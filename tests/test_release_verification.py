import importlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def release_module():
    return importlib.import_module("rag.release_verification")


def public_paths() -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in (PROJECT_ROOT / "release" / "public-files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def git_tracked_paths() -> set[str]:
    process = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in process.stdout.split(b"\0")
        if item
    }


def test_release_verifier_recomputes_committed_evidence():
    report = release_module().verify_release(PROJECT_ROOT)

    assert report["status"] == "pass"
    assert report["dataset"] == {
        "questions": 40,
        "answerable": 30,
        "unanswerable": 10,
    }
    assert report["dataset_sha256"] == (
        "760e33eaa0821001d37ff974bc037043d019fc670b8f3621b6e713030274ca07"
    )
    assert report["ablation"] == {"configurations": 8, "rows": 320}
    assert report["primary_retrieval"]["hit_at_5"] == pytest.approx(29 / 30)
    assert report["primary_retrieval"]["mrr_at_10"] == pytest.approx(
        0.9055555555555554
    )
    assert report["e2e"]["answered"] == 29
    assert report["e2e"]["refused"] == 11
    assert report["e2e"]["generation_calls"] == 31
    assert report["e2e"]["refusal_by_stage"]["threshold"]["count"] == 9
    assert report["e2e"]["refusal_by_stage"]["llm"]["count"] == 2
    assert report["provider_evidence"] == {
        "classification": "archived_provider_evidence",
        "judged": 29,
        "avg_faithfulness": pytest.approx(4.896551724137931),
        "avg_relevancy": pytest.approx(5.0),
    }
    assert report["privacy"] == {
        "official_trace_issues": 0,
        "public_scan_issues": 0,
    }
    assert report["source_data"] == {
        "dataset_id": 18290,
        "license": "政府資料開放授權條款－第1版",
        "redistribution": "allowed_with_attribution",
        "samples_verified": 2,
    }
    assert report["ci"] == {
        "action_pins": 2,
        "all_pinned": True,
        "lint": True,
        "tag_trigger": "v*",
    }
    assert report["tooling"]["ruff"]
    expected_tracking = (
        "exact_public_allowlist"
        if (PROJECT_ROOT / ".git").exists()
        else "not_applicable_no_git_metadata"
    )
    assert report["publication"]["tracking"] == expected_tracking


def test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions():
    manifest = json.loads(
        (PROJECT_ROOT / "release" / "manifest.json").read_text(encoding="utf-8")
    )
    tracked = git_tracked_paths()

    assert manifest["release_type"] == "public_source_only_portfolio_release"
    assert manifest["publication"]["tracked_excluded"] == []
    assert tracked == public_paths()
    assert len(tracked) == 94
    forbidden_prefixes = (
        ".claude/",
        ".worktrees/",
        "data/raw/",
        "docs/superpowers/",
        "eval/runs/",
        "storage/",
    )
    assert not any(
        path.startswith(prefix)
        for path in tracked
        for prefix in forbidden_prefixes
    )


def test_ruff_is_locked_and_ci_enforces_publication_gates():
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    dev_requirements = project["dependency-groups"]["dev"]
    assert any(re.match(r"^ruff(?:\W|$)", requirement) for requirement in dev_requirements)
    assert any(
        package["name"] == "ruff" and package.get("version")
        for package in lock["package"]
    )
    required_commands = [
        "uv lock --check",
        "uv sync --locked",
        "uv run ruff check .",
        "uv run python scripts/verify_release.py",
        "uv run pytest",
        "uv build",
        "uv run pytest tests/test_packaging.py -q",
        "import rag.api.main",
        "scripts/ask.py --help",
    ]
    positions = [workflow.index(command) for command in required_commands]
    assert positions == sorted(positions)
    assert re.search(r"(?m)^\s+branches:\s*\[main\]\s*$", workflow)
    assert re.search(r'(?m)^\s+tags:\s*\["v\*"\]\s*$', workflow)
    assert re.search(r"(?m)^\s+pull_request:\s*$", workflow)


def test_readme_first_screen_links_english_and_ci():
    first_screen = "\n".join(
        (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").splitlines()[:12]
    )

    assert "README.en.md" in first_screen
    assert "actions/workflows/ci.yml/badge.svg?branch=main" in first_screen


def test_design_does_not_expand_observed_corpus_scale():
    design = (PROJECT_ROOT / "DESIGN.md").read_text(encoding="utf-8")

    assert "88萬候選" not in design
    assert "884 個候選" in design


def test_reachable_history_rejects_a_forbidden_path(tmp_path):
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "kuotunyu"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "user.email",
            "61350295+kuotunyu@users.noreply.github.com",
        ],
        check=True,
    )
    (tmp_path / "README.md").write_text("public", encoding="utf-8")
    forbidden = tmp_path / ".claude"
    forbidden.mkdir()
    (forbidden / "launch.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--all"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "test fixture"],
        check=True,
        capture_output=True,
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="reachable history tree",
    ):
        release_module()._verify_reachable_git_history(
            tmp_path,
            {"README.md"},
            max_commits=1,
        )


def test_source_archive_skips_only_git_tracking_check(tmp_path):
    module = release_module()

    assert module._tracked_files(tmp_path) is None


def test_source_archive_inventory_rejects_non_generated_extras(tmp_path):
    (tmp_path / "README.md").write_text("public", encoding="utf-8")
    generated = tmp_path / ".venv" / "Lib"
    generated.mkdir(parents=True)
    (generated / "installed.txt").write_text("generated", encoding="utf-8")
    runtime_storage = tmp_path / "storage" / "dict"
    runtime_storage.mkdir(parents=True)
    (runtime_storage / "dict.txt.big").write_text("generated", encoding="utf-8")
    (tmp_path / "private.txt").write_text("not allowlisted", encoding="utf-8")

    extras = release_module()._source_archive_extra_files(tmp_path, ["README.md"])

    assert extras == ["private.txt"]


@pytest.mark.parametrize(
    "row",
    [
        {
            "qid": "high-threshold",
            "refused": True,
            "refusal_stage": "threshold",
            "top_score": 0.9,
        },
        {
            "qid": "low-answer",
            "refused": False,
            "refusal_stage": None,
            "top_score": 0.01,
        },
        {
            "qid": "low-llm",
            "refused": True,
            "refusal_stage": "llm",
            "top_score": 0.01,
        },
        {
            "qid": "nonfinite-score",
            "refused": False,
            "refusal_stage": None,
            "top_score": float("nan"),
        },
    ],
)
def test_threshold_contract_rejects_score_stage_disagreement(row):
    with pytest.raises(release_module().ReleaseVerificationError, match="threshold"):
        release_module()._verify_e2e_threshold_contract([row], threshold=0.03)


def test_public_scan_reports_categories_without_leaking_values(tmp_path):
    private_value = "do-not-print-this-provider-response"
    private_name = "PrivatePerson"
    private_email = "private.person" + "@" + "example.test"
    separator = chr(92)
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": private_value,
                "provider_response": private_value,
                "contact": private_email,
                "debug_path": (
                    f"C:{separator}Users{separator}{private_name}{separator}run.json"
                ),
            }
        ),
        encoding="utf-8",
    )

    issues = release_module().scan_public_files(tmp_path, ["trace.jsonl"])
    serialized = json.dumps(issues)

    assert {issue["category"] for issue in issues} == {
        "local_path",
        "personal_identifier",
        "provider_payload",
    }
    assert private_value not in serialized
    assert private_name not in serialized
    assert private_email not in serialized


def test_public_scan_allows_example_keys_but_rejects_real_assignments(tmp_path):
    example = tmp_path / ".env.example"
    actual = tmp_path / "leak.txt"
    actual_token = "sk-" + "ant-api03-" + "not-public-credential-material"
    example.write_text("OPENAI_API_KEY=\nGEMINI_API_KEY=your_key_here\n", encoding="utf-8")
    actual.write_text(f"ANTHROPIC_API_KEY={actual_token}\n", encoding="utf-8")

    module = release_module()
    assert module.scan_public_files(tmp_path, [".env.example"]) == []
    issues = module.scan_public_files(tmp_path, ["leak.txt"])

    assert {issue["category"] for issue in issues} == {
        "provider_token",
        "secret_assignment",
    }
    assert "sk-ant" not in json.dumps(issues)
    assert actual_token not in json.dumps(issues)


def test_public_scan_rejects_standalone_supported_provider_tokens(tmp_path):
    anthropic_token = "sk-" + "ant-api03-" + "A" * 24
    openai_token = "sk-" + "B" * 32
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps({"credential": anthropic_token, "opaque": openai_token}),
        encoding="utf-8",
    )

    issues = release_module().scan_public_files(tmp_path, ["payload.json"])
    serialized = json.dumps(issues)

    assert [issue["category"] for issue in issues] == ["provider_token"]
    assert anthropic_token not in serialized
    assert openai_token not in serialized


def test_reviewed_binary_hashes_fail_closed(tmp_path):
    path = tmp_path / "approved.png"
    path.write_bytes(b"reviewed image bytes")
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    module = release_module()

    assert module._verify_reviewed_binaries(
        tmp_path,
        ["approved.png"],
        {"approved.png": digest},
    ) == 1
    with pytest.raises(module.ReleaseVerificationError, match="binary SHA-256"):
        module._verify_reviewed_binaries(
            tmp_path,
            ["approved.png"],
            {"approved.png": "0" * 64},
        )


def test_trace_schema_fails_closed_on_provider_fields():
    module = release_module()
    row = {
        "answerable": True,
        "chunking": "structure",
        "elapsed_ms": 1.0,
        "qid": "eval-01",
        "rank": 1,
        "reranker": True,
        "retrieval": "hybrid",
        "top_score": 0.9,
        "provider_response": "private",
    }

    issues = module.scan_trace_rows([row], "ablation", "trace.jsonl")

    assert issues == [
        {
            "path": "trace.jsonl",
            "category": "unexpected_trace_field",
            "location": "row 1 field provider_response",
        }
    ]


def test_action_pin_scan_rejects_mutable_tags_without_leaking_ref(tmp_path):
    workflow = tmp_path / "ci.yml"
    workflow.write_text("- uses: actions/checkout@v6\n", encoding="utf-8")

    issues = release_module().scan_action_pins(workflow, "ci.yml")

    assert issues == [
        {
            "path": "ci.yml",
            "category": "mutable_action_ref",
            "location": "line 1",
        }
    ]
    assert "checkout@v6" not in json.dumps(issues)
