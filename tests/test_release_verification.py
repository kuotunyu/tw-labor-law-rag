import importlib
import io
import json
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = "kuotunyu"
PUBLIC_EMAIL = "61350295+kuotunyu@users.noreply.github.com"


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def init_public_repo(repo: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    run_git(repo, "config", "user.name", PUBLIC_NAME)
    run_git(repo, "config", "user.email", PUBLIC_EMAIL)


def commit_file(repo: Path, relative_path: str, content: bytes, message: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    run_git(repo, "add", "--", relative_path)
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def tar_bytes(*members: tuple[str, bytes, bytes | None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data, link_target in members:
            info = tarfile.TarInfo(name=name)
            if link_target is None:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = link_target.decode("utf-8")
                archive.addfile(info)
    return buffer.getvalue()


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


def write_version_contract_fixture(
    root: Path,
    *,
    package_version: str = "0.2.0",
    release_version: str = "v0.2.0",
    evidence_version: str = "v0.1.0",
) -> dict:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "labor-rag"\nversion = "{package_version}"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"這是 `{release_version}` source-only reliability release。",
        encoding="utf-8",
    )
    (root / "README.en.md").write_text(
        f"This is the `{release_version}` source-only reliability release.",
        encoding="utf-8",
    )
    return {
        "release_version": release_version,
        "formal_evidence_version": evidence_version,
    }


def test_release_version_contract_is_explicit_and_consistent(tmp_path):
    module = release_module()
    manifest = write_version_contract_fixture(tmp_path)

    assert module._verify_release_version_contract(tmp_path, manifest) == {
        "version": "v0.2.0",
        "package_version": "0.2.0",
        "formal_evidence_version": "v0.1.0",
    }

    manifest["release_version"] = "v0.3.0"
    with pytest.raises(module.ReleaseVerificationError, match="package release version"):
        module._verify_release_version_contract(tmp_path, manifest)


def test_release_version_contract_rejects_changed_formal_evidence_baseline(tmp_path):
    module = release_module()
    manifest = write_version_contract_fixture(
        tmp_path,
        evidence_version="v0.2.0",
    )

    with pytest.raises(module.ReleaseVerificationError, match="formal evidence version"):
        module._verify_release_version_contract(tmp_path, manifest)


def test_release_verifier_recomputes_committed_evidence():
    report = release_module().verify_release(PROJECT_ROOT)

    assert report["status"] == "pass"
    assert report["release"] == {
        "version": "v0.2.0",
        "package_version": "0.2.0",
        "formal_evidence_version": "v0.1.0",
    }
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
        "full_history_checkout": True,
    }
    assert report["tooling"]["ruff"]
    expected_tracking = (
        "exact_public_allowlist"
        if (PROJECT_ROOT / ".git").exists()
        else "not_applicable_no_git_metadata"
    )
    assert report["publication"]["tracking"] == expected_tracking
    expected_history = len(
        {
            line
            for line in run_git(
                PROJECT_ROOT,
                "rev-list",
                "--branches",
                "--tags",
                "--exclude=pull/*",
                "--remotes",
            )
            .stdout.decode("ascii")
            .splitlines()
            if line
        }
    )
    assert report["publication"]["history_commits"] == expected_history
    assert report["publication"]["history_ref_namespaces"] == [
        "heads",
        "tags",
        "remotes",
    ]


def test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions():
    manifest = json.loads(
        (PROJECT_ROOT / "release" / "manifest.json").read_text(encoding="utf-8")
    )
    tracked = git_tracked_paths()

    assert manifest["release_type"] == "public_source_only_portfolio_release"
    assert manifest["publication"]["tracked_excluded"] == []
    assert tracked == public_paths()
    assert len(tracked) == 110
    assert "docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md" in tracked
    assert "docs/release/HUGGINGFACE_ZERO_COST_IMPLEMENTATION_PLAN.md" in tracked
    assert "docs/release/RELEASE_EVOLUTION_DESIGN.md" in tracked
    assert "docs/release/RELEASE_EVOLUTION_IMPLEMENTATION_PLAN.md" in tracked
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


def test_ci_contract_rejects_shallow_history_checkout(tmp_path):
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    workflow_path = tmp_path / "ci.yml"
    workflow_path.write_text(
        workflow.replace("          fetch-depth: 0\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="full Git history",
    ):
        release_module()._verify_ci_publication_contract(workflow_path)


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


def test_publishable_commit_ids_ignore_nonstandard_recovery_refs(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    recovered_commit = commit_file(
        tmp_path,
        ".claude/launch.json",
        b"{}",
        "local recovery only",
    )
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/archive/recovery", recovered_commit)

    assert release_module()._publishable_commit_ids(tmp_path) == [public_commit]


def test_publishable_commit_ids_include_archive_named_branch(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    second_commit = commit_file(tmp_path, "CHANGELOG.md", b"change", "change")
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/heads/archive/review", second_commit)

    assert set(release_module()._publishable_commit_ids(tmp_path)) == {
        public_commit,
        second_commit,
    }


def test_publishable_commit_ids_include_and_deduplicate_tags_and_remotes(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    release_commit = commit_file(tmp_path, "CHANGELOG.md", b"release", "release")
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "tag", "-a", "v0.2.0", "-m", "release", release_commit)
    run_git(tmp_path, "update-ref", "refs/remotes/origin/release", release_commit)

    assert set(release_module()._publishable_commit_ids(tmp_path)) == {
        public_commit,
        release_commit,
    }


def test_publishable_commit_ids_ignore_synthetic_pull_merge_remote(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    synthetic_merge = commit_file(
        tmp_path,
        "PR_MERGE.md",
        b"ephemeral merge",
        "synthetic pull merge",
    )
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(
        tmp_path,
        "update-ref",
        "refs/remotes/pull/1/merge",
        synthetic_merge,
    )

    assert release_module()._publishable_commit_ids(tmp_path) == [public_commit]


def test_publishable_commit_ids_reject_empty_ref_set(tmp_path):
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="no publishable commits",
    ):
        release_module()._publishable_commit_ids(tmp_path)


def test_parse_git_archive_returns_regular_entries():
    module = release_module()
    payload = tar_bytes(("README.md", b"public", None))

    assert module._parse_git_archive(payload, commit="a" * 40) == [
        module.PublicEntry(path="README.md", data=b"public")
    ]


@pytest.mark.parametrize(
    "unsafe_path",
    ["../secret.txt", "/absolute.txt", "C:" + "/private.txt"],
)
def test_parse_git_archive_rejects_unsafe_paths(unsafe_path):
    payload = tar_bytes((unsafe_path, b"private", None))

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unsafe Git archive path",
    ):
        release_module()._parse_git_archive(payload, commit="b" * 40)


def test_parse_git_archive_rejects_links():
    payload = tar_bytes(("link.txt", b"", b"README.md"))

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unsupported Git archive entry",
    ):
        release_module()._parse_git_archive(payload, commit="c" * 40)


def test_parse_git_archive_rejects_duplicate_paths():
    payload = tar_bytes(
        ("README.md", b"first", None),
        ("README.md", b"second", None),
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="duplicate Git archive path",
    ):
        release_module()._parse_git_archive(payload, commit="d" * 40)


def test_parse_git_archive_rejects_duplicate_directory_paths():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for _ in range(2):
            info = tarfile.TarInfo(name="docs/")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="duplicate Git archive path",
    ):
        release_module()._parse_git_archive(buffer.getvalue(), commit="e" * 40)


def test_git_archive_entries_read_committed_tree(tmp_path):
    init_public_repo(tmp_path)
    commit = commit_file(tmp_path, "docs/README.md", b"public", "public")

    assert release_module()._git_archive_entries(tmp_path, commit) == [
        release_module().PublicEntry(path="docs/README.md", data=b"public")
    ]


def test_git_archive_entries_wrap_git_failure(tmp_path):
    init_public_repo(tmp_path)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="failed to read Git archive for commit",
    ):
        release_module()._git_archive_entries(tmp_path, "f" * 40)


def test_publishable_history_rejects_forbidden_path_on_public_branch(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    forbidden_commit = commit_file(
        tmp_path,
        ".claude/launch.json",
        b"{}",
        "forbidden",
    )
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/heads/archive/review", forbidden_commit)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="publishable history issues",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )


def test_publishable_history_rejects_secret_removed_from_current_tree(tmp_path):
    init_public_repo(tmp_path)
    secret = "sk-" + "A" * 32
    commit_file(tmp_path, "README.md", secret.encode("utf-8"), "leak")
    commit_file(tmp_path, "README.md", b"clean", "remove leak")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="provider_token",
    ) as caught:
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )

    assert secret not in str(caught.value)


def test_publishable_history_allows_explicit_legacy_path(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "legacy.md", b"old public file", "old release")
    (tmp_path / "legacy.md").unlink()
    (tmp_path / "README.md").write_bytes(b"current")
    run_git(tmp_path, "add", "--all")
    run_git(tmp_path, "commit", "-m", "new release")

    count = release_module()._verify_publishable_git_history(
        tmp_path,
        {"README.md"},
        legacy_public_paths={"legacy.md"},
        reviewed_binary_hashes=set(),
    )

    assert count == 2


def test_publishable_history_allows_current_file_absent_from_older_commit(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "README.md", b"first", "first release")
    commit_file(tmp_path, "CHANGELOG.md", b"added later", "second release")

    assert release_module()._verify_publishable_git_history(
        tmp_path,
        {"README.md", "CHANGELOG.md"},
        legacy_public_paths=set(),
        reviewed_binary_hashes=set(),
    ) == 2


def test_publishable_history_rejects_unlisted_legacy_path(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "legacy.md", b"old public file", "old release")
    (tmp_path / "legacy.md").unlink()
    (tmp_path / "README.md").write_bytes(b"current")
    run_git(tmp_path, "add", "--all")
    run_git(tmp_path, "commit", "-m", "new release")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unexpected_history_path",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )


def test_publishable_history_requires_reviewed_binary_hash(tmp_path):
    init_public_repo(tmp_path)
    binary = b"reviewed image bytes"
    commit_file(tmp_path, "image.png", binary, "image")
    digest = __import__("hashlib").sha256(binary).hexdigest()
    module = release_module()

    assert module._verify_publishable_git_history(
        tmp_path,
        {"image.png"},
        legacy_public_paths=set(),
        reviewed_binary_hashes={digest},
    ) == 1
    with pytest.raises(
        module.ReleaseVerificationError,
        match="unreviewed_history_binary",
    ):
        module._verify_publishable_git_history(
            tmp_path,
            {"image.png"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )


def test_publishable_history_rejects_unexpected_identity(tmp_path):
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    run_git(tmp_path, "config", "user.name", "Unexpected Author")
    private_email = "unexpected" + "@" + "example.test"
    run_git(tmp_path, "config", "user.email", private_email)
    commit_file(tmp_path, "README.md", b"public", "wrong identity")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="public commit identity",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
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


def test_public_entry_scan_reports_secret_without_leaking_value():
    module = release_module()
    secret = "sk-" + "A" * 32
    entries = [module.PublicEntry(path="README.md", data=secret.encode("utf-8"))]

    issues = module._scan_public_entries(entries)
    serialized = json.dumps(issues)

    assert [issue["category"] for issue in issues] == ["provider_token"]
    assert secret not in serialized


def test_public_file_scan_preserves_missing_sensitive_path_findings(tmp_path):
    issues = release_module().scan_public_files(tmp_path, [".env"])

    assert {issue["category"] for issue in issues} == {
        "missing_public_file",
        "sensitive_public_path",
    }


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
