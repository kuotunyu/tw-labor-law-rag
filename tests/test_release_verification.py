import hashlib
import importlib
import io
import json
import re
import subprocess
import tarfile
import tomllib
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = "kuotunyu"
PUBLIC_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
CURRENT_PROVIDER_EVIDENCE_RELEASE = "v0.3.2 provider safety cross-check"
CURRENT_PROVIDER_EVIDENCE_DOCUMENTS = {
    "README.md": 2,
    "README.en.md": 2,
    "DESIGN.md": 1,
    "EVAL_REPORT.md": 1,
    "docs/release/CLAIM_MATRIX.md": 2,
    "docs/release/PUBLICATION_BOUNDARY.md": 1,
    "docs/release/REVIEWER_GUIDE.md": 2,
    "eval/official/README.md": 1,
}
PROVIDER_EVIDENCE_FULL_CLAIM = re.compile(
    r"Gemini(?: `gemini-3\.5-flash-lite`)?(?: observed)? "
    r"refusal(?: accuracy)? `(?P<gemini_refusal>[^`]+)`.{0,80}?"
    r"citation(?: success)? `(?P<gemini_citation>[^`]+)`.{0,80}?"
    r"(?:estimated )?cost `US\$(?P<gemini_cost>[^`]+)`.{0,80}?"
    r"OpenAI(?: `gpt-5\.6-luna`)?(?: observed)? "
    r"refusal(?: accuracy)? `(?P<openai_refusal>[^`]+)`.{0,80}?"
    r"citation(?: success)? `(?P<openai_citation>[^`]+)`.{0,80}?"
    r"(?:estimated )?cost `US\$(?P<openai_cost>[^`]+)`",
    re.DOTALL,
)
PROVIDER_EVIDENCE_COMPACT_CLAIM = re.compile(
    r"Gemini refusal/citation `(?P<gemini_refusal>[^`]+)`/"
    r"`(?P<gemini_citation>[^`]+)`[、,]\s*cost "
    r"`US\$(?P<gemini_cost>[^`]+)`[；;].{0,40}?"
    r"OpenAI `(?P<openai_refusal>[^`]+)`/"
    r"`(?P<openai_citation>[^`]+)`[、,]\s*"
    r"`US\$(?P<openai_cost>[^`]+)`",
    re.DOTALL,
)


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


def provider_evidence_metric_claims(content: str) -> list[dict[str, str]]:
    matches = [
        (match.start(), match.groupdict())
        for pattern in (
            PROVIDER_EVIDENCE_FULL_CLAIM,
            PROVIDER_EVIDENCE_COMPACT_CLAIM,
        )
        for match in pattern.finditer(content)
    ]
    return [claim for _, claim in sorted(matches)]


def assert_provider_evidence_doc_contract(relative_path: str, content: str) -> None:
    claims = provider_evidence_metric_claims(content)

    assert len(claims) == CURRENT_PROVIDER_EVIDENCE_DOCUMENTS[relative_path], (
        relative_path
    )
    assert "gemini-3.5-flash-lite" in content, relative_path
    assert "gpt-5.6-luna" in content, relative_path
    assert re.search(
        r"(?:five requests(?: per provider| each)?|五筆(?: safety cross-check|請求)?)",
        content,
        re.IGNORECASE,
    ), relative_path
    for claim in claims:
        assert claim == {
            "gemini_refusal": "0.8",
            "gemini_citation": "1.0",
            "gemini_cost": "0.0022620",
            "openai_refusal": "1.0",
            "openai_citation": "1.0",
            "openai_cost": "0.0026414",
        }, relative_path
        assert Decimal(claim["gemini_cost"]) + Decimal(claim["openai_cost"]) == Decimal(
            "0.0049034"
        ), relative_path
    assert "safety cross-check" in content, relative_path
    assert "v0.1.0" in content, relative_path
    assert re.search(r"content-free|嚴格不含", content), relative_path
    assert "question/answer" in content, relative_path
    assert "provider payload" in content, relative_path
    assert re.search(r"credentials|憑證", content), relative_path


@pytest.mark.parametrize("relative_path", CURRENT_PROVIDER_EVIDENCE_DOCUMENTS)
def test_current_provider_evidence_docs_pin_completed_release_claims(relative_path):
    content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    assert_provider_evidence_doc_contract(relative_path, content)


def test_current_provider_evidence_docs_validate_every_matching_claim(
):
    content = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")
    canonical_claim = next(
        paragraph
        for paragraph in re.split(r"\n\s*\n", content)
        if "US$0.0022620" in paragraph and "US$0.0026414" in paragraph
    )
    drifted_claim = canonical_claim.replace("US$0.0026414", "US$0.0099999")

    with pytest.raises(AssertionError):
        assert_provider_evidence_doc_contract(
            "README.en.md",
            f"{canonical_claim}\n\n{drifted_claim}\n",
        )


def test_current_provider_evidence_docs_publish_completed_release_contract():
    for relative_path in CURRENT_PROVIDER_EVIDENCE_DOCUMENTS:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert CURRENT_PROVIDER_EVIDENCE_RELEASE in content, relative_path
        current_section = content.split(CURRENT_PROVIDER_EVIDENCE_RELEASE, 1)[1]
        current_section = current_section.split("\n## ", 1)[0]
        assert "pending_credentials" not in current_section, relative_path

def test_official_provider_trace_documentation_has_token_usage_carve_out():
    content = (PROJECT_ROOT / "eval/official/README.md").read_text(encoding="utf-8")

    assert "其他 official traces 不含 token usage 或 API metadata。" in content
    assert (
        "provider trace 僅含 strict allowlisted "
        "provider/model/verdict/token count/cost/latency，仍排除 "
        "prompts/questions/answers/provider payload/credentials/private paths/PII。"
    ) in content


def write_version_contract_fixture(
    root: Path,
    *,
    package_version: str = "0.3.4",
    release_version: str = "v0.3.4",
    evidence_version: str = "v0.1.0",
) -> dict:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "labor-rag"\nversion = "{package_version}"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"這是 `{release_version}` source-only runtime and deployment release。",
        encoding="utf-8",
    )
    (root / "README.en.md").write_text(
        f"This is the `{release_version}` source-only runtime and deployment release.",
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
        "version": "v0.3.4",
        "package_version": "0.3.4",
        "formal_evidence_version": "v0.1.0",
    }

    manifest["release_version"] = "v0.4.0"
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


def write_wage_arrears_regression_fixture(root: Path) -> dict:
    dataset_path = root / "eval" / "dataset" / "wage.jsonl"
    result_path = root / "eval" / "official" / "wage.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = []
    for index in range(1, 21):
        positive = index <= 10
        qid = f"wage-reg-{index:03d}"
        rows.append(
            {
                "qid": qid,
                "question": (
                    f"公司欠薪，我想直接離職。案例 {index}"
                    if positive
                    else f"公司欠薪，如何追討？案例 {index}"
                ),
                "expect_expansion": positive,
                "sources": (
                    [{"doc": "勞動基準法", "article": "第 14 條"}]
                    if positive
                    else []
                ),
                "style_tags": ["fixture"],
            }
        )
        cases.append(
            {
                "qid": qid,
                "expected_expansion": positive,
                "expansion_applied": positive,
                "rank": 1 if positive else None,
                "top_score": 0.1,
            }
        )
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    revision = "a" * 40
    configuration = {
        "chunking": "structure",
        "retrieval": "hybrid",
        "reranker": True,
        "top_k_retrieve": 20,
        "top_k_final": 5,
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "embed-revision",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "rerank-revision",
    }
    result = {
        "schema_version": "1.0",
        "dataset": {
            "path": "eval/dataset/wage.jsonl",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "questions": 20,
        },
        "code_revision": revision,
        "configuration": configuration,
        "summary": {
            "positive_routes": 10,
            "collision_routes_avoided": 10,
            "positive_hit_at_5": 10,
            "positive_hit_at_1": 10,
            "passed": True,
        },
        "cases": cases,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset": {
            "path": "eval/dataset/wage.jsonl",
            "sha256": result["dataset"]["sha256"],
            "questions": 20,
        },
        "results_path": "eval/official/wage.json",
        "code_revision": revision,
        "configuration": configuration,
        "summary": result["summary"],
    }


def test_wage_arrears_regression_contract_recomputes_evidence(tmp_path):
    module = release_module()
    contract = write_wage_arrears_regression_fixture(tmp_path)

    assert module._verify_wage_arrears_regression_evidence(tmp_path, contract) == {
        "questions": 20,
        "positive_routes": 10,
        "collision_routes_avoided": 10,
        "positive_hit_at_5": 10,
        "positive_hit_at_1": 10,
        "passed": True,
    }

    result_path = tmp_path / contract["results_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["cases"][0]["rank"] = 6
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(module.ReleaseVerificationError, match="positive Hit@5"):
        module._verify_wage_arrears_regression_evidence(tmp_path, contract)


def test_pending_provider_crosscheck_contract_is_explicit(tmp_path):
    module = release_module()
    contract = {
        "status": "pending_credentials",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "results_path": "eval/official/provider_crosscheck_results.json",
        "trace_path": "eval/official/provider_crosscheck_trace.jsonl",
    }

    assert module._verify_provider_crosscheck_contract(tmp_path, contract) == {
        "status": "pending_credentials",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
    }


def write_completed_provider_crosscheck_fixture(root: Path) -> dict:
    contract = {
        "status": "complete",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "results_path": "eval/official/provider_crosscheck_results.json",
        "trace_path": "eval/official/provider_crosscheck_trace.jsonl",
        "dataset": {
            "path": "eval/dataset/reliability_stress_v0.3.1.jsonl",
            "sha256": "7641d78e8434d8832319a70af019c1e0d860079a23fca161b488497e1c6b1b7f",
        },
    }
    dataset_path = root / contract["dataset"]["path"]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(
        (PROJECT_ROOT / contract["dataset"]["path"]).read_bytes()
    )
    reliability_path = root / "eval/official/reliability_trace.jsonl"
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    reliability_path.write_bytes(
        (PROJECT_ROOT / "eval/official/reliability_trace.jsonl").read_bytes()
    )
    qids = ["stress-001", "stress-002", "stress-003", "stress-042", "stress-043"]
    answerable = [True, True, True, False, False]
    provider_rows = {
        "gemini": {
            "model": "gemini-3.5-flash-lite",
            "inputs": [1308] * 5,
            "outputs": [24] * 5,
            "elapsed": [100.0, 101.0, 102.0, 103.0, 104.0],
            "costs": ["0.0004524"] * 5,
            "input_per_million": "0.30",
            "output_per_million": "2.50",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "total_cost": "0.0022620",
            "remaining_budget": "4.9977380",
        },
        "openai": {
            "model": "gpt-5.6-luna",
            "inputs": [2521, 2521, 2521, 2522, 2522],
            "outputs": [20] * 5,
            "elapsed": [110.0, 111.0, 112.0, 113.0, 114.0],
            "costs": ["0.0005282", "0.0005282", "0.0005282", "0.0005284", "0.0005284"],
            "input_per_million": "0.20",
            "output_per_million": "1.20",
            "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
            "total_cost": "0.0026414",
            "remaining_budget": "4.9973586",
        },
    }
    trace_rows = []
    for provider, values in provider_rows.items():
        for index, qid in enumerate(qids):
            trace_rows.append(
                {
                    "qid": qid,
                    "answerable": answerable[index],
                    "requested_provider": provider,
                    "actual_provider": provider,
                    "model": values["model"],
                    "refused": not answerable[index],
                    "citation_count": 1 if answerable[index] else 0,
                    "input_tokens": values["inputs"][index],
                    "output_tokens": values["outputs"][index],
                    "estimated_cost_usd": values["costs"][index],
                    "refusal_verdict": 1,
                    "citation_verdict": 1,
                    "elapsed_ms": values["elapsed"][index],
                }
            )
    results = {
        "schema_version": "1.0",
        "run_date": "2026-08-29",
        "dataset": contract["dataset"],
        "corpus_snapshot": {
            "path": "release/corpus_snapshot.json",
            "snapshot_date": "2026-08-29",
            "laws": 15,
            "articles": 884,
        },
        "selection": {
            "initial_per_provider": 5,
            "maximum_per_provider": 5,
            "generation_eligible_only": True,
        },
        "authorization": {
            "per_provider_cap_usd": "5.00",
            "max_input_tokens_per_request": 20_000,
            "max_output_tokens_per_request": 1_024,
        },
        "pricing": {
            provider: {
                "model": values["model"],
                "input_per_million_usd": values["input_per_million"],
                "output_per_million_usd": values["output_per_million"],
                "source": values["source"],
            }
            for provider, values in provider_rows.items()
        },
        "provider_status": {
            provider: {"status": "complete", "reason": None}
            for provider in provider_rows
        },
        "provider_metrics": {
            provider: {
                "requests": 5,
                "refusal_accuracy": 1.0,
                "citation_success_rate": 1.0,
                "input_tokens": sum(values["inputs"]),
                "output_tokens": sum(values["outputs"]),
                "estimated_cost_usd": values["total_cost"],
                "avg_latency_ms": sum(values["elapsed"]) / 5,
            }
            for provider, values in provider_rows.items()
        },
        "budget_ledgers": {
            provider: {
                "cap_usd": "5.00",
                "spent_usd": values["total_cost"],
                "remaining_usd": values["remaining_budget"],
                "requests": 5,
                "input_tokens": sum(values["inputs"]),
                "output_tokens": sum(values["outputs"]),
                "input_per_million_usd": values["input_per_million"],
                "output_per_million_usd": values["output_per_million"],
            }
            for provider, values in provider_rows.items()
        },
        "privacy": {
            "public_trace_contains_question_or_answer": False,
            "public_trace_contains_provider_payload": False,
            "public_trace_contains_credentials": False,
            "raw_trace_path": "ignored eval/runs only",
        },
    }
    results_path = root / contract["results_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results), encoding="utf-8")
    (root / contract["trace_path"]).write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    return contract


def test_completed_provider_crosscheck_recomputes_public_evidence(tmp_path):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)

    assert module._verify_provider_crosscheck_contract(tmp_path, contract) == {
        "status": "complete",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "requests": {"gemini": 5, "openai": 5},
        "estimated_cost_usd": {
            "gemini": "0.0022620",
            "openai": "0.0026414",
        },
    }


def test_completed_provider_crosscheck_contract_mismatch_never_reflects_values(tmp_path):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    marker = "SENSITIVE-MARKER-NEVER-REFLECT"
    contract["unexpected_private_field"] = marker

    with pytest.raises(module.ReleaseVerificationError) as error:
        module._verify_provider_crosscheck_contract(tmp_path, contract)

    assert "provider cross-check complete contract fields" in str(error.value)
    assert marker not in str(error.value)


@pytest.mark.parametrize(
    "requested_provider",
    [
        ["gemini", "SENSITIVE-LIST-MARKER"],
        {"provider": "gemini", "marker": "SENSITIVE-OBJECT-MARKER"},
    ],
)
def test_completed_provider_crosscheck_rejects_non_string_requested_provider(
    tmp_path,
    requested_provider,
):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    trace_path = tmp_path / contract["trace_path"]
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trace_rows[0]["requested_provider"] = requested_provider
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )

    with pytest.raises(
        module.ReleaseVerificationError,
        match="provider cross-check requested provider type",
    ) as error:
        module._verify_provider_crosscheck_contract(tmp_path, contract)

    assert "SENSITIVE" not in str(error.value)


@pytest.mark.parametrize(
    ("replacement_qid", "message"),
    [
        ("stress-037", "provider cross-check generation eligibility"),
        ("stress-004", "provider cross-check gemini selected qids"),
    ],
)
def test_completed_provider_crosscheck_rejects_non_selected_qids(
    tmp_path,
    replacement_qid,
    message,
):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    trace_path = tmp_path / contract["trace_path"]
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for row in trace_rows:
        if row["qid"] == "stress-001":
            row["qid"] = replacement_qid
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )

    with pytest.raises(module.ReleaseVerificationError, match=message):
        module._verify_provider_crosscheck_contract(tmp_path, contract)


def test_completed_provider_crosscheck_pins_qids_if_reliability_is_co_tampered(
    tmp_path,
):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    reliability_path = tmp_path / "eval/official/reliability_trace.jsonl"
    reliability_rows = [
        json.loads(line) for line in reliability_path.read_text().splitlines()
    ]
    for row in reliability_rows:
        if row["qid"] == "stress-001":
            row["threshold_refused"] = True
    reliability_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reliability_rows),
        encoding="utf-8",
    )

    trace_path = tmp_path / contract["trace_path"]
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    reordered_rows = []
    for provider in ("gemini", "openai"):
        rows_by_qid = {
            row["qid"]: row
            for row in trace_rows
            if row["requested_provider"] == provider
        }
        replacement = rows_by_qid.pop("stress-001")
        replacement["qid"] = "stress-004"
        reordered_rows.extend(
            [
                rows_by_qid["stress-002"],
                rows_by_qid["stress-003"],
                replacement,
                rows_by_qid["stress-042"],
                rows_by_qid["stress-043"],
            ]
        )
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reordered_rows),
        encoding="utf-8",
    )

    with pytest.raises(
        module.ReleaseVerificationError,
        match="provider cross-check deterministic selected qids",
    ):
        module._verify_provider_crosscheck_contract(tmp_path, contract)


def test_completed_provider_crosscheck_pins_run_date_to_snapshot(tmp_path):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    results_path = tmp_path / contract["results_path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["run_date"] = "2026-08-28"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    with pytest.raises(
        module.ReleaseVerificationError,
        match="provider cross-check run date",
    ):
        module._verify_provider_crosscheck_contract(tmp_path, contract)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("unknown_trace_field", "provider cross-check trace fields"),
        ("provider_fallback", "provider cross-check actual provider"),
        ("model_name", "provider cross-check gemini model"),
        ("request_count", "provider cross-check gemini.requests"),
        ("aggregate_tokens", "provider cross-check gemini.input_tokens"),
        ("cost", "provider cross-check gemini.estimated_cost_usd"),
        ("privacy_flag", "provider cross-check privacy"),
        ("dataset_hash", "provider cross-check dataset"),
        ("input_limit", "provider cross-check input tokens maximum"),
        ("dataset_qid", "provider cross-check dataset qids"),
        ("dataset_answerable", "provider cross-check dataset answerable"),
    ],
)
def test_completed_provider_crosscheck_rejects_tampered_public_evidence(
    tmp_path,
    tamper,
    message,
):
    module = release_module()
    contract = write_completed_provider_crosscheck_fixture(tmp_path)
    results_path = tmp_path / contract["results_path"]
    trace_path = tmp_path / contract["trace_path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    if tamper == "unknown_trace_field":
        trace_rows[0]["request_id"] = "forbidden"
    elif tamper == "provider_fallback":
        trace_rows[0]["actual_provider"] = "openai"
    elif tamper == "model_name":
        trace_rows[0]["model"] = "gemini-3.5-pro"
    elif tamper == "request_count":
        results["provider_metrics"]["gemini"]["requests"] = 4
    elif tamper == "aggregate_tokens":
        results["provider_metrics"]["gemini"]["input_tokens"] = 1
    elif tamper == "cost":
        results["provider_metrics"]["gemini"]["estimated_cost_usd"] = "0.01"
    elif tamper == "privacy_flag":
        results["privacy"]["public_trace_contains_credentials"] = True
    elif tamper == "dataset_hash":
        results["dataset"]["sha256"] = "0" * 64
    elif tamper == "input_limit":
        trace_rows[0]["input_tokens"] = 20_001
    elif tamper == "dataset_qid":
        trace_rows[0]["qid"] = "stress-999"
        trace_rows[5]["qid"] = "stress-999"
    elif tamper == "dataset_answerable":
        for index in (0, 5):
            trace_rows[index]["answerable"] = False
            trace_rows[index]["refused"] = True
            trace_rows[index]["citation_count"] = 0
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(f"unknown tamper case: {tamper}")

    results_path.write_text(json.dumps(results), encoding="utf-8")
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )

    with pytest.raises(module.ReleaseVerificationError, match=message):
        module._verify_provider_crosscheck_contract(tmp_path, contract)


def test_reliability_run_date_accepts_latest_global_civil_date():
    module = release_module()

    assert module._verify_evidence_run_date(
        "2026-08-29",
        latest_valid_date=date(2026, 8, 29),
    ) == date(2026, 8, 29)


def test_reliability_run_date_rejects_date_beyond_global_civil_limit():
    module = release_module()

    with pytest.raises(module.ReleaseVerificationError, match="run date is in the future"):
        module._verify_evidence_run_date(
            "2026-08-30",
            latest_valid_date=date(2026, 8, 29),
        )


@pytest.mark.parametrize(
    "artifact_path",
    [
        "eval/official/provider_crosscheck_results.json",
        "eval/official/provider_crosscheck_trace.jsonl",
    ],
)
def test_pending_provider_crosscheck_rejects_published_artifacts(
    tmp_path,
    artifact_path,
):
    module = release_module()
    contract = {
        "status": "pending_credentials",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "results_path": "eval/official/provider_crosscheck_results.json",
        "trace_path": "eval/official/provider_crosscheck_trace.jsonl",
    }
    artifact = tmp_path / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(
        module.ReleaseVerificationError,
        match="pending provider cross-check must not publish unverified artifacts",
    ):
        module._verify_provider_crosscheck_contract(tmp_path, contract)


def test_release_verifier_recomputes_committed_evidence():
    report = release_module().verify_release(PROJECT_ROOT)

    assert report["status"] == "pass"
    assert report["release"] == {
        "version": "v0.3.4",
        "package_version": "0.3.4",
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
    assert report["reliability"] == {
        "questions": 60,
        "hit_at_5": pytest.approx(0.95),
        "mrr_at_10": pytest.approx(0.9083333333333334),
        "direct_false_refusals": 1,
        "direct_unanswerable_coverage": pytest.approx(0.85),
        "decision": "retain_0.03",
    }
    assert report["wage_arrears_regression"] == {
        "questions": 20,
        "positive_routes": 10,
        "collision_routes_avoided": 10,
        "positive_hit_at_5": 10,
        "positive_hit_at_1": 10,
        "passed": True,
    }
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
    assert report["provider_crosscheck"] == {
        "status": "complete",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "requests": {"gemini": 5, "openai": 5},
        "estimated_cost_usd": {
            "gemini": "0.0022620",
            "openai": "0.0026414",
        },
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
        "full_snapshot_date": "2026-08-29",
        "full_snapshot_laws": 15,
        "full_snapshot_articles": 884,
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
    assert report["publication"]["files"] == 154
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


def test_english_readme_documents_provider_crosscheck_metadata_boundary():
    readme = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")

    assert (
        "Provider cross-check traces publish only strict allowlisted metadata"
        in readme
    )
    assert "token counts, estimated cost, and elapsed time" in readme
    assert (
        "exclude prompts, questions, answers, provider payloads, credentials, "
        "private paths, and personal identifiers"
    ) in readme
    assert "Public official traces do not contain" not in readme


def test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions():
    manifest = json.loads(
        (PROJECT_ROOT / "release" / "manifest.json").read_text(encoding="utf-8")
    )
    tracked = git_tracked_paths()

    assert manifest["release_type"] == "public_source_only_portfolio_release"
    assert manifest["publication"]["tracked_excluded"] == []
    assert tracked == public_paths()
    assert "docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md" in tracked
    assert "docs/release/HUGGINGFACE_ZERO_COST_IMPLEMENTATION_PLAN.md" in tracked
    assert "docs/release/RELEASE_EVOLUTION_DESIGN.md" in tracked
    assert "docs/release/RELEASE_EVOLUTION_IMPLEMENTATION_PLAN.md" in tracked
    assert "docs/release/BLUE_GREEN_QDRANT_MAINTENANCE_DESIGN.md" in tracked
    assert "docs/release/BLUE_GREEN_QDRANT_MAINTENANCE_IMPLEMENTATION_PLAN.md" in tracked
    assert "scripts/rebuild_qdrant_blue_green.py" in tracked
    assert "src/rag/qdrant_blue_green.py" in tracked
    assert "src/rag/qdrant_maintenance.py" in tracked
    assert "tests/test_qdrant_blue_green.py" in tracked
    assert "tests/test_qdrant_blue_green_cli.py" in tracked
    assert "tests/test_qdrant_maintenance.py" in tracked
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
    for tool in ("bandit", "pip-audit", "ruff"):
        assert any(
            re.match(rf"^{re.escape(tool)}(?:\W|$)", requirement)
            for requirement in dev_requirements
        )
        assert any(
            package["name"] == tool and package.get("version")
            for package in lock["package"]
        )
    required_commands = [
        "uv lock --check",
        "uv sync --locked",
        "uv run ruff check .",
        "uv run bandit -r src scripts -ll",
        "uv run pip-audit --local",
        "uv run python scripts/verify_release.py",
        "uv run pytest",
        "uv build",
        "uv run pytest tests/test_packaging.py -q",
        "import rag.api.main",
        "scripts/ask.py --help",
    ]
    positions = [workflow.index(command) for command in required_commands]
    assert positions == sorted(positions)
    assert "--ignore-vuln" not in workflow
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


def test_readmes_present_the_private_demo_and_reviewer_paths_truthfully():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_en = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")
    live_url = "https://steven0226-tw-labor-law-rag-demo.hf.space"

    for content in (readme, readme_en):
        assert "V035_REVIEWER_TOUR.md" in content
        assert "V035_INTERVIEW_DEMO.md" in content
        assert "2026-08-29" in content
        assert "private Space" in content
        assert live_url not in content
        assert "bm25_*.pkl" not in content
    assert "公開 BYOK Docker Space（已上線）" not in readme
    assert "Public BYOK Docker Space (live)" not in readme_en


def test_full_corpus_snapshot_verifier_proves_all_15_laws_and_article_arithmetic():
    result = release_module()._verify_full_corpus_snapshot(
        PROJECT_ROOT,
        {
            "path": "release/corpus_snapshot.json",
            "schema_version": "1.0",
            "snapshot_date": "2026-08-29",
            "laws": 15,
            "articles": 884,
        },
    )

    assert result == {"snapshot_date": "2026-08-29", "laws": 15, "articles": 884}


def test_full_corpus_snapshot_verifier_rejects_duplicate_law_names(tmp_path):
    snapshot = json.loads(
        (PROJECT_ROOT / "release" / "corpus_snapshot.json").read_text(encoding="utf-8")
    )
    snapshot["laws"][1]["name"] = snapshot["laws"][0]["name"]
    (tmp_path / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="corpus snapshot unique law names",
    ):
        release_module()._verify_full_corpus_snapshot(
            tmp_path,
            {
                "path": "snapshot.json",
                "schema_version": "1.0",
                "snapshot_date": "2026-08-29",
                "laws": 15,
                "articles": 884,
            },
        )


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


def test_publishable_history_allows_github_squash_committer(tmp_path):
    init_public_repo(tmp_path)
    run_git(tmp_path, "config", "user.name", "GitHub")
    run_git(tmp_path, "config", "user.email", "noreply" + "@" + "github.com")
    (tmp_path / "README.md").write_bytes(b"public")
    run_git(tmp_path, "add", "--", "README.md")
    run_git(
        tmp_path,
        "commit",
        f"--author={PUBLIC_NAME} <{PUBLIC_EMAIL}>",
        "-m",
        "GitHub squash",
    )

    assert release_module()._verify_publishable_git_history(
        tmp_path,
        {"README.md"},
        legacy_public_paths=set(),
        reviewed_binary_hashes=set(),
    ) == 1


def test_publishable_history_rejects_wrong_author_with_github_committer(tmp_path):
    init_public_repo(tmp_path)
    run_git(tmp_path, "config", "user.name", "GitHub")
    run_git(tmp_path, "config", "user.email", "noreply" + "@" + "github.com")
    (tmp_path / "README.md").write_bytes(b"public")
    run_git(tmp_path, "add", "--", "README.md")
    wrong_email = "unexpected" + "@" + "example.test"
    run_git(
        tmp_path,
        "commit",
        f"--author=Unexpected Author <{wrong_email}>",
        "-m",
        "wrong author",
    )

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


def test_reliability_trace_schema_fails_closed_on_content_fields():
    module = release_module()
    row = {
        "qid": "stress-001",
        "answerable": True,
        "rank": 1,
        "top_score": 0.9,
        "threshold_refused": False,
        "elapsed_ms": 1.0,
        "question": "private",
    }

    issues = module.scan_trace_rows([row], "reliability", "trace.jsonl")

    assert issues == [
        {
            "path": "trace.jsonl",
            "category": "unexpected_trace_field",
            "location": "row 1 field question",
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
