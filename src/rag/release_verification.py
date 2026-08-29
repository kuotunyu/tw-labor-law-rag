"""Offline verification for the public portfolio release contract.

This module only reads committed text artifacts. It never instantiates a model,
provider client, vector store, API lifespan, or network service.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

from rag.config import Settings
from rag.evaluation import canonical_text_sha256, compute_e2e_metrics
from rag.reliability import (
    PUBLIC_TRACE_FIELDS,
    compute_reliability_metrics,
    pareto_better_thresholds,
)

ABLATION_FIELDS = frozenset(
    {
        "answerable",
        "chunking",
        "elapsed_ms",
        "qid",
        "rank",
        "reranker",
        "retrieval",
        "top_score",
    }
)
E2E_FIELDS = frozenset(
    {
        "answerable",
        "cited_sources",
        "elapsed_ms",
        "judge",
        "q_type",
        "qid",
        "refusal_stage",
        "refused",
        "top_score",
    }
)
JUDGE_FIELDS = frozenset({"faithfulness", "relevancy"})
CITED_SOURCE_FIELDS = frozenset({"article", "doc"})
RELIABILITY_FIELDS = frozenset(PUBLIC_TRACE_FIELDS)
PROVIDER_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-5.6-luna",
}
PROVIDER_CROSSCHECK_SNAPSHOT_DATE = "2026-08-29"
PROVIDER_CROSSCHECK_QIDS = (
    "stress-001",
    "stress-002",
    "stress-003",
    "stress-042",
    "stress-043",
)
PROVIDER_RELIABILITY_TRACE_PATH = "eval/official/reliability_trace.jsonl"
PROVIDER_PRICING = {
    "gemini": {
        "input_per_million_usd": "0.30",
        "output_per_million_usd": "2.50",
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
    },
    "openai": {
        "input_per_million_usd": "0.20",
        "output_per_million_usd": "1.20",
        "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    },
}
PROVIDER_TRACE_FIELDS = frozenset(
    {
        "qid",
        "answerable",
        "requested_provider",
        "actual_provider",
        "model",
        "refused",
        "citation_count",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "refusal_verdict",
        "citation_verdict",
        "elapsed_ms",
    }
)
PROVIDER_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_date",
        "dataset",
        "corpus_snapshot",
        "selection",
        "authorization",
        "pricing",
        "provider_status",
        "provider_metrics",
        "budget_ledgers",
        "privacy",
    }
)
PROVIDER_METRIC_FIELDS = frozenset(
    {
        "requests",
        "refusal_accuracy",
        "citation_success_rate",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "avg_latency_ms",
    }
)
PROVIDER_LEDGER_FIELDS = frozenset(
    {
        "cap_usd",
        "spent_usd",
        "remaining_usd",
        "requests",
        "input_tokens",
        "output_tokens",
        "input_per_million_usd",
        "output_per_million_usd",
    }
)

RUNTIME_CONFIG_FIELDS = (
    "embedding_model",
    "reranker_model",
    "retrieval_mode",
    "use_reranker",
    "top_k_retrieve",
    "top_k_final",
    "rrf_k",
    "rerank_score_threshold",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
)

SENSITIVE_PUBLIC_PATHS = (
    ".claude/",
    ".env",
    ".venv/",
    ".worktrees/",
    "data/raw/",
    "docs/superpowers/",
    "eval/runs/",
    "indexes/",
    "prompts/",
    "provider/",
    "storage/",
    "answers/",
    "INTERVIEW_PREP.md",
    "STARTUP.md",
    "plan.md",
)

_LOCAL_PATH = re.compile(
    r"(?i)(?:(?<![A-Za-z0-9])[A-Z]:[\\/]|(?<![:/])/(?:Users|home)/[^/\s]+/)"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_STANDALONE_TOKEN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,})"
)
_EMAIL_ADDRESS = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_IP_ADDRESS = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
)
_PROVIDER_PAYLOAD_KEY = re.compile(
    r'(?i)"(?:api[_-]?key|judge_reason|prompt|provider_response|request_id|response_body|system_prompt|token_usage)"\s*:'
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(?:export[ \t]+)?(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY)[ \t]*=[ \t]*([^\r\n#]*)"
)

BINARY_PUBLIC_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".pdf"})
SOURCE_ARCHIVE_GENERATED_PREFIXES = (
    ".cache/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "build/",
    "dist/",
    "htmlcov/",
    "storage/",
)

TARGET_LAW_NAMES = frozenset(
    {
        "勞動基準法",
        "勞工退休金條例",
        "勞工保險條例",
        "性別平等工作法",
        "職業安全衛生法",
        "就業服務法",
        "最低工資法",
        "勞資爭議處理法",
        "勞動事件法",
        "勞工職業災害保險及保護法",
        "工會法",
        "團體協約法",
        "大量解僱勞工保護法",
        "勞動基準法施行細則",
        "勞工請假規則",
    }
)
CORPUS_SOURCE_URLS = {
    "acts": "https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=XML&AuData=CF",
    "regulations": "https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=XML&AuData=CM",
}


class ReleaseVerificationError(ValueError):
    """Raised when committed release evidence violates its contract."""


def _verify_evidence_run_date(
    value: object,
    *,
    latest_valid_date: date | None = None,
) -> date:
    """Validate a civil run date without rejecting valid UTC+14 evidence in UTC CI."""

    try:
        run_date = date.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ReleaseVerificationError(
            "reliability run date is not ISO YYYY-MM-DD"
        ) from exc
    if latest_valid_date is None:
        latest_valid_date = datetime.now(
            timezone(timedelta(hours=14))
        ).date()
    if run_date > latest_valid_date:
        raise ReleaseVerificationError("reliability run date is in the future")
    return run_date


@dataclass(frozen=True)
class PublicEntry:
    path: str
    data: bytes


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReleaseVerificationError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ReleaseVerificationError(f"{label}: actual={actual!r}, expected={expected!r}")


def _assert_public_equal(label: str, actual: Any, expected: Any) -> None:
    """Reject public-evidence mismatches without reflecting artifact values into logs."""
    expected_type = type(expected)
    if (
        actual != expected
        or (expected_type in (bool, int, str) and type(actual) is not expected_type)
    ):
        raise ReleaseVerificationError(f"{label} mismatch")


def _compare_public_tree(label: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        try:
            valid = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
            )
        except (TypeError, ValueError):
            valid = False
        _assert_public_equal(label, valid, True)
        return
    if isinstance(expected, dict):
        _assert_public_equal(label, isinstance(actual, Mapping), True)
        _assert_public_equal(f"{label} fields", set(actual), set(expected))
        for key in expected:
            _compare_public_tree(f"{label}.{key}", actual[key], expected[key])
        return
    if isinstance(expected, list):
        _assert_public_equal(label, isinstance(actual, list), True)
        _assert_public_equal(f"{label} length", len(actual), len(expected))
        for index, item in enumerate(expected):
            _compare_public_tree(f"{label}[{index}]", actual[index], item)
        return
    _assert_public_equal(label, actual, expected)


def _assert_close(
    label: str, actual: float, expected: float, *, abs_tol: float = 1e-12
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=abs_tol):
        raise ReleaseVerificationError(
            f"{label}: actual={actual!r}, expected={expected!r}"
        )


def _compare_tree(label: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        _assert_close(label, float(actual), expected)
        return
    if isinstance(expected, dict):
        _assert_equal(f"{label} keys", set(actual), set(expected))
        for key in expected:
            _compare_tree(f"{label}.{key}", actual[key], expected[key])
        return
    if isinstance(expected, list):
        _assert_equal(f"{label} length", len(actual), len(expected))
        for index, item in enumerate(expected):
            _compare_tree(f"{label}[{index}]", actual[index], item)
        return
    _assert_equal(label, actual, expected)


def _issue(path: str, category: str, location: str) -> dict[str, str]:
    return {"path": path, "category": category, "location": location}


def _is_sensitive_public_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    for forbidden in SENSITIVE_PUBLIC_PATHS:
        if forbidden.endswith("/") and normalized.startswith(forbidden):
            return True
        if normalized == forbidden:
            return True
    return False


def _is_placeholder_secret(raw: str) -> bool:
    value = raw.strip().rstrip(",").strip("'\"")
    lowered = value.lower()
    return (
        not value
        or lowered in {"dummy", "fake", "test"}
        or lowered.startswith("your_")
        or lowered.startswith("example")
        or lowered.startswith("replace_")
        or value.startswith("${")
        or (value.startswith("<") and value.endswith(">"))
    )


def _scan_public_entries(
    entries: Sequence[PublicEntry],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for entry in entries:
        normalized = entry.path.replace("\\", "/")
        suffix = Path(normalized).suffix.lower()
        if _is_sensitive_public_path(normalized):
            issues.append(_issue(normalized, "sensitive_public_path", "path"))
        if suffix in BINARY_PUBLIC_SUFFIXES:
            continue
        try:
            text = entry.data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(_issue(normalized, "non_utf8_public_text", "file"))
            continue

        categories: set[tuple[str, str]] = set()
        if _LOCAL_PATH.search(text):
            categories.add(("local_path", "text"))
        if _PRIVATE_KEY.search(text):
            categories.add(("private_key", "text"))
        if _STANDALONE_TOKEN.search(text):
            categories.add(("provider_token", "text"))
        private_email = any(
            not match.group(0).lower().endswith("@users.noreply.github.com")
            for match in _EMAIL_ADDRESS.finditer(text)
        )
        if private_email or (
            suffix in {".json", ".jsonl"} and _IP_ADDRESS.search(text)
        ):
            categories.add(("personal_identifier", "text"))
        if suffix in {".json", ".jsonl"} and _PROVIDER_PAYLOAD_KEY.search(text):
            categories.add(("provider_payload", "JSON field"))
        for match in _SECRET_ASSIGNMENT.finditer(text):
            if not _is_placeholder_secret(match.group(1)):
                line_number = text.count("\n", 0, match.start()) + 1
                categories.add(("secret_assignment", f"line {line_number}"))
        for category, location in sorted(categories):
            issues.append(_issue(normalized, category, location))

    return sorted(
        issues,
        key=lambda item: (item["path"], item["category"], item["location"]),
    )


def scan_public_files(
    project_root: Path, relative_paths: Sequence[str]
) -> list[dict[str, str]]:
    """Return sanitized privacy/secret issues without returning matched values."""

    issues: list[dict[str, str]] = []
    entries: list[PublicEntry] = []
    for relative_path in relative_paths:
        normalized = relative_path.replace("\\", "/")
        path = project_root / Path(normalized)
        if not path.is_file():
            if _is_sensitive_public_path(normalized):
                issues.append(_issue(normalized, "sensitive_public_path", "path"))
            issues.append(_issue(normalized, "missing_public_file", "path"))
            continue
        entries.append(PublicEntry(path=normalized, data=path.read_bytes()))
    return sorted(
        [*issues, *_scan_public_entries(entries)],
        key=lambda item: (item["path"], item["category"], item["location"]),
    )


def scan_trace_rows(
    rows: Sequence[Mapping[str, Any]], artifact_type: str, path: str
) -> list[dict[str, str]]:
    """Check privacy-reduced trace schemas, failing closed on unknown fields."""

    if artifact_type == "ablation":
        expected = ABLATION_FIELDS
    elif artifact_type == "e2e":
        expected = E2E_FIELDS
    elif artifact_type == "reliability":
        expected = RELIABILITY_FIELDS
    else:
        raise ValueError(f"unknown trace artifact type: {artifact_type!r}")

    issues: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        for field in sorted(set(row) - expected):
            issues.append(
                _issue(path, "unexpected_trace_field", f"row {index} field {field}")
            )
        for field in sorted(expected - set(row)):
            issues.append(_issue(path, "missing_trace_field", f"row {index} field {field}"))
        if artifact_type != "e2e":
            continue
        judge = row.get("judge")
        if judge is not None:
            if not isinstance(judge, Mapping):
                issues.append(_issue(path, "invalid_judge_shape", f"row {index}"))
            else:
                for field in sorted(set(judge) - JUDGE_FIELDS):
                    issues.append(
                        _issue(
                            path,
                            "unexpected_judge_field",
                            f"row {index} judge field {field}",
                        )
                    )
                for field in sorted(JUDGE_FIELDS - set(judge)):
                    issues.append(
                        _issue(
                            path,
                            "missing_judge_field",
                            f"row {index} judge field {field}",
                        )
                    )
        cited_sources = row.get("cited_sources")
        if isinstance(cited_sources, list):
            for source_index, source in enumerate(cited_sources, start=1):
                if not isinstance(source, Mapping):
                    issues.append(
                        _issue(
                            path,
                            "invalid_cited_source_shape",
                            f"row {index} source {source_index}",
                        )
                    )
                    continue
                for field in sorted(set(source) - CITED_SOURCE_FIELDS):
                    issues.append(
                        _issue(
                            path,
                            "unexpected_cited_source_field",
                            f"row {index} source {source_index} field {field}",
                        )
                    )
                for field in sorted(CITED_SOURCE_FIELDS - set(source)):
                    issues.append(
                        _issue(
                            path,
                            "missing_cited_source_field",
                            f"row {index} source {source_index} field {field}",
                        )
                    )
    return issues


def scan_action_pins(workflow_path: Path, display_path: str) -> list[dict[str, str]]:
    """Report third-party Actions references that are not full commit SHAs."""

    issues: list[dict[str, str]] = []
    for line_number, line in enumerate(
        workflow_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "uses:" not in line:
            continue
        reference = line.split("uses:", 1)[1].split("#", 1)[0].strip()
        if reference.startswith("./"):
            continue
        if not re.fullmatch(r"[^@\s]+@[0-9a-fA-F]{40}", reference):
            issues.append(
                _issue(display_path, "mutable_action_ref", f"line {line_number}")
            )
    return issues


def _load_public_file_list(path: Path) -> list[str]:
    paths = [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    _assert_equal("publication allowlist duplicates", len(paths), len(set(paths)))
    _assert_equal("publication allowlist sort order", paths, sorted(paths))
    return paths


def _tracked_files(project_root: Path) -> set[str] | None:
    if not (project_root / ".git").exists():
        return None
    process = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in process.stdout.split(b"\0")
        if item
    }


def _publishable_commit_ids(project_root: Path) -> list[str]:
    try:
        process = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "rev-list",
                "--branches",
                "--tags",
                "--exclude=pull/*",
                "--remotes",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ReleaseVerificationError(
            "failed to enumerate publishable Git refs"
        ) from exc
    commits = sorted({line for line in process.stdout.splitlines() if line})
    if not commits:
        raise ReleaseVerificationError("Git checkout has no publishable commits")
    return commits


def _parse_git_archive(data: bytes, *, commit: str) -> list[PublicEntry]:
    entries: list[PublicEntry] = []
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive.getmembers():
                raw = member.name
                candidate = raw[:-1] if member.isdir() and raw.endswith("/") else raw
                posix = PurePosixPath(candidate)
                if (
                    not candidate
                    or candidate.startswith("/")
                    or "\\" in candidate
                    or not posix.parts
                    or ":" in posix.parts[0]
                    or ".." in posix.parts
                    or posix.as_posix() != candidate
                ):
                    raise ReleaseVerificationError(
                        f"unsafe Git archive path in commit {commit}"
                    )
                if candidate in seen:
                    raise ReleaseVerificationError(
                        f"duplicate Git archive path in commit {commit}: {candidate}"
                    )
                seen.add(candidate)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseVerificationError(
                        f"unsupported Git archive entry in commit {commit}: {raw}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReleaseVerificationError(
                        f"unreadable Git archive entry in commit {commit}: {raw}"
                    )
                entries.append(PublicEntry(path=candidate, data=handle.read()))
    except (tarfile.TarError, OSError) as exc:
        raise ReleaseVerificationError(
            f"failed to parse Git archive for commit {commit}"
        ) from exc
    return sorted(entries, key=lambda entry: entry.path)


def _git_archive_entries(project_root: Path, commit: str) -> list[PublicEntry]:
    try:
        process = subprocess.run(
            ["git", "-C", str(project_root), "archive", "--format=tar", commit],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ReleaseVerificationError(
            f"failed to read Git archive for commit {commit}"
        ) from exc
    return _parse_git_archive(process.stdout, commit=commit)


def _verify_publishable_git_history(
    project_root: Path,
    current_public_paths: set[str],
    *,
    legacy_public_paths: set[str],
    reviewed_binary_hashes: set[str],
) -> int:
    commits = _publishable_commit_ids(project_root)
    allowed_paths = current_public_paths | legacy_public_paths
    owner_identity = (
        "kuotunyu",
        "61350295+kuotunyu@users.noreply.github.com",
    )
    allowed_identities = {
        (
            *owner_identity,
            *owner_identity,
        ),
        (
            *owner_identity,
            "GitHub",
            "noreply" + "@" + "github.com",
        ),
    }
    for commit in commits:
        entries = _git_archive_entries(project_root, commit)
        issues = _scan_public_entries(entries)
        for entry in entries:
            if entry.path not in allowed_paths:
                issues.append(_issue(entry.path, "unexpected_history_path", commit))
            if Path(entry.path).suffix.lower() in BINARY_PUBLIC_SUFFIXES:
                digest = hashlib.sha256(entry.data).hexdigest()
                if digest not in reviewed_binary_hashes:
                    issues.append(
                        _issue(entry.path, "unreviewed_history_binary", commit)
                    )
        if issues:
            sanitized = [
                {**issue, "commit": commit}
                for issue in sorted(
                    issues,
                    key=lambda item: (
                        item["path"],
                        item["category"],
                        item["location"],
                    ),
                )
            ]
            raise ReleaseVerificationError(
                f"publishable history issues: {sanitized}"
            )
        try:
            identity_process = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project_root),
                    "show",
                    "-s",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                    commit,
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            raise ReleaseVerificationError(
                f"failed to read identity for commit {commit}"
            ) from exc
        identity = tuple(identity_process.stdout.strip().split("\0"))
        if identity not in allowed_identities:
            raise ReleaseVerificationError(
                f"public commit identity {commit}: actual={identity!r}, "
                f"allowed={sorted(allowed_identities)!r}"
            )
    return len(commits)


def _verify_locked_ruff(project_root: Path) -> str:
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    dev_requirements = project["dependency-groups"]["dev"]
    if not any(re.match(r"^ruff(?:\W|$)", requirement) for requirement in dev_requirements):
        raise ReleaseVerificationError("Ruff is missing from the dev dependency group")
    versions = [
        package["version"]
        for package in lock["package"]
        if package.get("name") == "ruff" and package.get("version")
    ]
    _assert_equal("locked Ruff package count", len(versions), 1)
    return versions[0]


def _verify_ci_publication_contract(workflow_path: Path) -> dict[str, Any]:
    workflow = workflow_path.read_text(encoding="utf-8")
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
    missing = [command for command in required_commands if command not in workflow]
    _assert_equal("CI required commands", missing, [])
    positions = [workflow.index(command) for command in required_commands]
    _assert_equal("CI command order", positions, sorted(positions))
    if not re.search(r"(?m)^\s+branches:\s*\[main\]\s*$", workflow):
        raise ReleaseVerificationError("CI push branch trigger is not exactly main")
    if not re.search(r'(?m)^\s+tags:\s*\["v\*"\]\s*$', workflow):
        raise ReleaseVerificationError('CI tag trigger is not tags: ["v*"]')
    if not re.search(r"(?m)^\s+pull_request:\s*$", workflow):
        raise ReleaseVerificationError("CI pull_request trigger is missing")
    checkout_with_full_history = re.search(
        r"(?m)^[ \t]+- uses: actions/checkout@[0-9a-fA-F]{40}[^\n]*\n"
        r"[ \t]+with:\n(?:[ \t]+[^\n]*\n)*?"
        r"[ \t]+fetch-depth:\s*0[ \t]*$",
        workflow,
    )
    if checkout_with_full_history is None:
        raise ReleaseVerificationError("CI checkout does not fetch full Git history")
    return {
        "lint": True,
        "tag_trigger": "v*",
        "full_history_checkout": True,
    }


def _is_generated_source_archive_file(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    parts = normalized.split("/")
    return (
        normalized in {".coverage"}
        or normalized.endswith((".pyc", ".pyo"))
        or any(part == "__pycache__" or part.endswith(".egg-info") for part in parts)
        or any(normalized.startswith(prefix) for prefix in SOURCE_ARCHIVE_GENERATED_PREFIXES)
    )


def _source_archive_extra_files(
    project_root: Path, public_paths: Sequence[str]
) -> list[str]:
    allowed = {path.replace("\\", "/") for path in public_paths}
    return sorted(
        path.relative_to(project_root).as_posix()
        for path in project_root.rglob("*")
        if path.is_file()
        and path.relative_to(project_root).as_posix() not in allowed
        and not _is_generated_source_archive_file(
            path.relative_to(project_root).as_posix()
        )
    )


def _verify_reviewed_binaries(
    project_root: Path,
    public_paths: Sequence[str],
    reviewed_binaries: Mapping[str, str],
) -> int:
    binary_paths = sorted(
        path
        for path in public_paths
        if Path(path).suffix.lower() in BINARY_PUBLIC_SUFFIXES
    )
    _assert_equal(
        "reviewed public binary set",
        binary_paths,
        sorted(reviewed_binaries),
    )
    for relative_path in binary_paths:
        digest = hashlib.sha256((project_root / relative_path).read_bytes()).hexdigest()
        _assert_equal(
            f"binary SHA-256 {relative_path}",
            digest,
            reviewed_binaries[relative_path],
        )
    return len(binary_paths)


def _verify_e2e_threshold_contract(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> None:
    for row in rows:
        qid = row.get("qid", "unknown")
        stage = row.get("refusal_stage")
        top_score = row.get("top_score")
        if stage == "no_hits":
            if top_score is not None:
                raise ReleaseVerificationError(
                    f"threshold contract {qid}: no_hits must have no top_score"
                )
            continue
        if top_score is None:
            raise ReleaseVerificationError(
                f"threshold contract {qid}: retrieval result requires top_score"
            )
        score = float(top_score)
        if not math.isfinite(score):
            raise ReleaseVerificationError(
                f"threshold contract {qid}: top_score must be finite"
            )
        if stage == "threshold" and score >= threshold:
            raise ReleaseVerificationError(
                f"threshold contract {qid}: threshold refusal score is not below gate"
            )
        if stage != "threshold" and score < threshold:
            raise ReleaseVerificationError(
                f"threshold contract {qid}: score below gate reached generation layer"
            )


def _config_defaults() -> dict[str, Any]:
    return {
        field: Settings.model_fields[field].default for field in RUNTIME_CONFIG_FIELDS
    }


def _config_key(chunking: str, retrieval: str, reranker: bool) -> str:
    suffix = "hybrid+rerank" if retrieval == "hybrid" and reranker else retrieval
    return f"{chunking}/{suffix}"


def _verify_ablation(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    dataset_by_qid: Mapping[str, Mapping[str, Any]],
    expected_configs: set[str],
) -> dict[str, dict[str, float]]:
    _assert_equal("ablation row count", len(rows), 320)
    config_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = row["qid"]
        if qid not in dataset_by_qid:
            raise ReleaseVerificationError(f"ablation unknown qid: {qid}")
        _assert_equal(
            f"ablation {qid} answerable", row["answerable"], dataset_by_qid[qid]["answerable"]
        )
        key = _config_key(row["chunking"], row["retrieval"], row["reranker"])
        config_groups.setdefault(key, []).append(row)

    _assert_equal("ablation configuration set", set(config_groups), expected_configs)
    for key, group in config_groups.items():
        _assert_equal(f"{key} qid count", len(group), len(dataset_by_qid))
        _assert_equal(f"{key} qids", {row["qid"] for row in group}, set(dataset_by_qid))

    summaries = {
        _config_key(row["chunking"], row["retrieval"], row["reranker"]): row
        for row in result["results"]
    }
    _assert_equal("ablation summary configuration set", set(summaries), expected_configs)

    recomputed: dict[str, dict[str, float]] = {}
    for key, group in config_groups.items():
        answerable = [row for row in group if row["answerable"]]
        _assert_equal(f"{key} answerable count", len(answerable), 30)
        hit_at_5 = sum(
            row["rank"] is not None and row["rank"] <= 5 for row in answerable
        ) / len(answerable)
        mrr_at_10 = sum(
            1 / row["rank"]
            for row in answerable
            if row["rank"] is not None and row["rank"] <= 10
        ) / len(answerable)
        latency = sum(float(row["elapsed_ms"]) for row in answerable) / len(answerable)
        summary = summaries[key]
        _assert_close(f"{key} Hit@5", hit_at_5, float(summary["hit_at_5"]))
        _assert_close(f"{key} MRR@10", mrr_at_10, float(summary["mrr_at_10"]))
        _assert_close(
            f"{key} latency", latency, float(summary["avg_latency_ms"]), abs_tol=0.1
        )
        recomputed[key] = {
            "hit_at_5": hit_at_5,
            "mrr_at_10": mrr_at_10,
            "avg_latency_ms": latency,
        }
    return recomputed


def _verify_source_data(project_root: Path, source_data: Mapping[str, Any]) -> int:
    _assert_equal("source dataset id", source_data["dataset_id"], 18290)
    _assert_equal("source provider", source_data["provider"], "法務部資訊處")
    _assert_equal(
        "source license", source_data["license"], "政府資料開放授權條款－第1版"
    )
    _assert_equal(
        "source redistribution",
        source_data["redistribution"],
        "allowed_with_attribution",
    )
    samples = source_data["samples"]
    _assert_equal("source sample count", len(samples), 2)
    for sample in samples:
        relative_path = sample["path"]
        path = project_root / relative_path
        _assert_equal(f"sample exists {relative_path}", path.is_file(), True)
        _assert_equal(
            f"sample SHA-256 {relative_path}",
            canonical_text_sha256(path),
            sample["sha256"],
        )
        payload = _read_json(path)
        _assert_equal(
            f"sample last amended {relative_path}",
            payload["last_amended"],
            sample["last_amended"],
        )
        _assert_equal(f"sample nature {relative_path}", payload["nature"], "命令")
        if not str(payload["url"]).startswith("https://law.moj.gov.tw/"):
            raise ReleaseVerificationError(f"sample source URL is not law.moj.gov.tw: {relative_path}")
    return len(samples)


def _verify_full_corpus_snapshot(
    project_root: Path,
    snapshot_contract: Mapping[str, Any],
) -> dict[str, Any]:
    relative_path = snapshot_contract["path"]
    path = project_root / relative_path
    _assert_equal(f"corpus snapshot exists {relative_path}", path.is_file(), True)
    snapshot = _read_json(path)

    _assert_equal(
        "corpus snapshot schema",
        snapshot.get("schema_version"),
        snapshot_contract["schema_version"],
    )
    _assert_equal(
        "corpus snapshot date",
        snapshot.get("snapshot_date"),
        snapshot_contract["snapshot_date"],
    )
    try:
        date.fromisoformat(snapshot["snapshot_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseVerificationError("corpus snapshot date is not ISO YYYY-MM-DD") from exc

    sources = snapshot.get("sources", [])
    source_ids = [row.get("id") for row in sources]
    _assert_equal("corpus snapshot source ids", source_ids, sorted(CORPUS_SOURCE_URLS))
    for source in sources:
        source_id = source["id"]
        _assert_equal(
            f"corpus snapshot source URL {source_id}",
            source.get("url"),
            CORPUS_SOURCE_URLS[source_id],
        )
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
            raise ReleaseVerificationError(f"corpus snapshot source SHA-256: {source_id}")

    laws = snapshot.get("laws", [])
    law_names = [law.get("name") for law in laws]
    _assert_equal("corpus snapshot sorted law names", law_names, sorted(law_names))
    _assert_equal("corpus snapshot unique law names", len(set(law_names)), len(law_names))
    _assert_equal("corpus snapshot target laws", set(law_names), set(TARGET_LAW_NAMES))
    _assert_equal("corpus snapshot law count", snapshot.get("law_count"), len(laws))
    _assert_equal("corpus snapshot contract laws", len(laws), snapshot_contract["laws"])

    article_count = 0
    for law in laws:
        name = law["name"]
        if law.get("nature") not in {"法律", "命令"}:
            raise ReleaseVerificationError(f"corpus snapshot law nature: {name}")
        if not str(law.get("url", "")).startswith("https://law.moj.gov.tw/"):
            raise ReleaseVerificationError(f"corpus snapshot law URL: {name}")
        if not re.fullmatch(r"\d{8}", str(law.get("last_amended", ""))):
            raise ReleaseVerificationError(f"corpus snapshot last amended: {name}")
        effective_date = str(law.get("effective_date", ""))
        if effective_date and not re.fullmatch(r"\d{8}", effective_date):
            raise ReleaseVerificationError(f"corpus snapshot effective date: {name}")
        num_articles = law.get("num_articles")
        if not isinstance(num_articles, int) or isinstance(num_articles, bool) or num_articles <= 0:
            raise ReleaseVerificationError(f"corpus snapshot article count: {name}")
        article_count += num_articles
        if not re.fullmatch(r"[0-9a-f]{64}", str(law.get("content_sha256", ""))):
            raise ReleaseVerificationError(f"corpus snapshot content SHA-256: {name}")

    _assert_equal("corpus snapshot article arithmetic", snapshot.get("article_count"), article_count)
    _assert_equal(
        "corpus snapshot contract articles",
        article_count,
        snapshot_contract["articles"],
    )
    return {
        "snapshot_date": snapshot["snapshot_date"],
        "laws": len(laws),
        "articles": article_count,
    }


def _verify_reliability_evidence(
    project_root: Path,
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    formal_trace_rows: Sequence[Mapping[str, Any]],
    *,
    formal_dataset_rows: Sequence[Mapping[str, Any]],
    formal_dataset_sha: str,
    runtime_config: Mapping[str, Any],
    snapshot_contract: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_contract = contract["dataset"]
    dataset_path = project_root / dataset_contract["path"]
    dataset_rows = _read_jsonl(dataset_path)
    dataset_by_qid = {row["qid"]: row for row in dataset_rows}
    _assert_equal("reliability unique qids", len(dataset_by_qid), len(dataset_rows))
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    _assert_equal("reliability dataset SHA-256", dataset_sha, dataset_contract["sha256"])
    _assert_equal("reliability questions", len(dataset_rows), dataset_contract["questions"])
    _assert_equal(
        "reliability answerable",
        sum(bool(row["answerable"]) for row in dataset_rows),
        dataset_contract["answerable"],
    )
    _assert_equal(
        "reliability unanswerable",
        sum(not bool(row["answerable"]) for row in dataset_rows),
        dataset_contract["unanswerable"],
    )

    _assert_equal("reliability trace rows", len(trace_rows), len(dataset_rows))
    _assert_equal(
        "reliability trace qids",
        {row["qid"] for row in trace_rows},
        set(dataset_by_qid),
    )
    production_threshold = float(contract["production_threshold"])
    for row in trace_rows:
        qid = row["qid"]
        _assert_equal(
            f"reliability {qid} answerable",
            row["answerable"],
            dataset_by_qid[qid]["answerable"],
        )
        _assert_equal(
            f"reliability {qid} threshold decision",
            row["threshold_refused"],
            float(row["top_score"]) < production_threshold,
        )
    try:
        metrics = compute_reliability_metrics(trace_rows, contract["threshold_candidates"])
    except ValueError as exc:
        raise ReleaseVerificationError(f"invalid reliability trace: {exc}") from exc

    _assert_equal("reliability schema", result["schema_version"], "1.0")
    try:
        _verify_evidence_run_date(result["run_date"])
    except KeyError as exc:
        raise ReleaseVerificationError(
            "reliability run date is not ISO YYYY-MM-DD"
        ) from exc
    _assert_equal(
        "reliability result dataset",
        result["dataset"],
        {"path": dataset_contract["path"], "sha256": dataset_sha},
    )
    _assert_equal(
        "reliability formal guard dataset",
        result["formal_guard_dataset"],
        {"path": "eval/dataset/eval_set.jsonl", "sha256": formal_dataset_sha},
    )
    _assert_equal(
        "reliability corpus snapshot",
        result["corpus_snapshot"],
        {
            "path": snapshot_contract["path"],
            "snapshot_date": snapshot_contract["snapshot_date"],
            "laws": snapshot_contract["laws"],
            "articles": snapshot_contract["articles"],
        },
    )
    defaults = Settings(_env_file=None)
    _assert_equal(
        "reliability configuration",
        result["configuration"],
        {
            "chunking": runtime_config["chunking_strategy"],
            "retrieval": runtime_config["retrieval_mode"],
            "reranker": runtime_config["use_reranker"],
            "top_k_retrieve": runtime_config["top_k_retrieve"],
            "top_k_final": runtime_config["top_k_final"],
            "embedding_model": runtime_config["embedding_model"],
            "embedding_revision": defaults.embedding_model_revision,
            "reranker_model": runtime_config["reranker_model"],
            "reranker_revision": defaults.reranker_model_revision,
        },
    )
    _assert_equal(
        "reliability production threshold",
        result["production_threshold"],
        contract["production_threshold"],
    )
    _assert_equal(
        "reliability threshold candidates",
        result["threshold_candidates"],
        contract["threshold_candidates"],
    )
    _compare_tree("reliability stress metrics", metrics, result["stress_metrics"])

    production_key = str(production_threshold)
    production = metrics["threshold_sweep"][production_key]
    expected_stress = contract["stress"]
    _assert_close("reliability stress hit@5", metrics["hit_at_5"], expected_stress["hit_at_5"])
    _assert_close(
        "reliability stress MRR@10", metrics["mrr_at_10"], expected_stress["mrr_at_10"]
    )
    _assert_equal(
        "reliability stress direct false refusals",
        production["direct_false_refusals"],
        expected_stress["direct_false_refusals"],
    )
    _assert_close(
        "reliability stress direct unanswerable coverage",
        production["direct_unanswerable_coverage"],
        expected_stress["direct_unanswerable_coverage"],
    )

    formal_dataset_by_qid = {row["qid"]: row for row in formal_dataset_rows}
    _assert_equal(
        "reliability formal guard unique qids",
        len(formal_dataset_by_qid),
        len(formal_dataset_rows),
    )
    _assert_equal(
        "reliability formal guard trace rows",
        len(formal_trace_rows),
        len(formal_dataset_rows),
    )
    _assert_equal(
        "reliability formal guard trace qids",
        {row["qid"] for row in formal_trace_rows},
        set(formal_dataset_by_qid),
    )
    for row in formal_trace_rows:
        qid = row["qid"]
        _assert_equal(
            f"reliability formal guard {qid} answerable",
            row["answerable"],
            formal_dataset_by_qid[qid]["answerable"],
        )
        _assert_equal(
            f"reliability formal guard {qid} threshold decision",
            row["threshold_refused"],
            float(row["top_score"]) < production_threshold,
        )
    try:
        formal_metrics = compute_reliability_metrics(
            formal_trace_rows,
            contract["threshold_candidates"],
        )
    except ValueError as exc:
        raise ReleaseVerificationError(
            f"invalid reliability formal guard trace: {exc}"
        ) from exc
    _compare_tree(
        "reliability formal guard metrics",
        formal_metrics,
        result["formal_guard_metrics"],
    )
    expected_formal = contract["formal_guard"]
    _assert_close(
        "reliability formal hit@5", formal_metrics["hit_at_5"], expected_formal["hit_at_5"]
    )
    _assert_close(
        "reliability formal MRR@10",
        formal_metrics["mrr_at_10"],
        expected_formal["mrr_at_10"],
    )
    formal_production = formal_metrics["threshold_sweep"][production_key]
    _assert_equal(
        "reliability formal direct false refusals",
        formal_production["direct_false_refusals"],
        expected_formal["direct_false_refusals"],
    )
    _assert_close(
        "reliability formal direct unanswerable coverage",
        formal_production["direct_unanswerable_coverage"],
        expected_formal["direct_unanswerable_coverage"],
    )
    try:
        pareto_candidates = pareto_better_thresholds(
            metrics,
            formal_metrics,
            production=production_threshold,
        )
    except ValueError as exc:
        raise ReleaseVerificationError(f"invalid reliability decision: {exc}") from exc
    outcome = "retain_0.03" if not pareto_candidates else "manual_review_required"
    _assert_equal("reliability decision contract", contract["decision"], outcome)
    _assert_equal(
        "reliability decision",
        result["decision"],
        {
            "pareto_better_candidates": pareto_candidates,
            "outcome": outcome,
            "automatic_config_change": False,
        },
    )
    return {
        "questions": metrics["questions"],
        "hit_at_5": metrics["hit_at_5"],
        "mrr_at_10": metrics["mrr_at_10"],
        "direct_false_refusals": production["direct_false_refusals"],
        "direct_unanswerable_coverage": production["direct_unanswerable_coverage"],
        "decision": contract["decision"],
    }


def _verify_release_version_contract(
    project_root: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_version = project["project"]["version"]
    release_version = manifest["release_version"]
    formal_evidence_version = manifest["formal_evidence_version"]
    _assert_equal("package release version", f"v{package_version}", release_version)
    _assert_equal("formal evidence version", formal_evidence_version, "v0.1.0")
    _assert_equal("release version", release_version, "v0.3.4")
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    readme_en = (project_root / "README.en.md").read_text(encoding="utf-8")
    release_phrase = f"`{release_version}` source-only runtime and deployment release"
    if release_phrase not in readme:
        raise ReleaseVerificationError("README release wording mismatch")
    if release_phrase not in readme_en:
        raise ReleaseVerificationError("README.en release wording mismatch")
    return {
        "version": release_version,
        "package_version": package_version,
        "formal_evidence_version": formal_evidence_version,
    }


def _verify_provider_crosscheck_contract(
    project_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    pending_contract = {
        "status": "pending_credentials",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": ["gemini", "openai"],
        "results_path": "eval/official/provider_crosscheck_results.json",
        "trace_path": "eval/official/provider_crosscheck_trace.jsonl",
    }
    if contract.get("status") == "pending_credentials":
        _compare_public_tree("provider cross-check pending contract", contract, pending_contract)
        for artifact_key in ("results_path", "trace_path"):
            if (project_root / contract[artifact_key]).exists():
                raise ReleaseVerificationError(
                    "pending provider cross-check must not publish unverified artifacts"
                )
        return {
            "status": contract["status"],
            "authorized_cap_usd_per_provider": contract[
                "authorized_cap_usd_per_provider"
            ],
            "required_providers": contract["required_providers"],
        }

    complete_contract = {
        **pending_contract,
        "status": "complete",
        "dataset": {
            "path": "eval/dataset/reliability_stress_v0.3.1.jsonl",
            "sha256": "7641d78e8434d8832319a70af019c1e0d860079a23fca161b488497e1c6b1b7f",
        },
    }
    _compare_public_tree("provider cross-check complete contract", contract, complete_contract)

    results_path = project_root / contract["results_path"]
    trace_path = project_root / contract["trace_path"]
    _assert_equal("provider cross-check results exists", results_path.is_file(), True)
    _assert_equal("provider cross-check trace exists", trace_path.is_file(), True)
    results = _read_json(results_path)
    rows = _read_jsonl(trace_path)
    _assert_public_equal(
        "provider cross-check results fields", set(results), PROVIDER_RESULT_FIELDS
    )
    _assert_public_equal(
        "provider cross-check result schema", results["schema_version"], "1.0"
    )
    _verify_evidence_run_date(results["run_date"])
    _assert_public_equal(
        "provider cross-check run date",
        results["run_date"],
        PROVIDER_CROSSCHECK_SNAPSHOT_DATE,
    )
    _assert_public_equal(
        "provider cross-check dataset", results["dataset"], contract["dataset"]
    )
    dataset_path = project_root / contract["dataset"]["path"]
    _assert_public_equal("provider cross-check dataset exists", dataset_path.is_file(), True)
    _assert_public_equal(
        "provider cross-check dataset SHA-256",
        hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        contract["dataset"]["sha256"],
    )
    dataset_rows = _read_jsonl(dataset_path)
    dataset_by_qid = {row.get("qid"): row for row in dataset_rows}
    _assert_public_equal(
        "provider cross-check dataset unique qids",
        len(dataset_by_qid),
        len(dataset_rows),
    )
    reliability_path = project_root / PROVIDER_RELIABILITY_TRACE_PATH
    _assert_public_equal(
        "provider cross-check reliability trace exists",
        reliability_path.is_file(),
        True,
    )
    reliability_rows = _read_jsonl(reliability_path)
    reliability_by_qid = {row.get("qid"): row for row in reliability_rows}
    _assert_public_equal(
        "provider cross-check reliability unique qids",
        len(reliability_by_qid),
        len(reliability_rows),
    )
    _assert_public_equal(
        "provider cross-check reliability qid coverage",
        set(reliability_by_qid),
        set(dataset_by_qid),
    )
    eligible_answerable_qids: list[str] = []
    eligible_unanswerable_qids: list[str] = []
    for dataset_row in dataset_rows:
        qid = dataset_row["qid"]
        reliability_row = reliability_by_qid[qid]
        _assert_public_equal(
            "provider cross-check reliability answerable",
            reliability_row.get("answerable"),
            dataset_row["answerable"],
        )
        _assert_public_equal(
            "provider cross-check reliability threshold type",
            isinstance(reliability_row.get("threshold_refused"), bool),
            True,
        )
        if not reliability_row["threshold_refused"]:
            target = (
                eligible_answerable_qids
                if dataset_row["answerable"]
                else eligible_unanswerable_qids
            )
            target.append(qid)
    expected_qids = [
        *eligible_answerable_qids[:3],
        *eligible_unanswerable_qids[:2],
    ]
    _assert_public_equal(
        "provider cross-check deterministic selection size",
        len(expected_qids),
        5,
    )
    _assert_public_equal(
        "provider cross-check deterministic selected qids",
        expected_qids,
        list(PROVIDER_CROSSCHECK_QIDS),
    )
    _assert_public_equal(
        "provider cross-check corpus snapshot",
        results["corpus_snapshot"],
        {
            "path": "release/corpus_snapshot.json",
            "snapshot_date": PROVIDER_CROSSCHECK_SNAPSHOT_DATE,
            "laws": 15,
            "articles": 884,
        },
    )
    _assert_public_equal(
        "provider cross-check selection",
        results["selection"],
        {
            "initial_per_provider": 5,
            "maximum_per_provider": 5,
            "generation_eligible_only": True,
        },
    )
    _assert_public_equal(
        "provider cross-check authorization",
        results["authorization"],
        {
            "per_provider_cap_usd": "5.00",
            "max_input_tokens_per_request": 20_000,
            "max_output_tokens_per_request": 1_024,
        },
    )
    expected_pricing = {
        provider: {"model": PROVIDER_MODELS[provider], **PROVIDER_PRICING[provider]}
        for provider in contract["required_providers"]
    }
    _assert_public_equal(
        "provider cross-check pricing", results["pricing"], expected_pricing
    )
    _assert_public_equal(
        "provider cross-check provider status",
        results["provider_status"],
        {
            provider: {"status": "complete", "reason": None}
            for provider in contract["required_providers"]
        },
    )
    _assert_public_equal(
        "provider cross-check privacy",
        results["privacy"],
        {
            "public_trace_contains_question_or_answer": False,
            "public_trace_contains_provider_payload": False,
            "public_trace_contains_credentials": False,
            "raw_trace_path": "ignored eval/runs only",
        },
    )
    _assert_public_equal("provider cross-check trace rows", len(rows), 10)

    grouped: dict[str, list[dict[str, Any]]] = {
        provider: [] for provider in contract["required_providers"]
    }
    for row in rows:
        _assert_public_equal(
            "provider cross-check trace fields", set(row), PROVIDER_TRACE_FIELDS
        )
        provider = row["requested_provider"]
        _assert_public_equal(
            "provider cross-check requested provider type",
            type(provider) is str,
            True,
        )
        _assert_public_equal(
            "provider cross-check requested provider",
            provider in grouped,
            True,
        )
        _assert_public_equal(
            "provider cross-check actual provider", row["actual_provider"], provider
        )
        _assert_public_equal(
            f"provider cross-check {provider} model",
            row["model"],
            PROVIDER_MODELS[provider],
        )
        _assert_public_equal(
            "provider cross-check qid type",
            isinstance(row["qid"], str)
            and bool(re.fullmatch(r"stress-\d{3}", row["qid"])),
            True,
        )
        for field in ("answerable", "refused"):
            _assert_public_equal(
                f"provider cross-check {field} type",
                isinstance(row[field], bool),
                True,
            )
        for field in ("citation_count", "input_tokens", "output_tokens"):
            _assert_public_equal(
                f"provider cross-check {field} type",
                type(row[field]) is int and row[field] >= 0,
                True,
            )
        _assert_public_equal(
            "provider cross-check refusal verdict type",
            type(row["refusal_verdict"]) is int,
            True,
        )
        _assert_public_equal(
            "provider cross-check refusal verdict",
            row["refusal_verdict"],
            int(row["refused"] == (not row["answerable"])),
        )
        _assert_public_equal(
            "provider cross-check citation verdict type",
            type(row["citation_verdict"]) is int,
            True,
        )
        _assert_public_equal(
            "provider cross-check citation verdict",
            row["citation_verdict"],
            int(
                (row["refused"] and row["citation_count"] == 0)
                or (not row["refused"] and row["citation_count"] > 0)
            ),
        )
        _assert_public_equal(
            "provider cross-check latency",
            isinstance(row["elapsed_ms"], (int, float))
            and not isinstance(row["elapsed_ms"], bool)
            and math.isfinite(float(row["elapsed_ms"]))
            and row["elapsed_ms"] >= 0,
            True,
        )
        _assert_public_equal(
            "provider cross-check input tokens maximum",
            row["input_tokens"] <= results["authorization"]["max_input_tokens_per_request"],
            True,
        )
        _assert_public_equal(
            "provider cross-check output tokens maximum",
            row["output_tokens"] <= results["authorization"]["max_output_tokens_per_request"],
            True,
        )
        _assert_public_equal(
            "provider cross-check dataset qids", row["qid"] in dataset_by_qid, True
        )
        _assert_public_equal(
            "provider cross-check dataset answerable",
            row["answerable"],
            dataset_by_qid[row["qid"]]["answerable"],
        )
        _assert_public_equal(
            "provider cross-check generation eligibility",
            reliability_by_qid[row["qid"]]["threshold_refused"],
            False,
        )
        expected_cost = _provider_cost(
            provider,
            row["input_tokens"],
            row["output_tokens"],
        )
        _assert_public_equal(
            f"provider cross-check {provider} row cost",
            row["estimated_cost_usd"],
            str(expected_cost),
        )
        grouped[provider].append(row)

    requests: dict[str, int] = {}
    estimated_costs: dict[str, str] = {}
    for provider in contract["required_providers"]:
        provider_rows = grouped[provider]
        qids = [row["qid"] for row in provider_rows]
        _assert_public_equal(
            f"provider cross-check {provider} rows", len(provider_rows), 5
        )
        _assert_public_equal(
            f"provider cross-check {provider} unique qids", len(set(qids)), 5
        )
        _assert_public_equal(
            f"provider cross-check {provider} answerable rows",
            sum(row["answerable"] for row in provider_rows),
            3,
        )
        _assert_public_equal(
            f"provider cross-check {provider} unanswerable rows",
            sum(not row["answerable"] for row in provider_rows),
            2,
        )
        _assert_public_equal(
            f"provider cross-check {provider} selected qids",
            qids,
            expected_qids,
        )
        metrics = _provider_metrics(provider, provider_rows)
        _assert_public_equal(
            "provider cross-check provider metrics mapping",
            isinstance(results["provider_metrics"], Mapping),
            True,
        )
        _assert_public_equal(
            f"provider cross-check {provider} metric fields",
            set(results["provider_metrics"].get(provider, {})),
            PROVIDER_METRIC_FIELDS,
        )
        _compare_public_tree(
            f"provider cross-check {provider}",
            metrics,
            results["provider_metrics"][provider],
        )
        ledger = {
            "cap_usd": "5.00",
            "spent_usd": metrics["estimated_cost_usd"],
            "remaining_usd": str(Decimal("5.00") - Decimal(metrics["estimated_cost_usd"])),
            "requests": metrics["requests"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "input_per_million_usd": PROVIDER_PRICING[provider]["input_per_million_usd"],
            "output_per_million_usd": PROVIDER_PRICING[provider]["output_per_million_usd"],
        }
        _assert_public_equal(
            "provider cross-check budget ledgers mapping",
            isinstance(results["budget_ledgers"], Mapping),
            True,
        )
        _assert_public_equal(
            f"provider cross-check {provider} ledger fields",
            set(results["budget_ledgers"].get(provider, {})),
            PROVIDER_LEDGER_FIELDS,
        )
        _assert_public_equal(
            f"provider cross-check {provider} ledger",
            results["budget_ledgers"][provider],
            ledger,
        )
        _assert_public_equal(
            f"provider cross-check {provider} budget cap",
            Decimal(metrics["estimated_cost_usd"]) <= Decimal("5.00"),
            True,
        )
        requests[provider] = metrics["requests"]
        estimated_costs[provider] = metrics["estimated_cost_usd"]

    _assert_public_equal(
        "provider cross-check metrics providers",
        set(results["provider_metrics"]),
        set(contract["required_providers"]),
    )
    _assert_public_equal(
        "provider cross-check ledger providers",
        set(results["budget_ledgers"]),
        set(contract["required_providers"]),
    )
    return {
        "status": "complete",
        "authorized_cap_usd_per_provider": "5.00",
        "required_providers": contract["required_providers"],
        "requests": requests,
        "estimated_cost_usd": estimated_costs,
    }


def _provider_cost(provider: str, input_tokens: int, output_tokens: int) -> Decimal:
    try:
        input_rate = Decimal(PROVIDER_PRICING[provider]["input_per_million_usd"])
        output_rate = Decimal(PROVIDER_PRICING[provider]["output_per_million_usd"])
    except (KeyError, InvalidOperation) as exc:
        raise ReleaseVerificationError("provider cross-check pricing configuration") from exc
    return (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / Decimal(1_000_000)


def _provider_metrics(provider: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ReleaseVerificationError("provider cross-check provider rows must not be empty")
    return {
        "requests": count,
        "refusal_accuracy": sum(row["refusal_verdict"] for row in rows) / count,
        "citation_success_rate": sum(row["citation_verdict"] for row in rows) / count,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "estimated_cost_usd": str(
            sum(
                (_provider_cost(provider, row["input_tokens"], row["output_tokens"]) for row in rows),
                Decimal(0),
            )
        ),
        "avg_latency_ms": round(sum(float(row["elapsed_ms"]) for row in rows) / count, 1),
    }


def verify_release(project_root: Path) -> dict[str, Any]:
    """Verify all committed release evidence and return a deterministic summary."""

    root = project_root.resolve()
    manifest = _read_json(root / "release" / "manifest.json")
    _assert_equal("release schema", manifest.get("schema_version"), "1.0")
    release_contract = _verify_release_version_contract(root, manifest)
    _assert_equal(
        "release base commit",
        manifest.get("base_commit"),
        "24bd48f9a5ae0257f962e7401cbb3c759fbefe35",
    )

    dataset_path = root / manifest["evidence"]["dataset"]["path"]
    dataset = _read_jsonl(dataset_path)
    dataset_by_qid = {row["qid"]: row for row in dataset}
    _assert_equal("dataset unique qids", len(dataset_by_qid), len(dataset))
    answerable = [row for row in dataset if row["answerable"]]
    unanswerable = [row for row in dataset if not row["answerable"]]
    dataset_sha = canonical_text_sha256(dataset_path)
    expected_dataset = manifest["evidence"]["dataset"]
    _assert_equal("dataset SHA-256", dataset_sha, expected_dataset["canonical_sha256"])
    _assert_equal("dataset questions", len(dataset), expected_dataset["questions"])
    _assert_equal("dataset answerable", len(answerable), expected_dataset["answerable"])
    _assert_equal("dataset unanswerable", len(unanswerable), expected_dataset["unanswerable"])
    _assert_equal(
        "dataset question types",
        dict(sorted(Counter(row["q_type"] for row in dataset).items())),
        expected_dataset["question_types"],
    )
    _assert_equal(
        "dataset law coverage",
        len({source["doc"] for row in answerable for source in row["sources"]}),
        expected_dataset["laws_covered"],
    )

    runtime_config = manifest["runtime_config"]
    _assert_equal("runtime config defaults", runtime_config, _config_defaults())

    official_dir = root / "eval" / "official"
    ablation_result = _read_json(official_dir / "ablation_results.json")
    e2e_result = _read_json(official_dir / "e2e_results.json")
    reliability_contract = manifest["evidence"]["reliability"]
    reliability_result = _read_json(root / reliability_contract["results_path"])
    for label, result in (("ablation", ablation_result), ("e2e", e2e_result)):
        _assert_equal(f"{label} dataset SHA", result["dataset"]["sha256"], dataset_sha)
        _assert_equal(f"{label} dataset questions", result["dataset"]["n_questions"], len(dataset))
        _assert_equal(f"{label} settings", result["settings"], runtime_config)

    ablation_path = official_dir / "ablation_trace.jsonl"
    e2e_path = official_dir / "e2e_trace.jsonl"
    ablation_rows = _read_jsonl(ablation_path)
    e2e_rows = _read_jsonl(e2e_path)
    reliability_rows = _read_jsonl(root / reliability_contract["trace_path"])
    reliability_formal_rows = _read_jsonl(
        root / reliability_contract["formal_trace_path"]
    )
    trace_issues = scan_trace_rows(
        ablation_rows, "ablation", "eval/official/ablation_trace.jsonl"
    ) + scan_trace_rows(
        e2e_rows, "e2e", "eval/official/e2e_trace.jsonl"
    ) + scan_trace_rows(
        reliability_rows,
        "reliability",
        reliability_contract["trace_path"],
    ) + scan_trace_rows(
        reliability_formal_rows,
        "reliability",
        reliability_contract["formal_trace_path"],
    )
    if trace_issues:
        raise ReleaseVerificationError(f"official trace privacy/schema issues: {trace_issues}")
    _verify_e2e_threshold_contract(
        e2e_rows,
        float(runtime_config["rerank_score_threshold"]),
    )

    expected_configs = set(manifest["evidence"]["ablation"]["configurations"])
    ablation_metrics = _verify_ablation(
        ablation_rows, ablation_result, dataset_by_qid, expected_configs
    )
    _assert_equal(
        "manifest ablation rows", len(ablation_rows), manifest["evidence"]["ablation"]["rows"]
    )

    _assert_equal("e2e row count", len(e2e_rows), manifest["evidence"]["e2e"]["rows"])
    _assert_equal("e2e qids", {row["qid"] for row in e2e_rows}, set(dataset_by_qid))
    for row in e2e_rows:
        _assert_equal(
            f"e2e {row['qid']} answerable",
            row["answerable"],
            dataset_by_qid[row["qid"]]["answerable"],
        )
    metric_rows = [
        {key: value for key, value in row.items() if not (key == "judge" and value is None)}
        for row in e2e_rows
    ]
    e2e_metrics = compute_e2e_metrics(metric_rows)
    _compare_tree("e2e metrics", e2e_metrics, e2e_result["metrics"])
    _assert_equal("manifest e2e answered", e2e_metrics["n_answered"], manifest["evidence"]["e2e"]["answered"])
    _assert_equal(
        "manifest e2e refused",
        sum(1 for row in e2e_rows if row["refused"]),
        manifest["evidence"]["e2e"]["refused"],
    )
    _assert_equal("manifest e2e judged", e2e_metrics["n_judged"], manifest["evidence"]["e2e"]["judged"])
    _assert_equal("e2e generator", e2e_result["generator"], manifest["provider_evidence"]["generator"])
    _assert_equal("e2e judge", e2e_result["judge"], manifest["provider_evidence"]["judge"])
    _assert_equal(
        "provider evidence classification",
        manifest["provider_evidence"]["classification"],
        "archived_provider_evidence",
    )

    samples_verified = _verify_source_data(root, manifest["source_data"])
    full_snapshot = _verify_full_corpus_snapshot(
        root,
        manifest["source_data"]["full_snapshot"],
    )
    reliability_summary = _verify_reliability_evidence(
        root,
        reliability_contract,
        reliability_result,
        reliability_rows,
        reliability_formal_rows,
        formal_dataset_rows=dataset,
        formal_dataset_sha=dataset_sha,
        runtime_config=runtime_config,
        snapshot_contract=manifest["source_data"]["full_snapshot"],
    )
    provider_crosscheck_contract = manifest["evidence"]["provider_crosscheck"]
    provider_crosscheck_summary = _verify_provider_crosscheck_contract(
        root,
        provider_crosscheck_contract,
    )

    public_paths = _load_public_file_list(root / manifest["publication"]["allowlist"])
    history_config = manifest["publication"]["history"]
    _assert_equal(
        "publication history ref namespaces",
        history_config["refs"],
        ["heads", "tags", "remotes"],
    )
    legacy_public_paths = set(history_config["legacy_public_paths"])
    historical_binary_hashes = set(history_config["reviewed_binary_sha256"])
    current_binary_hashes = set(
        manifest["publication"]["reviewed_binaries"].values()
    )
    _assert_equal(
        "current reviewed binaries included in history",
        current_binary_hashes <= historical_binary_hashes,
        True,
    )
    public_issues = scan_public_files(root, public_paths)
    if public_issues:
        raise ReleaseVerificationError(f"public privacy/secret issues: {public_issues}")
    reviewed_binary_count = _verify_reviewed_binaries(
        root,
        public_paths,
        manifest["publication"]["reviewed_binaries"],
    )
    missing_exclusions = sorted(
        set(SENSITIVE_PUBLIC_PATHS) - set(manifest["publication"]["excluded"])
    )
    _assert_equal("manifest publication exclusions", missing_exclusions, [])
    _assert_equal(
        "public tracked_excluded",
        manifest["publication"]["tracked_excluded"],
        [],
    )
    tracked_files = _tracked_files(root)
    tracking_status = "not_applicable_no_git_metadata"
    archive_extra_count = 0
    history_commits = 0
    if tracked_files is not None:
        _assert_equal(
            "public Git tracked set",
            tracked_files,
            set(public_paths),
        )
        tracking_status = "exact_public_allowlist"
        history_commits = _verify_publishable_git_history(
            root,
            set(public_paths),
            legacy_public_paths=legacy_public_paths,
            reviewed_binary_hashes=historical_binary_hashes,
        )
    else:
        archive_extras = _source_archive_extra_files(root, public_paths)
        _assert_equal("source archive extra files", archive_extras, [])
        archive_extra_count = len(archive_extras)

    workflow_path = root / ".github" / "workflows" / "ci.yml"
    action_issues = scan_action_pins(workflow_path, ".github/workflows/ci.yml")
    if action_issues:
        raise ReleaseVerificationError(f"GitHub Actions pin issues: {action_issues}")
    action_pin_count = sum(
        "uses:" in line
        for line in workflow_path.read_text(encoding="utf-8").splitlines()
    )
    ci_contract = _verify_ci_publication_contract(workflow_path)
    ruff_version = _verify_locked_ruff(root)

    primary = ablation_metrics["structure/hybrid+rerank"]
    return {
        "status": "pass",
        "release": release_contract,
        "dataset": {
            "questions": len(dataset),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "dataset_sha256": dataset_sha,
        "ablation": {
            "configurations": len(ablation_metrics),
            "rows": len(ablation_rows),
        },
        "primary_retrieval": {
            "hit_at_5": primary["hit_at_5"],
            "mrr_at_10": primary["mrr_at_10"],
        },
        "reliability": reliability_summary,
        "e2e": {
            "answered": e2e_metrics["n_answered"],
            "refused": sum(1 for row in e2e_rows if row["refused"]),
            "generation_calls": e2e_metrics["n_generation_calls"],
            "refusal_by_stage": e2e_metrics["refusal_by_stage"],
            "refusal_accuracy": e2e_metrics["refusal_accuracy"],
            "false_refusal_rate": e2e_metrics["false_refusal_rate"],
            "threshold_contract": True,
        },
        "provider_evidence": {
            "classification": "archived_provider_evidence",
            "judged": e2e_metrics["n_judged"],
            "avg_faithfulness": e2e_metrics["avg_faithfulness"],
            "avg_relevancy": e2e_metrics["avg_relevancy"],
        },
        "provider_crosscheck": provider_crosscheck_summary,
        "privacy": {
            "official_trace_issues": len(trace_issues),
            "public_scan_issues": len(public_issues),
        },
        "source_data": {
            "dataset_id": manifest["source_data"]["dataset_id"],
            "license": manifest["source_data"]["license"],
            "redistribution": manifest["source_data"]["redistribution"],
            "samples_verified": samples_verified,
            "full_snapshot_date": full_snapshot["snapshot_date"],
            "full_snapshot_laws": full_snapshot["laws"],
            "full_snapshot_articles": full_snapshot["articles"],
        },
        "ci": {
            "action_pins": action_pin_count,
            "all_pinned": True,
            **ci_contract,
        },
        "tooling": {"ruff": ruff_version},
        "publication": {
            "files": len(public_paths),
            "tracking": tracking_status,
            "archive_extra_files": archive_extra_count,
            "reviewed_binaries": reviewed_binary_count,
            "history_commits": history_commits,
            "history_ref_namespaces": history_config["refs"],
        },
    }
