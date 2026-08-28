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
from pathlib import Path, PurePosixPath
from typing import Any

from rag.config import Settings
from rag.evaluation import canonical_text_sha256, compute_e2e_metrics

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


class ReleaseVerificationError(ValueError):
    """Raised when committed release evidence violates its contract."""


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
    expected_identity = (
        "kuotunyu",
        "61350295+kuotunyu@users.noreply.github.com",
        "kuotunyu",
        "61350295+kuotunyu@users.noreply.github.com",
    )
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
        _assert_equal(
            f"public commit identity {commit}",
            identity,
            expected_identity,
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
    return {"lint": True, "tag_trigger": "v*"}


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


def verify_release(project_root: Path) -> dict[str, Any]:
    """Verify all committed release evidence and return a deterministic summary."""

    root = project_root.resolve()
    manifest = _read_json(root / "release" / "manifest.json")
    _assert_equal("release schema", manifest.get("schema_version"), "1.0")
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
    for label, result in (("ablation", ablation_result), ("e2e", e2e_result)):
        _assert_equal(f"{label} dataset SHA", result["dataset"]["sha256"], dataset_sha)
        _assert_equal(f"{label} dataset questions", result["dataset"]["n_questions"], len(dataset))
        _assert_equal(f"{label} settings", result["settings"], runtime_config)

    ablation_path = official_dir / "ablation_trace.jsonl"
    e2e_path = official_dir / "e2e_trace.jsonl"
    ablation_rows = _read_jsonl(ablation_path)
    e2e_rows = _read_jsonl(e2e_path)
    trace_issues = scan_trace_rows(
        ablation_rows, "ablation", "eval/official/ablation_trace.jsonl"
    ) + scan_trace_rows(e2e_rows, "e2e", "eval/official/e2e_trace.jsonl")
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
        "privacy": {
            "official_trace_issues": len(trace_issues),
            "public_scan_issues": len(public_issues),
        },
        "source_data": {
            "dataset_id": manifest["source_data"]["dataset_id"],
            "license": manifest["source_data"]["license"],
            "redistribution": manifest["source_data"]["redistribution"],
            "samples_verified": samples_verified,
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
