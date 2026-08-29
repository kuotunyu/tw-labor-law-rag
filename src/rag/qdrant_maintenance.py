"""Fail-closed contracts for manual blue-green Qdrant maintenance.

This module is deliberately free of network clients.  It validates local
corpus evidence and candidate metadata, and it emits only a small redacted
receipt after a caller has completed both candidate builds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.corpus_audit import build_snapshot
from rag.ingestion.loader import load_law_data
from rag.models import SourceUnit

_CANDIDATE_BASE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_DATE_YYYYMMDD = re.compile(r"^\d{8}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
_OFFICIAL_LAW_PREFIX = "https://law.moj.gov.tw/"
_STRATEGIES = ("fixed", "structure")
_RECEIPT_ROOT = Path("eval/runs/qdrant-maintenance")


@dataclass(frozen=True)
class AuditedCorpus:
    """One immutable-in-practice view used by validation and candidate builds."""

    snapshot: dict[str, Any]
    units: tuple[SourceUnit, ...]


def candidate_collections(base: str) -> dict[str, str]:
    """Return the two candidate collection names for *base*."""
    return {strategy: f"{base}_{strategy}" for strategy in _STRATEGIES}


def validate_candidate_base(active_base: str, candidate_base: str) -> None:
    """Reject an active, ambiguous, or non-portable candidate base name."""
    if not _CANDIDATE_BASE.fullmatch(active_base):
        raise ValueError("active base must be a portable lowercase name")
    if candidate_base == active_base:
        raise ValueError("candidate base must differ from active base")
    if not _CANDIDATE_BASE.fullmatch(candidate_base):
        raise ValueError("candidate base must be a portable lowercase name")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_local_snapshot(
    *,
    source_archives: Mapping[str, tuple[str, Path]],
    laws_dir: Path,
    snapshot_date: str,
) -> dict[str, Any]:
    """Reconstruct a canonical snapshot from already-downloaded local files."""
    return audit_local_corpus(
        source_archives=source_archives,
        laws_dir=laws_dir,
        snapshot_date=snapshot_date,
    ).snapshot


def audit_local_corpus(
    *,
    source_archives: Mapping[str, tuple[str, Path]],
    laws_dir: Path,
    snapshot_date: str,
) -> AuditedCorpus:
    """Read the exact top-level law JSON set once and reject every extra entry."""
    corpus_root = Path(laws_dir)
    if not corpus_root.is_dir():
        raise FileNotFoundError("local law corpus is unavailable")

    law_paths: list[Path] = []
    for path in sorted(corpus_root.rglob("*")):
        relative = path.relative_to(corpus_root)
        is_allowed_manifest = relative == Path("manifest.json") and path.is_file()
        is_top_level_law = (
            len(relative.parts) == 1
            and path.is_file()
            and path.suffix.lower() == ".json"
            and path.name != "manifest.json"
        )
        if path.is_symlink() or not (is_allowed_manifest or is_top_level_law):
            raise ValueError("unexpected corpus entry")
        if is_top_level_law:
            law_paths.append(path)
    if not law_paths:
        raise ValueError("law corpus contains no law JSON")

    sources = [
        {
            "id": source_id,
            "url": url,
            "sha256": _sha256_file(Path(path)),
        }
        for source_id, (url, path) in source_archives.items()
    ]
    laws: list[Mapping[str, Any]] = []
    units: list[SourceUnit] = []
    for path in law_paths:
        law = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(law, Mapping):
            raise ValueError("law JSON must be an object")
        laws.append(law)
        units.extend(load_law_data(law, source_path=path.name))
    snapshot = build_snapshot(sources=sources, laws=laws, snapshot_date=snapshot_date)
    return AuditedCorpus(snapshot=snapshot, units=tuple(units))


def _difference_paths(expected: Any, observed: Any, path: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        paths: list[str] = []
        all_keys = sorted(set(expected) | set(observed), key=str)
        for key in all_keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in observed:
                paths.append(child)
                continue
            paths.extend(_difference_paths(expected[key], observed[key], child))
        return paths
    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes, bytearray))
        and isinstance(observed, Sequence)
        and not isinstance(observed, (str, bytes, bytearray))
    ):
        paths = []
        if len(expected) != len(observed):
            paths.append(f"{path}.length")
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=False)
        ):
            paths.extend(
                _difference_paths(expected_item, observed_item, f"{path}[{index}]")
            )
        return paths
    return [] if expected == observed else [path or "root"]


def validate_snapshot_match(committed: Mapping[str, Any], local: Mapping[str, Any]) -> None:
    """Require exact snapshot equality except for the observation date."""
    expected = dict(committed)
    observed = dict(local)
    expected.pop("snapshot_date", None)
    observed.pop("snapshot_date", None)
    differences = _difference_paths(expected, observed)
    if differences:
        raise ValueError(f"corpus snapshot drift: {','.join(differences[:8])}")


def _validate_date(value: str, *, field: str, allow_empty: bool) -> None:
    if not value:
        if allow_empty:
            return
        raise ValueError(f"candidate payload {field} is required")
    if not _DATE_YYYYMMDD.fullmatch(value):
        raise ValueError(f"candidate payload {field} must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"candidate payload {field} must be YYYYMMDD") from exc


def validate_candidate_payloads(
    strategy: str,
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
) -> None:
    """Validate exact count and public citation provenance for a candidate."""
    if strategy not in _STRATEGIES:
        raise ValueError("candidate strategy is invalid")
    if expected_count < 0 or len(payloads) != expected_count:
        raise ValueError("candidate payload count mismatch")

    required_fields = (
        "doc_title",
        "article_label",
        "source_url",
        "last_amended",
        "effective_date",
    )
    for payload in payloads:
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"candidate payload missing {field}")

        for field in ("doc_title", "article_label", "source_url", "last_amended"):
            if not str(payload[field]).strip():
                raise ValueError(f"candidate payload {field} is required")

        source_url = str(payload["source_url"]).strip()
        if not source_url.startswith(_OFFICIAL_LAW_PREFIX):
            raise ValueError("candidate payload source_url must be official")

        _validate_date(
            str(payload["last_amended"]).strip(),
            field="last_amended",
            allow_empty=False,
        )
        _validate_date(
            str(payload["effective_date"]).strip(),
            field="effective_date",
            allow_empty=True,
        )


def _require_sha256(value: str, *, field: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def build_maintenance_receipt(
    *,
    completed_at: datetime,
    active_base: str,
    candidate_base: str,
    point_counts: Mapping[str, int],
    corpus_snapshot_sha256: str,
    source_sha256: Mapping[str, str],
    embedding_model: str,
    embedding_revision: str,
    vector_dimension: int,
) -> dict[str, Any]:
    """Build the exact allowlisted, credential-free maintenance receipt."""
    validate_candidate_base(active_base, candidate_base)
    if set(point_counts) != set(_STRATEGIES):
        raise ValueError("point_counts must contain fixed and structure")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in point_counts.values()):
        raise ValueError("point_counts must be non-negative integers")
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("completed_at must include a timezone")
    if not _MODEL_ID.fullmatch(embedding_model):
        raise ValueError("embedding_model must be a portable model identifier")
    if not _REVISION.fullmatch(embedding_revision):
        raise ValueError("embedding_revision must be a pinned commit")
    if isinstance(vector_dimension, bool) or not isinstance(vector_dimension, int) or vector_dimension <= 0:
        raise ValueError("vector_dimension must be a positive integer")

    sources: dict[str, str] = {}
    for source_id, digest in sorted(source_sha256.items()):
        if not _SOURCE_ID.fullmatch(source_id):
            raise ValueError("source id must be portable")
        sources[source_id] = _require_sha256(digest, field="source sha256")

    names = candidate_collections(candidate_base)
    completed_utc = completed_at.astimezone(timezone.utc).replace(microsecond=0)
    return {
        "schema_version": "1.0",
        "completed_at": completed_utc.isoformat().replace("+00:00", "Z"),
        "active_base": active_base,
        "candidate_base": candidate_base,
        "collections": {
            strategy: {"name": names[strategy], "points": point_counts[strategy]}
            for strategy in _STRATEGIES
        },
        "corpus_snapshot_sha256": _require_sha256(
            corpus_snapshot_sha256, field="corpus snapshot sha256"
        ),
        "source_sha256": sources,
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "vector_dimension": vector_dimension,
    }


def write_receipt_atomic(
    receipt: Mapping[str, Any],
    target: Path,
    *,
    project_root: Path,
) -> Path:
    """Atomically publish a receipt below the project maintenance-run folder."""
    target = Path(target)
    if target.is_absolute():
        raise ValueError("receipt target must be project-relative")

    root = Path(project_root).resolve()
    allowed_root = (root / _RECEIPT_ROOT).resolve()
    destination = (root / target).resolve()
    try:
        destination.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("receipt target must be under qdrant-maintenance") from exc
    if destination == allowed_root:
        raise ValueError("receipt target must name a file under qdrant-maintenance")
    if destination.exists():
        raise FileExistsError("maintenance receipt already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(receipt, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError("maintenance receipt already exists") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
