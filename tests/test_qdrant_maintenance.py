from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag.corpus_audit import build_snapshot
from rag.qdrant_maintenance import (
    build_local_snapshot,
    build_maintenance_receipt,
    candidate_collections,
    validate_candidate_base,
    validate_candidate_payloads,
    validate_snapshot_match,
    write_receipt_atomic,
)


def _law(
    name: str = "測試法",
    *,
    effective_date: str = "",
) -> dict[str, object]:
    return {
        "name": name,
        "nature": "法律",
        "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0000001",
        "last_amended": "20260829",
        "effective_date": effective_date,
        "articles": [
            {"no": "第 1 條", "chapter": "第一章", "content": "第一條內容。"},
        ],
    }


def _snapshot() -> dict[str, object]:
    return build_snapshot(
        sources=[
            {
                "id": "acts",
                "url": "https://sendlaw.moj.gov.tw/acts",
                "sha256": "a" * 64,
            }
        ],
        laws=[_law()],
        snapshot_date="2026-08-29",
    )


def _valid_payload(*, effective_date: str = "") -> dict[str, object]:
    return {
        "doc_title": "勞動基準法",
        "article_label": "第 1 條",
        "source_url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",
        "last_amended": "20240731",
        "effective_date": effective_date,
    }


def test_candidate_base_must_be_distinct_and_portable():
    base = "labor_laws_20260830_deadbeef"
    assert candidate_collections(base) == {
        "fixed": f"{base}_fixed",
        "structure": f"{base}_structure",
    }

    with pytest.raises(ValueError, match="active base"):
        validate_candidate_base("labor_laws", "labor_laws")

    for invalid in ("ab", "_candidate", "Labor Laws/next", "a" * 65):
        with pytest.raises(ValueError, match="portable"):
            validate_candidate_base("labor_laws", invalid)


def test_snapshot_match_ignores_only_observation_date():
    committed = _snapshot()
    local = deepcopy(committed)
    local["snapshot_date"] = "2026-08-30"
    validate_snapshot_match(committed, local)

    local["laws"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content_sha256"):
        validate_snapshot_match(committed, local)


def test_snapshot_match_rejects_metadata_not_reported_by_legacy_diff_helper():
    committed = _snapshot()
    local = deepcopy(committed)
    local["sources"][0]["url"] = "https://example.invalid/changed"

    with pytest.raises(ValueError, match="sources.*url"):
        validate_snapshot_match(committed, local)


def test_build_local_snapshot_hashes_archives_and_ignores_manifest(tmp_path: Path):
    archive = tmp_path / "acts.zip"
    archive.write_bytes(b"official archive bytes")
    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / "test-law.json").write_text(
        json.dumps(_law(), ensure_ascii=False), encoding="utf-8"
    )
    (laws_dir / "manifest.json").write_text("{}", encoding="utf-8")

    observed = build_local_snapshot(
        source_archives={
            "acts": ("https://sendlaw.moj.gov.tw/acts", archive),
        },
        laws_dir=laws_dir,
        snapshot_date="2026-08-30",
    )

    assert observed["law_count"] == 1
    assert observed["article_count"] == 1
    assert observed["sources"] == [
        {
            "id": "acts",
            "url": "https://sendlaw.moj.gov.tw/acts",
            "sha256": hashlib.sha256(b"official archive bytes").hexdigest(),
        }
    ]


def test_payloads_require_public_provenance_and_exact_count():
    payload = _valid_payload()
    validate_candidate_payloads("structure", [payload], expected_count=1)

    for field in ("doc_title", "article_label", "source_url", "last_amended"):
        broken = dict(payload)
        broken[field] = ""
        with pytest.raises(ValueError, match=field):
            validate_candidate_payloads("structure", [broken], expected_count=1)

    broken = dict(payload)
    del broken["effective_date"]
    with pytest.raises(ValueError, match="effective_date"):
        validate_candidate_payloads("structure", [broken], expected_count=1)

    with pytest.raises(ValueError, match="count"):
        validate_candidate_payloads("structure", [payload], expected_count=2)


def test_payload_effective_date_may_be_officially_unknown_but_not_malformed():
    validate_candidate_payloads("fixed", [_valid_payload()], expected_count=1)
    validate_candidate_payloads(
        "fixed", [_valid_payload(effective_date="20260101")], expected_count=1
    )

    with pytest.raises(ValueError, match="effective_date"):
        validate_candidate_payloads(
            "fixed", [_valid_payload(effective_date="2026-01-01")], expected_count=1
        )


def test_payload_rejects_unknown_strategy_and_nonofficial_url():
    payload = _valid_payload()
    with pytest.raises(ValueError, match="strategy"):
        validate_candidate_payloads("other", [payload], expected_count=1)

    payload["source_url"] = "https://example.invalid/law"
    with pytest.raises(ValueError, match="source_url"):
        validate_candidate_payloads("fixed", [payload], expected_count=1)


def _receipt() -> dict[str, object]:
    return build_maintenance_receipt(
        completed_at=datetime(2026, 8, 30, 1, 2, 3, tzinfo=timezone.utc),
        active_base="labor_laws",
        candidate_base="labor_laws_20260830_deadbeef",
        point_counts={"fixed": 100, "structure": 884},
        corpus_snapshot_sha256="c" * 64,
        source_sha256={"acts": "a" * 64, "regulations": "b" * 64},
        embedding_model="BAAI/bge-m3",
        embedding_revision="d111557e9e14b7eb54e935b5cc6b049a7f16dd65",
        vector_dimension=1024,
    )


def test_receipt_has_exact_redacted_schema():
    receipt = _receipt()

    assert set(receipt) == {
        "schema_version",
        "completed_at",
        "active_base",
        "candidate_base",
        "collections",
        "corpus_snapshot_sha256",
        "source_sha256",
        "embedding_model",
        "embedding_revision",
        "vector_dimension",
    }
    assert receipt["completed_at"] == "2026-08-30T01:02:03Z"
    assert receipt["collections"]["fixed"] == {
        "name": "labor_laws_20260830_deadbeef_fixed",
        "points": 100,
    }
    serialized = json.dumps(receipt)
    assert "https://" not in serialized
    assert "api_key" not in serialized.lower()
    assert ("D:" + "\\\\") not in serialized


def test_receipt_write_is_atomic_and_restricted_to_maintenance_directory(
    tmp_path: Path,
):
    target = Path("eval/runs/qdrant-maintenance/receipt.json")
    written = write_receipt_atomic(_receipt(), target, project_root=tmp_path)

    assert written == tmp_path / target
    assert json.loads(written.read_text(encoding="utf-8")) == _receipt()
    assert list(written.parent.glob("*.tmp")) == []

    with pytest.raises(ValueError, match="project-relative"):
        write_receipt_atomic(_receipt(), tmp_path / "absolute.json", project_root=tmp_path)
    with pytest.raises(ValueError, match="qdrant-maintenance"):
        write_receipt_atomic(
            _receipt(), Path("eval/runs/elsewhere/receipt.json"), project_root=tmp_path
        )
