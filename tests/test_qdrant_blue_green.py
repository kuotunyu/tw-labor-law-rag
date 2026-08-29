from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from rag.config import Settings
from rag.qdrant_blue_green import BuildDependencies, BuildRequest, build_candidates


class FakeStore:
    def __init__(self):
        self.existing: set[str] = set()
        self.checked: list[str] = []
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.payloads: dict[str, list[dict]] = {}
        self.fail_upsert_for: str | None = None
        self.count_offset = 0
        self.closed = False

    def collection_exists(self, name: str) -> bool:
        self.checked.append(name)
        return name in self.existing

    def create_collection(self, name: str, dim: int) -> None:
        assert dim == 4
        if name in self.existing:
            raise AssertionError("orchestrator must preflight every candidate")
        self.created.append(name)
        self.existing.add(name)

    def upsert_chunks(self, name, chunks, vectors) -> None:
        if name == self.fail_upsert_for:
            raise RuntimeError("simulated write failure")
        assert vectors.shape == (len(chunks), 4)
        self.payloads[name] = [chunk.payload() for chunk in chunks]

    def count(self, name: str) -> int:
        return len(self.payloads.get(name, [])) + self.count_offset

    def scroll_payloads(self, name: str) -> list[dict]:
        return self.payloads[name]

    def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.fail_on_call: int | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated local embedding failure")
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    path = tmp_path / "laws"
    path.mkdir()
    law = {
        "name": "測試法",
        "nature": "法律",
        "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0000001",
        "last_amended": "20260829",
        "effective_date": "",
        "articles": [
            {"no": "第 1 條", "chapter": "第一章", "content": "第一條測試內容。"},
            {"no": "第 2 條", "chapter": "第一章", "content": "第二條測試內容。"},
        ],
    }
    (path / "test-law.json").write_text(
        json.dumps(law, ensure_ascii=False), encoding="utf-8"
    )
    return path


@pytest.fixture
def build_request(corpus_dir: Path) -> BuildRequest:
    return BuildRequest(
        active_base="labor_laws",
        candidate_base="labor_laws_20260830_deadbeef",
        corpus_dir=corpus_dir,
        receipt_path=Path("eval/runs/qdrant-maintenance/test.json"),
        snapshot_sha256="c" * 64,
        source_sha256={"acts": "a" * 64, "regulations": "b" * 64},
    )


@pytest.fixture
def dependencies() -> BuildDependencies:
    return BuildDependencies(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        settings=Settings(
            _env_file=None,
            embedding_model="BAAI/bge-m3",
            embedding_model_revision="d" * 40,
            chunk_size=400,
            chunk_overlap=80,
        ),
        completed_at=lambda: datetime(2026, 8, 30, 1, 2, 3, tzinfo=timezone.utc),
    )


def test_existing_candidate_blocks_before_any_create(dependencies, build_request):
    dependencies.store.existing.add(build_request.collections["fixed"])

    with pytest.raises(ValueError, match="candidate collection exists"):
        build_candidates(build_request, dependencies)

    assert dependencies.store.checked == list(build_request.collections.values())
    assert dependencies.store.created == []
    assert dependencies.embedder.calls == []
    assert dependencies.store.deleted == []


def test_local_preparation_failure_happens_before_any_create(
    dependencies, build_request
):
    dependencies.embedder.fail_on_call = 2

    with pytest.raises(RuntimeError, match="local embedding failure"):
        build_candidates(build_request, dependencies)

    assert len(dependencies.embedder.calls) == 2
    assert dependencies.store.created == []
    assert dependencies.store.deleted == []


def test_write_failure_never_deletes_partial_candidate(dependencies, build_request):
    dependencies.store.fail_upsert_for = build_request.collections["structure"]

    with pytest.raises(RuntimeError, match="write failure"):
        build_candidates(build_request, dependencies)

    assert dependencies.store.created == [
        build_request.collections["fixed"],
        build_request.collections["structure"],
    ]
    assert dependencies.store.deleted == []


def test_count_mismatch_fails_closed_without_delete(dependencies, build_request):
    dependencies.store.count_offset = 1

    with pytest.raises(ValueError, match="count mismatch: fixed"):
        build_candidates(build_request, dependencies)

    assert dependencies.store.created == [build_request.collections["fixed"]]
    assert dependencies.store.deleted == []


def test_success_builds_fixed_then_structure_and_returns_redacted_receipt(
    dependencies, build_request
):
    receipt = build_candidates(build_request, dependencies)

    assert dependencies.store.created == [
        build_request.collections["fixed"],
        build_request.collections["structure"],
    ]
    assert receipt["collections"]["fixed"]["points"] == len(
        dependencies.store.payloads[build_request.collections["fixed"]]
    )
    assert receipt["collections"]["structure"]["points"] == 2
    assert receipt["vector_dimension"] == 4
    serialized = json.dumps(receipt)
    assert "https://" not in serialized
    assert "api_key" not in serialized.lower()


def test_request_derives_only_candidate_collection_names(build_request):
    assert build_request.collections == {
        "fixed": "labor_laws_20260830_deadbeef_fixed",
        "structure": "labor_laws_20260830_deadbeef_structure",
    }
