from types import SimpleNamespace

import numpy as np
import pytest

from rag.config import Settings
from rag.indexing import vector_store


def test_server_store_uses_secret_api_key_and_public_byok_blocks_writes(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    settings = Settings(
        _env_file=None,
        deployment_mode="public_byok",
        qdrant_mode="server",
        qdrant_url="https://demo.cloud.qdrant.io:6333",
        qdrant_api_key="read-only-secret",
    )

    store = vector_store.VectorStore(settings)

    assert captured == {
        "url": "https://demo.cloud.qdrant.io:6333",
        "api_key": "read-only-secret",
        "timeout": 60.0,
    }
    with pytest.raises(RuntimeError, match="read-only Qdrant runtime"):
        store.recreate_collection("labor_laws_structure", 1024)
    with pytest.raises(RuntimeError, match="read-only Qdrant runtime"):
        store.create_collection("labor_laws_candidate_structure", 1024)
    with pytest.raises(RuntimeError, match="read-only Qdrant runtime"):
        store.upsert_chunks("labor_laws_structure", [], np.empty((0, 1024)))


def test_create_collection_refuses_existing_without_delete(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def collection_exists(self, name):
            calls.append(("exists", name))
            return True

        def create_collection(self, **kwargs):
            calls.append(("create", kwargs))

        def delete_collection(self, name):
            calls.append(("delete", name))

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(
        Settings(_env_file=None, qdrant_mode="server", qdrant_url="https://example.test")
    )

    with pytest.raises(ValueError, match="already exists"):
        store.create_collection("labor_laws_candidate_structure", dim=1024)

    assert calls == [("exists", "labor_laws_candidate_structure")]


def test_create_collection_creates_absent_target_without_delete(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def collection_exists(self, name):
            calls.append(("exists", name))
            return False

        def create_collection(self, **kwargs):
            calls.append(("create", kwargs))

        def delete_collection(self, name):
            calls.append(("delete", name))

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(
        Settings(_env_file=None, qdrant_mode="server", qdrant_url="https://example.test")
    )

    store.create_collection("labor_laws_candidate_structure", dim=1024)

    assert calls[0] == ("exists", "labor_laws_candidate_structure")
    operation, kwargs = calls[1]
    assert operation == "create"
    assert kwargs["collection_name"] == "labor_laws_candidate_structure"
    assert kwargs["vectors_config"].size == 1024
    assert kwargs["vectors_config"].distance == vector_store.qm.Distance.COSINE
    assert len(calls) == 2


def test_upsert_rejects_chunk_vector_count_mismatch_when_assertions_are_disabled(
    monkeypatch,
):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def upsert(self, **_kwargs):
            pytest.fail("mismatched chunks and vectors must fail before Qdrant is called")

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(
        Settings(_env_file=None, qdrant_mode="server", qdrant_url="https://example.test")
    )

    with pytest.raises(ValueError, match="chunk and vector counts must match"):
        store.upsert_chunks(
            "labor_laws_structure", [], np.empty((1, 1024), dtype=np.float32)
        )


def test_scroll_payloads_reads_all_pages_without_vectors(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def scroll(self, **kwargs):
            calls.append(kwargs)
            if kwargs["offset"] is None:
                return (
                    [
                        SimpleNamespace(payload={"chunk_id": "a", "text": "甲"}),
                        SimpleNamespace(payload={"chunk_id": "b", "text": "乙"}),
                    ],
                    "next-page",
                )
            return ([SimpleNamespace(payload={"chunk_id": "c", "text": "丙"})], None)

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(
        Settings(_env_file=None, qdrant_mode="server", qdrant_url="https://example.test")
    )

    assert store.scroll_payloads("labor_laws_structure", batch_size=2) == [
        {"chunk_id": "a", "text": "甲"},
        {"chunk_id": "b", "text": "乙"},
        {"chunk_id": "c", "text": "丙"},
    ]
    assert calls == [
        {
            "collection_name": "labor_laws_structure",
            "limit": 2,
            "offset": None,
            "with_payload": True,
            "with_vectors": False,
        },
        {
            "collection_name": "labor_laws_structure",
            "limit": 2,
            "offset": "next-page",
            "with_payload": True,
            "with_vectors": False,
        },
    ]


@pytest.mark.parametrize("payload", [None, [], "not-a-dict"])
def test_scroll_payloads_rejects_missing_or_invalid_payload(monkeypatch, payload):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def scroll(self, **_kwargs):
            return ([SimpleNamespace(payload=payload)], None)

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(
        Settings(_env_file=None, qdrant_mode="server", qdrant_url="https://example.test")
    )

    with pytest.raises(ValueError, match="missing or invalid payload"):
        store.scroll_payloads("labor_laws_structure")
