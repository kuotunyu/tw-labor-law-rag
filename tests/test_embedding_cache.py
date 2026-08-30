import threading
from types import SimpleNamespace

import numpy as np
import pytest

from rag.indexing.embedder import BGEM3Embedder, EmbeddingCache, resolve_model_snapshot
from rag.retrieval.reranker import Reranker, ensure_prepare_for_model


def test_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    v1 = np.arange(4, dtype=np.float32)
    v2 = np.ones(4, dtype=np.float32)
    cache.put_many({"k1": v1, "k2": v2})

    out = cache.get_many(["k1", "k2", "k-missing"])
    assert set(out) == {"k1", "k2"}
    np.testing.assert_array_equal(out["k1"], v1)
    assert len(cache) == 2


def test_cache_upsert_overwrites(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put_many({"k": np.zeros(3, dtype=np.float32)})
    cache.put_many({"k": np.ones(3, dtype=np.float32)})
    np.testing.assert_array_equal(cache.get_many(["k"])["k"], np.ones(3, dtype=np.float32))
    assert len(cache) == 1


def test_cache_handles_large_key_batches(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    items = {f"k{i}": np.full(2, i, dtype=np.float32) for i in range(1200)}
    cache.put_many(items)
    out = cache.get_many(list(items))
    assert len(out) == 1200


def test_cache_usable_from_multiple_threads(tmp_path):
    """Regression test: FastAPI serves requests from a threadpool, so the
    connection created at startup must be usable from worker threads too."""
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            cache.put_many({f"t{i}": np.full(4, i, dtype=np.float32)})
            cache.get_many([f"t{i}"])
        except Exception as exc:  # pragma: no cover - only hit on regression
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(cache) == 8


def test_embedder_uses_cache_without_model(tmp_path):
    """If every text is cached, the model must never be loaded."""
    embedder = BGEM3Embedder(cache_path=tmp_path / "emb.sqlite")
    texts = ["勞工請假", "特別休假"]
    keys = [embedder._key(t) for t in texts]
    embedder.cache.put_many(
        {keys[0]: np.zeros(4, dtype=np.float32), keys[1]: np.ones(4, dtype=np.float32)}
    )
    vectors = embedder.encode(texts)
    assert vectors.shape == (2, 4)
    assert embedder._model is None, "model should stay unloaded on full cache hit"
    np.testing.assert_array_equal(vectors[1], np.ones(4, dtype=np.float32))


def test_embedder_close_releases_embedding_cache_file(tmp_path):
    cache_path = tmp_path / "emb.sqlite"
    embedder = BGEM3Embedder(cache_path=cache_path)

    embedder.close()
    cache_path.unlink()

    assert not cache_path.exists()


def test_embedder_pins_revision_and_disables_remote_code(monkeypatch):
    captured = {}
    revision = "a" * 40

    def fake_snapshot_download(**kwargs):
        captured["snapshot"] = kwargs
        return "immutable-bge-m3-snapshot"

    def fake_model(model_name, **kwargs):
        captured.update(model_name=model_name, **kwargs)
        return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=fake_model),
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    embedder = BGEM3Embedder(
        model_name="BAAI/bge-m3",
        model_revision=revision,
        device="cpu",
    )

    assert embedder.model is not None
    assert captured["snapshot"] == {"repo_id": "BAAI/bge-m3", "revision": revision}
    assert captured["model_name"] == "immutable-bge-m3-snapshot"
    assert "revision" not in captured
    assert captured["trust_remote_code"] is False


def test_maintenance_embedder_resolves_snapshot_local_only(monkeypatch):
    captured = {}
    revision = "a" * 40

    def fake_snapshot_download(**kwargs):
        captured["snapshot"] = kwargs
        return "immutable-bge-m3-snapshot"

    def fake_model(model_name, **kwargs):
        captured.update(model_name=model_name, **kwargs)
        return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=fake_model),
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    embedder = BGEM3Embedder(
        model_name="BAAI/bge-m3",
        model_revision=revision,
        device="cpu",
        local_files_only=True,
    )

    assert embedder.model is not None
    assert captured["snapshot"] == {
        "repo_id": "BAAI/bge-m3",
        "revision": revision,
        "local_files_only": True,
    }


def test_embedding_cache_key_includes_model_revision():
    first = BGEM3Embedder(model_revision="revision-a", device="cpu")
    second = BGEM3Embedder(model_revision="revision-b", device="cpu")

    assert first._key("相同文字") != second._key("相同文字")


def test_reranker_pins_revision_and_disables_remote_code(monkeypatch):
    captured = {}
    revision = "b" * 40

    def fake_snapshot_download(**kwargs):
        captured["snapshot"] = kwargs
        return "immutable-bge-reranker-v2-m3-snapshot"

    def fake_model(model_name, **kwargs):
        captured.update(model_name=model_name, **kwargs)
        return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=fake_model),
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    reranker = Reranker(
        model_name="BAAI/bge-reranker-v2-m3",
        model_revision=revision,
        device="cpu",
    )

    assert reranker.model is not None
    assert captured["snapshot"] == {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "revision": revision,
    }
    assert captured["model_name"] == "immutable-bge-reranker-v2-m3-snapshot"
    assert "revision" not in captured
    assert captured["trust_remote_code"] is False


def test_model_snapshot_requires_full_commit_sha(monkeypatch):
    called = False

    def fake_snapshot_download(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    with pytest.raises(ValueError, match="40-character commit SHA"):
        resolve_model_snapshot("BAAI/bge-m3", "main")
    assert called is False


def test_reranker_restores_removed_prepare_for_model_api():
    class XLMRobertaTokenizer:
        bos_token_id = 0
        eos_token_id = 2

    tokenizer = XLMRobertaTokenizer()
    ensure_prepare_for_model(tokenizer)

    encoded = tokenizer.prepare_for_model(
        [10, 11],
        [20, 21, 22, 23],
        truncation="only_second",
        max_length=8,
        padding=False,
    )

    assert encoded == {
        "input_ids": [0, 10, 11, 2, 2, 20, 21, 2],
        "attention_mask": [1] * 8,
        "token_type_ids": [0] * 8,
    }
