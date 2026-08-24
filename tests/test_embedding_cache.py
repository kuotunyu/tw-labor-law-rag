import threading

import numpy as np

from rag.indexing.embedder import BGEM3Embedder, EmbeddingCache


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
