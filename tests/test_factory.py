from rag.config import Settings
from rag.factory import build_retrieval_pipeline
from rag.indexing.bm25_index import BM25Index
from rag.retrieval.retriever import BM25Retriever


def test_factory_uses_supplied_in_memory_bm25_without_disk_load(monkeypatch):
    supplied = BM25Index.from_payloads(
        [
            {"chunk_id": "a", "text": "工資應全額直接給付勞工"},
            {"chunk_id": "b", "text": "退休金相關規定"},
        ]
    )

    def fail_disk_load(_path):
        raise AssertionError("disk-backed BM25 must not load when an index is supplied")

    monkeypatch.setattr(BM25Index, "load", fail_disk_load)
    pipeline = build_retrieval_pipeline(
        Settings(_env_file=None),
        object(),
        object(),
        mode="bm25",
        use_reranker=False,
        bm25_index=supplied,
    )

    assert isinstance(pipeline.retriever, BM25Retriever)
    assert pipeline.retriever.index is supplied
