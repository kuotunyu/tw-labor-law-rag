import pytest

from rag.models import RetrievedChunk
from rag.retrieval.retriever import HybridRetriever, build_retriever


def hit(chunk_id, score=1.0):
    return RetrievedChunk(score=score, payload={"chunk_id": chunk_id})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


def test_hybrid_fuses_vector_and_bm25():
    vec = FakeRetriever([hit("a"), hit("b"), hit("c")])
    bm25 = FakeRetriever([hit("b"), hit("d")])
    hybrid = HybridRetriever(vec, bm25, rrf_k=60)

    fused = hybrid.retrieve("query", top_k=10)
    assert fused[0].payload["chunk_id"] == "b"  # ranked in both lists
    assert {h.payload["chunk_id"] for h in fused} == {"a", "b", "c", "d"}


def test_hybrid_respects_top_k():
    vec = FakeRetriever([hit("a"), hit("b"), hit("c")])
    bm25 = FakeRetriever([hit("d"), hit("e")])
    hybrid = HybridRetriever(vec, bm25)
    assert len(hybrid.retrieve("query", top_k=2)) == 2


def test_hybrid_fetch_k_overrides_top_k_for_candidates():
    vec = FakeRetriever([hit(f"v{i}") for i in range(20)])
    bm25 = FakeRetriever([hit(f"b{i}") for i in range(20)])
    hybrid = HybridRetriever(vec, bm25, fetch_k=20)
    hybrid.retrieve("query", top_k=5)
    assert vec.calls[0] == ("query", 20)
    assert bm25.calls[0] == ("query", 20)


def test_build_retriever_vector_mode():
    vec = FakeRetriever([hit("a")])
    assert build_retriever("vector", vector_retriever=vec) is vec


def test_build_retriever_bm25_mode():
    bm25 = FakeRetriever([hit("a")])
    assert build_retriever("bm25", bm25_retriever=bm25) is bm25


def test_build_retriever_hybrid_mode():
    vec, bm25 = FakeRetriever([hit("a")]), FakeRetriever([hit("b")])
    result = build_retriever("hybrid", vector_retriever=vec, bm25_retriever=bm25)
    assert isinstance(result, HybridRetriever)


def test_build_retriever_missing_dependency_raises():
    with pytest.raises(ValueError):
        build_retriever("vector")
    with pytest.raises(ValueError):
        build_retriever("hybrid", vector_retriever=FakeRetriever([]))


def test_build_retriever_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_retriever("magic")
