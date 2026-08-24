from rag.models import RetrievedChunk
from rag.retrieval.pipeline import RetrievalPipeline


def hit(chunk_id, score=1.0):
    return RetrievedChunk(score=score, payload={"chunk_id": chunk_id, "text": chunk_id})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k):
        return self.hits[:top_k]


class FakeReranker:
    """Reverses order, deterministically, so tests can tell rerank ran."""

    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))[:top_k]


def test_pipeline_without_reranker_slices_candidates():
    hits = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    pipeline = RetrievalPipeline(FakeRetriever(hits), reranker=None, top_k_retrieve=20, top_k_final=2)
    result = pipeline.run("query")
    assert [h.payload["chunk_id"] for h in result.hits] == ["a", "b"]
    assert result.top_score == 0.9
    assert len(result.candidates) == 3


def test_pipeline_with_reranker_reorders():
    hits = [hit("a"), hit("b"), hit("c")]
    pipeline = RetrievalPipeline(FakeRetriever(hits), reranker=FakeReranker(), top_k_retrieve=20, top_k_final=2)
    result = pipeline.run("query")
    assert [h.payload["chunk_id"] for h in result.hits] == ["c", "b"]


def test_pipeline_empty_candidates():
    pipeline = RetrievalPipeline(FakeRetriever([]), reranker=None, top_k_retrieve=20, top_k_final=5)
    result = pipeline.run("query")
    assert result.hits == []
    assert result.top_score == 0.0


def test_pipeline_passes_top_k_retrieve_to_retriever():
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(top_k)
            return []

    pipeline = RetrievalPipeline(RecordingRetriever(), reranker=None, top_k_retrieve=20, top_k_final=5)
    pipeline.run("query")
    assert calls == [20]
