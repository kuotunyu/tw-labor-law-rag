"""Cross-encoder reranking with bge-reranker-v2-m3.

Scores are sigmoid-normalized to roughly [0, 1] (``normalize=True``), which is
what makes ``rerank_score_threshold`` in :mod:`rag.config` meaningful — raw
cosine/BM25/RRF scores from the first-stage retrievers are not comparable
across queries (Phase 1 finding: unanswerable questions scored just as high as
answerable ones), but the cross-encoder score is.
"""

from __future__ import annotations

from rag.indexing.embedder import resolve_device
from rag.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"):
        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: avoid loading the cross-encoder until first use
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name, use_fp16=self.device.startswith("cuda"), devices=[self.device]
            )
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [[query, c.payload["text"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        reranked = [
            RetrievedChunk(score=float(s), payload=c.payload) for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k]
