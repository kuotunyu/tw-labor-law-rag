"""Composes first-stage retrieval with an optional reranking stage.

Reused by the answerer (Phase 2) and the ablation study (Phase 4): every
setting in the 6-way ablation grid — {vector, hybrid} x rerank on/off, x 2
chunking strategies — is just a different ``RetrievalPipeline`` wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.models import RetrievedChunk
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]  # final top_k_final, after optional rerank
    candidates: list[RetrievedChunk] = field(default_factory=list)  # pre-rerank top_k_retrieve
    top_score: float = 0.0


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None,
        top_k_retrieve: int,
        top_k_final: int,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

    def run(self, query: str) -> RetrievalResult:
        candidates = self.retriever.retrieve(query, top_k=self.top_k_retrieve)
        if self.reranker is not None:
            hits = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
        else:
            hits = candidates[: self.top_k_final]
        top_score = hits[0].score if hits else 0.0
        return RetrievalResult(hits=hits, candidates=candidates, top_score=top_score)
