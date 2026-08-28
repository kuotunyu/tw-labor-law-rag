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

_EMPLOYER_CUES = ("老闆", "主管", "雇主")
_OFF_HOURS_CUES = ("假日", "休假日", "休息日", "例假", "下班後", "非上班時間")
_MESSAGE_CUES = ("群組傳訊", "群組訊息", "傳訊", "回訊", "line群組", "line訊息")
_OFF_HOURS_LEGAL_TERMS = "雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"


def _retrieval_query(query: str) -> str:
    """Bridge one observed colloquial gap without spending a provider call.

    Expansion is deliberately gated on employer, off-hours, and messaging
    cues occurring together.  The original question remains intact for the
    answer-generation layer; only retrieval and reranking see the legal terms.
    """
    folded = query.casefold()
    cue_groups = (_EMPLOYER_CUES, _OFF_HOURS_CUES, _MESSAGE_CUES)
    if all(any(cue in folded for cue in cues) for cues in cue_groups):
        return f"{query} {_OFF_HOURS_LEGAL_TERMS}"
    return query


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
        search_query = _retrieval_query(query)
        candidates = self.retriever.retrieve(search_query, top_k=self.top_k_retrieve)
        if self.reranker is not None:
            hits = self.reranker.rerank(search_query, candidates, top_k=self.top_k_final)
        else:
            hits = candidates[: self.top_k_final]
        top_score = hits[0].score if hits else 0.0
        return RetrievalResult(hits=hits, candidates=candidates, top_score=top_score)
