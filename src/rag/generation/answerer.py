"""Assembles the final answer: retrieval -> LLM -> citation parsing -> refusal.

Two refusal layers (see ``DESIGN.md``, section 4):
  1. Retrieval layer — if the reranked top score is below
     ``refusal_threshold``, refuse without ever calling the LLM. Only applied
     when a reranker is in the pipeline: raw vector/BM25/RRF scores are not
     calibrated for this gate. The cross-encoder is useful as a coarse filter,
     but high-scoring unanswerable questions can still reach the second layer.
  2. Generation layer — the prompt instructs the LLM to emit
     ``REFUSAL_PHRASE`` verbatim when the retrieved context can't answer the
     question, even if retrieval itself passed the threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.evaluation import RefusalStage
from rag.generation.llm import LLMAdapter
from rag.generation.prompts import REFUSAL_PHRASE, SYSTEM_PROMPT, build_user_prompt
from rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult

# Models writing Traditional Chinese sometimes emit full-width brackets ［1］
# instead of ASCII [1] (observed with gpt-5.1), so match both.
_CITATION_PATTERN = re.compile(r"[\[［](\d+)[\]］]")


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    refused: bool = False
    retrieval: RetrievalResult | None = None
    # Kept last to preserve positional construction compatibility for callers
    # written before refusal-layer observability was added.
    refusal_stage: RefusalStage | None = None


class Answerer:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        llm: LLMAdapter,
        refusal_threshold: float = 0.0,
        temperature: float = 0.0,
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.refusal_threshold = refusal_threshold
        self.temperature = temperature

    def answer(self, question: str) -> Answer:
        retrieval = self.pipeline.run(question)

        if not retrieval.hits:
            return self._refuse(retrieval, stage="no_hits")
        if self.pipeline.reranker is not None and retrieval.top_score < self.refusal_threshold:
            return self._refuse(retrieval, stage="threshold")

        raw = self.llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, retrieval.hits), temperature=self.temperature
        )
        refused = REFUSAL_PHRASE in raw
        sources = [] if refused else self._parse_sources(raw, retrieval.hits)
        return Answer(
            text=raw,
            sources=sources,
            refused=refused,
            refusal_stage="llm" if refused else None,
            retrieval=retrieval,
        )

    @staticmethod
    def _refuse(retrieval: RetrievalResult, stage: RefusalStage) -> Answer:
        return Answer(
            text=f"{REFUSAL_PHRASE},無法回答此問題。",
            sources=[],
            refused=True,
            refusal_stage=stage,
            retrieval=retrieval,
        )

    @staticmethod
    def _parse_sources(raw: str, hits) -> list[dict]:
        indices = sorted({int(m) for m in _CITATION_PATTERN.findall(raw)})
        sources = []
        for idx in indices:
            if 1 <= idx <= len(hits):
                hit = hits[idx - 1]
                sources.append(
                    {
                        "index": idx,
                        "doc": hit.payload["doc_title"],
                        "article": hit.payload["article_label"],
                        "content": hit.payload["content"],
                    }
                )
        return sources
