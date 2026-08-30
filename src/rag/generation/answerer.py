"""Assembles the final answer: retrieval -> LLM -> citation parsing -> refusal.

Two refusal layers (see ``DESIGN.md``, section 4):
  1. Retrieval layer — the shared global policy refuses reranked results
     without an LLM call when the primary-query top score is below the global
     threshold. Routes remain retrieval evidence and never select a threshold.
     Raw vector/BM25/RRF scores are not calibrated for this gate. The
     cross-encoder is useful as a coarse filter, but high-scoring unanswerable
     questions can still reach the second layer.
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
from rag.generation.router import RoutedLLM
from rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from rag.retrieval.refusal_policy import decide_retrieval_refusal

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
    generation_called: bool = False
    requested_provider: str | None = None
    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    fallback_from: str | None = None


class Answerer:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        llm: LLMAdapter | RoutedLLM,
        refusal_threshold: float = 0.0,
        temperature: float = 0.0,
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.refusal_threshold = refusal_threshold
        self.temperature = temperature

    def answer(self, question: str) -> Answer:
        retrieval = self.pipeline.run(question)

        decision = decide_retrieval_refusal(
            has_hits=bool(retrieval.hits),
            reranker_enabled=self.pipeline.reranker is not None,
            applied_routes=retrieval.applied_routes,
            top_score=retrieval.top_score,
            global_threshold=self.refusal_threshold,
        )
        if decision.refusal_stage is not None:
            return self._refuse(retrieval, stage=decision.refusal_stage)

        generation = self.llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, retrieval.hits), temperature=self.temperature
        )
        refused = REFUSAL_PHRASE in generation.text
        sources = [] if refused else self._parse_sources(generation.text, retrieval.hits)
        return Answer(
            text=generation.text,
            sources=sources,
            refused=refused,
            refusal_stage="llm" if refused else None,
            retrieval=retrieval,
            generation_called=True,
            requested_provider=self._requested_provider(),
            provider=generation.provider,
            model=generation.model,
            fallback_used=generation.fallback_used,
            fallback_from=generation.fallback_from,
        )

    def _refuse(self, retrieval: RetrievalResult, stage: RefusalStage) -> Answer:
        return Answer(
            text=f"{REFUSAL_PHRASE},無法回答此問題。",
            sources=[],
            refused=True,
            refusal_stage=stage,
            retrieval=retrieval,
            generation_called=False,
            requested_provider=self._requested_provider(),
            provider=None,
            model=None,
            fallback_used=False,
            fallback_from=None,
        )

    def _requested_provider(self) -> str:
        primary_provider = getattr(self.llm, "primary_provider", None)
        if primary_provider is not None:
            return primary_provider
        return self.llm.provider

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
                        "source_url": hit.payload.get("source_url", ""),
                        "last_amended": hit.payload.get("last_amended", ""),
                        "effective_date": hit.payload.get("effective_date", ""),
                    }
                )
        return sources
