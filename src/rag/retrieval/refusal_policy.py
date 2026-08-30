import math
from dataclasses import dataclass

from rag.evaluation import RefusalStage


def _validated_unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return normalized


@dataclass(frozen=True)
class RetrievalRefusalDecision:
    refusal_stage: RefusalStage | None
    effective_threshold: float | None

    @property
    def refused(self) -> bool:
        return self.refusal_stage is not None


def decide_retrieval_refusal(
    *,
    has_hits: bool,
    reranker_enabled: bool,
    applied_routes: tuple[str, ...],
    top_score: float,
    global_threshold: float,
) -> RetrievalRefusalDecision:
    if type(has_hits) is not bool or type(reranker_enabled) is not bool:
        raise ValueError("hit and reranker flags must be booleans")
    if not isinstance(applied_routes, tuple) or not all(
        isinstance(route, str) and route.strip() for route in applied_routes
    ):
        raise ValueError("applied_routes must be a tuple of non-blank strings")
    score = _validated_unit_interval(top_score, name="top_score")
    global_value = _validated_unit_interval(
        global_threshold, name="global_threshold"
    )
    if not has_hits:
        return RetrievalRefusalDecision("no_hits", None)
    if not reranker_enabled:
        return RetrievalRefusalDecision(None, None)
    stage: RefusalStage | None = "threshold" if score < global_value else None
    return RetrievalRefusalDecision(stage, global_value)
