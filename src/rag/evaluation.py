"""Pure helpers for end-to-end evaluation and refusal-layer reporting.

The production answerer records *where* a refusal happened.  Keeping the
aggregation here (rather than inside the CLI script) makes the metric schema
unit-testable and lets the portfolio exporter recompute metrics from legacy
runs without calling an LLM again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

RefusalStage: TypeAlias = Literal["no_hits", "threshold", "llm"]
REFUSAL_STAGES: tuple[RefusalStage, ...] = ("no_hits", "threshold", "llm")


def canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text after normalizing checkout-dependent line endings."""

    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def infer_refusal_stage(trace: Mapping[str, Any], threshold: float) -> RefusalStage | None:
    """Return a refusal stage, including for traces written before it was recorded.

    New traces carry ``refusal_stage`` explicitly.  Legacy traces can be
    reconstructed deterministically from ``refused``, the retrieved-hit list,
    the top reranker score, and the configured threshold.
    """

    if "refusal_stage" in trace:
        stage = trace["refusal_stage"]
        if stage is not None and stage not in REFUSAL_STAGES:
            raise ValueError(f"unknown refusal_stage: {stage!r}")
        if bool(trace.get("refused")) != (stage is not None):
            raise ValueError("refused and refusal_stage disagree")
        return stage

    if not trace.get("refused"):
        return None
    if trace.get("retrieved") == []:
        return "no_hits"
    top_score = float(trace.get("top_score", 0.0))
    if top_score == threshold:
        raise ValueError(
            "legacy top_score equals the threshold after rounding; refusal stage is ambiguous"
        )
    return "threshold" if top_score < threshold else "llm"


def compute_e2e_metrics(traces: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate final quality and refusal behavior by decision layer."""

    rows = list(traces)
    for row in rows:
        # Explicit stage data is required here.  Legacy callers should first
        # normalize with ``infer_refusal_stage`` so bad data cannot silently
        # turn into a misleading layer breakdown.
        if "refusal_stage" not in row:
            raise ValueError(
                "compute_e2e_metrics requires explicit refusal_stage; "
                "normalize legacy traces with infer_refusal_stage first"
            )
        infer_refusal_stage(row, threshold=0.0)

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    judged = [row for row in answerable if "judge" in row]
    answered = [row for row in rows if not row["refused"]]
    answered_with_citations = [row for row in answered if row.get("cited_sources")]

    by_stage: dict[str, dict[str, Any]] = {}
    for stage in REFUSAL_STAGES:
        stage_rows = [row for row in rows if row.get("refusal_stage") == stage]
        by_stage[stage] = {
            "count": len(stage_rows),
            "answerable_qids": [row["qid"] for row in stage_rows if row["answerable"]],
            "unanswerable_qids": [
                row["qid"] for row in stage_rows if not row["answerable"]
            ],
        }

    direct_stages = {"no_hits", "threshold"}
    answerable_direct = [
        row for row in answerable if row.get("refusal_stage") in direct_stages
    ]
    unanswerable_direct = [
        row for row in unanswerable if row.get("refusal_stage") in direct_stages
    ]

    return {
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_judged": len(judged),
        "n_answered": len(answered),
        "n_generation_calls": sum(
            1 for row in rows if row.get("refusal_stage") not in direct_stages
        ),
        "false_refusals": [row["qid"] for row in answerable if row["refused"]],
        "false_refusal_rate": (
            sum(1 for row in answerable if row["refused"]) / len(answerable)
            if answerable
            else 0.0
        ),
        "avg_faithfulness": (
            sum(row["judge"]["faithfulness"] for row in judged) / len(judged)
            if judged
            else None
        ),
        "avg_relevancy": (
            sum(row["judge"]["relevancy"] for row in judged) / len(judged)
            if judged
            else None
        ),
        "pct_faithfulness_ge4": (
            sum(1 for row in judged if row["judge"]["faithfulness"] >= 4) / len(judged)
            if judged
            else None
        ),
        "n_answers_with_citations": len(answered_with_citations),
        "citation_parse_coverage": (
            len(answered_with_citations) / len(answered) if answered else None
        ),
        "uncited_answers": [row["qid"] for row in answered if not row.get("cited_sources")],
        "n_unanswerable": len(unanswerable),
        "refusal_accuracy": (
            sum(1 for row in unanswerable if row["refused"]) / len(unanswerable)
            if unanswerable
            else None
        ),
        "missed_refusals": [row["qid"] for row in unanswerable if not row["refused"]],
        # Direct refusal = no retrieval hit or score below the calibrated gate.
        # These two rates make the cost-saving first layer distinct from final
        # answerability, which can still be decided by the LLM layer.
        "direct_false_refusal_rate": (
            len(answerable_direct) / len(answerable) if answerable else 0.0
        ),
        "direct_unanswerable_coverage": (
            len(unanswerable_direct) / len(unanswerable) if unanswerable else None
        ),
        "refusal_by_stage": by_stage,
    }
