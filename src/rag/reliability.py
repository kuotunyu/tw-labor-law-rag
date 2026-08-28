"""Deterministic metrics and privacy reduction for the reliability stress run."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

PUBLIC_TRACE_FIELDS = (
    "qid",
    "answerable",
    "rank",
    "top_score",
    "threshold_refused",
    "elapsed_ms",
)


def _validated_row(row: Mapping[str, Any]) -> tuple[str, bool, int | None, float, float]:
    required = {"qid", "answerable", "rank", "top_score", "elapsed_ms"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"row missing fields: {sorted(missing)}")
    qid = row["qid"]
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("qid must be a non-empty string")
    answerable = row["answerable"]
    if not isinstance(answerable, bool):
        raise ValueError("answerable must be a boolean")
    rank = row["rank"]
    if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
        raise ValueError("rank must be null or a positive integer")
    score = float(row["top_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("top_score must be finite and between zero and one")
    elapsed_ms = float(row["elapsed_ms"])
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("elapsed_ms must be finite and non-negative")
    return qid, answerable, rank, score, elapsed_ms


def _validated_thresholds(thresholds: Iterable[float]) -> list[float]:
    values = [float(value) for value in thresholds]
    if not values:
        raise ValueError("thresholds must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("thresholds must be unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("threshold must be finite and between zero and one")
    return sorted(values)


def privacy_reduced_trace(row: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    """Return the exact, content-free trace schema approved for publication."""
    threshold_value = _validated_thresholds([threshold])[0]
    qid, answerable, rank, score, elapsed_ms = _validated_row(row)
    reduced = {
        "qid": qid,
        "answerable": answerable,
        "rank": rank,
        "top_score": round(score, 4),
        "threshold_refused": score < threshold_value,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if tuple(reduced) != PUBLIC_TRACE_FIELDS:
        raise AssertionError("public reliability trace schema changed")
    return reduced


def compute_reliability_metrics(
    rows: Iterable[Mapping[str, Any]], thresholds: Iterable[float]
) -> dict[str, Any]:
    """Compute retrieval and direct-refusal evidence from privacy-safe rows."""
    validated = [_validated_row(row) for row in rows]
    if not validated:
        raise ValueError("rows must not be empty")
    threshold_values = _validated_thresholds(thresholds)
    answerable_rows = [row for row in validated if row[1]]
    unanswerable_rows = [row for row in validated if not row[1]]
    if not answerable_rows or not unanswerable_rows:
        raise ValueError("rows must include answerable and unanswerable examples")

    answerable_count = len(answerable_rows)
    unanswerable_count = len(unanswerable_rows)
    ranks = [row[2] for row in answerable_rows]
    sweep: dict[str, dict[str, int | float]] = {}
    for threshold in threshold_values:
        false_refusals = sum(row[3] < threshold for row in answerable_rows)
        unanswerable_refusals = sum(row[3] < threshold for row in unanswerable_rows)
        sweep[str(threshold)] = {
            "direct_false_refusals": false_refusals,
            "direct_false_refusal_rate": false_refusals / answerable_count,
            "direct_unanswerable_refusals": unanswerable_refusals,
            "direct_unanswerable_coverage": unanswerable_refusals / unanswerable_count,
        }

    return {
        "questions": len(validated),
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / answerable_count,
        "mrr_at_10": sum(1 / rank for rank in ranks if rank is not None and rank <= 10)
        / answerable_count,
        "avg_latency_ms": round(sum(row[4] for row in validated) / len(validated), 1),
        "threshold_sweep": sweep,
    }
