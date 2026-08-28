"""Privacy-safe evidence reduction for budgeted provider cross-checks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

PUBLIC_PROVIDER_TRACE_FIELDS = (
    "qid",
    "answerable",
    "requested_provider",
    "actual_provider",
    "model",
    "refused",
    "citation_count",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "refusal_verdict",
    "citation_verdict",
    "elapsed_ms",
)


def select_crosscheck_rows(
    dataset_rows: Iterable[Mapping[str, Any]],
    reliability_rows: Iterable[Mapping[str, Any]],
    *,
    initial_count: int,
    maximum_count: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Select a five-row safety batch, then a deterministic eligible expansion."""
    if initial_count < 5 or maximum_count < initial_count:
        raise ValueError("cross-check counts require maximum >= initial >= 5")
    dataset = list(dataset_rows)
    evidence = list(reliability_rows)
    dataset_by_qid = {row.get("qid"): row for row in dataset}
    evidence_by_qid = {row.get("qid"): row for row in evidence}
    if (
        len(dataset_by_qid) != len(dataset)
        or len(evidence_by_qid) != len(evidence)
        or set(dataset_by_qid) != set(evidence_by_qid)
    ):
        raise ValueError("dataset and reliability qid coverage must match exactly")
    eligible = []
    for row in dataset:
        qid = row["qid"]
        answerable = row.get("answerable")
        evidence_row = evidence_by_qid[qid]
        if evidence_row.get("answerable") is not answerable:
            raise ValueError(f"answerable evidence mismatch for {qid}")
        refused = evidence_row.get("threshold_refused")
        if not isinstance(refused, bool):
            raise ValueError(f"threshold_refused must be boolean for {qid}")
        if not refused:
            eligible.append(row)

    answerable_rows = [row for row in eligible if row["answerable"]]
    unanswerable_rows = [row for row in eligible if not row["answerable"]]
    initial = [*answerable_rows[:3], *unanswerable_rows[:2]]
    if len(initial) != 5:
        raise ValueError("not enough generation-eligible rows for the initial safety batch")
    initial_qids = {row["qid"] for row in initial}
    remaining = [row for row in eligible if row["qid"] not in initial_qids]
    expansion = remaining[: maximum_count - len(initial)]
    return initial, expansion


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_decimal(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("cost must be a non-negative decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("cost must be a non-negative decimal")
    return result


def privacy_reduced_provider_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a raw generation trace to the exact public, content-free schema."""
    qid = row.get("qid")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("qid must be a non-empty string")
    answerable = row.get("answerable")
    refused = row.get("refused")
    if not isinstance(answerable, bool) or not isinstance(refused, bool):
        raise ValueError("answerable and refused must be booleans")
    requested = row.get("requested_provider")
    actual = row.get("actual_provider")
    if not isinstance(requested, str) or not requested or requested != actual:
        raise ValueError("requested and actual provider must match")
    model = row.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    citations = _nonnegative_int(row.get("citation_count"), label="citation count")
    inputs = _nonnegative_int(row.get("input_tokens"), label="token usage")
    outputs = _nonnegative_int(row.get("output_tokens"), label="token usage")
    cost = _nonnegative_decimal(row.get("estimated_cost_usd"))
    elapsed_ms = float(row.get("elapsed_ms"))
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("elapsed must be finite and non-negative")

    reduced = {
        "qid": qid,
        "answerable": answerable,
        "requested_provider": requested,
        "actual_provider": actual,
        "model": model,
        "refused": refused,
        "citation_count": citations,
        "input_tokens": inputs,
        "output_tokens": outputs,
        "estimated_cost_usd": str(cost),
        "refusal_verdict": int(refused == (not answerable)),
        "citation_verdict": int((refused and citations == 0) or (not refused and citations > 0)),
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if tuple(reduced) != PUBLIC_PROVIDER_TRACE_FIELDS:
        raise AssertionError("public provider trace schema changed")
    return reduced


def compute_provider_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    """Aggregate bounded verdicts and exact token costs by provider."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        reduced = privacy_reduced_provider_trace(row)
        grouped[reduced["actual_provider"]].append(reduced)
    if not grouped:
        raise ValueError("provider rows must not be empty")

    metrics: dict[str, dict[str, object]] = {}
    for provider, provider_rows in sorted(grouped.items()):
        count = len(provider_rows)
        metrics[provider] = {
            "requests": count,
            "refusal_accuracy": sum(row["refusal_verdict"] for row in provider_rows)
            / count,
            "citation_success_rate": sum(
                row["citation_verdict"] for row in provider_rows
            )
            / count,
            "input_tokens": sum(row["input_tokens"] for row in provider_rows),
            "output_tokens": sum(row["output_tokens"] for row in provider_rows),
            "estimated_cost_usd": str(
                sum(
                    (Decimal(row["estimated_cost_usd"]) for row in provider_rows),
                    Decimal(0),
                )
            ),
            "avg_latency_ms": round(
                sum(row["elapsed_ms"] for row in provider_rows) / count, 1
            ),
        }
    return metrics
