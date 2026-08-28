"""Privacy-safe evidence reduction for budgeted provider cross-checks."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rag.provider_budget import BudgetLedger

AUTHORIZED_MAX_INPUT_TOKENS = 20_000
AUTHORIZED_MAX_OUTPUT_TOKENS = 1_024
MESSAGE_ENVELOPE_TOKEN_ALLOWANCE = 1_024

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


class BudgetSafetyError(RuntimeError):
    """Sanitized stop signal for preflight or post-response budget violations."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def validate_request_maxima(
    max_input_tokens: object,
    max_output_tokens: object,
) -> tuple[int, int]:
    """Keep every request inside the reviewed pricing/context envelope."""
    for value, label, authorized in (
        (max_input_tokens, "input", AUTHORIZED_MAX_INPUT_TOKENS),
        (max_output_tokens, "output", AUTHORIZED_MAX_OUTPUT_TOKENS),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} maximum must be a positive integer")
        if value > authorized:
            raise ValueError(f"{label} maximum exceeds the authorized {label} maximum")
    return max_input_tokens, max_output_tokens


def conservative_input_token_bound(*texts: str) -> int:
    """Bound chat input by UTF-8 bytes plus a large message-envelope allowance."""
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("prompt texts must be strings")
    return (
        sum(len(text.encode("utf-8")) for text in texts)
        + MESSAGE_ENVELOPE_TOKEN_ALLOWANCE
    )


def generate_with_budget(
    adapter: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    ledger: BudgetLedger,
    max_input_tokens: int,
    max_output_tokens: int,
) -> tuple[Any, Decimal, float]:
    """Preflight the complete prompt before making exactly one paid request."""
    input_maximum, output_maximum = validate_request_maxima(
        max_input_tokens,
        max_output_tokens,
    )
    prompt_bound = conservative_input_token_bound(system_prompt, user_prompt)
    if prompt_bound > input_maximum:
        raise BudgetSafetyError("prompt_exceeds_conservative_maximum")
    if not ledger.can_start(
        max_input_tokens=prompt_bound,
        max_output_tokens=output_maximum,
    ):
        raise BudgetSafetyError("conservative_budget_preflight")

    started = time.perf_counter()
    generation = adapter.generate(
        system_prompt,
        user_prompt,
        temperature=0.0,
        max_tokens=output_maximum,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if generation.input_tokens is None or generation.output_tokens is None:
        raise BudgetSafetyError("missing_usage_metadata")
    if (
        generation.input_tokens > prompt_bound
        or generation.output_tokens > output_maximum
    ):
        raise BudgetSafetyError("usage_exceeded_conservative_maximum")
    charge = ledger.record_usage(
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
    )
    return generation, charge, elapsed_ms


def resolve_private_run_dir(
    requested: Path | None,
    runs_root: Path,
    default_name: str,
) -> Path:
    """Resolve a raw-output directory strictly below the ignored runs root."""
    root = runs_root.resolve()
    candidate = (requested if requested is not None else root / default_name).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("work directory must remain inside eval/runs") from exc
    if not relative.parts:
        raise ValueError("work directory must be a child directory inside eval/runs")
    return candidate


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
