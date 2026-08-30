import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_reliability_eval  # noqa: E402

from rag.reliability import (
    compute_reliability_metrics,
    pareto_better_thresholds,
    privacy_reduced_trace,
)


def trace(qid, answerable, *, rank, score, elapsed_ms=10.0):
    return {
        "qid": qid,
        "answerable": answerable,
        "rank": rank,
        "top_score": score,
        "elapsed_ms": elapsed_ms,
    }


def test_reliability_metrics_and_threshold_sweep_are_deterministic():
    metrics = compute_reliability_metrics(
        rows=[
            trace("a1", True, rank=1, score=0.02, elapsed_ms=10.0),
            trace("a2", True, rank=6, score=0.04, elapsed_ms=20.0),
            trace("u1", False, rank=None, score=0.01, elapsed_ms=30.0),
            trace("u2", False, rank=None, score=0.05, elapsed_ms=40.0),
        ],
        thresholds=[0.0, 0.03],
    )

    assert metrics["questions"] == 4
    assert metrics["answerable"] == 2
    assert metrics["unanswerable"] == 2
    assert metrics["hit_at_5"] == 0.5
    assert metrics["mrr_at_10"] == pytest.approx((1 + 1 / 6) / 2)
    assert metrics["avg_latency_ms"] == 25.0
    assert metrics["threshold_sweep"]["0.03"] == {
        "direct_false_refusals": 1,
        "direct_false_refusal_rate": 0.5,
        "direct_unanswerable_refusals": 1,
        "direct_unanswerable_coverage": 0.5,
    }


def test_privacy_reduced_trace_has_exact_public_fields():
    reduced = privacy_reduced_trace(
        {
            **trace("stress-001", True, rank=2, score=0.02994, elapsed_ms=12.345),
            "question": "must not survive",
            "hits": [{"content": "must not survive"}],
        },
        threshold_refused=True,
    )

    assert reduced == {
        "qid": "stress-001",
        "answerable": True,
        "rank": 2,
        "top_score": 0.0299,
        "threshold_refused": True,
        "elapsed_ms": 12.3,
    }


def test_privacy_reduced_trace_uses_the_precomputed_threshold_decision():
    reduced = privacy_reduced_trace(
        trace("stress-001", True, rank=2, score=0.01),
        threshold_refused=False,
    )

    assert reduced["threshold_refused"] is False


@pytest.mark.parametrize("threshold_refused", [None, 0, 1, "false"])
def test_privacy_reduced_trace_requires_a_boolean_decision(threshold_refused):
    with pytest.raises(ValueError, match="threshold_refused must be a boolean"):
        privacy_reduced_trace(
            trace("stress-001", True, rank=2, score=0.01),
            threshold_refused=threshold_refused,
        )


def test_reliability_runner_reduces_with_shared_route_aware_decision(monkeypatch):
    calls = []

    def decide(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(refusal_stage=None)

    monkeypatch.setattr(run_reliability_eval, "decide_retrieval_refusal", decide)
    settings = SimpleNamespace(
        rerank_score_threshold=0.9,
        severance_comparison_score_threshold=0.2,
    )
    row = {
        **trace("stress-001", True, rank=2, score=0.1),
        "hits": [{"chunk_id": "c1"}],
        "applied_routes": ["severance_comparison"],
    }

    reduced = run_reliability_eval._reduce_trace(row, settings)

    assert reduced["threshold_refused"] is False
    assert calls == [
        {
            "has_hits": True,
            "reranker_enabled": True,
            "applied_routes": ("severance_comparison",),
            "top_score": 0.1,
            "global_threshold": 0.9,
            "severance_comparison_threshold": 0.2,
        }
    ]


def test_reliability_runner_records_routes_only_in_private_raw_row():
    retrieval = SimpleNamespace(
        hits=[],
        top_score=0.0,
        applied_routes=("off_hours_employer_message",),
    )
    pipeline = SimpleNamespace(run=lambda _question: retrieval)

    rows = run_reliability_eval._run_rows(
        pipeline,
        [
            {
                "qid": "stress-001",
                "question": "private question",
                "answerable": False,
                "sources": [],
            }
        ],
    )

    assert rows[0]["applied_routes"] == ["off_hours_employer_message"]
    reduced = privacy_reduced_trace(rows[0], threshold_refused=True)
    assert tuple(reduced) == (
        "qid",
        "answerable",
        "rank",
        "top_score",
        "threshold_refused",
        "elapsed_ms",
    )
    assert "applied_routes" not in reduced


@pytest.mark.parametrize(
    "rows, thresholds, message",
    [
        ([], [0.03], "rows"),
        ([trace("a", True, rank=1, score=-0.1)], [0.03], "top_score"),
        ([trace("a", True, rank=1, score=0.1)], [-0.1], "threshold"),
        ([trace("a", True, rank=1, score=0.1)], [0.03, 0.03], "unique"),
    ],
)
def test_reliability_metrics_fail_closed_on_invalid_input(rows, thresholds, message):
    with pytest.raises(ValueError, match=message):
        compute_reliability_metrics(rows, thresholds)


def test_pareto_decision_is_recomputed_from_both_sweeps() -> None:
    stress = compute_reliability_metrics(
        [
            trace("a", True, rank=1, score=0.05),
            trace("u", False, rank=None, score=0.035),
        ],
        [0.0, 0.03, 0.04],
    )
    formal = compute_reliability_metrics(
        [
            trace("a", True, rank=1, score=0.05),
            trace("u", False, rank=None, score=0.035),
        ],
        [0.0, 0.03, 0.04],
    )

    assert pareto_better_thresholds(stress, formal, production=0.03) == [0.04]


def test_pareto_decision_rejects_mismatched_threshold_sweeps() -> None:
    stress = {"threshold_sweep": {"0.03": {}, "0.04": {}}}
    formal = {"threshold_sweep": {"0.03": {}}}

    with pytest.raises(ValueError, match="same thresholds"):
        pareto_better_thresholds(stress, formal, production=0.03)
