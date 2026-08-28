import pytest

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
        threshold=0.03,
    )

    assert reduced == {
        "qid": "stress-001",
        "answerable": True,
        "rank": 2,
        "top_score": 0.0299,
        "threshold_refused": True,
        "elapsed_ms": 12.3,
    }


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
