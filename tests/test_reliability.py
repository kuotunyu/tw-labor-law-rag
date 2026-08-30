import json
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


def test_reliability_main_uses_shared_policy_for_each_public_trace(
    monkeypatch,
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.jsonl"
    formal_dataset = tmp_path / "formal.jsonl"
    snapshot = tmp_path / "snapshot.json"
    dataset.write_text("{}\n", encoding="utf-8")
    formal_dataset.write_text("{}\n", encoding="utf-8")
    snapshot.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-01-01",
                "law_count": 15,
                "article_count": 100,
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        rerank_score_threshold=0.03,
        severance_comparison_score_threshold=0.015,
        top_k_retrieve=20,
        top_k_final=5,
        embedding_model="embedding-test",
        embedding_model_revision="embedding-revision",
        reranker_model="reranker-test",
        reranker_model_revision="reranker-revision",
        device="cpu",
    )
    retrieval = SimpleNamespace(
        hits=[],
        top_score=0.01,
        applied_routes=("severance_comparison",),
    )
    pipeline = SimpleNamespace(reranker=object(), run=lambda _question: retrieval)
    store = SimpleNamespace(close=lambda: None)
    rows = [
        {
            "qid": "answerable",
            "question": "private answerable question",
            "answerable": True,
            "sources": [],
        },
        {
            "qid": "unanswerable",
            "question": "private unanswerable question",
            "answerable": False,
            "sources": [],
        },
    ]
    policy_calls = []

    def decide(**kwargs):
        policy_calls.append(kwargs)
        return SimpleNamespace(refusal_stage=None)

    public_jsonl = {}

    def record_jsonl(path, value):
        public_jsonl[path.name] = value

    monkeypatch.setattr(run_reliability_eval, "Settings", lambda **_kwargs: settings)
    monkeypatch.setattr(
        run_reliability_eval,
        "_materialize_audited_corpus",
        lambda *_args: tmp_path,
    )
    monkeypatch.setattr(
        run_reliability_eval,
        "_build_indexes",
        lambda *_args: (object(), store),
    )
    monkeypatch.setattr(run_reliability_eval, "Reranker", lambda **_kwargs: object())
    monkeypatch.setattr(
        run_reliability_eval,
        "build_retrieval_pipeline",
        lambda *_args, **_kwargs: pipeline,
    )
    monkeypatch.setattr(run_reliability_eval.lib, "load_dataset", lambda _path: rows)
    monkeypatch.setattr(run_reliability_eval, "decide_retrieval_refusal", decide)
    monkeypatch.setattr(run_reliability_eval, "_write_jsonl", record_jsonl)
    monkeypatch.setattr(run_reliability_eval, "_write_json", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reliability_eval.py",
            "--dataset",
            str(dataset),
            "--formal-dataset",
            str(formal_dataset),
            "--snapshot",
            str(snapshot),
            "--work-dir",
            str(tmp_path / "run"),
            "--export-official",
        ],
    )

    assert run_reliability_eval.main() == 0
    assert len(policy_calls) == 4
    assert all(
        call
        == {
            "has_hits": False,
            "reranker_enabled": True,
            "applied_routes": ("severance_comparison",),
            "top_score": 0.01,
            "global_threshold": 0.03,
            "severance_comparison_threshold": 0.015,
        }
        for call in policy_calls
    )
    assert [
        row["threshold_refused"]
        for row in public_jsonl["reliability_trace.jsonl"]
    ] == [False, False]
    assert [
        row["threshold_refused"]
        for row in public_jsonl["reliability_formal_trace.jsonl"]
    ] == [False, False]


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
