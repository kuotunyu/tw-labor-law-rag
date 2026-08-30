"""Contracts for the compact v0.3.5 portfolio regression."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import run_portfolio_demo_regression
from eval.run_portfolio_demo_regression import run_cases, write_public_json
from rag import factory
from rag.portfolio_demo_regression import (
    build_artifact,
    build_result,
    load_cases,
    summarize_results,
)
from rag.retrieval import reranker as reranker_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_reliability_eval  # noqa: E402

PROJECT_ROOT = Path(__file__).parents[1]
DATASET = PROJECT_ROOT / "eval/dataset/portfolio_demo_v0.3.5.jsonl"


@pytest.fixture
def cases():
    return load_cases(DATASET)


def test_portfolio_dataset_has_exact_representative_contract() -> None:
    cases = load_cases(DATASET)

    assert [case.qid for case in cases] == [
        f"portfolio-{number:03d}" for number in range(1, 11)
    ]
    assert sum(case.answerable for case in cases) == 6
    assert sum(case.expect_threshold_refusal for case in cases) == 2
    assert sum(case.expected_refusal_stage == "llm" for case in cases) == 2
    assert {source["law"] for case in cases for source in case.sources} >= {
        "勞動基準法",
        "勞工請假規則",
        "勞工退休金條例",
        "勞動基準法施行細則",
    }
    assert next(
        case for case in cases if case.qid == "portfolio-006"
    ).required_routes == ("wage_arrears",)
    collision = next(case for case in cases if case.qid == "portfolio-002")
    assert collision.prohibited_routes == ("wage_arrears",)
    assert collision.prohibited_sources == (
        {"law": "勞動基準法", "article": "第 14 條"},
    )


def _valid_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update(extra=1), "portfolio case"),
        (lambda rows: rows[0].update(question=" "), "portfolio case"),
        (lambda rows: rows[1].update(qid=rows[0]["qid"]), "portfolio case"),
        (lambda rows: rows[0].update(sources=[]), "portfolio case"),
        (
            lambda rows: rows[6].update(
                sources=[{"law": "勞動基準法", "article": "第 30 條"}]
            ),
            "portfolio case",
        ),
        (lambda rows: rows[0].update(expect_threshold_refusal=True), "portfolio case"),
        (lambda rows: rows[0].update(expected_refusal_stage="threshold"), "portfolio case"),
        (lambda rows: rows[6].update(expected_refusal_stage="threshold"), "portfolio case"),
        (
            lambda rows: rows[0].update(
                prohibited_sources=list(rows[0]["sources"])
            ),
            "portfolio case",
        ),
        (
            lambda rows: rows[0].update(
                required_routes=["route"], prohibited_routes=["route"]
            ),
            "portfolio case",
        ),
    ],
)
def test_portfolio_parser_rejects_invalid_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    rows = _valid_rows()
    mutate(rows)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_cases(bad)


def test_result_builder_scores_sources_and_refusal_without_answer_text(cases) -> None:
    result = build_result(
        cases[0],
        retrieved=[("勞動基準法", "第 30 條", 1)],
        applied_routes=[],
        threshold_refused=False,
        top_score=0.61,
    )

    assert result["source_ranks"] == {"勞動基準法|第 30 條": 1}
    assert result["refusal_stage"] is None
    assert result["generation_allowed"] is True
    assert result["generation_called"] is False
    assert result["passed"] is True
    assert "answer" not in result


def test_unanswerable_llm_stage_passes_retrieval_boundary_without_generation(
    cases,
) -> None:
    case = next(case for case in cases if case.expected_refusal_stage == "llm")

    result = build_result(
        case,
        retrieved=[("最低工資法", "第 4 條", 1)],
        applied_routes=[],
        threshold_refused=False,
        top_score=0.1,
    )

    assert result["expected_refusal_stage"] == "llm"
    assert result["refusal_stage"] is None
    assert result["generation_allowed"] is True
    assert result["generation_called"] is False
    assert result["passed"] is True


def test_summary_requires_all_expected_sources_at_five_and_exact_refusal(
    cases,
) -> None:
    results = [
        build_result(
            case,
            retrieved=[],
            applied_routes=list(case.required_routes),
            threshold_refused=case.expect_threshold_refusal,
            top_score=0.0,
        )
        for case in cases
    ]

    summary = summarize_results(results)

    assert summary["threshold_refusal_accuracy"] == 1.0
    assert summary["source_recall_at_5"] == 0.0
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "retrieved",
    [
        [("勞動基準法", "第 30 條", 0)],
        [
            ("勞動基準法", "第 30 條", 1),
            ("勞動基準法", "第 30 條", 2),
        ],
    ],
)
def test_result_builder_rejects_invalid_retrieved_contract(cases, retrieved) -> None:
    with pytest.raises(ValueError, match="retrieved"):
        build_result(
            cases[0],
            retrieved=retrieved,
            applied_routes=[],
            threshold_refused=False,
            top_score=0.1,
        )


def test_runner_with_fake_retrieval_is_content_free_and_deterministic(
    tmp_path: Path,
    cases,
) -> None:
    calls: list[str] = []

    def fake_retrieval(case):
        calls.append(case.qid)
        return {
            "retrieved": [
                (source["law"], source["article"], rank)
                for rank, source in enumerate(case.sources, 1)
            ],
            "applied_routes": list(case.required_routes),
            "threshold_refused": case.expect_threshold_refusal,
            "top_score": 0.0 if case.expect_threshold_refusal else 0.5,
        }

    results = run_cases(cases, fake_retrieval)
    artifact = build_artifact(
        dataset_path=DATASET,
        snapshot_path=PROJECT_ROOT / "release/corpus_snapshot.json",
        code_revision="a" * 40,
        configuration={"chunking": "structure", "retrieval": "hybrid"},
        results=results,
    )
    output = tmp_path / "portfolio.json"
    write_public_json(output, artifact)

    assert calls == [f"portfolio-{number:03d}" for number in range(1, 11)]
    assert artifact["summary"]["passed"] is True
    assert artifact["summary"]["source_recall_at_5"] == 1.0
    assert len(artifact["cases"]) == 10
    serialized = output.read_text(encoding="utf-8")
    assert b"\r\n" not in output.read_bytes()
    assert serialized.endswith("\n")
    assert serialized == json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    assert all(case.question not in serialized for case in cases)
    assert '"answer"' not in serialized
    assert "api_key" not in serialized
    runner_source = (
        PROJECT_ROOT / "eval/run_portfolio_demo_regression.py"
    ).read_text(encoding="utf-8")
    assert "rag.generation" not in runner_source
    assert "provider_crosscheck" not in runner_source


def test_portfolio_runner_uses_retrieval_routes_and_shared_decision(
    monkeypatch,
    cases,
) -> None:
    calls = []

    def decide(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(refusal_stage=None)

    monkeypatch.setattr(
        run_portfolio_demo_regression,
        "decide_retrieval_refusal",
        decide,
    )
    retrieval = SimpleNamespace(
        hits=[],
        top_score=0.01,
        applied_routes=("route_from_retrieval",),
    )
    settings = SimpleNamespace(
        rerank_score_threshold=0.03,
        severance_comparison_score_threshold=0.015,
    )

    observation = run_portfolio_demo_regression._evaluate_retrieval(
        cases[0],
        retrieval,
        settings,
        reranker_enabled=True,
    )

    assert observation == {
        "retrieved": [],
        "applied_routes": ["route_from_retrieval"],
        "threshold_refused": False,
        "top_score": 0.01,
    }
    assert calls == [
        {
            "has_hits": False,
            "reranker_enabled": True,
            "applied_routes": ("route_from_retrieval",),
            "top_score": 0.01,
            "global_threshold": 0.03,
            "severance_comparison_threshold": 0.015,
        }
    ]


def test_portfolio_local_index_wiring_uses_shared_policy_observation(
    monkeypatch,
    tmp_path: Path,
    cases,
) -> None:
    case = cases[0]
    expected_source = case.sources[0]
    hit = SimpleNamespace(
        payload={
            "doc_title": expected_source["law"],
            "articles": [expected_source["article"]],
        }
    )
    retrieval = SimpleNamespace(
        hits=[hit],
        top_score=0.01,
        applied_routes=("route_from_retrieval",),
    )
    pipeline = SimpleNamespace(reranker=object(), run=lambda _question: retrieval)
    settings = SimpleNamespace(
        rerank_score_threshold=0.03,
        severance_comparison_score_threshold=0.015,
        top_k_retrieve=20,
        top_k_final=5,
        rrf_k=60,
        embedding_model="embedding-test",
        embedding_model_revision="embedding-revision",
        reranker_model="reranker-test",
        reranker_model_revision="reranker-revision",
        device="cpu",
    )
    store = SimpleNamespace(close=lambda: None)
    embedder = SimpleNamespace(close=lambda: None)
    policy_calls = []

    def decide(**kwargs):
        policy_calls.append(kwargs)
        return SimpleNamespace(refusal_stage=None)

    monkeypatch.setattr(
        run_portfolio_demo_regression,
        "Settings",
        lambda **_kwargs: settings,
    )
    monkeypatch.setattr(
        run_portfolio_demo_regression,
        "require_cached_models",
        lambda _settings: None,
    )
    monkeypatch.setattr(run_portfolio_demo_regression, "load_cases", lambda _path: [case])
    monkeypatch.setattr(
        run_portfolio_demo_regression,
        "decide_retrieval_refusal",
        decide,
    )
    monkeypatch.setattr(
        run_reliability_eval,
        "_build_indexes",
        lambda *_args: (embedder, store),
    )
    monkeypatch.setattr(
        factory,
        "build_retrieval_pipeline",
        lambda *_args, **_kwargs: pipeline,
    )
    monkeypatch.setattr(reranker_module, "Reranker", lambda **_kwargs: object())
    args = SimpleNamespace(
        offline=False,
        device="cpu",
        dataset=DATASET,
        data_dir=tmp_path,
        snapshot=tmp_path / "unused-snapshot.json",
    )

    results, _configuration = run_portfolio_demo_regression._run_with_local_index(args)

    assert len(policy_calls) == 1
    assert policy_calls[0]["top_score"] == 0.01
    assert results[0]["applied_routes"] == ["route_from_retrieval"]
    assert results[0]["threshold_refused"] is False
