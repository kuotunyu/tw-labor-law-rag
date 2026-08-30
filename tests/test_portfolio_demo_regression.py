"""Contracts for the compact v0.3.5 portfolio regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_portfolio_demo_regression import run_cases, write_public_json
from rag.portfolio_demo_regression import (
    build_artifact,
    build_result,
    load_cases,
    summarize_results,
)

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
    assert sum(case.expect_threshold_refusal for case in cases) == 4
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
