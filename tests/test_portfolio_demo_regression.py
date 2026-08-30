"""Contracts for the compact v0.3.5 portfolio regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.portfolio_demo_regression import load_cases

PROJECT_ROOT = Path(__file__).parents[1]
DATASET = PROJECT_ROOT / "eval/dataset/portfolio_demo_v0.3.5.jsonl"


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
