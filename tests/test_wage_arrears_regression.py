import json
from pathlib import Path

import pytest

from rag.wage_arrears_regression import (
    load_regression_dataset,
    route_expansion_applied,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "eval" / "dataset" / "wage_arrears_regression_v0.3.4.jsonl"
)
ARTICLE_14_SOURCE = [{"doc": "勞動基準法", "article": "第 14 條"}]


def _write_rows(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_rows() -> list[dict]:
    return [
        {
            "qid": f"wage-reg-{index:03d}",
            "question": f"測試問題 {index}：公司欠薪，我想直接離職。",
            "expect_expansion": index <= 10,
            "sources": ARTICLE_14_SOURCE if index <= 10 else [],
            "style_tags": ["test"],
        }
        for index in range(1, 21)
    ]


def test_targeted_dataset_has_reviewed_shape_and_routes() -> None:
    rows = load_regression_dataset(DATASET_PATH)

    assert len(rows) == 20
    assert [row["qid"] for row in rows] == [
        f"wage-reg-{index:03d}" for index in range(1, 21)
    ]
    assert sum(row["expect_expansion"] for row in rows) == 10
    assert all(
        route_expansion_applied(row["question"]) is row["expect_expansion"]
        for row in rows
    )
    assert all(row["sources"] == ARTICLE_14_SOURCE for row in rows[:10])
    assert all(row["sources"] == [] for row in rows[10:])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[1].update(qid=rows[0]["qid"]), "duplicate qid"),
        (lambda rows: rows[0].pop("style_tags"), "fields"),
        (lambda rows: rows[0].update(expect_expansion="yes"), "boolean"),
        (lambda rows: rows[0].update(sources=[]), "sources"),
        (lambda rows: rows[10].update(sources=ARTICLE_14_SOURCE), "sources"),
    ],
)
def test_targeted_dataset_rejects_invalid_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    rows = _valid_rows()
    mutate(rows)
    path = tmp_path / "invalid.jsonl"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=message):
        load_regression_dataset(path)

