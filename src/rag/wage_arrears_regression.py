"""Contracts for the v0.3.4 wage-arrears targeted regression evidence."""

from __future__ import annotations

import json
from pathlib import Path

from rag.retrieval.pipeline import _WAGE_ARREARS_LEGAL_TERMS, _retrieval_query

REQUIRED_FIELDS = {
    "qid",
    "question",
    "expect_expansion",
    "sources",
    "style_tags",
}
ARTICLE_14_SOURCE = [{"doc": "勞動基準法", "article": "第 14 條"}]
EXPECTED_QIDS = [f"wage-reg-{index:03d}" for index in range(1, 21)]


def load_regression_dataset(path: Path) -> list[dict]:
    """Load the frozen 20-row dataset and fail closed on contract drift."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: expected object")
        qid = row.get("qid", f"row {index}")
        if set(row) != REQUIRED_FIELDS:
            raise ValueError(f"{qid}: fields must equal {sorted(REQUIRED_FIELDS)}")
        if not isinstance(row["qid"], str) or not row["qid"]:
            raise ValueError(f"row {index}: qid must be a non-empty string")
        if row["qid"] in seen:
            raise ValueError(f"{row['qid']}: duplicate qid")
        seen.add(row["qid"])
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{row['qid']}: question must be a non-empty string")
        if type(row["expect_expansion"]) is not bool:
            raise ValueError(f"{row['qid']}: expect_expansion must be boolean")
        expected_sources = ARTICLE_14_SOURCE if row["expect_expansion"] else []
        if row["sources"] != expected_sources:
            raise ValueError(f"{row['qid']}: sources do not match expansion decision")
        if not isinstance(row["style_tags"], list) or not row["style_tags"]:
            raise ValueError(f"{row['qid']}: style_tags must be a non-empty list")
        if not all(isinstance(tag, str) and tag for tag in row["style_tags"]):
            raise ValueError(f"{row['qid']}: style_tags must contain non-empty strings")
    if [row["qid"] for row in rows] != EXPECTED_QIDS:
        raise ValueError("dataset qids must be wage-reg-001 through wage-reg-020")
    return rows


def route_expansion_applied(question: str) -> bool:
    """Return whether the shipped retrieval route appends Article 14 terms."""
    return _WAGE_ARREARS_LEGAL_TERMS in _retrieval_query(question)

