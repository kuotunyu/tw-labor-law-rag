"""Strict contracts for the compact v0.3.5 portfolio regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "qid",
    "question",
    "category",
    "answerable",
    "sources",
    "prohibited_sources",
    "expect_threshold_refusal",
    "expected_refusal_stage",
    "required_routes",
    "prohibited_routes",
    "rationale",
    "style_tags",
}
EXPECTED_QIDS = tuple(f"portfolio-{number:03d}" for number in range(1, 11))


@dataclass(frozen=True)
class PortfolioCase:
    """One content-bearing private input with content-free scoring contracts."""

    qid: str
    question: str
    category: str
    answerable: bool
    sources: tuple[dict[str, str], ...]
    prohibited_sources: tuple[dict[str, str], ...]
    expect_threshold_refusal: bool
    expected_refusal_stage: str | None
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    rationale: str
    style_tags: tuple[str, ...]


def _invalid(identity: object, message: str) -> ValueError:
    return ValueError(f"portfolio case {identity}: {message}")


def _non_blank(row: dict[str, Any], field: str, identity: object) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise _invalid(identity, f"{field} must be a non-blank string")
    return value.strip()


def _sources(
    value: object,
    *,
    field: str,
    identity: object,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise _invalid(identity, f"{field} must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {"law", "article"}:
            raise _invalid(identity, f"{field} entries require law and article")
        law = source["law"]
        article = source["article"]
        if not isinstance(law, str) or not law.strip():
            raise _invalid(identity, f"{field} law must be a non-blank string")
        if not isinstance(article, str) or not article.strip():
            raise _invalid(identity, f"{field} article must be a non-blank string")
        source_id = (law.strip(), article.strip())
        if source_id in seen:
            raise _invalid(identity, f"{field} contains a duplicate source")
        seen.add(source_id)
        normalized.append({"law": source_id[0], "article": source_id[1]})
    return tuple(normalized)


def _strings(
    value: object,
    *,
    field: str,
    identity: object,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _invalid(identity, f"{field} must be a list")
    if not allow_empty and not value:
        raise _invalid(identity, f"{field} must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise _invalid(identity, f"{field} must contain non-blank strings")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise _invalid(identity, f"{field} contains duplicates")
    return normalized


def _parse_case(row: object, index: int) -> PortfolioCase:
    identity = f"row {index}"
    if not isinstance(row, dict):
        raise _invalid(identity, "must be an object")
    identity = row.get("qid", identity)
    if set(row) != REQUIRED_FIELDS:
        raise _invalid(identity, f"fields must equal {sorted(REQUIRED_FIELDS)}")

    qid = _non_blank(row, "qid", identity)
    question = _non_blank(row, "question", qid)
    category = _non_blank(row, "category", qid)
    rationale = _non_blank(row, "rationale", qid)
    if type(row["answerable"]) is not bool:
        raise _invalid(qid, "answerable must be boolean")
    if type(row["expect_threshold_refusal"]) is not bool:
        raise _invalid(qid, "expect_threshold_refusal must be boolean")

    answerable = row["answerable"]
    expect_threshold_refusal = row["expect_threshold_refusal"]
    sources = _sources(row["sources"], field="sources", identity=qid)
    prohibited_sources = _sources(
        row["prohibited_sources"], field="prohibited_sources", identity=qid
    )
    required_routes = _strings(
        row["required_routes"],
        field="required_routes",
        identity=qid,
        allow_empty=True,
    )
    prohibited_routes = _strings(
        row["prohibited_routes"],
        field="prohibited_routes",
        identity=qid,
        allow_empty=True,
    )
    style_tags = _strings(
        row["style_tags"], field="style_tags", identity=qid, allow_empty=False
    )

    if answerable and not sources:
        raise _invalid(qid, "answerable cases require sources")
    if not answerable and sources:
        raise _invalid(qid, "unanswerable cases must not have sources")
    if answerable == expect_threshold_refusal:
        raise _invalid(qid, "answerability must oppose threshold refusal")
    expected_stage = row["expected_refusal_stage"]
    required_stage = "threshold" if expect_threshold_refusal else None
    if expected_stage != required_stage:
        raise _invalid(qid, f"expected_refusal_stage must be {required_stage!r}")
    source_ids = {(source["law"], source["article"]) for source in sources}
    prohibited_ids = {
        (source["law"], source["article"]) for source in prohibited_sources
    }
    if source_ids & prohibited_ids:
        raise _invalid(qid, "sources overlap prohibited_sources")
    if set(required_routes) & set(prohibited_routes):
        raise _invalid(qid, "required_routes overlap prohibited_routes")

    return PortfolioCase(
        qid=qid,
        question=question,
        category=category,
        answerable=answerable,
        sources=sources,
        prohibited_sources=prohibited_sources,
        expect_threshold_refusal=expect_threshold_refusal,
        expected_refusal_stage=expected_stage,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        rationale=rationale,
        style_tags=style_tags,
    )


def load_cases(path: Path) -> list[PortfolioCase]:
    """Load the exact reviewed ten-case contract and fail closed on drift."""

    rows: list[object] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise _invalid(f"row {index}", "must be valid JSON") from exc
    cases = [_parse_case(row, index) for index, row in enumerate(rows, 1)]
    qids = [case.qid for case in cases]
    if len(qids) != len(set(qids)):
        raise _invalid("dataset", "contains duplicate qids")
    if tuple(qids) != EXPECTED_QIDS:
        raise _invalid("dataset", "qids must be portfolio-001 through portfolio-010")
    return cases
