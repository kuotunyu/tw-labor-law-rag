"""Strict contracts for the compact v0.3.5 portfolio regression."""

from __future__ import annotations

import hashlib
import json
import math
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


def _source_key(law: str, article: str) -> str:
    return f"{law}|{article}"


def build_result(
    case: PortfolioCase,
    *,
    retrieved: list[tuple[str, str, int]],
    applied_routes: list[str],
    threshold_refused: bool,
    top_score: float,
) -> dict[str, Any]:
    """Score one retrieval result without retaining questions or law content."""

    source_ranks: dict[str, int] = {}
    for item in retrieved:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or not all(isinstance(value, str) and value.strip() for value in item[:2])
            or type(item[2]) is not int
            or item[2] <= 0
        ):
            raise ValueError("retrieved entries require law, article, and positive rank")
        key = _source_key(item[0].strip(), item[1].strip())
        if key in source_ranks:
            raise ValueError("retrieved entries contain a duplicate source")
        source_ranks[key] = item[2]

    if not isinstance(applied_routes, list) or not all(
        isinstance(route, str) and route.strip() for route in applied_routes
    ):
        raise ValueError("applied_routes must contain non-blank strings")
    normalized_routes = sorted(route.strip() for route in applied_routes)
    if len(normalized_routes) != len(set(normalized_routes)):
        raise ValueError("applied_routes contains duplicates")
    if type(threshold_refused) is not bool:
        raise ValueError("threshold_refused must be boolean")
    if (
        isinstance(top_score, bool)
        or not isinstance(top_score, (int, float))
        or not math.isfinite(top_score)
        or top_score < 0
    ):
        raise ValueError("top_score must be a finite non-negative number")

    expected_keys = [
        _source_key(source["law"], source["article"]) for source in case.sources
    ]
    prohibited_keys = [
        _source_key(source["law"], source["article"])
        for source in case.prohibited_sources
    ]
    required_source_hits_at_5 = sum(
        key in source_ranks and source_ranks[key] <= 5 for key in expected_keys
    )
    prohibited_source_hits_at_5 = sum(
        key in source_ranks and source_ranks[key] <= 5 for key in prohibited_keys
    )
    source_contract_passed = (
        required_source_hits_at_5 == len(expected_keys)
        and prohibited_source_hits_at_5 == 0
    )
    route_contract_passed = all(
        route in normalized_routes for route in case.required_routes
    ) and all(route not in normalized_routes for route in case.prohibited_routes)
    threshold_contract_passed = threshold_refused is case.expect_threshold_refusal
    answerability_contract_passed = (
        (not threshold_refused and source_contract_passed)
        if case.answerable
        else threshold_refused
    )
    passed = (
        answerability_contract_passed
        and threshold_contract_passed
        and route_contract_passed
    )
    return {
        "qid": case.qid,
        "answerable": case.answerable,
        "expected_source_count": len(expected_keys),
        "source_ranks": dict(sorted(source_ranks.items())),
        "required_source_hits_at_5": required_source_hits_at_5,
        "prohibited_source_hits_at_5": prohibited_source_hits_at_5,
        "applied_routes": normalized_routes,
        "route_contract_passed": route_contract_passed,
        "threshold_expected": case.expect_threshold_refusal,
        "threshold_refused": threshold_refused,
        "threshold_contract_passed": threshold_contract_passed,
        "refusal_stage": "threshold" if threshold_refused else None,
        "generation_allowed": not threshold_refused,
        "generation_called": False,
        "top_score": round(float(top_score), 6),
        "passed": passed,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the compact regression acceptance summary."""

    if not results:
        raise ValueError("portfolio results must not be empty")
    qids = [result.get("qid") for result in results]
    if len(qids) != len(set(qids)):
        raise ValueError("portfolio results contain duplicate qids")
    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]
    expected_sources = sum(result["expected_source_count"] for result in results)
    source_hits = sum(result["required_source_hits_at_5"] for result in results)
    summary = {
        "total": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "source_recall_at_5": _rate(source_hits, expected_sources),
        "answerable_pass_rate": _rate(
            sum(bool(result["passed"]) for result in answerable), len(answerable)
        ),
        "threshold_refusal_accuracy": _rate(
            sum(bool(result["threshold_contract_passed"]) for result in results),
            len(results),
        ),
        "route_accuracy": _rate(
            sum(bool(result["route_contract_passed"]) for result in results),
            len(results),
        ),
        "passed": all(result["passed"] for result in results),
    }
    return summary


def build_artifact(
    *,
    dataset_path: Path,
    snapshot_path: Path,
    code_revision: str,
    configuration: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind content-free case results to exact public inputs and code."""

    if len(code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in code_revision
    ):
        raise ValueError("code_revision must be a lowercase 40-character Git hash")
    ordered = sorted(results, key=lambda result: result["qid"])
    if tuple(result["qid"] for result in ordered) != EXPECTED_QIDS:
        raise ValueError("portfolio results must contain the exact ten qids")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "dataset": {
            "path": "eval/dataset/portfolio_demo_v0.3.5.jsonl",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "questions": len(ordered),
        },
        "corpus_snapshot": {
            "path": "release/corpus_snapshot.json",
            "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "snapshot_date": snapshot["snapshot_date"],
            "laws": snapshot["law_count"],
            "articles": snapshot["article_count"],
        },
        "code_revision": code_revision,
        "configuration": configuration,
        "summary": summarize_results(ordered),
        "cases": ordered,
    }
