"""Strict, content-free scoring for the severance refusal calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from rag.retrieval.refusal_policy import decide_retrieval_refusal

CANDIDATE_THRESHOLDS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
EXPECTED_QIDS = tuple(
    f"severance-policy-{number:03d}" for number in range(1, 31)
)
FORMAL_HIT_AT_5_BASELINE = 0.9666666666666667
FORMAL_MRR_AT_10_BASELINE = 0.9055555555555554

_FIELDS = {
    "qid",
    "question",
    "case_type",
    "answerable",
    "sources",
    "required_routes",
    "prohibited_routes",
    "expect_generation",
    "style_tags",
}
_OBSERVATION_FIELDS = {
    "qid",
    "case_type",
    "answerable",
    "source_ranks",
    "applied_routes",
    "top_score",
    "source_contract_passed",
    "route_contract_passed",
    "generation_expected",
}
_CASE_RESULT_FIELDS = _OBSERVATION_FIELDS | {
    "effective_threshold",
    "refused",
    "refusal_stage",
    "generation_allowed",
    "generation_contract_passed",
    "passed",
}
_TARGET_FIELDS = {
    "total",
    "passed_cases",
    "positive_routes",
    "positive_sources_at_5",
    "positive_generation_allowed",
    "collision_contracts",
    "passed",
}
_STRESS_FIELDS = {
    "questions",
    "answerable",
    "unanswerable",
    "direct_false_refusals",
    "direct_unanswerable_refusals",
    "direct_unanswerable_coverage",
    "passed",
}
_FORMAL_FIELDS = {
    "questions",
    "answerable",
    "unanswerable",
    "hit_at_5",
    "mrr_at_10",
    "direct_false_refusals",
    "passed",
}
_CANDIDATE_FIELDS = {
    "candidate_threshold",
    "target",
    "stress",
    "formal",
    "cases",
    "passed",
}
_POSITIVE_STYLES = {
    "statutory_chinese",
    "colloquial_chinese",
    "code_switch",
    "punctuation",
    "long_narrative",
    "reversed_regime_order",
    "formula_wording",
    "cap_wording",
    "mixed_tenure",
}
_COLLISION_STYLES = {
    "single_regime",
    "ordinary_termination",
    "notice_only",
    "wage_arrears",
    "generic_retirement",
    "unrelated_old_new",
    "partial_cue_collision",
}
_SEVERANCE_ROUTE = ("severance_comparison",)
_KNOWN_ROUTES = {
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
}
_SEVERANCE_SOURCES = (
    {"law": "勞工退休金條例", "article": "第 12 條"},
    {"law": "勞動基準法", "article": "第 17 條"},
)
_GUARD_FIELDS = {"qid", "answerable", "rank", "top_score", "applied_routes"}
_PROVENANCE_FIELDS = {
    "dataset_sha256",
    "corpus_snapshot_sha256",
    "source_artifact_sha256",
    "embedding_model",
    "embedding_revision",
    "reranker_model",
    "reranker_revision",
    "retrieval_configuration",
    "code_revision",
    "provider_adapters",
    "provider_requests",
}
_SOURCE_ARTIFACT_FIELDS = {
    "reliability_results",
    "reliability_trace",
    "reliability_formal_trace",
}
_RETRIEVAL_CONFIGURATION_FIELDS = {
    "chunking",
    "retrieval",
    "reranker",
    "top_k_retrieve",
    "top_k_final",
}


@dataclass(frozen=True)
class SeverancePolicyCase:
    qid: str
    question: str
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    sources: tuple[dict[str, str], ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expect_generation: bool
    style_tags: tuple[str, ...]


def _invalid(identity: object, message: str) -> ValueError:
    return ValueError(f"severance policy {identity}: {message}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _non_blank(value: object, *, field: str, identity: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(identity, f"{field} must be a non-blank string")
    return value.strip()


def _boolean(value: object, *, field: str, identity: object) -> bool:
    if type(value) is not bool:
        raise _invalid(identity, f"{field} must be boolean")
    return value


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
    normalized = tuple(
        _non_blank(item, field=field, identity=identity) for item in value
    )
    if len(normalized) != len(set(normalized)):
        raise _invalid(identity, f"{field} contains duplicates")
    return normalized


def _sources(value: object, *, identity: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise _invalid(identity, "sources must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {"law", "article"}:
            raise _invalid(identity, "source fields must equal law and article")
        source_id = (
            _non_blank(source["law"], field="source law", identity=identity),
            _non_blank(source["article"], field="source article", identity=identity),
        )
        if source_id in seen:
            raise _invalid(identity, "sources contain a duplicate source")
        seen.add(source_id)
        normalized.append({"law": source_id[0], "article": source_id[1]})
    return tuple(normalized)


def _parse_case(row: object, index: int) -> SeverancePolicyCase:
    identity: object = f"row {index}"
    if not isinstance(row, dict):
        raise _invalid(identity, "must be an object")
    identity = row.get("qid", identity)
    if set(row) != _FIELDS:
        raise _invalid(identity, f"fields must equal {sorted(_FIELDS)}")
    qid = _non_blank(row["qid"], field="qid", identity=identity)
    question = _non_blank(row["question"], field="question", identity=qid)
    case_type = row["case_type"]
    if case_type not in {"positive", "collision_negative"}:
        raise _invalid(qid, "case_type is invalid")
    expected_type = "positive" if index <= 15 else "collision_negative"
    if case_type != expected_type:
        raise _invalid(qid, "case_type ordering must be fifteen positives then negatives")

    answerable = _boolean(row["answerable"], field="answerable", identity=qid)
    expect_generation = _boolean(
        row["expect_generation"], field="expect_generation", identity=qid
    )
    sources = _sources(row["sources"], identity=qid)
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
    if set(required_routes) & set(prohibited_routes):
        raise _invalid(qid, "required_routes overlap prohibited_routes")
    if not set(required_routes + prohibited_routes) <= _KNOWN_ROUTES:
        raise _invalid(qid, "route contract contains an unknown route")
    if answerable and not sources:
        raise _invalid(qid, "answerable cases require sources")
    if not answerable and sources:
        raise _invalid(qid, "unanswerable cases must not have sources")

    if case_type == "positive":
        if not answerable or sources != _SEVERANCE_SOURCES:
            raise _invalid(qid, "positive sources must be the reviewed two-law contract")
        if required_routes != _SEVERANCE_ROUTE or prohibited_routes:
            raise _invalid(qid, "positive route contract must be severance-only")
        if not expect_generation:
            raise _invalid(qid, "positive cases must reach generation")
    elif required_routes == _SEVERANCE_ROUTE and not prohibited_routes:
        raise _invalid(qid, "collision case cannot request the severance-only policy")
    if answerable and not expect_generation:
        raise _invalid(qid, "answerable cases must reach generation")

    return SeverancePolicyCase(
        qid=qid,
        question=question,
        case_type=case_type,
        answerable=answerable,
        sources=sources,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        expect_generation=expect_generation,
        style_tags=style_tags,
    )


def load_cases(path: Path) -> list[SeverancePolicyCase]:
    """Load and validate the exact thirty reviewed cases."""

    rows: list[object] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, object_pairs_hook=_strict_object))
        except (json.JSONDecodeError, ValueError) as exc:
            raise _invalid(f"row {index}", "must be strict valid JSON") from exc
    cases = [_parse_case(row, index) for index, row in enumerate(rows, start=1)]
    qids = [case.qid for case in cases]
    if len(qids) != len(set(qids)):
        raise _invalid("dataset", "contains duplicate qids")
    if tuple(qids) != EXPECTED_QIDS:
        raise _invalid(
            "dataset", "qids must be severance-policy-001 through -030"
        )
    positive_styles = {tag for case in cases[:15] for tag in case.style_tags}
    if not _POSITIVE_STYLES <= positive_styles:
        raise _invalid("dataset", "positive style coverage is incomplete")
    collision_styles = {tag for case in cases[15:] for tag in case.style_tags}
    if not _COLLISION_STYLES <= collision_styles:
        raise _invalid("dataset", "collision style coverage is incomplete")
    return cases


def _unit_interval(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be finite and between zero and one")
    return normalized


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _routes(value: object, *, field: str, require_tuple: bool) -> tuple[str, ...]:
    expected_type = tuple if require_tuple else list
    if not isinstance(value, expected_type):
        raise ValueError(f"{field} must be a {expected_type.__name__}")
    if not all(isinstance(route, str) and route.strip() for route in value):
        raise ValueError(f"{field} must contain non-blank strings")
    normalized = tuple(route.strip() for route in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} contains duplicates")
    return normalized


def build_case_observation(
    case: SeverancePolicyCase,
    *,
    source_ranks: dict[str, int],
    applied_routes: tuple[str, ...],
    top_score: float,
) -> dict[str, Any]:
    """Return one validated content-free retrieval observation."""

    if not isinstance(case, SeverancePolicyCase):
        raise ValueError("case must be a SeverancePolicyCase")
    if not isinstance(source_ranks, dict):
        raise ValueError("source_ranks must be a dict")
    expected_sources = {_source_key(source) for source in case.sources}
    if not set(source_ranks) <= expected_sources:
        raise ValueError("source_ranks may contain only reviewed source identities")
    normalized_ranks: dict[str, int] = {}
    for source, rank in source_ranks.items():
        if (
            not isinstance(source, str)
            or not source.strip()
            or type(rank) is not int
            or rank < 1
        ):
            raise ValueError("source_ranks require non-blank keys and positive integers")
        normalized_ranks[source.strip()] = rank
    routes = _routes(applied_routes, field="applied_routes", require_tuple=True)
    score = _unit_interval(top_score, field="top_score")
    source_contract_passed = case.case_type != "positive" or all(
        normalized_ranks.get(source, 6) <= 5 for source in expected_sources
    )
    if case.case_type == "positive":
        route_contract_passed = routes == _SEVERANCE_ROUTE
    else:
        route_contract_passed = (
            routes != _SEVERANCE_ROUTE
            and all(route in routes for route in case.required_routes)
            and all(route not in routes for route in case.prohibited_routes)
        )
    return {
        "qid": case.qid,
        "case_type": case.case_type,
        "answerable": case.answerable,
        "source_ranks": dict(sorted(normalized_ranks.items())),
        "applied_routes": list(routes),
        "top_score": score,
        "source_contract_passed": source_contract_passed,
        "route_contract_passed": route_contract_passed,
        "generation_expected": case.expect_generation,
    }


def _validated_observations(observations: object) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(observations, start=1):
        if not isinstance(row, dict) or set(row) != _OBSERVATION_FIELDS:
            raise ValueError(f"observation fields are invalid at row {index}")
        qid = _non_blank(row["qid"], field="qid", identity=f"observation {index}")
        case_type = row["case_type"]
        if case_type not in {"positive", "collision_negative"}:
            raise ValueError(f"observation {qid}: invalid case_type")
        answerable = _boolean(row["answerable"], field="answerable", identity=qid)
        generation_expected = _boolean(
            row["generation_expected"], field="generation_expected", identity=qid
        )
        source_contract = _boolean(
            row["source_contract_passed"],
            field="source_contract_passed",
            identity=qid,
        )
        route_contract = _boolean(
            row["route_contract_passed"],
            field="route_contract_passed",
            identity=qid,
        )
        if not isinstance(row["source_ranks"], dict):
            raise ValueError(f"observation {qid}: source_ranks must be a dict")
        ranks: dict[str, int] = {}
        for source, rank in row["source_ranks"].items():
            if (
                not isinstance(source, str)
                or not source.strip()
                or type(rank) is not int
                or rank < 1
            ):
                raise ValueError(f"observation {qid}: invalid source_ranks")
            ranks[source.strip()] = rank
        routes = _routes(
            row["applied_routes"], field="applied_routes", require_tuple=False
        )
        normalized.append(
            {
                "qid": qid,
                "case_type": case_type,
                "answerable": answerable,
                "source_ranks": dict(sorted(ranks.items())),
                "applied_routes": list(routes),
                "top_score": _unit_interval(row["top_score"], field="top_score"),
                "source_contract_passed": source_contract,
                "route_contract_passed": route_contract,
                "generation_expected": generation_expected,
            }
        )
    qids = [row["qid"] for row in normalized]
    if len(qids) != len(set(qids)):
        raise ValueError("observations contain duplicate qids")
    if tuple(qids) != EXPECTED_QIDS:
        raise ValueError("observations must contain the exact thirty qids in order")
    expected_types = ["positive"] * 15 + ["collision_negative"] * 15
    if [row["case_type"] for row in normalized] != expected_types:
        raise ValueError("observation case types must preserve reviewed ordering")
    return normalized


def _validated_guard_rows(
    rows: object,
    *,
    label: str,
    expected_qids: tuple[str, ...],
    answerable_count: int,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows must be a list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != _GUARD_FIELDS:
            raise ValueError(f"{label} rows have invalid fields at row {index}")
        qid = _non_blank(row["qid"], field="qid", identity=f"{label} row {index}")
        answerable = _boolean(row["answerable"], field="answerable", identity=qid)
        rank = row["rank"]
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError(f"{label} row {qid}: rank must be null or positive")
        if not answerable and rank is not None:
            raise ValueError(f"{label} row {qid}: unanswerable rank must be null")
        routes = _routes(
            row["applied_routes"], field="applied_routes", require_tuple=False
        )
        normalized.append(
            {
                "qid": qid,
                "answerable": answerable,
                "rank": rank,
                "top_score": _unit_interval(row["top_score"], field="top_score"),
                "applied_routes": list(routes),
            }
        )
    if tuple(row["qid"] for row in normalized) != expected_qids:
        raise ValueError(f"{label} rows must contain the exact committed qids")
    if sum(row["answerable"] for row in normalized) != answerable_count:
        raise ValueError(f"{label} rows have the wrong answerable split")
    return normalized


def _guard_refused(
    row: dict[str, Any], *, candidate_threshold: float, global_threshold: float
) -> bool:
    decision = decide_retrieval_refusal(
        has_hits=True,
        reranker_enabled=True,
        applied_routes=tuple(row["applied_routes"]),
        top_score=row["top_score"],
        global_threshold=global_threshold,
        severance_comparison_threshold=candidate_threshold,
    )
    return decision.refused


def evaluate_candidate(
    observations: list[dict[str, Any]],
    *,
    candidate_threshold: float,
    global_threshold: float,
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute target and guard gates through the shared policy."""

    candidate = _unit_interval(
        candidate_threshold, field="candidate_threshold"
    )
    if candidate not in CANDIDATE_THRESHOLDS:
        raise ValueError("candidate_threshold must belong to the committed grid")
    global_value = _unit_interval(global_threshold, field="global_threshold")
    target_rows = _validated_observations(observations)
    stress = _validated_guard_rows(
        stress_rows,
        label="stress",
        expected_qids=tuple(f"stress-{number:03d}" for number in range(1, 61)),
        answerable_count=40,
    )
    formal = _validated_guard_rows(
        formal_rows,
        label="formal",
        expected_qids=tuple(f"eval-{number:02d}" for number in range(1, 41)),
        answerable_count=30,
    )

    case_results = []
    for row in target_rows:
        decision = decide_retrieval_refusal(
            has_hits=True,
            reranker_enabled=True,
            applied_routes=tuple(row["applied_routes"]),
            top_score=row["top_score"],
            global_threshold=global_value,
            severance_comparison_threshold=candidate,
        )
        generation_allowed = not decision.refused
        generation_contract = generation_allowed is row["generation_expected"]
        required_contracts = row["route_contract_passed"] and generation_contract
        passed = required_contracts and (
            row["source_contract_passed"] or row["case_type"] == "collision_negative"
        )
        case_results.append(
            {
                **row,
                "effective_threshold": decision.effective_threshold,
                "refused": decision.refused,
                "refusal_stage": decision.refusal_stage,
                "generation_allowed": generation_allowed,
                "generation_contract_passed": generation_contract,
                "passed": passed,
            }
        )

    positives = case_results[:15]
    collisions = case_results[15:]
    passed_cases = sum(row["passed"] for row in case_results)
    target_summary = {
        "total": 30,
        "passed_cases": passed_cases,
        "positive_routes": sum(row["route_contract_passed"] for row in positives),
        "positive_sources_at_5": sum(
            row["source_contract_passed"] for row in positives
        ),
        "positive_generation_allowed": sum(
            row["generation_allowed"] for row in positives
        ),
        "collision_contracts": sum(
            row["route_contract_passed"]
            and row["generation_contract_passed"]
            for row in collisions
        ),
        "passed": passed_cases == 30,
    }

    stress_decisions = [
        _guard_refused(
            row,
            candidate_threshold=candidate,
            global_threshold=global_value,
        )
        for row in stress
    ]
    stress_false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(stress, stress_decisions, strict=True)
    )
    stress_unanswerable_refusals = sum(
        refused and not row["answerable"]
        for row, refused in zip(stress, stress_decisions, strict=True)
    )
    stress_summary = {
        "questions": 60,
        "answerable": 40,
        "unanswerable": 20,
        "direct_false_refusals": stress_false_refusals,
        "direct_unanswerable_refusals": stress_unanswerable_refusals,
        "direct_unanswerable_coverage": stress_unanswerable_refusals / 20,
        "passed": stress_false_refusals == 0
        and stress_unanswerable_refusals >= 17,
    }

    formal_answerable = [row for row in formal if row["answerable"]]
    formal_decisions = [
        _guard_refused(
            row,
            candidate_threshold=candidate,
            global_threshold=global_value,
        )
        for row in formal
    ]
    formal_false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(formal, formal_decisions, strict=True)
    )
    hit_at_5 = sum(
        row["rank"] is not None and row["rank"] <= 5 for row in formal_answerable
    ) / 30
    mrr_at_10 = sum(
        1 / row["rank"]
        for row in formal_answerable
        if row["rank"] is not None and row["rank"] <= 10
    ) / 30
    formal_summary = {
        "questions": 40,
        "answerable": 30,
        "unanswerable": 10,
        "hit_at_5": hit_at_5,
        "mrr_at_10": mrr_at_10,
        "direct_false_refusals": formal_false_refusals,
        "passed": hit_at_5 >= FORMAL_HIT_AT_5_BASELINE
        and mrr_at_10 >= FORMAL_MRR_AT_10_BASELINE
        and formal_false_refusals == 0,
    }
    passed = target_summary["passed"] and stress_summary["passed"] and formal_summary[
        "passed"
    ]
    return {
        "candidate_threshold": candidate,
        "target": target_summary,
        "stress": stress_summary,
        "formal": formal_summary,
        "cases": case_results,
        "passed": passed,
    }


def _count(value: object, *, field: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an integer from zero through {maximum}")
    return value


def _metric(value: object, *, field: str) -> float:
    return _unit_interval(value, field=field)


def _validated_candidate_results(
    candidate_results: object,
) -> list[dict[str, Any]]:
    if not isinstance(candidate_results, list):
        raise ValueError("candidate_results must be a list")
    normalized: list[dict[str, Any]] = []
    for result in candidate_results:
        if not isinstance(result, dict) or set(result) != _CANDIDATE_FIELDS:
            raise ValueError("candidate result fields are invalid")
        threshold = _unit_interval(
            result["candidate_threshold"], field="candidate_threshold"
        )
        target = result["target"]
        stress = result["stress"]
        formal = result["formal"]
        if not isinstance(target, dict) or set(target) != _TARGET_FIELDS:
            raise ValueError("candidate target fields are invalid")
        if not isinstance(stress, dict) or set(stress) != _STRESS_FIELDS:
            raise ValueError("candidate stress fields are invalid")
        if not isinstance(formal, dict) or set(formal) != _FORMAL_FIELDS:
            raise ValueError("candidate formal fields are invalid")
        cases = result["cases"]
        if not isinstance(cases, list) or len(cases) != 30:
            raise ValueError("candidate cases must contain thirty rows")
        if any(not isinstance(row, dict) or set(row) != _CASE_RESULT_FIELDS for row in cases):
            raise ValueError("candidate case fields are invalid")
        if tuple(row["qid"] for row in cases) != EXPECTED_QIDS:
            raise ValueError("candidate cases have invalid qids")
        for index, row in enumerate(cases, start=1):
            expected_type = "positive" if index <= 15 else "collision_negative"
            if row["case_type"] != expected_type:
                raise ValueError("candidate cases have invalid case types")
            _boolean(row["answerable"], field="answerable", identity=row["qid"])
            if not isinstance(row["source_ranks"], dict) or any(
                not isinstance(source, str)
                or not source.strip()
                or type(rank) is not int
                or rank < 1
                for source, rank in row["source_ranks"].items()
            ):
                raise ValueError("candidate case source_ranks are invalid")
            _routes(
                row["applied_routes"],
                field="applied_routes",
                require_tuple=False,
            )
            _unit_interval(row["top_score"], field="top_score")
            _unit_interval(
                row["effective_threshold"], field="effective_threshold"
            )
            for field in (
                "source_contract_passed",
                "route_contract_passed",
                "generation_expected",
                "refused",
                "generation_allowed",
                "generation_contract_passed",
                "passed",
            ):
                _boolean(row[field], field=field, identity=row["qid"])
            if row["refusal_stage"] not in {None, "threshold"}:
                raise ValueError("candidate case refusal_stage is invalid")
            if row["refused"] is not (row["refusal_stage"] is not None):
                raise ValueError("candidate case refusal decision is inconsistent")
            if row["generation_allowed"] is row["refused"]:
                raise ValueError("candidate case generation decision is inconsistent")
            expected_generation_contract = (
                row["generation_allowed"] is row["generation_expected"]
            )
            if row["generation_contract_passed"] is not expected_generation_contract:
                raise ValueError("candidate generation contract is inconsistent")

        target_total = _count(target["total"], field="target total", maximum=30)
        target_passed_cases = _count(
            target["passed_cases"], field="target passed_cases", maximum=30
        )
        if target_total != 30:
            raise ValueError("target total must equal thirty")
        for field in (
            "positive_routes",
            "positive_sources_at_5",
            "positive_generation_allowed",
            "collision_contracts",
        ):
            _count(target[field], field=f"target {field}", maximum=15)
        target_passed = _boolean(
            target["passed"], field="target passed", identity=threshold
        )
        if target_passed is not (target_passed_cases == 30):
            raise ValueError("candidate target passed flag is inconsistent")

        if (
            stress["questions"],
            stress["answerable"],
            stress["unanswerable"],
        ) != (60, 40, 20):
            raise ValueError("candidate stress counts are invalid")
        stress_false = _count(
            stress["direct_false_refusals"],
            field="stress direct_false_refusals",
            maximum=40,
        )
        stress_refusals = _count(
            stress["direct_unanswerable_refusals"],
            field="stress direct_unanswerable_refusals",
            maximum=20,
        )
        coverage = _metric(
            stress["direct_unanswerable_coverage"], field="stress coverage"
        )
        if coverage != stress_refusals / 20:
            raise ValueError("candidate stress coverage is inconsistent")
        stress_passed = _boolean(
            stress["passed"], field="stress passed", identity=threshold
        )
        if stress_passed is not (stress_false == 0 and stress_refusals >= 17):
            raise ValueError("candidate stress passed flag is inconsistent")

        if (
            formal["questions"],
            formal["answerable"],
            formal["unanswerable"],
        ) != (40, 30, 10):
            raise ValueError("candidate formal counts are invalid")
        formal_hit = _metric(formal["hit_at_5"], field="formal hit_at_5")
        formal_mrr = _metric(formal["mrr_at_10"], field="formal mrr_at_10")
        formal_false = _count(
            formal["direct_false_refusals"],
            field="formal direct_false_refusals",
            maximum=30,
        )
        formal_passed = _boolean(
            formal["passed"], field="formal passed", identity=threshold
        )
        formal_gate = (
            formal_hit >= FORMAL_HIT_AT_5_BASELINE
            and formal_mrr >= FORMAL_MRR_AT_10_BASELINE
            and formal_false == 0
        )
        if formal_passed is not formal_gate:
            raise ValueError("candidate formal passed flag is inconsistent")
        passed = _boolean(result["passed"], field="candidate passed", identity=threshold)
        if passed is not (target_passed and stress_passed and formal_passed):
            raise ValueError("candidate passed flag is inconsistent")
        normalized.append(result)

    thresholds = [result["candidate_threshold"] for result in normalized]
    if len(thresholds) != len(set(thresholds)) or set(thresholds) != set(
        CANDIDATE_THRESHOLDS
    ):
        raise ValueError("candidate grid must equal the committed seven thresholds")
    return sorted(normalized, key=lambda result: result["candidate_threshold"])


def _complete_gate_set(result: dict[str, Any]) -> bool:
    return (
        result["target"]["passed_cases"] == 30
        and result["stress"]["direct_false_refusals"] == 0
        and result["stress"]["direct_unanswerable_refusals"] >= 17
        and result["formal"]["hit_at_5"] >= FORMAL_HIT_AT_5_BASELINE
        and result["formal"]["mrr_at_10"] >= FORMAL_MRR_AT_10_BASELINE
        and result["formal"]["direct_false_refusals"] == 0
    )


def select_highest_passing_threshold(
    candidate_results: list[dict[str, Any]],
) -> float:
    """Return the greatest candidate whose complete gate set passes."""

    validated = _validated_candidate_results(candidate_results)
    passing = [
        result["candidate_threshold"]
        for result in validated
        if _complete_gate_set(result)
    ]
    if not passing:
        raise RuntimeError("no candidate threshold satisfies the complete gate set")
    return max(passing)


def _hex(value: object, *, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase {length}-character hex value")
    return value


def _public_identifier(value: object, *, field: str) -> str:
    normalized = _non_blank(value, field=field, identity="provenance")
    if "://" in normalized or Path(normalized).is_absolute() or PureWindowsPath(
        normalized
    ).is_absolute():
        raise ValueError(f"{field} must not contain a URL or local absolute path")
    return normalized


def _validated_provenance(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError(f"provenance fields must equal {sorted(_PROVENANCE_FIELDS)}")
    source_hashes = provenance["source_artifact_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != _SOURCE_ARTIFACT_FIELDS:
        raise ValueError("source_artifact_sha256 fields are invalid")
    configuration = provenance["retrieval_configuration"]
    if (
        not isinstance(configuration, dict)
        or set(configuration) != _RETRIEVAL_CONFIGURATION_FIELDS
    ):
        raise ValueError("retrieval_configuration fields are invalid")
    for field in ("chunking", "retrieval"):
        _public_identifier(configuration[field], field=field)
    if type(configuration["reranker"]) is not bool or not configuration["reranker"]:
        raise ValueError("retrieval_configuration reranker must be true")
    for field in ("top_k_retrieve", "top_k_final"):
        if type(configuration[field]) is not int or configuration[field] < 1:
            raise ValueError(f"retrieval_configuration {field} must be positive")
    for field in ("provider_adapters", "provider_requests"):
        if type(provenance[field]) is not int or provenance[field] != 0:
            raise ValueError(f"{field} must be zero")
    normalized = {
        "dataset_sha256": _hex(
            provenance["dataset_sha256"], field="dataset_sha256", length=64
        ),
        "corpus_snapshot_sha256": _hex(
            provenance["corpus_snapshot_sha256"],
            field="corpus_snapshot_sha256",
            length=64,
        ),
        "source_artifact_sha256": {
            field: _hex(source_hashes[field], field=field, length=64)
            for field in sorted(_SOURCE_ARTIFACT_FIELDS)
        },
        "embedding_model": _public_identifier(
            provenance["embedding_model"], field="embedding_model"
        ),
        "embedding_revision": _hex(
            provenance["embedding_revision"], field="embedding_revision", length=40
        ),
        "reranker_model": _public_identifier(
            provenance["reranker_model"], field="reranker_model"
        ),
        "reranker_revision": _hex(
            provenance["reranker_revision"], field="reranker_revision", length=40
        ),
        "retrieval_configuration": dict(configuration),
        "code_revision": _hex(
            provenance["code_revision"], field="code_revision", length=40
        ),
        "provider_adapters": 0,
        "provider_requests": 0,
    }
    return normalized


def build_official_artifact(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict content-free official schema."""

    target_observations = _validated_observations(observations)
    candidates = _validated_candidate_results(candidate_results)
    selected_threshold = select_highest_passing_threshold(candidates)
    selected = next(
        result
        for result in candidates
        if result["candidate_threshold"] == selected_threshold
    )
    for observation, result in zip(
        target_observations, selected["cases"], strict=True
    ):
        for field in _OBSERVATION_FIELDS:
            if result[field] != observation[field]:
                raise ValueError(f"selected case {result['qid']} does not match observation")

    public_cases = []
    for result in selected["cases"]:
        public_cases.append(
            {
                "qid": result["qid"],
                "case_type": result["case_type"],
                "answerable": result["answerable"],
                "source_ranks": dict(result["source_ranks"]),
                "applied_routes": list(result["applied_routes"]),
                "top_score": round(result["top_score"], 6),
                "effective_threshold": round(result["effective_threshold"], 6),
                "refused": result["refused"],
                "refusal_stage": result["refusal_stage"],
                "source_contract_passed": result["source_contract_passed"],
                "route_contract_passed": result["route_contract_passed"],
                "generation_expected": result["generation_expected"],
                "generation_allowed": result["generation_allowed"],
                "generation_contract_passed": result[
                    "generation_contract_passed"
                ],
                "passed": result["passed"],
            }
        )
    public_candidates = [
        {
            "candidate_threshold": result["candidate_threshold"],
            "target": dict(result["target"]),
            "stress": dict(result["stress"]),
            "formal": dict(result["formal"]),
            "passed": result["passed"],
        }
        for result in candidates
    ]
    return {
        "schema_version": "1.0",
        "provenance": _validated_provenance(provenance),
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "selected_threshold": selected_threshold,
        "candidates": public_candidates,
        "cases": public_cases,
    }
