"""Strict, content-free scoring for the severance refusal calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag.retrieval.pipeline import (
    MULTI_VIEW_MERGE_POLICY_VERSION,
    SEVERANCE_SEMANTIC_VIEW_SHA256,
)
from rag.retrieval.refusal_policy import decide_retrieval_refusal

CANDIDATE_THRESHOLDS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
REQUIRED_TRACKED_CODE_PATHS = frozenset(
    {
        ".python-version",
        "Dockerfile",
        "pyproject.toml",
        "src/rag/indexing/dict/legal_terms.txt",
        "uv.lock",
    }
)
DECLARED_INPUT_PATHS = {
    "corpus_snapshot": "release/corpus_snapshot.json",
    "formal_dataset": "eval/dataset/eval_set.jsonl",
    "stress_dataset": "eval/dataset/reliability_stress_v0.3.1.jsonl",
    "target_dataset": "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
}
EXPECTED_QIDS = tuple(
    f"severance-policy-{number:03d}" for number in range(1, 31)
)
FORMAL_HIT_AT_5_BASELINE = 0.9666666666666667
FORMAL_MRR_AT_10_BASELINE = 0.9055555555555554
_GLOBAL_THRESHOLD = 0.03
_SCHEMA_VERSION = "1.3"
_PRECISION_MODE = "fp32"
_SEMANTIC_VIEW_SHA256 = SEVERANCE_SEMANTIC_VIEW_SHA256
_MERGE_POLICY_VERSION = MULTI_VIEW_MERGE_POLICY_VERSION
_PRIMARY_SCORE_SEMANTICS = "full_precision_primary_query_top_score"
_WINDOWS_PYWIN32_RUNTIME_LAYOUT = (
    "Lib/site-packages/win32",
    "Lib/site-packages/win32/lib",
)

_DATASET_FIELDS = {
    "qid",
    "question",
    "case_type",
    "answerable",
    "sources",
    "required_routes",
    "prohibited_routes",
    "expected_outcome",
    "style_tags",
}
_OBSERVATION_FIELDS = {
    "qid",
    "source_ranks",
    "applied_routes",
    "hit_count",
    "top_score",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
}
_CASE_RESULT_FIELDS = {
    "qid",
    "case_type",
    "answerable",
    "source_ranks",
    "applied_routes",
    "hit_count",
    "top_score",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
    "effective_threshold",
    "refused",
    "refusal_stage",
    "source_contract_passed",
    "route_contract_passed",
    "expected_outcome",
    "generation_allowed",
    "outcome_contract_passed",
    "passed",
}
_GUARD_INPUT_FIELDS = {
    "qid",
    "answerable",
    "rank",
    "hit_count",
    "top_score",
    "applied_routes",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
}
_GUARD_EVIDENCE_FIELDS = _GUARD_INPUT_FIELDS | {
    "has_hits",
    "reranker_enabled",
}
_CANDIDATE_FIELDS = {
    "candidate_threshold",
    "global_threshold",
    "target",
    "stress",
    "formal",
    "cases",
    "stress_evidence",
    "formal_evidence",
    "passed",
}
_PROVENANCE_FIELDS = {
    "dataset_sha256",
    "corpus_snapshot_sha256",
    "source_artifact_sha256",
    "revision_binding",
    "environment_binding",
    "embedding_model",
    "embedding_revision",
    "reranker_model",
    "reranker_revision",
    "retrieval_configuration",
    "execution_device",
    "precision_mode",
    "local_files_only",
    "semantic_view_sha256",
    "merge_policy_version",
    "primary_score_semantics",
    "source_tree_clean",
    "code_revision",
    "run_origin",
    "provider_adapters",
    "provider_requests",
}
_SOURCE_ARTIFACT_FIELDS = {
    "stress_dataset",
    "formal_dataset",
}
_TOP_K_FINAL = 5
_RETRIEVAL_CONFIGURATION = {
    "chunking": "structure",
    "retrieval": "hybrid",
    "reranker": True,
    "top_k_retrieve": 20,
    "top_k_final": _TOP_K_FINAL,
    "rrf_k": 60,
}
_EMBEDDING_MODEL = "BAAI/bge-m3"
_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_RUN_ORIGIN = "fresh_offline_retrieval"
_KNOWN_ROUTES = {
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
}
_SEVERANCE_ROUTE = ("severance_comparison",)
_WAGE_ROUTE = ("wage_arrears_termination",)
_MULTI_ROUTE = ("severance_comparison", "wage_arrears_termination")
_PENSION_12 = "勞工退休金條例|第 12 條"
_LABOR_17 = "勞動基準法|第 17 條"
_LABOR_11 = "勞動基準法|第 11 條"
_LABOR_16 = "勞動基準法|第 16 條"
_LABOR_14 = "勞動基準法|第 14 條"
_PENSION_24 = "勞工退休金條例|第 24 條"
_LABOR_54 = "勞動基準法|第 54 條"
_LABOR_30 = "勞動基準法|第 30 條"
_CANONICAL_SOURCE_KEYS = {
    _PENSION_12,
    _LABOR_17,
    _LABOR_11,
    _LABOR_16,
    _LABOR_14,
    _PENSION_24,
    _LABOR_54,
    _LABOR_30,
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


@dataclass(frozen=True)
class SeverancePolicyCase:
    qid: str
    question: str
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    sources: tuple[dict[str, str], ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expected_outcome: Literal["generation", "no_hits", "threshold"]
    style_tags: tuple[str, ...]


@dataclass(frozen=True)
class _CaseContract:
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    source_keys: tuple[str, ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expected_outcome: Literal["generation", "no_hits", "threshold"]


def _collision(
    source_keys: tuple[str, ...],
    *,
    answerable: bool = True,
    required_routes: tuple[str, ...] = (),
    prohibited_routes: tuple[str, ...] = _SEVERANCE_ROUTE,
    expected_outcome: Literal["generation", "no_hits", "threshold"] = "generation",
) -> _CaseContract:
    return _CaseContract(
        case_type="collision_negative",
        answerable=answerable,
        source_keys=source_keys,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        expected_outcome=expected_outcome,
    )


_POSITIVE_CONTRACT = _CaseContract(
    case_type="positive",
    answerable=True,
    source_keys=(_PENSION_12, _LABOR_17),
    required_routes=_SEVERANCE_ROUTE,
    prohibited_routes=(),
    expected_outcome="generation",
)
_CASE_CONTRACTS = {
    **{qid: _POSITIVE_CONTRACT for qid in EXPECTED_QIDS[:15]},
    "severance-policy-016": _collision((_PENSION_12,)),
    "severance-policy-017": _collision((_LABOR_17,)),
    "severance-policy-018": _collision((_LABOR_11,)),
    "severance-policy-019": _collision((_LABOR_16,)),
    "severance-policy-020": _collision(
        (_LABOR_14,), required_routes=_WAGE_ROUTE
    ),
    "severance-policy-021": _collision((_PENSION_24,)),
    "severance-policy-022": _collision((_LABOR_54,)),
    "severance-policy-023": _collision(
        (), answerable=False, expected_outcome="no_hits"
    ),
    "severance-policy-024": _collision((), answerable=False),
    "severance-policy-025": _collision((_PENSION_12,)),
    "severance-policy-026": _collision((_LABOR_17,)),
    "severance-policy-027": _collision(
        (), answerable=False, expected_outcome="threshold"
    ),
    "severance-policy-028": _collision((_LABOR_16,)),
    "severance-policy-029": _collision((_LABOR_30,)),
    "severance-policy-030": _collision(
        (_LABOR_14, _PENSION_12, _LABOR_17),
        required_routes=_MULTI_ROUTE,
        prohibited_routes=(),
    ),
}
_STRESS_QIDS = tuple(f"stress-{number:03d}" for number in range(1, 61))
_FORMAL_QIDS = tuple(f"eval-{number:02d}" for number in range(1, 41))
_STRESS_ROUTES = {
    "stress-003": _SEVERANCE_ROUTE,
    "stress-010": _WAGE_ROUTE,
    "stress-037": _SEVERANCE_ROUTE,
    "stress-038": _WAGE_ROUTE,
}
_FORMAL_ROUTES = {
    "eval-03": _SEVERANCE_ROUTE,
    "eval-10": _WAGE_ROUTE,
}


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


def _unit_interval(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be finite and between zero and one")
    return normalized


def _validated_global_threshold(value: object) -> float:
    normalized = _unit_interval(value, field="global_threshold")
    if normalized != _GLOBAL_THRESHOLD:
        raise ValueError("global_threshold must equal the committed 0.03")
    return normalized


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


def _routes(value: object, *, field: str, require_tuple: bool) -> tuple[str, ...]:
    expected_type = tuple if require_tuple else list
    if not isinstance(value, expected_type):
        raise ValueError(f"{field} must be a {expected_type.__name__}")
    if not all(isinstance(route, str) and route.strip() for route in value):
        raise ValueError(f"{field} must contain non-blank strings")
    normalized = tuple(route.strip() for route in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} contains duplicates")
    if not set(normalized) <= _KNOWN_ROUTES:
        raise ValueError(f"{field} values must belong to the route allowlist")
    return normalized


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _sources(value: object, *, identity: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise _invalid(identity, "sources must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {"law", "article"}:
            raise _invalid(identity, "source fields must equal law and article")
        normalized_source = {
            "law": _non_blank(
                source["law"], field="source law", identity=identity
            ),
            "article": _non_blank(
                source["article"], field="source article", identity=identity
            ),
        }
        key = _source_key(normalized_source)
        if key in seen:
            raise _invalid(identity, "sources contain a duplicate source")
        seen.add(key)
        normalized.append(normalized_source)
    return tuple(normalized)


def _parse_case(row: object, index: int) -> SeverancePolicyCase:
    identity: object = f"row {index}"
    if not isinstance(row, dict):
        raise _invalid(identity, "must be an object")
    identity = row.get("qid", identity)
    if set(row) != _DATASET_FIELDS:
        raise _invalid(identity, f"fields must equal {sorted(_DATASET_FIELDS)}")
    qid = _non_blank(row["qid"], field="qid", identity=identity)
    question = _non_blank(row["question"], field="question", identity=qid)
    case_type = row["case_type"]
    expected_type = "positive" if index <= 15 else "collision_negative"
    if case_type != expected_type:
        raise _invalid(qid, "case_type ordering must be fifteen positives then negatives")
    answerable = _boolean(row["answerable"], field="answerable", identity=qid)
    expected_outcome = row["expected_outcome"]
    if expected_outcome not in {"generation", "no_hits", "threshold"}:
        raise _invalid(qid, "expected_outcome must be a supported exact outcome")
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
    contract = _CASE_CONTRACTS.get(qid)
    actual_contract = (
        case_type,
        answerable,
        tuple(_source_key(source) for source in sources),
        required_routes,
        prohibited_routes,
        expected_outcome,
    )
    expected_contract = (
        contract.case_type,
        contract.answerable,
        contract.source_keys,
        contract.required_routes,
        contract.prohibited_routes,
        contract.expected_outcome,
    ) if contract else None
    if actual_contract != expected_contract:
        raise _invalid(qid, "does not match its canonical contract")
    return SeverancePolicyCase(
        qid=qid,
        question=question,
        case_type=case_type,
        answerable=answerable,
        sources=sources,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        expected_outcome=expected_outcome,
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
        raise _invalid("dataset", "qids must be severance-policy-001 through -030")
    if not _POSITIVE_STYLES <= {
        tag for case in cases[:15] for tag in case.style_tags
    }:
        raise _invalid("dataset", "positive style coverage is incomplete")
    if not _COLLISION_STYLES <= {
        tag for case in cases[15:] for tag in case.style_tags
    }:
        raise _invalid("dataset", "collision style coverage is incomplete")
    return cases


def _validated_source_ranks(
    value: object, *, qid: str, allowed: tuple[str, ...]
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{qid}: source_ranks must be a dict")
    if not set(value) <= set(allowed):
        raise ValueError(f"{qid}: source_ranks keys are not canonical")
    normalized: dict[str, int] = {}
    for source, rank in value.items():
        if type(rank) is not int or rank < 1:
            raise ValueError(f"{qid}: source_ranks require positive integer ranks")
        normalized[source] = rank
    return dict(sorted(normalized.items()))


def _case_matches_contract(case: SeverancePolicyCase) -> _CaseContract:
    contract = _CASE_CONTRACTS.get(case.qid)
    if contract is None:
        raise ValueError("case qid is not canonical")
    actual = (
        case.case_type,
        case.answerable,
        tuple(_source_key(source) for source in case.sources),
        case.required_routes,
        case.prohibited_routes,
        case.expected_outcome,
    )
    expected = (
        contract.case_type,
        contract.answerable,
        contract.source_keys,
        contract.required_routes,
        contract.prohibited_routes,
        contract.expected_outcome,
    )
    if actual != expected:
        raise ValueError(f"{case.qid}: case does not match canonical contract")
    return contract


def build_case_observation(
    case: SeverancePolicyCase,
    *,
    source_ranks: dict[str, int],
    applied_routes: tuple[str, ...],
    top_score: float,
    hit_count: int,
    candidate_count: int,
    route_plan_matched: bool,
    first_stage_retrieval_calls: int,
    reranker_calls: int,
    reranker_scored_pairs: tuple[int, ...],
) -> dict[str, Any]:
    """Return one validated content-free retrieval observation."""

    if not isinstance(case, SeverancePolicyCase):
        raise ValueError("case must be a SeverancePolicyCase")
    contract = _case_matches_contract(case)
    ranks = _validated_source_ranks(
        source_ranks, qid=case.qid, allowed=contract.source_keys
    )
    routes = _routes(applied_routes, field="applied_routes", require_tuple=True)
    if type(hit_count) is not int or not 0 <= hit_count <= _TOP_K_FINAL:
        raise ValueError(
            f"{case.qid}: hit_count must be between zero and {_TOP_K_FINAL}"
        )
    score = _unit_interval(top_score, field="top_score")
    if hit_count == 0 and (ranks or score != 0.0):
        raise ValueError(
            f"{case.qid}: zero-hit observation requires no ranks and zero score"
        )
    if hit_count > 0 and score == 0.0:
        raise ValueError(
            f"{case.qid}: positive hit_count requires a positive score"
        )
    if any(rank > hit_count for rank in ranks.values()):
        raise ValueError(f"{case.qid}: source rank must not exceed hit_count")
    execution = _validated_execution_evidence(
        qid=case.qid,
        routes=routes,
        hit_count=hit_count,
        candidate_count=candidate_count,
        route_plan_matched=route_plan_matched,
        first_stage_retrieval_calls=first_stage_retrieval_calls,
        reranker_calls=reranker_calls,
        reranker_scored_pairs=reranker_scored_pairs,
        require_tuple=True,
    )
    return {
        "qid": case.qid,
        "source_ranks": ranks,
        "applied_routes": list(routes),
        "hit_count": hit_count,
        "top_score": score,
        **execution,
    }


def _validated_execution_evidence(
    *,
    qid: str,
    routes: tuple[str, ...],
    hit_count: int,
    candidate_count: object,
    route_plan_matched: object,
    first_stage_retrieval_calls: object,
    reranker_calls: object,
    reranker_scored_pairs: object,
    require_tuple: bool,
) -> dict[str, Any]:
    if type(candidate_count) is not int or not 0 <= candidate_count <= 20:
        raise ValueError(f"{qid}: candidate_count must be between zero and twenty")
    if hit_count > candidate_count or (candidate_count == 0) != (hit_count == 0):
        raise ValueError(f"{qid}: candidate_count and hit_count disagree")
    if route_plan_matched is not True:
        raise ValueError(f"{qid}: route_plan_matched must be true")
    if first_stage_retrieval_calls != 1 or type(first_stage_retrieval_calls) is not int:
        raise ValueError(f"{qid}: first_stage_retrieval_calls must equal one")
    expected_type = tuple if require_tuple else list
    if not isinstance(reranker_scored_pairs, expected_type):
        raise ValueError(
            f"{qid}: reranker_scored_pairs must be a {expected_type.__name__}"
        )
    pairs = tuple(reranker_scored_pairs)
    expected_calls = 0 if candidate_count == 0 else 2 if routes == _SEVERANCE_ROUTE else 1
    expected_pairs = () if expected_calls == 0 else (candidate_count,) * expected_calls
    if type(reranker_calls) is not int or reranker_calls != expected_calls:
        raise ValueError(f"{qid}: reranker_calls do not match the exact route contract")
    if pairs != expected_pairs:
        raise ValueError(f"{qid}: reranker_scored_pairs do not match candidate_count")
    return {
        "candidate_count": candidate_count,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": expected_calls,
        "reranker_scored_pairs": list(pairs),
    }


def _validated_observations(observations: object) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    if len(observations) != 30:
        raise ValueError("observations must contain thirty rows")
    normalized = []
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or set(row) != _OBSERVATION_FIELDS:
            raise ValueError(f"observation fields are invalid at row {index + 1}")
        qid = row["qid"]
        if qid != EXPECTED_QIDS[index]:
            raise ValueError("observations must contain the exact thirty qids in order")
        contract = _CASE_CONTRACTS[qid]
        normalized.append(
            {
                "qid": qid,
                "source_ranks": _validated_source_ranks(
                    row["source_ranks"], qid=qid, allowed=contract.source_keys
                ),
                "applied_routes": list(
                    _routes(
                        row["applied_routes"],
                        field="applied_routes",
                        require_tuple=False,
                    )
                ),
                "hit_count": row["hit_count"],
                "top_score": _unit_interval(row["top_score"], field="top_score"),
            }
        )
        hit_count = normalized[-1]["hit_count"]
        if type(hit_count) is not int or not 0 <= hit_count <= _TOP_K_FINAL:
            raise ValueError(
                f"{qid}: hit_count must be between zero and {_TOP_K_FINAL}"
            )
        if hit_count == 0 and (
            normalized[-1]["source_ranks"] or normalized[-1]["top_score"] != 0.0
        ):
            raise ValueError(
                f"{qid}: zero-hit observation requires no ranks and zero score"
            )
        if hit_count > 0 and normalized[-1]["top_score"] == 0.0:
            raise ValueError(f"{qid}: positive hit_count requires a positive score")
        if any(
            rank > hit_count for rank in normalized[-1]["source_ranks"].values()
        ):
            raise ValueError(f"{qid}: source rank must not exceed hit_count")
        normalized[-1].update(
            _validated_execution_evidence(
                qid=qid,
                routes=tuple(normalized[-1]["applied_routes"]),
                hit_count=hit_count,
                candidate_count=row["candidate_count"],
                route_plan_matched=row["route_plan_matched"],
                first_stage_retrieval_calls=row["first_stage_retrieval_calls"],
                reranker_calls=row["reranker_calls"],
                reranker_scored_pairs=row["reranker_scored_pairs"],
                require_tuple=False,
            )
        )
    return normalized


def _expected_guard_answerability(label: str, index: int) -> bool:
    return index < (40 if label == "stress" else 30)


def _validated_guard_rows(
    rows: object,
    *,
    label: str,
    published_evidence: bool = False,
) -> list[dict[str, Any]]:
    expected_qids = _STRESS_QIDS if label == "stress" else _FORMAL_QIDS
    expected_routes = _STRESS_ROUTES if label == "stress" else _FORMAL_ROUTES
    expected_fields = (
        _GUARD_EVIDENCE_FIELDS if published_evidence else _GUARD_INPUT_FIELDS
    )
    if not isinstance(rows, list) or len(rows) != len(expected_qids):
        raise ValueError(f"{label} rows must contain the exact committed qids")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"{label} rows have invalid fields at row {index + 1}")
        qid = row["qid"]
        if qid != expected_qids[index]:
            raise ValueError(f"{label} rows must contain the exact committed qids")
        answerable = _boolean(row["answerable"], field="answerable", identity=qid)
        if answerable is not _expected_guard_answerability(label, index):
            raise ValueError(f"{label} answerability is not canonical for {qid}")
        rank = row["rank"]
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError(f"{label} row {qid}: rank must be null or positive")
        if not answerable and rank is not None:
            raise ValueError(f"{label} row {qid}: unanswerable rank must be null")
        hit_count = row["hit_count"]
        if (
            type(hit_count) is not int
            or hit_count < 0
            or hit_count > _TOP_K_FINAL
        ):
            raise ValueError(
                f"{label} row {qid}: hit_count must be between zero and "
                f"{_TOP_K_FINAL}"
            )
        top_score = _unit_interval(row["top_score"], field="top_score")
        if hit_count == 0 and (top_score != 0.0 or rank is not None):
            raise ValueError(
                f"{label} row {qid}: zero-hit rows require zero score and null rank"
            )
        if hit_count > 0 and top_score == 0.0:
            raise ValueError(
                f"{label} row {qid}: positive hit_count requires a positive score"
            )
        if rank is not None and rank > hit_count:
            raise ValueError(f"{label} row {qid}: rank must not exceed hit_count")
        has_hits = hit_count > 0
        if published_evidence:
            if row["has_hits"] is not has_hits:
                raise ValueError(f"{label} row {qid}: has_hits derivation mismatch")
            if row["reranker_enabled"] is not True:
                raise ValueError(
                    f"{label} row {qid}: reranker_enabled must match configuration"
                )
        routes = _routes(
            row["applied_routes"], field="applied_routes", require_tuple=False
        )
        if routes != expected_routes.get(qid, ()):
            raise ValueError(f"{label} route identity is not canonical for {qid}")
        normalized.append(
            {
                "qid": qid,
                "answerable": answerable,
                "rank": rank,
                "hit_count": hit_count,
                "has_hits": has_hits,
                "reranker_enabled": True,
                "top_score": top_score,
                "applied_routes": list(routes),
            }
        )
        normalized[-1].update(
            _validated_execution_evidence(
                qid=qid,
                routes=routes,
                hit_count=hit_count,
                candidate_count=row["candidate_count"],
                route_plan_matched=row["route_plan_matched"],
                first_stage_retrieval_calls=row["first_stage_retrieval_calls"],
                reranker_calls=row["reranker_calls"],
                reranker_scored_pairs=row["reranker_scored_pairs"],
                require_tuple=False,
            )
        )
    if not any(row["top_score"] != round(row["top_score"], 4) for row in normalized):
        raise ValueError(
            f"{label} guard scores cannot be entirely four-decimal values"
        )
    return normalized


def _route_ablation_decision(
    *,
    has_hits: bool,
    routes: list[str],
    score: float,
    candidate: float,
    global_threshold: float,
):
    evaluation_threshold = (
        candidate if tuple(routes) == _SEVERANCE_ROUTE else global_threshold
    )
    return decide_retrieval_refusal(
        has_hits=has_hits,
        reranker_enabled=True,
        applied_routes=tuple(routes),
        top_score=score,
        global_threshold=evaluation_threshold,
    )


def _evaluate_target(
    observations: list[dict[str, Any]],
    *,
    candidate: float,
    global_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_results = []
    for row in observations:
        contract = _CASE_CONTRACTS[row["qid"]]
        source_contract = all(
            row["source_ranks"].get(source, 6) <= 5
            for source in contract.source_keys
        )
        applied_routes = set(row["applied_routes"])
        if contract.case_type == "positive":
            route_contract = tuple(row["applied_routes"]) == _SEVERANCE_ROUTE
        elif row["qid"] == "severance-policy-027":
            route_contract = tuple(row["applied_routes"]) == ()
        else:
            route_contract = set(contract.required_routes) <= applied_routes and not (
                set(contract.prohibited_routes) & applied_routes
            )
        decision = _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        )
        generation_allowed = not decision.refused
        actual_outcome = (
            "generation" if decision.refusal_stage is None else decision.refusal_stage
        )
        outcome_contract = actual_outcome == contract.expected_outcome
        passed = route_contract and outcome_contract and (
            source_contract or contract.case_type == "collision_negative"
        )
        case_results.append(
            {
                "qid": row["qid"],
                "case_type": contract.case_type,
                "answerable": contract.answerable,
                "source_ranks": dict(row["source_ranks"]),
                "applied_routes": list(row["applied_routes"]),
                "hit_count": row["hit_count"],
                "top_score": row["top_score"],
                "candidate_count": row["candidate_count"],
                "route_plan_matched": row["route_plan_matched"],
                "first_stage_retrieval_calls": row["first_stage_retrieval_calls"],
                "reranker_calls": row["reranker_calls"],
                "reranker_scored_pairs": list(row["reranker_scored_pairs"]),
                "effective_threshold": decision.effective_threshold,
                "refused": decision.refused,
                "refusal_stage": decision.refusal_stage,
                "source_contract_passed": source_contract,
                "route_contract_passed": route_contract,
                "expected_outcome": contract.expected_outcome,
                "generation_allowed": generation_allowed,
                "outcome_contract_passed": outcome_contract,
                "passed": passed,
            }
        )
    positives = case_results[:15]
    collisions = case_results[15:]
    passed_cases = sum(row["passed"] for row in case_results)
    summary = {
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
            and row["outcome_contract_passed"]
            for row in collisions
        ),
        "passed": passed_cases == 30,
    }
    return case_results, summary


def _stress_summary(
    rows: list[dict[str, Any]], *, candidate: float, global_threshold: float
) -> dict[str, Any]:
    decisions = [
        _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        ).refused
        for row in rows
    ]
    false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    unanswerable_refusals = sum(
        refused and not row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    return {
        "questions": 60,
        "answerable": 40,
        "unanswerable": 20,
        "direct_false_refusals": false_refusals,
        "direct_unanswerable_refusals": unanswerable_refusals,
        "direct_unanswerable_coverage": unanswerable_refusals / 20,
        "passed": false_refusals == 0 and unanswerable_refusals >= 17,
    }


def _formal_summary(
    rows: list[dict[str, Any]], *, candidate: float, global_threshold: float
) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    decisions = [
        _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        ).refused
        for row in rows
    ]
    false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    hit_at_5 = sum(
        row["rank"] is not None and row["rank"] <= 5 for row in answerable
    ) / 30
    mrr_at_10 = sum(
        1 / row["rank"]
        for row in answerable
        if row["rank"] is not None and row["rank"] <= 10
    ) / 30
    return {
        "questions": 40,
        "answerable": 30,
        "unanswerable": 10,
        "hit_at_5": hit_at_5,
        "mrr_at_10": mrr_at_10,
        "direct_false_refusals": false_refusals,
        "passed": hit_at_5 >= FORMAL_HIT_AT_5_BASELINE
        and mrr_at_10 >= FORMAL_MRR_AT_10_BASELINE
        and false_refusals == 0,
    }


def evaluate_route_ablation_candidate(
    observations: list[dict[str, Any]],
    *,
    candidate_threshold: float,
    global_threshold: float,
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute target and guard gates through the shared policy."""

    candidate = _unit_interval(candidate_threshold, field="candidate_threshold")
    if candidate not in CANDIDATE_THRESHOLDS:
        raise ValueError("candidate_threshold must belong to the committed grid")
    global_value = _validated_global_threshold(global_threshold)
    target_evidence = _validated_observations(observations)
    stress_evidence = _validated_guard_rows(stress_rows, label="stress")
    formal_evidence = _validated_guard_rows(formal_rows, label="formal")
    cases, target = _evaluate_target(
        target_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    stress = _stress_summary(
        stress_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    formal = _formal_summary(
        formal_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    passed = target["passed"] and stress["passed"] and formal["passed"]
    return {
        "candidate_threshold": candidate,
        "global_threshold": global_value,
        "target": target,
        "stress": stress,
        "formal": formal,
        "cases": cases,
        "stress_evidence": stress_evidence,
        "formal_evidence": formal_evidence,
        "passed": passed,
    }


# Compatibility for callers predating the v0.3.6 retrieval-coverage pivot.
evaluate_candidate = evaluate_route_ablation_candidate


def _observations_from_case_results(cases: object) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) != 30:
        raise ValueError("candidate cases must contain thirty rows")
    observations = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != _CASE_RESULT_FIELDS:
            raise ValueError(f"candidate case fields are invalid at row {index + 1}")
        observations.append(
            {
                "qid": row["qid"],
                "source_ranks": row["source_ranks"],
                "applied_routes": row["applied_routes"],
                "hit_count": row["hit_count"],
                "top_score": row["top_score"],
                "candidate_count": row["candidate_count"],
                "route_plan_matched": row["route_plan_matched"],
                "first_stage_retrieval_calls": row["first_stage_retrieval_calls"],
                "reranker_calls": row["reranker_calls"],
                "reranker_scored_pairs": row["reranker_scored_pairs"],
            }
        )
    return _validated_observations(observations)


def _recompute_candidate(result: object) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != _CANDIDATE_FIELDS:
        raise ValueError("candidate result fields are invalid")
    candidate = _unit_interval(
        result["candidate_threshold"], field="candidate_threshold"
    )
    global_threshold = _validated_global_threshold(result["global_threshold"])
    observations = _observations_from_case_results(result["cases"])
    stress_evidence = _validated_guard_rows(
        result["stress_evidence"], label="stress", published_evidence=True
    )
    formal_evidence = _validated_guard_rows(
        result["formal_evidence"], label="formal", published_evidence=True
    )
    cases, target = _evaluate_target(
        observations,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    stress = _stress_summary(
        stress_evidence,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    formal = _formal_summary(
        formal_evidence,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    passed = target["passed"] and stress["passed"] and formal["passed"]
    if result["cases"] != cases:
        raise ValueError(f"candidate {candidate}: case decision mismatch")
    if result["target"] != target:
        raise ValueError(f"candidate {candidate}: target aggregate mismatch")
    if result["stress"] != stress:
        raise ValueError(f"candidate {candidate}: stress aggregate mismatch")
    if result["formal"] != formal:
        raise ValueError(f"candidate {candidate}: formal aggregate mismatch")
    if type(result["passed"]) is not bool or result["passed"] is not passed:
        raise ValueError(f"candidate {candidate}: complete gate mismatch")
    return {
        "candidate_threshold": candidate,
        "global_threshold": global_threshold,
        "target": target,
        "stress": stress,
        "formal": formal,
        "cases": cases,
        "stress_evidence": stress_evidence,
        "formal_evidence": formal_evidence,
        "passed": passed,
    }


def _validated_candidate_results(
    candidate_results: object,
) -> list[dict[str, Any]]:
    if not isinstance(candidate_results, list):
        raise ValueError("candidate_results must be a list")
    results = [_recompute_candidate(result) for result in candidate_results]
    thresholds = [result["candidate_threshold"] for result in results]
    if len(thresholds) != len(set(thresholds)) or set(thresholds) != set(
        CANDIDATE_THRESHOLDS
    ):
        raise ValueError("candidate grid must equal the committed seven thresholds")
    first = results[0]
    first_observations = _observations_from_case_results(first["cases"])
    for result in results[1:]:
        if result["global_threshold"] != first["global_threshold"]:
            raise ValueError("candidate global threshold must be identical across grid")
        if _observations_from_case_results(result["cases"]) != first_observations:
            raise ValueError("candidate target evidence must be identical across grid")
        if result["stress_evidence"] != first["stress_evidence"]:
            raise ValueError("candidate stress evidence must be identical across grid")
        if result["formal_evidence"] != first["formal_evidence"]:
            raise ValueError("candidate formal evidence must be identical across grid")
    return sorted(results, key=lambda result: result["candidate_threshold"])


def _selected_threshold(results: list[dict[str, Any]]) -> float:
    passing = [result["candidate_threshold"] for result in results if result["passed"]]
    if not passing:
        raise RuntimeError("no candidate threshold satisfies the complete gate set")
    return max(passing)


def select_highest_passing_threshold(
    candidate_results: list[dict[str, Any]],
) -> float:
    """Return the greatest candidate whose complete gate set passes."""

    return _selected_threshold(_validated_candidate_results(candidate_results))


def _hex(value: object, *, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase {length}-character hex value")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_repo_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a canonical POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} must be a canonical POSIX path")
    if "/".join(value.split("/")) != value:
        raise ValueError(f"{field} must be a canonical POSIX path")
    return value


def _validated_bound_entry(value: object, *, field: str) -> dict[str, str]:
    expected_fields = {"path", "mode", "object_type", "blob_oid", "sha256"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field} fields are invalid")
    path = _canonical_repo_path(value["path"], field=f"{field}.path")
    if value["mode"] not in {"100644", "100755"}:
        raise ValueError(f"{field} must bind a regular Git mode")
    if value["object_type"] != "blob":
        raise ValueError(f"{field} must bind a Git blob")
    return {
        "path": path,
        "mode": value["mode"],
        "object_type": "blob",
        "blob_oid": _hex(value["blob_oid"], field=f"{field}.blob_oid", length=40),
        "sha256": _hex(value["sha256"], field=f"{field}.sha256", length=64),
    }


def _require_unique_bound_paths(entries: list[dict[str, str]], *, field: str) -> None:
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} contains a duplicate path")
    casefolded = [path.casefold() for path in paths]
    if len(casefolded) != len(set(casefolded)):
        raise ValueError(f"{field} contains a case-fold path collision")


def _validated_revision_binding(
    value: object, *, code_revision: str
) -> dict[str, Any]:
    fields = {"format_version", "revision", "tracked_files", "declared_inputs"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("revision_binding fields are invalid")
    if value["format_version"] != "1":
        raise ValueError("revision_binding format_version must equal 1")
    revision = _hex(value["revision"], field="revision_binding.revision", length=40)
    if revision != code_revision:
        raise ValueError("revision_binding revision must equal code_revision")
    if not isinstance(value["tracked_files"], list):
        raise ValueError("revision_binding tracked_files must be a list")
    tracked = [
        _validated_bound_entry(entry, field="revision_binding.tracked_file")
        for entry in value["tracked_files"]
    ]
    _require_unique_bound_paths(tracked, field="revision_binding tracked_files")
    if tracked != sorted(tracked, key=lambda entry: entry["path"]):
        raise ValueError("revision_binding tracked_files must use canonical order")
    tracked_paths = {entry["path"] for entry in tracked}
    if not REQUIRED_TRACKED_CODE_PATHS <= tracked_paths:
        raise ValueError("revision_binding is missing a required tracked-code path")
    if any(
        path not in REQUIRED_TRACKED_CODE_PATHS
        and Path(path).suffix.casefold() != ".py"
        for path in tracked_paths
    ):
        raise ValueError("revision_binding contains an extra non-code binding")

    if not isinstance(value["declared_inputs"], list):
        raise ValueError("revision_binding declared_inputs must be a list")
    declared: list[dict[str, str]] = []
    for raw_entry in value["declared_inputs"]:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "label",
            "path",
            "mode",
            "object_type",
            "blob_oid",
            "sha256",
        }:
            raise ValueError("revision_binding declared input fields are invalid")
        label = raw_entry["label"]
        if not isinstance(label, str) or not label:
            raise ValueError("revision_binding declared input label is invalid")
        declared.append(
            {
                "label": label,
                **_validated_bound_entry(
                    {key: nested for key, nested in raw_entry.items() if key != "label"},
                    field=f"revision_binding declared input {label}",
                ),
            }
        )
    _require_unique_bound_paths(declared, field="revision_binding declared_inputs")
    if declared != sorted(declared, key=lambda entry: entry["label"]):
        raise ValueError("revision_binding declared_inputs must use canonical order")
    if {entry["label"]: entry["path"] for entry in declared} != DECLARED_INPUT_PATHS:
        raise ValueError("revision_binding declared_inputs must equal approved inputs")
    return {
        "format_version": "1",
        "revision": revision,
        "tracked_files": tracked,
        "declared_inputs": declared,
    }


def _safe_binding_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a non-blank string")
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError(f"{field} must not contain an absolute path")
    return value


def _validated_package_inventory(value: object, *, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    normalized: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"name", "version"}:
            raise ValueError(f"{field} entry fields are invalid")
        name = _safe_binding_string(entry["name"], field=f"{field}.name")
        if re.sub(r"[-_.]+", "-", name).lower() != name or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name
        ):
            raise ValueError(f"{field} names must be PEP 503 normalized")
        version = _safe_binding_string(entry["version"], field=f"{field}.version")
        normalized.append({"name": name, "version": version})
    if normalized != sorted(normalized, key=lambda entry: entry["name"]):
        raise ValueError(f"{field} must use canonical name order")
    names = [entry["name"] for entry in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains a duplicate normalized name")
    return normalized


def _validated_runtime_import_layout(
    value: object,
    *,
    selected_packages: list[dict[str, str]],
    installed_distributions: list[dict[str, str]],
    markers: dict[str, str],
) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("environment_binding runtime import layout must be a list of strings")
    is_windows = (
        markers["os_name"] == "nt"
        and markers["sys_platform"] == "win32"
        and markers["platform_system"] == "Windows"
    )
    selected_names = {entry["name"] for entry in selected_packages}
    installed_names = {entry["name"] for entry in installed_distributions}
    expected = (
        list(_WINDOWS_PYWIN32_RUNTIME_LAYOUT)
        if is_windows
        and "pywin32" in selected_names
        and "pywin32" in installed_names
        else []
    )
    if value != expected:
        raise ValueError(
            "environment_binding runtime import layout does not match "
            "the selected inventory and platform markers"
        )
    return expected


def _validated_environment_binding(value: object) -> dict[str, Any]:
    fields = {
        "format_version",
        "interpreter",
        "pyvenv",
        "site_layout",
        "runtime_import_layout",
        "lock_selection",
        "installed_distributions",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("environment_binding fields are invalid")
    if value["format_version"] != "1":
        raise ValueError("environment_binding format_version must equal 1")
    interpreter_fields = {
        "implementation",
        "full_version",
        "abi",
        "os_name",
        "sys_platform",
        "platform_system",
        "platform_machine",
        "executable_layout",
    }
    interpreter = value["interpreter"]
    if not isinstance(interpreter, dict) or set(interpreter) != interpreter_fields:
        raise ValueError("environment_binding interpreter fields are invalid")
    normalized_interpreter = {
        field: _safe_binding_string(
            interpreter[field], field=f"environment_binding.interpreter.{field}"
        )
        for field in sorted(interpreter_fields)
    }
    if normalized_interpreter["implementation"] != "cpython":
        raise ValueError("environment_binding interpreter must be CPython")
    if not re.fullmatch(r"\d+\.\d+\.\d+", normalized_interpreter["full_version"]):
        raise ValueError("environment_binding full Python version is invalid")
    if normalized_interpreter["os_name"] == "nt":
        expected_executable = "Scripts/python.exe"
        expected_sites = ["Lib/site-packages"]
    elif normalized_interpreter["os_name"] == "posix":
        major, minor, _patch = normalized_interpreter["full_version"].split(".")
        expected_executable = "bin/python"
        expected_sites = [f"lib/python{major}.{minor}/site-packages"]
    else:
        raise ValueError("environment_binding os_name is invalid")
    if normalized_interpreter["executable_layout"] != expected_executable:
        raise ValueError("environment_binding executable layout is invalid")
    if value["pyvenv"] != {"include_system_site_packages": False}:
        raise ValueError("environment_binding pyvenv contract is invalid")
    if value["site_layout"] != expected_sites:
        raise ValueError("environment_binding site layout is invalid")

    lock_fields = {
        "lock_sha256",
        "offline",
        "frozen",
        "no_dev",
        "selected_dependency_groups",
        "excluded_dependency_groups",
        "markers",
        "active_resolution_markers",
        "selected_packages",
    }
    lock = value["lock_selection"]
    if not isinstance(lock, dict) or set(lock) != lock_fields:
        raise ValueError("environment_binding lock_selection fields are invalid")
    for flag in ("offline", "frozen", "no_dev"):
        if lock[flag] is not True:
            raise ValueError(f"environment_binding lock_selection {flag} must be true")
    if lock["selected_dependency_groups"] != []:
        raise ValueError("environment_binding must select no dependency groups")
    excluded_groups = lock["excluded_dependency_groups"]
    if (
        not isinstance(excluded_groups, list)
        or excluded_groups != sorted(set(excluded_groups))
        or not all(isinstance(group, str) and group for group in excluded_groups)
    ):
        raise ValueError("environment_binding excluded dependency groups are invalid")
    marker_fields = {
        "implementation_name",
        "os_name",
        "platform_machine",
        "platform_python_implementation",
        "platform_system",
        "python_full_version",
        "python_version",
        "sys_platform",
    }
    markers = lock["markers"]
    if not isinstance(markers, dict) or set(markers) != marker_fields:
        raise ValueError("environment_binding marker fields are invalid")
    normalized_markers = {
        field: _safe_binding_string(
            markers[field], field=f"environment_binding.marker.{field}"
        )
        for field in sorted(marker_fields)
    }
    active_markers = lock["active_resolution_markers"]
    if (
        not isinstance(active_markers, list)
        or active_markers != sorted(set(active_markers))
        or not all(isinstance(marker, str) and marker for marker in active_markers)
    ):
        raise ValueError("environment_binding active resolution markers are invalid")
    selected_packages = _validated_package_inventory(
        lock["selected_packages"], field="environment_binding selected_packages"
    )
    installed = _validated_package_inventory(
        value["installed_distributions"],
        field="environment_binding installed_distributions",
    )
    if installed != selected_packages:
        raise ValueError("environment_binding installed inventory must equal selected lock")
    runtime_import_layout = _validated_runtime_import_layout(
        value["runtime_import_layout"],
        selected_packages=selected_packages,
        installed_distributions=installed,
        markers=normalized_markers,
    )
    return {
        "format_version": "1",
        "interpreter": normalized_interpreter,
        "pyvenv": {"include_system_site_packages": False},
        "site_layout": list(expected_sites),
        "runtime_import_layout": runtime_import_layout,
        "lock_selection": {
            "lock_sha256": _hex(
                lock["lock_sha256"], field="environment_binding.lock_sha256", length=64
            ),
            "offline": True,
            "frozen": True,
            "no_dev": True,
            "selected_dependency_groups": [],
            "excluded_dependency_groups": list(excluded_groups),
            "markers": normalized_markers,
            "active_resolution_markers": list(active_markers),
            "selected_packages": selected_packages,
        },
        "installed_distributions": installed,
    }


def _validated_provenance(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError(f"provenance fields must equal {sorted(_PROVENANCE_FIELDS)}")
    source_hashes = provenance["source_artifact_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != _SOURCE_ARTIFACT_FIELDS:
        raise ValueError("source_artifact_sha256 fields are invalid")
    normalized_hashes = {
        field: _hex(source_hashes[field], field=field, length=64)
        for field in sorted(_SOURCE_ARTIFACT_FIELDS)
    }
    code_revision = _hex(
        provenance["code_revision"], field="code_revision", length=40
    )
    revision_binding = _validated_revision_binding(
        provenance["revision_binding"], code_revision=code_revision
    )
    environment_binding = _validated_environment_binding(
        provenance["environment_binding"]
    )
    declared_hashes = {
        entry["label"]: entry["sha256"]
        for entry in revision_binding["declared_inputs"]
    }
    if declared_hashes != {
        "corpus_snapshot": provenance["corpus_snapshot_sha256"],
        "formal_dataset": normalized_hashes["formal_dataset"],
        "stress_dataset": normalized_hashes["stress_dataset"],
        "target_dataset": provenance["dataset_sha256"],
    }:
        raise ValueError("revision_binding declared input hashes are inconsistent")
    tracked_hashes = {
        entry["path"]: entry["sha256"]
        for entry in revision_binding["tracked_files"]
    }
    if environment_binding["lock_selection"]["lock_sha256"] != tracked_hashes["uv.lock"]:
        raise ValueError("environment_binding lock hash is inconsistent with revision_binding")
    exact_strings = {
        "embedding_model": _EMBEDDING_MODEL,
        "embedding_revision": _EMBEDDING_REVISION,
        "reranker_model": _RERANKER_MODEL,
        "reranker_revision": _RERANKER_REVISION,
    }
    for field, expected in exact_strings.items():
        if provenance[field] != expected:
            raise ValueError(f"{field} must equal the approved pinned value")
    configuration = provenance["retrieval_configuration"]
    if not isinstance(configuration, dict) or set(configuration) != set(
        _RETRIEVAL_CONFIGURATION
    ):
        raise ValueError("retrieval_configuration fields are invalid")
    if configuration != _RETRIEVAL_CONFIGURATION:
        raise ValueError("retrieval_configuration must equal the approved primitives")
    if provenance["run_origin"] != _RUN_ORIGIN:
        raise ValueError("run_origin must equal fresh_offline_retrieval")
    execution_device = provenance["execution_device"]
    if execution_device != "cpu":
        raise ValueError("execution_device must equal cpu for authoritative evidence")
    if provenance["precision_mode"] != _PRECISION_MODE:
        raise ValueError("precision_mode must equal fp32")
    if provenance["local_files_only"] is not True:
        raise ValueError("local_files_only must be true")
    if provenance["semantic_view_sha256"] != _SEMANTIC_VIEW_SHA256:
        raise ValueError("semantic_view_sha256 must equal the approved view hash")
    if provenance["merge_policy_version"] != _MERGE_POLICY_VERSION:
        raise ValueError("merge_policy_version must equal the approved version")
    if provenance["primary_score_semantics"] != _PRIMARY_SCORE_SEMANTICS:
        raise ValueError("primary_score_semantics must bind full-precision PRIMARY scores")
    if provenance["source_tree_clean"] is not True:
        raise ValueError("source_tree_clean must be true")
    for field in ("provider_adapters", "provider_requests"):
        if type(provenance[field]) is not int or provenance[field] != 0:
            raise ValueError(f"{field} must be zero")
    return {
        "dataset_sha256": _hex(
            provenance["dataset_sha256"], field="dataset_sha256", length=64
        ),
        "corpus_snapshot_sha256": _hex(
            provenance["corpus_snapshot_sha256"],
            field="corpus_snapshot_sha256",
            length=64,
        ),
        "source_artifact_sha256": normalized_hashes,
        "revision_binding": revision_binding,
        "environment_binding": environment_binding,
        **exact_strings,
        "retrieval_configuration": dict(_RETRIEVAL_CONFIGURATION),
        "execution_device": execution_device,
        "precision_mode": _PRECISION_MODE,
        "local_files_only": True,
        "semantic_view_sha256": _SEMANTIC_VIEW_SHA256,
        "merge_policy_version": _MERGE_POLICY_VERSION,
        "primary_score_semantics": _PRIMARY_SCORE_SEMANTICS,
        "source_tree_clean": True,
        "code_revision": code_revision,
        "run_origin": _RUN_ORIGIN,
        "provider_adapters": 0,
        "provider_requests": 0,
    }


_PUBLIC_KEYS = {
    "schema_version",
    "provenance",
    "candidate_thresholds",
    "production_threshold",
    "route_ablation",
    "highest_passing_candidate",
    "guard_evidence",
    "guard_evidence_binding_sha256",
    "target_evidence_binding_sha256",
    "candidates",
    "cases",
    "evidence_class",
    "outcome",
    "official_export_allowed",
    "target_observations",
    "failed_gates",
    "gates",
    "candidate_threshold",
    "target",
    "stress",
    "formal",
    "passed",
    "total",
    "passed_cases",
    "positive_routes",
    "positive_sources_at_5",
    "positive_generation_allowed",
    "collision_contracts",
    "questions",
    "answerable",
    "unanswerable",
    "direct_false_refusals",
    "direct_unanswerable_refusals",
    "direct_unanswerable_coverage",
    "hit_at_5",
    "mrr_at_10",
    "qid",
    "case_type",
    "rank",
    "hit_count",
    "has_hits",
    "reranker_enabled",
    "source_ranks",
    "applied_routes",
    "top_score",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
    "effective_threshold",
    "refused",
    "refusal_stage",
    "source_contract_passed",
    "route_contract_passed",
    "expected_outcome",
    "generation_allowed",
    "outcome_contract_passed",
    *_PROVENANCE_FIELDS,
    *_SOURCE_ARTIFACT_FIELDS,
    *_RETRIEVAL_CONFIGURATION,
    *_CANONICAL_SOURCE_KEYS,
}
_PUBLIC_STRINGS = {
    _SCHEMA_VERSION,
    "non_release_pivot_no_go",
    "no_go",
    "target",
    "stress",
    "formal",
    "route_ablation",
    "positive",
    "collision_negative",
    "threshold",
    "no_hits",
    "generation",
    "structure",
    "hybrid",
    _EMBEDDING_MODEL,
    _EMBEDDING_REVISION,
    _RERANKER_MODEL,
    _RERANKER_REVISION,
    _RUN_ORIGIN,
    _PRECISION_MODE,
    _MERGE_POLICY_VERSION,
    _PRIMARY_SCORE_SEMANTICS,
    "cpu",
    *EXPECTED_QIDS,
    *_STRESS_QIDS,
    *_FORMAL_QIDS,
    *_KNOWN_ROUTES,
}


def _validate_public_tree(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str) or key not in _PUBLIC_KEYS:
                raise ValueError("public artifact contains a non-allowlisted key")
            if key not in {"revision_binding", "environment_binding"}:
                _validate_public_tree(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_public_tree(nested)
    elif isinstance(value, str):
        is_hash = len(value) in {40, 64} and all(
            character in "0123456789abcdef" for character in value
        )
        if value not in _PUBLIC_STRINGS and not is_hash:
            raise ValueError("public artifact contains a non-allowlisted string value")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("public artifact contains a non-finite number")
    elif value is not None and type(value) not in {bool, int, float}:
        raise ValueError("public artifact contains an unsupported value")


def build_official_artifact(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict content-free official schema."""

    target_evidence = _validated_observations(observations)
    candidates = _validated_candidate_results(candidate_results)
    selected_threshold = _selected_threshold(candidates)
    selected = next(
        result
        for result in candidates
        if result["candidate_threshold"] == selected_threshold
    )
    if _observations_from_case_results(selected["cases"]) != target_evidence:
        raise ValueError("selected candidate cases do not match target observations")
    normalized_provenance = _validated_provenance(provenance)
    guard_evidence = {
        label: [
            {**row, "applied_routes": list(row["applied_routes"])}
            for row in selected[f"{label}_evidence"]
        ]
        for label in ("stress", "formal")
    }
    guard_evidence_binding = _canonical_sha256(
        {
            "guard_evidence": guard_evidence,
            "provenance": normalized_provenance,
        }
    )
    public_cases = [
        {
            **row,
            "source_ranks": dict(row["source_ranks"]),
            "applied_routes": list(row["applied_routes"]),
        }
        for row in selected["cases"]
    ]
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
    target_evidence_binding = _canonical_sha256(
        {
            "cases": public_cases,
            "provenance": normalized_provenance,
        }
    )
    artifact = {
        "schema_version": _SCHEMA_VERSION,
        "provenance": normalized_provenance,
        "production_threshold": selected["global_threshold"],
        "route_ablation": {
            "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
            "highest_passing_candidate": selected_threshold,
            "candidates": public_candidates,
        },
        "guard_evidence": guard_evidence,
        "guard_evidence_binding_sha256": guard_evidence_binding,
        "target_evidence_binding_sha256": target_evidence_binding,
        "cases": public_cases,
    }
    _validate_public_tree(artifact)
    return artifact


_OFFICIAL_FIELDS = {
    "schema_version",
    "provenance",
    "production_threshold",
    "route_ablation",
    "guard_evidence",
    "guard_evidence_binding_sha256",
    "target_evidence_binding_sha256",
    "cases",
}
_ROUTE_ABLATION_FIELDS = {
    "candidate_thresholds",
    "highest_passing_candidate",
    "candidates",
}


def replay_official_artifact(artifact: object) -> dict[str, Any]:
    """Recompute schema 1.3 acceptance without retrieval or model execution."""

    if not isinstance(artifact, dict) or set(artifact) != _OFFICIAL_FIELDS:
        raise ValueError("official artifact fields are invalid")
    if artifact["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("schema_version must equal 1.3")
    if artifact["production_threshold"] != _GLOBAL_THRESHOLD:
        raise ValueError("production_threshold must equal 0.03")
    production_threshold = _validated_global_threshold(artifact["production_threshold"])
    route_ablation = artifact["route_ablation"]
    if (
        not isinstance(route_ablation, dict)
        or set(route_ablation) != _ROUTE_ABLATION_FIELDS
    ):
        raise ValueError("route ablation fields are invalid")
    if route_ablation["candidate_thresholds"] != list(CANDIDATE_THRESHOLDS):
        raise ValueError("route ablation candidate grid mismatch")
    if route_ablation["highest_passing_candidate"] != _GLOBAL_THRESHOLD:
        raise ValueError("route ablation highest passing candidate must equal 0.03")
    provenance = _validated_provenance(artifact["provenance"])
    observations = _observations_from_case_results(artifact["cases"])
    guard_evidence = artifact["guard_evidence"]
    if not isinstance(guard_evidence, dict) or set(guard_evidence) != {
        "stress",
        "formal",
    }:
        raise ValueError("official guard evidence fields are invalid")
    stress = _validated_guard_rows(
        guard_evidence["stress"], label="stress", published_evidence=True
    )
    formal = _validated_guard_rows(
        guard_evidence["formal"], label="formal", published_evidence=True
    )
    candidates = [
        evaluate_route_ablation_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=production_threshold,
            stress_rows=_guard_inputs(stress),
            formal_rows=_guard_inputs(formal),
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    rebuilt = build_official_artifact(
        observations=observations,
        candidate_results=candidates,
        provenance=provenance,
    )
    if artifact != rebuilt:
        raise ValueError("official artifact replay mismatch")
    return rebuilt


_NO_GO_FIELDS = {
    "schema_version",
    "evidence_class",
    "outcome",
    "official_export_allowed",
    "provenance",
    "production_threshold",
    "route_ablation",
    "target_observations",
    "guard_evidence",
    "failed_gates",
}


def _guard_inputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in _GUARD_INPUT_FIELDS
        }
        for row in rows
    ]


def _candidate_summaries(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_threshold": result["candidate_threshold"],
            "target": dict(result["target"]),
            "stress": dict(result["stress"]),
            "formal": dict(result["formal"]),
            "passed": result["passed"],
        }
        for result in candidates
    ]


def build_no_go_evidence(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic, replayable evidence for a non-release NO-GO."""

    target_evidence = _validated_observations(observations)
    candidates = _validated_candidate_results(candidate_results)
    if _observations_from_case_results(candidates[0]["cases"]) != target_evidence:
        raise ValueError("candidate target evidence does not match observations")
    passing = [
        result["candidate_threshold"] for result in candidates if result["passed"]
    ]
    selected = max(passing) if passing else None
    if selected == _GLOBAL_THRESHOLD:
        raise ValueError("passing 0.03 requires the official artifact")
    normalized_provenance = _validated_provenance(provenance)
    guard_evidence = {
        label: [dict(row) for row in candidates[0][f"{label}_evidence"]]
        for label in ("stress", "formal")
    }
    failed_gates = []
    for result in candidates:
        gates = [
            label
            for label in ("target", "stress", "formal")
            if not result[label]["passed"]
        ]
        if result["candidate_threshold"] == _GLOBAL_THRESHOLD:
            gates.append("route_ablation")
        if gates:
            failed_gates.append(
                {
                    "candidate_threshold": result["candidate_threshold"],
                    "gates": gates,
                }
            )
    envelope = {
        "schema_version": _SCHEMA_VERSION,
        "evidence_class": "non_release_pivot_no_go",
        "outcome": "no_go",
        "official_export_allowed": False,
        "provenance": normalized_provenance,
        "production_threshold": candidates[0]["global_threshold"],
        "route_ablation": {
            "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
            "highest_passing_candidate": selected,
            "candidates": _candidate_summaries(candidates),
        },
        "target_observations": target_evidence,
        "guard_evidence": guard_evidence,
        "failed_gates": failed_gates,
    }
    _validate_public_tree(envelope)
    return envelope


def replay_no_go_evidence(envelope: object) -> dict[str, Any]:
    """Recompute a NO-GO envelope without retrieval or model construction."""

    if not isinstance(envelope, dict) or set(envelope) != _NO_GO_FIELDS:
        raise ValueError("NO-GO evidence fields are invalid")
    provenance = _validated_provenance(envelope["provenance"])
    observations = _validated_observations(envelope["target_observations"])
    guard_evidence = envelope["guard_evidence"]
    if not isinstance(guard_evidence, dict) or set(guard_evidence) != {
        "stress",
        "formal",
    }:
        raise ValueError("NO-GO guard evidence fields are invalid")
    stress = _validated_guard_rows(
        guard_evidence["stress"], label="stress", published_evidence=True
    )
    formal = _validated_guard_rows(
        guard_evidence["formal"], label="formal", published_evidence=True
    )
    candidates = [
        evaluate_route_ablation_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=envelope["production_threshold"],
            stress_rows=_guard_inputs(stress),
            formal_rows=_guard_inputs(formal),
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    rebuilt = build_no_go_evidence(
        observations=observations,
        candidate_results=candidates,
        provenance=provenance,
    )
    if envelope != rebuilt:
        raise ValueError("NO-GO evidence replay mismatch")
    return rebuilt
