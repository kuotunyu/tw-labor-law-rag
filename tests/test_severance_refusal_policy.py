import json
import math
from pathlib import Path

import pytest

from rag.retrieval.pipeline import plan_retrieval_query
from rag.severance_refusal_policy import (
    CANDIDATE_THRESHOLDS,
    EXPECTED_QIDS,
    FORMAL_HIT_AT_5_BASELINE,
    FORMAL_MRR_AT_10_BASELINE,
    build_case_observation,
    build_official_artifact,
    evaluate_candidate,
    load_cases,
    select_highest_passing_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "eval/dataset/severance_refusal_policy_v0.3.6.jsonl"
POSITIVE_STYLES = {
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
COLLISION_STYLES = {
    "single_regime",
    "ordinary_termination",
    "notice_only",
    "wage_arrears",
    "generic_retirement",
    "unrelated_old_new",
    "partial_cue_collision",
}
SEVERANCE_SOURCES = (
    {"law": "勞工退休金條例", "article": "第 12 條"},
    {"law": "勞動基準法", "article": "第 17 條"},
)
OBSERVATION_FIELDS = {
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
OFFICIAL_CASE_FIELDS = {
    "qid",
    "case_type",
    "answerable",
    "source_ranks",
    "applied_routes",
    "top_score",
    "effective_threshold",
    "refused",
    "refusal_stage",
    "source_contract_passed",
    "route_contract_passed",
    "generation_expected",
    "generation_allowed",
    "generation_contract_passed",
    "passed",
}
FORMAL_RANKS = [
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    None,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


@pytest.fixture(scope="module")
def cases():
    return load_cases(DATASET)


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _observations(cases):
    observations = []
    for index, case in enumerate(cases):
        source_ranks = {
            _source_key(source): rank
            for rank, source in enumerate(case.sources, start=1)
        }
        score = 0.015 if index == 0 else 0.5
        if not case.expect_generation:
            score = 0.0
        observations.append(
            build_case_observation(
                case,
                source_ranks=source_ranks,
                applied_routes=case.required_routes,
                top_score=score,
            )
        )
    return observations


def _stress_rows():
    answerable = [
        {
            "qid": f"stress-{number:03d}",
            "answerable": True,
            "rank": 1,
            "top_score": 0.0175 if number == 1 else 0.5,
            "applied_routes": ["severance_comparison"] if number == 1 else [],
        }
        for number in range(1, 41)
    ]
    unanswerable = [
        {
            "qid": f"stress-{number:03d}",
            "answerable": False,
            "rank": None,
            "top_score": 0.0 if number <= 57 else 0.5,
            "applied_routes": [],
        }
        for number in range(41, 61)
    ]
    return answerable + unanswerable


def _formal_rows():
    answerable = [
        {
            "qid": f"eval-{number:02d}",
            "answerable": True,
            "rank": rank,
            "top_score": 0.5,
            "applied_routes": [],
        }
        for number, rank in enumerate(FORMAL_RANKS, start=1)
    ]
    unanswerable = [
        {
            "qid": f"eval-{number:02d}",
            "answerable": False,
            "rank": None,
            "top_score": 0.0 if number <= 39 else 0.5,
            "applied_routes": [],
        }
        for number in range(31, 41)
    ]
    return answerable + unanswerable


def _candidate_results(observations):
    return [
        evaluate_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=0.03,
            stress_rows=_stress_rows(),
            formal_rows=_formal_rows(),
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]


def _provenance():
    return {
        "dataset_sha256": "a" * 64,
        "corpus_snapshot_sha256": "b" * 64,
        "source_artifact_sha256": {
            "reliability_results": "c" * 64,
            "reliability_trace": "d" * 64,
            "reliability_formal_trace": "e" * 64,
        },
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "1" * 40,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "2" * 40,
        "retrieval_configuration": {
            "chunking": "structure",
            "retrieval": "hybrid",
            "reranker": True,
            "top_k_retrieve": 20,
            "top_k_final": 5,
        },
        "code_revision": "f" * 40,
        "provider_adapters": 0,
        "provider_requests": 0,
    }


def test_reviewed_dataset_has_exact_order_contracts_and_style_coverage(cases):
    assert tuple(case.qid for case in cases) == EXPECTED_QIDS
    assert [case.case_type for case in cases] == ["positive"] * 15 + [
        "collision_negative"
    ] * 15
    positives = cases[:15]
    collisions = cases[15:]
    assert all(case.answerable for case in positives)
    assert all(case.sources == SEVERANCE_SOURCES for case in positives)
    assert all(case.required_routes == ("severance_comparison",) for case in positives)
    assert all(not case.prohibited_routes for case in positives)
    assert all(case.expect_generation for case in positives)
    assert POSITIVE_STYLES <= {
        tag for case in positives for tag in case.style_tags
    }
    assert COLLISION_STYLES <= {
        tag for case in collisions for tag in case.style_tags
    }
    assert all(len(case.question) >= 12 and case.question[-1] in "？?" for case in cases)


def test_reviewed_questions_match_committed_route_semantics(cases):
    for case in cases:
        routes = plan_retrieval_query(case.question).routes
        if case.case_type == "positive":
            assert routes == ("severance_comparison",), case.qid
        else:
            assert routes != ("severance_comparison",), case.qid
            assert all(route in routes for route in case.required_routes), case.qid
            assert all(route not in routes for route in case.prohibited_routes), case.qid


def _dataset_rows():
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update(extra="drift"), "fields"),
        (lambda rows: rows[0].pop("question"), "fields"),
        (lambda rows: rows[1].update(qid=rows[0]["qid"]), "duplicate qids"),
        (lambda rows: rows[0].update(question=" "), "question"),
        (
            lambda rows: rows[0].update(sources=[rows[0]["sources"][0]] * 2),
            "duplicate source",
        ),
        (
            lambda rows: rows[0].update(
                required_routes=["severance_comparison"] * 2
            ),
            "duplicates",
        ),
        (lambda rows: rows[0].update(answerable=1), "answerable"),
        (lambda rows: rows[0].update(expect_generation=1), "expect_generation"),
        (lambda rows: rows[0].update(case_type="collision_negative"), "ordering"),
        (lambda rows: rows.pop(), "qids"),
    ],
)
def test_loader_fails_closed_on_dataset_drift(tmp_path, mutate, message):
    rows = _dataset_rows()
    mutate(rows)
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_observation_is_content_free_and_scores_real_contracts(cases):
    observation = build_case_observation(
        cases[0],
        source_ranks={
            "勞工退休金條例|第 12 條": 1,
            "勞動基準法|第 17 條": 5,
        },
        applied_routes=("severance_comparison",),
        top_score=0.0150004,
    )

    assert set(observation) == OBSERVATION_FIELDS
    assert observation["source_contract_passed"] is True
    assert observation["route_contract_passed"] is True
    assert observation["top_score"] == 0.0150004
    assert "question" not in observation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_ranks": {"未知法規|第 1 條": 1}}, "source_ranks"),
        (
            {
                "source_ranks": {
                    "勞工退休金條例|第 12 條": True,
                    "勞動基準法|第 17 條": 2,
                }
            },
            "source_ranks",
        ),
        ({"applied_routes": ["severance_comparison"]}, "applied_routes"),
        (
            {
                "applied_routes": (
                    "severance_comparison",
                    "severance_comparison",
                )
            },
            "duplicates",
        ),
        ({"top_score": math.nan}, "top_score"),
        ({"top_score": math.inf}, "top_score"),
    ],
)
def test_observation_rejects_invalid_content_free_inputs(cases, kwargs, message):
    valid = {
        "source_ranks": {
            "勞工退休金條例|第 12 條": 1,
            "勞動基準法|第 17 條": 2,
        },
        "applied_routes": ("severance_comparison",),
        "top_score": 0.1,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_case_observation(cases[0], **valid)


def test_candidate_recomputes_target_stress_and_formal_gates(cases):
    result = evaluate_candidate(
        _observations(cases),
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    assert result["target"] == {
        "total": 30,
        "passed_cases": 30,
        "positive_routes": 15,
        "positive_sources_at_5": 15,
        "positive_generation_allowed": 15,
        "collision_contracts": 15,
        "passed": True,
    }
    assert result["stress"] == {
        "questions": 60,
        "answerable": 40,
        "unanswerable": 20,
        "direct_false_refusals": 0,
        "direct_unanswerable_refusals": 17,
        "direct_unanswerable_coverage": 0.85,
        "passed": True,
    }
    assert result["formal"]["hit_at_5"] == FORMAL_HIT_AT_5_BASELINE
    assert result["formal"]["mrr_at_10"] == FORMAL_MRR_AT_10_BASELINE
    assert result["formal"]["direct_false_refusals"] == 0
    assert result["formal"]["passed"] is True
    assert result["passed"] is True


def test_candidate_uses_unrounded_score_and_shared_equality_behavior(cases):
    observations = _observations(cases)
    observations[0] = {
        **observations[0],
        "top_score": 0.0149996,
    }

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    first = result["cases"][0]
    assert round(first["top_score"], 6) == 0.015
    assert first["refused"] is True
    assert first["refusal_stage"] == "threshold"
    assert result["target"]["passed_cases"] == 29


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate_nan", "candidate_threshold"),
        ("candidate_outside_grid", "candidate_threshold"),
        ("stress_nonfinite", "top_score"),
        ("formal_wrong_count", "formal rows"),
    ],
)
def test_candidate_rejects_bad_grid_and_guard_inputs(cases, mutation, message):
    threshold = 0.015
    stress = _stress_rows()
    formal = _formal_rows()
    if mutation == "candidate_nan":
        threshold = math.nan
    elif mutation == "candidate_outside_grid":
        threshold = 0.017
    elif mutation == "stress_nonfinite":
        stress[0]["top_score"] = math.inf
    else:
        formal.pop()

    with pytest.raises(ValueError, match=message):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=threshold,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=formal,
        )


def test_selector_sorts_grid_and_returns_highest_complete_pass(cases):
    results = list(reversed(_candidate_results(_observations(cases))))

    assert select_highest_passing_threshold(results) == 0.015


def test_selector_rejects_grid_tampering_and_no_go(cases):
    results = _candidate_results(_observations(cases))
    with pytest.raises(ValueError, match="candidate grid"):
        select_highest_passing_threshold(results[:-1])

    failing = [
        {
            **result,
            "target": {**result["target"], "passed_cases": 29, "passed": False},
            "passed": False,
        }
        for result in results
    ]
    with pytest.raises(RuntimeError, match="no candidate"):
        select_highest_passing_threshold(failing)


def test_selector_rejects_nonfinite_case_decision_inputs(cases):
    results = _candidate_results(_observations(cases))
    results[0]["cases"][0]["top_score"] = math.nan

    with pytest.raises(ValueError, match="top_score"):
        select_highest_passing_threshold(results)


def test_official_artifact_has_strict_content_free_schema_and_rounding(cases):
    observations = _observations(cases)
    observations[0] = {**observations[0], "top_score": 0.0150004}
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=_provenance(),
    )

    assert tuple(artifact) == (
        "schema_version",
        "provenance",
        "candidate_thresholds",
        "selected_threshold",
        "candidates",
        "cases",
    )
    assert artifact["schema_version"] == "1.0"
    assert artifact["candidate_thresholds"] == list(CANDIDATE_THRESHOLDS)
    assert artifact["selected_threshold"] == 0.015
    assert len(artifact["candidates"]) == 7
    assert all("cases" not in result for result in artifact["candidates"])
    assert len(artifact["cases"]) == 30
    assert set(artifact["cases"][0]) == OFFICIAL_CASE_FIELDS
    assert artifact["cases"][0]["top_score"] == 0.015
    assert artifact["cases"][0]["effective_threshold"] == 0.015
    serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
    assert all(case.question not in serialized for case in cases)
    assert not {
        "question",
        "content",
        "answer",
        "endpoint",
        "url",
        "credential",
        "response",
        "local_path",
        "account",
    } & {key.casefold() for key in _all_keys(artifact)}


def _all_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


def test_official_artifact_rejects_content_bearing_rows_and_provenance(cases):
    observations = _observations(cases)
    results = _candidate_results(observations)
    content_bearing = [{**observations[0], "question": cases[0].question}]
    content_bearing.extend(observations[1:])
    with pytest.raises(ValueError, match="observation fields"):
        build_official_artifact(
            observations=content_bearing,
            candidate_results=results,
            provenance=_provenance(),
        )

    provenance = {**_provenance(), "endpoint": "https://example.invalid"}
    with pytest.raises(ValueError, match="provenance fields"):
        build_official_artifact(
            observations=observations,
            candidate_results=results,
            provenance=provenance,
        )
