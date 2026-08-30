import math

import pytest

from rag.retrieval.refusal_policy import decide_retrieval_refusal


@pytest.mark.parametrize(
    ("case", "expected_stage", "expected_threshold"),
    [
        ("no hits before other conditions", "no_hits", None),
        ("hits with no reranker", None, None),
        ("severance score below threshold", "threshold", 0.7),
        ("severance score equal to threshold", None, 0.7),
        ("severance score above threshold", None, 0.7),
        ("empty routes use global", "threshold", 0.03),
        ("unknown routes use global", "threshold", 0.03),
        ("duplicate routes use global", "threshold", 0.03),
        ("multiple routes use global", "threshold", 0.03),
    ],
)
def test_decision_table(case, expected_stage, expected_threshold):
    scores = {
        "no hits before other conditions": 0.0,
        "hits with no reranker": 0.0,
        "severance score below threshold": 0.69,
        "severance score equal to threshold": 0.7,
        "severance score above threshold": 0.71,
        "empty routes use global": 0.02,
        "unknown routes use global": 0.02,
        "duplicate routes use global": 0.02,
        "multiple routes use global": 0.02,
    }
    routes = {
        "severance score below threshold": ("severance_comparison",),
        "severance score equal to threshold": ("severance_comparison",),
        "severance score above threshold": ("severance_comparison",),
        "unknown routes use global": ("other",),
        "duplicate routes use global": ("severance_comparison", "severance_comparison"),
        "multiple routes use global": ("severance_comparison", "other"),
    }.get(case, ())
    decision = decide_retrieval_refusal(
        has_hits=case != "no hits before other conditions",
        reranker_enabled=case != "hits with no reranker",
        applied_routes=routes,
        top_score=scores[case],
        global_threshold=0.03,
        severance_comparison_threshold=0.7,
    )
    assert decision.refusal_stage == expected_stage
    assert decision.effective_threshold == expected_threshold


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_score", -0.01),
        ("top_score", 1.01),
        ("top_score", math.nan),
        ("top_score", math.inf),
        ("global_threshold", -0.01),
        ("global_threshold", 1.01),
        ("global_threshold", math.nan),
        ("global_threshold", math.inf),
        ("severance_comparison_threshold", -0.01),
        ("severance_comparison_threshold", 1.01),
        ("severance_comparison_threshold", math.nan),
        ("severance_comparison_threshold", math.inf),
    ],
)
def test_invalid_scores_and_thresholds_are_rejected(field, value):
    kwargs = dict(
        has_hits=True,
        reranker_enabled=True,
        applied_routes=(),
        top_score=0.5,
        global_threshold=0.03,
        severance_comparison_threshold=0.7,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        decide_retrieval_refusal(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"has_hits": 1},
        {"reranker_enabled": 0},
        {"applied_routes": ["severance_comparison"]},
        {"applied_routes": ("",)},
        {"applied_routes": ("  ",)},
        {"applied_routes": (1,)},
    ],
)
def test_input_shapes_are_rejected(kwargs):
    base = dict(
        has_hits=True,
        reranker_enabled=True,
        applied_routes=(),
        top_score=0.5,
        global_threshold=0.03,
        severance_comparison_threshold=0.7,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        decide_retrieval_refusal(**base)
