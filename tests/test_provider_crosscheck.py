"""Provider cross-check evidence must be useful without leaking content."""

import pytest

from rag.provider_crosscheck import (
    PUBLIC_PROVIDER_TRACE_FIELDS,
    compute_provider_metrics,
    privacy_reduced_provider_trace,
    select_crosscheck_rows,
)


def _raw_row(**overrides):
    row = {
        "qid": "stress-001",
        "answerable": True,
        "requested_provider": "gemini",
        "actual_provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "refused": False,
        "citation_count": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "estimated_cost_usd": "0.000080",
        "elapsed_ms": 123.45,
        "question": "private question",
        "answer": "private answer",
        "api_key": "secret",
        "provider_response": {"secret": "payload"},
        "judge_reason": "private reasoning",
    }
    row.update(overrides)
    return row


def test_privacy_reducer_keeps_only_approved_numeric_verdicts() -> None:
    reduced = privacy_reduced_provider_trace(_raw_row())

    assert tuple(reduced) == PUBLIC_PROVIDER_TRACE_FIELDS
    assert reduced == {
        "qid": "stress-001",
        "answerable": True,
        "requested_provider": "gemini",
        "actual_provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "refused": False,
        "citation_count": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "estimated_cost_usd": "0.000080",
        "refusal_verdict": 1,
        "citation_verdict": 1,
        "elapsed_ms": 123.5,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested_provider": "openai", "actual_provider": "gemini"}, "provider"),
        ({"input_tokens": None}, "token"),
        ({"output_tokens": -1}, "token"),
        ({"estimated_cost_usd": "-0.1"}, "cost"),
        ({"citation_count": -1}, "citation"),
        ({"elapsed_ms": float("nan")}, "elapsed"),
    ],
)
def test_privacy_reducer_rejects_untrustworthy_evidence(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        privacy_reduced_provider_trace(_raw_row(**overrides))


def test_metrics_compare_each_provider_without_content() -> None:
    rows = [
        privacy_reduced_provider_trace(_raw_row()),
        privacy_reduced_provider_trace(
            _raw_row(
                qid="stress-042",
                answerable=False,
                refused=True,
                citation_count=0,
                input_tokens=80,
                output_tokens=5,
                estimated_cost_usd="0.000037",
            )
        ),
        privacy_reduced_provider_trace(
            _raw_row(
                requested_provider="openai",
                actual_provider="openai",
                model="gpt-5.6-luna",
                input_tokens=90,
                output_tokens=15,
                estimated_cost_usd="0.000036",
            )
        ),
    ]

    metrics = compute_provider_metrics(rows)

    assert metrics["gemini"] == {
        "requests": 2,
        "refusal_accuracy": 1.0,
        "citation_success_rate": 1.0,
        "input_tokens": 180,
        "output_tokens": 25,
        "estimated_cost_usd": "0.000117",
        "avg_latency_ms": 123.5,
    }
    assert metrics["openai"]["requests"] == 1
    assert metrics["openai"]["estimated_cost_usd"] == "0.000036"


def test_selection_starts_with_three_answerable_and_two_generation_unanswerable() -> None:
    dataset = [
        {"qid": f"stress-{index:03d}", "answerable": index <= 6}
        for index in range(1, 11)
    ]
    reliability = [
        {
            "qid": row["qid"],
            "answerable": row["answerable"],
            "threshold_refused": row["qid"] in {"stress-007", "stress-010"},
        }
        for row in dataset
    ]

    initial, expansion = select_crosscheck_rows(
        dataset,
        reliability,
        initial_count=5,
        maximum_count=8,
    )

    assert [row["qid"] for row in initial] == [
        "stress-001",
        "stress-002",
        "stress-003",
        "stress-008",
        "stress-009",
    ]
    assert [row["qid"] for row in expansion] == [
        "stress-004",
        "stress-005",
        "stress-006",
    ]


def test_selection_rejects_missing_dataset_evidence() -> None:
    with pytest.raises(ValueError, match="qid coverage"):
        select_crosscheck_rows(
            [{"qid": "stress-001", "answerable": True}],
            [],
            initial_count=5,
            maximum_count=5,
        )
