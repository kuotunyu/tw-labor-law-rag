import hashlib

import pytest

import rag.evaluation as evaluation
from rag.evaluation import compute_e2e_metrics, infer_refusal_stage


def test_canonical_text_sha256_ignores_checkout_line_endings(tmp_path):
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"qid":"a"}\n')
    crlf.write_bytes(b'{"qid":"a"}\r\n')
    expected = hashlib.sha256(b'{"qid":"a"}\n').hexdigest()

    assert evaluation.canonical_text_sha256(lf) == expected
    assert evaluation.canonical_text_sha256(crlf) == expected


def test_infer_refusal_stage_for_legacy_traces():
    assert infer_refusal_stage({"refused": False}, threshold=0.03) is None
    assert (
        infer_refusal_stage(
            {"refused": True, "retrieved": [], "top_score": 0.0}, threshold=0.03
        )
        == "no_hits"
    )
    assert (
        infer_refusal_stage(
            {"refused": True, "retrieved": [{}], "top_score": 0.01}, threshold=0.03
        )
        == "threshold"
    )
    assert (
        infer_refusal_stage(
            {"refused": True, "retrieved": [{}], "top_score": 0.9}, threshold=0.03
        )
        == "llm"
    )


def test_infer_refusal_stage_rejects_inconsistent_explicit_data():
    with pytest.raises(ValueError, match="disagree"):
        infer_refusal_stage(
            {"refused": False, "refusal_stage": "threshold"}, threshold=0.03
        )

    with pytest.raises(ValueError, match="ambiguous"):
        infer_refusal_stage(
            {"refused": True, "retrieved": [{}], "top_score": 0.03}, threshold=0.03
        )


def test_compute_e2e_metrics_separates_refusal_layers():
    traces = [
        {
            "qid": "answer-ok",
            "answerable": True,
            "refused": False,
            "refusal_stage": None,
            "cited_sources": [{"doc": "法規", "article": "第 1 條"}],
            "judge": {"faithfulness": 5, "relevancy": 4},
        },
        {
            "qid": "answer-llm-refusal",
            "answerable": True,
            "refused": True,
            "refusal_stage": "llm",
        },
        {
            "qid": "unknown-threshold",
            "answerable": False,
            "refused": True,
            "refusal_stage": "threshold",
        },
        {
            "qid": "unknown-llm",
            "answerable": False,
            "refused": True,
            "refusal_stage": "llm",
        },
    ]

    metrics = compute_e2e_metrics(traces)

    assert metrics["false_refusals"] == ["answer-llm-refusal"]
    assert metrics["n_judged"] == 1
    assert metrics["n_answered"] == 1
    assert metrics["n_generation_calls"] == 3
    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["direct_false_refusal_rate"] == 0.0
    assert metrics["direct_unanswerable_coverage"] == 0.5
    assert metrics["avg_faithfulness"] == 5.0
    assert metrics["avg_relevancy"] == 4.0
    assert metrics["n_answers_with_citations"] == 1
    assert metrics["citation_parse_coverage"] == 1.0
    assert metrics["uncited_answers"] == []
    assert metrics["refusal_by_stage"]["threshold"] == {
        "count": 1,
        "answerable_qids": [],
        "unanswerable_qids": ["unknown-threshold"],
    }
    assert metrics["refusal_by_stage"]["llm"] == {
        "count": 2,
        "answerable_qids": ["answer-llm-refusal"],
        "unanswerable_qids": ["unknown-llm"],
    }


def test_compute_e2e_metrics_requires_explicit_stage_and_tracks_uncited_answers():
    with pytest.raises(ValueError, match="explicit refusal_stage"):
        compute_e2e_metrics(
            [{"qid": "legacy", "answerable": True, "refused": False}]
        )

    metrics = compute_e2e_metrics(
        [
            {
                "qid": "cited",
                "answerable": True,
                "refused": False,
                "refusal_stage": None,
                "cited_sources": [{"doc": "法規", "article": "第 1 條"}],
            },
            {
                "qid": "uncited",
                "answerable": True,
                "refused": False,
                "refusal_stage": None,
                "cited_sources": [],
            },
        ]
    )
    assert metrics["n_answered"] == 2
    assert metrics["n_answers_with_citations"] == 1
    assert metrics["citation_parse_coverage"] == 0.5
    assert metrics["uncited_answers"] == ["uncited"]
