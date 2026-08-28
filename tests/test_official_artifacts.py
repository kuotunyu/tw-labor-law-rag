import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rag.evaluation import canonical_text_sha256, compute_e2e_metrics
from rag.reliability import PUBLIC_TRACE_FIELDS, compute_reliability_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT_ROOT / "eval" / "official"
DATASET = PROJECT_ROOT / "eval" / "dataset" / "eval_set.jsonl"
STRESS_DATASET = PROJECT_ROOT / "eval" / "dataset" / "reliability_stress_v0.3.1.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_official_artifacts_match_public_dataset():
    dataset = read_jsonl(DATASET)
    expected_qids = {row["qid"] for row in dataset}
    expected_hash = canonical_text_sha256(DATASET)

    ablation = json.loads((OFFICIAL / "ablation_results.json").read_text(encoding="utf-8"))
    e2e = json.loads((OFFICIAL / "e2e_results.json").read_text(encoding="utf-8"))
    assert ablation["dataset"]["sha256"] == expected_hash
    assert e2e["dataset"]["sha256"] == expected_hash

    ablation_trace = read_jsonl(OFFICIAL / "ablation_trace.jsonl")
    assert len(ablation_trace) == 8 * len(dataset)
    configs = {
        (row["chunking"], row["retrieval"], row["reranker"])
        for row in ablation_trace
    }
    assert len(configs) == 8
    for config in configs:
        assert {row["qid"] for row in ablation_trace if (
            row["chunking"], row["retrieval"], row["reranker"]
        ) == config} == expected_qids

    e2e_trace = read_jsonl(OFFICIAL / "e2e_trace.jsonl")
    assert {row["qid"] for row in e2e_trace} == expected_qids
    assert len(e2e_trace) == len(dataset)


def test_official_e2e_metrics_recompute_from_trace():
    result = json.loads((OFFICIAL / "e2e_results.json").read_text(encoding="utf-8"))
    traces = read_jsonl(OFFICIAL / "e2e_trace.jsonl")
    metric_rows = [
        {key: value for key, value in row.items() if not (key == "judge" and value is None)}
        for row in traces
    ]
    recomputed = compute_e2e_metrics(metric_rows)
    assert recomputed == result["metrics"]
    assert recomputed["direct_false_refusal_rate"] == 0.0
    assert recomputed["direct_unanswerable_coverage"] == 0.9
    assert recomputed["n_answers_with_citations"] == 28
    assert recomputed["citation_parse_coverage"] == pytest.approx(28 / 29)
    assert recomputed["uncited_answers"] == ["eval-26"]
    assert recomputed["refusal_by_stage"]["threshold"]["count"] == 9
    assert recomputed["refusal_by_stage"]["llm"] == {
        "count": 2,
        "answerable_qids": ["eval-10"],
        "unanswerable_qids": ["eval-32"],
    }


def test_official_ablation_metrics_recompute_from_trace():
    result = json.loads((OFFICIAL / "ablation_results.json").read_text(encoding="utf-8"))
    traces = read_jsonl(OFFICIAL / "ablation_trace.jsonl")

    for summary in result["results"]:
        rows = [
            row
            for row in traces
            if row["chunking"] == summary["chunking"]
            and row["retrieval"] == summary["retrieval"]
            and row["reranker"] == summary["reranker"]
            and row["answerable"]
        ]
        assert len(rows) == summary["n_answerable"]
        assert sum(row["rank"] is not None and row["rank"] <= 5 for row in rows) / len(
            rows
        ) == pytest.approx(summary["hit_at_5"])
        assert sum(
            1 / row["rank"]
            for row in rows
            if row["rank"] is not None and row["rank"] <= 10
        ) / len(rows) == pytest.approx(summary["mrr_at_10"])
        assert sum(row["elapsed_ms"] for row in rows) / len(rows) == pytest.approx(
            summary["avg_latency_ms"], abs=0.1
        )


def test_official_reliability_metrics_recompute_from_privacy_reduced_trace():
    result = json.loads((OFFICIAL / "reliability_results.json").read_text(encoding="utf-8"))
    traces = read_jsonl(OFFICIAL / "reliability_trace.jsonl")
    dataset = read_jsonl(STRESS_DATASET)

    assert len(traces) == 60
    assert {row["qid"] for row in traces} == {row["qid"] for row in dataset}
    assert all(tuple(row) == PUBLIC_TRACE_FIELDS for row in traces)
    recomputed = compute_reliability_metrics(traces, result["threshold_candidates"])
    assert recomputed == result["stress_metrics"]
    assert recomputed["hit_at_5"] == 0.95
    assert recomputed["mrr_at_10"] == pytest.approx(0.9083333333333334)
    assert recomputed["threshold_sweep"]["0.03"]["direct_false_refusals"] == 1
    assert recomputed["threshold_sweep"]["0.03"]["direct_unanswerable_coverage"] == 0.85
    assert result["formal_guard_metrics"]["hit_at_5"] == pytest.approx(29 / 30)
    assert result["decision"] == {
        "pareto_better_candidates": [],
        "outcome": "retain_0.03",
        "automatic_config_change": False,
    }


def test_official_artifacts_have_no_local_paths_or_secret_fields():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in OFFICIAL.iterdir()
        if path.is_file()
    )
    assert not re.search(r"[A-Za-z]:[\\/]", combined)
    assert "/Users/" not in combined
    assert "/home/" not in combined
    assert not re.search(r'(?i)api[_-]?key|password|bearer\\s', combined)


@pytest.mark.parametrize(
    "filename",
    ["ablation_results.json", "e2e_results.json", "reliability_results.json"],
)
def test_official_json_ends_with_newline(filename):
    assert (OFFICIAL / filename).read_bytes().endswith(b"\n")


def test_direct_exporter_bootstrap_can_hash_the_dataset():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    code = (
        "import runpy, sys; "
        "from pathlib import Path; "
        "sys.path.insert(0, 'eval'); "
        "namespace = runpy.run_path('eval/export_official.py'); "
        "metadata, _ = namespace['_dataset_metadata'](Path('eval/dataset/eval_set.jsonl')); "
        f"assert metadata['sha256'] == {canonical_text_sha256(DATASET)!r}"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stdout + process.stderr
