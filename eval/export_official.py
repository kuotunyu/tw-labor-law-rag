"""Export deterministic, de-identified portfolio artifacts from local eval runs.

Raw runs stay under ``eval/runs/`` (gitignored) because they contain local
paths, verbose model output, and debugging logs.  This exporter validates the
selected complete runs and writes a compact, reviewable evidence set under
``eval/official/`` without making any model or network calls.

Usage:
    python eval/export_official.py \
        --ablation-run eval/runs/<timestamp>-ablation \
        --e2e-run eval/runs/<timestamp>-e2e
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from rag.evaluation import (
        canonical_text_sha256,
        compute_e2e_metrics,
        infer_refusal_stage,
    )
except ModuleNotFoundError:  # direct ``python eval/export_official.py`` execution
    import _bootstrap  # noqa: F401

    from rag.evaluation import (
        canonical_text_sha256,
        compute_e2e_metrics,
        infer_refusal_stage,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "dataset" / "eval_set.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "official"
STRATEGIES = ("structure", "fixed")
CONFIGS = ("bm25", "vector", "hybrid", "hybrid+rerank")
SETTINGS_ALLOWLIST = (
    "embedding_model",
    "reranker_model",
    "retrieval_mode",
    "use_reranker",
    "top_k_retrieve",
    "top_k_final",
    "rrf_k",
    "rerank_score_threshold",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _allowed_settings(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in SETTINGS_ALLOWLIST if key in raw}


def _dataset_metadata(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = _read_jsonl(path)
    qids = [row["qid"] for row in dataset]
    if len(qids) != len(set(qids)):
        raise ValueError("evaluation dataset contains duplicate qids")
    digest = canonical_text_sha256(path)
    metadata = {
        "path": "eval/dataset/eval_set.jsonl",
        "sha256": digest,
        "n_questions": len(dataset),
        "n_answerable": sum(1 for row in dataset if row["answerable"]),
        "n_unanswerable": sum(1 for row in dataset if not row["answerable"]),
    }
    return metadata, dataset


def _assert_close(label: str, actual: float, expected: float, *, abs_tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=abs_tol):
        raise ValueError(f"{label} mismatch: recomputed={actual!r}, raw={expected!r}")


def _validate_qids(
    label: str, rows: list[dict[str, Any]], dataset: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    order = {row["qid"]: index for index, row in enumerate(dataset)}
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"{label} contains duplicate qids")
    if set(qids) != set(order):
        missing = sorted(set(order) - set(qids))
        extra = sorted(set(qids) - set(order))
        raise ValueError(f"{label} qids differ from dataset: missing={missing}, extra={extra}")
    answerable = {row["qid"]: row["answerable"] for row in dataset}
    for row in rows:
        if row["answerable"] != answerable[row["qid"]]:
            raise ValueError(f"{label} answerable flag differs for {row['qid']}")
    return sorted(rows, key=lambda row: order[row["qid"]])


def _build_ablation_artifacts(
    run_dir: Path, dataset_meta: dict[str, Any], dataset: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _read_json(run_dir / "results.json")
    if raw.get("n_questions") != dataset_meta["n_questions"]:
        raise ValueError("ablation results question count differs from dataset")

    result_rows = []
    trace_rows = []
    for strategy in STRATEGIES:
        for config in CONFIGS:
            raw_key = f"{strategy}/{config}"
            filename = f"trace_{strategy}_{config.replace('+', '-')}.jsonl"
            rows = _validate_qids(filename, _read_jsonl(run_dir / filename), dataset)
            mode, reranker = ("hybrid", True) if config == "hybrid+rerank" else (config, False)

            sanitized = [
                {
                    "qid": row["qid"],
                    "chunking": strategy,
                    "retrieval": mode,
                    "reranker": reranker,
                    "answerable": row["answerable"],
                    "rank": row["rank"],
                    "top_score": row["top_score"],
                    "elapsed_ms": row["elapsed_ms"],
                }
                for row in rows
            ]
            trace_rows.extend(sanitized)

            answerable_rows = [row for row in sanitized if row["answerable"]]
            n_answerable = len(answerable_rows)
            hit_at_5 = sum(
                1 for row in answerable_rows if row["rank"] and row["rank"] <= 5
            ) / n_answerable
            mrr_at_10 = sum(
                1 / row["rank"]
                for row in answerable_rows
                if row["rank"] and row["rank"] <= 10
            ) / n_answerable
            avg_latency_ms = sum(row["elapsed_ms"] for row in answerable_rows) / n_answerable

            raw_metrics = raw["metrics"][raw_key]
            _assert_close(f"{raw_key} hit@5", hit_at_5, raw_metrics["hit_rate@5"])
            _assert_close(f"{raw_key} mrr@10", mrr_at_10, raw_metrics["mrr@10"])
            # Traces round each latency to 0.1 ms; results use pre-rounded values.
            _assert_close(
                f"{raw_key} latency", avg_latency_ms, raw["avg_latency_ms"][raw_key], abs_tol=0.1
            )
            raw_unanswerable_scores = raw_metrics["unanswerable_top1_scores"]
            trace_unanswerable_scores = [
                row["top_score"] for row in sanitized if not row["answerable"]
            ]
            if trace_unanswerable_scores != raw_unanswerable_scores:
                raise ValueError(f"{raw_key} unanswerable score list differs from results.json")

            result_rows.append(
                {
                    "chunking": strategy,
                    "retrieval": mode,
                    "reranker": reranker,
                    "n_answerable": n_answerable,
                    "hit_at_5": raw_metrics["hit_rate@5"],
                    "mrr_at_10": raw_metrics["mrr@10"],
                    "avg_latency_ms": raw["avg_latency_ms"][raw_key],
                }
            )

    result = {
        "schema_version": "1.0",
        "artifact_type": "retrieval_ablation",
        "evaluation_date": raw["timestamp"].split("T", 1)[0],
        "dataset": dataset_meta,
        "settings": _allowed_settings(raw.get("settings", {})),
        "results": result_rows,
    }
    return result, trace_rows


def _build_e2e_artifacts(
    run_dir: Path, dataset_meta: dict[str, Any], dataset: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_results = _read_json(run_dir / "results.json")
    raw_traces = _validate_qids("e2e trace", _read_jsonl(run_dir / "trace.jsonl"), dataset)
    threshold = float(raw_results["settings"]["rerank_score_threshold"])

    traces = []
    for row in raw_traces:
        stage = infer_refusal_stage(row, threshold)
        judge = row.get("judge")
        traces.append(
            {
                "qid": row["qid"],
                "q_type": row["q_type"],
                "answerable": row["answerable"],
                "refused": row["refused"],
                "refusal_stage": stage,
                "top_score": row["top_score"],
                "elapsed_ms": row["elapsed_ms"],
                "cited_sources": row["cited_sources"],
                "judge": (
                    {
                        "faithfulness": judge["faithfulness"],
                        "relevancy": judge["relevancy"],
                    }
                    if judge
                    else None
                ),
            }
        )

    # Omit null judge values before aggregation because the runtime trace only
    # adds the key when a judge call actually happened.
    metric_rows = [
        {key: value for key, value in row.items() if not (key == "judge" and value is None)}
        for row in traces
    ]
    metrics = compute_e2e_metrics(metric_rows)
    for key, expected in raw_results["metrics"].items():
        actual = metrics[key]
        if isinstance(expected, float):
            _assert_close(f"e2e {key}", actual, expected)
        elif actual != expected:
            raise ValueError(f"e2e {key} mismatch: recomputed={actual!r}, raw={expected!r}")

    result = {
        "schema_version": "1.0",
        "artifact_type": "end_to_end",
        "evaluation_date": raw_results["timestamp"].split("T", 1)[0],
        "dataset": dataset_meta,
        "config": raw_results["config"],
        "generator": raw_results["generator"],
        "judge": raw_results["judge"],
        "settings": _allowed_settings(raw_results.get("settings", {})),
        "metrics": metrics,
    }
    return result, traces


def build_official_artifacts(
    ablation_run: Path, e2e_run: Path, dataset_path: Path = DEFAULT_DATASET
) -> dict[str, str]:
    """Validate the selected raw runs and return deterministic artifact text."""

    dataset_meta, dataset = _dataset_metadata(dataset_path)
    ablation_results, ablation_trace = _build_ablation_artifacts(
        ablation_run, dataset_meta, dataset
    )
    e2e_results, e2e_trace = _build_e2e_artifacts(e2e_run, dataset_meta, dataset)
    return {
        "ablation_results.json": _json_text(ablation_results),
        "ablation_trace.jsonl": _jsonl_text(ablation_trace),
        "e2e_results.json": _json_text(e2e_results),
        "e2e_trace.jsonl": _jsonl_text(e2e_trace),
    }


def _write_or_check(artifacts: dict[str, str], output_dir: Path, check: bool) -> None:
    mismatches = []
    for filename, content in artifacts.items():
        destination = output_dir / filename
        if check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != content:
                mismatches.append(filename)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    if mismatches:
        raise SystemExit(f"official artifacts are stale or missing: {', '.join(mismatches)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-run", type=Path, required=True)
    parser.add_argument("--e2e-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="fail if committed artifacts differ")
    args = parser.parse_args()

    artifacts = build_official_artifacts(args.ablation_run, args.e2e_run, args.dataset)
    _write_or_check(artifacts, args.output_dir, args.check)
    action = "verified" if args.check else "wrote"
    print(f"{action} {len(artifacts)} official artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
