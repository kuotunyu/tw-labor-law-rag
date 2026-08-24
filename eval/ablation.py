"""Retrieval ablation study: every (retrieval config x chunking strategy) cell
on the full eval set, at zero LLM cost.

Grid (8 cells): {bm25, vector, hybrid, hybrid+rerank} x {structure, fixed}.
Outputs a markdown comparison table (for EVAL_REPORT.md) plus per-question
traces for failure analysis.

Usage:
    python eval/ablation.py [--dataset eval/dataset/eval_set.jsonl]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import lib

from rag.config import PROJECT_ROOT, get_settings
from rag.factory import build_retrieval_pipeline
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.retrieval.reranker import Reranker

RUNS_DIR = PROJECT_ROOT / "eval" / "runs"

CONFIGS = ["bm25", "vector", "hybrid", "hybrid+rerank"]
STRATEGIES = ["structure", "fixed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "eval" / "dataset" / "eval_set.jsonl"
    )
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--mrr-k", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    rows = lib.load_dataset(args.dataset)
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    store = VectorStore(settings)
    reranker = Reranker(model_name=settings.reranker_model, device=settings.device)

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-ablation"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    latencies: dict[str, float] = {}
    for strategy in STRATEGIES:
        for config in CONFIGS:
            mode, use_reranker = lib.parse_config(config)
            key = f"{strategy}/{config}"
            pipeline = build_retrieval_pipeline(
                settings,
                embedder,
                store,
                strategy=strategy,
                mode=mode,
                use_reranker=use_reranker,
                reranker=reranker,
            )
            metrics, traces = lib.evaluate_pipeline(pipeline, rows, args.hit_k, args.mrr_k)
            all_metrics[key] = metrics
            answered = [t for t in traces if t["answerable"]]
            latencies[key] = sum(t["elapsed_ms"] for t in answered) / len(answered)

            with open(run_dir / f"trace_{strategy}_{config.replace('+', '-')}.jsonl", "w", encoding="utf-8") as f:
                for trace in traces:
                    f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            print(f"[done] {key}: hit@{args.hit_k}={metrics[f'hit_rate@{args.hit_k}']:.3f} "
                  f"MRR@{args.mrr_k}={metrics[f'mrr@{args.mrr_k}']:.3f}")

    # ── markdown table ───────────────────────────────
    hit_key, mrr_key = f"hit_rate@{args.hit_k}", f"mrr@{args.mrr_k}"
    lines = [
        f"| chunking | 檢索設定 | hit@{args.hit_k} | MRR@{args.mrr_k} | 平均延遲 (ms) |",
        "|---|---|---|---|---|",
    ]
    for strategy in STRATEGIES:
        for config in CONFIGS:
            key = f"{strategy}/{config}"
            m = all_metrics[key]
            lines.append(
                f"| {strategy} | {config} | {m[hit_key]:.3f} | {m[mrr_key]:.3f} | {latencies[key]:.0f} |"
            )
    table = "\n".join(lines)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_questions": len(rows),
        "settings": {
            k: v for k, v in settings.model_dump(mode="json").items() if "api_key" not in k
        },
        "metrics": all_metrics,
        "avg_latency_ms": latencies,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    (run_dir / "ablation.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + table)
    print(f"\nrun artifacts -> {run_dir}")
    store.close()


if __name__ == "__main__":
    main()
