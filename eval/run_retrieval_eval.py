"""Retrieval evaluation: hit_rate@K and MRR@K over an eval dataset.

Compares one or more retrieval configs (e.g. ``vector``, ``hybrid``,
``hybrid+rerank``) across one or both chunking strategies. A retrieved chunk
counts as a hit when its document matches a gold source and the gold article
is among the articles the chunk covers. Unanswerable questions are excluded
from hit_rate/MRR but their top score is recorded — this is what calibrates
``rerank_score_threshold`` in config.py.

Every run writes ``eval/runs/<timestamp>-retrieval/`` with a config snapshot
and per-question traces, so failure cases can be analysed after the fact.

Usage:
    python eval/run_retrieval_eval.py --configs vector,hybrid,hybrid+rerank --strategy all
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


def snapshot_settings(settings) -> dict:
    data = settings.model_dump(mode="json")
    return {k: v for k, v in data.items() if "api_key" not in k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "eval" / "dataset" / "mini_eval.jsonl"
    )
    parser.add_argument(
        "--configs",
        default="vector,hybrid,hybrid+rerank",
        help="comma-separated: vector | bm25 | hybrid, each optionally suffixed with +rerank",
    )
    parser.add_argument("--strategy", choices=["structure", "fixed", "all"], default="all")
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--mrr-k", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    strategies = ["structure", "fixed"] if args.strategy == "all" else [args.strategy]
    configs = args.configs.split(",")
    rows = lib.load_dataset(args.dataset)

    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    store = VectorStore(settings)
    # Shared across every config/strategy that needs it — loading bge-reranker-v2-m3 is expensive.
    needs_reranker = any(lib.parse_config(c)[1] for c in configs)
    reranker = Reranker(model_name=settings.reranker_model, device=settings.device) if needs_reranker else None

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-retrieval"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_questions": len(rows),
        "settings": snapshot_settings(settings),
        "runs": {},
    }

    for strategy in strategies:
        for config in configs:
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
            results["runs"][key] = metrics

            trace_name = f"trace_{strategy}_{config.replace('+', '-')}.jsonl"
            with open(run_dir / trace_name, "w", encoding="utf-8") as f:
                for trace in traces:
                    f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\n=== retrieval eval ({args.dataset.name}) ===")
    header = f"{'strategy/config':<26}{'hit@' + str(args.hit_k):<10}{'MRR@' + str(args.mrr_k):<10}{'n':<4}"
    print(header)
    for key, m in results["runs"].items():
        print(
            f"{key:<26}"
            f"{m[f'hit_rate@{args.hit_k}']:<10.3f}"
            f"{m[f'mrr@{args.mrr_k}']:<10.3f}"
            f"{m['n_answerable']:<4}"
        )
    print(f"\nrun artifacts -> {run_dir}")

    print("\nunanswerable top-score (for refusal threshold calibration):")
    for key, m in results["runs"].items():
        print(f"  {key}: {m['unanswerable_top1_scores']}")

    store.close()


if __name__ == "__main__":
    main()
