"""End-to-end evaluation: generation quality (LLM-as-judge) + refusal accuracy.

Runs the full answer pipeline on every question:
  - answerable questions -> judge scores faithfulness / relevancy against the
    retrieved context; also tracks false refusals (refusing an answerable
    question is a miss)
  - unanswerable questions -> refusal accuracy (the system must decline)

By default this evaluates only the primary config (settings' strategy / mode /
reranker): a full e2e run costs ~60 LLM calls, so running it across all six
ablation cells would blow free-tier quotas for little insight — the retrieval
ablation (ablation.py) covers the config comparison at zero LLM cost.

Usage:
    python eval/run_e2e_eval.py [--limit N] [--sleep 2] [--judge-provider gemini]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import lib
from judge import Judge

from rag.config import PROJECT_ROOT, get_settings
from rag.evaluation import compute_e2e_metrics
from rag.factory import build_answerer
from rag.generation.llm import build_llm
from rag.generation.prompts import build_context_block
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore

RUNS_DIR = PROJECT_ROOT / "eval" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "eval" / "dataset" / "eval_set.jsonl"
    )
    parser.add_argument("--strategy", choices=["structure", "fixed"], default=None)
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--sleep", type=float, default=2.0, help="pause between questions (rate-limit politeness)")
    parser.add_argument("--judge-provider", default=None, help="override judge provider (cross-provider check)")
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    settings = get_settings()
    strategy = args.strategy or settings.chunking_strategy
    mode = args.mode or settings.retrieval_mode
    use_reranker = not args.no_rerank

    rows = lib.load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    store = VectorStore(settings)
    answerer = build_answerer(
        settings, embedder, store, strategy=strategy, mode=mode, use_reranker=use_reranker
    )
    judge_llm = build_llm(
        settings,
        provider=args.judge_provider,
        model=args.judge_model or settings.resolved_judge_model,
    )
    judge = Judge(judge_llm)

    config_label = f"{strategy}/{mode}{'+rerank' if use_reranker else ''}"
    print(f"e2e eval: {len(rows)} questions, config={config_label}, "
          f"generator={settings.llm_provider}({settings.resolved_generation_model}), "
          f"judge={args.judge_provider or settings.llm_provider}({judge_llm.model})\n")

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-e2e"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    traces = []
    for i, row in enumerate(rows, start=1):
        t0 = time.perf_counter()
        result = lib.retry_rate_limited(lambda: answerer.answer(row["question"]))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace = {
            "qid": row["qid"],
            "question": row["question"],
            "q_type": row["q_type"],
            "answerable": row["answerable"],
            "gold_answer": row["answer"],
            "gold_sources": row["sources"],
            "refused": result.refused,
            "refusal_stage": result.refusal_stage,
            "answer": result.text,
            "cited_sources": [{"doc": s["doc"], "article": s["article"]} for s in result.sources],
            "top_score": round(result.retrieval.top_score, 4),
            "retrieved": [
                {"citation": h.citation, "score": round(h.score, 4)} for h in result.retrieval.hits
            ],
            "elapsed_ms": elapsed_ms,
        }

        if row["answerable"] and not result.refused:
            context = build_context_block(result.retrieval.hits)
            trace["judge"] = judge.score(row["question"], context, result.text)

        traces.append(trace)
        # Flush incrementally so a mid-run crash (e.g. quota exhaustion) keeps progress.
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        status = "REFUSED" if result.refused else f"answered ({len(result.sources)} cites)"
        judge_str = ""
        if "judge" in trace:
            judge_str = f"  F={trace['judge']['faithfulness']} R={trace['judge']['relevancy']}"
        print(f"[{i:>2}/{len(rows)}] {row['qid']} {status}{judge_str}")
        if args.sleep:
            time.sleep(args.sleep)

    # ── metrics ──────────────────────────────────────
    metrics = compute_e2e_metrics(traces)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "config": config_label,
        "generator": f"{settings.llm_provider}/{settings.resolved_generation_model}",
        "judge": f"{args.judge_provider or settings.llm_provider}/{judge_llm.model}",
        "settings": {
            k: v for k, v in settings.model_dump(mode="json").items() if "api_key" not in k
        },
        "metrics": metrics,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\n=== e2e results ({config_label}) ===")
    print(f"answerable: {metrics['n_answerable']}  "
          f"false_refusal_rate: {metrics['false_refusal_rate']:.3f}  "
          f"faithfulness: {metrics['avg_faithfulness']}  "
          f"relevancy: {metrics['avg_relevancy']}")
    print(f"unanswerable: {metrics['n_unanswerable']}  "
          f"refusal_accuracy: {metrics['refusal_accuracy']}")
    direct_coverage = metrics["direct_unanswerable_coverage"]
    direct_coverage_text = "n/a" if direct_coverage is None else f"{direct_coverage:.3f}"
    print(f"direct refusal layer: unanswerable coverage="
          f"{direct_coverage_text}  "
          f"answerable false-refusal rate={metrics['direct_false_refusal_rate']:.3f}")
    for stage, breakdown in metrics["refusal_by_stage"].items():
        print(f"  {stage}: {breakdown['count']} "
              f"(answerable={breakdown['answerable_qids']}, "
              f"unanswerable={breakdown['unanswerable_qids']})")
    if metrics["false_refusals"]:
        print(f"false refusals: {metrics['false_refusals']}")
    if metrics["missed_refusals"]:
        print(f"missed refusals: {metrics['missed_refusals']}")
    print(f"\nrun artifacts -> {run_dir}")
    store.close()


if __name__ == "__main__":
    main()
