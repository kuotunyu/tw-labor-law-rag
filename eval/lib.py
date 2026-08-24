"""Shared helpers for retrieval evaluation, used by both the Phase 2 quick
comparison (run_retrieval_eval.py) and the Phase 4 ablation study
(ablation.py) — every ablation cell is just one more (strategy, config) pair
evaluated the same way.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag.retrieval.pipeline import RetrievalPipeline


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def retry_rate_limited(fn, *, max_retries: int = 6, backoff_base: float = 15.0):
    """Call ``fn()``, retrying with exponential backoff on 429s.

    Free-tier LLM quotas (e.g. Gemini free tier: 5 requests/min) make 429s a
    fact of life for eval runs; anything else propagates immediately.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limit(exc):
                raise
            last_error = exc
            time.sleep(min(backoff_base * (2**attempt), 120.0))
    raise RuntimeError(f"rate-limited after {max_retries} retries: {last_error}")

# "hybrid+rerank" -> ("hybrid", True); "vector" -> ("vector", False)
def parse_config(config: str) -> tuple[str, bool]:
    mode, _, flag = config.partition("+")
    return mode, flag == "rerank"


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def match_rank(hits, gold_sources: list[dict]) -> int | None:
    """1-based rank of the first hit matching any gold (doc, article); None if absent."""
    for rank, hit in enumerate(hits, start=1):
        doc = hit.payload.get("doc_title", "")
        articles = hit.payload.get("articles", [])
        for src in gold_sources:
            if doc == src["doc"] and src["article"] in articles:
                return rank
    return None


def evaluate_pipeline(
    pipeline: RetrievalPipeline, rows: list[dict], hit_k: int, mrr_k: int
) -> tuple[dict, list[dict]]:
    traces = []
    ranks: list[int | None] = []
    refusal_scores = []

    for row in rows:
        t0 = time.perf_counter()
        result = pipeline.run(row["question"])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        rank = match_rank(result.hits, row["sources"]) if row["answerable"] else None
        if row["answerable"]:
            ranks.append(rank)
        else:
            refusal_scores.append(result.top_score)

        traces.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "answerable": row["answerable"],
                "gold": row["sources"],
                "rank": rank,
                "top_score": round(result.top_score, 4),
                "elapsed_ms": round(elapsed_ms, 1),
                "hits": [
                    {
                        "citation": h.citation,
                        "score": round(h.score, 4),
                        "chunk_id": h.payload.get("chunk_id"),
                    }
                    for h in result.hits
                ],
            }
        )

    n = len(ranks)
    metrics = {
        "n_answerable": n,
        f"hit_rate@{hit_k}": sum(1 for r in ranks if r and r <= hit_k) / n if n else 0.0,
        f"mrr@{mrr_k}": sum(1 / r for r in ranks if r and r <= mrr_k) / n if n else 0.0,
        "unanswerable_top1_scores": [round(s, 4) for s in refusal_scores],
    }
    return metrics, traces
