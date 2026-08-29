"""Run the v0.3.4 wage-arrears regression against an isolated local index."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import lib
from run_reliability_eval import (
    _build_indexes,
    _materialize_audited_corpus,
    _write_jsonl,
)

from rag.config import PROJECT_ROOT, Settings
from rag.factory import build_retrieval_pipeline
from rag.retrieval.reranker import Reranker
from rag.wage_arrears_regression import (
    build_public_result,
    load_regression_dataset,
    require_cached_models,
    route_expansion_applied,
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "eval" / "dataset" / "wage_arrears_regression_v0.3.4.jsonl"
)
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release" / "corpus_snapshot.json"
OFFICIAL_RESULT = (
    PROJECT_ROOT / "eval" / "official" / "wage_arrears_regression_v0.3.4.json"
)
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip()


def _write_public_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_cases(pipeline, rows: list[dict]) -> list[dict]:
    raw_cases = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result = pipeline.run(row["question"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        rank = lib.match_rank(result.hits, row["sources"]) if row["expect_expansion"] else None
        raw_cases.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "expected_expansion": row["expect_expansion"],
                "expansion_applied": route_expansion_applied(row["question"]),
                "rank": rank,
                "top_score": result.top_score,
                "elapsed_ms": elapsed_ms,
                "hits": [
                    {
                        "chunk_id": hit.payload.get("chunk_id", ""),
                        "citation": hit.citation,
                        "score": hit.score,
                    }
                    for hit in result.hits
                ],
            }
        )
        print(f"[query] {index}/{len(rows)} {row['qid']}", flush=True)
    return raw_cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--export-official", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    rows = load_regression_dataset(args.dataset)
    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        device=args.device,
        chunking_strategy="structure",
        retrieval_mode="hybrid",
        use_reranker=True,
    )
    require_cached_models(settings)

    run_dir = args.work_dir or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-wage-arrears"
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"work directory must be absent or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    committed_snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    corpus_dir = _materialize_audited_corpus(run_dir, committed_snapshot)

    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        qdrant_path=str(run_dir / "qdrant"),
        storage_dir=run_dir / "storage",
        data_dir=run_dir / "data",
        collection_name="wage_arrears_regression",
        device=args.device,
        chunking_strategy="structure",
        retrieval_mode="hybrid",
        use_reranker=True,
    )
    embedder, store = _build_indexes(settings, corpus_dir)
    try:
        reranker = Reranker(
            model_name=settings.reranker_model,
            model_revision=settings.reranker_model_revision,
            device=settings.device,
        )
        pipeline = build_retrieval_pipeline(
            settings,
            embedder,
            store,
            strategy="structure",
            mode="hybrid",
            use_reranker=True,
            reranker=reranker,
        )
        raw_cases = _run_cases(pipeline, rows)
        _write_jsonl(run_dir / "raw_cases.jsonl", raw_cases)
        result = build_public_result(
            dataset_path=args.dataset,
            code_revision=_git_revision(),
            configuration={
                "chunking": "structure",
                "retrieval": "hybrid",
                "reranker": True,
                "top_k_retrieve": settings.top_k_retrieve,
                "top_k_final": settings.top_k_final,
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_model_revision,
                "reranker_model": settings.reranker_model,
                "reranker_revision": settings.reranker_model_revision,
            },
            cases=raw_cases,
        )
        _write_public_json(run_dir / "results.json", result)
        if args.export_official and result["summary"]["passed"]:
            _write_public_json(OFFICIAL_RESULT, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        print(f"[done] raw artifacts: {run_dir}", flush=True)
        return 0 if result["summary"]["passed"] else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
