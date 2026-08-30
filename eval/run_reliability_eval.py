"""Run the 60-question reliability benchmark against an isolated local index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lib  # noqa: E402

from rag.config import Settings  # noqa: E402
from rag.corpus_audit import build_snapshot, compare_snapshots  # noqa: E402
from rag.factory import build_retrieval_pipeline  # noqa: E402
from rag.indexing.bm25_index import BM25Index  # noqa: E402
from rag.indexing.embedder import BGEM3Embedder  # noqa: E402
from rag.indexing.vector_store import VectorStore  # noqa: E402
from rag.ingestion.chunkers import get_chunker  # noqa: E402
from rag.ingestion.loader import load_corpus  # noqa: E402
from rag.reliability import (  # noqa: E402
    compute_reliability_metrics,
    pareto_better_thresholds,
    privacy_reduced_trace,
)
from rag.retrieval.refusal_policy import decide_retrieval_refusal  # noqa: E402
from rag.retrieval.reranker import Reranker  # noqa: E402
from rag.retrieval.retriever import bm25_path_for, collection_for  # noqa: E402
from scripts.audit_corpus import download_bytes  # noqa: E402
from scripts.download_corpus import (  # noqa: E402
    DUMPS,
    iter_laws,
    normalize_name,
    validate_dump_zip,
)

DEFAULT_STRESS_DATASET = PROJECT_ROOT / "eval" / "dataset" / "reliability_stress_v0.3.1.jsonl"
DEFAULT_FORMAL_DATASET = PROJECT_ROOT / "eval" / "dataset" / "eval_set.jsonl"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release" / "corpus_snapshot.json"
OFFICIAL_DIR = PROJECT_ROOT / "eval" / "official"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _materialize_audited_corpus(work_dir: Path, committed: dict) -> Path:
    raw_dir = work_dir / "raw"
    corpus_dir = work_dir / "corpus"
    raw_dir.mkdir(parents=True, exist_ok=False)
    corpus_dir.mkdir(parents=True, exist_ok=False)
    sources = []
    laws = []
    missing = []
    for source_id, (url, targets) in DUMPS.items():
        archive_bytes = download_bytes(url)
        archive_path = raw_dir / f"{source_id}.zip"
        archive_path.write_bytes(archive_bytes)
        validate_dump_zip(archive_path)
        sources.append(
            {"id": source_id, "url": url, "sha256": hashlib.sha256(archive_bytes).hexdigest()}
        )
        wanted = set(targets)
        found = {}
        for law in iter_laws(archive_path):
            name = normalize_name(law["name"])
            if name in wanted and not law["abolished"]:
                if name in found:
                    raise ValueError(f"duplicate target law: {name}")
                law["name"] = name
                found[name] = law
        for name in targets:
            if name not in found:
                missing.append(name)
            else:
                laws.append(found[name])
    if missing:
        raise ValueError(f"missing target laws: {', '.join(missing)}")

    live = build_snapshot(
        sources=sources,
        laws=laws,
        snapshot_date=committed["snapshot_date"],
    )
    changes = compare_snapshots(committed, live)
    if changes or live["sources"] != committed["sources"]:
        raise RuntimeError(
            "official corpus no longer matches the committed audited snapshot: "
            + json.dumps(changes, ensure_ascii=False)
        )
    for law in laws:
        (corpus_dir / f"{law['name']}.json").write_text(
            json.dumps(law, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return corpus_dir


def _build_indexes(
    settings: Settings,
    corpus_dir: Path,
    *,
    local_files_only: bool = False,
) -> tuple[BGEM3Embedder, VectorStore]:
    units = load_corpus(corpus_dir)
    if len({unit.doc_id for unit in units}) != 15:
        raise RuntimeError("isolated corpus must contain exactly 15 laws")
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
        local_files_only=local_files_only,
    )
    store = VectorStore(settings)
    for strategy in ("structure", "fixed"):
        chunks = get_chunker(strategy, settings.chunk_size, settings.chunk_overlap).chunk(units)
        chunks_path = settings.storage_dir / f"chunks_{strategy}.jsonl"
        _write_jsonl(chunks_path, [chunk.payload() for chunk in chunks])
        started = time.perf_counter()
        vectors = embedder.encode([chunk.text for chunk in chunks])
        collection = collection_for(settings, strategy)
        store.recreate_collection(collection, dim=vectors.shape[1])
        store.upsert_chunks(collection, chunks, vectors)
        bm25 = BM25Index.from_payloads([chunk.payload() for chunk in chunks])
        bm25.save(bm25_path_for(settings, strategy))
        print(
            f"[index] {strategy}: {len(chunks)} chunks, {store.count(collection)} points, "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return embedder, store


def _run_rows(pipeline, dataset_rows: list[dict]) -> list[dict]:
    traces = []
    for index, row in enumerate(dataset_rows, start=1):
        started = time.perf_counter()
        result = pipeline.run(row["question"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        rank = lib.match_rank(result.hits, row["sources"]) if row["answerable"] else None
        traces.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "answerable": row["answerable"],
                "rank": rank,
                "top_score": result.top_score,
                "applied_routes": list(result.applied_routes),
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
        print(f"[query] {index}/{len(dataset_rows)} {row['qid']}", flush=True)
    return traces


def _reduce_trace(row: dict, settings: Settings) -> dict:
    decision = decide_retrieval_refusal(
        has_hits=bool(row["hits"]),
        reranker_enabled=True,
        applied_routes=tuple(row["applied_routes"]),
        top_score=row["top_score"],
        global_threshold=settings.rerank_score_threshold,
        severance_comparison_threshold=(
            settings.severance_comparison_score_threshold
        ),
    )
    return privacy_reduced_trace(
        row,
        threshold_refused=decision.refusal_stage == "threshold",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_STRESS_DATASET)
    parser.add_argument("--formal-dataset", type=Path, default=DEFAULT_FORMAL_DATASET)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--export-official", action="store_true")
    args = parser.parse_args()

    run_dir = args.work_dir or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-reliability"
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"work directory must be absent or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    committed = json.loads(args.snapshot.read_text(encoding="utf-8"))
    corpus_dir = _materialize_audited_corpus(run_dir, committed)

    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        qdrant_path=str(run_dir / "qdrant"),
        storage_dir=run_dir / "storage",
        data_dir=run_dir / "data",
        collection_name="reliability_laws",
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
        stress_rows = lib.load_dataset(args.dataset)
        formal_rows = lib.load_dataset(args.formal_dataset)
        stress_raw = _run_rows(pipeline, stress_rows)
        formal_raw = _run_rows(pipeline, formal_rows)
        _write_jsonl(run_dir / "stress_raw.jsonl", stress_raw)
        _write_jsonl(run_dir / "formal_raw.jsonl", formal_raw)

        production_threshold = settings.rerank_score_threshold
        stress_metrics = compute_reliability_metrics(stress_raw, THRESHOLDS)
        formal_metrics = compute_reliability_metrics(formal_raw, THRESHOLDS)
        public_trace = [_reduce_trace(row, settings) for row in stress_raw]
        candidates = pareto_better_thresholds(
            stress_metrics,
            formal_metrics,
            production=production_threshold,
        )
        results = {
            "schema_version": "1.0",
            "run_date": date.today().isoformat(),
            "dataset": {
                "path": "eval/dataset/reliability_stress_v0.3.1.jsonl",
                "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            },
            "formal_guard_dataset": {
                "path": "eval/dataset/eval_set.jsonl",
                "sha256": hashlib.sha256(args.formal_dataset.read_bytes()).hexdigest(),
            },
            "corpus_snapshot": {
                "path": "release/corpus_snapshot.json",
                "snapshot_date": committed["snapshot_date"],
                "laws": committed["law_count"],
                "articles": committed["article_count"],
            },
            "configuration": {
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
            "production_threshold": production_threshold,
            "threshold_candidates": THRESHOLDS,
            "stress_metrics": stress_metrics,
            "formal_guard_metrics": formal_metrics,
            "decision": {
                "pareto_better_candidates": candidates,
                "outcome": "retain_0.03" if not candidates else "manual_review_required",
                "automatic_config_change": False,
            },
        }
        _write_json(run_dir / "results.json", results)
        if args.export_official:
            _write_json(OFFICIAL_DIR / "reliability_results.json", results)
            _write_jsonl(OFFICIAL_DIR / "reliability_trace.jsonl", public_trace)
            _write_jsonl(
                OFFICIAL_DIR / "reliability_formal_trace.jsonl",
                [
                    _reduce_trace(row, settings) for row in formal_raw
                ],
            )
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        print(f"[done] raw artifacts: {run_dir}", flush=True)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
