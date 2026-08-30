"""Run the content-free v0.3.5 portfolio regression without an LLM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from eval import _bootstrap as _eval_bootstrap
except ModuleNotFoundError:  # Direct execution starts with eval/ as the import root.
    import _bootstrap as _eval_bootstrap

_eval_bootstrap  # Keep the import for its documented src/ bootstrap side effect.

from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.portfolio_demo_regression import (  # noqa: E402
    PortfolioCase,
    build_artifact,
    build_result,
    load_cases,
)
from rag.retrieval.pipeline import (  # noqa: E402
    _WAGE_ARREARS_LEGAL_TERMS,
    _retrieval_query,
)
from rag.wage_arrears_regression import require_cached_models  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "eval/dataset/portfolio_demo_v0.3.5.jsonl"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release/corpus_snapshot.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval/runs/portfolio_demo_v0.3.5.json"
Observation = dict[str, Any]


def write_public_json(path: Path, value: object) -> None:
    """Atomically write deterministic UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_cases(
    cases: list[PortfolioCase],
    evaluate: Callable[[PortfolioCase], Observation],
) -> list[dict[str, Any]]:
    """Evaluate all reviewed cases through an injected retrieval-only boundary."""

    results = []
    for case in cases:
        observation = evaluate(case)
        if set(observation) != {
            "retrieved",
            "applied_routes",
            "threshold_refused",
            "top_score",
        }:
            raise ValueError(f"{case.qid}: invalid retrieval observation fields")
        results.append(build_result(case, **observation))
    return results


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


def _applied_routes(question: str) -> list[str]:
    expanded = _retrieval_query(question)
    return ["wage_arrears"] if _WAGE_ARREARS_LEGAL_TERMS in expanded else []


def _retrieved_identities(hits) -> list[tuple[str, str, int]]:
    ranks: dict[tuple[str, str], int] = {}
    for rank, hit in enumerate(hits, 1):
        law = hit.payload.get("doc_title", "")
        for article in hit.payload.get("articles", []):
            identity = (law, article)
            if all(identity) and identity not in ranks:
                ranks[identity] = rank
    return [(law, article, rank) for (law, article), rank in ranks.items()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional pre-materialized audited 15-law corpus directory.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require pinned retrieval models from local caches; never load an LLM.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _run_with_local_index(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    preflight = Settings(_env_file=None, device=args.device)
    require_cached_models(preflight)

    from run_reliability_eval import _build_indexes, _materialize_audited_corpus

    from rag.factory import build_retrieval_pipeline
    from rag.retrieval.reranker import Reranker

    cases = load_cases(args.dataset)
    with tempfile.TemporaryDirectory(prefix="tw-labor-portfolio-") as temp:
        work_dir = Path(temp)
        if args.data_dir is None:
            committed_snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
            corpus_dir = _materialize_audited_corpus(work_dir, committed_snapshot)
        else:
            corpus_dir = args.data_dir.resolve()
        settings = Settings(
            _env_file=None,
            qdrant_mode="local",
            qdrant_path=str(work_dir / "qdrant"),
            storage_dir=work_dir / "storage",
            data_dir=work_dir / "data",
            collection_name="portfolio_demo_regression",
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

            def evaluate(case: PortfolioCase) -> Observation:
                retrieval = pipeline.run(case.question)
                return {
                    "retrieved": _retrieved_identities(retrieval.hits),
                    "applied_routes": _applied_routes(case.question),
                    "threshold_refused": (
                        not retrieval.hits
                        or retrieval.top_score < settings.rerank_score_threshold
                    ),
                    "top_score": retrieval.top_score,
                }

            results = run_cases(cases, evaluate)
        finally:
            store.close()
    configuration = {
        "chunking": "structure",
        "retrieval": "hybrid",
        "reranker": True,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_final": settings.top_k_final,
        "rrf_k": settings.rrf_k,
        "rerank_score_threshold": settings.rerank_score_threshold,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_model_revision,
    }
    return results, configuration


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results, configuration = _run_with_local_index(args)
    artifact = build_artifact(
        dataset_path=args.dataset,
        snapshot_path=args.snapshot,
        code_revision=_git_revision(),
        configuration=configuration,
        results=results,
    )
    write_public_json(args.output, artifact)
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if artifact["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
