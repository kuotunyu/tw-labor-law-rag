"""Build the content-free v0.3.6 retrieval-pivot calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if "--offline" in sys.argv[1:]:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

try:
    from eval import _bootstrap as _eval_bootstrap
except ModuleNotFoundError:  # Direct execution starts with eval/ as import root.
    import _bootstrap as _eval_bootstrap

_eval_bootstrap  # Keep the import for its documented src/ bootstrap side effect.

from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.retrieval.pipeline import (  # noqa: E402
    MULTI_VIEW_MERGE_POLICY_VERSION,
    SEVERANCE_SEMANTIC_VIEW_SHA256,
    plan_retrieval_query,
)
from rag.severance_refusal_policy import (  # noqa: E402
    CANDIDATE_THRESHOLDS,
    DECISION_CODE_PATHS,
    SeverancePolicyCase,
    build_case_observation,
    build_no_go_evidence,
    build_official_artifact,
    evaluate_route_ablation_candidate,
    load_cases,
)
from rag.wage_arrears_regression import require_cached_models  # noqa: E402

DEFAULT_DATASET = (
    PROJECT_ROOT / "eval/dataset/severance_refusal_policy_v0.3.6.jsonl"
)
DEFAULT_STRESS_DATASET = (
    PROJECT_ROOT / "eval/dataset/reliability_stress_v0.3.1.jsonl"
)
DEFAULT_FORMAL_DATASET = PROJECT_ROOT / "eval/dataset/eval_set.jsonl"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release/corpus_snapshot.json"
OFFICIAL_RESULT = (
    PROJECT_ROOT / "eval/official/severance_refusal_policy_v0.3.6.json"
)
DIAGNOSTIC_RESULT = (
    PROJECT_ROOT
    / "eval/diagnostics/severance_retrieval_pivot_v0.3.6_no_go.json"
)
RUNS_DIR = PROJECT_ROOT / "eval/runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--stress-dataset", type=Path, default=DEFAULT_STRESS_DATASET
    )
    parser.add_argument(
        "--formal-dataset", type=Path, default=DEFAULT_FORMAL_DATASET
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--diagnostics-output", type=Path, default=DIAGNOSTIC_RESULT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--export-official", action="store_true")
    return parser


def _offline_preflight(args: argparse.Namespace) -> Settings:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    settings = Settings(_env_file=None, device=args.device)
    require_cached_models(settings)
    return settings


def _prepare_work_dir(path: Path) -> Path:
    work_dir = path.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"work directory must be absent or empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _source_ranks(
    hits: list[Any], sources: tuple[dict[str, str], ...]
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for source in sources:
        key = f"{source['law']}|{source['article']}"
        for rank, hit in enumerate(hits, start=1):
            if (
                hit.payload.get("doc_title") == source["law"]
                and source["article"] in hit.payload.get("articles", [])
            ):
                ranks[key] = rank
                break
    return ranks


def _run_target_cases(
    pipeline: Any, cases: list[SeverancePolicyCase]
) -> list[dict[str, Any]]:
    observations = []
    for index, case in enumerate(cases, start=1):
        retrieval = pipeline.run(case.question)
        planned_routes = plan_retrieval_query(case.question).routes
        if retrieval.applied_routes != planned_routes:
            raise RuntimeError(f"route mismatch for {case.qid}")
        observations.append(
            build_case_observation(
                case,
                source_ranks=_source_ranks(retrieval.hits, case.sources),
                applied_routes=retrieval.applied_routes,
                top_score=retrieval.top_score,
                hit_count=len(retrieval.hits),
                candidate_count=len(retrieval.candidates),
                route_plan_matched=True,
                first_stage_retrieval_calls=retrieval.first_stage_retrieval_calls,
                reranker_calls=retrieval.reranker_calls,
                reranker_scored_pairs=retrieval.reranker_scored_pairs,
            )
        )
        print(f"[target] {index}/{len(cases)} {case.qid}", flush=True)
    return observations


def _match_rank(hits: list[Any], sources: list[dict[str, str]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        law = hit.payload.get("doc_title", "")
        articles = hit.payload.get("articles", [])
        if any(
            law == source["doc"] and source["article"] in articles
            for source in sources
        ):
            return rank
    return None


def _run_guard_cases(
    pipeline: Any, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evidence = []
    for index, row in enumerate(rows, start=1):
        retrieval = pipeline.run(row["question"])
        planned_routes = plan_retrieval_query(row["question"]).routes
        if retrieval.applied_routes != planned_routes:
            raise RuntimeError(f"route mismatch for {row['qid']}")
        evidence.append(
            {
                "qid": row["qid"],
                "answerable": row["answerable"],
                "rank": (
                    _match_rank(retrieval.hits, row["sources"])
                    if row["answerable"]
                    else None
                ),
                "hit_count": len(retrieval.hits),
                "top_score": retrieval.top_score,
                "applied_routes": list(retrieval.applied_routes),
                "candidate_count": len(retrieval.candidates),
                "route_plan_matched": True,
                "first_stage_retrieval_calls": retrieval.first_stage_retrieval_calls,
                "reranker_calls": retrieval.reranker_calls,
                "reranker_scored_pairs": list(retrieval.reranker_scored_pairs),
            }
        )
        print(f"[guard] {index}/{len(rows)} {row['qid']}", flush=True)
    return evidence


def _evaluate_route_ablation(
    observations: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        evaluate_route_ablation_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=0.03,
            stress_rows=stress_rows,
            formal_rows=formal_rows,
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]


def _build_accepted_artifact(
    *,
    observations: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    candidate_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = candidate_results or _evaluate_route_ablation(
        observations, stress_rows, formal_rows
    )
    try:
        artifact = build_official_artifact(
            observations=observations,
            candidate_results=candidates,
            provenance=provenance,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"NO-GO: {exc}") from exc
    if artifact["production_threshold"] != 0.03:
        raise RuntimeError(
            "NO-GO: production threshold does not equal the approved 0.03"
        )
    if artifact["route_ablation"]["highest_passing_candidate"] != 0.03:
        raise RuntimeError(
            "NO-GO: route ablation highest passing candidate must equal 0.03"
        )
    return artifact


def _write_public_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    temporary.replace(path)


def _build_provenance(
    args: argparse.Namespace, settings: Settings, code_revision: str
) -> dict[str, Any]:
    return {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "corpus_snapshot_sha256": hashlib.sha256(
            args.snapshot.read_bytes()
        ).hexdigest(),
        "source_artifact_sha256": {
            "stress_dataset": hashlib.sha256(
                args.stress_dataset.read_bytes()
            ).hexdigest(),
            "formal_dataset": hashlib.sha256(
                args.formal_dataset.read_bytes()
            ).hexdigest(),
        },
        "decision_code_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for name, path in DECISION_CODE_PATHS.items()
        },
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_model_revision,
        "retrieval_configuration": {
            "chunking": "structure",
            "retrieval": "hybrid",
            "reranker": True,
            "top_k_retrieve": settings.top_k_retrieve,
            "top_k_final": settings.top_k_final,
            "rrf_k": settings.rrf_k,
        },
        "execution_device": settings.device,
        "precision_mode": "fp32",
        "local_files_only": True,
        "semantic_view_sha256": SEVERANCE_SEMANTIC_VIEW_SHA256,
        "merge_policy_version": MULTI_VIEW_MERGE_POLICY_VERSION,
        "primary_score_semantics": "full_precision_primary_query_top_score",
        "source_tree_clean": True,
        "code_revision": code_revision,
        "run_origin": "fresh_offline_retrieval",
        "provider_adapters": 0,
        "provider_requests": 0,
    }


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"dataset rows must be objects: {path.name}")
    return rows


def _clean_git_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    if status.stdout:
        raise RuntimeError(
            "calibration requires a clean tracked and untracked tree"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return revision.stdout.strip()


def _reliability_helpers():
    eval_dir = str(PROJECT_ROOT / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from run_reliability_eval import (  # noqa: PLC0415
        _build_indexes as build_indexes,
    )
    from run_reliability_eval import (  # noqa: PLC0415
        _materialize_audited_corpus as materialize_audited_corpus,
    )

    return materialize_audited_corpus, build_indexes


def _materialize_audited_corpus(work_dir: Path, committed: dict) -> Path:
    materialize, _ = _reliability_helpers()
    return materialize(work_dir, committed)


def _build_indexes(
    settings: Settings, corpus_dir: Path, *, local_files_only: bool = False
):
    _, build_indexes = _reliability_helpers()
    return build_indexes(
        settings, corpus_dir, local_files_only=local_files_only
    )


def _build_local_pipeline(
    args: argparse.Namespace, work_dir: Path, corpus_dir: Path
):
    from rag.factory import build_retrieval_pipeline  # noqa: PLC0415
    from rag.indexing.embedder import resolve_device  # noqa: PLC0415
    from rag.retrieval.reranker import Reranker  # noqa: PLC0415

    execution_device = resolve_device(args.device)
    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        qdrant_path=str(work_dir / "qdrant"),
        storage_dir=work_dir / "storage",
        data_dir=work_dir / "data",
        collection_name="severance_refusal_policy",
        device=execution_device,
        chunking_strategy="structure",
        retrieval_mode="hybrid",
        use_reranker=True,
    )
    embedder, store = _build_indexes(
        settings, corpus_dir, local_files_only=True
    )
    try:
        reranker = Reranker(
            model_name=settings.reranker_model,
            model_revision=settings.reranker_model_revision,
            device=settings.device,
            local_files_only=True,
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
    except Exception:
        store.close()
        embedder.close()
        raise
    return settings, embedder, store, pipeline


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.offline:
        parser.error("--offline is required for calibration")
    _offline_preflight(args)
    candidate_revision = _clean_git_revision()
    default_work_dir = RUNS_DIR / (
        f"{datetime.now():%Y%m%d-%H%M%S}-severance-refusal-policy"
    )
    work_dir = _prepare_work_dir(args.work_dir or default_work_dir)
    committed = json.loads(args.snapshot.read_text(encoding="utf-8"))
    corpus_dir = _materialize_audited_corpus(work_dir, committed)
    settings, embedder, store, pipeline = _build_local_pipeline(
        args, work_dir, corpus_dir
    )
    try:
        cases = load_cases(args.dataset)
        stress_rows = _load_dataset(args.stress_dataset)
        formal_rows = _load_dataset(args.formal_dataset)
        observations = _run_target_cases(pipeline, cases)
        stress_evidence = _run_guard_cases(pipeline, stress_rows)
        formal_evidence = _run_guard_cases(pipeline, formal_rows)
        candidates = _evaluate_route_ablation(
            observations, stress_evidence, formal_evidence
        )
        provenance = _build_provenance(args, settings, candidate_revision)
        try:
            artifact = _build_accepted_artifact(
                observations=observations,
                stress_rows=stress_evidence,
                formal_rows=formal_evidence,
                provenance=provenance,
                candidate_results=candidates,
            )
        except RuntimeError:
            artifact = None
        if (
            artifact is None
            or artifact["production_threshold"] != 0.03
            or artifact["route_ablation"]["highest_passing_candidate"] != 0.03
        ):
            diagnostic = build_no_go_evidence(
                observations=observations,
                candidate_results=candidates,
                provenance=provenance,
            )
            _write_public_json(args.diagnostics_output, diagnostic)
            print(
                json.dumps(
                    {
                        "outcome": "no_go",
                        "highest_passing_candidate": diagnostic["route_ablation"][
                            "highest_passing_candidate"
                        ],
                        "failed_gates": diagnostic["failed_gates"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 1
        _write_public_json(work_dir / "results.json", artifact)
        if args.export_official:
            _write_public_json(OFFICIAL_RESULT, artifact)
        print(
            json.dumps(
                {
                    "outcome": "accepted",
                    "production_threshold": artifact["production_threshold"],
                    "highest_passing_candidate": artifact["route_ablation"][
                        "highest_passing_candidate"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        store.close()
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
