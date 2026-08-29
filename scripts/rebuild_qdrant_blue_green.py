"""Safely plan or execute a create-only blue-green Qdrant corpus rebuild.

The default is a local dry-run.  Cloud execution additionally requires an
explicit flag, repeated candidate confirmation, a temporary writer key, and
both pinned model snapshots already present in the local Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

if __package__ in {None, ""}:
    import _bootstrap  # noqa: F401
    from download_corpus import DUMPS
else:
    from scripts.download_corpus import DUMPS

from rag.config import (
    DEFAULT_EMBEDDING_MODEL_REVISION,
    DEFAULT_RERANKER_MODEL_REVISION,
    Settings,
)
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.qdrant_blue_green import BuildDependencies, BuildRequest, build_candidates
from rag.qdrant_maintenance import (
    build_local_snapshot,
    candidate_collections,
    validate_candidate_base,
    validate_snapshot_match,
    write_receipt_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path("data/raw/laws")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_SNAPSHOT = Path("release/corpus_snapshot.json")
DEFAULT_RECEIPT = Path("eval/runs/qdrant-maintenance/receipt.json")
_PINNED_MODELS = (
    ("BAAI/bge-m3", DEFAULT_EMBEDDING_MODEL_REVISION),
    ("BAAI/bge-reranker-v2-m3", DEFAULT_RERANKER_MODEL_REVISION),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-base", required=True, metavar="NAME")
    parser.add_argument("--confirm-candidate-base", metavar="NAME")
    parser.add_argument(
        "--active-base",
        default=os.environ.get("COLLECTION_NAME", "").strip() or "labor_laws",
        metavar="NAME",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, metavar="PATH")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, metavar="PATH")
    parser.add_argument(
        "--snapshot", type=Path, default=DEFAULT_SNAPSHOT, metavar="PATH"
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT, metavar="PATH")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _snapshot_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dry_run_plan(args: argparse.Namespace) -> dict[str, object]:
    """Validate only local evidence and return the redacted execution plan."""
    validate_candidate_base(args.active_base, args.candidate_base)
    raw_dir = _project_path(args.raw_dir)
    snapshot_path = _project_path(args.snapshot)
    committed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    source_archives = {
        source_id: (url, raw_dir / f"chlaw_{source_id}.zip")
        for source_id, (url, _targets) in DUMPS.items()
    }
    local = build_local_snapshot(
        source_archives=source_archives,
        laws_dir=_project_path(args.corpus),
        snapshot_date=date.today().isoformat(),
    )
    validate_snapshot_match(committed, local)
    return {
        "status": "dry_run_ready",
        "active_base": args.active_base,
        "candidate_base": args.candidate_base,
        "collections": candidate_collections(args.candidate_base),
        "snapshot_sha256": _snapshot_sha256(snapshot_path),
        "execution_required": True,
    }


def _emit_error(code: str) -> int:
    print(json.dumps({"status": "error", "code": code}), file=sys.stderr)
    return 2


def _writer_environment_ready() -> bool:
    return bool(
        os.environ.get("QDRANT_URL", "").strip()
        and os.environ.get("QDRANT_WRITER_API_KEY", "").strip()
    )


def _pinned_models_are_cached() -> bool:
    missing = False
    for repo_id, revision in _PINNED_MODELS:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=True,
            )
        except (LocalEntryNotFoundError, OSError):
            missing = True
    return not missing


def _receipt_target_is_safe(target: Path) -> bool:
    if target.is_absolute() or ".." in target.parts:
        return False
    return target.parts[:3] == ("eval", "runs", "qdrant-maintenance") and len(
        target.parts
    ) > 3


def _execute(args: argparse.Namespace, plan: dict[str, object]) -> int:
    store = None
    writer_settings = None
    try:
        writer_settings = Settings(
            _env_file=None,
            deployment_mode="standard",
            qdrant_mode="server",
            qdrant_url=os.environ["QDRANT_URL"].strip(),
            qdrant_api_key=os.environ["QDRANT_WRITER_API_KEY"].strip(),
            collection_name=args.active_base,
            device=args.device,
            embedding_model="BAAI/bge-m3",
            embedding_model_revision=DEFAULT_EMBEDDING_MODEL_REVISION,
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_model_revision=DEFAULT_RERANKER_MODEL_REVISION,
            anthropic_api_key="",
            openai_api_key="",
            gemini_api_key="",
        )
        committed = json.loads(_project_path(args.snapshot).read_text(encoding="utf-8"))
        source_sha256 = {
            str(source["id"]): str(source["sha256"])
            for source in committed["sources"]
        }
        request = BuildRequest(
            active_base=args.active_base,
            candidate_base=args.candidate_base,
            corpus_dir=_project_path(args.corpus),
            receipt_path=args.receipt,
            snapshot_sha256=str(plan["snapshot_sha256"]),
            source_sha256=source_sha256,
        )
        store = VectorStore(writer_settings)
        embedder = BGEM3Embedder(
            model_name=writer_settings.embedding_model,
            model_revision=writer_settings.embedding_model_revision,
            device=args.device,
            cache_path=writer_settings.storage_dir / "emb_cache.sqlite",
        )
        dependencies = BuildDependencies(
            store=store,
            embedder=embedder,
            settings=writer_settings,
            completed_at=lambda: datetime.now(timezone.utc),
        )
        receipt = build_candidates(request, dependencies)
        written = write_receipt_atomic(
            receipt,
            args.receipt,
            project_root=PROJECT_ROOT,
        )
    except Exception:
        return _emit_error("candidate_build_failed")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        writer_settings = None

    output = {
        "status": "candidate_ready",
        "candidate_base": args.candidate_base,
        "collections": plan["collections"],
        "receipt": written.relative_to(PROJECT_ROOT).as_posix(),
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_candidate_base(args.active_base, args.candidate_base)
    except ValueError:
        return _emit_error("invalid_candidate")

    if args.execute and args.confirm_candidate_base != args.candidate_base:
        return _emit_error("candidate_confirmation_mismatch")

    try:
        plan = build_dry_run_plan(args)
    except FileNotFoundError:
        return _emit_error("missing_local_corpus")
    except (json.JSONDecodeError, KeyError, TypeError):
        return _emit_error("invalid_local_corpus")
    except (OSError, ValueError):
        return _emit_error("snapshot_drift")

    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    if not _writer_environment_ready():
        return _emit_error("missing_writer_environment")
    if not _receipt_target_is_safe(args.receipt):
        return _emit_error("invalid_receipt_target")
    if not _pinned_models_are_cached():
        return _emit_error("missing_model_snapshot")
    return _execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
