"""Audit the 15-law corpus against current Ministry of Justice XML dumps.

The default mode compares the live sources with ``release/corpus_snapshot.json``.
Use ``--write`` only when deliberately refreshing the committed provenance.
Raw ZIP/XML/law content is held in a task-specific temporary directory and is
not copied into the public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import httpx

if __package__ in {None, ""}:
    import _bootstrap  # noqa: F401
    from download_corpus import (
        DUMPS,
        CorpusArchiveError,
        iter_laws,
        normalize_name,
        validate_dump_zip,
    )
else:
    from scripts.download_corpus import (
        DUMPS,
        CorpusArchiveError,
        iter_laws,
        normalize_name,
        validate_dump_zip,
    )

from rag.corpus_audit import build_snapshot, compare_snapshots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release" / "corpus_snapshot.json"


def download_bytes(url: str) -> bytes:
    """Download one official archive, failing on redirects/errors/timeouts."""
    response = httpx.get(url, timeout=180.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def build_live_snapshot(
    *,
    snapshot_date: str,
    downloader: Callable[[str], bytes] = download_bytes,
) -> dict:
    """Download, validate, and reduce the official corpus to public provenance."""
    sources = []
    selected_laws = []
    missing = []
    with tempfile.TemporaryDirectory(prefix="tw_law_corpus_audit_") as temp_dir:
        root = Path(temp_dir)
        for source_id, (url, targets) in DUMPS.items():
            archive_bytes = downloader(url)
            digest = hashlib.sha256(archive_bytes).hexdigest()
            archive_path = root / f"{source_id}.zip"
            archive_path.write_bytes(archive_bytes)
            validate_dump_zip(archive_path)

            wanted = set(targets)
            found = {}
            for law in iter_laws(archive_path):
                name = normalize_name(law["name"])
                if name in wanted and not law["abolished"]:
                    if name in found:
                        raise ValueError(f"duplicate target law in {source_id}: {name}")
                    law["name"] = name
                    found[name] = law
            for name in targets:
                if name not in found:
                    missing.append(name)
                else:
                    selected_laws.append(found[name])
            sources.append({"id": source_id, "url": url, "sha256": digest})

    if missing:
        raise ValueError(f"missing target laws: {', '.join(missing)}")
    return build_snapshot(
        sources=sources,
        laws=selected_laws,
        snapshot_date=snapshot_date,
    )


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _run(args: argparse.Namespace) -> int:
    live = build_live_snapshot(snapshot_date=args.snapshot_date)
    if args.write is not None:
        target = _resolve_project_path(args.write)
        _write_snapshot(target, live)
        print(
            f"wrote {target}: {live['law_count']} laws, "
            f"{live['article_count']} non-deleted articles"
        )
        return 0

    target = _resolve_project_path(args.check)
    if not target.is_file():
        raise FileNotFoundError(f"committed snapshot not found: {target}")
    committed = json.loads(target.read_text(encoding="utf-8"))
    changes = compare_snapshots(committed, live)
    if changes:
        print(json.dumps({"status": "changed", "changes": changes}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "current", "changes": []}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", type=Path, metavar="PATH")
    mode.add_argument("--check", type=Path, metavar="PATH", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (CorpusArchiveError, httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {"status": "invalid_source", "error_type": type(exc).__name__}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
