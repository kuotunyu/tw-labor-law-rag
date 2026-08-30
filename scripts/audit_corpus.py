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

from rag.corpus_audit import (
    build_article_snapshot,
    build_snapshot,
    compare_article_snapshots,
    compare_snapshots,
    summarize_changes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "release" / "corpus_snapshot.json"
DEFAULT_ARTICLE_SNAPSHOT = PROJECT_ROOT / "release" / "corpus_article_snapshot.json"


def download_bytes(url: str) -> bytes:
    """Download one official archive, failing on redirects/errors/timeouts."""
    response = httpx.get(url, timeout=180.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def build_live_snapshots(
    *,
    snapshot_date: str,
    downloader: Callable[[str], bytes] = download_bytes,
) -> tuple[dict, dict]:
    """Download once and reduce the corpus to law and article provenance."""
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
    return (
        build_snapshot(
            sources=sources,
            laws=selected_laws,
            snapshot_date=snapshot_date,
        ),
        build_article_snapshot(selected_laws, snapshot_date=snapshot_date),
    )


def build_live_snapshot(
    *,
    snapshot_date: str,
    downloader: Callable[[str], bytes] = download_bytes,
) -> dict:
    """Backward-compatible law/source snapshot wrapper."""

    return build_live_snapshots(
        snapshot_date=snapshot_date,
        downloader=downloader,
    )[0]


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    temporary.replace(path)


def _read_snapshot(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"committed {label} snapshot not found")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"committed {label} snapshot must be an object")
    return value


def _empty_article_summary() -> dict[str, int]:
    return {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}


def _report(
    *,
    law_changes: list[dict],
    article_report: dict,
) -> dict:
    return {
        "status": "changed" if law_changes or article_report["changes"] else "current",
        "changes": {
            "laws": law_changes,
            "articles": article_report["changes"],
        },
        "summary": {
            "laws": summarize_changes(law_changes),
            "articles": article_report["summary"],
        },
    }


def _run(args: argparse.Namespace) -> int:
    live, live_articles = build_live_snapshots(snapshot_date=args.snapshot_date)
    if args.write is not None:
        law_target = _resolve_project_path(args.write)
        article_target = _resolve_project_path(args.article_write)
        _write_snapshot(law_target, live)
        _write_snapshot(article_target, live_articles)
        print(
            json.dumps(
                {
                    "status": "written",
                    "laws": live["law_count"],
                    "articles": live_articles["article_count"],
                }
            )
        )
        return 0

    law_target = _resolve_project_path(args.check)
    committed = _read_snapshot(law_target, "law/source")
    law_changes = compare_snapshots(committed, live)

    if args.bootstrap_article_snapshot is not None:
        article_report = {"changes": [], "summary": _empty_article_summary()}
        report = _report(law_changes=law_changes, article_report=article_report)
        if law_changes:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
        target = _resolve_project_path(args.bootstrap_article_snapshot)
        _write_snapshot(target, live_articles)
        print(
            json.dumps(
                {
                    **report,
                    "article_snapshot_written": True,
                    "article_count": live_articles["article_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    article_target = _resolve_project_path(args.article_check)
    committed_articles = _read_snapshot(article_target, "article")
    article_report = compare_article_snapshots(committed_articles, live_articles)
    report = _report(law_changes=law_changes, article_report=article_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "changed":
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", type=Path, metavar="PATH")
    mode.add_argument("--bootstrap-article-snapshot", type=Path, metavar="PATH")
    parser.add_argument("--article-write", type=Path, metavar="PATH")
    parser.add_argument("--check", type=Path, metavar="PATH", default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--article-check",
        type=Path,
        metavar="PATH",
        default=DEFAULT_ARTICLE_SNAPSHOT,
    )
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    if (args.write is None) != (args.article_write is None):
        parser.error("--write and --article-write must be provided together")
    if args.bootstrap_article_snapshot is not None and args.article_write is not None:
        parser.error("bootstrap mode cannot use --article-write")

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
