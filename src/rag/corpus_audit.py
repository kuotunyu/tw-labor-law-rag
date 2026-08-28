"""Deterministic, privacy-safe corpus snapshot helpers.

The complete law corpus remains outside the public repository.  This module
reduces it to hashes and public provenance fields so a release can prove which
official snapshot it used without redistributing every article.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DELETED_ARTICLE = re.compile(r"^[（(]\s*刪除\s*[）)]$")
_LAW_FIELDS = ("last_amended", "effective_date", "num_articles", "content_sha256")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _snapshot_law(raw: Mapping[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("law name is required")
    articles = []
    for raw_article in raw.get("articles", []):
        content = str(raw_article.get("content", "")).strip()
        if not content or _DELETED_ARTICLE.fullmatch(content):
            continue
        articles.append(
            {
                "no": str(raw_article.get("no", "")).strip(),
                "chapter": str(raw_article.get("chapter", "")).strip(),
                "content": content,
            }
        )
    canonical_law = {
        "name": name,
        "nature": str(raw.get("nature", "")).strip(),
        "url": str(raw.get("url", "")).strip(),
        "last_amended": str(raw.get("last_amended", "")).strip(),
        "effective_date": str(raw.get("effective_date", "")).strip(),
        "articles": articles,
    }
    return {
        "name": canonical_law["name"],
        "nature": canonical_law["nature"],
        "url": canonical_law["url"],
        "last_amended": canonical_law["last_amended"],
        "effective_date": canonical_law["effective_date"],
        "num_articles": len(articles),
        "content_sha256": _canonical_sha256(canonical_law),
    }


def build_snapshot(
    *,
    sources: Iterable[Mapping[str, Any]],
    laws: Iterable[Mapping[str, Any]],
    snapshot_date: str,
) -> dict[str, Any]:
    """Build a deterministic public snapshot from official source metadata and laws."""
    try:
        date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise ValueError("snapshot_date must be ISO YYYY-MM-DD") from exc

    source_rows: list[dict[str, str]] = []
    source_ids: set[str] = set()
    for source in sources:
        source_id = str(source.get("id", "")).strip()
        if not source_id:
            raise ValueError("source id is required")
        if source_id in source_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        source_ids.add(source_id)
        digest = str(source.get("sha256", "")).strip()
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"source sha256 is invalid: {source_id}")
        source_rows.append(
            {
                "id": source_id,
                "url": str(source.get("url", "")).strip(),
                "sha256": digest,
            }
        )

    law_rows: list[dict[str, Any]] = []
    law_names: set[str] = set()
    for law in laws:
        row = _snapshot_law(law)
        if row["name"] in law_names:
            raise ValueError(f"duplicate law name: {row['name']}")
        law_names.add(row["name"])
        law_rows.append(row)

    source_rows.sort(key=lambda row: row["id"])
    law_rows.sort(key=lambda row: row["name"])
    return {
        "schema_version": "1.0",
        "snapshot_date": snapshot_date,
        "sources": source_rows,
        "law_count": len(law_rows),
        "article_count": sum(row["num_articles"] for row in law_rows),
        "laws": law_rows,
    }


def compare_snapshots(
    committed: Mapping[str, Any], live: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return deterministic field-level changes between two corpus snapshots."""
    changes: list[dict[str, Any]] = []

    old_sources = {row["id"]: row for row in committed.get("sources", [])}
    new_sources = {row["id"]: row for row in live.get("sources", [])}
    for source_id in sorted(old_sources.keys() | new_sources.keys()):
        if source_id not in old_sources:
            changes.append({"source": source_id, "kind": "added"})
            continue
        if source_id not in new_sources:
            changes.append({"source": source_id, "kind": "removed"})
            continue
        if old_sources[source_id]["sha256"] != new_sources[source_id]["sha256"]:
            changes.append(
                {
                    "source": source_id,
                    "kind": "sha256",
                    "old": old_sources[source_id]["sha256"],
                    "new": new_sources[source_id]["sha256"],
                }
            )

    old_laws = {row["name"]: row for row in committed.get("laws", [])}
    new_laws = {row["name"]: row for row in live.get("laws", [])}
    for name in sorted(old_laws.keys() | new_laws.keys()):
        if name not in old_laws:
            changes.append({"law": name, "kind": "added"})
            continue
        if name not in new_laws:
            changes.append({"law": name, "kind": "removed"})
            continue
        for field in _LAW_FIELDS:
            if old_laws[name][field] != new_laws[name][field]:
                changes.append(
                    {
                        "law": name,
                        "kind": field,
                        "old": old_laws[name][field],
                        "new": new_laws[name][field],
                    }
                )
    return changes
