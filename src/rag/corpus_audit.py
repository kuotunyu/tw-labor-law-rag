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
_CHANGE_KINDS = (
    "added",
    "removed",
    "sha256",
    "last_amended",
    "effective_date",
    "num_articles",
    "content_sha256",
)


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


def summarize_changes(changes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Summarize source/law changes without retaining corpus content."""

    counts = {kind: 0 for kind in _CHANGE_KINDS}
    laws: set[str] = set()
    has_source_change = False
    total = 0
    for change in changes:
        kind = change.get("kind")
        if kind not in counts:
            raise ValueError(f"unknown corpus change kind: {kind}")
        law = str(change.get("law", "")).strip()
        source = str(change.get("source", "")).strip()
        if bool(law) == bool(source):
            raise ValueError("corpus change requires exactly one law or source")
        if law:
            laws.add(law)
        else:
            has_source_change = True
        counts[kind] += 1
        total += 1
    return {
        "total_changes": total,
        "subjects_changed": len(laws) + int(has_source_change),
        **counts,
    }


def build_article_snapshot(
    laws: Iterable[Mapping[str, Any]],
    *,
    snapshot_date: str,
) -> dict[str, Any]:
    """Build content-free per-article fingerprints from normalized law records."""

    try:
        date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise ValueError("snapshot_date must be ISO YYYY-MM-DD") from exc

    law_rows: list[dict[str, Any]] = []
    law_names: set[str] = set()
    for raw_law in laws:
        name = str(raw_law.get("name", "")).strip()
        if not name:
            raise ValueError("law name is required")
        if name in law_names:
            raise ValueError(f"duplicate law name: {name}")
        law_names.add(name)
        articles: list[dict[str, str]] = []
        article_labels: set[str] = set()
        for raw_article in raw_law.get("articles", []):
            content = str(raw_article.get("content", "")).strip()
            if not content or _DELETED_ARTICLE.fullmatch(content):
                continue
            article = str(raw_article.get("no", "")).strip()
            if not article:
                raise ValueError(f"article label is required: {name}")
            if article in article_labels:
                raise ValueError(f"duplicate article: {name} {article}")
            article_labels.add(article)
            canonical = {
                "no": article,
                "chapter": str(raw_article.get("chapter", "")).strip(),
                "content": content,
            }
            articles.append(
                {"article": article, "sha256": _canonical_sha256(canonical)}
            )
        articles.sort(key=lambda row: row["article"])
        law_rows.append({"name": name, "articles": articles})

    law_rows.sort(key=lambda row: row["name"])
    return {
        "schema_version": "1.0",
        "snapshot_date": snapshot_date,
        "law_count": len(law_rows),
        "article_count": sum(len(row["articles"]) for row in law_rows),
        "laws": law_rows,
    }


def _article_snapshot_map(snapshot: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if "laws" not in snapshot:
        rows: dict[str, dict[str, str]] = {}
        for raw_law, raw_articles in snapshot.items():
            law = str(raw_law).strip()
            if not law or not isinstance(raw_articles, Mapping):
                raise ValueError("article snapshot laws must map to article hashes")
            articles: dict[str, str] = {}
            for raw_article, raw_digest in raw_articles.items():
                article = str(raw_article).strip()
                digest = str(raw_digest).strip()
                if not article or not _SHA256.fullmatch(digest):
                    raise ValueError("article snapshot identity or sha256 is invalid")
                articles[article] = digest
            rows[law] = articles
        return rows

    raw_laws = snapshot.get("laws")
    if not isinstance(raw_laws, list):
        raise ValueError("article snapshot laws must be a list")
    rows = {}
    for raw_law in raw_laws:
        if not isinstance(raw_law, Mapping) or set(raw_law) != {"name", "articles"}:
            raise ValueError("article snapshot law fields are invalid")
        law = str(raw_law["name"]).strip()
        if not law or law in rows or not isinstance(raw_law["articles"], list):
            raise ValueError("article snapshot law identity is invalid")
        articles = {}
        for row in raw_law["articles"]:
            if not isinstance(row, Mapping) or set(row) != {"article", "sha256"}:
                raise ValueError("article snapshot article fields are invalid")
            article = str(row["article"]).strip()
            digest = str(row["sha256"]).strip()
            if not article or article in articles or not _SHA256.fullmatch(digest):
                raise ValueError("article snapshot identity or sha256 is invalid")
            articles[article] = digest
        rows[law] = articles
    return rows


def compare_article_snapshots(
    committed: Mapping[str, Any], live: Mapping[str, Any]
) -> dict[str, Any]:
    """Return deterministic added/removed/changed article fingerprints."""

    old_laws = _article_snapshot_map(committed)
    new_laws = _article_snapshot_map(live)
    summary = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    changes: list[dict[str, Any]] = []
    for law in sorted(old_laws.keys() | new_laws.keys()):
        old_articles = old_laws.get(law, {})
        new_articles = new_laws.get(law, {})
        for article in sorted(old_articles.keys() | new_articles.keys()):
            if article not in old_articles:
                summary["added"] += 1
                changes.append(
                    {
                        "law": law,
                        "article": article,
                        "kind": "added",
                        "new": new_articles[article],
                    }
                )
            elif article not in new_articles:
                summary["removed"] += 1
                changes.append(
                    {
                        "law": law,
                        "article": article,
                        "kind": "removed",
                        "old": old_articles[article],
                    }
                )
            elif old_articles[article] != new_articles[article]:
                summary["changed"] += 1
                changes.append(
                    {
                        "law": law,
                        "article": article,
                        "kind": "changed",
                        "old": old_articles[article],
                        "new": new_articles[article],
                    }
                )
            else:
                summary["unchanged"] += 1
    return {"summary": summary, "changes": changes}
