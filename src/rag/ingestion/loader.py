"""Load source files into :class:`~rag.models.SourceUnit` lists.

Supported formats:
  - law JSON produced by ``scripts/download_corpus.py`` (one unit per article)
  - Markdown (one unit per heading section)
  - plain text (one unit per file)
  - PDF (one unit per page, via pypdf)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.ingestion.cleaner import clean_text, normalize_label
from rag.models import SourceUnit

_DELETED_ARTICLE = re.compile(r"^[（(]\s*刪除\s*[）)]$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def load_law_json(path: Path) -> list[SourceUnit]:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data["name"]
    units = []
    for article in data.get("articles", []):
        text = clean_text(article.get("content", ""))
        if not text or _DELETED_ARTICLE.match(text):
            continue
        units.append(
            SourceUnit(
                text=text,
                doc_id=title,
                doc_title=title,
                article_no=normalize_label(article.get("no", "")),
                chapter=normalize_label(article.get("chapter", "")),
                source_path=str(path),
            )
        )
    return units


def load_markdown(path: Path) -> list[SourceUnit]:
    text = clean_text(path.read_text(encoding="utf-8"))
    title = path.stem
    units: list[SourceUnit] = []
    heading_stack: dict[int, str] = {}
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        heading_path = " > ".join(v for _, v in sorted(heading_stack.items()))
        units.append(
            SourceUnit(
                text=body,
                doc_id=title,
                doc_title=title,
                chapter=heading_path,
                source_path=str(path),
            )
        )

    for line in text.split("\n"):
        match = _MD_HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_stack[level] = match.group(2).strip()
            for deeper in [k for k in heading_stack if k > level]:
                del heading_stack[deeper]
        else:
            buffer.append(line)
    flush()
    return units


def load_text(path: Path) -> list[SourceUnit]:
    body = clean_text(path.read_text(encoding="utf-8"))
    if not body:
        return []
    title = path.stem
    return [SourceUnit(text=body, doc_id=title, doc_title=title, source_path=str(path))]


def load_pdf(path: Path) -> list[SourceUnit]:
    from pypdf import PdfReader  # lazy: pypdf only needed when PDFs are ingested

    title = path.stem
    units = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        body = clean_text(page.extract_text() or "")
        if not body:
            continue
        units.append(
            SourceUnit(
                text=body,
                doc_id=title,
                doc_title=title,
                chapter=f"第 {page_no} 頁",
                source_path=str(path),
            )
        )
    return units


_LOADERS = {
    ".json": load_law_json,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".pdf": load_pdf,
}

_SKIP_FILES = {"manifest.json"}


def load_file(path: Path) -> list[SourceUnit]:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path}")
    return loader(path)


def load_corpus(root: Path) -> list[SourceUnit]:
    """Load every supported file under ``root`` (a file or a directory)."""
    root = Path(root)
    if root.is_file():
        return load_file(root)
    units: list[SourceUnit] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS and path.name not in _SKIP_FILES:
            units.extend(load_file(path))
    return units
