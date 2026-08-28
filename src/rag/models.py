"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath, PureWindowsPath


def _public_source_path(value: str) -> str:
    """Return portable provenance without serializing host filesystem layout."""
    if not value:
        return ""
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value.replace("\\", "/"))
    unsafe = bool(
        windows_path.drive
        or windows_path.root
        or posix_path.root
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    )
    if unsafe:
        return windows_path.name or posix_path.name
    return posix_path.as_posix()


@dataclass
class SourceUnit:
    """A structural unit of a source document.

    One law article, one markdown section, one PDF page, or one plain-text
    document — the smallest unit that still carries citation metadata.
    """

    text: str
    doc_id: str
    doc_title: str
    article_no: str = ""  # e.g. "第 24 條" (law documents only)
    chapter: str = ""  # e.g. "第 三 章 工資", or a markdown heading path
    source_path: str = ""


@dataclass
class Chunk:
    """An indexable chunk with citation metadata.

    ``text`` is what gets embedded and retrieved: a context header (document
    title / chapter / article number) followed by the content. ``content`` is
    the bare text used when displaying citations.
    """

    chunk_id: str
    text: str
    content: str
    doc_id: str
    doc_title: str
    articles: list[str] = field(default_factory=list)  # article numbers covered
    chapter: str = ""
    seq: int = 0
    source_path: str = ""

    @property
    def article_label(self) -> str:
        """Human-readable article reference, e.g. '第 24 條' or '第 24 條～第 26 條'."""
        if not self.articles:
            return ""
        if len(self.articles) == 1:
            return self.articles[0]
        return f"{self.articles[0]}～{self.articles[-1]}"

    def payload(self) -> dict:
        data = asdict(self)
        data["source_path"] = _public_source_path(self.source_path)
        data["article_label"] = self.article_label
        return data


@dataclass
class RetrievedChunk:
    """A search hit: similarity/fusion score plus the stored chunk payload."""

    score: float
    payload: dict

    @property
    def citation(self) -> str:
        title = self.payload.get("doc_title", "")
        label = self.payload.get("article_label", "")
        return f"{title} {label}".strip()
