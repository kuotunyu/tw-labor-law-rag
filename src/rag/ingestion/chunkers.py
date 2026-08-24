"""Chunking strategies (strategy pattern).

Two implementations, switchable via config for the ablation study:

- :class:`StructureAwareChunker` — one chunk per source unit (law article /
  markdown section). Long units are split at line boundaries. Every chunk is
  prefixed with a context header (document title, chapter, article number) so
  the embedding carries its provenance.
- :class:`FixedSizeChunker` — classic sliding window over the whole document,
  packed on sentence boundaries with overlap, ignoring document structure.
  Citation metadata records every article the window overlaps.
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol

from rag.models import Chunk, SourceUnit

# Sentence boundaries for Chinese legal text: sentence-ending punctuation or newline.
_SENTENCE_END = re.compile(r"(?<=[。！？；\n])")


def split_sentences(text: str, hard_limit: int) -> list[str]:
    """Split on sentence boundaries; hard-split any sentence longer than ``hard_limit``.

    Whitespace-only fragments (e.g. a lone newline between items) are glued to
    the previous sentence so no text is lost.
    """
    sentences: list[str] = []
    for sent in _SENTENCE_END.split(text):
        if not sent:
            continue
        if not sent.strip():
            if sentences:
                sentences[-1] += sent
            continue
        while len(sent) > hard_limit:
            sentences.append(sent[:hard_limit])
            sent = sent[hard_limit:]
        if sent:
            sentences.append(sent)
    return sentences


def _header(unit: SourceUnit, with_article: bool = True) -> str:
    parts = [unit.doc_title]
    if unit.chapter:
        parts.append(unit.chapter)
    if with_article and unit.article_no:
        parts.append(unit.article_no)
    return " ".join(parts)


class Chunker(Protocol):
    def chunk(self, units: Iterable[SourceUnit]) -> list[Chunk]: ...


class StructureAwareChunker:
    """One chunk per structural unit; long units split at line boundaries."""

    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def chunk(self, units: Iterable[SourceUnit]) -> list[Chunk]:
        chunks: list[Chunk] = []
        seq_by_doc: dict[str, int] = {}
        for unit in units:
            header = _header(unit)
            for part in self._split_unit(unit.text):
                seq = seq_by_doc.get(unit.doc_id, 0)
                seq_by_doc[unit.doc_id] = seq + 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{unit.doc_id}#s{seq:04d}",
                        text=f"{header}\n{part}",
                        content=part,
                        doc_id=unit.doc_id,
                        doc_title=unit.doc_title,
                        articles=[unit.article_no] if unit.article_no else [],
                        chapter=unit.chapter,
                        seq=seq,
                        source_path=unit.source_path,
                    )
                )
        return chunks

    def _split_unit(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        parts: list[str] = []
        buffer: list[str] = []
        length = 0
        for line in split_sentences(text, hard_limit=self.max_chars):
            if buffer and length + len(line) > self.max_chars:
                parts.append("".join(buffer).strip())
                buffer, length = [], 0
            buffer.append(line)
            length += len(line)
        if buffer:
            parts.append("".join(buffer).strip())
        return [p for p in parts if p]


class FixedSizeChunker:
    """Sliding window with overlap, packed on sentence boundaries per document."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, units: Iterable[SourceUnit]) -> list[Chunk]:
        by_doc: dict[str, list[SourceUnit]] = {}
        for unit in units:
            by_doc.setdefault(unit.doc_id, []).append(unit)

        chunks: list[Chunk] = []
        for doc_id, doc_units in by_doc.items():
            chunks.extend(self._chunk_doc(doc_id, doc_units))
        return chunks

    def _chunk_doc(self, doc_id: str, units: list[SourceUnit]) -> list[Chunk]:
        # Flatten the document into (sentence, owning unit) pairs.
        segments: list[tuple[str, SourceUnit]] = []
        for unit in units:
            for sent in split_sentences(unit.text, hard_limit=self.chunk_size):
                segments.append((sent, unit))
        if not segments:
            return []

        doc_title = units[0].doc_title
        chunks: list[Chunk] = []
        buffer: list[tuple[str, SourceUnit]] = []
        carried = 0  # sentences at buffer start carried over as overlap

        def emit() -> None:
            body = "".join(s for s, _ in buffer).strip()
            if not body:
                return
            seen: list[str] = []
            for _, u in buffer:
                if u.article_no and u.article_no not in seen:
                    seen.append(u.article_no)
            seq = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#f{seq:04d}",
                    text=f"{doc_title}\n{body}",
                    content=body,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    articles=seen,
                    chapter=buffer[0][1].chapter,
                    seq=seq,
                    source_path=buffer[0][1].source_path,
                )
            )

        for segment in segments:
            length = sum(len(s) for s, _ in buffer)
            fresh = len(buffer) - carried
            if buffer and fresh > 0 and length + len(segment[0]) > self.chunk_size:
                emit()
                buffer = self._tail_for_overlap(buffer)
                carried = len(buffer)
            buffer.append(segment)
        if len(buffer) > carried:
            emit()
        return chunks

    def _tail_for_overlap(self, buffer: list[tuple[str, SourceUnit]]) -> list[tuple[str, SourceUnit]]:
        tail: list[tuple[str, SourceUnit]] = []
        total = 0
        for segment in reversed(buffer):
            if total >= self.overlap:
                break
            tail.insert(0, segment)
            total += len(segment[0])
        return tail


def get_chunker(strategy: str, chunk_size: int = 400, overlap: int = 80) -> Chunker:
    if strategy == "structure":
        return StructureAwareChunker()
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"unknown chunking strategy: {strategy}")
