"""Create-only orchestration for validated blue-green Qdrant candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from rag.config import Settings
from rag.ingestion.chunkers import get_chunker
from rag.models import Chunk, SourceUnit
from rag.qdrant_maintenance import (
    build_maintenance_receipt,
    candidate_collections,
    validate_candidate_base,
    validate_candidate_payloads,
)

_STRATEGIES = ("fixed", "structure")
MAINTENANCE_CHUNK_SIZE = 400
MAINTENANCE_CHUNK_OVERLAP = 80
MAINTENANCE_VECTOR_DIMENSION = 1024


class CandidateStore(Protocol):
    """The intentionally narrow, deletion-free candidate store interface."""

    def collection_exists(self, name: str) -> bool: ...

    def create_collection(self, name: str, dim: int) -> None: ...

    def upsert_chunks(
        self, name: str, chunks: list[Chunk], vectors: np.ndarray
    ) -> None: ...

    def count(self, name: str) -> int: ...

    def scroll_payloads(self, name: str) -> list[dict]: ...

    def close(self) -> None: ...


class CandidateEmbedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class BuildRequest:
    active_base: str
    candidate_base: str
    units: tuple[SourceUnit, ...]
    expected_point_counts: Mapping[str, int]
    snapshot_sha256: str
    source_sha256: Mapping[str, str]

    @property
    def collections(self) -> dict[str, str]:
        return candidate_collections(self.candidate_base)


@dataclass(frozen=True)
class BuildDependencies:
    store: CandidateStore
    embedder: CandidateEmbedder
    settings: Settings
    completed_at: Callable[[], datetime]


def _prepare_all_strategies(
    request: BuildRequest,
    dependencies: BuildDependencies,
) -> dict[str, tuple[list[Chunk], np.ndarray]]:
    units = list(request.units)
    if not units:
        raise ValueError("candidate corpus has no source units")
    if set(request.expected_point_counts) != set(_STRATEGIES) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in request.expected_point_counts.values()
    ):
        raise ValueError("candidate expected point counts are invalid")

    prepared: dict[str, tuple[list[Chunk], np.ndarray]] = {}
    vector_dimension: int | None = None
    for strategy in _STRATEGIES:
        chunker = get_chunker(
            strategy,
            MAINTENANCE_CHUNK_SIZE,
            MAINTENANCE_CHUNK_OVERLAP,
        )
        chunks = chunker.chunk(units)
        if not chunks:
            raise ValueError(f"candidate strategy has no chunks: {strategy}")
        if len(chunks) != request.expected_point_counts[strategy]:
            raise ValueError(f"candidate expected point count mismatch: {strategy}")
        vectors = np.asarray(
            dependencies.embedder.encode([chunk.text for chunk in chunks]),
            dtype=np.float32,
        )
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(chunks)
            or vectors.shape[1] != MAINTENANCE_VECTOR_DIMENSION
        ):
            raise ValueError(
                f"candidate vectors must be 1024-dimensional: {strategy}"
            )
        if vector_dimension is None:
            vector_dimension = int(vectors.shape[1])
        elif vectors.shape[1] != vector_dimension:
            raise ValueError("candidate vector dimensions differ between strategies")
        prepared[strategy] = (chunks, vectors)
    return prepared


def build_candidates(
    request: BuildRequest,
    dependencies: BuildDependencies,
) -> dict[str, object]:
    """Build and validate both candidates without any delete or overwrite path."""
    validate_candidate_base(request.active_base, request.candidate_base)
    collections = request.collections

    existing = {
        strategy: dependencies.store.collection_exists(collections[strategy])
        for strategy in _STRATEGIES
    }
    if any(existing.values()):
        raise ValueError("candidate collection exists")

    # Both chunking and embedding passes complete before the first cloud write.
    prepared = _prepare_all_strategies(request, dependencies)
    point_counts: dict[str, int] = {}
    vector_dimension: int | None = None
    for strategy in _STRATEGIES:
        chunks, vectors = prepared[strategy]
        name = collections[strategy]
        dimension = int(vectors.shape[1])
        dependencies.store.create_collection(name, dim=dimension)
        dependencies.store.upsert_chunks(name, chunks, vectors)

        actual_count = dependencies.store.count(name)
        if actual_count != request.expected_point_counts[strategy]:
            raise ValueError(f"candidate count mismatch: {strategy}")
        payloads = dependencies.store.scroll_payloads(name)
        validate_candidate_payloads(
            strategy,
            payloads,
            expected_count=request.expected_point_counts[strategy],
        )
        point_counts[strategy] = actual_count
        vector_dimension = dimension

    if vector_dimension is None:  # pragma: no cover - guarded by preparation
        raise RuntimeError("candidate vector dimension is unavailable")
    return build_maintenance_receipt(
        completed_at=dependencies.completed_at(),
        active_base=request.active_base,
        candidate_base=request.candidate_base,
        point_counts=point_counts,
        corpus_snapshot_sha256=request.snapshot_sha256,
        source_sha256=request.source_sha256,
        embedding_model=dependencies.settings.embedding_model,
        embedding_revision=dependencies.settings.embedding_model_revision,
        vector_dimension=vector_dimension,
    )
