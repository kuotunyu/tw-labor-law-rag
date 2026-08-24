"""Qdrant wrapper: in-process local mode (no Docker) or server mode, via config.

Point IDs are UUIDv5 hashes of the deterministic chunk_id, so re-indexing the
same corpus overwrites in place instead of duplicating.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rag.config import PROJECT_ROOT, Settings
from rag.models import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, settings: Settings):
        if settings.qdrant_mode == "local":
            path = Path(settings.qdrant_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(url=settings.qdrant_url)

    def recreate_collection(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert_chunks(
        self, name: str, chunks: list[Chunk], vectors: np.ndarray, batch_size: int = 256
    ) -> None:
        assert len(chunks) == len(vectors)
        for start in range(0, len(chunks), batch_size):
            points = [
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=chunk.payload(),
                )
                for chunk, vector in zip(
                    chunks[start : start + batch_size], vectors[start : start + batch_size]
                )
            ]
            self.client.upsert(collection_name=name, points=points)

    def search(self, name: str, vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        result = self.client.query_points(
            collection_name=name, query=vector.tolist(), limit=top_k, with_payload=True
        )
        return [RetrievedChunk(score=p.score, payload=p.payload) for p in result.points]

    def count(self, name: str) -> int:
        return self.client.count(collection_name=name, exact=True).count

    def close(self) -> None:
        self.client.close()
