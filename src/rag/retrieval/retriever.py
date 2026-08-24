"""Retrieval front-ends: dense vector, BM25 keyword, and RRF-fused hybrid.

All three implement the same ``retrieve(query, top_k) -> list[RetrievedChunk]``
interface, so :mod:`rag.retrieval.pipeline` and the ablation study can swap
between them purely via config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.config import Settings
from rag.indexing.bm25_index import BM25Index
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.models import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion


def collection_for(settings: Settings, strategy: str) -> str:
    """Each chunking strategy gets its own Qdrant collection, e.g. ``labor_laws_structure``."""
    return f"{settings.collection_name}_{strategy}"


def bm25_path_for(settings: Settings, strategy: str) -> Path:
    return settings.storage_dir / f"bm25_{strategy}.pkl"


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class VectorRetriever:
    def __init__(self, embedder: BGEM3Embedder, store: VectorStore, collection: str):
        self.embedder = embedder
        self.store = store
        self.collection = collection

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vector = self.embedder.encode_query(query)
        return self.store.search(self.collection, vector, top_k)


class BM25Retriever:
    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.index.search(query, top_k)


class HybridRetriever:
    """BM25 + vector, each fetched independently and fused with RRF."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        fetch_k: int | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k  # candidates pulled from each side before fusion; defaults to top_k

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        fetch_k = self.fetch_k or top_k
        vec_hits = self.vector_retriever.retrieve(query, top_k=fetch_k)
        bm25_hits = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        fused = reciprocal_rank_fusion([vec_hits, bm25_hits], k=self.rrf_k)
        return fused[:top_k]


def build_retriever(
    mode: str,
    *,
    vector_retriever: VectorRetriever | None = None,
    bm25_retriever: BM25Retriever | None = None,
    rrf_k: int = 60,
    fetch_k: int | None = None,
) -> Retriever:
    """Factory used by ask.py / the eval scripts / the ablation study."""
    if mode == "vector":
        if vector_retriever is None:
            raise ValueError("mode='vector' requires vector_retriever")
        return vector_retriever
    if mode == "bm25":
        if bm25_retriever is None:
            raise ValueError("mode='bm25' requires bm25_retriever")
        return bm25_retriever
    if mode == "hybrid":
        if vector_retriever is None or bm25_retriever is None:
            raise ValueError("mode='hybrid' requires both vector_retriever and bm25_retriever")
        return HybridRetriever(vector_retriever, bm25_retriever, rrf_k=rrf_k, fetch_k=fetch_k)
    raise ValueError(f"unknown retrieval mode: {mode}")
