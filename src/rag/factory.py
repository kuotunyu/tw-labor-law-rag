"""Wires concrete components (embedder, vector store, BM25 index, reranker,
LLM) into a :class:`~rag.retrieval.pipeline.RetrievalPipeline` /
:class:`~rag.generation.answerer.Answerer`, based on :class:`~rag.config.Settings`.

Centralizing this means every entry point — CLI scripts, eval/ablation
scripts, the API — asks for "a hybrid+rerank pipeline for strategy X"
the same way, instead of each reimplementing the wiring.
"""

from __future__ import annotations

from rag.config import Settings
from rag.generation.answerer import Answerer
from rag.generation.llm import LLMAdapter, build_llm
from rag.indexing.bm25_index import BM25Index
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.retrieval.pipeline import RetrievalPipeline
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import (
    BM25Retriever,
    VectorRetriever,
    bm25_path_for,
    build_retriever,
    collection_for,
)


def build_retrieval_pipeline(
    settings: Settings,
    embedder: BGEM3Embedder,
    store: VectorStore,
    *,
    strategy: str | None = None,
    mode: str | None = None,
    use_reranker: bool | None = None,
    reranker: Reranker | None = None,
) -> RetrievalPipeline:
    strategy = strategy or settings.chunking_strategy
    mode = mode or settings.retrieval_mode
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker

    vector_retriever = None
    if mode in ("vector", "hybrid"):
        vector_retriever = VectorRetriever(embedder, store, collection_for(settings, strategy))

    bm25_retriever = None
    if mode in ("bm25", "hybrid"):
        bm25_retriever = BM25Retriever(BM25Index.load(bm25_path_for(settings, strategy)))

    retriever = build_retriever(
        mode,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=settings.rrf_k,
        fetch_k=settings.top_k_retrieve,
    )

    active_reranker = None
    if use_reranker:
        active_reranker = reranker or Reranker(
            model_name=settings.reranker_model, device=settings.device
        )

    return RetrievalPipeline(
        retriever,
        active_reranker,
        top_k_retrieve=settings.top_k_retrieve,
        top_k_final=settings.top_k_final,
    )


def build_answerer(
    settings: Settings,
    embedder: BGEM3Embedder,
    store: VectorStore,
    *,
    strategy: str | None = None,
    mode: str | None = None,
    use_reranker: bool | None = None,
    reranker: Reranker | None = None,
    llm: LLMAdapter | None = None,
) -> Answerer:
    pipeline = build_retrieval_pipeline(
        settings,
        embedder,
        store,
        strategy=strategy,
        mode=mode,
        use_reranker=use_reranker,
        reranker=reranker,
    )
    return Answerer(
        pipeline,
        llm or build_llm(settings),
        refusal_threshold=settings.rerank_score_threshold,
        temperature=settings.llm_temperature,
    )
