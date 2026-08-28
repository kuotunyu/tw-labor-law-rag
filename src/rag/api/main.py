"""FastAPI front-end: POST /query, GET /health, GET /models.

Retrieval components are loaded once at startup. Provider adapters and routed
LLMs are created lazily and cached by provider; answerers are cached per
(strategy, mode, use_reranker, provider) combination.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from threading import RLock
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import PUBLIC_LLM_PROVIDERS, Settings, get_settings
from rag.factory import build_answerer
from rag.generation.answerer import Answerer
from rag.generation.llm import (
    LLMAdapter,
    ProviderOperationalError,
    ProviderPolicyError,
    build_llm,
)
from rag.generation.router import RoutedLLM, build_routed_llm
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.retrieval.reranker import Reranker


class AppState:
    settings: Settings
    embedder: BGEM3Embedder
    store: VectorStore
    reranker: Reranker

    def __init__(self) -> None:
        self._adapter_cache: dict[str, LLMAdapter] = {}
        self._routed_llm_cache: dict[str, RoutedLLM] = {}
        self._answerer_cache: dict[tuple[str, str, bool, str], Answerer] = {}
        self._cache_lock = RLock()

    def clear_caches(self) -> None:
        with self._cache_lock:
            self._adapter_cache.clear()
            self._routed_llm_cache.clear()
            self._answerer_cache.clear()

    def resolve_provider_catalog(self) -> tuple[str | None, tuple[str, ...]]:
        provider_keys = {
            "gemini": self.settings.gemini_api_key,
            "openai": self.settings.openai_api_key,
        }
        available = tuple(
            provider
            for provider in PUBLIC_LLM_PROVIDERS
            if provider_keys[provider].strip()
        )
        default = self.settings.llm_provider
        if default not in available:
            default = available[0] if available else None
        return default, available

    def is_provider_available(self, provider: str) -> bool:
        _, available = self.resolve_provider_catalog()
        return provider in available

    def _get_adapter(self, provider: str) -> LLMAdapter:
        if provider not in self._adapter_cache:
            self._adapter_cache[provider] = build_llm(self.settings, provider=provider)
        return self._adapter_cache[provider]

    def _get_routed_llm(self, provider: str) -> RoutedLLM:
        if not self.is_provider_available(provider):
            raise ValueError(f"unavailable public LLM provider: {provider}")
        if provider not in self._routed_llm_cache:
            adapter_providers = {provider}
            if self.settings.llm_fallback_enabled:
                adapter_providers.update(
                    candidate
                    for candidate in PUBLIC_LLM_PROVIDERS
                    if self.is_provider_available(candidate)
                )
            adapters = {
                candidate: self._get_adapter(candidate) for candidate in adapter_providers
            }
            self._routed_llm_cache[provider] = build_routed_llm(
                self.settings,
                provider,
                adapters=adapters,
            )
        return self._routed_llm_cache[provider]

    def get_answerer(
        self,
        strategy: str,
        mode: str,
        use_reranker: bool,
        provider: str,
    ) -> Answerer:
        key = (strategy, mode, use_reranker, provider)
        with self._cache_lock:
            if key not in self._answerer_cache:
                self._answerer_cache[key] = build_answerer(
                    self.settings,
                    self.embedder,
                    self.store,
                    strategy=strategy,
                    mode=mode,
                    use_reranker=use_reranker,
                    reranker=self.reranker,
                    llm=self._get_routed_llm(provider),
                )
            return self._answerer_cache[key]


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    state.clear_caches()
    state.embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    state.store = VectorStore(settings)
    state.reranker = Reranker(model_name=settings.reranker_model, device=settings.device)
    yield
    state.store.close()


app = FastAPI(title="繁體中文 Hybrid RAG 知識問答系統", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    provider: Optional[Literal["gemini", "openai"]] = None
    strategy: Optional[Literal["structure", "fixed"]] = None
    mode: Optional[Literal["vector", "bm25", "hybrid"]] = None
    use_reranker: Optional[bool] = None


class SourceOut(BaseModel):
    index: int
    doc: str
    article: str
    content: str


class RetrievalHitOut(BaseModel):
    citation: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    retrieval_hits: list[RetrievalHitOut]
    strategy: str
    mode: str
    use_reranker: bool
    provider: Optional[str]
    model: Optional[str]
    # Optional keeps payload construction compatible with clients/tests written
    # before refusal-layer observability was added.
    refusal_stage: Optional[Literal["no_hits", "threshold", "llm"]] = None
    generation_called: bool = True
    requested_provider: Optional[str] = None
    fallback_used: bool = False
    fallback_from: Optional[str] = None


class ModelOut(BaseModel):
    provider: Literal["gemini", "openai"]
    model: str


class ModelsResponse(BaseModel):
    default_provider: Optional[Literal["gemini", "openai"]]
    providers: list[ModelOut]


@app.get("/health")
def health():
    settings = state.settings
    info = {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "generation_model": settings.resolved_generation_model,
        "qdrant_mode": settings.qdrant_mode,
    }
    for strategy in ("structure", "fixed"):
        collection = f"{settings.collection_name}_{strategy}"
        try:
            info[f"collection_{strategy}_points"] = state.store.count(collection)
        except Exception:
            info[f"collection_{strategy}_points"] = None
    return info


@app.get("/models", response_model=ModelsResponse)
def models():
    settings = state.settings
    default_provider, available_providers = state.resolve_provider_catalog()
    return {
        "default_provider": default_provider,
        "providers": [
            {
                "provider": provider,
                "model": settings.generation_model_for(provider),
            }
            for provider in PUBLIC_LLM_PROVIDERS
            if provider in available_providers
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = state.settings
    strategy = req.strategy or settings.chunking_strategy
    mode = req.mode or settings.retrieval_mode
    use_reranker = settings.use_reranker if req.use_reranker is None else req.use_reranker

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    default_provider, available_providers = state.resolve_provider_catalog()
    provider = req.provider or default_provider
    if provider is None:
        raise HTTPException(status_code=503, detail="generation_unavailable")
    if req.provider is not None and provider not in available_providers:
        raise HTTPException(status_code=400, detail="provider_unavailable")

    try:
        answerer = state.get_answerer(strategy, mode, use_reranker, provider)
        result = answerer.answer(req.question)
    except ProviderOperationalError as exc:
        raise HTTPException(status_code=502, detail="generation_unavailable") from exc
    except ProviderPolicyError as exc:
        raise HTTPException(status_code=422, detail="generation_rejected") from exc

    return QueryResponse(
        answer=result.text,
        refused=result.refused,
        sources=[SourceOut(**s) for s in result.sources],
        retrieval_hits=[
            RetrievalHitOut(citation=h.citation, score=h.score) for h in result.retrieval.hits
        ],
        strategy=strategy,
        mode=mode,
        use_reranker=use_reranker,
        provider=result.provider,
        model=result.model,
        refusal_stage=result.refusal_stage,
        generation_called=result.generation_called,
        requested_provider=result.requested_provider,
        fallback_used=result.fallback_used,
        fallback_from=result.fallback_from,
    )
