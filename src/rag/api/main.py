"""FastAPI front-end: POST /query, GET /health, GET /models.

Retrieval components are loaded once at startup. Provider adapters and routed
LLMs are created lazily and cached by provider; answerers are cached per
(strategy, mode, use_reranker, provider) combination.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from threading import RLock
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag.api.byok import (
    BYOK_MAX_KEY_CHARS,
    ByokConcurrencyGate,
    ByokSessionManager,
    DemoBusy,
    InvalidDemoSession,
    SessionCapacityExceeded,
    SessionQuotaExceeded,
)
from rag.config import PUBLIC_LLM_PROVIDERS, Settings, get_settings
from rag.factory import build_answerer
from rag.generation.answerer import Answerer
from rag.generation.llm import (
    LLMAdapter,
    ProviderOperationalError,
    ProviderPolicyError,
    build_llm,
)
from rag.generation.router import (
    ProviderRouteOperationalError,
    RoutedLLM,
    build_routed_llm,
)
from rag.indexing.bm25_index import BM25Index
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
        self._byok_bm25_indexes: dict[str, BM25Index] = {}
        self.byok_sessions: ByokSessionManager | None = None
        self.byok_gate: ByokConcurrencyGate | None = None
        self._cache_lock = RLock()

    def clear_caches(self) -> None:
        with self._cache_lock:
            self._adapter_cache.clear()
            self._routed_llm_cache.clear()
            self._answerer_cache.clear()

    def resolve_provider_catalog(self) -> tuple[str | None, tuple[str, ...]]:
        if self.settings.public_byok_enabled:
            default = self.settings.llm_provider
            if default not in PUBLIC_LLM_PROVIDERS:
                default = PUBLIC_LLM_PROVIDERS[0]
            return default, PUBLIC_LLM_PROVIDERS
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

    def configure_runtime(self, settings: Settings) -> None:
        self._byok_bm25_indexes = {}
        self.byok_sessions = None
        self.byok_gate = None
        if not settings.public_byok_enabled:
            return
        if (
            settings.qdrant_mode != "server"
            or not settings.qdrant_api_key.get_secret_value().strip()
            or not settings.session_signing_secret.get_secret_value().strip()
        ):
            raise RuntimeError("public BYOK runtime configuration is invalid")
        try:
            for strategy in ("structure", "fixed"):
                collection = f"{settings.collection_name}_{strategy}"
                index = BM25Index.from_payloads(self.store.scroll_payloads(collection))
                if len(index) != self.store.count(collection):
                    raise RuntimeError
                self._byok_bm25_indexes[strategy] = index
        except Exception:
            self._byok_bm25_indexes = {}
            raise RuntimeError("Qdrant BM25 bootstrap failed") from None
        self.byok_sessions = ByokSessionManager(
            secret=settings.session_signing_secret.get_secret_value(),
            query_limit=settings.byok_session_query_limit,
            ttl_seconds=settings.byok_session_ttl_seconds,
            max_tracked_sessions=settings.byok_max_tracked_sessions,
        )
        self.byok_gate = ByokConcurrencyGate(settings.byok_max_concurrency)

    def get_byok_answerer(
        self,
        strategy: str,
        mode: str,
        use_reranker: bool,
        provider: str,
        api_key: str,
    ) -> Answerer:
        if provider not in PUBLIC_LLM_PROVIDERS:
            raise ValueError(f"unavailable public LLM provider: {provider}")
        adapter = build_llm(
            self.settings,
            provider=provider,
            api_key=api_key,
            timeout_seconds=self.settings.byok_request_timeout_seconds,
        )
        return build_answerer(
            self.settings,
            self.embedder,
            self.store,
            strategy=strategy,
            mode=mode,
            use_reranker=use_reranker,
            reranker=self.reranker,
            llm=adapter,
            bm25_index=self._byok_bm25_indexes.get(strategy),
        )


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    state.clear_caches()
    state.embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    state.store = VectorStore(settings)
    state.reranker = Reranker(
        model_name=settings.reranker_model,
        model_revision=settings.reranker_model_revision,
        device=settings.device,
    )
    try:
        state.configure_runtime(settings)
        yield
    finally:
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
    requires_api_key: bool = False
    session_query_limit: Optional[int] = None


class SessionResponse(BaseModel):
    token: str
    query_limit: int


@app.get("/health")
def health():
    settings = state.settings
    default_provider, available_providers = state.resolve_provider_catalog()
    info = {
        "status": "ok" if available_providers else "degraded",
        "default_provider": default_provider,
        "available_providers": list(available_providers),
        "llm_provider": default_provider,
        "generation_model": (
            settings.generation_model_for(default_provider)
            if default_provider is not None
            else None
        ),
        "qdrant_mode": settings.qdrant_mode,
    }
    for strategy in ("structure", "fixed"):
        collection = f"{settings.collection_name}_{strategy}"
        try:
            info[f"collection_{strategy}_points"] = state.store.count(collection)
        except Exception:
            info[f"collection_{strategy}_points"] = None
    if settings.public_byok_enabled and any(
        not info[f"collection_{strategy}_points"]
        for strategy in ("structure", "fixed")
    ):
        info["status"] = "degraded"
    return info


@app.get("/models", response_model=ModelsResponse)
def models():
    settings = state.settings
    default_provider, available_providers = state.resolve_provider_catalog()
    response = {
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
    if settings.public_byok_enabled:
        response.update(
            requires_api_key=True,
            session_query_limit=settings.byok_session_query_limit,
        )
    return response


@app.post("/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    settings = state.settings
    if not settings.public_byok_enabled:
        raise HTTPException(status_code=404, detail="byok_not_enabled")
    if state.byok_sessions is None:
        raise HTTPException(status_code=503, detail="runtime_not_ready")
    try:
        token = state.byok_sessions.issue()
    except SessionCapacityExceeded as exc:
        raise HTTPException(status_code=429, detail="session_capacity_exceeded") from exc
    return SessionResponse(token=token, query_limit=settings.byok_session_query_limit)


def _query_response(result, *, strategy: str, mode: str, use_reranker: bool) -> QueryResponse:
    return QueryResponse(
        answer=result.text,
        refused=result.refused,
        sources=[SourceOut(**source) for source in result.sources],
        retrieval_hits=[
            RetrievalHitOut(citation=hit.citation, score=hit.score)
            for hit in result.retrieval.hits
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


def _raise_byok_provider_error(exc: ProviderOperationalError) -> None:
    if exc.reason_code in {"http_401", "http_403"}:
        raise HTTPException(status_code=401, detail="provider_key_rejected") from exc
    if exc.reason_code == "http_429":
        raise HTTPException(status_code=429, detail="provider_rate_limited") from exc
    if exc.reason_code in {"timeout", "http_504"}:
        raise HTTPException(status_code=504, detail="provider_timeout") from exc
    raise HTTPException(status_code=502, detail="generation_unavailable") from exc


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    provider_api_key: Annotated[
        Optional[str], Header(alias="X-Provider-Api-Key")
    ] = None,
    demo_session: Annotated[Optional[str], Header(alias="X-Demo-Session")] = None,
) -> QueryResponse:
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

    if settings.public_byok_enabled:
        if len(req.question) > settings.byok_max_question_chars:
            raise HTTPException(status_code=400, detail="question_too_long")
        visitor_key = provider_api_key.strip() if provider_api_key is not None else ""
        if not visitor_key:
            raise HTTPException(status_code=401, detail="provider_api_key_required")
        if len(visitor_key) > BYOK_MAX_KEY_CHARS:
            raise HTTPException(status_code=400, detail="provider_api_key_too_long")
        if not demo_session or state.byok_sessions is None or state.byok_gate is None:
            raise HTTPException(status_code=401, detail="invalid_demo_session")
        try:
            state.byok_sessions.consume(demo_session)
        except InvalidDemoSession as exc:
            raise HTTPException(status_code=401, detail="invalid_demo_session") from exc
        except SessionQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail="session_quota_exceeded") from exc
        try:
            with state.byok_gate.acquire():
                answerer = state.get_byok_answerer(
                    strategy,
                    mode,
                    use_reranker,
                    provider,
                    visitor_key,
                )
                result = answerer.answer(req.question)
        except DemoBusy as exc:
            raise HTTPException(status_code=429, detail="demo_busy") from exc
        except ProviderOperationalError as exc:
            _raise_byok_provider_error(exc)
        except ProviderPolicyError as exc:
            raise HTTPException(status_code=422, detail="generation_rejected") from exc
        return _query_response(
            result,
            strategy=strategy,
            mode=mode,
            use_reranker=use_reranker,
        )

    try:
        answerer = state.get_answerer(strategy, mode, use_reranker, provider)
        result = answerer.answer(req.question)
    except ProviderOperationalError as exc:
        status_code = (
            502
            if isinstance(exc, ProviderRouteOperationalError) and exc.fallback_attempted
            else 503
        )
        raise HTTPException(
            status_code=status_code, detail="generation_unavailable"
        ) from exc
    except ProviderPolicyError as exc:
        raise HTTPException(status_code=422, detail="generation_rejected") from exc

    return _query_response(
        result,
        strategy=strategy,
        mode=mode,
        use_reranker=use_reranker,
    )
