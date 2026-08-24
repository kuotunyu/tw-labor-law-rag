"""FastAPI front-end: POST /query, GET /health.

Heavy components (embedder, vector store, reranker, LLM client) are loaded
once at startup and reused across requests. Answerer instances are cached per
(strategy, mode, use_reranker) combo so the UI can let a user flip those
settings — for live ablation demos — without reloading any model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import Settings, get_settings
from rag.factory import build_answerer
from rag.generation.answerer import Answerer
from rag.generation.llm import LLMAdapter, build_llm
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.retrieval.reranker import Reranker


class AppState:
    settings: Settings
    embedder: BGEM3Embedder
    store: VectorStore
    reranker: Reranker
    llm: LLMAdapter

    def __init__(self) -> None:
        self._answerer_cache: dict[tuple[str, str, bool], Answerer] = {}

    def get_answerer(self, strategy: str, mode: str, use_reranker: bool) -> Answerer:
        key = (strategy, mode, use_reranker)
        if key not in self._answerer_cache:
            self._answerer_cache[key] = build_answerer(
                self.settings,
                self.embedder,
                self.store,
                strategy=strategy,
                mode=mode,
                use_reranker=use_reranker,
                reranker=self.reranker,
                llm=self.llm,
            )
        return self._answerer_cache[key]


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    state.embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    state.store = VectorStore(settings)
    state.reranker = Reranker(model_name=settings.reranker_model, device=settings.device)
    state.llm = build_llm(settings)
    yield
    state.store.close()


app = FastAPI(title="繁體中文 Hybrid RAG 知識問答系統", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
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
    provider: str
    model: str
    # Optional keeps payload construction compatible with clients/tests written
    # before refusal-layer observability was added.
    refusal_stage: Optional[Literal["no_hits", "threshold", "llm"]] = None


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


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = state.settings
    strategy = req.strategy or settings.chunking_strategy
    mode = req.mode or settings.retrieval_mode
    use_reranker = settings.use_reranker if req.use_reranker is None else req.use_reranker

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    answerer = state.get_answerer(strategy, mode, use_reranker)
    try:
        result = answerer.answer(req.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

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
        provider=settings.llm_provider,
        model=settings.resolved_generation_model,
        refusal_stage=result.refusal_stage,
    )
