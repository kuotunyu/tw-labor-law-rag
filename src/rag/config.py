"""Application settings, loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default models per provider; overridable via provider-specific settings or
# the legacy GENERATION_MODEL setting.
DEFAULT_GENERATION_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.6-luna",
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "qwen3:8b",
}
PUBLIC_LLM_PROVIDERS = ("gemini", "openai")
DEFAULT_JUDGE_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:8b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "gemini", "ollama"] = "gemini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = ""  # legacy override for the active provider
    gemini_generation_model: str = ""
    openai_generation_model: str = ""
    llm_fallback_enabled: bool = True
    judge_model: str = ""  # empty → provider default
    llm_temperature: float = 0.0  # deterministic for reproducible evals

    # ── Embedding / Reranker ─────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"  # auto | cuda | cpu

    # ── Qdrant ───────────────────────────────────────
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: str = "storage/qdrant"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "labor_laws"

    # ── Retrieval pipeline ───────────────────────────
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    use_reranker: bool = True
    top_k_retrieve: int = 20  # candidates fed into RRF / reranker
    top_k_final: int = 5  # chunks handed to the LLM
    rrf_k: int = 60
    # First-layer gate: scores below this value are refused without an LLM call.
    # The 0.03 value was selected on the mini eval. On the 40-question formal
    # eval it passed all 30 answerable questions and directly rejected 9/10
    # unanswerable ones. The remaining high-scoring unanswerable question still
    # required the LLM refusal layer, so this is not a universal answerability
    # boundary. Recalibrate whenever the corpus, reranker, or query mix changes.
    rerank_score_threshold: float = 0.03

    # ── Chunking ─────────────────────────────────────
    chunking_strategy: Literal["fixed", "structure"] = "structure"
    chunk_size: int = 400  # characters (Chinese text)
    chunk_overlap: int = 80

    # ── Paths ────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    storage_dir: Path = PROJECT_ROOT / "storage"

    def generation_model_for(self, provider: str) -> str:
        if provider not in DEFAULT_GENERATION_MODELS:
            raise ValueError(f"unknown LLM provider: {provider}")
        provider_override = {
            "gemini": self.gemini_generation_model,
            "openai": self.openai_generation_model,
        }.get(provider, "")
        if provider_override:
            return provider_override
        if provider == self.llm_provider and self.generation_model:
            return self.generation_model
        return DEFAULT_GENERATION_MODELS[provider]

    @property
    def resolved_generation_model(self) -> str:
        return self.generation_model_for(self.llm_provider)

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or DEFAULT_JUDGE_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
