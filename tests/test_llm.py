"""build_llm() dispatch tests. Client constructors are lazy (no network call
on init) so these run offline with dummy keys — only .generate() would hit
the network, and we never call that here.
"""

import pytest

from rag.config import Settings
from rag.generation.llm import (
    AnthropicAdapter,
    GeminiAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    build_llm,
)


def settings_for(provider: str) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider=provider,
        anthropic_api_key="dummy",
        openai_api_key="dummy",
        gemini_api_key="dummy",
    )


def test_build_llm_anthropic():
    llm = build_llm(settings_for("anthropic"))
    assert isinstance(llm, AnthropicAdapter)
    assert llm.model == "claude-sonnet-5"


def test_build_llm_openai():
    llm = build_llm(settings_for("openai"))
    assert isinstance(llm, OpenAIAdapter)
    assert llm.model == "gpt-5.1"


def test_build_llm_gemini():
    llm = build_llm(settings_for("gemini"))
    assert isinstance(llm, GeminiAdapter)
    assert llm.model == "gemini-2.5-pro"


def test_build_llm_ollama():
    llm = build_llm(settings_for("ollama"))
    assert isinstance(llm, OllamaAdapter)
    assert llm.model == "qwen3:8b"


def test_build_llm_model_override():
    llm = build_llm(settings_for("gemini"), model="gemini-2.5-flash")
    assert llm.model == "gemini-2.5-flash"


def test_build_llm_provider_override_ignores_settings_provider():
    settings = settings_for("anthropic")
    llm = build_llm(settings, provider="gemini")
    assert isinstance(llm, GeminiAdapter)


def test_build_llm_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_llm(settings_for("anthropic"), provider="bedrock")
