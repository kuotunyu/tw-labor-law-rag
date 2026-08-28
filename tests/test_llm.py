"""build_llm() dispatch tests. Client constructors are lazy (no network call
on init) so these run offline with dummy keys — only .generate() would hit
the network, and we never call that here.
"""

import pytest

import rag.generation.llm as llm_module
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
    assert llm.model == "gpt-5.6-luna"


def test_build_llm_gemini():
    llm = build_llm(settings_for("gemini"))
    assert isinstance(llm, GeminiAdapter)
    assert llm.model == "gemini-3.5-flash-lite"


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("正常答案", "正常答案"),
        ("<think>private reasoning</think>\n正常答案", "正常答案"),
        ("前言<think>unfinished reasoning", "前言"),
        ("private reasoning</think>\n正常答案", "正常答案"),
        ("<think>one</think><think>two</think>正常答案", "正常答案"),
    ],
)
def test_sanitize_ollama_content(raw, expected):
    assert llm_module.sanitize_ollama_content(raw) == expected


def test_ollama_generate_disables_thinking_and_sanitizes_content(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "qwen3:8b",
                "created_at": "2026-08-28T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "<think>private</think>\n可見答案",
                    "thinking": "",
                },
                "done": True,
            }

    def fake_post(url, *, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("httpx.post", fake_post)
    adapter = OllamaAdapter("http://localhost:11434/", "qwen3:8b")

    assert adapter.generate("system", "user") == "可見答案"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False


def test_ollama_generate_keeps_malformed_response_visible(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "qwen3:8b",
                "created_at": "2026-08-28T00:00:00Z",
                "message": {},
                "done": True,
            }

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())

    with pytest.raises(KeyError):
        OllamaAdapter("http://localhost:11434", "qwen3:8b").generate(
            "system", "user"
        )
