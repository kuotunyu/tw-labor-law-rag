"""Offline provider-adapter contract tests using local fake SDK clients."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import rag.generation.llm as llm_module
from rag.config import Settings
from rag.generation.llm import (
    AnthropicAdapter,
    GeminiAdapter,
    LLMOutput,
    OllamaAdapter,
    OpenAIAdapter,
    ProviderOperationalError,
    ProviderPolicyError,
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


def test_llm_output_defaults_to_no_fallback():
    output = LLMOutput(text="回答", provider="gemini", model="gemini-test")

    assert output.text == "回答"
    assert output.fallback_used is False
    assert output.fallback_from is None


def test_llm_output_is_immutable():
    output = LLMOutput(text="回答", provider="gemini", model="gemini-test")

    with pytest.raises(FrozenInstanceError):
        output.text = "修改後答案"


def test_build_llm_uses_provider_specific_model():
    settings = settings_for("gemini")
    settings.gemini_generation_model = "gemini-specific"
    settings.openai_generation_model = "openai-specific"

    assert build_llm(settings, provider="gemini").model == "gemini-specific"
    assert build_llm(settings, provider="openai").model == "openai-specific"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "ollama"])
def test_build_llm_adapter_reports_its_fixed_provider(provider):
    assert build_llm(settings_for(provider)).provider == provider


def test_empty_provider_response_is_operational_failure(monkeypatch):
    llm = build_llm(settings_for("gemini"), model="gemini-test")
    monkeypatch.setattr(
        llm.client.models,
        "generate_content",
        lambda **_kwargs: SimpleNamespace(text=""),
    )

    with pytest.raises(ProviderOperationalError) as exc_info:
        llm.generate("system", "user")

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.reason_code == "empty_response"


def test_openai_structured_refusal_is_policy_error(monkeypatch):
    llm = build_llm(settings_for("openai"), model="openai-test")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=None, refusal="I cannot help with that."),
            )
        ]
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **_kwargs: response)

    with pytest.raises(ProviderPolicyError) as exc_info:
        llm.generate("system", "user")

    assert exc_info.value.provider == "openai"
    assert not isinstance(exc_info.value, ProviderOperationalError)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            "prompt_feedback",
            id="prompt-safety-block",
        ),
        pytest.param(
            "candidate_finish_reason",
            id="candidate-safety-finish",
        ),
        pytest.param(
            "candidate_recitation_finish_reason",
            id="candidate-recitation-finish",
        ),
        pytest.param(
            "candidate_image_recitation_finish_reason",
            id="candidate-image-recitation-finish",
        ),
        pytest.param(
            "candidate_safety_rating",
            id="candidate-safety-rating",
        ),
    ],
)
def test_gemini_structured_safety_block_is_policy_error(monkeypatch, response):
    from google.genai import types

    llm = build_llm(settings_for("gemini"), model="gemini-test")
    if response == "prompt_feedback":
        provider_response = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            )
        )
    elif response == "candidate_finish_reason":
        provider_response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]
        )
    elif response == "candidate_recitation_finish_reason":
        provider_response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.RECITATION)]
        )
    elif response == "candidate_image_recitation_finish_reason":
        provider_response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.IMAGE_RECITATION)]
        )
    else:
        provider_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    safety_ratings=[types.SafetyRating(blocked=True)]
                )
            ]
        )
    monkeypatch.setattr(
        llm.client.models,
        "generate_content",
        lambda **_kwargs: provider_response,
    )

    with pytest.raises(ProviderPolicyError) as exc_info:
        llm.generate("system", "user")

    assert exc_info.value.provider == "gemini"
    assert not isinstance(exc_info.value, ProviderOperationalError)


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

    output = adapter.generate("system", "user")

    assert output == LLMOutput("可見答案", "ollama", "qwen3:8b")
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


@pytest.mark.parametrize("status_code", [429, 503])
def test_http_provider_failures_are_operational(status_code):
    class ProviderSDKError(Exception):
        def __init__(self):
            self.status_code = status_code

    error = llm_module._normalized_provider_error("gemini", ProviderSDKError())

    assert isinstance(error, ProviderOperationalError)
    assert error.provider == "gemini"
    assert error.reason_code == f"http_{status_code}"


def test_safety_provider_failure_is_policy_error():
    class SafetyBlockedError(Exception):
        pass

    error = llm_module._normalized_provider_error("openai", SafetyBlockedError())

    assert isinstance(error, ProviderPolicyError)
    assert error.provider == "openai"
    assert error.reason_code == "policy_rejection"


def test_unknown_provider_failure_is_not_operational():
    error = llm_module._normalized_provider_error("gemini", ValueError("bad input"))

    assert type(error) is RuntimeError
    assert not isinstance(error, ProviderOperationalError)
