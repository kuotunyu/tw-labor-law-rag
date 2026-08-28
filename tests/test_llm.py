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


def test_gemini_output_includes_billable_usage(monkeypatch):
    """Catches thinking tokens being omitted from Gemini output cost."""
    llm = build_llm(settings_for("gemini"), model="gemini-test")
    response = SimpleNamespace(
        text="回答",
        prompt_feedback=None,
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=7,
        ),
    )
    monkeypatch.setattr(llm.client.models, "generate_content", lambda **_kwargs: response)

    output = llm.generate("system", "user")

    assert output.input_tokens == 100
    assert output.output_tokens == 27


def test_openai_output_includes_billable_usage(monkeypatch):
    llm = build_llm(settings_for("openai"), model="openai-test")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="回答", refusal=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=27),
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **_kwargs: response)

    output = llm.generate("system", "user")

    assert output.input_tokens == 100
    assert output.output_tokens == 27


def test_gemini_35_flash_lite_uses_minimal_thinking_level(monkeypatch):
    """Catches Gemini 3.x being sent the Gemini 2.5 thinking-budget contract."""
    from google.genai import types

    captured = {}
    llm = build_llm(
        settings_for("gemini"),
        model="gemini-3.5-flash-lite",
    )

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="回答", prompt_feedback=None, candidates=[])

    monkeypatch.setattr(llm.client.models, "generate_content", generate_content)

    llm.generate("system", "user")

    thinking = captured["config"].thinking_config
    assert thinking.thinking_level == types.ThinkingLevel.MINIMAL
    assert thinking.thinking_budget is None


def test_gemini_25_flash_keeps_zero_thinking_budget(monkeypatch):
    """Catches the Gemini 3.x compatibility change regressing Gemini 2.5."""
    captured = {}
    llm = build_llm(
        settings_for("gemini"),
        model="gemini-2.5-flash",
    )

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="回答", prompt_feedback=None, candidates=[])

    monkeypatch.setattr(llm.client.models, "generate_content", generate_content)

    llm.generate("system", "user")

    thinking = captured["config"].thinking_config
    assert thinking.thinking_budget == 0
    assert thinking.thinking_level is None


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


def test_build_llm_uses_request_key_without_mutating_owner_settings(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, model, timeout_seconds=None):
            captured.update(
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
            )

    monkeypatch.setattr(llm_module, "OpenAIAdapter", FakeOpenAI)
    settings = Settings(_env_file=None, openai_api_key="owner-key")

    llm_module.build_llm(
        settings,
        provider="openai",
        api_key="visitor-key",
        timeout_seconds=60.0,
    )

    assert captured == {
        "api_key": "visitor-key",
        "model": "gpt-5.6-luna",
        "timeout_seconds": 60.0,
    }
    assert settings.openai_api_key == "owner-key"


def test_openai_adapter_passes_request_timeout_to_sdk(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    OpenAIAdapter("visitor-key", "openai-test", timeout_seconds=60.0)

    assert captured == {"api_key": "visitor-key", "timeout": 60.0}


def test_gemini_adapter_converts_request_timeout_to_milliseconds(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("google.genai.Client", FakeClient)

    GeminiAdapter("visitor-key", "gemini-test", timeout_seconds=60.0)

    assert captured["api_key"] == "visitor-key"
    assert captured["http_options"].timeout == 60_000


def test_timeout_failure_has_specific_safe_reason_code():
    secret = "raw-provider-timeout-body"

    class ProviderTimeoutError(Exception):
        pass

    error = llm_module._normalized_provider_error(
        "gemini", ProviderTimeoutError(secret)
    )

    assert isinstance(error, ProviderOperationalError)
    assert error.reason_code == "timeout"
    assert secret not in str(error)
    assert secret not in repr(error)


def test_gemini_invalid_key_http_400_is_sanitized_auth_failure():
    from google.genai import errors

    secret = "visitor-key-must-not-escape"
    provider_error = errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": f"API key not valid. Credential: {secret}",
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    error = llm_module._normalized_provider_error("gemini", provider_error)

    assert isinstance(error, ProviderOperationalError)
    assert error.reason_code == "http_401"
    assert secret not in str(error)
    assert secret not in repr(error)


def test_gemini_non_auth_http_400_is_sanitized_operational_failure():
    from google.genai import errors

    provider_detail = "thinking budget is not supported"
    provider_error = errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": provider_detail,
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    error = llm_module._normalized_provider_error("gemini", provider_error)

    assert isinstance(error, ProviderOperationalError)
    assert error.reason_code == "http_400"
    assert provider_detail not in str(error)
    assert provider_detail not in repr(error)


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
