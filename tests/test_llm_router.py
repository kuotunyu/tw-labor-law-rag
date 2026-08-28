"""One-shot cross-provider routing contracts using local adapter fakes."""

import pytest

from rag.config import Settings
from rag.generation.llm import LLMOutput, ProviderOperationalError, ProviderPolicyError
from rag.generation.router import (
    ProviderRouteOperationalError,
    RoutedLLM,
    build_routed_llm,
)


class FakeAdapter:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        output: str | None = None,
        error: Exception | None = None,
    ):
        self.provider = provider
        self.model = model
        self.output = output
        self.error = error
        self.calls: list[tuple[str, str, float, int]] = []

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMOutput:
        self.calls.append((system, user, temperature, max_tokens))
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return LLMOutput(self.output, self.provider, self.model)


def router_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "llm_provider": "gemini",
        "gemini_api_key": "gemini-key",
        "openai_api_key": "openai-key",
    }
    values.update(overrides)
    return Settings(**values)


def test_primary_success_does_not_call_fallback():
    """Catches routing every request to the fallback despite primary success."""
    primary = FakeAdapter("gemini", "gemini-test", output="Gemini answer")
    fallback = FakeAdapter("openai", "openai-test", output="GPT answer")

    result = RoutedLLM(primary, fallback).generate("system", "user")

    assert result == LLMOutput("Gemini answer", "gemini", "gemini-test")
    assert primary.calls == [("system", "user", 0.0, 2048)]
    assert fallback.calls == []


def test_operational_failure_falls_back_once():
    """Catches omission of the one allowed operational recovery attempt."""
    primary = FakeAdapter(
        "gemini",
        "gemini-test",
        error=ProviderOperationalError("gemini", "http_503"),
    )
    fallback = FakeAdapter("openai", "openai-test", output="GPT answer")

    result = RoutedLLM(primary, fallback).generate("system", "user")

    assert result == LLMOutput(
        "GPT answer",
        "openai",
        "openai-test",
        fallback_used=True,
        fallback_from="gemini",
    )
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


@pytest.mark.parametrize(
    "error",
    [ProviderPolicyError("gemini"), ValueError("programming bug")],
)
def test_non_operational_errors_never_fallback(error):
    """Catches policy or programming failures being hidden by provider switching."""
    primary = FakeAdapter("gemini", "gemini-test", error=error)
    fallback = FakeAdapter("openai", "openai-test", output="GPT answer")

    with pytest.raises(type(error)):
        RoutedLLM(primary, fallback).generate("system", "user")

    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_dual_operational_failure_makes_exactly_two_calls():
    """Catches retry loops after the single fallback has already failed."""
    primary = FakeAdapter(
        "gemini", "gemini-test", error=ProviderOperationalError("gemini", "http_503")
    )
    fallback = FakeAdapter(
        "openai", "openai-test", error=ProviderOperationalError("openai", "http_429")
    )

    with pytest.raises(ProviderRouteOperationalError) as exc_info:
        RoutedLLM(primary, fallback).generate("system", "user")

    assert isinstance(exc_info.value, ProviderOperationalError)
    assert exc_info.value.attempted_providers == ("gemini", "openai")
    assert exc_info.value.fallback_attempted is True
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_primary_only_operational_failure_has_safe_immutable_route_context():
    """Catches losing route-attempt context or retaining raw provider details."""
    secret = "sdk-message-with-private-value"
    primary = FakeAdapter(
        "gemini", "gemini-test", error=ProviderOperationalError("gemini", secret)
    )

    with pytest.raises(ProviderRouteOperationalError) as exc_info:
        RoutedLLM(primary).generate("system", "user")

    error = exc_info.value
    assert isinstance(error, ProviderOperationalError)
    assert error.attempted_providers == ("gemini",)
    assert error.fallback_attempted is False
    assert secret not in str(error)
    assert secret not in repr(error)
    with pytest.raises(AttributeError):
        error.attempted_providers = ("openai",)
    with pytest.raises(AttributeError):
        error.provider = "openai"
    with pytest.raises(AttributeError):
        error.reason_code = "http_429"


@pytest.mark.parametrize(
    "fallback_error",
    [ProviderPolicyError("openai"), ValueError("programming bug")],
)
def test_fallback_policy_and_programming_errors_remain_unwrapped(fallback_error):
    """Catches route context masking non-operational fallback failures."""
    primary = FakeAdapter(
        "gemini", "gemini-test", error=ProviderOperationalError("gemini", "http_503")
    )
    fallback = FakeAdapter("openai", "openai-test", error=fallback_error)

    with pytest.raises(type(fallback_error)) as exc_info:
        RoutedLLM(primary, fallback).generate("system", "user")

    assert exc_info.value is fallback_error


def test_openai_primary_uses_gemini_as_the_single_fallback():
    """Catches a router that only supports Gemini-to-OpenAI recovery."""
    primary = FakeAdapter(
        "openai", "openai-test", error=ProviderOperationalError("openai", "http_503")
    )
    fallback = FakeAdapter("gemini", "gemini-test", output="Gemini answer")

    result = RoutedLLM(primary, fallback).generate("system", "user")

    assert result == LLMOutput(
        "Gemini answer",
        "gemini",
        "gemini-test",
        fallback_used=True,
        fallback_from="openai",
    )
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_build_routed_llm_uses_the_other_public_provider_when_enabled():
    """Catches selecting the configured primary twice rather than its alternate."""
    gemini = FakeAdapter("gemini", "gemini-test", output="Gemini answer")
    openai = FakeAdapter("openai", "openai-test", output="GPT answer")

    routed = build_routed_llm(
        router_settings(llm_fallback_enabled=True),
        "gemini",
        adapters={"gemini": gemini, "openai": openai},
    )

    assert routed.primary is gemini
    assert routed.fallback is openai
    assert routed.primary_provider == "gemini"


@pytest.mark.parametrize(
    "settings",
    [
        router_settings(llm_fallback_enabled=False),
        router_settings(openai_api_key=""),
    ],
)
def test_build_routed_llm_omits_fallback_when_disabled_or_unconfigured(settings):
    """Catches building an alternate adapter without explicit enabled credentials."""
    gemini = FakeAdapter("gemini", "gemini-test", output="Gemini answer")
    openai = FakeAdapter("openai", "openai-test", output="GPT answer")

    routed = build_routed_llm(
        settings,
        "gemini",
        adapters={"gemini": gemini, "openai": openai},
    )

    assert routed.primary is gemini
    assert routed.fallback is None


def test_build_routed_llm_treats_whitespace_fallback_key_as_unconfigured():
    """Catches disagreement between API discovery and router key availability."""
    gemini = FakeAdapter("gemini", "gemini-test", output="Gemini answer")

    routed = build_routed_llm(
        router_settings(openai_api_key="   "),
        "gemini",
        adapters={"gemini": gemini},
    )

    assert routed.primary is gemini
    assert routed.fallback is None


def test_build_routed_llm_rejects_non_public_primary_provider():
    """Catches exposing a router for providers outside the dual public contract."""
    with pytest.raises(ValueError, match="unknown public LLM provider: anthropic"):
        build_routed_llm(router_settings(), "anthropic", adapters={})
