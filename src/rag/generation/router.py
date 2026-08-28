"""One-shot routing between the public Gemini and OpenAI generation providers."""

from collections.abc import Mapping
from dataclasses import replace

from rag.config import PUBLIC_LLM_PROVIDERS, Settings
from rag.generation.llm import (
    DEFAULT_MAX_TOKENS,
    LLMAdapter,
    LLMOutput,
    ProviderOperationalError,
    build_llm,
)


class RoutedLLM:
    """Use a primary adapter, recovering once through an operational fallback."""

    def __init__(self, primary: LLMAdapter, fallback: LLMAdapter | None = None):
        self.primary = primary
        self.fallback = fallback
        self.primary_provider = primary.provider

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMOutput:
        try:
            return self.primary.generate(system, user, temperature, max_tokens)
        except ProviderOperationalError:
            if self.fallback is None:
                raise
            output = self.fallback.generate(system, user, temperature, max_tokens)
            return replace(
                output,
                fallback_used=True,
                fallback_from=self.primary.provider,
            )


def build_routed_llm(
    settings: Settings,
    primary_provider: str,
    adapters: Mapping[str, LLMAdapter] | None = None,
) -> RoutedLLM:
    """Build a public-provider router, optionally using test-injected adapters."""
    if primary_provider not in PUBLIC_LLM_PROVIDERS:
        raise ValueError(f"unknown public LLM provider: {primary_provider}")

    def adapter_for(provider: str) -> LLMAdapter:
        if adapters is not None:
            return adapters[provider]
        return build_llm(settings, provider=provider)

    primary = adapter_for(primary_provider)
    fallback = None
    alternate_provider = next(
        provider for provider in PUBLIC_LLM_PROVIDERS if provider != primary_provider
    )
    provider_keys = {
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }
    if settings.llm_fallback_enabled and provider_keys[alternate_provider].strip():
        fallback = adapter_for(alternate_provider)

    return RoutedLLM(primary, fallback)
