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


class ProviderRouteOperationalError(ProviderOperationalError):
    """Sanitized operational failure with immutable public route context."""

    _IMMUTABLE_FIELDS = {"provider", "reason_code", "_attempted_providers"}

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_FIELDS and hasattr(self, name):
            raise AttributeError(f"{name} is read-only")
        super().__setattr__(name, value)

    def __init__(self, attempted_providers: tuple[str, ...]):
        if not attempted_providers or any(
            provider not in PUBLIC_LLM_PROVIDERS for provider in attempted_providers
        ):
            raise ValueError("attempted providers must use the public provider catalog")
        super().__init__(attempted_providers[-1], "route_operational_failure")
        self._attempted_providers = tuple(attempted_providers)

    @property
    def attempted_providers(self) -> tuple[str, ...]:
        return self._attempted_providers

    @property
    def fallback_attempted(self) -> bool:
        return len(self._attempted_providers) > 1


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
                raise ProviderRouteOperationalError((self.primary.provider,)) from None
            try:
                output = self.fallback.generate(system, user, temperature, max_tokens)
            except ProviderOperationalError:
                raise ProviderRouteOperationalError(
                    (self.primary.provider, self.fallback.provider)
                ) from None
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
