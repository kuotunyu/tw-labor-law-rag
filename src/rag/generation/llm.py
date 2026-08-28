"""Thin adapters over Anthropic / OpenAI / Gemini / Ollama chat-completion APIs.

Each provider returns structured generation metadata while keeping provider
selection isolated from retrieval and answer assembly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag.config import Settings

# Gemini 2.5's "thinking" tokens are deducted from max_output_tokens too, so a
# low ceiling can silently truncate the visible answer after an invisible
# reasoning pass (observed on gemini-2.5-flash: 980/1024 tokens spent
# thinking, leaving 40 for the actual answer -> cut off mid-sentence). 2048
# gives every provider headroom for a multi-citation answer; see GeminiAdapter
# below for the flash-specific fix that avoids spending the budget on thinking.
DEFAULT_MAX_TOKENS = 2048


@dataclass(frozen=True)
class LLMOutput:
    text: str
    provider: str
    model: str
    fallback_used: bool = False
    fallback_from: str | None = None


class ProviderOperationalError(RuntimeError):
    def __init__(self, provider: str, reason_code: str):
        super().__init__(f"{provider} provider unavailable")
        self.provider = provider
        self.reason_code = reason_code


class ProviderPolicyError(RuntimeError):
    def __init__(self, provider: str, reason_code: str = "policy_rejection"):
        super().__init__(f"{provider} provider rejected the request")
        self.provider = provider
        self.reason_code = reason_code


_OPERATIONAL_STATUS_CODES = {401, 403, 404, 408, 409, 429, 500, 502, 503, 504}
_OPERATIONAL_CLASS_TOKENS = (
    "connection",
    "timeout",
    "authentication",
    "permission",
    "ratelimit",
    "servererror",
    "notfound",
    "serviceunavailable",
)
_POLICY_TOKENS = ("safety", "policy", "content_filter", "blocked")


def _normalized_provider_error(provider: str, exc: Exception) -> RuntimeError:
    class_name = type(exc).__name__.lower()
    message = str(exc).lower()
    if any(token in class_name or token in message for token in _POLICY_TOKENS):
        return ProviderPolicyError(provider)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        numeric_status = int(status)
    except (TypeError, ValueError):
        numeric_status = None
    if numeric_status in _OPERATIONAL_STATUS_CODES:
        return ProviderOperationalError(provider, f"http_{numeric_status}")
    if any(token in class_name for token in _OPERATIONAL_CLASS_TOKENS):
        return ProviderOperationalError(provider, "transport_or_service")
    return RuntimeError(f"unclassified {provider} provider failure")


def _provider_output(provider: str, model: str, text: str) -> LLMOutput:
    if not text or not text.strip():
        raise ProviderOperationalError(provider, "empty_response")
    return LLMOutput(text=text, provider=provider, model=model)


class LLMAdapter(Protocol):
    provider: str
    model: str

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMOutput: ...


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMOutput:
        from anthropic import AnthropicError

        try:
            resp = self.client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
        except AnthropicError as exc:
            raise _normalized_provider_error(self.provider, exc) from None
        text = "".join(block.text for block in resp.content if block.type == "text")
        return _provider_output(self.provider, self.model, text)


class OpenAIAdapter:
    """GPT-5-era compatibility notes (all discovered the hard way):
    - ``max_tokens`` is rejected; use ``max_completion_tokens``.
    - Reasoning models silently burn the whole token budget on hidden
      reasoning, returning an EMPTY string once ``max_completion_tokens``
      runs out — ``reasoning_effort="low"`` keeps that in check for
      RAG-synthesis/judging workloads that don't need deep reasoning.
    - Some models reject non-default ``temperature``.
    Unsupported parameters are detected from the API error once, then dropped
    for all subsequent calls on this adapter instance.
    """

    provider = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._unsupported_params: set[str] = set()

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMOutput:
        from openai import APIError, BadRequestError

        kwargs = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        optional = {"temperature": temperature, "reasoning_effort": "low"}
        for name, value in optional.items():
            if name not in self._unsupported_params:
                kwargs[name] = value

        while True:
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                message = str(exc)
                dropped = False
                for name in optional:
                    if name in kwargs and name in message:
                        self._unsupported_params.add(name)
                        kwargs.pop(name)
                        dropped = True
                        break
                if not dropped:
                    raise _normalized_provider_error(self.provider, exc) from None
            except APIError as exc:
                raise _normalized_provider_error(self.provider, exc) from None
            else:
                text = resp.choices[0].message.content or ""
                return _provider_output(self.provider, self.model, text)


class GeminiAdapter:
    provider = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMOutput:
        from google.genai import errors, types

        # RAG synthesis over supplied context doesn't need multi-step reasoning, so
        # turn thinking off on flash models (budget=0 is supported there) to spend
        # the whole token budget on the visible answer. gemini-2.5-pro requires a
        # non-zero thinking budget (would error on 0), so it's left at the default.
        thinking_config = types.ThinkingConfig(thinking_budget=0) if "flash" in self.model else None

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_config,
                ),
            )
        except errors.APIError as exc:
            raise _normalized_provider_error(self.provider, exc) from None
        return _provider_output(self.provider, self.model, resp.text or "")


_COMPLETE_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def sanitize_ollama_content(text: str) -> str:
    """Remove leaked Qwen-style thinking markup from user-visible content."""
    cleaned = _COMPLETE_THINK_BLOCK.sub("", text)
    unclosed = cleaned.find("<think>")
    if unclosed != -1:
        cleaned = cleaned[:unclosed]
    orphan = cleaned.find("</think>")
    if orphan != -1:
        cleaned = cleaned[orphan + len("</think>") :]
    return cleaned.lstrip()


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMOutput:
        import httpx

        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=180.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _normalized_provider_error(self.provider, exc) from None
        content = resp.json()["message"]["content"]
        return _provider_output(self.provider, self.model, sanitize_ollama_content(content))


def build_llm(settings: Settings, *, provider: str | None = None, model: str | None = None) -> LLMAdapter:
    """``provider``/``model`` overrides let eval scripts request e.g. a cross-provider judge."""
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        return AnthropicAdapter(settings.anthropic_api_key, model or settings.generation_model_for(provider))
    if provider == "openai":
        return OpenAIAdapter(settings.openai_api_key, model or settings.generation_model_for(provider))
    if provider == "gemini":
        return GeminiAdapter(settings.gemini_api_key, model or settings.generation_model_for(provider))
    if provider == "ollama":
        return OllamaAdapter(settings.ollama_base_url, model or settings.generation_model_for(provider))
    raise ValueError(f"unknown LLM provider: {provider}")
