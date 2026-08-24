"""Thin adapters over Anthropic / OpenAI / Gemini / Ollama chat-completion APIs.

Deliberately minimal: one ``generate(system, user) -> str`` method per
provider, so swapping ``LLM_PROVIDER`` never touches the retrieval or
answer-assembly logic that calls it.
"""

from __future__ import annotations

from typing import Protocol

from rag.config import Settings

# Gemini 2.5's "thinking" tokens are deducted from max_output_tokens too, so a
# low ceiling can silently truncate the visible answer after an invisible
# reasoning pass (observed on gemini-2.5-flash: 980/1024 tokens spent
# thinking, leaving 40 for the actual answer -> cut off mid-sentence). 2048
# gives every provider headroom for a multi-citation answer; see GeminiAdapter
# below for the flash-specific fix that avoids spending the budget on thinking.
DEFAULT_MAX_TOKENS = 2048


class LLMAdapter(Protocol):
    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str: ...


class AnthropicAdapter:
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


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

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._unsupported_params: set[str] = set()

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        from openai import BadRequestError

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
                return resp.choices[0].message.content or ""
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
                    raise


class GeminiAdapter:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        from google.genai import types

        # RAG synthesis over supplied context doesn't need multi-step reasoning, so
        # turn thinking off on flash models (budget=0 is supported there) to spend
        # the whole token budget on the visible answer. gemini-2.5-pro requires a
        # non-zero thinking budget (would error on 0), so it's left at the default.
        thinking_config = types.ThinkingConfig(thinking_budget=0) if "flash" in self.model else None

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
        return resp.text or ""


class OllamaAdapter:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def build_llm(settings: Settings, *, provider: str | None = None, model: str | None = None) -> LLMAdapter:
    """``provider``/``model`` overrides let eval scripts request e.g. a cross-provider judge."""
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        return AnthropicAdapter(settings.anthropic_api_key, model or settings.resolved_generation_model)
    if provider == "openai":
        return OpenAIAdapter(settings.openai_api_key, model or settings.resolved_generation_model)
    if provider == "gemini":
        return GeminiAdapter(settings.gemini_api_key, model or settings.resolved_generation_model)
    if provider == "ollama":
        return OllamaAdapter(settings.ollama_base_url, model or settings.resolved_generation_model)
    raise ValueError(f"unknown LLM provider: {provider}")
