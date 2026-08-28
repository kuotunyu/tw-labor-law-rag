"""Small HTTP boundary for the Streamlit UI's public API calls."""

from __future__ import annotations

import httpx

_PUBLIC_PROVIDERS = {"gemini", "openai"}
_DISCOVERY_ERROR = "invalid model discovery payload"


def _validate_model_catalog(payload: object) -> dict:
    if not isinstance(payload, dict) or "default_provider" not in payload:
        raise ValueError(_DISCOVERY_ERROR)
    records = payload.get("providers")
    if not isinstance(records, list):
        raise ValueError(_DISCOVERY_ERROR)

    providers: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(_DISCOVERY_ERROR)
        provider = record.get("provider")
        model = record.get("model")
        if (
            not isinstance(provider, str)
            or provider not in _PUBLIC_PROVIDERS
            or provider in seen
            or not isinstance(model, str)
            or not model.strip()
        ):
            raise ValueError(_DISCOVERY_ERROR)
        seen.add(provider)
        providers.append({"provider": provider, "model": model.strip()})

    default_provider = payload["default_provider"]
    if (not providers and default_provider is not None) or (
        providers
        and (
            not isinstance(default_provider, str)
            or default_provider not in seen
        )
    ):
        raise ValueError(_DISCOVERY_ERROR)
    return {"default_provider": default_provider, "providers": providers}


def actual_generation_metadata(payload: dict) -> tuple[str, str] | None:
    """Return display-safe actual provider/model metadata from a query response."""
    provider = payload.get("provider")
    model = payload.get("model")
    if (
        isinstance(provider, str)
        and provider.strip()
        and isinstance(model, str)
        and model.strip()
    ):
        return provider, model
    return None


def requested_provider_for_display(payload: dict) -> str | None:
    """Return allowlisted requested-route metadata for plain historical display."""
    provider = payload.get("requested_provider")
    if isinstance(provider, str) and provider.strip() in _PUBLIC_PROVIDERS:
        return provider.strip()
    return None


def fetch_models(
    api_url: str, transport: httpx.BaseTransport | None = None
) -> dict:
    """Return the configured public model catalog from the API."""
    with httpx.Client(transport=transport, timeout=5.0) as client:
        response = client.get(f"{api_url.rstrip('/')}/models")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            raise ValueError(_DISCOVERY_ERROR) from None
    return _validate_model_catalog(payload)


def submit_query(
    api_url: str, payload: dict, transport: httpx.BaseTransport | None = None
) -> dict:
    """Submit one UI query payload and return the API response."""
    with httpx.Client(transport=transport, timeout=120.0) as client:
        response = client.post(f"{api_url.rstrip('/')}/query", json=payload)
        response.raise_for_status()
        return response.json()
