"""Small HTTP boundary for the Streamlit UI's public API calls."""

from __future__ import annotations

import httpx

_PUBLIC_PROVIDERS = {"gemini", "openai"}
_DISCOVERY_ERROR = "invalid model discovery payload"
_SESSION_ERROR = "invalid session payload"


class ApiRequestError(RuntimeError):
    """Status-only error that cannot retain a secret-bearing httpx request."""

    def __init__(self, status_code: int):
        super().__init__("API request failed")
        self.status_code = status_code


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
    result = {"default_provider": default_provider, "providers": providers}
    has_byok_flag = "requires_api_key" in payload
    has_query_limit = "session_query_limit" in payload
    if has_byok_flag or has_query_limit:
        requires_api_key = payload.get("requires_api_key")
        query_limit = payload.get("session_query_limit")
        if not isinstance(requires_api_key, bool):
            raise ValueError(_DISCOVERY_ERROR)
        if requires_api_key:
            if (
                not isinstance(query_limit, int)
                or isinstance(query_limit, bool)
                or query_limit < 1
            ):
                raise ValueError(_DISCOVERY_ERROR)
        elif query_limit is not None:
            raise ValueError(_DISCOVERY_ERROR)
        result.update(
            requires_api_key=requires_api_key,
            session_query_limit=query_limit,
        )
    return result


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


def fetch_session(
    api_url: str, transport: httpx.BaseTransport | None = None
) -> dict:
    """Create one opaque, bounded demo session for a BYOK browser session."""
    with httpx.Client(transport=transport, timeout=5.0) as client:
        response = client.post(f"{api_url.rstrip('/')}/session")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            raise ValueError(_SESSION_ERROR) from None
    if not isinstance(payload, dict):
        raise ValueError(_SESSION_ERROR)
    token = payload.get("token")
    query_limit = payload.get("query_limit")
    if (
        not isinstance(token, str)
        or not token.strip()
        or not isinstance(query_limit, int)
        or isinstance(query_limit, bool)
        or query_limit < 1
    ):
        raise ValueError(_SESSION_ERROR)
    return {"token": token, "query_limit": query_limit}


def submit_query(
    api_url: str,
    payload: dict,
    *,
    api_key: str | None = None,
    session_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Submit one UI query payload and return the API response."""
    headers = {}
    if api_key is not None:
        headers["X-Provider-Api-Key"] = api_key
    if session_token is not None:
        headers["X-Demo-Session"] = session_token
    with httpx.Client(transport=transport, timeout=120.0) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/query",
            json=payload,
            headers=headers,
        )
        if headers and not response.is_success:
            raise ApiRequestError(response.status_code)
        response.raise_for_status()
        return response.json()
