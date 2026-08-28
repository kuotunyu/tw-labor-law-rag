"""Small HTTP boundary for the Streamlit UI's public API calls."""

from __future__ import annotations

import httpx


def fetch_models(
    api_url: str, transport: httpx.BaseTransport | None = None
) -> dict:
    """Return the configured public model catalog from the API."""
    with httpx.Client(transport=transport, timeout=5.0) as client:
        response = client.get(f"{api_url.rstrip('/')}/models")
        response.raise_for_status()
        return response.json()


def submit_query(
    api_url: str, payload: dict, transport: httpx.BaseTransport | None = None
) -> dict:
    """Submit one UI query payload and return the API response."""
    with httpx.Client(transport=transport, timeout=120.0) as client:
        response = client.post(f"{api_url.rstrip('/')}/query", json=payload)
        response.raise_for_status()
        return response.json()
