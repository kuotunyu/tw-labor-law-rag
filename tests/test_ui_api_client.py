import json

import httpx
import pytest

from ui.api_client import (
    actual_generation_metadata,
    fetch_models,
    requested_provider_for_display,
    submit_query,
)


def test_fetch_models_returns_discovery_payload():
    """Catches fetching any route other than the public model catalog."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        return httpx.Response(
            200,
            json={
                "default_provider": "gemini",
                "providers": [
                    {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
                    {"provider": "openai", "model": "gpt-5.6-luna"},
                ],
            },
        )

    payload = fetch_models("http://api", transport=httpx.MockTransport(handler))

    assert payload["default_provider"] == "gemini"
    assert [item["provider"] for item in payload["providers"]] == ["gemini", "openai"]


@pytest.mark.parametrize("malformed_payload", [[], None, "not-a-catalog"])
def test_fetch_models_rejects_non_object_json_payloads(malformed_payload):
    """Catches discovery JSON that would crash the UI's catalog parsing."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        return httpx.Response(200, json=malformed_payload)

    with pytest.raises(ValueError, match="invalid model discovery payload"):
        fetch_models("http://api", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "malformed_payload",
    [
        {},
        {"default_provider": None, "providers": None},
        {"default_provider": "gemini", "providers": [None]},
        {
            "default_provider": "anthropic",
            "providers": [{"provider": "anthropic", "model": "claude"}],
        },
        {
            "default_provider": "gemini",
            "providers": [{"provider": " gemini ", "model": "gemini-model"}],
        },
        {
            "default_provider": "gemini",
            "providers": [{"provider": ["gemini"], "model": "gemini-model"}],
        },
        {
            "default_provider": "gemini",
            "providers": [{"provider": "gemini", "model": "   "}],
        },
        {
            "default_provider": "gemini",
            "providers": [{"provider": "gemini", "model": 42}],
        },
        {
            "default_provider": "gemini",
            "providers": [
                {"provider": "gemini", "model": "first"},
                {"provider": "gemini", "model": "second"},
            ],
        },
        {
            "default_provider": None,
            "providers": [{"provider": "gemini", "model": "gemini-model"}],
        },
        {
            "default_provider": "openai",
            "providers": [{"provider": "gemini", "model": "gemini-model"}],
        },
        {
            "default_provider": ["gemini"],
            "providers": [{"provider": "gemini", "model": "gemini-model"}],
        },
        {"default_provider": "gemini", "providers": []},
    ],
)
def test_fetch_models_rejects_malformed_nested_catalog(malformed_payload):
    """Catches malformed nested discovery values reaching selectors or payloads."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=malformed_payload)

    with pytest.raises(ValueError, match="^invalid model discovery payload$"):
        fetch_models("http://api", transport=httpx.MockTransport(handler))


def test_fetch_models_trims_models_and_accepts_empty_catalog():
    """Catches display/payload model metadata retaining boundary whitespace."""
    responses = iter(
        [
            {
                "default_provider": "gemini",
                "providers": [
                    {"provider": "gemini", "model": "  gemini-model  "},
                ],
            },
            {"default_provider": None, "providers": []},
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    transport = httpx.MockTransport(handler)
    assert fetch_models("http://api", transport=transport) == {
        "default_provider": "gemini",
        "providers": [{"provider": "gemini", "model": "gemini-model"}],
    }
    assert fetch_models("http://api", transport=transport) == {
        "default_provider": None,
        "providers": [],
    }


def test_submit_query_includes_selected_provider():
    """Catches dropping the provider selected from API discovery."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/query"
        assert payload["provider"] == "openai"
        return httpx.Response(200, json={"answer": "ok"})

    response = submit_query(
        "http://api",
        {"question": "問題", "provider": "openai"},
        transport=httpx.MockTransport(handler),
    )

    assert response == {"answer": "ok"}


def test_actual_generation_metadata_keeps_stale_catalog_fallback_response():
    """Catches hiding the actual fallback route when cached choices are stale."""
    response = {
        "requested_provider": "gemini",
        "provider": "openai",
        "model": "gpt-5.6-luna-new",
        "fallback_used": True,
    }

    assert actual_generation_metadata(response) == ("openai", "gpt-5.6-luna-new")


def test_requested_provider_keeps_allowlisted_stale_history_route_for_display():
    """Catches a stale but safe historical route disappearing with catalog churn."""
    stale_response = {"requested_provider": "gemini"}

    assert requested_provider_for_display(stale_response) == "gemini"


@pytest.mark.parametrize(
    "requested_provider",
    [None, "", "   ", "anthropic", "<script>alert(1)</script>", 42],
)
def test_requested_provider_display_rejects_non_public_values(requested_provider):
    """Catches response metadata bypassing the fixed display allowlist."""
    assert (
        requested_provider_for_display({"requested_provider": requested_provider})
        is None
    )


@pytest.mark.parametrize(
    "response",
    [
        {"provider": None, "model": "gpt-5.6-luna"},
        {"provider": "openai", "model": ""},
        {"provider": "openai", "model": 1},
    ],
)
def test_actual_generation_metadata_requires_non_empty_text(response):
    """Catches rendering incomplete or non-text route metadata."""

    assert actual_generation_metadata(response) is None


@pytest.mark.parametrize(
    ("client_call", "path"),
    [
        (lambda transport: fetch_models("http://api", transport=transport), "/models"),
        (
            lambda transport: submit_query(
                "http://api", {"question": "問題", "provider": "gemini"}, transport=transport
            ),
            "/query",
        ),
    ],
)
def test_non_success_responses_raise_httpx_status_errors_without_response_body(
    client_call, path: str
):
    """Catches clients that conceal HTTP failures or echo response bodies in errors."""
    private_body = "do-not-display-provider-failure-details"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == path
        return httpx.Response(503, text=private_body, request=request)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client_call(httpx.MockTransport(handler))

    assert private_body not in str(exc_info.value)
