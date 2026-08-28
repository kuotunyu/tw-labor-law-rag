import warnings
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag.api import main as api_main
from rag.api.main import QueryResponse
from rag.config import Settings
from rag.generation.llm import LLMOutput, ProviderOperationalError, ProviderPolicyError


class FakeAdapter:
    def __init__(self, provider: str):
        self.provider = provider
        self.model = f"{provider}-test"

    def generate(self, system, user, temperature=0.0, max_tokens=2048):
        return LLMOutput(text="unused", provider=self.provider, model=self.model)


def configured_settings(
    *,
    llm_provider="gemini",
    providers=("gemini", "openai"),
    key_values=None,
    **overrides,
) -> Settings:
    values = {
        "_env_file": None,
        "llm_provider": llm_provider,
        "gemini_api_key": "",
        "openai_api_key": "",
    }
    for provider in providers:
        values[f"{provider}_api_key"] = "dummy"
    for provider, value in (key_values or {}).items():
        values[f"{provider}_api_key"] = value
    values.update(overrides)
    return Settings(**values)


def configured_state(monkeypatch, **settings_overrides):
    app_state = api_main.AppState()
    app_state.settings = configured_settings(**settings_overrides)
    app_state.embedder = object()
    app_state.store = object()
    app_state.reranker = object()
    built_adapters = []

    def fake_build_llm(_settings, *, provider, model=None):
        adapter = FakeAdapter(provider)
        built_adapters.append(adapter)
        return adapter

    def fake_build_answerer(_settings, _embedder, _store, **kwargs):
        return SimpleNamespace(llm=kwargs["llm"])

    monkeypatch.setattr(api_main, "build_llm", fake_build_llm)
    monkeypatch.setattr(api_main, "build_answerer", fake_build_answerer)
    return app_state, built_adapters


def _response_payload() -> dict:
    return {
        "answer": "根據檢索到的法規內容無法回答此問題。",
        "refused": True,
        "sources": [],
        "retrieval_hits": [],
        "strategy": "structure",
        "mode": "hybrid",
        "use_reranker": True,
        "provider": "openai",
        "model": "example-model",
    }


def test_query_response_keeps_refusal_stage_optional_for_older_callers():
    response = QueryResponse(**_response_payload())

    assert response.refusal_stage is None


@pytest.mark.parametrize("stage", ["no_hits", "threshold", "llm"])
def test_query_response_accepts_known_refusal_stages(stage):
    response = QueryResponse(**_response_payload(), refusal_stage=stage)

    assert response.refusal_stage == stage


def test_query_response_rejects_unknown_refusal_stage():
    with pytest.raises(ValidationError):
        QueryResponse(**_response_payload(), refusal_stage="other")


def test_query_request_accepts_only_public_providers():
    assert api_main.QueryRequest(question="問題", provider="gemini").provider == "gemini"
    assert api_main.QueryRequest(question="問題", provider="openai").provider == "openai"
    with pytest.raises(ValidationError):
        api_main.QueryRequest(question="問題", provider="anthropic")


def test_models_lists_only_configured_public_providers(monkeypatch):
    settings = configured_settings(providers=("gemini",))
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main,
        "build_llm",
        lambda *_args, **_kwargs: pytest.fail("discovery must not instantiate an adapter"),
    )

    response = api_main.models()

    assert response == {
        "default_provider": "gemini",
        "providers": [
            {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        ],
    }


def test_models_returns_no_provider_when_no_public_key_is_configured(monkeypatch):
    settings = configured_settings(providers=())
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)

    assert api_main.models() == {"default_provider": None, "providers": []}


@pytest.mark.parametrize(
    ("settings", "expected_default"),
    [
        (
            configured_settings(providers=("openai",)),
            "openai",
        ),
        (
            configured_settings(
                llm_provider="anthropic",
                providers=("gemini",),
            ),
            "gemini",
        ),
    ],
)
def test_models_default_is_an_available_public_provider(
    monkeypatch,
    settings,
    expected_default,
):
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)

    response = api_main.models()

    assert response["default_provider"] == expected_default
    assert expected_default in {item["provider"] for item in response["providers"]}


def test_models_route_has_validated_response_schema(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
        monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
        response = TestClient(api_main.app).get("/models")
    response_schema = api_main.app.openapi()["paths"]["/models"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response.status_code == 200
    assert response.json()["default_provider"] == "gemini"
    assert response_schema["$ref"].endswith("/ModelsResponse")


def test_models_never_exposes_server_secrets(monkeypatch):
    settings = configured_settings(
        providers=(),
        key_values={
            "gemini": "gemini-top-secret",
            "openai": "openai-top-secret",
        },
    )
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)

    response_text = repr(api_main.models())

    assert "gemini-top-secret" not in response_text
    assert "openai-top-secret" not in response_text


def test_same_provider_reuses_answerer_and_routed_llm(monkeypatch):
    app_state, built_adapters = configured_state(monkeypatch)

    first = app_state.get_answerer("structure", "hybrid", True, "gemini")
    second = app_state.get_answerer("structure", "hybrid", True, "gemini")

    assert first is second
    assert first.llm.primary_provider == "gemini"
    assert len(built_adapters) == 2


def test_different_providers_have_isolated_answerers_and_reuse_concrete_adapters(monkeypatch):
    app_state, built_adapters = configured_state(monkeypatch)

    gemini_answerer = app_state.get_answerer("structure", "hybrid", True, "gemini")
    openai_answerer = app_state.get_answerer("structure", "hybrid", True, "openai")

    assert gemini_answerer is not openai_answerer
    assert gemini_answerer.llm.primary_provider == "gemini"
    assert openai_answerer.llm.primary_provider == "openai"
    assert gemini_answerer.llm.primary is openai_answerer.llm.fallback
    assert gemini_answerer.llm.fallback is openai_answerer.llm.primary
    assert len(built_adapters) == 2


def test_whitespace_key_is_not_built_as_a_fallback_adapter(monkeypatch):
    app_state, built_adapters = configured_state(
        monkeypatch,
        key_values={"openai": "   "},
    )

    answerer = app_state.get_answerer("structure", "hybrid", True, "gemini")

    assert answerer.llm.primary_provider == "gemini"
    assert answerer.llm.fallback is None
    assert [adapter.provider for adapter in built_adapters] == ["gemini"]


def test_query_exposes_answerer_refusal_stage(monkeypatch):
    settings = configured_settings(llm_provider="openai")
    result = SimpleNamespace(
        text="根據檢索到的法規內容無法回答此問題。",
        refused=True,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage="threshold",
        generation_called=False,
        requested_provider="openai",
        provider=None,
        model=None,
        fallback_used=False,
        fallback_from=None,
    )
    answerer = SimpleNamespace(answer=lambda _question: result)
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    response = api_main.query(api_main.QueryRequest(question="雇主可以問不在法規庫的問題嗎?"))

    assert response.refused is True
    assert response.refusal_stage == "threshold"


def test_query_reports_actual_fallback_route(monkeypatch):
    result = SimpleNamespace(
        text="回答",
        refused=False,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage=None,
        generation_called=True,
        requested_provider="gemini",
        provider="openai",
        model="gpt-5.6-luna",
        fallback_used=True,
        fallback_from="gemini",
    )
    answerer = SimpleNamespace(answer=lambda _question: result)
    monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    response = api_main.query(api_main.QueryRequest(question="問題", provider="gemini"))

    assert response.generation_called is True
    assert response.requested_provider == "gemini"
    assert response.provider == "openai"
    assert response.model == "gpt-5.6-luna"
    assert response.fallback_used is True
    assert response.fallback_from == "gemini"


def test_query_uses_configured_default_provider_when_omitted(monkeypatch):
    settings = configured_settings(llm_provider="openai")
    result = SimpleNamespace(
        text="回答",
        refused=False,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage=None,
        generation_called=True,
        requested_provider="openai",
        provider="openai",
        model="gpt-5.6-luna",
        fallback_used=False,
        fallback_from=None,
    )
    requested_providers = []
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "get_answerer",
        lambda _strategy, _mode, _reranker, provider: (
            requested_providers.append(provider)
            or SimpleNamespace(answer=lambda _question: result)
        ),
    )

    response = api_main.query(api_main.QueryRequest(question="問題"))

    assert requested_providers == ["openai"]
    assert response.requested_provider == "openai"


def test_query_omission_uses_the_same_available_default_as_models(monkeypatch):
    settings = configured_settings(providers=("openai",))
    result = SimpleNamespace(
        text="回答",
        refused=False,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage=None,
        generation_called=True,
        requested_provider="openai",
        provider="openai",
        model="gpt-5.6-luna",
        fallback_used=False,
        fallback_from=None,
    )
    requested_providers = []
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "get_answerer",
        lambda _strategy, _mode, _reranker, provider: (
            requested_providers.append(provider)
            or SimpleNamespace(answer=lambda _question: result)
        ),
    )

    response = api_main.query(api_main.QueryRequest(question="問題"))

    assert api_main.models()["default_provider"] == "openai"
    assert requested_providers == ["openai"]
    assert response.requested_provider == "openai"


def test_query_omission_without_available_provider_is_service_unavailable(monkeypatch):
    settings = configured_settings(providers=())
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "get_answerer",
        lambda *_args: pytest.fail("no-provider request must stop before retrieval"),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "generation_unavailable"


def test_query_rejects_unconfigured_provider_before_answerer_lookup(monkeypatch):
    settings = configured_settings(providers=("gemini",))
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "get_answerer",
        lambda *_args: pytest.fail("unavailable provider must be rejected before retrieval"),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題", provider="openai"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "provider_unavailable"


def test_query_sanitizes_operational_provider_failure(monkeypatch):
    secret = "sdk-message-with-sk-top-secret"

    def fail(_question):
        raise ProviderOperationalError("gemini", secret)

    answerer = SimpleNamespace(answer=fail)
    monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題"))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "generation_unavailable"
    assert secret not in str(exc_info.value.detail)


def test_query_sanitizes_policy_provider_failure(monkeypatch):
    secret = "policy-message-with-gemini-top-secret"

    def fail(_question):
        raise ProviderPolicyError("gemini", secret)

    answerer = SimpleNamespace(answer=fail)
    monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題"))

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "generation_rejected"
    assert secret not in str(exc_info.value.detail)


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ProviderOperationalError("gemini", "sdk-secret"), 502, "generation_unavailable"),
        (ProviderPolicyError("gemini", "sdk-secret"), 422, "generation_rejected"),
    ],
)
def test_query_sanitizes_provider_failures_during_lazy_answerer_construction(
    monkeypatch,
    error,
    status_code,
    detail,
):
    monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
    monkeypatch.setattr(
        api_main.state,
        "get_answerer",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題"))

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail
    assert "sdk-secret" not in str(exc_info.value.detail)


def test_query_does_not_mislabel_programming_errors_as_provider_failures(monkeypatch):
    answerer = SimpleNamespace(
        answer=lambda _question: (_ for _ in ()).throw(ValueError("programming bug"))
    )
    monkeypatch.setattr(api_main.state, "settings", configured_settings(), raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    with pytest.raises(ValueError, match="programming bug"):
        api_main.query(api_main.QueryRequest(question="問題"))
