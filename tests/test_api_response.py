import warnings
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag.api import main as api_main
from rag.api.byok import ByokConcurrencyGate, ByokSessionManager
from rag.api.main import QueryResponse
from rag.config import Settings
from rag.generation.llm import LLMOutput, ProviderOperationalError, ProviderPolicyError
from rag.generation.router import build_routed_llm


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

    def fake_build_llm(
        _settings,
        *,
        provider,
        model=None,
        api_key=None,
        timeout_seconds=None,
    ):
        adapter = FakeAdapter(provider)
        adapter.api_key = api_key
        adapter.timeout_seconds = timeout_seconds
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


def test_health_uses_the_same_effective_catalog_as_models(monkeypatch):
    """Catches health reporting the configured default when only another route is ready."""
    settings = configured_settings(llm_provider="gemini", providers=("openai",))
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "store",
        SimpleNamespace(count=lambda _collection: 0),
        raising=False,
    )

    health = api_main.health()
    catalog = api_main.models()

    assert health["status"] == "ok"
    assert health["default_provider"] == catalog["default_provider"] == "openai"
    assert health["available_providers"] == ["openai"]
    assert health["llm_provider"] == "openai"
    assert health["generation_model"] == "gpt-5.6-luna"


def test_health_is_degraded_without_a_public_generator(monkeypatch):
    """Catches health declaring an unconfigured Gemini route ready."""
    settings = configured_settings(llm_provider="gemini", providers=())
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "store",
        SimpleNamespace(count=lambda _collection: 0),
        raising=False,
    )

    health = api_main.health()

    assert health["status"] == "degraded"
    assert health["default_provider"] is None
    assert health["available_providers"] == []
    assert health["llm_provider"] is None
    assert health["generation_model"] is None


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

    assert exc_info.value.status_code == 503
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
        (ProviderOperationalError("gemini", "sdk-secret"), 503, "generation_unavailable"),
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


def _route_failure_answerer(settings, *, fallback_error=None):
    adapters = {
        "gemini": FakeAdapter("gemini"),
        "openai": FakeAdapter("openai"),
    }
    adapters["gemini"].generate = lambda *_args: (_ for _ in ()).throw(
        ProviderOperationalError("gemini", "http_503")
    )
    if fallback_error is not None:
        adapters["openai"].generate = lambda *_args: (_ for _ in ()).throw(
            fallback_error
        )
    routed = build_routed_llm(settings, "gemini", adapters=adapters)
    return SimpleNamespace(answer=lambda _question: routed.generate("system", "user"))


@pytest.mark.parametrize(
    "settings",
    [
        configured_settings(llm_fallback_enabled=False),
        configured_settings(key_values={"openai": "   "}),
    ],
)
def test_query_primary_only_operational_failure_is_503(monkeypatch, settings):
    """Catches disabled or blank-key routes being reported as a bad gateway."""
    answerer = _route_failure_answerer(settings)
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題", provider="gemini"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "generation_unavailable"


def test_query_dual_operational_failure_is_502(monkeypatch):
    """Catches an attempted and failed fallback being reported as primary-only."""
    settings = configured_settings()
    answerer = _route_failure_answerer(
        settings,
        fallback_error=ProviderOperationalError("openai", "http_429"),
    )
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(api_main.QueryRequest(question="問題", provider="gemini"))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "generation_unavailable"


def byok_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "deployment_mode": "public_byok",
        "qdrant_mode": "server",
        "qdrant_url": "https://qdrant.example.test",
        "qdrant_api_key": "read-only-secret",
        "session_signing_secret": "session-secret",
        "gemini_api_key": "owner-gemini-must-not-be-used",
        "openai_api_key": "owner-openai-must-not-be-used",
    }
    values.update(overrides)
    return Settings(**values)


def byok_result(provider="gemini", model="gemini-3.5-flash-lite"):
    return SimpleNamespace(
        text="回答",
        refused=False,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage=None,
        generation_called=True,
        requested_provider=provider,
        provider=provider,
        model=model,
        fallback_used=False,
        fallback_from=None,
    )


def install_byok_query_state(
    monkeypatch,
    *,
    query_limit=20,
    max_concurrency=2,
    answer_error=None,
):
    settings = byok_settings(
        byok_session_query_limit=query_limit,
        byok_max_concurrency=max_concurrency,
    )
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=query_limit,
        ttl_seconds=60,
    )
    answerer = SimpleNamespace(
        answer=(
            (lambda _question: (_ for _ in ()).throw(answer_error))
            if answer_error is not None
            else (lambda _question: byok_result())
        )
    )
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "byok_sessions", manager, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "byok_gate",
        ByokConcurrencyGate(settings.byok_max_concurrency),
        raising=False,
    )
    monkeypatch.setattr(
        api_main.state,
        "get_byok_answerer",
        lambda *_args, **_kwargs: answerer,
        raising=False,
    )
    return manager.issue()


def test_byok_catalog_is_ready_without_using_owner_keys(monkeypatch):
    settings = byok_settings(gemini_api_key="", openai_api_key="")
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "store",
        SimpleNamespace(count=lambda _collection: 2),
        raising=False,
    )

    assert api_main.models() == {
        "default_provider": "gemini",
        "providers": [
            {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
            {"provider": "openai", "model": "gpt-5.6-luna"},
        ],
        "requires_api_key": True,
        "session_query_limit": 20,
    }
    assert api_main.health()["status"] == "ok"


def test_byok_health_degrades_when_qdrant_is_unavailable(monkeypatch):
    settings = byok_settings(gemini_api_key="", openai_api_key="")
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(
        api_main.state,
        "store",
        SimpleNamespace(
            count=lambda _collection: (_ for _ in ()).throw(
                RuntimeError("endpoint detail must stay hidden")
            )
        ),
        raising=False,
    )

    response = api_main.health()

    assert response["status"] == "degraded"
    assert response["collection_structure_points"] is None
    assert response["collection_fixed_points"] is None
    assert "endpoint detail must stay hidden" not in repr(response)


def test_session_endpoint_issues_token_only_in_byok_mode(monkeypatch):
    settings = byok_settings()
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=20,
        ttl_seconds=60,
        clock=lambda: 1_000.0,
        token_factory=lambda: "fixed-session",
    )
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "byok_sessions", manager, raising=False)

    response = api_main.create_session()

    assert response.token.startswith("fixed-session.")
    assert response.query_limit == 20

    monkeypatch.setattr(
        api_main.state,
        "settings",
        configured_settings(),
        raising=False,
    )
    with pytest.raises(HTTPException) as exc_info:
        api_main.create_session()
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "byok_not_enabled"


def test_configure_runtime_builds_each_bm25_index_once_from_qdrant():
    payloads = {
        "labor_laws_structure": [
            {"chunk_id": "s1", "text": "正常工作時間"},
            {"chunk_id": "s2", "text": "延長工作時間"},
        ],
        "labor_laws_fixed": [
            {"chunk_id": "f1", "text": "工資給付"},
            {"chunk_id": "f2", "text": "退休金規定"},
        ],
    }
    calls = []
    app_state = api_main.AppState()
    app_state.store = SimpleNamespace(
        scroll_payloads=lambda collection: (
            calls.append(collection) or payloads[collection]
        ),
        count=lambda collection: len(payloads[collection]),
    )

    app_state.configure_runtime(byok_settings())

    assert calls == ["labor_laws_structure", "labor_laws_fixed"]
    assert len(app_state._byok_bm25_indexes["structure"]) == 2
    assert len(app_state._byok_bm25_indexes["fixed"]) == 2
    assert app_state.byok_sessions is not None
    assert app_state.byok_gate is not None


def test_configure_runtime_fails_closed_on_qdrant_count_mismatch():
    app_state = api_main.AppState()
    app_state.store = SimpleNamespace(
        scroll_payloads=lambda _collection: [{"chunk_id": "x", "text": "內容"}],
        count=lambda _collection: 2,
    )

    with pytest.raises(RuntimeError, match="Qdrant BM25 bootstrap failed"):
        app_state.configure_runtime(byok_settings())


def test_standard_runtime_does_not_scroll_qdrant_payloads():
    app_state = api_main.AppState()
    app_state.store = SimpleNamespace(
        scroll_payloads=lambda _collection: pytest.fail(
            "standard runtime must not bootstrap BM25 from Qdrant"
        )
    )

    app_state.configure_runtime(configured_settings())

    assert app_state._byok_bm25_indexes == {}
    assert app_state.byok_sessions is None
    assert app_state.byok_gate is None


def test_get_byok_answerer_uses_only_request_key_and_never_caches(monkeypatch):
    app_state = api_main.AppState()
    app_state.settings = byok_settings()
    app_state.embedder = object()
    app_state.store = object()
    app_state.reranker = object()
    in_memory_index = object()
    app_state._byok_bm25_indexes = {"structure": in_memory_index}
    adapter_calls = []
    answerer_calls = []

    def fake_build_llm(_settings, **kwargs):
        adapter_calls.append(kwargs)
        return SimpleNamespace(provider=kwargs["provider"], model="test-model")

    def fake_build_answerer(*_args, **kwargs):
        answerer_calls.append(kwargs)
        return SimpleNamespace(llm=kwargs["llm"])

    monkeypatch.setattr(api_main, "build_llm", fake_build_llm)
    monkeypatch.setattr(api_main, "build_answerer", fake_build_answerer)

    first = app_state.get_byok_answerer(
        "structure", "hybrid", True, "gemini", "visitor-one"
    )
    second = app_state.get_byok_answerer(
        "structure", "hybrid", True, "gemini", "visitor-two"
    )

    assert first is not second
    assert [call["api_key"] for call in adapter_calls] == [
        "visitor-one",
        "visitor-two",
    ]
    assert all(call["provider"] == "gemini" for call in adapter_calls)
    assert all(call["timeout_seconds"] == 60.0 for call in adapter_calls)
    assert all(call["bm25_index"] is in_memory_index for call in answerer_calls)
    assert app_state._adapter_cache == {}
    assert app_state._routed_llm_cache == {}
    assert app_state._answerer_cache == {}
    assert "owner-gemini-must-not-be-used" not in repr(adapter_calls)
    assert "owner-openai-must-not-be-used" not in repr(adapter_calls)


@pytest.mark.parametrize(
    ("api_key", "session", "status_code", "detail"),
    [
        (None, "valid", 401, "provider_api_key_required"),
        ("   ", "valid", 401, "provider_api_key_required"),
        ("x" * 513, "valid", 400, "provider_api_key_too_long"),
        ("visitor-key", None, 401, "invalid_demo_session"),
        ("visitor-key", "invalid", 401, "invalid_demo_session"),
    ],
)
def test_byok_query_rejects_bad_key_or_session_before_answerer(
    monkeypatch,
    api_key,
    session,
    status_code,
    detail,
):
    token = install_byok_query_state(monkeypatch)
    monkeypatch.setattr(
        api_main.state,
        "get_byok_answerer",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid BYOK request must stop before retrieval"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(
            api_main.QueryRequest(question="問題", provider="gemini"),
            provider_api_key=api_key,
            demo_session=token if session == "valid" else session,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail


def test_byok_query_rejects_overlong_question_before_answerer(monkeypatch):
    token = install_byok_query_state(monkeypatch)
    monkeypatch.setattr(
        api_main.state,
        "get_byok_answerer",
        lambda *_args, **_kwargs: pytest.fail(
            "overlong question must stop before retrieval"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(
            api_main.QueryRequest(question="問" * 2001, provider="gemini"),
            provider_api_key="visitor-key",
            demo_session=token,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "question_too_long"


def test_byok_session_quota_and_concurrency_map_to_429(monkeypatch):
    token = install_byok_query_state(monkeypatch, query_limit=1)
    request = api_main.QueryRequest(question="問題", provider="gemini")

    response = api_main.query(
        request,
        provider_api_key="visitor-key",
        demo_session=token,
    )
    assert response.answer == "回答"

    with pytest.raises(HTTPException) as quota_error:
        api_main.query(
            request,
            provider_api_key="visitor-key",
            demo_session=token,
        )
    assert quota_error.value.status_code == 429
    assert quota_error.value.detail == "session_quota_exceeded"

    token = install_byok_query_state(monkeypatch, max_concurrency=1)
    with api_main.state.byok_gate.acquire():
        with pytest.raises(HTTPException) as busy_error:
            api_main.query(
                request,
                provider_api_key="visitor-key",
                demo_session=token,
            )
    assert busy_error.value.status_code == 429
    assert busy_error.value.detail == "demo_busy"


@pytest.mark.parametrize(
    ("reason_code", "status_code", "detail"),
    [
        ("http_401", 401, "provider_key_rejected"),
        ("http_403", 401, "provider_key_rejected"),
        ("http_429", 429, "provider_rate_limited"),
        ("timeout", 504, "provider_timeout"),
        ("http_504", 504, "provider_timeout"),
        ("http_400", 502, "generation_unavailable"),
        ("http_503", 502, "generation_unavailable"),
    ],
)
def test_byok_provider_failures_use_safe_specific_statuses(
    monkeypatch,
    caplog,
    reason_code,
    status_code,
    detail,
):
    token = install_byok_query_state(
        monkeypatch,
        answer_error=ProviderOperationalError("gemini", reason_code),
    )
    visitor_key = "visitor-secret-key"
    unique_question = "unique-question-must-not-be-logged"

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(
            api_main.QueryRequest(question=unique_question, provider="gemini"),
            provider_api_key=visitor_key,
            demo_session=token,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail
    evidence = f"{exc_info.value!r}\n{caplog.text}"
    assert visitor_key not in evidence
    assert unique_question not in caplog.text
    assert "https://qdrant.example.test" not in evidence


def test_byok_policy_failure_remains_422(monkeypatch):
    token = install_byok_query_state(
        monkeypatch,
        answer_error=ProviderPolicyError("gemini", "provider-body-must-stay-hidden"),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_main.query(
            api_main.QueryRequest(question="問題", provider="gemini"),
            provider_api_key="visitor-key",
            demo_session=token,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "generation_rejected"
    assert "provider-body-must-stay-hidden" not in repr(exc_info.value)
