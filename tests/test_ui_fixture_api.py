from scripts.run_ui_fixture_api import response_for


def test_fixture_exposes_the_complete_public_byok_contract():
    status, models = response_for("GET", "/models", {}, {})
    assert status == 200
    assert models == {
        "default_provider": "gemini",
        "providers": [
            {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
            {"provider": "openai", "model": "gpt-5.6-luna"},
        ],
        "requires_api_key": True,
        "session_query_limit": 20,
    }

    status, session = response_for("POST", "/session", {}, {})
    assert status == 200
    assert session == {"token": "local-fixture-session", "query_limit": 20}


def test_fixture_requires_a_key_and_never_echoes_it():
    payload = {
        "question": "勞工每天和每週的正常工作時間上限是多少？",
        "provider": "gemini",
        "strategy": "structure",
        "mode": "hybrid",
        "use_reranker": True,
    }

    status, missing = response_for("POST", "/query", payload, {})
    assert status == 401
    assert missing == {"detail": "fixture provider key required"}

    secret = "test-only-not-a-provider-key"
    status, query = response_for(
        "POST",
        "/query",
        payload,
        {"X-Provider-Api-Key": secret, "X-Demo-Session": "local-fixture-session"},
    )
    assert status == 200
    assert query["answer"] == "一般情況下，勞工每日正常工作時間不得超過 8 小時。[1]"
    assert query["sources"][0]["doc"] == "勞動基準法"
    assert query["sources"][0]["article"] == "第 30 條"
    assert query["generation_called"] is True
    assert query["fallback_used"] is False
    assert secret not in repr(query)


def test_fixture_rejects_unknown_routes_without_reflecting_input():
    status, response = response_for(
        "POST",
        "/unknown",
        {"question": "private fixture input"},
        {"X-Provider-Api-Key": "private fixture key"},
    )

    assert status == 404
    assert response == {"detail": "fixture route not found"}
