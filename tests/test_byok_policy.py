import pytest

from rag.api.byok import (
    ByokConcurrencyGate,
    ByokSessionManager,
    DemoBusy,
    InvalidDemoSession,
    SessionQuotaExceeded,
)


def test_signed_session_allows_exact_limit_and_rejects_tampering():
    now = [1_000.0]
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=2,
        ttl_seconds=60,
        clock=lambda: now[0],
        token_factory=lambda: "fixed-session",
    )
    token = manager.issue()

    manager.consume(token)
    manager.consume(token)
    with pytest.raises(SessionQuotaExceeded):
        manager.consume(token)
    with pytest.raises(InvalidDemoSession):
        manager.consume(token + "tampered")


def test_session_expires_and_is_removed_from_quota_state():
    now = [1_000.0]
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=20,
        ttl_seconds=60,
        clock=lambda: now[0],
        token_factory=lambda: "fixed-session",
    )
    token = manager.issue()
    now[0] = 1_061.0

    with pytest.raises(InvalidDemoSession):
        manager.consume(token)
    assert manager.tracked_sessions == 0


def test_new_issue_cleans_other_expired_session_state():
    now = [1_000.0]
    session_ids = iter(["expired-session", "fresh-session"])
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=20,
        ttl_seconds=60,
        clock=lambda: now[0],
        token_factory=lambda: next(session_ids),
    )
    manager.issue()
    now[0] = 1_061.0

    manager.issue()

    assert manager.tracked_sessions == 1


def test_invalid_session_errors_do_not_retain_token_text():
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=20,
        ttl_seconds=60,
        clock=lambda: 1_000.0,
        token_factory=lambda: "fixed-session",
    )
    secret_token = "attacker-controlled-session-token"

    with pytest.raises(InvalidDemoSession) as exc_info:
        manager.consume(secret_token)

    assert secret_token not in str(exc_info.value)
    assert secret_token not in repr(exc_info.value)


def test_concurrency_gate_rejects_instead_of_queueing():
    gate = ByokConcurrencyGate(limit=1)

    with gate.acquire():
        with pytest.raises(DemoBusy):
            with gate.acquire():
                pass

    with gate.acquire():
        pass
