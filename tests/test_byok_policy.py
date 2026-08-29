from pathlib import Path

import pytest

from rag.api.byok import (
    ByokConcurrencyGate,
    ByokSessionManager,
    DemoBusy,
    InvalidDemoSession,
    SessionCapacityExceeded,
    SessionQuotaExceeded,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_session_capacity_recovers_after_expired_state_is_cleaned():
    now = [1_000.0]
    session_ids = iter(["first-session", "replacement-session"])
    manager = ByokSessionManager(
        secret="session-secret",
        query_limit=20,
        ttl_seconds=60,
        max_tracked_sessions=1,
        clock=lambda: now[0],
        token_factory=lambda: next(session_ids),
    )
    manager.issue()

    with pytest.raises(SessionCapacityExceeded):
        manager.issue()

    now[0] = 1_061.0
    replacement = manager.issue()
    assert replacement.startswith("replacement-session.")
    assert manager.tracked_sessions == 1


def test_concurrency_gate_rejects_instead_of_queueing():
    gate = ByokConcurrencyGate(limit=1)

    with gate.acquire():
        with pytest.raises(DemoBusy):
            with gate.acquire():
                pass

    with gate.acquire():
        pass


def test_hugging_face_runbook_requires_zero_cost_cpu_and_forbids_paid_fallback():
    runbook = (PROJECT_ROOT / "docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    assert "`cpu-basic`" in runbook
    assert "US$0" in runbook
    assert "CPU 驗收失敗時保持 private/paused，禁止自動改用任何付費硬體。" in runbook
    assert "持久 storage" in runbook
    assert "replica" in runbook
    assert "`t4-small`" not in runbook
    assert "3600 秒" not in runbook


def test_runbook_requires_blue_green_and_forbids_in_place_cloud_rebuild():
    runbook = (PROJECT_ROOT / "docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    assert "rebuild_qdrant_blue_green.py" in runbook
    assert "QDRANT_WRITER_API_KEY" in runbook
    assert "--confirm-candidate-base" in runbook
    assert "不得對正式 collection 執行 build_index.py" in runbook
    assert "撤銷 temporary writer key" in runbook
    assert "舊 COLLECTION_NAME" in runbook
    assert "rollback" in runbook


def test_runbook_keeps_refresh_manual_and_free():
    runbook = (PROJECT_ROOT / "docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    assert "不建立排程" in runbook
    assert "CPU Basic" in runbook
    assert "Qdrant Free" in runbook
    assert "不自動刪除" in runbook


def test_readme_links_safe_manual_qdrant_maintenance():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "### 人工更新 Qdrant 法規索引" in readme
    assert "scripts/rebuild_qdrant_blue_green.py" in readme
    assert "BLUE_GREEN_QDRANT_MAINTENANCE_DESIGN.md" in readme
