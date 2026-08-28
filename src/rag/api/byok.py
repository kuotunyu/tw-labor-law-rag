"""In-memory safeguards for the public visitor-owned-key demo."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore, RLock

BYOK_MAX_KEY_CHARS = 512


class InvalidDemoSession(RuntimeError):
    def __init__(self):
        super().__init__("invalid or expired demo session")


class SessionQuotaExceeded(RuntimeError):
    def __init__(self):
        super().__init__("demo session query quota exceeded")


class SessionCapacityExceeded(RuntimeError):
    def __init__(self):
        super().__init__("demo session capacity exceeded")


class DemoBusy(RuntimeError):
    def __init__(self):
        super().__init__("demo is currently busy")


class ByokSessionManager:
    def __init__(
        self,
        *,
        secret: str,
        query_limit: int,
        ttl_seconds: int,
        max_tracked_sessions: int = 1000,
        clock: Callable[[], float] = time.time,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(24),
    ):
        if not secret:
            raise ValueError("session signing secret must not be empty")
        if query_limit < 1 or ttl_seconds < 1 or max_tracked_sessions < 1:
            raise ValueError("session limits and TTL must be positive")
        self._secret = secret.encode("utf-8")
        self._query_limit = query_limit
        self._ttl_seconds = ttl_seconds
        self._max_tracked_sessions = max_tracked_sessions
        self._clock = clock
        self._token_factory = token_factory
        self._sessions: dict[str, tuple[int, int]] = {}
        self._lock = RLock()

    def _signature(self, payload: str) -> str:
        return hmac.new(
            self._secret,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _cleanup_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, (expiry, _consumed) in self._sessions.items()
            if now >= expiry
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def issue(self) -> str:
        with self._lock:
            now = self._clock()
            self._cleanup_expired(now)
            if len(self._sessions) >= self._max_tracked_sessions:
                raise SessionCapacityExceeded()
            session_id = self._token_factory()
            if not session_id or "." in session_id:
                raise ValueError("session token factory returned an invalid identifier")
            expiry = int(now + self._ttl_seconds)
            self._sessions[session_id] = (expiry, 0)
            payload = f"{session_id}.{expiry}"
            return f"{payload}.{self._signature(payload)}"

    def consume(self, token: str) -> None:
        with self._lock:
            now = self._clock()
            self._cleanup_expired(now)
            try:
                session_id, raw_expiry, signature = token.split(".")
                expiry = int(raw_expiry)
            except (AttributeError, TypeError, ValueError):
                raise InvalidDemoSession() from None

            payload = f"{session_id}.{expiry}"
            if not hmac.compare_digest(signature, self._signature(payload)):
                raise InvalidDemoSession()

            session = self._sessions.get(session_id)
            if session is None or session[0] != expiry or now >= expiry:
                raise InvalidDemoSession()
            consumed = session[1]
            if consumed >= self._query_limit:
                raise SessionQuotaExceeded()
            self._sessions[session_id] = (expiry, consumed + 1)

    @property
    def tracked_sessions(self) -> int:
        with self._lock:
            self._cleanup_expired(self._clock())
            return len(self._sessions)


class ByokConcurrencyGate:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("concurrency limit must be positive")
        self._semaphore = BoundedSemaphore(limit)

    @contextmanager
    def acquire(self) -> Iterator[None]:
        if not self._semaphore.acquire(blocking=False):
            raise DemoBusy()
        try:
            yield
        finally:
            self._semaphore.release()
