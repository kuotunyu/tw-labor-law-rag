"""Run a deterministic localhost-only API for UI screenshots and acceptance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-5.6-luna",
}
_SESSION_TOKEN = "local-fixture-session"


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.casefold()
    return next(
        (str(value).strip() for key, value in headers.items() if key.casefold() == wanted),
        "",
    )


def response_for(
    method: str,
    path: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    """Return one complete fixture response without retaining request secrets."""
    if method == "GET" and path == "/models":
        return 200, {
            "default_provider": "gemini",
            "providers": [
                {"provider": provider, "model": model}
                for provider, model in _MODELS.items()
            ],
            "requires_api_key": True,
            "session_query_limit": 20,
        }

    if method == "POST" and path == "/session":
        return 200, {"token": _SESSION_TOKEN, "query_limit": 20}

    if method == "POST" and path == "/query":
        if not _header(headers, "X-Provider-Api-Key") or (
            _header(headers, "X-Demo-Session") != _SESSION_TOKEN
        ):
            return 401, {"detail": "fixture provider key required"}

        provider = payload.get("provider")
        if provider not in _MODELS:
            return 400, {"detail": "fixture provider is invalid"}

        return 200, {
            "answer": "一般情況下，勞工每日正常工作時間不得超過 8 小時。[1]",
            "refused": False,
            "sources": [
                {
                    "index": 1,
                    "doc": "勞動基準法",
                    "article": "第 30 條",
                    "content": "勞工正常工作時間，每日不得超過八小時，每週不得超過四十小時。",
                    "source_url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",
                    "last_amended": "20250718",
                    "effective_date": "20250718",
                }
            ],
            "retrieval_hits": [{"citation": "勞動基準法 第 30 條", "score": 0.91}],
            "strategy": payload.get("strategy", "structure"),
            "mode": payload.get("mode", "hybrid"),
            "use_reranker": payload.get("use_reranker", True),
            "provider": provider,
            "model": _MODELS[provider],
            "refusal_stage": None,
            "generation_called": True,
            "requested_provider": provider,
            "fallback_used": False,
            "fallback_from": None,
        }

    return 404, {"detail": "fixture route not found"}


class FixtureHandler(BaseHTTPRequestHandler):
    """HTTP adapter around the pure fixture response contract."""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond(self, method: str) -> None:
        payload: Mapping[str, Any] = {}
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                decoded = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(decoded, dict):
                    raise ValueError
                payload = decoded
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"detail": "fixture JSON is invalid"})
                return
        status, response = response_for(method, self.path, payload, self.headers)
        self._send_json(status, response)

    def do_GET(self) -> None:
        self._respond("GET")

    def do_POST(self) -> None:
        self._respond("POST")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), FixtureHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
