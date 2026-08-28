import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from streamlit.testing.v1 import AppTest


def test_streamlit_byok_flow_keeps_visitor_key_out_of_rendered_history(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def _send_json(self, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/models"
            self._send_json(
                {
                    "default_provider": "gemini",
                    "providers": [
                        {
                            "provider": "gemini",
                            "model": "gemini-3.5-flash-lite",
                        },
                        {"provider": "openai", "model": "gpt-5.6-luna"},
                    ],
                    "requires_api_key": True,
                    "session_query_limit": 20,
                }
            )

        def do_POST(self):
            if self.path == "/session":
                requests.append({"path": self.path})
                self._send_json({"token": "signed-session", "query_limit": 20})
                return
            assert self.path == "/query"
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            requests.append(
                {
                    "path": self.path,
                    "provider_key": self.headers.get("X-Provider-Api-Key"),
                    "demo_session": self.headers.get("X-Demo-Session"),
                    "payload": payload,
                }
            )
            self._send_json(
                {
                    "answer": "依勞動基準法規定計算。[1]",
                    "refused": False,
                    "sources": [
                        {
                            "index": 1,
                            "doc": "勞動基準法",
                            "article": "第 24 條",
                            "content": "延長工作時間之工資應依法加給。",
                            "source_url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",
                            "last_amended": "20250718",
                            "effective_date": "20250718",
                        }
                    ],
                    "retrieval_hits": [
                        {"citation": "勞動基準法 第 24 條", "score": 0.9}
                    ],
                    "strategy": payload["strategy"],
                    "mode": payload["mode"],
                    "use_reranker": payload["use_reranker"],
                    "provider": payload["provider"],
                    "model": "gemini-3.5-flash-lite",
                    "refusal_stage": None,
                    "generation_called": True,
                    "requested_provider": payload["provider"],
                    "fallback_used": False,
                    "fallback_from": None,
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("API_URL", api_url)

    try:
        app_path = Path(__file__).parents[1] / "ui" / "app.py"
        app = AppTest.from_file(app_path, default_timeout=10).run()

        assert not app.exception
        assert len(app.segmented_control) == 1
        assert app.segmented_control[0].label == "回答模型"
        assert app.segmented_control[0].options == [
            "Gemini · gemini-3.5-flash-lite",
            "OpenAI · gpt-5.6-luna",
        ]
        assert app.segmented_control[0].value == "gemini"
        assert all(widget.label != "回答模型" for widget in app.sidebar.selectbox)
        assert any(
            element.value == "🔐 開始安全問答" for element in app.subheader
        )
        assert app.chat_input[0].disabled is True
        assert requests == [{"path": "/session"}]

        app.text_input[0].set_value("gemini-visitor-secret-key").run()
        assert app.chat_input[0].disabled is False
        assert any(
            element.value == "API Key 已填入，可以開始問答。"
            for element in app.success
        )

        app.segmented_control[0].set_value("openai").run()
        assert app.text_input[0].value == ""
        assert app.chat_input[0].disabled is True

        visitor_key = "openai-visitor-secret-key"
        app.text_input[0].set_value(visitor_key).run()
        assert app.chat_input[0].disabled is False

        app.chat_input[0].set_value("加班費如何計算？").run()

        query_request = next(item for item in requests if item["path"] == "/query")
        assert query_request["provider_key"] == visitor_key
        assert query_request["demo_session"] == "signed-session"
        assert query_request["payload"]["provider"] == "openai"
        assert visitor_key not in repr(query_request["payload"])
        assert visitor_key not in repr(app.session_state["history"])
        rendered = "\n".join(
            str(element.value)
            for collection in (app.markdown, app.caption, app.info, app.warning)
            for element in collection
        )
        assert visitor_key not in rendered
        assert "最新異動：2025-07-18" in rendered
        assert any(
            "[全國法規資料庫](https://law.moj.gov.tw/" in str(element.value)
            for element in app.markdown
        )

        app.session_state["history"].append(
            {
                "role": "assistant",
                "content": "舊索引引用仍可顯示。",
                "sources": [
                    {
                        "index": 1,
                        "doc": "工會法",
                        "article": "第 11 條",
                        "content": "舊資料沒有來源網址與日期。",
                    }
                ],
            }
        )
        app.run()
        assert not app.exception

        clear_button = next(
            button for button in app.button if button.label == "清除 API Key"
        )
        clear_button.click().run()
        assert app.text_input[0].value == ""
        assert app.chat_input[0].disabled is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
