"""Run the private FastAPI backend and public Streamlit UI in one Space."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable

import httpx


def api_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "rag.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def ui_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/app.py",
        "--server.address",
        # Hugging Face requires the single public UI port to bind all interfaces.
        "0.0.0.0",  # nosec B104
        "--server.port",
        "7860",
        "--server.headless",
        "true",
    ]


def wait_for_api(
    process: subprocess.Popen,
    url: str,
    timeout_seconds: float,
    *,
    client=httpx,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        if process.poll() is not None:
            raise RuntimeError("API process exited before readiness")
        try:
            response = client.get(url, timeout=2.0)
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            sleep(0.5)
            continue
        if (
            response.status_code == 200
            and isinstance(payload, dict)
            and payload.get("status") == "ok"
        ):
            return
        sleep(0.5)
    raise TimeoutError("API readiness timed out")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def supervise(
    api_process: subprocess.Popen,
    ui_process: subprocess.Popen,
    *,
    poll_interval: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    while True:
        api_code = api_process.poll()
        if api_code is not None:
            _stop_process(ui_process)
            return api_code
        ui_code = ui_process.poll()
        if ui_code is not None:
            _stop_process(api_process)
            return ui_code
        sleep(poll_interval)


def run() -> int:
    api_process = subprocess.Popen(api_command())
    ui_process = None
    try:
        wait_for_api(
            api_process,
            "http://127.0.0.1:8000/health",
            timeout_seconds=300,
        )
        ui_process = subprocess.Popen(ui_command())
        return supervise(api_process, ui_process)
    finally:
        _stop_process(ui_process)
        _stop_process(api_process)


if __name__ == "__main__":
    raise SystemExit(run())
