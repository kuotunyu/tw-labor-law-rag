import subprocess

import pytest

from scripts import run_space


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeProcess:
    def __init__(self, returncode=None, wait_times_out=False):
        self.returncode = returncode
        self.wait_times_out = wait_times_out
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_times_out and not self.killed:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self):
        self.killed = True


def test_wait_for_api_fails_when_process_exits_before_readiness():
    process = FakeProcess(returncode=3)

    class Client:
        @staticmethod
        def get(*_args, **_kwargs):
            pytest.fail("exited API process must be detected before an HTTP call")

    with pytest.raises(RuntimeError, match="API process exited before readiness"):
        run_space.wait_for_api(
            process,
            "http://127.0.0.1:8000/health",
            10,
            client=Client,
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )


def test_wait_for_api_requires_200_with_ok_health_payload():
    responses = iter(
        [
            FakeResponse(503, {"status": "degraded"}),
            FakeResponse(200, {"status": "degraded"}),
            FakeResponse(200, {"status": "ok"}),
        ]
    )
    calls = []
    now = [0.0]

    class Client:
        @staticmethod
        def get(url, timeout):
            calls.append((url, timeout))
            return next(responses)

    def advance(seconds):
        now[0] += seconds

    run_space.wait_for_api(
        FakeProcess(),
        "http://127.0.0.1:8000/health",
        10,
        client=Client,
        clock=lambda: now[0],
        sleep=advance,
    )

    assert calls == [
        ("http://127.0.0.1:8000/health", 2.0),
        ("http://127.0.0.1:8000/health", 2.0),
        ("http://127.0.0.1:8000/health", 2.0),
    ]


def test_supervise_terminates_api_when_ui_exits():
    api_process = FakeProcess()
    ui_process = FakeProcess(returncode=7)

    result = run_space.supervise(
        api_process,
        ui_process,
        poll_interval=0,
        sleep=lambda _seconds: None,
    )

    assert result == 7
    assert api_process.terminated is True
    assert api_process.killed is False


def test_supervise_escalates_to_kill_after_bounded_wait():
    api_process = FakeProcess(returncode=2)
    ui_process = FakeProcess(wait_times_out=True)

    result = run_space.supervise(
        api_process,
        ui_process,
        poll_interval=0,
        sleep=lambda _seconds: None,
    )

    assert result == 2
    assert ui_process.terminated is True
    assert ui_process.killed is True
    assert ui_process.wait_calls == 2


def test_space_commands_bind_only_streamlit_to_public_port():
    api_command = run_space.api_command()
    ui_command = run_space.ui_command()

    assert api_command[-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    assert ui_command[-6:] == [
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "7860",
        "--server.headless",
        "true",
    ]
