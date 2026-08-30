import json

import httpx

from scripts.verify_qdrant_reader import main, verify_reader

CANDIDATE_BASE = "labor_laws_20260830_3ec5ade"
LEGACY_BASE = "labor_laws"
EXPECTED_COUNTS = {"fixed": 481, "structure": 884}


def reader_transport(*, candidate_counts=EXPECTED_COUNTS, legacy_status=403):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        for suffix, count in candidate_counts.items():
            if name == f"{CANDIDATE_BASE}_{suffix}":
                return httpx.Response(
                    200,
                    json={"result": {"points_count": count}, "status": "ok"},
                )
        if name in {f"{LEGACY_BASE}_fixed", f"{LEGACY_BASE}_structure"}:
            return httpx.Response(legacy_status, json={"status": "forbidden"})
        return httpx.Response(404, json={"status": "missing"})

    return httpx.MockTransport(handler), requests


def test_reader_verifier_reads_candidate_counts_and_requires_legacy_denial():
    transport, requests = reader_transport()
    with httpx.Client(transport=transport, base_url="https://qdrant.invalid") as client:
        report = verify_reader(
            client,
            candidate_base=CANDIDATE_BASE,
            legacy_base=LEGACY_BASE,
            expected_counts=EXPECTED_COUNTS,
        )

    assert report["passed"] is True
    assert report["candidate"] == {
        "fixed": {"actual": 481, "expected": 481, "readable": True},
        "structure": {"actual": 884, "expected": 884, "readable": True},
    }
    assert report["legacy_denied"] == {"fixed": True, "structure": True}
    assert {request.method for request in requests} == {"GET"}


def test_reader_verifier_rejects_wrong_count_and_legacy_access():
    transport, requests = reader_transport(
        candidate_counts={"fixed": 480, "structure": 884},
        legacy_status=200,
    )
    with httpx.Client(transport=transport, base_url="https://qdrant.invalid") as client:
        report = verify_reader(
            client,
            candidate_base=CANDIDATE_BASE,
            legacy_base=LEGACY_BASE,
            expected_counts=EXPECTED_COUNTS,
        )

    assert report["passed"] is False
    assert {item["code"] for item in report["violations"]} == {
        "point_count_mismatch",
        "legacy_access_allowed",
    }
    assert {request.method for request in requests} == {"GET"}


def test_qdrant_cli_output_is_redacted(monkeypatch, capsys):
    transport, _requests = reader_transport()

    def client_factory(**kwargs):
        assert kwargs["base_url"] == "https://private-qdrant.example"
        assert kwargs["headers"]["api-key"] == "qdrant-secret"
        return httpx.Client(transport=transport, base_url=kwargs["base_url"])

    monkeypatch.setenv("QDRANT_URL", "https://private-qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    exit_code = main(
        [
            "--candidate-base",
            CANDIDATE_BASE,
            "--legacy-base",
            LEGACY_BASE,
            "--fixed-count",
            "481",
            "--structure-count",
            "884",
            "--json",
        ],
        client_factory=client_factory,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["passed"] is True
    assert "private-qdrant" not in captured.out
    assert "qdrant-secret" not in captured.out
