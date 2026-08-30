import json
from types import SimpleNamespace

from scripts.verify_private_space import evaluate_space_policy, inspect_space, main

EXPECTED_BASE = "labor_laws_20260830_3ec5ade"


def test_private_free_space_policy_accepts_runtime_only_secret_names():
    report = evaluate_space_policy(
        private=True,
        stage="RUNNING",
        current_hardware="cpu-basic",
        requested_hardware="cpu-basic",
        secret_names={"QDRANT_API_KEY", "SESSION_SIGNING_SECRET"},
        variable_names={"QDRANT_URL", "COLLECTION_NAME", "DEPLOYMENT_MODE"},
        collection_base=EXPECTED_BASE,
    )

    assert report["passed"] is True
    assert report["ready"] is True
    assert set(report["secret_names"]) == {
        "QDRANT_API_KEY",
        "SESSION_SIGNING_SECRET",
    }
    assert report["violations"] == []


def test_private_free_space_policy_rejects_owner_model_keys_and_paid_hardware():
    report = evaluate_space_policy(
        private=True,
        stage="RUNNING",
        current_hardware="t4-small",
        requested_hardware="t4-small",
        secret_names={"QDRANT_API_KEY", "GEMINI_API_KEY"},
        variable_names={"COLLECTION_NAME"},
        collection_base=EXPECTED_BASE,
    )

    assert report["passed"] is False
    assert {item["code"] for item in report["violations"]} == {
        "paid_hardware",
        "owner_model_key",
    }


def test_inspection_never_reads_secret_values():
    class ExplodingSecret:
        @property
        def value(self):
            raise AssertionError("secret value must not be read")

    class FakeApi:
        def repo_info(self, repo_id, *, repo_type):
            assert repo_id == "owner/demo"
            assert repo_type == "space"
            return SimpleNamespace(private=True)

        def get_space_runtime(self, repo_id):
            assert repo_id == "owner/demo"
            return SimpleNamespace(
                stage="RUNNING",
                hardware="cpu-basic",
                requested_hardware="cpu-basic",
            )

        def get_space_variables(self, repo_id):
            return {
                "COLLECTION_NAME": SimpleNamespace(value=EXPECTED_BASE),
                "QDRANT_URL": SimpleNamespace(value="must-not-be-emitted"),
            }

        def get_space_secrets(self, repo_id):
            return {
                "QDRANT_API_KEY": ExplodingSecret(),
                "SESSION_SIGNING_SECRET": ExplodingSecret(),
            }

    report = inspect_space(FakeApi(), "owner/demo")

    assert report["passed"] is True
    serialized = json.dumps(report, sort_keys=True)
    assert "must-not-be-emitted" not in serialized


def test_cli_api_failure_is_generic_and_redacted(capsys):
    class BrokenApi:
        def repo_info(self, repo_id, *, repo_type):
            raise RuntimeError("token=hf_secret private-endpoint.example")

    exit_code = main(
        ["--repo-id", "owner/demo", "--json"],
        api_factory=BrokenApi,
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload == {
        "error_class": "RuntimeError",
        "message": "space metadata unavailable",
        "status": "error",
    }
    assert "hf_secret" not in captured.err
    assert "private-endpoint" not in captured.err
