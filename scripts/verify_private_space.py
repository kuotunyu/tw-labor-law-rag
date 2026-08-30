"""Read-only, redacted policy preflight for the private Hugging Face Space."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from huggingface_hub import HfApi

ALLOWED_SECRET_NAMES = frozenset({"QDRANT_API_KEY", "SESSION_SIGNING_SECRET"})
OWNER_KEY_MARKERS = ("OPENAI", "GEMINI", "GOOGLE", "ANTHROPIC", "PROVIDER")
ALLOWED_STAGES = frozenset({"RUNNING", "BUILDING", "SLEEPING"})
EXPECTED_COLLECTION_BASE = "labor_laws_20260830_3ec5ade"


def _value_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).rsplit(".", 1)[-1]


def evaluate_space_policy(
    *,
    private: bool,
    stage: str,
    current_hardware: str,
    requested_hardware: str,
    secret_names: Collection[str],
    variable_names: Collection[str],
    collection_base: str,
) -> dict[str, Any]:
    """Return a deterministic policy report containing names and status only."""
    normalized_stage = _value_text(stage).upper()
    normalized_current = _value_text(current_hardware).lower()
    normalized_requested = _value_text(requested_hardware).lower()
    normalized_secrets = sorted(str(name) for name in secret_names)
    normalized_variables = sorted(str(name) for name in variable_names)
    violations: list[dict[str, str]] = []

    if not private:
        violations.append({"code": "public_visibility"})
    if normalized_current != "cpu-basic" or normalized_requested != "cpu-basic":
        violations.append({"code": "paid_hardware"})
    if normalized_stage not in ALLOWED_STAGES:
        violations.append({"code": "invalid_stage"})

    owner_key_names = {
        name
        for name in normalized_secrets
        if any(marker in name.upper() for marker in OWNER_KEY_MARKERS)
    }
    if owner_key_names:
        violations.append({"code": "owner_model_key"})
    if set(normalized_secrets) - ALLOWED_SECRET_NAMES - owner_key_names:
        violations.append({"code": "unexpected_secret"})
    if collection_base != EXPECTED_COLLECTION_BASE:
        violations.append({"code": "collection_base"})

    return {
        "passed": not violations,
        "private": bool(private),
        "stage": normalized_stage,
        "ready": normalized_stage == "RUNNING",
        "current_hardware": normalized_current,
        "requested_hardware": normalized_requested,
        "secret_names": normalized_secrets,
        "variable_names": normalized_variables,
        "collection_base": collection_base,
        "violations": violations,
    }


def inspect_space(api: Any, repo_id: str) -> dict[str, Any]:
    """Read Space metadata without reading or serializing any secret value."""
    info = api.repo_info(repo_id, repo_type="space")
    runtime = api.get_space_runtime(repo_id)
    variables: Mapping[str, Any] = api.get_space_variables(repo_id)
    secrets: Mapping[str, Any] = api.get_space_secrets(repo_id)
    collection_variable = variables.get("COLLECTION_NAME")
    collection_base = (
        str(getattr(collection_variable, "value", "")).strip()
        if collection_variable is not None
        else ""
    )
    return evaluate_space_policy(
        private=bool(getattr(info, "private", False)),
        stage=_value_text(getattr(runtime, "stage", "")),
        current_hardware=_value_text(getattr(runtime, "hardware", "")),
        requested_hardware=_value_text(
            getattr(runtime, "requested_hardware", "")
        ),
        secret_names=set(secrets),
        variable_names=set(variables),
        collection_base=collection_base,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[[], Any] = HfApi,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_space(api_factory(), args.repo_id)
    except Exception as exc:
        error = {
            "status": "error",
            "error_class": type(exc).__name__,
            "message": "space metadata unavailable",
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
