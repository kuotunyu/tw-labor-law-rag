"""Read-only, redacted verification for the scoped Qdrant runtime key."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

_BASE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")


def _valid_base(value: str) -> bool:
    return bool(_BASE_PATTERN.fullmatch(value))


def verify_reader(
    client: httpx.Client,
    *,
    candidate_base: str,
    legacy_base: str,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Read candidate counts and prove the same key cannot read the legacy pair."""
    if not _valid_base(candidate_base) or not _valid_base(legacy_base):
        raise ValueError("invalid collection base")
    if set(expected_counts) != {"fixed", "structure"} or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in expected_counts.values()
    ):
        raise ValueError("invalid expected point counts")

    candidate: dict[str, dict[str, Any]] = {}
    legacy_denied: dict[str, bool] = {}
    violations: list[dict[str, str]] = []
    for suffix in ("fixed", "structure"):
        expected = expected_counts[suffix]
        response = client.get(f"/collections/{candidate_base}_{suffix}")
        actual: int | None = None
        readable = response.status_code == 200
        if readable:
            try:
                candidate_value = response.json()["result"]["points_count"]
                if isinstance(candidate_value, bool) or not isinstance(
                    candidate_value, int
                ):
                    raise TypeError
                actual = candidate_value
            except (KeyError, TypeError, ValueError):
                violations.append({"code": "invalid_candidate_response"})
        else:
            violations.append({"code": "candidate_unavailable"})
        if actual is not None and actual != expected:
            violations.append({"code": "point_count_mismatch"})
        candidate[suffix] = {
            "actual": actual,
            "expected": expected,
            "readable": readable,
        }

    for suffix in ("fixed", "structure"):
        response = client.get(f"/collections/{legacy_base}_{suffix}")
        denied = response.status_code in {401, 403}
        legacy_denied[suffix] = denied
        if response.status_code == 200:
            violations.append({"code": "legacy_access_allowed"})
        elif not denied:
            violations.append({"code": "legacy_check_failed"})

    unique_violations = [
        {"code": code}
        for code in sorted({item["code"] for item in violations})
    ]
    return {
        "passed": not unique_violations,
        "candidate_base": candidate_base,
        "legacy_base": legacy_base,
        "candidate": candidate,
        "legacy_denied": legacy_denied,
        "violations": unique_violations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-base", required=True)
    parser.add_argument("--legacy-base", required=True)
    parser.add_argument("--fixed-count", required=True, type=int)
    parser.add_argument("--structure-count", required=True, type=int)
    parser.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> int:
    args = build_parser().parse_args(argv)
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        print(
            json.dumps(
                {"status": "error", "message": "qdrant environment unavailable"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        with client_factory(
            base_url=url.rstrip("/"),
            headers={"api-key": api_key},
            timeout=30.0,
        ) as client:
            report = verify_reader(
                client,
                candidate_base=args.candidate_base,
                legacy_base=args.legacy_base,
                expected_counts={
                    "fixed": args.fixed_count,
                    "structure": args.structure_count,
                },
            )
    except Exception as exc:
        error = {
            "status": "error",
            "error_class": type(exc).__name__,
            "message": "qdrant verification unavailable",
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
