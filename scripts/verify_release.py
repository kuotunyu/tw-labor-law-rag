"""Verify the public source-only release contract without models or providers."""

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    v036_artifact = (
        project_root / "eval/official/severance_refusal_policy_v0.3.6.json"
    )
    if v036_artifact.is_file():
        raise SystemExit(
            "v0.3.6 replay must start with: python -I -S "
            "scripts/v036_authoritative_bootstrap.py --project-root . "
            "--environment-root <external-environment> --mode verify-release"
        )

    import _bootstrap  # noqa: F401, PLC0415

    from rag.release_verification import verify_release  # noqa: PLC0415

    report = verify_release(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
