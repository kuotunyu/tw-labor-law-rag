"""Verify the public source-only release contract without models or providers."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from rag.release_verification import verify_release


def main() -> None:
    report = verify_release(Path(__file__).resolve().parents[1])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
