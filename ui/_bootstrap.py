"""Restore repository import roots when Streamlit executes ``ui/app.py``."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_roots() -> None:
    """Make both top-level UI and ``src`` packages importable."""

    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


ensure_import_roots()
