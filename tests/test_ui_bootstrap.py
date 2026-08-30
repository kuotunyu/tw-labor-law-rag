"""Regression tests for launching the Streamlit file as a script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_ui_bootstrap_restores_the_repository_import_root() -> None:
    """The Streamlit entrypoint can import ``ui.*`` from its script directory."""

    probe = r"""
import sys
from pathlib import Path

root = Path.cwd().resolve()
ui_dir = root / "ui"
sys.path = [str(ui_dir)] + [
    entry
    for entry in sys.path
    if entry and Path(entry).resolve() != root
]

from _bootstrap import ensure_import_roots

ensure_import_roots()
import ui.api_client
import ui.content
print("ui bootstrap: ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ui bootstrap: ok"
