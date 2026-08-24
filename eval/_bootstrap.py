"""Make ``src/`` importable for eval scripts without installing the package.

(Same rationale as scripts/_bootstrap.py: editable installs break on
non-ASCII project paths with Python <3.13.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
