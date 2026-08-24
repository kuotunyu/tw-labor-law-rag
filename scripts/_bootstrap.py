"""Make ``src/`` importable for scripts without installing the package.

(An editable install would write the project path into a .pth file, which
Python <3.13 decodes with the locale encoding and crashes on non-ASCII paths —
e.g. Chinese folder names on zh-TW Windows.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
