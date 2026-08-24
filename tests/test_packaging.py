import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = "rag/indexing/dict/legal_terms.txt"


def test_fastapi_import_has_no_dependency_warnings():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    process = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", "import rag.api.main"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stdout + process.stderr


def test_built_distributions_include_legal_terms_dictionary(tmp_path):
    process = subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    wheels = list(tmp_path.glob("*.whl"))
    source_distributions = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_distributions) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        assert LEGAL_TERMS in archive.namelist()

    with tarfile.open(source_distributions[0], "r:gz") as archive:
        assert any(
            name.endswith(f"/src/{LEGAL_TERMS}") for name in archive.getnames()
        )
