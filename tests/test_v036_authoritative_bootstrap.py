from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = PROJECT_ROOT / "scripts/v036_authoritative_bootstrap.py"
RELEASE_WRAPPER = PROJECT_ROOT / "scripts/verify_release.py"
FIXED_FILES = {
    ".python-version": "3.11\n",
    "Dockerfile": "FROM python:3.11-slim\n",
    "pyproject.toml": "[project]\nname='fixture'\nversion='1'\n",
    "uv.lock": "version = 1\nrevision = 3\nrequires-python = '>=3.11'\npackage = []\n",
    "src/rag/indexing/dict/legal_terms.txt": "term\n",
}
DECLARED_INPUTS = {
    "corpus_snapshot": "release/corpus_snapshot.json",
    "formal_dataset": "eval/dataset/eval_set.jsonl",
    "stress_dataset": "eval/dataset/reliability_stress_v0.3.1.jsonl",
    "target_dataset": "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
}


def _run(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=False,
    )


def _write(repository: Path, relative_path: str, payload: str) -> None:
    path = repository / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _commit(repository: Path, message: str = "fixture") -> str:
    _run("git", "add", "-A", cwd=repository)
    _run("git", "commit", "-m", message, cwd=repository)
    return _run("git", "rev-parse", "HEAD", cwd=repository).stdout.decode().strip()


@pytest.fixture(scope="module")
def bootstrap_module():
    assert BOOTSTRAP_PATH.is_file(), "Task 5 requires a committed stdlib-only bootstrap"
    spec = importlib.util.spec_from_file_location(
        "v036_authoritative_bootstrap", BOOTSTRAP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run("git", "init", cwd=repository)
    _run("git", "config", "user.email", "fixture@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "Fixture", cwd=repository)
    _run("git", "config", "core.autocrlf", "false", cwd=repository)
    _write(repository, ".gitignore", "*.pyc\n__pycache__/\n*.PTH\n")
    for path, payload in FIXED_FILES.items():
        _write(repository, path, payload)
    for label, path in DECLARED_INPUTS.items():
        _write(repository, path, f"{label}\n")
    _write(repository, "src/pkg/__init__.py", "VALUE = 1\n")
    _write(repository, "eval/Runner.PY", "VALUE = 2\n")
    _write(repository, "scripts/tool.py", "VALUE = 3\n")
    _write(repository, "tests/test_fixture.py", "def test_fixture(): pass\n")
    _write(repository, "docs/unbound.md", "not decision code\n")
    _commit(repository)
    return repository


def _binding(module, repository: Path) -> dict:
    revision = _run("git", "rev-parse", "HEAD", cwd=repository).stdout.decode().strip()
    return module.build_revision_binding(repository, revision, DECLARED_INPUTS)


def test_revision_binding_uses_exact_full_git_tree_metadata_and_casefolded_python_suffix(
    bootstrap_module, git_repository: Path
) -> None:
    binding = _binding(bootstrap_module, git_repository)

    paths = [entry["path"] for entry in binding["tracked_files"]]
    assert paths == sorted(
        [
            ".python-version",
            "Dockerfile",
            "eval/Runner.PY",
            "pyproject.toml",
            "scripts/tool.py",
            "src/pkg/__init__.py",
            "src/rag/indexing/dict/legal_terms.txt",
            "tests/test_fixture.py",
            "uv.lock",
        ]
    )
    assert set(binding) == {
        "format_version",
        "revision",
        "tracked_files",
        "declared_inputs",
    }
    assert (
        binding["revision"]
        == _run("git", "rev-parse", "HEAD", cwd=git_repository).stdout.decode().strip()
    )
    for entry in [*binding["tracked_files"], *binding["declared_inputs"]]:
        assert (
            set(entry) == {"label", "path", "mode", "object_type", "blob_oid", "sha256"}
            if "label" in entry
            else {
                "path",
                "mode",
                "object_type",
                "blob_oid",
                "sha256",
            }
        )
        assert entry["mode"] in {"100644", "100755"}
        assert entry["object_type"] == "blob"
        assert len(entry["blob_oid"]) == 40
        assert len(entry["sha256"]) == 64

    assert (
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )
        == binding
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "../escape.py", "canonical POSIX path"),
        ("mode", "120000", "regular file mode"),
        ("object_type", "commit", "blob object type"),
        ("blob_oid", "0" * 40, "recorded Git tree"),
        ("sha256", "0" * 64, "recorded Git tree"),
    ],
)
def test_revision_binding_rejects_path_mode_type_blob_and_sha_inconsistency(
    bootstrap_module,
    git_repository: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    binding["tracked_files"][0][field] = value

    with pytest.raises(ValueError, match=message):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


@pytest.mark.parametrize("mutation", ["duplicate", "casefold", "extra", "missing"])
def test_revision_binding_rejects_duplicate_collision_extra_and_missing_bindings(
    bootstrap_module, git_repository: Path, mutation: str
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    if mutation == "duplicate":
        binding["tracked_files"].append(dict(binding["tracked_files"][0]))
    elif mutation == "casefold":
        collision = dict(binding["tracked_files"][0])
        collision["path"] = collision["path"].swapcase()
        binding["tracked_files"].append(collision)
    elif mutation == "extra":
        extra = dict(binding["tracked_files"][0])
        extra["path"] = "extra.py"
        binding["tracked_files"].append(extra)
    else:
        binding["tracked_files"].pop()

    with pytest.raises(ValueError, match="tracked-code binding"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


@pytest.mark.parametrize("record_kind", ["tree", "index"])
def test_nul_git_metadata_rejects_nonselected_gitlinks_before_filtering(
    bootstrap_module, record_kind: str
) -> None:
    oid = b"a" * 40
    if record_kind == "tree":
        payload = b"160000 commit " + oid + b"\tvendor/dependency\0"
        parser = bootstrap_module._parse_tree_records
    else:
        payload = b"160000 " + oid + b" 0\tvendor/dependency\0"
        parser = bootstrap_module._parse_index_records

    with pytest.raises(ValueError, match="gitlink|submodule"):
        parser(payload)


def _create_local_dependency_repository(path: Path) -> str:
    path.mkdir()
    _run("git", "init", cwd=path)
    _run("git", "config", "user.email", "tests@example.invalid", cwd=path)
    _run("git", "config", "user.name", "Test Fixture", cwd=path)
    path.joinpath("README.md").write_text("local dependency\n", encoding="utf-8")
    _run("git", "add", "README.md", cwd=path)
    _run("git", "commit", "-m", "fixture", cwd=path)
    return _run("git", "rev-parse", "HEAD", cwd=path).stdout.decode().strip()


def test_revision_binding_rejects_real_nonselected_local_submodule_in_recorded_and_head_tree(
    bootstrap_module, git_repository: Path, tmp_path: Path
) -> None:
    recorded_binding = _binding(bootstrap_module, git_repository)
    dependency = tmp_path / "local-dependency"
    _create_local_dependency_repository(dependency)
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(dependency),
        "vendor/dependency",
        cwd=git_repository,
    )
    _commit(git_repository, "local submodule")
    current_revision = (
        _run("git", "rev-parse", "HEAD", cwd=git_repository).stdout.decode().strip()
    )

    with pytest.raises(ValueError, match="gitlink|submodule"):
        bootstrap_module.build_revision_binding(
            git_repository, current_revision, DECLARED_INPUTS
        )
    with pytest.raises(ValueError, match="gitlink|submodule"):
        bootstrap_module.verify_revision_binding(
            git_repository, recorded_binding, DECLARED_INPUTS
        )


def test_revision_binding_rejects_nonselected_gitlink_staged_in_current_index(
    bootstrap_module, git_repository: Path, tmp_path: Path
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    dependency_revision = _create_local_dependency_repository(
        tmp_path / "local-dependency"
    )
    _run(
        "git",
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{dependency_revision},vendor/staged-dependency",
        cwd=git_repository,
    )

    with pytest.raises(ValueError, match="gitlink|submodule"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


@pytest.mark.parametrize("mutation", ["added", "removed", "renamed", "changed", "mode"])
def test_revision_binding_rejects_current_head_tree_drift(
    bootstrap_module, git_repository: Path, mutation: str
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    target = git_repository / "scripts/tool.py"
    if mutation == "added":
        _write(git_repository, "src/pkg/added.py", "ADDED = True\n")
    elif mutation == "removed":
        target.unlink()
    elif mutation == "renamed":
        target.rename(git_repository / "scripts/renamed.py")
    elif mutation == "changed":
        target.write_text("VALUE = 99\n", encoding="utf-8")
    else:
        if os.name == "nt":
            pytest.skip(
                "Git executable-mode transitions are not represented on Windows"
            )
        target.chmod(0o755)
    _commit(git_repository, mutation)

    with pytest.raises(ValueError, match="current HEAD tracked-code set"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


def test_revision_binding_rejects_mode_only_drift_in_declared_input(
    bootstrap_module, git_repository: Path
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    _run(
        "git",
        "update-index",
        "--chmod=+x",
        DECLARED_INPUTS["target_dataset"],
        cwd=git_repository,
    )
    _run("git", "commit", "-m", "declared input mode", cwd=git_repository)

    with pytest.raises(ValueError, match="current HEAD declared input"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


@pytest.mark.parametrize(
    "mutation", ["staged_change", "staged_add", "checkout_change", "missing"]
)
def test_revision_binding_rejects_index_checkout_and_sparse_drift(
    bootstrap_module, git_repository: Path, mutation: str
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    target = git_repository / "scripts/tool.py"
    if mutation == "staged_change":
        target.write_text("VALUE = 99\n", encoding="utf-8")
        _run("git", "add", "scripts/tool.py", cwd=git_repository)
    elif mutation == "staged_add":
        _write(git_repository, "scripts/added.py", "ADDED = True\n")
        _run("git", "add", "scripts/added.py", cwd=git_repository)
    elif mutation == "checkout_change":
        target.write_text("VALUE = 99\n", encoding="utf-8")
    else:
        target.unlink()

    with pytest.raises(ValueError, match="index|checkout|clean tree"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/evil.PY",
        "eval/evil.pyc",
        "eval/evil.pyo",
        "scripts/evil.pyd",
        "src/evil.so",
        "eval/evil.dll",
        "scripts/evil.PTH",
        "src/evil.zip",
        "eval/evil.egg",
        "scripts/evil.whl",
        "src/evil.pyz",
    ],
)
def test_revision_binding_rejects_untracked_or_ignored_importable_artifacts(
    bootstrap_module, git_repository: Path, relative_path: str
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    _write(git_repository, relative_path, "untrusted\n")
    before = {
        path.relative_to(git_repository).as_posix(): path.read_bytes()
        for path in git_repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }

    with pytest.raises(ValueError, match="untracked or ignored importable"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )

    after = {
        path.relative_to(git_repository).as_posix(): path.read_bytes()
        for path in git_repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    assert after == before


@pytest.mark.parametrize(
    "cache_path", ["src/__pycache__", "eval/.pytest_cache", "scripts/CACHE"]
)
def test_revision_binding_rejects_cache_trees(
    bootstrap_module, git_repository: Path, cache_path: str
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    _write(git_repository, f"{cache_path}/artifact.bin", "cache\n")

    with pytest.raises(ValueError, match="cache tree"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


def test_revision_binding_rejects_dirty_unbound_files_and_declared_input_drift(
    bootstrap_module, git_repository: Path
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    _write(git_repository, "docs/dirty.md", "dirty but not importable\n")
    with pytest.raises(ValueError, match="clean tree"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )

    _run("git", "clean", "-f", "docs/dirty.md", cwd=git_repository)
    changed = git_repository / DECLARED_INPUTS["target_dataset"]
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declared input|checkout|clean tree"):
        bootstrap_module.verify_revision_binding(
            git_repository, binding, DECLARED_INPUTS
        )


def test_bootstrap_source_is_stdlib_only_and_has_no_project_import_at_load_time() -> (
    None
):
    assert BOOTSTRAP_PATH.is_file(), "Task 5 requires a committed stdlib-only bootstrap"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(BOOTSTRAP_PATH), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--environment-root" in completed.stdout
    assert "--project-root" in completed.stdout
    assert "--mode" in completed.stdout
    assert "--binding" in completed.stdout


def test_binding_json_is_canonical_and_contains_no_absolute_paths(
    bootstrap_module, git_repository: Path
) -> None:
    binding = _binding(bootstrap_module, git_repository)
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))

    assert str(git_repository) not in encoded
    assert "\\\\" not in encoded


def _environment_python(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Scripts/python.exe"
    return environment_root / "bin/python"


def _site_packages(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Lib/site-packages"
    return (
        environment_root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )


def _create_environment(path: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def _minimal_lock(*, dependencies: str = "", packages: str = "", dev: str = "") -> str:
    dependency_block = f"dependencies = [{dependencies}]\n" if dependencies else ""
    dev_block = f"dev-dependencies = {{ dev = [{dev}] }}\n" if dev else ""
    return (
        "version = 1\n"
        "revision = 3\n"
        "requires-python = '>=3.11'\n\n"
        "[[package]]\n"
        "name = 'fixture'\n"
        "version = '1'\n"
        "source = { virtual = '.' }\n"
        f"{dependency_block}{dev_block}"
        f"{packages}"
    )


def _record_command(repository: Path, environment_root: Path) -> list[str]:
    return [
        str(_environment_python(environment_root)),
        "-I",
        "-S",
        str(BOOTSTRAP_PATH),
        "--project-root",
        str(repository),
        "--environment-root",
        str(environment_root),
        "--mode",
        "record",
    ]


def _record(
    repository: Path,
    environment_root: Path,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    process_environment = os.environ.copy() if environment is None else environment
    process_environment.pop("PYTHONPATH", None)
    return subprocess.run(
        _record_command(repository, environment_root),
        cwd=repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _replace_lock(repository: Path, payload: str) -> None:
    (repository / "uv.lock").write_text(payload, encoding="utf-8")
    _commit(repository, "lock fixture")


def _install_metadata(
    site: Path, name: str, version: str, *, dirname: str | None = None
) -> None:
    dist_info = site / (dirname or f"{name}-{version}.dist-info")
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_record_mode_requires_isolated_no_site_external_environment_before_project_import(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    import_marker = git_repository / "project-imported.marker"
    _write(
        git_repository,
        "src/rag/factory.py",
        "from pathlib import Path\nPath('project-imported.marker').write_text('imported')\n",
    )
    _commit(git_repository, "import sentinel")

    missing_isolation = subprocess.run(
        [
            str(_environment_python(environment_root)),
            "-S",
            str(BOOTSTRAP_PATH),
            "--project-root",
            str(git_repository),
            "--environment-root",
            str(environment_root),
            "--mode",
            "record",
        ],
        cwd=git_repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    missing_no_site = subprocess.run(
        [
            argument
            for argument in _record_command(git_repository, environment_root)
            if argument != "-S"
        ],
        cwd=git_repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert missing_isolation.returncode != 0
    assert "-I" in missing_isolation.stderr
    assert missing_no_site.returncode != 0
    assert "-S" in missing_no_site.stderr
    assert not import_marker.exists()


def test_record_mode_rejects_pythonpath_even_when_isolated_mode_would_ignore_it(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = str(tmp_path / "injection")
    completed = subprocess.run(
        _record_command(git_repository, environment_root),
        cwd=git_repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "PYTHONPATH" in completed.stderr


@pytest.mark.parametrize("source", ["python-version", "lock-requires-python"])
def test_record_mode_rejects_interpreter_version_contract_drift_before_import(
    git_repository: Path, tmp_path: Path, source: str
) -> None:
    incompatible_major_minor = "3.12" if sys.version_info[:2] != (3, 12) else "3.11"
    if source == "python-version":
        git_repository.joinpath(".python-version").write_text(
            f"{incompatible_major_minor}\n", encoding="utf-8"
        )
        _replace_lock(git_repository, _minimal_lock())
    else:
        incompatible_requirement = (
            ">=3.12" if sys.version_info[:2] < (3, 12) else "<3.12"
        )
        _replace_lock(
            git_repository,
            _minimal_lock().replace(">=3.11", incompatible_requirement, 1),
        )
    environment_root = _create_environment(tmp_path / "authority-env")
    import_marker = git_repository / "project-imported.marker"
    _write(
        git_repository,
        "src/rag/factory.py",
        "from pathlib import Path\nPath('project-imported.marker').write_text('imported')\n",
    )
    _commit(git_repository, "import sentinel")

    completed = _record(git_repository, environment_root)

    assert completed.returncode != 0
    expected_source = (
        ".python-version" if source == "python-version" else "requires-python"
    )
    assert expected_source in completed.stderr
    assert not import_marker.exists()


def test_record_mode_binds_privacy_safe_external_interpreter_and_empty_inventory(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    completed = _record(git_repository, environment_root)

    assert completed.returncode == 0, completed.stderr
    attestation = json.loads(completed.stdout)
    environment = attestation["environment_binding"]
    assert set(environment) == {
        "format_version",
        "interpreter",
        "pyvenv",
        "site_layout",
        "lock_selection",
        "installed_distributions",
    }
    assert environment["interpreter"]["implementation"] == "cpython"
    assert environment["interpreter"]["full_version"] == ".".join(
        str(value) for value in sys.version_info[:3]
    )
    assert environment["pyvenv"] == {"include_system_site_packages": False}
    assert environment["installed_distributions"] == []
    assert environment["lock_selection"]["offline"] is True
    assert environment["lock_selection"]["frozen"] is True
    assert environment["lock_selection"]["no_dev"] is True
    assert environment["lock_selection"]["selected_dependency_groups"] == []
    encoded = json.dumps(environment, sort_keys=True)
    assert str(environment_root) not in encoded
    assert str(git_repository) not in encoded
    assert (environment_root / "pyvenv.cfg").read_text(encoding="utf-8") not in encoded


def test_record_mode_rejects_environment_inside_repository_and_pyvenv_relationship_drift(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    inside = _create_environment(git_repository / ".authority-env")
    inside_result = _record(git_repository, inside)
    assert inside_result.returncode != 0
    assert "outside the repository" in inside_result.stderr

    shutil.rmtree(inside)
    _run("git", "clean", "-fd", cwd=git_repository)
    environment_root = _create_environment(tmp_path / "authority-env")
    config = environment_root / "pyvenv.cfg"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        ),
        encoding="utf-8",
    )
    drifted = _record(git_repository, environment_root)
    assert drifted.returncode != 0
    assert "system site" in drifted.stderr


def test_record_mode_rejects_unapproved_prebootstrap_path_and_does_not_process_pth(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    injected = tmp_path / "injected"
    injected.mkdir()
    command = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(injected)!r});"
        f"sys.argv={_record_command(git_repository, environment_root)[3:]!r};"
        f"runpy.run_path({str(BOOTSTRAP_PATH)!r},run_name='__main__')"
    )
    rejected = subprocess.run(
        [str(_environment_python(environment_root)), "-I", "-S", "-c", command],
        cwd=git_repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert rejected.returncode != 0
    assert "sys.path" in rejected.stderr

    marker = tmp_path / "sitecustomize-imported.marker"
    injected.joinpath("sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    _site_packages(environment_root).mkdir(parents=True, exist_ok=True)
    _site_packages(environment_root).joinpath("injected.pth").write_text(
        str(injected), encoding="utf-8"
    )
    accepted = _record(git_repository, environment_root)
    assert accepted.returncode == 0, accepted.stderr
    assert not marker.exists()


def _directory_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_record_mode_rejects_approved_site_through_symlinked_parent_without_touching_target(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    site_parent = _site_packages(environment_root).parent
    external_target = tmp_path / "external-site-parent"
    site_parent.rename(external_target)
    external_target.joinpath("sentinel.txt").write_text("untouched\n", encoding="utf-8")
    try:
        site_parent.symlink_to(external_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"OS denied directory symlink creation: {exc}")
    before = _directory_snapshot(external_target)

    completed = _record(git_repository, environment_root)

    assert completed.returncode != 0
    assert "site-packages" in completed.stderr and "alias" in completed.stderr
    assert _directory_snapshot(external_target) == before


def test_record_mode_rejects_approved_site_windows_junction_without_touching_target(
    git_repository: Path, tmp_path: Path
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows junctions are unavailable on this platform")
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    site_path = _site_packages(environment_root)
    site_path.rename(environment_root / "original-site-packages")
    external_target = tmp_path / "junction-site-target"
    external_target.mkdir()
    external_target.joinpath("sentinel.txt").write_text("untouched\n", encoding="utf-8")
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(site_path), str(external_target)],
        check=False,
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("OS denied junction creation")
    before = _directory_snapshot(external_target)

    completed = _record(git_repository, environment_root)

    assert completed.returncode != 0
    assert "site-packages" in completed.stderr and "alias" in completed.stderr
    assert _directory_snapshot(external_target) == before


def test_approved_site_validator_rejects_portable_windows_reparse_seam(
    bootstrap_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validator = getattr(bootstrap_module, "_validated_approved_sites", None)
    assert callable(validator), "bootstrap must expose approved-site alias validation"
    environment_root = tmp_path / "authority-env"
    _site_packages(environment_root).mkdir(parents=True)
    monkeypatch.setattr(bootstrap_module, "_is_alias", lambda _path_stat: True)

    with pytest.raises(ValueError, match="site-packages.*alias"):
        validator(environment_root)


def test_approved_site_validator_requires_resolved_containment_when_alias_seam_misses(
    bootstrap_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validator = getattr(bootstrap_module, "_validated_approved_sites", None)
    assert callable(validator), (
        "bootstrap must expose approved-site containment validation"
    )
    environment_root = tmp_path / "authority-env"
    site_path = _site_packages(environment_root)
    site_path.parent.mkdir(parents=True)
    external_target = tmp_path / "external-site"
    external_target.mkdir()
    external_target.joinpath("sentinel.txt").write_text("untouched\n", encoding="utf-8")
    if sys.platform == "win32":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(site_path), str(external_target)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            pytest.skip("OS denied junction creation")
    else:
        try:
            site_path.symlink_to(external_target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"OS denied directory symlink creation: {exc}")
    before = _directory_snapshot(external_target)
    monkeypatch.setattr(bootstrap_module, "_is_alias", lambda _path_stat: False)

    with pytest.raises(ValueError, match="site-packages.*outside environment"):
        validator(environment_root)

    assert _directory_snapshot(external_target) == before


def test_lock_markers_select_one_exact_version_and_exclude_dev_group(
    git_repository: Path, tmp_path: Path
) -> None:
    packages = """
[[package]]
name = 'chosen'
version = '1.0'
resolution-markers = ["python_full_version < '3.12'"]

[[package]]
name = 'chosen'
version = '2.0'
resolution-markers = ["python_full_version >= '3.12'"]

[[package]]
name = 'dev-only'
version = '9.0'
"""
    _replace_lock(
        git_repository,
        _minimal_lock(
            dependencies="{ name = 'chosen', version = '1.0', marker = \"python_full_version < '3.12'\" }, { name = 'chosen', version = '2.0', marker = \"python_full_version >= '3.12'\" }",
            packages=packages,
            dev="{ name = 'dev-only' }",
        ),
    )
    environment_root = _create_environment(tmp_path / "authority-env")
    expected_version = "1.0" if sys.version_info < (3, 12) else "2.0"
    _install_metadata(_site_packages(environment_root), "Chosen", expected_version)
    completed = _record(git_repository, environment_root)

    assert completed.returncode == 0, completed.stderr
    binding = json.loads(completed.stdout)["environment_binding"]
    assert binding["installed_distributions"] == [
        {"name": "chosen", "version": expected_version}
    ]
    assert binding["lock_selection"]["selected_packages"] == [
        {"name": "chosen", "version": expected_version}
    ]
    assert binding["lock_selection"]["excluded_dependency_groups"] == ["dev"]
    assert binding["lock_selection"]["markers"]["python_full_version"].startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )


def test_repository_lock_selects_a_duplicate_free_production_only_inventory(
    bootstrap_module,
) -> None:
    import tomllib

    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    selected, excluded_groups, active_markers = (
        bootstrap_module._select_locked_inventory(
            lock, bootstrap_module._marker_environment()
        )
    )
    names = [entry["name"] for entry in selected]

    assert names == sorted(set(names))
    assert {"flagembedding", "torch", "transformers"} <= set(names)
    assert {"bandit", "pip-audit", "pytest", "ruff"}.isdisjoint(names)
    assert excluded_groups == ["dev"]
    assert len(active_markers) == 1


@pytest.mark.parametrize("drift", ["missing", "extra", "version", "duplicate", "dev"])
def test_record_mode_rejects_exact_inventory_drift_and_pep503_duplicates(
    git_repository: Path, tmp_path: Path, drift: str
) -> None:
    packages = """
[[package]]
name = 'foo-bar'
version = '1.0'

[[package]]
name = 'dev-only'
version = '9.0'
"""
    _replace_lock(
        git_repository,
        _minimal_lock(
            dependencies="{ name = 'foo-bar' }",
            packages=packages,
            dev="{ name = 'dev-only' }",
        ),
    )
    environment_root = _create_environment(tmp_path / "authority-env")
    site = _site_packages(environment_root)
    if drift != "missing":
        _install_metadata(site, "Foo_Bar", "2.0" if drift == "version" else "1.0")
    if drift == "extra":
        _install_metadata(site, "unexpected", "1.0")
    elif drift == "duplicate":
        _install_metadata(site, "foo.bar", "1.0", dirname="duplicate-1.0.dist-info")
    elif drift == "dev":
        _install_metadata(site, "dev-only", "9.0")

    completed = _record(git_repository, environment_root)

    assert completed.returncode != 0
    assert "inventory" in completed.stderr or "duplicate" in completed.stderr


def test_environment_binding_rejects_interpreter_lock_marker_and_inventory_tampering(
    bootstrap_module, git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    environment_root = _create_environment(tmp_path / "authority-env")
    completed = _record(git_repository, environment_root)
    assert completed.returncode == 0, completed.stderr
    actual = json.loads(completed.stdout)["environment_binding"]

    for path, value in (
        (("interpreter", "abi"), "tampered"),
        (("lock_selection", "lock_sha256"), "0" * 64),
        (("lock_selection", "markers", "sys_platform"), "tampered"),
        (("installed_distributions",), [{"name": "extra", "version": "1"}]),
    ):
        expected = json.loads(json.dumps(actual))
        target = expected
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="environment binding"):
            bootstrap_module.verify_environment_binding(actual, expected)


def test_calibration_entrypoint_imports_project_only_after_all_bindings_pass(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    marker = tmp_path / "calibration-imported.json"
    _write(
        git_repository,
        "eval/run_severance_refusal_policy.py",
        """import json
import os
from pathlib import Path

def main(argv, *, trusted_runtime):
    Path(os.environ['BOOTSTRAP_TEST_MARKER']).write_text(
        json.dumps({'argv': argv, 'runtime': sorted(trusted_runtime)}),
        encoding='utf-8',
    )
    return 0
""",
    )
    _commit(git_repository, "calibration entrypoint")
    environment_root = _create_environment(tmp_path / "authority-env")
    process_environment = os.environ.copy()
    process_environment.pop("PYTHONPATH", None)
    process_environment["BOOTSTRAP_TEST_MARKER"] = str(marker)
    command = _record_command(git_repository, environment_root)
    command[command.index("record")] = "calibrate"
    command.extend(["--", "--offline", "--work-dir", str(tmp_path / "work")])

    accepted = subprocess.run(
        command,
        cwd=git_repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "argv": ["--offline", "--work-dir", str(tmp_path / "work")],
        "runtime": ["environment_binding", "revision_binding"],
    }

    marker.unlink()
    _write(git_repository, "src/ignored.PY", "UNTRUSTED = True\n")
    rejected = subprocess.run(
        command,
        cwd=git_repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert rejected.returncode != 0
    assert "untracked or ignored importable" in rejected.stderr
    assert not marker.exists()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    ("option", "value_kind", "option_form"),
    [
        ("--dataset", "relative", "separate"),
        ("--stress-dataset", "absolute", "equals"),
        ("--formal-dataset", "traversal", "separate"),
        ("--snapshot", "alias", "equals"),
        ("--data", "relative", "equals"),
        ("--stress", "absolute", "separate"),
        ("--formal", "traversal", "equals"),
        ("--snap", "alias", "separate"),
    ],
)
def test_calibration_rejects_all_input_overrides_before_project_import_or_work(
    git_repository: Path,
    tmp_path: Path,
    option: str,
    value_kind: str,
    option_form: str,
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    import_marker = tmp_path / "project-imported.marker"
    invocation_marker = tmp_path / "runner-invoked.marker"
    _write(
        git_repository,
        "eval/run_severance_refusal_policy.py",
        """import os
from pathlib import Path

Path(os.environ['BOOTSTRAP_IMPORT_MARKER']).write_text('imported')

def main(argv, *, trusted_runtime):
    Path(os.environ['BOOTSTRAP_INVOCATION_MARKER']).write_text('invoked')
    return 0
""",
    )
    _commit(git_repository, "calibration override sentinel")
    environment_root = _create_environment(tmp_path / "authority-env")
    work_dir = tmp_path / "work"
    values = {
        "relative": "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
        "absolute": str(tmp_path / "outside-stress.jsonl"),
        "traversal": "eval/dataset/../dataset/eval_set.jsonl",
        "alias": "release/./corpus_snapshot.json",
    }
    override = (
        [option, values[value_kind]]
        if option_form == "separate"
        else [f"{option}={values[value_kind]}"]
    )
    command = _record_command(git_repository, environment_root)
    command[command.index("record")] = "calibrate"
    command.extend(["--", "--offline", *override, "--work-dir", str(work_dir)])
    process_environment = os.environ.copy()
    process_environment.pop("PYTHONPATH", None)
    process_environment["BOOTSTRAP_IMPORT_MARKER"] = str(import_marker)
    process_environment["BOOTSTRAP_INVOCATION_MARKER"] = str(invocation_marker)

    completed = subprocess.run(
        command,
        cwd=git_repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "authoritative input" in completed.stderr
    assert not import_marker.exists()
    assert not invocation_marker.exists()
    assert not work_dir.exists()


def test_artifact_verification_entrypoint_passes_exact_runtime_after_validation(
    git_repository: Path, tmp_path: Path
) -> None:
    _replace_lock(git_repository, _minimal_lock())
    marker = tmp_path / "verified.json"
    _write(git_repository, "src/rag/__init__.py", "")
    _write(
        git_repository,
        "src/rag/release_verification.py",
        """import json
import os
from pathlib import Path

def _verify_severance_refusal_policy_artifact(project_root, artifact_path, *, trusted_runtime):
    Path(os.environ['BOOTSTRAP_TEST_MARKER']).write_text(
        json.dumps({'project': project_root.name, 'artifact': artifact_path.name,
                    'runtime': sorted(trusted_runtime)}),
        encoding='utf-8',
    )
    return {'verified': True}
""",
    )
    _commit(git_repository, "verification entrypoint")
    environment_root = _create_environment(tmp_path / "authority-env")
    recorded = _record(git_repository, environment_root)
    assert recorded.returncode == 0, recorded.stderr
    runtime = json.loads(recorded.stdout)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "provenance": {
                    "revision_binding": runtime["revision_binding"],
                    "environment_binding": runtime["environment_binding"],
                }
            }
        ),
        encoding="utf-8",
    )
    process_environment = os.environ.copy()
    process_environment.pop("PYTHONPATH", None)
    process_environment["BOOTSTRAP_TEST_MARKER"] = str(marker)
    completed = subprocess.run(
        [
            str(_environment_python(environment_root)),
            "-I",
            "-S",
            str(BOOTSTRAP_PATH),
            "--project-root",
            str(git_repository),
            "--environment-root",
            str(environment_root),
            "--mode",
            "verify-artifact",
            "--binding",
            str(artifact),
        ],
        cwd=git_repository,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"verified": True}
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "project": "repository",
        "artifact": "artifact.json",
        "runtime": ["environment_binding", "revision_binding"],
    }


def test_release_wrapper_defers_v036_to_bootstrap_before_any_project_import(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    official = repository / "eval/official"
    scripts.mkdir(parents=True)
    official.mkdir(parents=True)
    shutil.copyfile(RELEASE_WRAPPER, scripts / "verify_release.py")
    import_marker = tmp_path / "legacy-bootstrap-imported.marker"
    (scripts / "_bootstrap.py").write_text(
        f"from pathlib import Path\nPath({str(import_marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    (official / "severance_refusal_policy_v0.3.6.json").write_text(
        "{}\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, str(scripts / "verify_release.py")],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "v036_authoritative_bootstrap.py" in completed.stderr
    assert "--environment-root" in completed.stderr
    assert not import_marker.exists()
