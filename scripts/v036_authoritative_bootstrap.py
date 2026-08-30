"""Validate the v0.3.6 source and environment before project imports.

This file deliberately imports only the Python standard library.  The
authoritative Task 6/Task 7 commands invoke it with ``python -I -S``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

_FORMAT_VERSION = "1"
_FIXED_TRACKED_PATHS = frozenset(
    {
        ".python-version",
        "Dockerfile",
        "pyproject.toml",
        "src/rag/indexing/dict/legal_terms.txt",
        "uv.lock",
    }
)
_CODE_ROOTS = ("src", "eval", "scripts")
_REGULAR_MODES = frozenset({"100644", "100755"})
_IMPORTABLE_SUFFIXES = frozenset(
    {
        ".dll",
        ".egg",
        ".pht",
        ".pth",
        ".py",
        ".pyc",
        ".pyd",
        ".pyo",
        ".pyz",
        ".so",
        ".whl",
        ".zip",
    }
)
_CACHE_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", "cache"}
)
_HEX_OID = re.compile(r"[0-9a-f]{40,64}\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NORMALIZED_DISTRIBUTION = re.compile(r"[-_.]+")
_MARKER_TOKEN = re.compile(
    r"\s*(?:(and|or|not|in)|([A-Za-z_][A-Za-z0-9_]*)|"
    r"(===|==|!=|<=|>=|<|>)|('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")|"
    r"([()]))"
)
_MARKER_KEYS = (
    "implementation_name",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_system",
    "python_full_version",
    "python_version",
    "sys_platform",
)
_WINDOWS_PYWIN32_RUNTIME_LAYOUT = (
    PurePosixPath("Lib/site-packages/win32"),
    PurePosixPath("Lib/site-packages/win32/lib"),
)


def _git(repository: Path, *arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=False,
        capture_output=True,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed ({' '.join(arguments)}): {stderr}")
    return completed.stdout


def _canonical_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("binding path must be a canonical POSIX path")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("binding path must be UTF-8") from exc
    path = PurePosixPath(value)
    if (
        value.startswith("/")
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("binding path must be a canonical POSIX path")
    return value


def _decode_git_path(value: bytes) -> str:
    try:
        decoded = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Git path is not valid UTF-8") from exc
    return _canonical_path(decoded)


def _parse_tree_records(payload: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_record in payload.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            raw_mode, raw_type, raw_oid = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            object_type = raw_type.decode("ascii")
            oid = raw_oid.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("malformed NUL-delimited git tree metadata") from exc
        if not _HEX_OID.fullmatch(oid):
            raise ValueError("malformed Git object identity")
        if mode == "160000" or object_type == "commit":
            raise ValueError("Git tree contains a forbidden gitlink or submodule")
        records.append(
            {
                "path": _decode_git_path(raw_path),
                "mode": mode,
                "object_type": object_type,
                "blob_oid": oid,
            }
        )
    return records


def _parse_index_records(payload: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_record in payload.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            oid = raw_oid.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("malformed NUL-delimited git index metadata") from exc
        if stage != "0":
            raise ValueError("Git index contains an unresolved staged entry")
        if not _HEX_OID.fullmatch(oid):
            raise ValueError("malformed Git index object identity")
        if mode == "160000":
            raise ValueError("Git index contains a forbidden gitlink or submodule")
        records.append(
            {
                "path": _decode_git_path(raw_path),
                "mode": mode,
                "blob_oid": oid,
            }
        )
    return records


def _is_bound_code_path(path: str) -> bool:
    return (
        path in _FIXED_TRACKED_PATHS or PurePosixPath(path).suffix.casefold() == ".py"
    )


def _validate_unique_paths(entries: list[dict[str, str]], *, label: str) -> None:
    exact: set[str] = set()
    casefolded: dict[str, str] = {}
    for entry in entries:
        path = _canonical_path(entry.get("path"))
        if path in exact:
            raise ValueError(f"duplicate {label} path: {path}")
        folded = path.casefold()
        if folded in casefolded:
            raise ValueError(
                f"case-fold collision in {label}: {casefolded[folded]} and {path}"
            )
        exact.add(path)
        casefolded[folded] = path


def _blob_bytes(repository: Path, oid: str) -> bytes:
    return _git(repository, "cat-file", "blob", oid)


def _materialize_tree_entries(
    repository: Path, records: list[dict[str, str]], *, selected_only: bool
) -> list[dict[str, str]]:
    selected = [
        record
        for record in records
        if not selected_only or _is_bound_code_path(record["path"])
    ]
    _validate_unique_paths(selected, label="tracked-code binding")
    materialized: list[dict[str, str]] = []
    for record in selected:
        if record["mode"] not in _REGULAR_MODES:
            raise ValueError(
                f"bound path must have a regular file mode: {record['path']}"
            )
        if record.get("object_type") != "blob":
            raise ValueError(f"bound path must have blob object type: {record['path']}")
        payload = _blob_bytes(repository, record["blob_oid"])
        materialized.append(
            {
                "path": record["path"],
                "mode": record["mode"],
                "object_type": "blob",
                "blob_oid": record["blob_oid"],
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return sorted(materialized, key=lambda entry: entry["path"])


def _tree_entries(
    repository: Path, revision: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records = _parse_tree_records(_git(repository, "ls-tree", "-r", "-z", revision))
    _validate_unique_paths(records, label="Git tree")
    tracked_files = _materialize_tree_entries(repository, records, selected_only=True)
    missing_fixed = sorted(
        _FIXED_TRACKED_PATHS - {entry["path"] for entry in tracked_files}
    )
    if missing_fixed:
        raise ValueError(
            f"recorded Git tree is missing fixed bindings: {missing_fixed}"
        )
    return records, tracked_files


def _declared_entries(
    repository: Path,
    records: list[dict[str, str]],
    declared_inputs: Mapping[str, str],
) -> list[dict[str, str]]:
    if not isinstance(declared_inputs, Mapping) or not declared_inputs:
        raise ValueError("declared inputs must be a non-empty mapping")
    by_path = {record["path"]: record for record in records}
    entries: list[dict[str, str]] = []
    labels: set[str] = set()
    for label, raw_path in sorted(declared_inputs.items()):
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("declared input labels must be unique non-blank strings")
        labels.add(label)
        path = _canonical_path(raw_path)
        record = by_path.get(path)
        if record is None:
            raise ValueError(f"recorded Git tree is missing declared input {label}")
        materialized = _materialize_tree_entries(
            repository, [record], selected_only=False
        )[0]
        entries.append({"label": label, **materialized})
    _validate_unique_paths(entries, label="declared input")
    return entries


def build_revision_binding(
    repository: Path | str,
    revision: str,
    declared_inputs: Mapping[str, str],
) -> dict[str, Any]:
    """Build the conservative binding from one committed Git tree."""

    root = Path(repository).resolve(strict=True)
    if not isinstance(revision, str) or not _HEX_OID.fullmatch(revision):
        raise ValueError("revision must be a full lowercase Git object identity")
    commit_type = _git(root, "cat-file", "-t", revision).decode("ascii").strip()
    if commit_type != "commit":
        raise ValueError("recorded revision must identify a commit")
    records, tracked_files = _tree_entries(root, revision)
    return {
        "format_version": _FORMAT_VERSION,
        "revision": revision,
        "tracked_files": tracked_files,
        "declared_inputs": _declared_entries(root, records, declared_inputs),
    }


def _normalized_recorded_entry(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "mode",
        "object_type",
        "blob_oid",
        "sha256",
    }:
        raise ValueError(f"{label} entry fields are invalid")
    path = _canonical_path(value["path"])
    mode = value["mode"]
    if mode not in _REGULAR_MODES:
        raise ValueError(f"bound path must have a regular file mode: {path}")
    if value["object_type"] != "blob":
        raise ValueError(f"bound path must have blob object type: {path}")
    oid = value["blob_oid"]
    sha256 = value["sha256"]
    if not isinstance(oid, str) or not _HEX_OID.fullmatch(oid):
        raise ValueError(f"{label} blob_oid is invalid")
    if not isinstance(sha256, str) or not _HEX_SHA256.fullmatch(sha256):
        raise ValueError(f"{label} sha256 is invalid")
    return {
        "path": path,
        "mode": mode,
        "object_type": "blob",
        "blob_oid": oid,
        "sha256": sha256,
    }


def _normalize_binding(
    value: object, expected_declared_inputs: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "revision",
        "tracked_files",
        "declared_inputs",
    }:
        raise ValueError("revision binding fields are invalid")
    if value["format_version"] != _FORMAT_VERSION:
        raise ValueError("revision binding format_version is invalid")
    revision = value["revision"]
    if not isinstance(revision, str) or not _HEX_OID.fullmatch(revision):
        raise ValueError("revision binding revision is invalid")
    raw_tracked = value["tracked_files"]
    if not isinstance(raw_tracked, list):
        raise ValueError("tracked-code binding must be a list")
    tracked = [
        _normalized_recorded_entry(entry, label="tracked-code binding")
        for entry in raw_tracked
    ]
    _validate_unique_paths(tracked, label="tracked-code binding")
    if tracked != sorted(tracked, key=lambda entry: entry["path"]):
        raise ValueError("tracked-code binding must use canonical path order")

    raw_declared = value["declared_inputs"]
    if not isinstance(raw_declared, list):
        raise ValueError("declared input binding must be a list")
    declared: list[dict[str, str]] = []
    for item in raw_declared:
        if not isinstance(item, dict) or set(item) != {
            "label",
            "path",
            "mode",
            "object_type",
            "blob_oid",
            "sha256",
        }:
            raise ValueError("declared input binding fields are invalid")
        label = item["label"]
        if not isinstance(label, str) or not label:
            raise ValueError("declared input label is invalid")
        declared.append(
            {
                "label": label,
                **_normalized_recorded_entry(
                    {key: item[key] for key in item if key != "label"},
                    label=f"declared input {label}",
                ),
            }
        )
    expected_mapping = {
        label: _canonical_path(path) for label, path in expected_declared_inputs.items()
    }
    if {entry["label"]: entry["path"] for entry in declared} != expected_mapping:
        raise ValueError("declared input binding does not match the approved inputs")
    if declared != sorted(declared, key=lambda entry: entry["label"]):
        raise ValueError("declared input binding must use canonical label order")
    _validate_unique_paths(declared, label="declared input")
    return {
        "format_version": _FORMAT_VERSION,
        "revision": revision,
        "tracked_files": tracked,
        "declared_inputs": declared,
    }


def _index_bound_entries(
    repository: Path, bound_paths: set[str]
) -> list[dict[str, str]]:
    records = _parse_index_records(_git(repository, "ls-files", "--stage", "-z"))
    selected = [record for record in records if record["path"] in bound_paths]
    _validate_unique_paths(selected, label="Git index binding set")
    entries: list[dict[str, str]] = []
    for record in selected:
        object_type = (
            _git(repository, "cat-file", "-t", record["blob_oid"]).decode().strip()
        )
        entries.append(
            {
                "path": record["path"],
                "mode": record["mode"],
                "object_type": object_type,
                "blob_oid": record["blob_oid"],
                "sha256": hashlib.sha256(
                    _blob_bytes(repository, record["blob_oid"])
                ).hexdigest(),
            }
        )
    return sorted(entries, key=lambda entry: entry["path"])


def _is_alias(path_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & 0x400
    )


def _verify_checkout(repository: Path, entries: list[dict[str, str]]) -> None:
    root = repository.resolve(strict=True)
    for entry in entries:
        path = root.joinpath(*PurePosixPath(entry["path"]).parts)
        try:
            path_stat = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"bound checkout path is missing or sparse: {entry['path']}"
            ) from exc
        if _is_alias(path_stat) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(
                f"bound checkout path is not a regular file: {entry['path']}"
            )
        try:
            path.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"bound checkout path escapes repository: {entry['path']}"
            ) from exc
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"bound checkout bytes differ: {entry['path']}")


def _scan_repository_importables(repository: Path, tracked_paths: set[str]) -> None:
    root = repository.resolve(strict=True)
    for root_name in _CODE_ROOTS:
        code_root = root / root_name
        if not code_root.is_dir():
            raise ValueError(f"verified repository code root is missing: {root_name}")
        pending = [code_root]
        while pending:
            directory = pending.pop()
            for child in os.scandir(directory):
                path = Path(child.path)
                relative = path.relative_to(root).as_posix()
                child_stat = child.stat(follow_symlinks=False)
                if _is_alias(child_stat):
                    raise ValueError(
                        f"repository code root contains an alias: {relative}"
                    )
                if child.is_dir(follow_symlinks=False):
                    if child.name.casefold() in _CACHE_DIRECTORY_NAMES:
                        raise ValueError(
                            f"repository code root contains a cache tree: {relative}"
                        )
                    pending.append(path)
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise ValueError(
                        f"repository code root contains a special file: {relative}"
                    )
                if (
                    path.suffix.casefold() in _IMPORTABLE_SUFFIXES
                    and relative not in tracked_paths
                ):
                    raise ValueError(
                        f"repository contains untracked or ignored importable artifact: {relative}"
                    )


def verify_revision_binding(
    repository: Path | str,
    binding: object,
    expected_declared_inputs: Mapping[str, str],
) -> dict[str, Any]:
    """Verify recorded tree, current HEAD/index, checkout, scan, and cleanliness."""

    root = Path(repository).resolve(strict=True)
    normalized = _normalize_binding(binding, expected_declared_inputs)
    rebuilt = build_revision_binding(
        root, normalized["revision"], expected_declared_inputs
    )
    if rebuilt["tracked_files"] != normalized["tracked_files"]:
        raise ValueError("tracked-code binding differs from the recorded Git tree")
    if rebuilt["declared_inputs"] != normalized["declared_inputs"]:
        raise ValueError("declared input binding differs from the recorded Git tree")

    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    head_binding = build_revision_binding(root, head, expected_declared_inputs)
    if head_binding["tracked_files"] != normalized["tracked_files"]:
        raise ValueError("current HEAD tracked-code set differs from recorded revision")
    if head_binding["declared_inputs"] != normalized["declared_inputs"]:
        raise ValueError("current HEAD declared input differs from recorded revision")

    expected_index_entries = sorted(
        [
            *normalized["tracked_files"],
            *(
                {key: value for key, value in entry.items() if key != "label"}
                for entry in normalized["declared_inputs"]
            ),
        ],
        key=lambda entry: entry["path"],
    )
    index_entries = _index_bound_entries(
        root, {entry["path"] for entry in expected_index_entries}
    )
    if index_entries != expected_index_entries:
        raise ValueError("current index binding set differs from recorded revision")

    all_bound = [
        *normalized["tracked_files"],
        *(
            {key: value for key, value in item.items() if key != "label"}
            for item in normalized["declared_inputs"]
        ),
    ]
    _verify_checkout(root, all_bound)
    tracked_paths = {
        _decode_git_path(raw)
        for raw in _git(root, "ls-files", "-z").split(b"\0")
        if raw
    }
    _scan_repository_importables(root, tracked_paths)
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        raise ValueError("authoritative execution requires a clean tree")
    return normalized


def _pep503_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("distribution inventory contains a blank name")
    normalized = _NORMALIZED_DISTRIBUTION.sub("-", value).lower()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", normalized):
        raise ValueError(f"distribution inventory name is invalid: {value!r}")
    return normalized


def _marker_environment() -> dict[str, str]:
    full_version = platform.python_version()
    return {
        "implementation_name": sys.implementation.name,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "python_full_version": full_version,
        "python_version": ".".join(full_version.split(".")[:2]),
        "sys_platform": sys.platform,
    }


def _version_key(value: str) -> tuple[tuple[int, object], ...]:
    pieces = re.findall(r"[0-9]+|[A-Za-z]+", value)
    return tuple(
        (0, int(piece)) if piece.isdigit() else (1, piece.casefold())
        for piece in pieces
    )


def _validate_interpreter_version_contract(
    project: Path, lock: Mapping[str, Any]
) -> None:
    try:
        requested_version = (
            (project / ".python-version").read_text(encoding="utf-8").strip()
        )
    except OSError as exc:
        raise ValueError(".python-version is missing or unreadable") from exc
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", requested_version):
        raise ValueError(".python-version must contain one exact Python version")
    actual_version = platform.python_version()
    requested_parts = requested_version.split(".")
    if actual_version.split(".")[: len(requested_parts)] != requested_parts:
        raise ValueError(".python-version does not match the authoritative interpreter")

    requirement = lock.get("requires-python")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("uv.lock requires-python is missing or invalid")
    for raw_clause in requirement.split(","):
        clause = raw_clause.strip()
        match = re.fullmatch(
            r"(===|==|!=|<=|>=|<|>)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
            clause,
        )
        if match is None:
            raise ValueError("uv.lock requires-python contains unsupported syntax")
        operator, version = match.groups()
        expression = f"python_full_version {operator} '{version}'"
        if not _evaluate_marker(expression, _marker_environment()):
            raise ValueError(
                "uv.lock requires-python excludes the authoritative interpreter"
            )


class _MarkerParser:
    def __init__(self, expression: str, environment: Mapping[str, str]) -> None:
        self.environment = environment
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(expression):
            match = _MARKER_TOKEN.match(expression, position)
            if match is None:
                raise ValueError(f"unsupported lock marker syntax: {expression!r}")
            keyword, identifier, operator, literal, parenthesis = match.groups()
            if keyword:
                self.tokens.append(("keyword", keyword))
            elif identifier:
                self.tokens.append(("identifier", identifier))
            elif operator:
                self.tokens.append(("operator", operator))
            elif literal:
                self.tokens.append(("literal", literal[1:-1]))
            elif parenthesis:
                self.tokens.append((parenthesis, parenthesis))
            position = match.end()
        self.index = 0

    def _accept(self, kind: str, value: str | None = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        token_kind, token_value = self.tokens[self.index]
        if token_kind != kind or (value is not None and token_value != value):
            return False
        self.index += 1
        return True

    def _operand(self) -> tuple[str | None, str]:
        if self.index >= len(self.tokens):
            raise ValueError("lock marker is missing an operand")
        kind, value = self.tokens[self.index]
        self.index += 1
        if kind == "literal":
            return None, value
        if kind != "identifier" or value not in self.environment:
            raise ValueError(f"unsupported lock marker variable: {value}")
        return value, self.environment[value]

    def _comparison(self) -> bool:
        if self._accept("("):
            result = self._or_expression()
            if not self._accept(")"):
                raise ValueError("unclosed lock marker parenthesis")
            return result
        left_name, left = self._operand()
        if self._accept("keyword", "not"):
            if not self._accept("keyword", "in"):
                raise ValueError("unsupported lock marker 'not' expression")
            operator = "not in"
        elif self._accept("keyword", "in"):
            operator = "in"
        elif self.index < len(self.tokens) and self.tokens[self.index][0] == "operator":
            operator = self.tokens[self.index][1]
            self.index += 1
        else:
            raise ValueError("lock marker is missing a comparison operator")
        right_name, right = self._operand()
        if operator == "in":
            return left in right
        if operator == "not in":
            return left not in right
        if operator in {"==", "==="}:
            return left == right
        if operator == "!=":
            return left != right
        version_comparison = bool(
            (left_name and left_name.startswith("python_"))
            or (right_name and right_name.startswith("python_"))
        )
        comparable_left: object = _version_key(left) if version_comparison else left
        comparable_right: object = _version_key(right) if version_comparison else right
        return {
            "<": comparable_left < comparable_right,
            "<=": comparable_left <= comparable_right,
            ">": comparable_left > comparable_right,
            ">=": comparable_left >= comparable_right,
        }[operator]

    def _and_expression(self) -> bool:
        result = self._comparison()
        while self._accept("keyword", "and"):
            other = self._comparison()
            result = result and other
        return result

    def _or_expression(self) -> bool:
        result = self._and_expression()
        while self._accept("keyword", "or"):
            other = self._and_expression()
            result = result or other
        return result

    def parse(self) -> bool:
        result = self._or_expression()
        if self.index != len(self.tokens):
            raise ValueError("lock marker contains trailing unsupported syntax")
        return result


def _evaluate_marker(expression: object, environment: Mapping[str, str]) -> bool:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("lock marker must be a non-blank string")
    return _MarkerParser(expression, environment).parse()


def _package_selected(package: Mapping[str, Any], markers: Mapping[str, str]) -> bool:
    resolution_markers = package.get("resolution-markers", [])
    if not isinstance(resolution_markers, list) or not all(
        isinstance(marker, str) for marker in resolution_markers
    ):
        raise ValueError("lock package resolution-markers are invalid")
    return not resolution_markers or any(
        _evaluate_marker(marker, markers) for marker in resolution_markers
    )


def _select_locked_inventory(
    lock: Mapping[str, Any], markers: Mapping[str, str]
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock package records are invalid")
    roots = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("source") == {"virtual": "."}
    ]
    if len(roots) != 1:
        raise ValueError("uv.lock must contain exactly one virtual project root")
    root = roots[0]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock package record must be an object")
        name = _pep503_name(package.get("name"))
        version = package.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"uv.lock package version is invalid for {name}")
        by_name.setdefault(name, []).append(package)

    selected: dict[str, str] = {}
    traversed_dependencies: set[tuple[str, str]] = set()
    traversed_extras: dict[tuple[str, str], set[str]] = {}
    pending: list[dict[str, Any]] = list(root.get("dependencies", []))
    while pending:
        dependency = pending.pop(0)
        if not isinstance(dependency, dict):
            raise ValueError("uv.lock dependency record must be an object")
        marker = dependency.get("marker")
        if marker is not None and not _evaluate_marker(marker, markers):
            continue
        name = _pep503_name(dependency.get("name"))
        candidates = by_name.get(name, [])
        if "version" in dependency:
            candidates = [
                package
                for package in candidates
                if package.get("version") == dependency["version"]
            ]
        if "source" in dependency:
            candidates = [
                package
                for package in candidates
                if package.get("source") == dependency["source"]
            ]
        candidates = [
            package for package in candidates if _package_selected(package, markers)
        ]
        if len(candidates) != 1:
            raise ValueError(f"uv.lock does not select exactly one package for {name}")
        package = candidates[0]
        version = package["version"]
        prior = selected.get(name)
        if prior is not None and prior != version:
            raise ValueError(f"uv.lock selects duplicate normalized package {name}")
        selected[name] = version
        extras = dependency.get("extra", [])
        if not isinstance(extras, list) or not all(
            isinstance(extra, str) for extra in extras
        ):
            raise ValueError(f"uv.lock extras are invalid for {name}")
        traversal_key = (name, version)
        already = traversed_extras.setdefault(traversal_key, set())
        new_extras = set(extras) - already
        if traversal_key not in traversed_dependencies:
            child_dependencies = package.get("dependencies", [])
            if not isinstance(child_dependencies, list):
                raise ValueError(f"uv.lock dependencies are invalid for {name}")
            pending.extend(child_dependencies)
            traversed_dependencies.add(traversal_key)
        optional = package.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise ValueError(f"uv.lock optional dependencies are invalid for {name}")
        for extra in sorted(new_extras):
            if extra not in optional or not isinstance(optional[extra], list):
                raise ValueError(
                    f"uv.lock does not contain selected extra {name}[{extra}]"
                )
            pending.extend(optional[extra])
        already.update(new_extras)

    dev_dependencies = root.get("dev-dependencies", {})
    if not isinstance(dev_dependencies, dict):
        raise ValueError("uv.lock dev dependency groups are invalid")
    resolution_markers = lock.get("resolution-markers", [])
    if not isinstance(resolution_markers, list) or not all(
        isinstance(marker, str) for marker in resolution_markers
    ):
        raise ValueError("uv.lock resolution-markers are invalid")
    active_resolution = sorted(
        marker for marker in resolution_markers if _evaluate_marker(marker, markers)
    )
    return (
        [
            {"name": name, "version": version}
            for name, version in sorted(selected.items())
        ],
        sorted(dev_dependencies),
        active_resolution,
    )


def _installed_inventory(approved_sites: list[Path]) -> list[dict[str, str]]:
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[os.fspath(path) for path in approved_sites]
    ):
        name = _pep503_name(distribution.metadata.get("Name"))
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise ValueError(f"distribution inventory version is invalid for {name}")
        if name in inventory:
            raise ValueError(
                f"duplicate normalized distribution inventory entry: {name}"
            )
        inventory[name] = version
    return [
        {"name": name, "version": version}
        for name, version in sorted(inventory.items())
    ]


def _expected_runtime_import_layout(
    selected: Sequence[Mapping[str, object]], markers: Mapping[str, str]
) -> tuple[PurePosixPath, ...]:
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise ValueError("selected package inventory must be a sequence")
    if not isinstance(markers, Mapping) or set(markers) != set(_MARKER_KEYS):
        raise ValueError("runtime marker environment fields are invalid")
    if not all(isinstance(markers[key], str) for key in _MARKER_KEYS):
        raise ValueError("runtime marker environment values are invalid")

    selected_names: list[str] = []
    for entry in selected:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "version"}:
            raise ValueError("selected package inventory entry fields are invalid")
        name = entry["name"]
        version = entry["version"]
        if (
            not isinstance(name, str)
            or _pep503_name(name) != name
            or not isinstance(version, str)
            or not version
        ):
            raise ValueError("selected package inventory entry is invalid")
        selected_names.append(name)

    is_windows = (
        markers["os_name"] == "nt"
        and markers["sys_platform"] == "win32"
        and markers["platform_system"] == "Windows"
    )
    if is_windows and "pywin32" in selected_names:
        return _WINDOWS_PYWIN32_RUNTIME_LAYOUT
    return ()


def _parse_pyvenv(environment_root: Path) -> dict[str, str]:
    config_path = environment_root / "pyvenv.cfg"
    if not config_path.is_file():
        raise ValueError("environment root is missing pyvenv.cfg")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip() or key.strip().casefold() in values:
            raise ValueError("pyvenv.cfg is malformed or contains duplicate keys")
        values[key.strip().casefold()] = value.strip()
    return values


def _same_path(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.abspath(left))
    right_text = os.path.normcase(os.path.abspath(right))
    return left_text == right_text


def _approved_sites(environment_root: Path) -> list[Path]:
    if os.name == "nt":
        return [environment_root / "Lib/site-packages"]
    return [
        environment_root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    ]


def _validated_approved_sites(environment_root: Path) -> list[Path]:
    lexical_environment = Path(os.path.abspath(environment_root))
    try:
        resolved_environment = lexical_environment.resolve(strict=True)
    except OSError as exc:
        raise ValueError("approved environment root is missing") from exc
    approved_sites = _approved_sites(lexical_environment)
    for site_path in approved_sites:
        current = lexical_environment
        for part in site_path.relative_to(lexical_environment).parts:
            current /= part
            try:
                path_stat = os.lstat(current)
            except OSError as exc:
                raise ValueError(
                    "approved site-packages path or parent is missing"
                ) from exc
            if _is_alias(path_stat):
                raise ValueError(
                    "approved site-packages path or parent contains an alias"
                )
            if not stat.S_ISDIR(path_stat.st_mode):
                raise ValueError(
                    "approved site-packages path or parent is not a directory"
                )
        try:
            resolved_site = site_path.resolve(strict=True)
            resolved_site.relative_to(resolved_environment)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "approved site-packages path resolves outside environment"
            ) from exc
    return approved_sites


def _validated_runtime_import_roots(
    environment_root: Path,
    layout: Sequence[str],
    selected: Sequence[Mapping[str, object]],
    markers: Mapping[str, str],
) -> list[Path]:
    expected = _expected_runtime_import_layout(selected, markers)
    if type(layout) is not list or any(type(item) is not str for item in layout):
        raise ValueError("runtime import layout must be a list of strings")
    try:
        canonical_layout = tuple(
            PurePosixPath(_canonical_path(item)) for item in layout
        )
    except ValueError as exc:
        raise ValueError("runtime import layout contains a non-canonical path") from exc
    if canonical_layout != expected:
        raise ValueError(
            "runtime import layout differs from the package-conditioned layout"
        )

    lexical_environment = Path(os.path.abspath(environment_root))
    try:
        resolved_environment = lexical_environment.resolve(strict=True)
    except OSError as exc:
        raise ValueError("runtime import root environment is missing") from exc

    roots: list[Path] = []
    for relative in canonical_layout:
        root = lexical_environment.joinpath(*relative.parts)
        current = lexical_environment
        for part in relative.parts:
            current /= part
            try:
                path_stat = os.lstat(current)
            except OSError as exc:
                raise ValueError(
                    "runtime import root path or parent is missing"
                ) from exc
            if _is_alias(path_stat):
                raise ValueError(
                    "runtime import root path or parent contains an alias"
                )
            if not stat.S_ISDIR(path_stat.st_mode):
                raise ValueError(
                    "runtime import root path or parent is not a directory"
                )
        try:
            resolved_root = root.resolve(strict=True)
            resolved_root.relative_to(resolved_environment)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "runtime import root resolves outside environment"
            ) from exc
        roots.append(root)
    return roots


def _validate_preimport_runtime(
    project_root: Path, environment_root: Path
) -> tuple[list[Path], dict[str, str]]:
    if not (
        sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and bool(getattr(sys.flags, "safe_path", False))
    ):
        raise ValueError("authoritative bootstrap requires Python -I")
    if sys.flags.no_site != 1 or any(
        name in sys.modules for name in ("site", "sitecustomize", "usercustomize")
    ):
        raise ValueError(
            "authoritative bootstrap requires Python -S with no site customization"
        )
    if os.environ.get("PYTHONPATH"):
        raise ValueError("PYTHONPATH must be absent for authoritative bootstrap")
    if os.environ.get("PYTHONHOME") or os.environ.get("PYTHONUSERBASE"):
        raise ValueError(
            "Python home and user-site environment overrides must be absent"
        )

    project = project_root.resolve(strict=True)
    environment = environment_root.resolve(strict=True)
    try:
        environment.relative_to(project)
    except ValueError:
        pass
    else:
        raise ValueError("environment root must be outside the repository")
    try:
        project.relative_to(environment)
    except ValueError:
        pass
    else:
        raise ValueError("environment root must not contain the repository")

    executable = Path(os.path.abspath(sys.executable))
    try:
        executable_relative = executable.relative_to(environment)
    except ValueError as exc:
        raise ValueError("sys.executable must be inside the environment root") from exc
    approved_executables = {
        PurePosixPath("Scripts/python.exe"),
        PurePosixPath("bin/python"),
        PurePosixPath(f"bin/python{sys.version_info.major}"),
        PurePosixPath(f"bin/python{sys.version_info.major}.{sys.version_info.minor}"),
    }
    if PurePosixPath(executable_relative.as_posix()) not in approved_executables:
        raise ValueError(
            "sys.executable has an unapproved environment-relative location"
        )

    config = _parse_pyvenv(environment)
    if config.get("include-system-site-packages", "").casefold() != "false":
        raise ValueError("pyvenv.cfg must disable system site packages")
    home = config.get("home")
    if not home or not Path(home).is_absolute():
        raise ValueError("pyvenv.cfg home must identify the base interpreter directory")
    base_executable = Path(getattr(sys, "_base_executable", ""))
    if not base_executable.is_absolute() or not _same_path(
        base_executable.parent, Path(home)
    ):
        raise ValueError(
            "sys.executable and pyvenv.cfg base interpreter relationship is invalid"
        )
    recorded_version = config.get("version_info", config.get("version"))
    if recorded_version is not None and recorded_version != platform.python_version():
        raise ValueError("pyvenv.cfg interpreter version does not match sys.version")

    approved_sites = _validated_approved_sites(environment)

    home_path = Path(home)
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if os.name == "nt":
        allowed_paths = {
            home_path,
            home_path / "DLLs",
            home_path / "Lib",
            home_path / f"python{sys.version_info.major}{sys.version_info.minor}.zip",
        }
    else:
        installation_root = home_path.parent if home_path.name == "bin" else home_path
        stdlib = installation_root / "lib" / major_minor
        allowed_paths = {
            stdlib,
            stdlib / "lib-dynload",
            installation_root
            / "lib"
            / f"python{sys.version_info.major}{sys.version_info.minor}.zip",
        }
    for raw_path in sys.path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("pre-bootstrap sys.path contains an empty or invalid root")
        path = Path(os.path.abspath(raw_path))
        if not any(_same_path(path, allowed) for allowed in allowed_paths):
            raise ValueError(
                f"pre-bootstrap sys.path contains an unapproved root: {path.name}"
            )
        if path.suffix.casefold() == ".zip" and not any(
            _same_path(path, allowed) and allowed.suffix.casefold() == ".zip"
            for allowed in allowed_paths
        ):
            raise ValueError("pre-bootstrap sys.path contains an unapproved zip root")
    if any(
        any(_same_path(Path(raw_path), site_path) for raw_path in sys.path)
        for site_path in approved_sites
    ):
        raise ValueError(
            "approved sites must not be processed before bootstrap validation"
        )
    return approved_sites, config


def build_environment_binding(
    project_root: Path | str, environment_root: Path | str
) -> dict[str, Any]:
    """Validate and bind an isolated, frozen, no-development environment."""

    project = Path(project_root).resolve(strict=True)
    environment = Path(environment_root).resolve(strict=True)
    approved_sites, _config = _validate_preimport_runtime(project, environment)
    lock_path = project / "uv.lock"
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("uv.lock is missing or malformed") from exc
    if lock.get("version") != 1 or lock.get("revision") != 3:
        raise ValueError("uv.lock format version/revision is not approved")
    _validate_interpreter_version_contract(project, lock)
    markers = _marker_environment()
    selected, excluded_groups, active_resolution = _select_locked_inventory(
        lock, markers
    )
    installed = _installed_inventory(approved_sites)
    if installed != selected:
        raise ValueError(
            "installed distribution inventory does not exactly match the selected no-dev lock"
        )
    runtime_import_layout = [
        path.as_posix()
        for path in _expected_runtime_import_layout(selected, markers)
    ]
    _validated_runtime_import_roots(
        environment, runtime_import_layout, selected, markers
    )
    interpreter_relative = Path(os.path.abspath(sys.executable)).relative_to(
        environment
    )
    site_layout = [
        site_path.relative_to(environment).as_posix() for site_path in approved_sites
    ]
    return {
        "format_version": _FORMAT_VERSION,
        "interpreter": {
            "implementation": sys.implementation.name,
            "full_version": platform.python_version(),
            "abi": f"{sys.implementation.cache_tag}:{getattr(sys, 'abiflags', '')}",
            "os_name": os.name,
            "sys_platform": sys.platform,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "executable_layout": interpreter_relative.as_posix(),
        },
        "pyvenv": {"include_system_site_packages": False},
        "site_layout": site_layout,
        "runtime_import_layout": runtime_import_layout,
        "lock_selection": {
            "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "offline": True,
            "frozen": True,
            "no_dev": True,
            "selected_dependency_groups": [],
            "excluded_dependency_groups": excluded_groups,
            "markers": {key: markers[key] for key in _MARKER_KEYS},
            "active_resolution_markers": active_resolution,
            "selected_packages": selected,
        },
        "installed_distributions": installed,
    }


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    return False


def verify_environment_binding(actual: object, expected: object) -> dict[str, Any]:
    """Require an exact privacy-safe environment binding match."""

    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("environment binding must be an object")
    if _contains_absolute_path(actual) or _contains_absolute_path(expected):
        raise ValueError("environment binding must not contain absolute paths")
    if actual != expected:
        raise ValueError("environment binding differs from the recorded environment")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("record", "calibrate", "verify-artifact", "verify-release"),
        required=True,
    )
    parser.add_argument("--binding", type=Path)
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    return parser


def _declared_inputs() -> dict[str, str]:
    return {
        "corpus_snapshot": "release/corpus_snapshot.json",
        "formal_dataset": "eval/dataset/eval_set.jsonl",
        "stress_dataset": "eval/dataset/reliability_stress_v0.3.1.jsonl",
        "target_dataset": "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
    }


def _record_runtime(
    project_root: Path, environment_binding: dict[str, Any]
) -> dict[str, Any]:
    revision = _git(project_root, "rev-parse", "HEAD").decode("ascii").strip()
    revision_binding = build_revision_binding(
        project_root, revision, _declared_inputs()
    )
    verify_revision_binding(project_root, revision_binding, _declared_inputs())
    return {
        "revision_binding": revision_binding,
        "environment_binding": environment_binding,
    }


def _artifact_runtime(artifact_path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        provenance = artifact["provenance"]
        runtime = {
            "revision_binding": provenance["revision_binding"],
            "environment_binding": provenance["environment_binding"],
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "binding artifact does not contain authoritative provenance"
        ) from exc
    if not isinstance(runtime["revision_binding"], dict) or not isinstance(
        runtime["environment_binding"], dict
    ):
        raise ValueError("binding artifact runtime provenance is malformed")
    return runtime


def _activate_import_paths(
    project_root: Path,
    environment_root: Path,
    environment_binding: dict[str, Any],
) -> None:
    project_roots = [
        project_root / "src",
        project_root / "eval",
        project_root / "scripts",
    ]
    approved_sites = _validated_approved_sites(environment_root)
    if not all(path.is_dir() for path in project_roots):
        raise ValueError("verified project or environment import root is missing")

    if type(environment_binding) is not dict:
        raise ValueError("environment binding is malformed during import activation")
    lock_selection = environment_binding.get("lock_selection")
    if type(lock_selection) is not dict:
        raise ValueError("environment lock selection is malformed during import activation")
    try:
        layout = environment_binding["runtime_import_layout"]
        markers = lock_selection["markers"]
        selected = lock_selection["selected_packages"]
    except KeyError as exc:
        raise ValueError(
            "environment runtime binding is incomplete during import activation"
        ) from exc
    runtime_roots = _validated_runtime_import_roots(
        environment_root, layout, selected, markers
    )
    additions = [*project_roots, *approved_sites, *runtime_roots]
    sys.path.extend(os.fspath(path) for path in additions)


def _entry_arguments(arguments: list[str]) -> list[str]:
    return arguments[1:] if arguments[:1] == ["--"] else arguments


def _authoritative_calibration_arguments(arguments: list[str]) -> list[str]:
    entry_args = _entry_arguments(arguments)
    protected_options = (
        "--dataset",
        "--stress-dataset",
        "--formal-dataset",
        "--snapshot",
    )
    for argument in entry_args:
        option_name = argument.partition("=")[0]
        if option_name.startswith("--") and any(
            option.startswith(option_name) for option in protected_options
        ):
            raise ValueError(
                "authoritative input paths may not be overridden during calibration"
            )
    return entry_args


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        entry_args = (
            _authoritative_calibration_arguments(args.entry_args)
            if args.mode == "calibrate"
            else []
        )
        project_root = args.project_root.resolve(strict=True)
        environment_root = args.environment_root.resolve(strict=True)
        environment_binding = build_environment_binding(project_root, environment_root)
        if args.mode in {"record", "calibrate"}:
            runtime = _record_runtime(project_root, environment_binding)
        else:
            artifact_path = args.binding
            if args.mode == "verify-release":
                artifact_path = (
                    project_root / "eval/official/severance_refusal_policy_v0.3.6.json"
                )
            if artifact_path is None:
                raise ValueError("verification mode requires --binding")
            runtime = _artifact_runtime(artifact_path.resolve(strict=True))
            verify_environment_binding(
                environment_binding, runtime["environment_binding"]
            )
            verify_revision_binding(
                project_root,
                runtime["revision_binding"],
                _declared_inputs(),
            )

        if args.mode == "record":
            print(
                json.dumps(
                    runtime,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        _activate_import_paths(project_root, environment_root, environment_binding)
        if args.mode == "calibrate":
            if "--offline" not in entry_args:
                raise ValueError("calibration bootstrap requires --offline")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            runner = importlib.import_module("run_severance_refusal_policy")
            return int(runner.main(entry_args, trusted_runtime=runtime))
        release_verification = importlib.import_module("rag.release_verification")
        if args.mode == "verify-artifact":
            report = release_verification._verify_severance_refusal_policy_artifact(
                project_root,
                args.binding.resolve(strict=True),
                trusted_runtime=runtime,
            )
        else:
            report = release_verification.verify_release(
                project_root, trusted_runtime=runtime
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except ValueError as exc:
        print(f"authoritative bootstrap rejected execution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
