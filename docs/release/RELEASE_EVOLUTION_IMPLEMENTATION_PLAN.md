# Release Evolution Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline release verifier distinguish publishable Git history from local recovery refs, support normal multi-commit evolution, and scan every publishable historical tree for forbidden paths, private data, secrets, and unreviewed binaries.

**Architecture:** Keep the exact current-tree allowlist unchanged as the primary release boundary. Add a publishable-ref adapter (`heads`, `tags`, `remotes`), an in-memory `git archive` reader, and a shared path/bytes scanner so current and historical artifacts use the same redacted privacy rules. Local refs such as `refs/archive/*` remain intact and outside the publication graph.

**Tech Stack:** Python 3.11, stdlib `dataclasses`/`io`/`tarfile`/`subprocess`, pytest 8, Ruff, Git plumbing, existing `release/manifest.json` and release verifier.

## Global Constraints

- Work only on `feat/v0.2-release-evolution-foundation` in the isolated worktree.
- Keep `main`, `origin/main`, tag `v0.1.0`, and `refs/archive/*` unchanged.
- Do not fetch, push, merge, rebase, tag, or call any remote service.
- The clean reviewer path must not load models, call providers, start Qdrant/Docker, require a GPU, or require network access after dependencies are installed.
- Preserve the exact current-tree contract: `git ls-files` must equal `release/public-files.txt`.
- Publishable history means commits reachable from `refs/heads/*`, `refs/tags/*`, and `refs/remotes/*` only.
- A short branch name such as `archive/foo` remains publishable because its full name is `refs/heads/archive/foo`.
- Historical privacy errors may report only path, category, location, and commit ID; never include matched content.
- Current binaries remain governed by `publication.reviewed_binaries`; historical binaries must match the append-only `publication.history.reviewed_binary_sha256` set.
- `publication.history.legacy_public_paths` starts empty and cannot override forbidden-path or content scanning.
- Do not modify retrieval, reranking, generation, refusal thresholds, UI, evaluation datasets, or reported model-quality metrics.
- Every behavior change follows red-green-refactor TDD and is committed separately.

## File Map

- `src/rag/release_verification.py`: publishable ref enumeration, byte-oriented scanning, safe archive parsing, and historical verification orchestration.
- `tests/test_release_verification.py`: Git fixture helpers and all ref/history/privacy regression coverage.
- `release/manifest.json`: replace `max_commits` with explicit ref namespaces, legacy paths, and historical binary hashes.
- `release/public-files.txt`: add the approved design and implementation plan to the exact public inventory.
- `README.md` / `README.en.md`: describe publishable history rather than a one-commit local checkout assumption.
- `docs/release/PUBLICATION_BOUNDARY.md`: authoritative ref, historical scanning, and recovery-ref boundary.
- `docs/release/CLAIM_MATRIX.md`: map the new history claim to source, tests, and verifier output.
- `docs/release/REVIEWER_GUIDE.md`: update expected reviewer evidence and Git audit interpretation.

---

### Task 1: Select Publishable Commits Without Following Recovery Refs

**Files:**
- Modify: `tests/test_release_verification.py:1-205`
- Modify: `src/rag/release_verification.py:345-430`

**Interfaces:**
- Consumes: a local Git repository at `project_root: Path`.
- Produces: `_publishable_commit_ids(project_root: Path) -> list[str]`, returning sorted unique commits reachable from branches, tags, and remote-tracking refs.

- [ ] **Step 1: Add deterministic Git fixture helpers**

Add these helpers near the top of `tests/test_release_verification.py`:

```python
PUBLIC_NAME = "kuotunyu"
PUBLIC_EMAIL = "61350295+kuotunyu@users.noreply.github.com"


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def init_public_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    run_git(repo, "config", "user.name", PUBLIC_NAME)
    run_git(repo, "config", "user.email", PUBLIC_EMAIL)


def commit_file(repo: Path, relative_path: str, content: bytes, message: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    run_git(repo, "add", "--", relative_path)
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()
```

- [ ] **Step 2: Write the failing recovery-ref selection test**

Add:

```python
def test_publishable_commit_ids_ignore_nonstandard_recovery_refs(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    recovered_commit = commit_file(
        tmp_path,
        ".claude/launch.json",
        b"{}",
        "local recovery only",
    )
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/archive/recovery", recovered_commit)

    assert release_module()._publishable_commit_ids(tmp_path) == [public_commit]
```

- [ ] **Step 3: Run the test and verify RED**

Run:

```powershell
uv run --locked pytest tests/test_release_verification.py::test_publishable_commit_ids_ignore_nonstandard_recovery_refs -q -p no:cacheprovider
```

Expected: FAIL with `AttributeError` because `_publishable_commit_ids` does not exist.

- [ ] **Step 4: Implement publishable ref enumeration**

Add to `src/rag/release_verification.py` immediately after `_tracked_files`:

```python
def _publishable_commit_ids(project_root: Path) -> list[str]:
    try:
        process = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "rev-list",
                "--branches",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ReleaseVerificationError("failed to enumerate publishable Git refs") from exc
    commits = sorted({line for line in process.stdout.splitlines() if line})
    if not commits:
        raise ReleaseVerificationError("Git checkout has no publishable commits")
    return commits
```

- [ ] **Step 5: Run the recovery-ref test and verify GREEN**

Run the command from Step 3.

Expected: `1 passed`.

- [ ] **Step 6: Add failing coverage for public archive branches and ref deduplication**

Add:

```python
def test_publishable_commit_ids_include_archive_named_branch(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    second_commit = commit_file(tmp_path, "CHANGELOG.md", b"change", "change")
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/heads/archive/review", second_commit)

    assert set(release_module()._publishable_commit_ids(tmp_path)) == {
        public_commit,
        second_commit,
    }


def test_publishable_commit_ids_include_and_deduplicate_tags_and_remotes(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    release_commit = commit_file(tmp_path, "CHANGELOG.md", b"release", "release")
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "tag", "-a", "v0.2.0", "-m", "release", release_commit)
    run_git(tmp_path, "update-ref", "refs/remotes/origin/release", release_commit)

    assert set(release_module()._publishable_commit_ids(tmp_path)) == {
        public_commit,
        release_commit,
    }


def test_publishable_commit_ids_reject_empty_ref_set(tmp_path):
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="no publishable commits",
    ):
        release_module()._publishable_commit_ids(tmp_path)
```

Run all three tests. The archive-branch and empty-set tests pass, while the
tag/remote test fails because the initial implementation enumerates branches
only. Add `"--tags"` and `"--remotes"` after `"--branches"`, then rerun all
three tests.

Expected after expansion: `3 passed`.

- [ ] **Step 7: Run the focused release-verification tests**

```powershell
uv run --locked pytest tests/test_release_verification.py -q -p no:cacheprovider
```

Expected: only the pre-existing real-repository verifier test may still fail; the three new selection tests pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add tests/test_release_verification.py src/rag/release_verification.py
git commit -m "fix: scope release history to publishable refs"
```

---

### Task 2: Share Redacted Privacy Scanning Between Files and Bytes

**Files:**
- Modify: `tests/test_release_verification.py:261-347`
- Modify: `src/rag/release_verification.py:7-249`

**Interfaces:**
- Consumes: `PublicEntry(path: str, data: bytes)` values from either the filesystem adapter or Git archive adapter.
- Produces: `_scan_public_entries(entries: Sequence[PublicEntry]) -> list[dict[str, str]]`; preserves `scan_public_files(project_root, relative_paths)` as the public filesystem-facing adapter.

- [ ] **Step 1: Write a failing byte-scanner privacy test**

Add:

```python
def test_public_entry_scan_reports_secret_without_leaking_value():
    module = release_module()
    secret = "sk-" + "A" * 32
    entries = [module.PublicEntry(path="README.md", data=secret.encode("utf-8"))]

    issues = module._scan_public_entries(entries)
    serialized = json.dumps(issues)

    assert [issue["category"] for issue in issues] == ["provider_token"]
    assert secret not in serialized


def test_public_file_scan_preserves_missing_sensitive_path_findings(tmp_path):
    issues = release_module().scan_public_files(tmp_path, [".env"])

    assert {issue["category"] for issue in issues} == {
        "missing_public_file",
        "sensitive_public_path",
    }
```

- [ ] **Step 2: Run the byte-scanner test and verify RED**

```powershell
uv run --locked pytest tests/test_release_verification.py::test_public_entry_scan_reports_secret_without_leaking_value -q -p no:cacheprovider
```

Expected: FAIL because `PublicEntry` is undefined.

- [ ] **Step 3: Add `PublicEntry` and the pure scanner**

Add `from dataclasses import dataclass`, then add:

```python
@dataclass(frozen=True)
class PublicEntry:
    path: str
    data: bytes


def _scan_public_entries(
    entries: Sequence[PublicEntry],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for entry in entries:
        normalized = entry.path.replace("\\", "/")
        suffix = Path(normalized).suffix.lower()
        if _is_sensitive_public_path(normalized):
            issues.append(_issue(normalized, "sensitive_public_path", "path"))
        if suffix in BINARY_PUBLIC_SUFFIXES:
            continue
        try:
            text = entry.data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(_issue(normalized, "non_utf8_public_text", "file"))
            continue

        categories: set[tuple[str, str]] = set()
        if _LOCAL_PATH.search(text):
            categories.add(("local_path", "text"))
        if _PRIVATE_KEY.search(text):
            categories.add(("private_key", "text"))
        if _STANDALONE_TOKEN.search(text):
            categories.add(("provider_token", "text"))
        private_email = any(
            not match.group(0).lower().endswith("@users.noreply.github.com")
            for match in _EMAIL_ADDRESS.finditer(text)
        )
        if private_email or (
            suffix in {".json", ".jsonl"} and _IP_ADDRESS.search(text)
        ):
            categories.add(("personal_identifier", "text"))
        if suffix in {".json", ".jsonl"} and _PROVIDER_PAYLOAD_KEY.search(text):
            categories.add(("provider_payload", "JSON field"))
        for match in _SECRET_ASSIGNMENT.finditer(text):
            if not _is_placeholder_secret(match.group(1)):
                line_number = text.count("\n", 0, match.start()) + 1
                categories.add(("secret_assignment", f"line {line_number}"))
        for category, location in sorted(categories):
            issues.append(_issue(normalized, category, location))
    return sorted(
        issues,
        key=lambda item: (item["path"], item["category"], item["location"]),
    )
```

- [ ] **Step 4: Refactor `scan_public_files` into an adapter**

Replace its content loop with:

```python
def scan_public_files(
    project_root: Path, relative_paths: Sequence[str]
) -> list[dict[str, str]]:
    """Return sanitized privacy/secret issues without returning matched values."""

    issues: list[dict[str, str]] = []
    entries: list[PublicEntry] = []
    for relative_path in relative_paths:
        normalized = relative_path.replace("\\", "/")
        path = project_root / Path(normalized)
        if not path.is_file():
            if _is_sensitive_public_path(normalized):
                issues.append(_issue(normalized, "sensitive_public_path", "path"))
            issues.append(_issue(normalized, "missing_public_file", "path"))
            continue
        entries.append(PublicEntry(path=normalized, data=path.read_bytes()))
    return sorted(
        [*issues, *_scan_public_entries(entries)],
        key=lambda item: (item["path"], item["category"], item["location"]),
    )
```

- [ ] **Step 5: Run new and existing scanner tests**

```powershell
uv run --locked pytest tests/test_release_verification.py -k "public_scan or public_entry_scan or reviewed_binary" -q -p no:cacheprovider
```

Expected: all selected tests pass and secret values remain absent from serialized findings.

- [ ] **Step 6: Commit Task 2**

```powershell
git add tests/test_release_verification.py src/rag/release_verification.py
git commit -m "refactor: share release privacy scanner"
```

---

### Task 3: Parse Git Commit Archives Safely in Memory

**Files:**
- Modify: `tests/test_release_verification.py`
- Modify: `src/rag/release_verification.py:7-20,345-430`

**Interfaces:**
- Consumes: tar bytes returned by `git archive --format=tar <commit>`.
- Produces: `_parse_git_archive(data: bytes, *, commit: str) -> list[PublicEntry]` and `_git_archive_entries(project_root: Path, commit: str) -> list[PublicEntry]`.

- [ ] **Step 1: Add an in-memory tar fixture helper**

Add `import io` and `import tarfile` to the test file, then add:

```python
def tar_bytes(*members: tuple[str, bytes, bytes | None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data, link_target in members:
            info = tarfile.TarInfo(name=name)
            if link_target is None:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = link_target.decode("utf-8")
                archive.addfile(info)
    return buffer.getvalue()
```

- [ ] **Step 2: Write failing parser tests**

Add:

```python
def test_parse_git_archive_returns_regular_entries():
    module = release_module()
    payload = tar_bytes(("README.md", b"public", None))

    assert module._parse_git_archive(payload, commit="a" * 40) == [
        module.PublicEntry(path="README.md", data=b"public")
    ]


@pytest.mark.parametrize(
    "unsafe_path",
    ["../secret.txt", "/absolute.txt", "C:" + "/private.txt"],
)
def test_parse_git_archive_rejects_unsafe_paths(unsafe_path):
    payload = tar_bytes((unsafe_path, b"private", None))

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unsafe Git archive path",
    ):
        release_module()._parse_git_archive(payload, commit="b" * 40)


def test_parse_git_archive_rejects_links():
    payload = tar_bytes(("link.txt", b"", b"README.md"))

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unsupported Git archive entry",
    ):
        release_module()._parse_git_archive(payload, commit="c" * 40)


def test_parse_git_archive_rejects_duplicate_paths():
    payload = tar_bytes(
        ("README.md", b"first", None),
        ("README.md", b"second", None),
    )

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="duplicate Git archive path",
    ):
        release_module()._parse_git_archive(payload, commit="d" * 40)
```

- [ ] **Step 3: Run the parser tests and verify RED**

```powershell
uv run --locked pytest tests/test_release_verification.py -k "parse_git_archive" -q -p no:cacheprovider
```

Expected: FAIL because `_parse_git_archive` does not exist.

- [ ] **Step 4: Implement safe tar parsing**

Add imports `io`, `tarfile`, and `PurePosixPath`, then add:

```python
def _parse_git_archive(data: bytes, *, commit: str) -> list[PublicEntry]:
    entries: list[PublicEntry] = []
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive.getmembers():
                raw = member.name
                candidate = raw[:-1] if member.isdir() and raw.endswith("/") else raw
                posix = PurePosixPath(candidate)
                if (
                    not candidate
                    or candidate.startswith("/")
                    or "\\" in candidate
                    or not posix.parts
                    or ":" in posix.parts[0]
                    or ".." in posix.parts
                    or posix.as_posix() != candidate
                ):
                    raise ReleaseVerificationError(
                        f"unsafe Git archive path in commit {commit}"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseVerificationError(
                        f"unsupported Git archive entry in commit {commit}: {raw}"
                    )
                if raw in seen:
                    raise ReleaseVerificationError(
                        f"duplicate Git archive path in commit {commit}: {raw}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReleaseVerificationError(
                        f"unreadable Git archive entry in commit {commit}: {raw}"
                    )
                seen.add(raw)
                entries.append(PublicEntry(path=raw, data=handle.read()))
    except (tarfile.TarError, OSError) as exc:
        raise ReleaseVerificationError(
            f"failed to parse Git archive for commit {commit}"
        ) from exc
    return sorted(entries, key=lambda entry: entry.path)
```

- [ ] **Step 5: Run parser tests and verify GREEN**

Run the command from Step 3.

Expected: all parser tests pass.

- [ ] **Step 6: Write a failing real-Git archive adapter test**

```python
def test_git_archive_entries_read_committed_tree(tmp_path):
    init_public_repo(tmp_path)
    commit = commit_file(tmp_path, "docs/README.md", b"public", "public")

    assert release_module()._git_archive_entries(tmp_path, commit) == [
        release_module().PublicEntry(path="docs/README.md", data=b"public")
    ]


def test_git_archive_entries_wrap_git_failure(tmp_path):
    init_public_repo(tmp_path)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="failed to read Git archive for commit",
    ):
        release_module()._git_archive_entries(tmp_path, "f" * 40)
```

Run:

```powershell
uv run --locked pytest tests/test_release_verification.py::test_git_archive_entries_read_committed_tree -q -p no:cacheprovider
```

Expected: FAIL because `_git_archive_entries` does not exist.

- [ ] **Step 7: Implement the Git archive adapter**

```python
def _git_archive_entries(project_root: Path, commit: str) -> list[PublicEntry]:
    try:
        process = subprocess.run(
            ["git", "-C", str(project_root), "archive", "--format=tar", commit],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ReleaseVerificationError(
            f"failed to read Git archive for commit {commit}"
        ) from exc
    return _parse_git_archive(process.stdout, commit=commit)
```

Rerun the adapter test.

Expected: `1 passed`.

- [ ] **Step 8: Commit Task 3**

```powershell
git add tests/test_release_verification.py src/rag/release_verification.py
git commit -m "feat: inspect Git history archives safely"
```

---

### Task 4: Verify Every Publishable Historical Tree

**Files:**
- Modify: `tests/test_release_verification.py:167-205,261-347`
- Modify: `src/rag/release_verification.py:345-430,499-521`

**Interfaces:**
- Consumes: `_publishable_commit_ids`, `_git_archive_entries`, `_scan_public_entries`, current allowlist, legacy path set, and reviewed historical binary digest set.
- Produces: `_verify_publishable_git_history(project_root: Path, current_public_paths: set[str], *, legacy_public_paths: set[str], reviewed_binary_hashes: set[str]) -> int`.

- [ ] **Step 1: Replace the old forbidden-path test with a failing public-history test**

Replace `test_reachable_history_rejects_a_forbidden_path` with:

```python
def test_publishable_history_rejects_forbidden_path_on_public_branch(tmp_path):
    init_public_repo(tmp_path)
    public_commit = commit_file(tmp_path, "README.md", b"public", "public")
    forbidden_commit = commit_file(
        tmp_path,
        ".claude/launch.json",
        b"{}",
        "forbidden",
    )
    run_git(tmp_path, "reset", "--hard", public_commit)
    run_git(tmp_path, "update-ref", "refs/heads/archive/review", forbidden_commit)

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="publishable history issues",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )
```

- [ ] **Step 2: Run the public-history test and verify RED**

```powershell
uv run --locked pytest tests/test_release_verification.py::test_publishable_history_rejects_forbidden_path_on_public_branch -q -p no:cacheprovider
```

Expected: FAIL because `_verify_publishable_git_history` does not exist.

- [ ] **Step 3: Implement minimal historical verification**

Replace `_verify_reachable_git_history` with:

```python
def _verify_publishable_git_history(
    project_root: Path,
    current_public_paths: set[str],
    *,
    legacy_public_paths: set[str],
    reviewed_binary_hashes: set[str],
) -> int:
    commits = _publishable_commit_ids(project_root)
    allowed_paths = current_public_paths | legacy_public_paths
    expected_identity = (
        "kuotunyu",
        "61350295+kuotunyu@users.noreply.github.com",
        "kuotunyu",
        "61350295+kuotunyu@users.noreply.github.com",
    )
    for commit in commits:
        entries = _git_archive_entries(project_root, commit)
        issues = _scan_public_entries(entries)
        for entry in entries:
            if entry.path not in allowed_paths:
                issues.append(_issue(entry.path, "unexpected_history_path", commit))
            if Path(entry.path).suffix.lower() in BINARY_PUBLIC_SUFFIXES:
                digest = hashlib.sha256(entry.data).hexdigest()
                if digest not in reviewed_binary_hashes:
                    issues.append(_issue(entry.path, "unreviewed_history_binary", commit))
        if issues:
            sanitized = [
                {**issue, "commit": commit}
                for issue in sorted(
                    issues,
                    key=lambda item: (item["path"], item["category"], item["location"]),
                )
            ]
            raise ReleaseVerificationError(
                f"publishable history issues: {sanitized}"
            )
        try:
            identity_process = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project_root),
                    "show",
                    "-s",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                    commit,
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            raise ReleaseVerificationError(
                f"failed to read identity for commit {commit}"
            ) from exc
        identity = tuple(identity_process.stdout.strip().split("\0"))
        _assert_equal(f"public commit identity {commit}", identity, expected_identity)
    return len(commits)
```

- [ ] **Step 4: Run the public forbidden-path test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed` and the exception contains no file contents.

- [ ] **Step 5: Add a failing removed-secret regression test**

```python
def test_publishable_history_rejects_secret_removed_from_current_tree(tmp_path):
    init_public_repo(tmp_path)
    secret = "sk-" + "A" * 32
    commit_file(tmp_path, "README.md", secret.encode("utf-8"), "leak")
    commit_file(tmp_path, "README.md", b"clean", "remove leak")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="provider_token",
    ) as caught:
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )

    assert secret not in str(caught.value)
```

Temporarily replace `_scan_public_entries(entries)` with `[]`, run the test to
verify it fails because no exception is raised, restore the scanner call, and
rerun.

Expected after restore: `1 passed`.

- [ ] **Step 6: Add legacy-path and historical-binary tests**

```python
def test_publishable_history_allows_explicit_legacy_path(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "legacy.md", b"old public file", "old release")
    (tmp_path / "legacy.md").unlink()
    (tmp_path / "README.md").write_bytes(b"current")
    run_git(tmp_path, "add", "--all")
    run_git(tmp_path, "commit", "-m", "new release")

    count = release_module()._verify_publishable_git_history(
        tmp_path,
        {"README.md"},
        legacy_public_paths={"legacy.md"},
        reviewed_binary_hashes=set(),
    )

    assert count == 2


def test_publishable_history_allows_current_file_absent_from_older_commit(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "README.md", b"first", "first release")
    commit_file(tmp_path, "CHANGELOG.md", b"added later", "second release")

    assert release_module()._verify_publishable_git_history(
        tmp_path,
        {"README.md", "CHANGELOG.md"},
        legacy_public_paths=set(),
        reviewed_binary_hashes=set(),
    ) == 2


def test_publishable_history_rejects_unlisted_legacy_path(tmp_path):
    init_public_repo(tmp_path)
    commit_file(tmp_path, "legacy.md", b"old public file", "old release")
    (tmp_path / "legacy.md").unlink()
    (tmp_path / "README.md").write_bytes(b"current")
    run_git(tmp_path, "add", "--all")
    run_git(tmp_path, "commit", "-m", "new release")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="unexpected_history_path",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )


def test_publishable_history_requires_reviewed_binary_hash(tmp_path):
    init_public_repo(tmp_path)
    binary = b"reviewed image bytes"
    commit_file(tmp_path, "image.png", binary, "image")
    digest = __import__("hashlib").sha256(binary).hexdigest()
    module = release_module()

    assert module._verify_publishable_git_history(
        tmp_path,
        {"image.png"},
        legacy_public_paths=set(),
        reviewed_binary_hashes={digest},
    ) == 1
    with pytest.raises(module.ReleaseVerificationError, match="unreviewed_history_binary"):
        module._verify_publishable_git_history(
            tmp_path,
            {"image.png"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )
```

Run both tests and expect `2 passed`.

- [ ] **Step 7: Add wrong-identity coverage**

```python
def test_publishable_history_rejects_unexpected_identity(tmp_path):
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    run_git(tmp_path, "config", "user.name", "Unexpected Author")
    private_email = "unexpected" + "@" + "example.test"
    run_git(tmp_path, "config", "user.email", private_email)
    commit_file(tmp_path, "README.md", b"public", "wrong identity")

    with pytest.raises(
        release_module().ReleaseVerificationError,
        match="public commit identity",
    ):
        release_module()._verify_publishable_git_history(
            tmp_path,
            {"README.md"},
            legacy_public_paths=set(),
            reviewed_binary_hashes=set(),
        )
```

Run the test and expect `1 passed`.

- [ ] **Step 8: Run all focused history tests**

```powershell
uv run --locked pytest tests/test_release_verification.py -k "publishable or git_archive or public_entry" -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 4**

```powershell
git add tests/test_release_verification.py src/rag/release_verification.py
git commit -m "feat: audit publishable Git history contents"
```

---

### Task 5: Migrate the Release Contract and Documentation

**Files:**
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `src/rag/release_verification.py:754-860`
- Modify: `tests/test_release_verification.py:40-115`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `docs/release/PUBLICATION_BOUNDARY.md`
- Modify: `docs/release/CLAIM_MATRIX.md`
- Modify: `docs/release/REVIEWER_GUIDE.md`
- Existing: `docs/release/RELEASE_EVOLUTION_DESIGN.md`
- Existing: `docs/release/RELEASE_EVOLUTION_IMPLEMENTATION_PLAN.md`

**Interfaces:**
- Consumes: `_verify_publishable_git_history` and manifest history fields.
- Produces: `verify_release()` report fields `publication.history_commits` and `publication.history_ref_namespaces`; an exact 96-file public inventory.

- [ ] **Step 1: Write failing real-repository report assertions**

Extend `test_release_verifier_recomputes_committed_evidence`:

```python
    expected_history = len(
        {
            line
            for line in run_git(
                PROJECT_ROOT,
                "rev-list",
                "--branches",
                "--tags",
                "--remotes",
            ).stdout.decode("ascii").splitlines()
            if line
        }
    )
    assert report["publication"]["history_commits"] == expected_history
    assert report["publication"]["history_ref_namespaces"] == [
        "heads",
        "tags",
        "remotes",
    ]
```

Update `test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions`:

```python
    assert len(tracked) == 96
    assert "docs/release/RELEASE_EVOLUTION_DESIGN.md" in tracked
    assert "docs/release/RELEASE_EVOLUTION_IMPLEMENTATION_PLAN.md" in tracked
```

- [ ] **Step 2: Run the two integration tests and verify RED**

```powershell
uv run --locked pytest tests/test_release_verification.py::test_release_verifier_recomputes_committed_evidence tests/test_release_verification.py::test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions -q -p no:cacheprovider
```

Expected: FAIL because the manifest still has `max_commits`, the verifier still
calls `_verify_reachable_git_history`, and the two documents are absent from the
allowlist.

- [ ] **Step 3: Migrate the manifest history schema**

Replace `publication.history` in `release/manifest.json` with:

```json
"history": {
  "refs": [
    "heads",
    "tags",
    "remotes"
  ],
  "legacy_public_paths": [],
  "reviewed_binary_sha256": [
    "ee95af1a9a8b92d2e1b8521da48e0a56fcc7447982122193bcc251ef95b86ecb"
  ]
}
```

Do not change `release_version`, model evidence, dataset evidence, runtime
configuration, or `publication.reviewed_binaries`.

- [ ] **Step 4: Add both planning documents to the sorted public inventory**

Insert these lines in `release/public-files.txt` between
`PUBLICATION_BOUNDARY.md` and `REVIEWER_GUIDE.md`:

```text
docs/release/RELEASE_EVOLUTION_DESIGN.md
docs/release/RELEASE_EVOLUTION_IMPLEMENTATION_PLAN.md
```

Run:

```powershell
uv run --locked python -c "from pathlib import Path; p=Path('release/public-files.txt').read_text(encoding='utf-8').splitlines(); assert p == sorted(p); assert len(p) == len(set(p)) == 96"
```

Expected: exit 0.

- [ ] **Step 5: Integrate history configuration into `verify_release`**

After loading `public_paths`, add:

```python
    history_config = manifest["publication"]["history"]
    _assert_equal(
        "publication history ref namespaces",
        history_config["refs"],
        ["heads", "tags", "remotes"],
    )
    legacy_public_paths = set(history_config["legacy_public_paths"])
    historical_binary_hashes = set(history_config["reviewed_binary_sha256"])
    current_binary_hashes = set(manifest["publication"]["reviewed_binaries"].values())
    _assert_equal(
        "current reviewed binaries included in history",
        current_binary_hashes <= historical_binary_hashes,
        True,
    )
```

Move current `scan_public_files` and `_verify_reviewed_binaries` calls before
the Git/source-archive branch. Replace the old history call with:

```python
        history_commits = _verify_publishable_git_history(
            root,
            set(public_paths),
            legacy_public_paths=legacy_public_paths,
            reviewed_binary_hashes=historical_binary_hashes,
        )
```

Add to the publication report:

```python
            "history_ref_namespaces": history_config["refs"],
```

- [ ] **Step 6: Run integration and source-archive regression tests**

Run the command from Step 2, then run:

```powershell
uv run --locked pytest tests/test_release_verification.py::test_source_archive_skips_only_git_tracking_check tests/test_release_verification.py::test_source_archive_inventory_rejects_non_generated_extras -q -p no:cacheprovider
```

Expected: all four tests pass. The local `refs/archive/*` ref remains present
but is not counted, and the Gitless source-archive behavior remains unchanged.

- [ ] **Step 7: Update publication-boundary documentation**

Make these exact claim changes:

- `docs/release/PUBLICATION_BOUNDARY.md`: replace “every reachable commit tree
  equals the 94-file allowlist” with the current-tree exact 96-file contract;
  define publishable refs as heads/tags/remotes; state that every publishable
  historical tree is path/content/binary scanned; state that `refs/archive/*`
  is local recovery evidence and excluded.
- `docs/release/CLAIM_MATRIX.md`: replace the one-commit row with a publishable
  history row mapping manifest namespaces, historical scanner, focused tests,
  and `scripts/verify_release.py`.
- `docs/release/REVIEWER_GUIDE.md`: change expected inventory to 96 files and
  replace “at most one initial release commit” with “all commits reachable from
  heads/tags/remotes pass identity and historical privacy scanning.”
- `README.md`: add one sentence in “Release evidence boundary” explaining that
  verifier results cover publishable refs but do not publish local recovery
  refs.
- `README.en.md`: add the equivalent calibrated English sentence.

Do not change retrieval, refusal, LLM-judge, corpus, or legal-safety claims.

- [ ] **Step 8: Add documentation contract assertions**

Add to `tests/test_release_verification.py`:

```python
def test_release_docs_define_publishable_history_boundary():
    boundary = (PROJECT_ROOT / "docs/release/PUBLICATION_BOUNDARY.md").read_text(
        encoding="utf-8"
    )
    reviewer = (PROJECT_ROOT / "docs/release/REVIEWER_GUIDE.md").read_text(
        encoding="utf-8"
    )

    assert "refs/heads/*" in boundary
    assert "refs/tags/*" in boundary
    assert "refs/remotes/*" in boundary
    assert "refs/archive/*" in boundary
    assert "96" in reviewer
    assert "at most one" not in reviewer
```

Run this test and expect `1 passed`.

- [ ] **Step 9: Run the release verifier directly**

```powershell
uv run --locked python scripts/verify_release.py
```

Expected: JSON report with `"status": "pass"`,
`"history_ref_namespaces": ["heads", "tags", "remotes"]`, zero public scan
issues, and a history count matching `git rev-list --branches --tags --remotes`.

- [ ] **Step 10: Commit Task 5**

```powershell
git add release/manifest.json release/public-files.txt src/rag/release_verification.py tests/test_release_verification.py README.md README.en.md docs/release/PUBLICATION_BOUNDARY.md docs/release/CLAIM_MATRIX.md docs/release/REVIEWER_GUIDE.md
git commit -m "feat: support evolving public release history"
```

---

### Task 6: Run the Complete Offline Release Gate

**Files:**
- Verify only; no intended tracked-file changes.

**Interfaces:**
- Consumes: the completed feature branch.
- Produces: fresh evidence for lint, release verification, tests, packaging,
  imports, Git cleanliness, and protected-ref invariants.

- [ ] **Step 1: Verify branch and protected refs before running gates**

```powershell
git branch --show-current
git rev-parse main
git rev-parse refs/remotes/origin/main
git rev-parse refs/tags/v0.1.0^{commit}
git rev-parse refs/archive/amended-away-before-public-snapshot-20260828
```

Expected:

```text
feat/v0.2-release-evolution-foundation
61385a24155a97af6b5d0201f9d81a1ad3eaf379
61385a24155a97af6b5d0201f9d81a1ad3eaf379
61385a24155a97af6b5d0201f9d81a1ad3eaf379
6527cbd35ee559573d83f83d3019c3d67c190e46
```

- [ ] **Step 2: Install the locked environment and verify the lock**

```powershell
$env:UV_LINK_MODE = "copy"
uv sync --locked
uv lock --check
```

Expected: both commands exit 0 without changing `uv.lock`.

- [ ] **Step 3: Run Ruff**

```powershell
uv run --locked ruff check .
```

Expected: `All checks passed!`.

- [ ] **Step 4: Run the release verifier**

```powershell
uv run --locked python scripts/verify_release.py
```

Expected: status pass, 40 questions, 8 ablation configurations, zero trace
issues, zero public scan issues, 96 public files, and publishable-history count
matching Git.

- [ ] **Step 5: Run the full test suite**

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
uv run --locked pytest -q -p no:cacheprovider
```

Expected: all tests pass with zero failures.

- [ ] **Step 6: Build both distributions in a unique temporary directory**

```powershell
$reviewBuild = Join-Path ([IO.Path]::GetTempPath()) ("labor-rag-v0.2-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $reviewBuild | Out-Null
uv build --out-dir $reviewBuild
Get-ChildItem -LiteralPath $reviewBuild -File | Select-Object Name,Length
```

Expected: one `.whl` and one `.tar.gz`, both non-empty. Do not delete the
temporary directory from inside the worktree workflow.

- [ ] **Step 7: Run package, FastAPI, and CLI smoke tests**

```powershell
uv run --locked pytest tests/test_packaging.py -q -p no:cacheprovider
uv run --locked python -W error::UserWarning -c "import sys; sys.path.insert(0, 'src'); import rag; import rag.api.main; print('FastAPI import: ok')"
uv run --locked python -W error::UserWarning scripts/ask.py --help
```

Expected: packaging test passes, FastAPI prints `FastAPI import: ok`, and CLI
help exits 0 without loading a model or provider.

- [ ] **Step 8: Verify clean worktree and exact commit scope**

```powershell
git status --short
git diff --check main...HEAD
git diff --name-status main...HEAD
```

Expected: status is clean; diff check exits 0; changed files are limited to the
specification, implementation plan, release verifier, its tests, manifest,
public inventory, and release documentation listed in this plan.

- [ ] **Step 9: Reverify protected refs and absence of archive-branch tracking**

```powershell
git rev-parse main
git rev-parse refs/remotes/origin/main
git rev-parse refs/tags/v0.1.0^{commit}
git rev-parse refs/archive/amended-away-before-public-snapshot-20260828
git config --get branch.feat/v0.2-release-evolution-foundation.remote
```

Expected: the four hashes match Step 1. The final config command exits 1 with
no output, proving the feature branch has no upstream.

## Completion Evidence to Report

When execution finishes, report:

- feature branch and final commit list;
- exact changed-file list;
- Ruff result;
- release verifier summary, including publishable history count and ref namespaces;
- pytest pass count and warnings, if any;
- package artifact names and sizes;
- FastAPI/CLI smoke results;
- protected-ref hashes and recovery-ref preservation;
- confirmation that no fetch, push, merge, rebase, tag, or provider/model call occurred.
