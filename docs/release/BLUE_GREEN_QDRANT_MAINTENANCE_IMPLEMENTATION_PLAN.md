# Blue-Green Qdrant Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a dry-run-first, create-only manual maintenance command that can build and validate an audited candidate pair of Qdrant collections without modifying the active pair.

**Architecture:** Pure contracts in `rag.qdrant_maintenance` validate candidate names, corpus snapshots, payload provenance, and redacted receipts. A small orchestration module owns the create-only two-strategy build, while a thin CLI supplies environment credentials and explicit execution confirmation. Existing runtime retrieval and destructive local-development indexing remain unchanged.

**Tech Stack:** Python 3.11, Pydantic settings, qdrant-client 1.18, NumPy, BGE-M3, pytest, Ruff, Bandit, pip-audit, uv, GitHub CLI.

## Global Constraints

- Do not move, recreate, force-push, or delete the existing `v0.3.4` tag.
- Do not schedule a workflow, monitor, heartbeat, or unattended external write.
- Do not request paid Qdrant, Hugging Face hardware, storage, replica, or GPU resources.
- Do not call Gemini or OpenAI and do not read owner provider keys.
- Dry-run must not instantiate a cloud client, write a receipt, or load model weights.
- Execute mode must use `QDRANT_WRITER_API_KEY`, never the runtime `QDRANT_API_KEY`.
- The active collection base and both active collections must never be recreated, overwritten, or deleted.
- Candidate cleanup and old-collection deletion are outside this plan.
- Receipts and logs must not contain credentials, endpoints, absolute paths, law text, questions, answers, prompts, or provider payloads.
- Every new public path must be added to sorted `release/public-files.txt`.
- Tests must use fakes or isolated local Qdrant; no Qdrant Cloud mutation occurs in automated verification.

---

## File map

- `src/rag/qdrant_maintenance.py`: pure validation, local snapshot reconstruction, and redacted receipt contracts.
- `src/rag/qdrant_blue_green.py`: create-only candidate orchestration with injected stores and embedders.
- `src/rag/indexing/vector_store.py`: additive create-only and existence methods used by the orchestrator.
- `scripts/rebuild_qdrant_blue_green.py`: command-line parsing, dry-run output, environment credential gate, and dependency wiring.
- `tests/test_qdrant_maintenance.py`: pure contract and privacy tests.
- `tests/test_qdrant_blue_green.py`: fake-backed orchestration tests proving no delete or overwrite path.
- `tests/test_qdrant_blue_green_cli.py`: dry-run and execution-gate tests.
- `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`: exact operator commands, cutover, rollback, and writer-key revocation.
- `README.md`: short pointer to the safe maintenance command.
- `release/public-files.txt`: exact public inventory additions.

---

### Task 1: Pure maintenance contracts and audited local snapshot

**Files:**
- Create: `src/rag/qdrant_maintenance.py`
- Create: `tests/test_qdrant_maintenance.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: `rag.corpus_audit.build_snapshot`, the committed snapshot mapping, cached source archive paths, and normalized law JSON paths.
- Produces: `candidate_collections`, `validate_candidate_base`, `build_local_snapshot`, `validate_snapshot_match`, `validate_candidate_payloads`, `build_maintenance_receipt`, and `write_receipt_atomic`.

- [ ] **Step 1: Write failing candidate-name and snapshot tests**

```python
from copy import deepcopy

import pytest

from rag.qdrant_maintenance import (
    candidate_collections,
    validate_candidate_base,
    validate_snapshot_match,
)


def test_candidate_base_must_be_distinct_and_portable():
    assert candidate_collections("labor_laws_20260830_deadbeef") == {
        "fixed": "labor_laws_20260830_deadbeef_fixed",
        "structure": "labor_laws_20260830_deadbeef_structure",
    }
    with pytest.raises(ValueError, match="active base"):
        validate_candidate_base("labor_laws", "labor_laws")
    with pytest.raises(ValueError, match="portable"):
        validate_candidate_base("labor_laws", "Labor Laws/next")


def test_snapshot_match_ignores_only_observation_date(committed_snapshot):
    local = deepcopy(committed_snapshot)
    local["snapshot_date"] = "2026-08-30"
    validate_snapshot_match(committed_snapshot, local)
    local["laws"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content_sha256"):
        validate_snapshot_match(committed_snapshot, local)
```

- [ ] **Step 2: Run the focused tests and confirm the import failure**

Run: `uv run --locked pytest tests/test_qdrant_maintenance.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'rag.qdrant_maintenance'`.

- [ ] **Step 3: Implement candidate names and exact snapshot comparison**

```python
_CANDIDATE_BASE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_STRATEGIES = ("fixed", "structure")


def candidate_collections(base: str) -> dict[str, str]:
    return {strategy: f"{base}_{strategy}" for strategy in _STRATEGIES}


def validate_candidate_base(active_base: str, candidate_base: str) -> None:
    if candidate_base == active_base:
        raise ValueError("candidate base must differ from active base")
    if not _CANDIDATE_BASE.fullmatch(candidate_base):
        raise ValueError("candidate base must be a portable lowercase name")


def validate_snapshot_match(committed: Mapping, local: Mapping) -> None:
    expected = dict(committed)
    observed = dict(local)
    expected.pop("snapshot_date", None)
    observed.pop("snapshot_date", None)
    if expected != observed:
        changes = compare_snapshots(committed, local)
        kinds = sorted({str(row.get("kind", "unknown")) for row in changes})
        raise ValueError(f"corpus snapshot drift: {','.join(kinds)}")
```

- [ ] **Step 4: Add local archive/law reconstruction tests**

```python
def test_build_local_snapshot_hashes_archives_and_laws(tmp_path, committed_snapshot):
    archives, law_dir = write_local_corpus_fixture(tmp_path, committed_snapshot)
    observed = build_local_snapshot(
        source_archives=archives,
        laws_dir=law_dir,
        snapshot_date="2026-08-30",
    )
    assert observed["law_count"] == 15
    assert observed["article_count"] == 884
    assert {row["id"] for row in observed["sources"]} == {"acts", "regulations"}
```

- [ ] **Step 5: Implement local reconstruction using canonical helpers**

```python
def build_local_snapshot(
    *,
    source_archives: Mapping[str, tuple[str, Path]],
    laws_dir: Path,
    snapshot_date: str,
) -> dict[str, object]:
    sources = [
        {"id": source_id, "url": url, "sha256": sha256_file(path)}
        for source_id, (url, path) in source_archives.items()
    ]
    laws = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(laws_dir.glob("*.json")) if path.name != "manifest.json"]
    return build_snapshot(sources=sources, laws=laws, snapshot_date=snapshot_date)
```

- [ ] **Step 6: Add payload and receipt privacy tests**

```python
def test_payloads_require_complete_public_provenance(valid_payload):
    validate_candidate_payloads("structure", [valid_payload], expected_count=1)
    for field in ("doc_title", "article_label", "source_url", "last_amended"):
        broken = dict(valid_payload)
        broken[field] = ""
        with pytest.raises(ValueError, match=field):
            validate_candidate_payloads("structure", [broken], expected_count=1)

    missing_effective_date = dict(valid_payload)
    del missing_effective_date["effective_date"]
    with pytest.raises(ValueError, match="effective_date"):
        validate_candidate_payloads(
            "structure", [missing_effective_date], expected_count=1
        )


def test_receipt_has_exact_redacted_schema(receipt_input):
    receipt = build_maintenance_receipt(**receipt_input)
    assert set(receipt) == {
        "schema_version", "completed_at", "active_base", "candidate_base",
        "collections", "corpus_snapshot_sha256", "source_sha256",
        "embedding_model", "embedding_revision", "vector_dimension",
    }
    serialized = json.dumps(receipt)
    assert "https://" not in serialized
    assert "api_key" not in serialized.lower()
```

- [ ] **Step 7: Implement payload and receipt validation plus atomic write**

Implement fixed field allowlists, `YYYYMMDD` date validation, official
`https://law.moj.gov.tw/` URL validation, exact count validation, UTC ISO time,
and `temporary.replace(target)` atomic receipt publication. Require the
`effective_date` field but allow it to be empty when the audited official
source has no value; never synthesize one. Accept only a project-relative
receipt target beneath `eval/runs/qdrant-maintenance/`.

- [ ] **Step 8: Run Task 1 tests and release-boundary checks**

Run:

```powershell
uv run --locked pytest tests/test_qdrant_maintenance.py -q -p no:cacheprovider
uv run --locked ruff check src/rag/qdrant_maintenance.py tests/test_qdrant_maintenance.py
uv run --locked pytest tests/test_release_verification.py::test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions -q -p no:cacheprovider
```

Expected: all commands exit 0.

- [ ] **Step 9: Commit Task 1**

```powershell
git add src/rag/qdrant_maintenance.py tests/test_qdrant_maintenance.py release/public-files.txt
git commit -m "feat: add qdrant maintenance contracts"
```

---

### Task 2: Create-only VectorStore operations

**Files:**
- Modify: `src/rag/indexing/vector_store.py`
- Modify: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: existing `VectorStore.client` and Qdrant vector parameters.
- Produces: `collection_exists(name: str) -> bool` and `create_collection(name: str, dim: int) -> None`; `create_collection` must never call delete.

- [ ] **Step 1: Write failing create-only tests**

```python
def test_create_collection_refuses_existing_without_delete(fake_client, settings):
    fake_client.collection_exists.return_value = True
    store = make_store(settings, fake_client)
    with pytest.raises(ValueError, match="already exists"):
        store.create_collection("candidate_structure", dim=1024)
    fake_client.delete_collection.assert_not_called()
    fake_client.create_collection.assert_not_called()


def test_create_collection_creates_absent_target(fake_client, settings):
    fake_client.collection_exists.return_value = False
    store = make_store(settings, fake_client)
    store.create_collection("candidate_structure", dim=1024)
    fake_client.create_collection.assert_called_once()
    fake_client.delete_collection.assert_not_called()
```

- [ ] **Step 2: Run the two tests and confirm missing-method failures**

Run: `uv run --locked pytest tests/test_vector_store.py -q -p no:cacheprovider`

Expected: failures report that `VectorStore` has no `create_collection` method.

- [ ] **Step 3: Implement create-only operations**

```python
def collection_exists(self, name: str) -> bool:
    return bool(self.client.collection_exists(name))


def create_collection(self, name: str, dim: int) -> None:
    self._require_writable()
    if self.collection_exists(name):
        raise ValueError(f"collection already exists: {name}")
    self.client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
```

- [ ] **Step 4: Run vector-store and BYOK policy tests**

Run:

```powershell
uv run --locked pytest tests/test_vector_store.py tests/test_byok_policy.py -q -p no:cacheprovider
uv run --locked ruff check src/rag/indexing/vector_store.py tests/test_vector_store.py
```

Expected: all commands exit 0 and existing `recreate_collection` tests remain unchanged.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/rag/indexing/vector_store.py tests/test_vector_store.py
git commit -m "feat: add create-only qdrant collections"
```

---

### Task 3: Dry-run-first CLI and credential gates

**Files:**
- Create: `scripts/rebuild_qdrant_blue_green.py`
- Create: `tests/test_qdrant_blue_green_cli.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: Task 1 validation functions, `Settings`, `release/corpus_snapshot.json`, cached archives, and normalized laws.
- Produces: `parse_args(argv) -> argparse.Namespace`, `build_dry_run_plan(args, settings) -> dict`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing dry-run tests**

```python
def test_dry_run_never_reads_writer_credentials_or_creates_client(monkeypatch, corpus_fixture):
    monkeypatch.delenv("QDRANT_WRITER_API_KEY", raising=False)
    result = cli.main([
        "--candidate-base", "labor_laws_20260830_deadbeef",
        "--corpus", str(corpus_fixture.laws_dir),
        "--raw-dir", str(corpus_fixture.raw_dir),
    ])
    assert result == 0
    assert corpus_fixture.client_factory.calls == []
    assert not corpus_fixture.receipt_dir.exists()


def test_execute_requires_repeated_candidate_confirmation(monkeypatch):
    result = cli.main([
        "--execute",
        "--candidate-base", "labor_laws_20260830_deadbeef",
        "--confirm-candidate-base", "labor_laws_wrong",
    ])
    assert result == 2
```

- [ ] **Step 2: Run the CLI tests and confirm the missing-script failure**

Run: `uv run --locked pytest tests/test_qdrant_blue_green_cli.py -q -p no:cacheprovider`

Expected: collection fails because the CLI module does not exist.

- [ ] **Step 3: Implement argument parsing and the dry-run gate**

The parser must expose only:

```text
--candidate-base NAME          required
--confirm-candidate-base NAME  required only with --execute
--active-base NAME             defaults to COLLECTION_NAME or labor_laws
--corpus PATH                  defaults to data/raw/laws
--raw-dir PATH                 defaults to data/raw
--snapshot PATH                defaults to release/corpus_snapshot.json
--receipt PATH                 defaults inside eval/runs/qdrant-maintenance
--device cpu|cuda              defaults to cpu
--execute                      absent means dry-run
```

Dry-run validates the candidate and exact local snapshot, derives collection
names, and prints JSON containing only `status`, `active_base`,
`candidate_base`, `collections`, `snapshot_sha256`, and `execution_required`.

- [ ] **Step 4: Add execute-mode environment and cache tests**

```python
@pytest.mark.parametrize("missing", ["QDRANT_URL", "QDRANT_WRITER_API_KEY"])
def test_execute_fails_before_client_when_required_environment_missing(monkeypatch, missing):
    configure_other_required_environment(monkeypatch, missing)
    assert cli.main(valid_execute_args()) == 2
    assert fake_client_factory.calls == []


def test_execute_checks_both_pinned_snapshots_before_client(monkeypatch):
    monkeypatch.setattr(cli, "snapshot_download", raise_local_entry_not_found)
    assert cli.main(valid_execute_args()) == 2
    assert fake_client_factory.calls == []
```

- [ ] **Step 5: Implement sanitized execute preflight**

Use `huggingface_hub.snapshot_download(..., local_files_only=True)` for the
pinned embedding and reranker revisions. Print only fixed error codes such as
`missing_writer_environment`, `missing_model_snapshot`, or `snapshot_drift`;
never print exception text, endpoint values, or environment values.

- [ ] **Step 6: Run Task 3 tests and CLI help smoke test**

Run:

```powershell
uv run --locked pytest tests/test_qdrant_blue_green_cli.py -q -p no:cacheprovider
uv run --locked python scripts/rebuild_qdrant_blue_green.py --help
uv run --locked ruff check scripts/rebuild_qdrant_blue_green.py tests/test_qdrant_blue_green_cli.py
```

Expected: all commands exit 0; help text lists the exact options above and does not load a model.

- [ ] **Step 7: Commit Task 3**

```powershell
git add scripts/rebuild_qdrant_blue_green.py tests/test_qdrant_blue_green_cli.py release/public-files.txt
git commit -m "feat: add dry-run blue-green qdrant cli"
```

---

### Task 4: Candidate build orchestration and redacted receipt

**Files:**
- Create: `src/rag/qdrant_blue_green.py`
- Create: `tests/test_qdrant_blue_green.py`
- Modify: `scripts/rebuild_qdrant_blue_green.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: `VectorStore.create_collection`, `VectorStore.upsert_chunks`, `VectorStore.count`, `VectorStore.scroll_payloads`, `BGEM3Embedder`, `get_chunker`, and Task 1 validators.
- Produces: `BuildDependencies`, `build_candidates(request, dependencies) -> dict`, and a successful redacted receipt payload.

- [ ] **Step 1: Write failing no-overwrite orchestration tests**

```python
def test_existing_candidate_blocks_before_any_create(fake_dependencies, request):
    fake_dependencies.store.existing.add(request.collections["fixed"])
    with pytest.raises(ValueError, match="candidate collection exists"):
        build_candidates(request, fake_dependencies)
    assert fake_dependencies.store.created == []
    assert fake_dependencies.store.deleted == []


def test_failure_never_deletes_partial_candidate(fake_dependencies, request):
    fake_dependencies.store.fail_upsert_for = request.collections["structure"]
    with pytest.raises(RuntimeError):
        build_candidates(request, fake_dependencies)
    assert fake_dependencies.store.deleted == []
```

- [ ] **Step 2: Run Task 4 tests and confirm missing-module failure**

Run: `uv run --locked pytest tests/test_qdrant_blue_green.py -q -p no:cacheprovider`

Expected: collection fails with missing `rag.qdrant_blue_green`.

- [ ] **Step 3: Define explicit injected dependencies**

```python
@dataclass(frozen=True)
class BuildRequest:
    active_base: str
    candidate_base: str
    corpus_dir: Path
    receipt_path: Path
    snapshot_sha256: str

    @property
    def collections(self) -> dict[str, str]:
        return candidate_collections(self.candidate_base)


@dataclass(frozen=True)
class BuildDependencies:
    store: CandidateStore
    embedder: CandidateEmbedder
    settings: Settings
    completed_at: Callable[[], datetime]
```

The `CandidateStore` protocol contains only `collection_exists`,
`create_collection`, `upsert_chunks`, `count`, `scroll_payloads`, and `close`.
It deliberately contains no delete or recreate method.

- [ ] **Step 4: Implement preflight-all-then-build sequencing**

```python
def build_candidates(request: BuildRequest, dependencies: BuildDependencies) -> dict:
    collections = candidate_collections(request.candidate_base)
    if any(dependencies.store.collection_exists(name) for name in collections.values()):
        raise ValueError("candidate collection exists")
    prepared = prepare_all_strategies(request.corpus_dir, dependencies)
    for strategy in ("fixed", "structure"):
        chunks, vectors = prepared[strategy]
        name = collections[strategy]
        dependencies.store.create_collection(name, dim=vectors.shape[1])
        dependencies.store.upsert_chunks(name, chunks, vectors)
        if dependencies.store.count(name) != len(chunks):
            raise ValueError(f"candidate count mismatch: {strategy}")
        validate_candidate_payloads(
            strategy,
            dependencies.store.scroll_payloads(name),
            expected_count=len(chunks),
        )
    return build_maintenance_receipt(...)
```

Prepare chunks and vectors for both strategies before the first create so a
local model/chunking failure cannot leave a cloud collection behind.

- [ ] **Step 5: Add exact sequencing, count, payload, and receipt tests**

```python
def test_success_builds_fixed_then_structure_and_returns_redacted_receipt(fake_dependencies, request):
    receipt = build_candidates(request, fake_dependencies)
    assert fake_dependencies.store.created == [
        request.collections["fixed"],
        request.collections["structure"],
    ]
    assert receipt["collections"]["fixed"]["points"] == len(fixed_chunks)
    assert receipt["collections"]["structure"]["points"] == 884
    assert "url" not in json.dumps(receipt).lower()
```

- [ ] **Step 6: Wire execute mode and atomic receipt publication**

The CLI creates a Qdrant server-mode `Settings` instance using
`QDRANT_WRITER_API_KEY` copied into a process-local `SecretStr`, constructs the
store and pinned embedder, calls `build_candidates`, writes the receipt only
after success, closes the store in `finally`, and clears the temporary settings
reference. It must not modify `.env`.

- [ ] **Step 7: Run Task 4 focused verification**

Run:

```powershell
uv run --locked pytest tests/test_qdrant_blue_green.py tests/test_qdrant_blue_green_cli.py tests/test_vector_store.py -q -p no:cacheprovider
uv run --locked ruff check src/rag/qdrant_blue_green.py scripts/rebuild_qdrant_blue_green.py tests/test_qdrant_blue_green.py tests/test_qdrant_blue_green_cli.py
```

Expected: all commands exit 0, with no network provider or cloud Qdrant call.

- [ ] **Step 8: Commit Task 4**

```powershell
git add src/rag/qdrant_blue_green.py scripts/rebuild_qdrant_blue_green.py tests/test_qdrant_blue_green.py tests/test_qdrant_blue_green_cli.py release/public-files.txt
git commit -m "feat: build validated qdrant candidates"
```

---

### Task 5: Operator runbook and public release boundary

**Files:**
- Modify: `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`
- Modify: `README.md`
- Modify: `tests/test_byok_policy.py`
- Modify: `tests/test_release_verification.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: the exact CLI and receipt contracts from Tasks 1-4.
- Produces: copy-safe PowerShell commands, private acceptance steps, rollback, and mandatory writer-key revocation instructions.

- [ ] **Step 1: Write failing runbook policy tests**

```python
def test_runbook_requires_blue_green_and_forbids_in_place_cloud_rebuild():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "rebuild_qdrant_blue_green.py" in text
    assert "QDRANT_WRITER_API_KEY" in text
    assert "--confirm-candidate-base" in text
    assert "不得對正式 collection 執行 build_index.py" in text
    assert "撤銷 temporary writer key" in text


def test_runbook_keeps_refresh_manual_and_free():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "不建立排程" in text
    assert "CPU Basic" in text
    assert "Qdrant Free" in text
```

- [ ] **Step 2: Run the runbook tests and confirm missing-copy failures**

Run: `uv run --locked pytest tests/test_byok_policy.py -q -p no:cacheprovider`

Expected: the new assertions fail against the current runbook.

- [ ] **Step 3: Document the exact dry-run and execute commands**

The runbook must show:

```powershell
uv run python scripts/download_corpus.py --force-download
uv run python scripts/audit_corpus.py
uv run python scripts/rebuild_qdrant_blue_green.py --candidate-base labor_laws_YYYYMMDD_HASH

$env:QDRANT_URL = Read-Host 'Paste the Qdrant cluster endpoint'
$env:QDRANT_WRITER_API_KEY = Read-Host -MaskInput 'Paste the temporary writer key'
uv run python scripts/rebuild_qdrant_blue_green.py --execute `
  --candidate-base labor_laws_YYYYMMDD_HASH `
  --confirm-candidate-base labor_laws_YYYYMMDD_HASH
Remove-Item Env:QDRANT_WRITER_API_KEY
Remove-Item Env:QDRANT_URL
```

The surrounding copy must state that the placeholders are operator-selected
non-secret names, commands must run attended, the active collections remain
untouched, and the writer key must be revoked in Qdrant Cloud after either
success or failure.

- [ ] **Step 4: Document cutover without automating it**

Add the exact private acceptance sequence: record old `COLLECTION_NAME`, change
to the candidate base, restart, require RUNNING/READY/health 200, inspect logs,
rollback to the old base on failure, and revoke the writer key. State that old
collection deletion is a separate destructive operation not authorized by the
rebuild command.

- [ ] **Step 5: Add README pointer and release-boundary assertions**

Add a short `Manual corpus maintenance` section linking the runbook and design.
Update the exact public allowlist for every new source, test, script, and plan
path. Extend the release-boundary test to assert the new maintenance files are
present and `tracked_excluded` remains empty.

- [ ] **Step 6: Run documentation and release verification**

Run:

```powershell
uv run --locked pytest tests/test_byok_policy.py tests/test_release_verification.py -q -p no:cacheprovider
uv run --locked python scripts/verify_release.py
uv run --locked ruff check .
```

Expected: tests pass, verifier reports `status=pass`, public scan issues 0, and the updated exact publication count.

- [ ] **Step 7: Commit Task 5**

```powershell
git add README.md docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md tests/test_byok_policy.py tests/test_release_verification.py release/public-files.txt
git commit -m "docs: add safe qdrant maintenance runbook"
```

---

### Task 6: Full security gate, PR, and v0.3.4 GitHub Release

**Files:**
- No new repository files unless a verification defect requires a focused test and fix.

**Interfaces:**
- Consumes: Tasks 1-5 commits and the immutable `v0.3.4` annotated tag.
- Produces: a green PR against `main`, unchanged live Qdrant/Hugging Face state, and a published GitHub Release for the existing tag.

- [ ] **Step 1: Run the full local gate**

Run:

```powershell
uv lock --check
uv run --locked pytest -q -p no:cacheprovider
uv run --locked ruff check .
uv run --locked bandit -r src scripts -ll
uv run --locked pip-audit --local
uv run --locked python scripts/verify_release.py
uv build --out-dir dist
```

Expected: 0 test failures, Ruff clean, no Medium/High Bandit findings, no known
dependency vulnerabilities, verifier `status=pass` with 0 privacy findings,
and package build exit 0. A non-PyPI local CUDA torch build may be reported as
skipped by pip-audit but must not be reported as a known vulnerability.

- [ ] **Step 2: Prove no external runtime mutation occurred**

Record sanitized before/after evidence:

```text
Qdrant candidate collections created: 0
Hugging Face hardware mutation: none
Hugging Face visibility mutation: none
Hugging Face variable/secret mutation: none
Provider calls: 0
Scheduled automations created: 0
```

Use read-only Hugging Face runtime inspection and repository-local fakes; do not request or print Qdrant credentials.

- [ ] **Step 3: Push the branch and create a PR**

```powershell
git push -u origin codex/blue-green-qdrant-maintenance
$body = @'
## Summary
- add dry-run-first blue-green Qdrant maintenance
- forbid overwrite/delete of active collections
- validate audited corpus, provenance, counts, and redacted receipts

## Safety boundary
- no cloud mutation occurred in tests or branch preparation
- actual execution requires an attended temporary writer key
- no scheduled job, paid hardware, provider call, or owner LLM key

## Verification
- full pytest, Ruff, Bandit, pip-audit, build, and release verifier
'@
gh pr create --base main --head codex/blue-green-qdrant-maintenance `
  --title "feat: add safe blue-green Qdrant maintenance" `
  --body $body
```

The PR body states that the branch performs no cloud mutation and that actual
execution remains gated on an attended temporary writer key.

- [ ] **Step 4: Wait for required CI and merge without rewriting history**

Require the `test` check conclusion `SUCCESS`. Use the repository's allowed
merge strategy; do not force-push, move tags, or bypass branch protection.

- [ ] **Step 5: Publish the immutable v0.3.4 GitHub Release**

Before creation, confirm `gh release view v0.3.4` returns not found and
`git rev-parse v0.3.4^{commit}` still resolves to
`3ec5adefd6361ba92913106adbda18d4b65d7620`.

Create release notes that state:

- deterministic wage-arrears Article 14 retrieval expansion;
- no provider, threshold, Qdrant, prompt, or historical metric change;
- tagged source archive scope ends at the release-preparation commit;
- additive targeted regression evidence is available on `main` at commit
  `b26319b1f602250764d93a806b13c1c2bfec6ecc`;
- live BYOK demo URL;
- visitor-owned API keys and free CPU/Qdrant boundaries.

Run:

```powershell
$notes = @'
## v0.3.4 — Wage Arrears Retrieval Hardening

This release adds deterministic retrieval expansion for questions that combine
wage arrears with a worker's immediate termination right under Labor Standards
Act Article 14. BM25, dense retrieval, and reranking receive the legal expansion;
the generation provider still receives the original user question.

### Evidence boundary

- No provider, prompt, threshold, Qdrant data, or historical metric changed.
- This tag's source archive ends at the release-preparation commit.
- Additive targeted regression evidence is available on `main` at
  `b26319b1f602250764d93a806b13c1c2bfec6ecc`.

### Live demo and cost boundary

- Demo: https://steven0226-tw-labor-law-rag-demo.hf.space/
- Visitors use their own Gemini or OpenAI API key.
- The deployment uses Hugging Face CPU Basic and Qdrant Free Tier.
'@
gh release create v0.3.4 --verify-tag `
  --title "v0.3.4 — Wage Arrears Retrieval Hardening" `
  --notes $notes
```

- [ ] **Step 6: Verify final GitHub and local state**

Run:

```powershell
gh pr view --json state,mergedAt,url,statusCheckRollup
gh release view v0.3.4 --json tagName,name,url,isDraft,isPrerelease
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/blue-green-qdrant-maintenance
```

Expected: PR merged with required CI success, v0.3.4 release published and not
draft/prerelease, branch worktree clean, and local/remote branch heads equal.

The actual Qdrant blue-green execute/cutover is a later attended operation. It
must not be performed merely because this implementation plan has passed.
