# v0.3.5 Release and Private Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the verified portfolio/evidence work into a coherent v0.3.5 release, deploy the exact application candidate to the existing private free Hugging Face Space, and leave a non-secret audit trail.

**Architecture:** Treat the repository verifier as the release root of trust. First align version and documentation state, then add a read-only private-Space preflight, run the complete local gate, deploy one identified application commit, record the remote receipt, and merge/tag only after GitHub CI passes. The Space remains private and uses free CPU hardware; the GitHub repository remains the public technical portfolio.

**Ordering refinement:** The approved design listed tag creation before deployment. Because a Space revision does not exist until upload, that order cannot bind the deployment receipt into the same immutable release. This plan deploys the already verified candidate first, commits its non-secret receipt, then rebase-merges and tags the identical application tree. Any rebase that changes an application file invalidates the receipt and forces a repeat deployment before tagging.

**Tech Stack:** Python 3.11, uv, pytest, Ruff, Bandit, pip-audit, build, Git/GitHub CLI, huggingface_hub, Hugging Face Docker Spaces.

## Global Constraints

- Keep `steven0226/tw-labor-law-rag-demo` private, `cpu-basic`, one replica, no paid persistent storage, and no paid accelerator.
- Never add owner Gemini/OpenAI keys. The Space may have only runtime Qdrant/session secrets and non-secret deployment variables; visitors use their own provider keys.
- Never print or persist secret values, Qdrant endpoints, signed session tokens, local absolute paths, or provider request data.
- Qdrant remains Free Tier and runtime access remains read-only to the v0.3.4 candidate collection pair.
- Do not create schedules, monitors, heartbeats, cron jobs, or background paid resources.
- Do not move or overwrite existing release tags. Abort if `v0.3.5` already exists locally, remotely, or as a GitHub Release.
- Use a rebase merge to preserve the repository's linear history; do not force-push `main`.
- Do not clean legacy local repositories or worktrees as part of this release.
- Deployment/release receipts contain only non-secret state and exact immutable identifiers.

---

## File map

- `pyproject.toml`: package version `0.3.5`.
- `README.md`: current release label and private-demo state.
- `README.en.md`: equivalent English state.
- `release/manifest.json`: v0.3.5 version, publication count, new evidence, documentation, and receipt hashes.
- `release/public-files.txt`: Python-sorted exact public inventory.
- `tests/test_release_verification.py`: version/state/publication assertions.
- `scripts/verify_private_space.py`: read-only, redacted Hugging Face Space policy preflight.
- `tests/test_private_space_verification.py`: free/private/no-owner-key policy tests.
- `scripts/verify_qdrant_reader.py`: read-only candidate/count/legacy-denial verification with redacted output.
- `tests/test_qdrant_reader_verification.py`: Qdrant reader-policy tests using fake HTTP transports.
- `docs/release/V035_DEPLOYMENT_RECEIPT.md`: source SHA, remote revision, free/private posture, collection base, counts, and acceptance outcome.
- `docs/release/V035_RELEASE_NOTES.md`: evidence-calibrated v0.3.5 GitHub Release notes.
- `docs/release/CLAIM_MATRIX.md`: final claim-state alignment.
- `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`: exact v0.3.5 verification and rollback commands.

---

### Task 1: Release-version and documentation alignment

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `README.en.md`
- Create: `docs/release/V035_RELEASE_NOTES.md`
- Modify: `docs/release/CLAIM_MATRIX.md`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `tests/test_release_verification.py`

**Interfaces:**
- Consumes: completed portfolio/UI and evidence checkpoints.
- Produces: one coherent `v0.3.5` release identity across package metadata, manifest, README, claims, and notes.

- [ ] **Step 1: Write failing version/state assertions**

```python
def test_v035_version_and_release_documents_are_aligned():
    manifest = json.loads((PROJECT_ROOT / "release/manifest.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "0.3.5"
    assert manifest["release_version"] == "v0.3.5"
    public_paths = set(manifest["publication"]["public_paths"])
    assert "docs/release/V035_RELEASE_NOTES.md" in public_paths
    assert "docs/release/V035_DEPLOYMENT_RECEIPT.md" in public_paths
```

- [ ] **Step 2: Run the focused test and confirm v0.3.4 state fails**

Run: `uv run pytest tests/test_release_verification.py -q -p no:cacheprovider`

Expected: FAIL because package and manifest still identify v0.3.4 and the v0.3.5 documents are absent.

- [ ] **Step 3: Align version and release prose**

Set the package and manifest release to v0.3.5 without changing dependency constraints. Replace any README statement that implies the private Space is public. Write release notes with: reviewer journey/UI, ten-case deterministic demonstration evidence, manual freshness summary, preserved v0.3.4 Qdrant candidate, unchanged formal benchmark metrics, security/cost boundaries, known limitations, and rollback.

- [ ] **Step 4: Update exact publication inventory**

Add every v0.3.5 design, plan, reviewer, evaluation, fixture, receipt, and release-notes path. Regenerate manifest public-path/count/hash fields using repository helpers rather than hand-counting. Preserve exactly one reviewed binary.

- [ ] **Step 5: Run version/publication checks**

```powershell
uv run pytest tests/test_release_verification.py tests/test_packaging.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
uv lock --check
git diff --check
```

Expected: all commands exit 0 and report `v0.3.5` with the exact tracked public inventory.

- [ ] **Step 6: Commit Task 1**

```powershell
git add pyproject.toml README.md README.en.md docs/release/V035_RELEASE_NOTES.md docs/release/CLAIM_MATRIX.md release/manifest.json release/public-files.txt tests/test_release_verification.py
git commit -m "chore: align v0.3.5 release state"
```

---

### Task 2: Redacted private-Space preflight

**Files:**
- Create: `scripts/verify_private_space.py`
- Create: `tests/test_private_space_verification.py`
- Create: `scripts/verify_qdrant_reader.py`
- Create: `tests/test_qdrant_reader_verification.py`
- Modify: `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: `HfApi.repo_info(..., repo_type="space")`, `HfApi.get_space_runtime(...)`, `HfApi.get_space_variables(...)`, and `HfApi.get_space_secrets(...)` metadata.
- Produces: a redacted JSON policy report containing booleans, names, hardware/stage, and collection base; it never contains secret values.

- [ ] **Step 1: Write failing pure policy tests**

```python
from scripts.verify_private_space import evaluate_space_policy


def test_private_free_space_policy_accepts_runtime_only_secret_names():
    report = evaluate_space_policy(
        private=True,
        stage="RUNNING",
        current_hardware="cpu-basic",
        requested_hardware="cpu-basic",
        secret_names={"QDRANT_API_KEY", "SESSION_SIGNING_SECRET"},
        variable_names={"QDRANT_URL", "QDRANT_COLLECTION_BASE", "PUBLIC_DEMO_BYOK"},
        collection_base="labor_laws_20260830_3ec5ade",
    )
    assert report["passed"] is True
    assert set(report["secret_names"]) == {"QDRANT_API_KEY", "SESSION_SIGNING_SECRET"}


def test_private_free_space_policy_rejects_owner_model_keys_and_paid_hardware():
    report = evaluate_space_policy(
        private=True,
        stage="RUNNING",
        current_hardware="t4-small",
        requested_hardware="t4-small",
        secret_names={"QDRANT_API_KEY", "GEMINI_API_KEY"},
        variable_names={"QDRANT_COLLECTION_BASE"},
        collection_base="labor_laws_20260830_3ec5ade",
    )
    assert report["passed"] is False
    assert {item["code"] for item in report["violations"]} == {"paid_hardware", "owner_model_key"}
```

- [ ] **Step 2: Run the focused test and confirm the script is missing**

Run: `uv run pytest tests/test_private_space_verification.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.verify_private_space'`.

- [ ] **Step 3: Implement pure evaluation and the authenticated CLI**

Allow only secret names `QDRANT_API_KEY` and `SESSION_SIGNING_SECRET`. Reject any secret name containing `OPENAI`, `GEMINI`, `GOOGLE`, `ANTHROPIC`, or `PROVIDER`. Require private visibility, current/requested `cpu-basic`, and collection base `labor_laws_20260830_3ec5ade`. Accept stages `RUNNING`, `BUILDING`, and `SLEEPING`, but mark only `RUNNING` ready. The CLI accepts `--repo-id` and `--json`, returns `0` on policy pass and `1` on policy failure, and serializes names/status only.

- [ ] **Step 4: Prove redaction and failure handling**

Tests must pass fake secret objects whose values raise if accessed, prove values are never read, and prove authentication/API errors emit only the exception class plus a generic message. Run:

```powershell
uv run pytest tests/test_private_space_verification.py -q -p no:cacheprovider
uv run ruff check scripts/verify_private_space.py tests/test_private_space_verification.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Write and implement the read-only Qdrant reader verifier**

Using `httpx.MockTransport`, test that `verify_reader(client, candidate_base, legacy_base, expected_counts)` reads only:

- `GET /collections/{candidate_base}_fixed` → `481` points;
- `GET /collections/{candidate_base}_structure` → `884` points;
- `GET /collections/{legacy_base}_fixed` and `_structure` → access denied.

The implementation must never issue `PUT`, `POST`, `PATCH`, or `DELETE`. Its authenticated CLI reads `QDRANT_URL` and `QDRANT_API_KEY` from the environment, accepts candidate/legacy bases and counts, emits only names/counts/booleans, and converts HTTP errors to redacted status/class output. Application-level write denial remains proved by `tests/test_vector_store.py`; do not probe a live remote write.

- [ ] **Step 6: Run both policy test suites**

```powershell
uv run pytest tests/test_private_space_verification.py tests/test_qdrant_reader_verification.py tests/test_vector_store.py -q -p no:cacheprovider
uv run ruff check scripts/verify_private_space.py scripts/verify_qdrant_reader.py tests/test_private_space_verification.py tests/test_qdrant_reader_verification.py
```

Expected: all commands exit 0 and the fake transport proves that no write method can be emitted.

- [ ] **Step 7: Document the read-only preflights**

Add:

```powershell
uv run python scripts/verify_private_space.py --repo-id steven0226/tw-labor-law-rag-demo --json
uv run python scripts/verify_qdrant_reader.py --candidate-base labor_laws_20260830_3ec5ade --legacy-base labor_laws --fixed-count 481 --structure-count 884 --json
```

State that the commands read metadata/collection counts only, do not reveal values, do not restart or resize the Space, and do not change Qdrant.

- [ ] **Step 8: Commit Task 2**

```powershell
git add scripts/verify_private_space.py tests/test_private_space_verification.py scripts/verify_qdrant_reader.py tests/test_qdrant_reader_verification.py docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md release/public-files.txt
git commit -m "test: enforce private free Space policy"
```

---

### Task 3: Complete local release candidate gate

**Files:**
- No planned product changes; fix only reproducible failures attributable to v0.3.5.

**Interfaces:**
- Consumes: the complete branch after the portfolio/UI and evidence plans.
- Produces: one clean release-candidate commit and recorded local evidence.

- [ ] **Step 1: Run static, security, test, verifier, and build gates**

```powershell
uv lock --check
uv run ruff check .
uv run bandit -r src scripts -ll
$env:PYTHONUTF8='1'
uv run pip-audit --local
uv run pytest -q -p no:cacheprovider
uv run python scripts/verify_release.py
uv run pytest tests/test_packaging.py -q -p no:cacheprovider
uv run python -W error::UserWarning -c "import sys; sys.path.insert(0, 'src'); import rag.api.main; print('FastAPI import: ok')"
uv run python -W error::UserWarning scripts/ask.py --help
$artifactDir=Join-Path ([System.IO.Path]::GetTempPath()) ("tw-labor-law-rag-v035-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $artifactDir | Out-Null
uv build --out-dir $artifactDir
Get-ChildItem $artifactDir
git diff --check
git status --short
```

Expected: every gate exits 0, all 578 baseline tests plus new v0.3.5 tests pass, both source distribution artifacts exist only in the temporary directory, and the tracked worktree is clean.

- [ ] **Step 2: Run the authenticated, read-only Space preflight**

Run: `uv run python scripts/verify_private_space.py --repo-id steven0226/tw-labor-law-rag-demo --json`

Expected: exit 0; private true, `cpu-basic`, no owner model-key names, and base `labor_laws_20260830_3ec5ade`.

- [ ] **Step 3: Run the authenticated, read-only Qdrant preflight**

Run: `uv run python scripts/verify_qdrant_reader.py --candidate-base labor_laws_20260830_3ec5ade --legacy-base labor_laws --fixed-count 481 --structure-count 884 --json`

Expected: exit 0; candidate counts match, both old-collection reads are denied, no write method is sent, and output contains no URL/key.

- [ ] **Step 4: Record the application candidate SHA**

```powershell
$candidateSha=(git rev-parse HEAD).Trim()
git show --no-patch --format=fuller $candidateSha
git status --short
```

Expected: a 40-character SHA and clean worktree. Save it for Task 4; do not tag yet.

---

### Task 4: Deploy the exact private free-Space candidate and record receipt

**Files:**
- Create: `docs/release/V035_DEPLOYMENT_RECEIPT.md`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `tests/test_release_verification.py`

**Interfaces:**
- Consumes: Task 3 candidate SHA and the existing authenticated private Space.
- Produces: one remote Space revision plus a non-secret receipt bound into the release verifier.

- [ ] **Step 1: Export only the committed public candidate**

Create a temporary directory with `git archive $candidateSha`, then add no untracked or ignored files. Confirm `.env`, key downloads, `.git`, caches, run artifacts, and local absolute paths are absent. Compare the exported file list exactly with `release/public-files.txt`; abort on any missing or extra path.

- [ ] **Step 2: Upload to the existing private Space without changing hardware**

Use authenticated `HfApi.upload_folder(repo_id="steven0226/tw-labor-law-rag-demo", repo_type="space", folder_path=<export>, commit_message="deploy v0.3.5 candidate <short-sha>")`. Do not call `request_space_hardware`, `add_space_secret`, or `add_space_variable`. Capture only the returned Space commit OID/URL metadata; do not print authentication tokens.

- [ ] **Step 3: Wait for ready state with bounded read-only polling**

Poll `get_space_runtime` at 15-second intervals for at most 10 minutes. Accept only `RUNNING` on `cpu-basic`; abort on an error stage or any hardware mismatch. Run the policy preflight again after readiness.

- [ ] **Step 4: Run free, non-provider acceptance**

Verify `/models` lists Gemini/OpenAI BYOK with fallback disabled, `/session` issues a bounded session, unauthenticated `/query` is rejected, the private app page loads, and both redacted Space/Qdrant preflights still pass. Do not submit a real Gemini/OpenAI query and do not expose the private Space URL in the receipt.

- [ ] **Step 5: Write the deployment receipt and failing binding test**

The receipt contains: candidate source SHA, Space revision SHA, date `2026-08-30`, `private`, `cpu-basic`, one replica, no persistent storage, candidate base `labor_laws_20260830_3ec5ade`, expected point counts `481` fixed and `884` structure, read-only preflight pass, BYOK/fallback policy pass, and no-provider acceptance pass. It must not contain endpoint, URL, email, key suffix, secret, session token, provider payload, or local path.

Add a release-verifier test that rejects changed source/revision SHA, paid hardware, public visibility, owner-key wording, wrong collection base/counts, or a receipt not hashed in the manifest.

- [ ] **Step 6: Bind the receipt and rerun the release gate**

```powershell
uv run pytest tests/test_release_verification.py tests/test_private_space_verification.py tests/test_qdrant_reader_verification.py tests/test_vector_store.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: all commands exit 0 and the verifier reports one v0.3.5 deployment receipt.

- [ ] **Step 7: Commit Task 4**

```powershell
git add docs/release/V035_DEPLOYMENT_RECEIPT.md release/manifest.json release/public-files.txt tests/test_release_verification.py
git commit -m "docs: record v0.3.5 private deployment"
```

---

### Task 5: Pull request, CI, immutable tag, and GitHub Release

**Files:**
- No additional product files expected.

**Interfaces:**
- Consumes: clean v0.3.5 branch, full local gate, and private deployment receipt.
- Produces: merged public GitHub history, immutable `v0.3.5` tag, and GitHub Release.

- [ ] **Step 1: Rebase on current main and rerun decisive gates**

```powershell
git fetch origin main --tags
git rebase origin/main
uv run ruff check .
uv run pytest -q -p no:cacheprovider
uv run python scripts/verify_release.py
git status --short
```

Expected: rebase succeeds, all gates exit 0, and the worktree is clean. If rebase changes the deployed application tree, repeat Task 4 before continuing; documentation-only receipt conflict resolution does not require provider calls.

- [ ] **Step 2: Prove v0.3.5 is unused**

```powershell
if (git tag -l v0.3.5) { throw 'local tag v0.3.5 already exists' }
if (git ls-remote --tags origin refs/tags/v0.3.5) { throw 'remote tag v0.3.5 already exists' }
gh release view v0.3.5 2>$null; if ($LASTEXITCODE -eq 0) { throw 'GitHub Release v0.3.5 already exists' }
```

Expected: all three checks confirm the version is unused.

- [ ] **Step 3: Push branch, open PR, and wait for CI**

```powershell
git push -u origin codex/v035-portfolio-readiness
gh pr create --base main --head codex/v035-portfolio-readiness --title "release: v0.3.5 portfolio readiness" --body-file docs/release/V035_RELEASE_NOTES.md
$pr=(gh pr view --json number --jq .number).Trim()
gh pr checks $pr --watch --fail-fast
```

Expected: PR is open, every required check passes, and no unreviewed workflow is skipped.

- [ ] **Step 4: Rebase-merge and verify the main-branch tree**

```powershell
gh pr merge $pr --rebase
git fetch origin main
$releaseSha=(git rev-parse origin/main).Trim()
git diff --exit-code HEAD origin/main
git show --no-patch --format=fuller $releaseSha
```

Expected: PR is merged and `origin/main` contains the intended v0.3.5 application tree. If the diff command reports a product-file difference, stop before tagging and reconcile it.

- [ ] **Step 5: Wait for the exact main-branch CI run**

Find the workflow run whose `headSha` equals `$releaseSha` using `gh run list --branch main --json databaseId,headSha,status,conclusion`, then run `gh run watch <databaseId> --exit-status`.

Expected: the exact merged SHA passes CI.

- [ ] **Step 6: Create and publish the immutable release**

```powershell
git tag -a v0.3.5 $releaseSha -m "v0.3.5 portfolio readiness"
git push origin v0.3.5
gh release create v0.3.5 --verify-tag --title "v0.3.5 — Portfolio readiness" --notes-file docs/release/V035_RELEASE_NOTES.md
```

Expected: tag and release succeed without replacing any prior tag.

- [ ] **Step 7: Post-release verification**

```powershell
git fetch origin main --tags
git rev-parse refs/tags/v0.3.5^{}
gh release view v0.3.5 --json isDraft,isPrerelease,tagName,targetCommitish,url
gh pr view $pr --json state,mergeCommit,url
```

Expected: tag dereferences to `$releaseSha`, release is neither draft nor prerelease, PR state is merged, and the local worktree remains clean.
