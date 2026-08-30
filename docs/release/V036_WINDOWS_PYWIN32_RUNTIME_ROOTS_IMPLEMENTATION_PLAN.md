# v0.3.6 Windows PyWin32 Runtime Roots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the isolated Windows authoritative bootstrap import the minimum PyWin32 runtime needed by local Qdrant while preserving the frozen-environment, no-`.pth`, privacy, and fail-closed boundaries.

**Architecture:** Derive an exact environment-relative runtime layout from the already verified platform markers and locked inventory, bind it into the privacy-safe receipt, validate it before any third-party import, and revalidate it immediately before direct `sys.path` activation. The only non-empty layout is `Lib/site-packages/win32` plus `Lib/site-packages/win32/lib` on Windows when PyWin32 is selected and installed.

**Tech Stack:** Python 3.12, pytest, Ruff, isolated Python `-B -I -S`, uv frozen synchronization, portalocker, local Qdrant.

## Global constraints

- Never import `site`, `sitecustomize`, or `usercustomize`; never parse or execute `.pth` files.
- Approve exactly two package-conditioned Windows roots and no others. Do not add `pythonwin`, `pywin32_system32`, archives, user sites, or arbitrary distribution roots.
- Reject symlinks, junctions, reparse points, special files, escapes, duplicates, non-canonical spelling, wrong order, missing roots, and receipt mismatches.
- Store only environment-relative POSIX paths in the receipt; never publish an absolute environment path.
- Fail before project/cache/index/model/provider construction and before artifact writes.
- Do not change dependencies, the lock, the corpus, retrieval, reranking, thresholds, providers, public allowlists, or release metrics.
- Implementation and smoke tests use no network, provider, paid compute, secret, calibration, or artifact-generation path.
- Use strict RED-GREEN-REFACTOR TDD and independently review each task before proceeding.

---

## Task 1: Bind and validate the exact runtime layout

**Files:**

- Modify: `scripts/v036_authoritative_bootstrap.py`
- Modify: `tests/test_v036_authoritative_bootstrap.py`
- Update: `.superpowers/sdd/V036_WINDOWS_PYWIN32_RUNTIME_ROOTS_IMPLEMENTATION_PLAN/task-1-report.md`

- [ ] **Step 1: Add RED contract tests**

Add behavior-level tests proving that the expected layout is exactly:

```python
[
    "Lib/site-packages/win32",
    "Lib/site-packages/win32/lib",
]
```

only when Windows markers and the exact selected/installed inventory include PyWin32. Prove that non-Windows and no-PyWin32 inventories produce `[]`.

- [ ] **Step 2: Bind the layout into the environment receipt**

Extend the exact receipt contract with:

```python
"runtime_import_layout": list[str]
```

Update the existing exact-key-set and replay-equality tests. The values must be canonical environment-relative POSIX paths and must never be discovered from `.pth` contents.

- [ ] **Step 3: Implement the minimum derivation helper**

Add a fixed constant and a pure helper equivalent to:

```python
_WINDOWS_PYWIN32_RUNTIME_LAYOUT = (
    PurePosixPath("Lib/site-packages/win32"),
    PurePosixPath("Lib/site-packages/win32/lib"),
)

def _expected_runtime_import_layout(selected, markers) -> tuple[PurePosixPath, ...]:
    ...
```

Derive the result only after the selected lock inventory exactly equals the validated installed inventory.

- [ ] **Step 4: Add RED path-hardening tests**

Test rejection of missing and extra roots, wrong order or spelling, `pythonwin`, duplicate/non-canonical paths, special files, path escapes, symlinks, and Windows junction/reparse points. Every rejection must happen before constructors and artifact writes; preserve existing sentinels that prove this ordering.

- [ ] **Step 5: Implement validation without activation**

Add a helper equivalent to:

```python
def _validated_runtime_import_roots(
    environment_root: Path,
    layout: Sequence[str],
    selected: Sequence[Mapping[str, object]],
    markers: Mapping[str, str],
) -> list[Path]:
    ...
```

First re-derive the expected layout from `selected` and `markers`, then require
exact type, order, spelling, and canonical equality. Validate each component
with `lstat`, reject aliases/reparse points, resolve beneath the same external
environment, and require real directories.

- [ ] **Step 6: Run focused verification**

Run the authoritative-bootstrap unit tests, Ruff on changed Python files, and `git diff --check`. Confirm that dependency and public-surface files did not change.

- [ ] **Step 7: Commit and independently review Task 1**

Commit only the Task 1 implementation/tests/report. The reviewer must verify least privilege, receipt privacy, replay exactness, validation ordering, and the no-`.pth` boundary.

---

## Task 2: Activate verified roots and prove the real Windows lock path

**Files:**

- Modify: `scripts/v036_authoritative_bootstrap.py`
- Modify: `tests/test_v036_authoritative_bootstrap.py`
- Update: `.superpowers/sdd/V036_WINDOWS_PYWIN32_RUNTIME_ROOTS_IMPLEMENTATION_PLAN/task-2-report.md`

- [ ] **Step 1: Add isolated fake and real RED import regressions**

Create a temporary fake environment where one required module is available only from `win32` and another only from `win32/lib`. Include a malicious `.pth` side-effect marker. Launch the real bootstrap with `-B -I -S`; before activation support, the runner must fail and the marker must remain absent.

Also preserve the existing Task 6 pre-fix traceback as the real RED evidence.
Define one internal test subprocess harness that launches Python with
`-B -I -S`, imports the authoritative bootstrap as a private module, runs its
pre-import validation and direct activation helpers, and constructs
`portalocker.portalocker.Win32Locker`. This harness is not a public CLI mode
and must never enter evaluator, verifier, calibration, or artifact code.

- [ ] **Step 2: Revalidate and directly activate the bound roots**

Change activation to receive the verified environment binding. Immediately
before modifying `sys.path`, read
`environment_binding["lock_selection"]["markers"]` and its selected package
inventory, re-derive the expected runtime layout, require exact
type/order/canonical equality with `runtime_import_layout`, then repeat the
`lstat` and containment checks. Add the exact roots directly only after all
checks pass. Continue to avoid `site` and all `.pth` processing.

- [ ] **Step 3: Complete GREEN security coverage**

Prove the isolated runner imports the split fake modules, the `.pth` marker remains absent, `pythonwin` is not on `sys.path`, order is deterministic, replay mismatch is rejected, and all failures occur before project or artifact construction.

- [ ] **Step 4: Run the real model-free Windows smoke**

Using the already frozen external environment and the same private `-B -I -S`
subprocess harness:

1. run offline frozen sync-check and authoritative record validation, then
   load the resulting privacy-safe binding in the private harness;
2. import `pywintypes` through the authoritative bootstrap;
3. construct `portalocker.portalocker.Win32Locker`;
4. construct and close a local Qdrant store in an explicit temporary directory outside the repository.

The smoke must add no public CLI mode and perform zero evaluator/verifier
entry, calibration, model loads, embeddings, reranks, queries, providers,
network calls, and artifact writes. The before/after evidence is the preserved
Task 6 real traceback followed by success from this same model-free runtime
path.

- [ ] **Step 5: Run full scoped verification**

Run the combined Task 5/bootstrap test selection, Ruff, `git diff --check`, clean-code-root checks, old-artifact preservation checks, and public allowlist diff checks.

- [ ] **Step 6: Commit and independently review Task 2**

Commit only the Task 2 implementation/tests/report. The reviewer must confirm the original Qdrant/portalocker failure is reproduced then fixed without broadening the trust boundary.

- [ ] **Step 7: Return to pivot Task 6**

After a clean review, create a new clean candidate and restart the single authorized CPU/FP32 30/60/40 calibration. Do not reuse any index, cache, result, or artifact from the prior zero-query launch. Preserve the old NO-GO artifact unchanged until a new official result passes all gates.

---

## Completion gates

- [ ] Both task reviews have no unresolved P0-P2 findings.
- [ ] Frozen external-environment record validation succeeds.
- [ ] The real model-free Qdrant/portalocker smoke succeeds under `-B -I -S`.
- [ ] No `.pth` file is parsed or executed and no unapproved import root is active.
- [ ] Test, Ruff, diff, privacy, clean-root, old-artifact, and public-surface checks pass.
- [ ] Only then may the new authoritative calibration launch.
