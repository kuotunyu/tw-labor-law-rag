# v0.3.6 Windows PyWin32 Runtime Roots Design

Status: approved under the project controller's standing recommendation authority on 2026-08-31.

## Problem and evidence

The first authoritative calibration for candidate `119a8b5` passed the exact
Git and external-environment bindings, then stopped before indexing or the
first target query. Local Qdrant asked `portalocker` for its Windows lock
implementation, which could not import `pywintypes` under isolated Python
`-I -S`.

The frozen environment correctly contains `pywin32`. Its normal installer
layout relies on a `.pth` file to add `win32`, `win32/lib`, and `pythonwin`, and
to execute a bootstrap import. The authoritative launcher deliberately neither
processes nor executes `.pth` files. Read-only isolated probes established the
minimal dependency boundary:

- approved `site-packages` plus `win32/lib` fails because
  `_win32sysloader` is unavailable;
- approved `site-packages` plus `win32` fails because `pywintypes` is
  unavailable;
- approved `site-packages` plus both `win32` and `win32/lib` imports
  `pywintypes` and constructs `portalocker`'s `Win32Locker`;
- `pythonwin` is not required by the acceptance path.

This was a pre-query runtime blocker. It produced no target, stress, or formal
observation, no provider call, and no official or diagnostic artifact.

## Options considered

### 1. Explicit package-conditioned runtime roots (selected)

On Windows, when the exact frozen lock and installed inventory select
`pywin32`, approve only `Lib/site-packages/win32` and
`Lib/site-packages/win32/lib`. Record their environment-relative POSIX paths
in the environment receipt and validate them like every other approved import
root.

This preserves the no-`.pth` boundary, uses the installed package's real
layout, and grants only the two roots demonstrated necessary.

### 2. Parse path entries from `pywin32.pth` (rejected)

Even a path-only parser would make an executable configuration format part of
the trusted path-discovery mechanism. The installed file also contains an
`import` statement, creating additional parsing and rejection rules without a
release benefit.

### 3. Change storage or rewrite the installed environment (rejected)

Using an in-memory Qdrant path would make acceptance differ from the deployed
local-store behavior. Copying or linking PyWin32 modules into the approved site
root would mutate a frozen environment after synchronization and weaken its
inventory claim.

## Runtime-root contract

The environment binding gains an exact `runtime_import_layout` list.

- On Windows with lock-selected and installed `pywin32`, it is exactly:
  - `Lib/site-packages/win32`
  - `Lib/site-packages/win32/lib`
- On non-Windows platforms, or an environment whose exact selected inventory
  does not contain `pywin32`, it is empty.
- `pythonwin`, `pywin32_system32`, arbitrary distribution directories, user
  sites, archives, and paths discovered from `.pth` are not import roots.
  `pywin32_system32` remains importable as a package from the already approved
  site root; it is not added separately.

Each runtime root and every parent below the validated environment root must:

1. exist as a real directory;
2. pass `lstat` symlink/junction/reparse rejection;
3. resolve beneath the same resolved external environment;
4. equal the package- and platform-conditioned expected relative layout;
5. be revalidated immediately before `sys.path` activation.

The bootstrap still validates the exact lock selection and installed
distribution inventory before importing third-party or project code. It then
adds the verified project roots, approved site root, and the exact verified
runtime roots directly. It never imports `site`, executes `.pth` statements,
or derives paths from `.pth` contents.

Immediately before activation, the bootstrap reads the already verified
`lock_selection.markers` and selected package inventory from the environment
binding, re-derives the package- and platform-conditioned expected layout,
requires exact type, order, spelling, and canonical equality with
`runtime_import_layout`, and only then performs the second `lstat` and resolved
containment validation. `sys.path` is unchanged until every check succeeds.

The receipt contains only relative paths. Exact receipt equality on replay
binds the runtime-root layout without publishing an absolute environment
location.

## Failure behavior and tests

Strict TDD covers both the original failure and the trust boundary.

- RED must reproduce isolated `Win32Locker` construction failing with the
  approved site root alone.
- GREEN must prove that the exact two-root layout succeeds without processing
  `.pth` or adding `pythonwin`.
- The real RED/GREEN evidence uses an internal test subprocess harness that
  launches the authoritative bootstrap with `-B -I -S`, imports the bootstrap
  as a private module, runs its pre-import validation and direct activation
  helpers, and then constructs `portalocker.portalocker.Win32Locker`. It adds
  no public CLI mode and never enters the evaluator, verifier, calibration, or
  artifact path. The already preserved Task 6 pre-fix traceback is the real
  RED evidence; the same model-free harness must succeed after the fix.
- Unit tests reject a missing root, extra root, wrong order or spelling,
  non-Windows roots, roots without exact PyWin32 inventory, symlink, Windows
  junction/reparse point, path escape, special file, and receipt mismatch.
- A malicious `.pth` side effect marker must remain absent.
- Every validation failure occurs before project/cache/index/model/provider
  construction and before artifact writes.
- A model-free external-environment smoke uses the same private `-B -I -S`
  subprocess harness to construct and close a local Qdrant store before the
  next formal calibration. It uses an explicit temporary directory outside
  the repository and no server, key, provider, query, model, network, public
  CLI mode, calibration, or artifact path.

Existing schema `1.3`, exact revision binding, raw checkout-byte checks,
privacy gates, CPU/FP32 requirement, model pins, and release metrics remain
unchanged.

## Acceptance authorization

Candidate `119a8b5` consumed one calibration process launch but reached zero
index and zero query observations. The runtime-root fix changes the trusted
bootstrap and therefore invalidates that candidate under the existing
rerun-on-decision-change rule.

After the amended Task 5 implementation and independent review, Task 6 may
create a new clean candidate and perform one authoritative CPU/FP32 30/60/40
calibration for that new candidate. The partial external run is audit-only and
must not be reused as an index, cache, result, or artifact input.

## Out of scope

- generic `.pth` support;
- accepting arbitrary distribution-provided import roots;
- changing Qdrant persistence mode;
- changing the frozen dependency set or installed files;
- changing retrieval, reranking, threshold, dataset, corpus, or release gates;
- any provider, network, paid compute, or secret access.
