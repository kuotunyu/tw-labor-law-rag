# Task 5 report — authoritative binding redesign

Redesign base commit:
`376fd4e9d91a0e3e970a65ef46874b09bb3fd883`.

This is the replacement Task 5 design required by the revision-binding
amendment. It deletes the superseded AST/symtable import-closure,
dynamic-execution, and control-flow analyzer instead of extending it with a
sixth review fix.

## Root cause and RED evidence

Python execution dependencies cannot be proved complete by the former small
static analyzer. Aliases, namespace access, decorators, class bodies, lambdas,
and point-order effects repeatedly exposed new blind spots. The redesign uses
an exact conservative Git and environment boundary whose completeness does not
depend on Python control-flow inference.

The implementation followed strict RED/GREEN steps:

- the first bootstrap contract test failed because the committed stdlib-only
  entry point did not exist;
- record-mode environment tests failed until isolated `-I -S`, explicit
  external-environment, `pyvenv.cfg`, approved-site, lock, marker, and exact
  inventory validation existed;
- runner provenance and missing-attestation tests failed until calibration
  required the trusted revision/environment binding before preflight;
- the legacy release wrapper test failed until v0.3.6 dispatch was deferred to
  the authoritative bootstrap before importing project code;
- interpreter compatibility tests reported `2 failed` before both
  `.python-version` and `uv.lock` `requires-python` were enforced; and
- a declared-input mode-only commit reported `1 failed` before current `HEAD`
  and index metadata comparisons covered declared inputs as well as Python and
  fixed configuration files.

Every failure sentinel asserted that project/heavy imports, cache/model/index/
provider construction, and output writes/deletes remained at zero.

## Implementation

- `scripts/v036_authoritative_bootstrap.py` imports only the standard library
  and is invoked by the explicit external interpreter under `python -I -S`.
- The revision binding is derived from NUL-delimited full Git tree and index
  metadata. It binds every tracked path whose suffix case-folds to `.py`, plus
  `.python-version`, `Dockerfile`, `pyproject.toml`, `uv.lock`,
  `legal_terms.txt`, and the four declared corpus/target/stress/formal inputs.
- Replay rejects recorded-tree, current-`HEAD`, index, mode/type/OID, checkout
  byte/SHA, add/remove/rename, sparse/missing, alias, path-escape, duplicate,
  case-fold collision, extra/missing binding, dirty-tree, and inconsistent
  declared-input state.
- Exact `src`, `eval`, and `scripts` roots are scanned before imports for
  ignored or untracked Python case variants, bytecode, extension modules,
  `.pth`, importable archives, aliases, special files, `__pycache__`, and cache
  trees.
- The environment contract rejects missing isolation/no-site flags,
  `PYTHONPATH`/home/user-site overrides, repository-contained environments,
  executable/`pyvenv.cfg` relationship drift, system sites, unapproved
  pre-bootstrap paths or zip roots, and interpreter/version/ABI/platform drift.
  `.pth` files are never processed.
- The frozen no-development lock graph is selected against the exact recorded
  marker environment. Installed distributions are enumerated only with
  `importlib.metadata.distributions(path=approved_sites)` and must match the
  selected lock by duplicate-free PEP 503 normalized name and exact version.
- The published environment binding contains no absolute environment path and
  no raw `pyvenv.cfg` content. The trusted boundary remains the OS,
  interpreter, bootstrap/launcher, and approved installation; installed wheel
  bytes are not claimed.
- Only after both bindings pass does the bootstrap append the verified
  `src`/`eval`/`scripts` and approved site roots, import calibration or replay
  code, and pass an exact trusted-runtime attestation.
- Schema `1.3` provenance and model-free replay now validate
  `revision_binding` and `environment_binding`; the release verifier requires
  exact equality with the bootstrap's freshly verified runtime.
- The legacy release command remains valid for already-published releases but
  fails closed with the authoritative v0.3.6 command once that artifact
  exists. Task 7 verification is documented as read-only.

## GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_v036_authoritative_bootstrap.py tests\test_severance_refusal_policy.py tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
293 passed, 3 skipped, 2 deselected in 215.25s

.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
910 passed, 3 skipped, 2 deselected in 516.89s

.venv\Scripts\ruff.exe check .
All checks passed!

git diff --check
```

The two deselections are exactly the Task 7-owned public-file allowlist cases.
Task 5 does not change `release/public-files.txt`.

The immutable historical NO-GO working copy and its `HEAD` blob both equal Git
blob `2cdb13b36d98b5ebfbfcd2cec877e571f3ab2dd4`. No official or pivot diagnostic
artifact was generated, rewritten, deleted, or normalized. `progress.md` is
unchanged.

## Task 6 / Task 7 handoff

- Task 6 creates and synchronizes the real dedicated environment outside the
  repository, offline/frozen/no-dev, then uses this bootstrap for the fresh
  committed CPU/FP32 acceptance. Task 5 performed no environment sync,
  inference, model/provider construction, download, acceptance, export,
  deployment, or secret access.
- Task 7 reuses the committed bootstrap and replay implementation read-only.
  Any Python/test change returns the process to Task 5 and invalidates the
  Task 6 artifact; Task 7 owns public packaging and the two allowlist tests.

## Binding-redesign review-fix round 1

### Root causes and RED evidence

- The bootstrap attested four exact declared inputs but passed its calibration
  remainder directly to the runner. Four exact/equals-form overrides reported
  `4 failed`; all reached the stub project import and runner invocation. A
  mutation check then exposed downstream `argparse` long-option abbreviation:
  the four exact cases passed while `--data`, `--stress`, `--formal`, and
  `--snap` reported `4 failed`.
- Full NUL-delimited Git records were parsed, but gitlink type/mode validation
  occurred only while materializing selected paths. Synthetic tree/index
  records, a real offline local submodule, and a staged nonselected gitlink
  reported `4 failed`; the staged case reached only the generic dirty-tree
  fallback.
- Approved site paths were accepted with `Path.is_dir()`, which follows
  aliases. A real Windows site junction and the missing portable validator
  seam reported `2 failed, 1 skipped`; a forced alias-detector miss separately
  reported `1 failed` before resolved containment existed.

### Fix

- Authoritative calibration now rejects every exact, equals-form, or
  `argparse`-abbreviated spelling of `--dataset`, `--stress-dataset`,
  `--formal-dataset`, and `--snapshot` before project/environment resolution,
  import-path activation, offline preflight, work-directory preparation,
  materialization, model/index/provider construction, or writes. The
  authoritative path therefore always uses the four attested runner defaults.
- Tree parsing rejects any `160000`/`commit` gitlink globally, and index
  parsing rejects every `160000` entry globally, before selected-path
  filtering. This applies to recorded revisions, current `HEAD`, and the full
  current index, including non-Python paths.
- Approved environment-relative site parents and the site directory itself
  are checked with `lstat` plus the portable symlink/Windows-reparse detector.
  The resolved site must also remain below the resolved environment root even
  if the alias detector misses. These checks run before distribution inventory
  enumeration and again before verified roots are added to `sys.path`.
  Symlink/junction external targets are read-only and remain byte-identical.

### GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_v036_authoritative_bootstrap.py -q -p no:cacheprovider -k "rejects_all_input_overrides or calibration_entrypoint_imports_project_only_after_all_bindings_pass"
9 passed, 62 deselected in 17.95s

.venv\Scripts\python.exe -m pytest tests\test_v036_authoritative_bootstrap.py -q -p no:cacheprovider -k "gitlink or submodule"
4 passed, 59 deselected in 4.98s

.venv\Scripts\python.exe -m pytest tests\test_v036_authoritative_bootstrap.py -q -p no:cacheprovider -k "symlinked_parent or windows_junction_without or portable_windows_reparse_seam or resolved_containment_when_alias_seam_misses"
3 passed, 1 skipped, 63 deselected in 1.55s

.venv\Scripts\python.exe -m pytest tests\test_v036_authoritative_bootstrap.py tests\test_severance_refusal_policy.py tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
308 passed, 4 skipped, 2 deselected in 157.75s

.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
925 passed, 4 skipped, 2 deselected in 241.49s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 10.13s

.venv\Scripts\ruff.exe check .
All checks passed!

git diff --check
```

The historical NO-GO working copy and `HEAD` still resolve to Git blob
`2cdb13b36d98b5ebfbfcd2cec877e571f3ab2dd4`; the official and pivot diagnostic
artifacts remain absent, and `progress.md` remains unchanged. No network,
package sync, model/provider construction, acceptance, export, deployment,
secret access, or project artifact deletion occurred.

Ignored test-generated `__pycache__` trees remain under the exact code roots
(`src/rag` packages, `eval`, and `scripts`) and were intentionally not removed
in this review fix. Task 6 must begin from a cache-clean committed checkout;
the authoritative bootstrap correctly treats those ignored trees as a failing
precondition.
