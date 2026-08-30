# Task 5 report — pivot the calibration contract

Implementation commit: `a88f53b112f05107ff8e543ff207270d1900e1c6`.

## RED evidence

- The inherited evaluator suite initially reported `42 failed, 54 passed`.
  Every failure reached the removed production
  `severance_comparison_threshold` argument, proving the legacy candidate sweep
  was still coupled to production refusal plumbing.
- New pipeline evidence assertions initially failed twice because
  `RetrievalResult` did not expose first-stage/reranker call counts.
- Schema/replay tests initially failed because the artifact still used schema
  `1.2`, conflated the production and candidate thresholds, and had no strict
  schema `1.3` replay entry point.
- A target-score mutation test exposed a self-consistent replay gap; publishing
  a separate target-evidence binding closed it.
- CPU-only preflight, release-verifier replay, and committed-revision mismatch
  tests each failed before their respective gates were implemented.

## Implementation

- The seven historical thresholds are evaluated only through the named
  `route_ablation` path. Its singleton severance decision is local to the
  evaluator and never reintroduces a production route-threshold parameter.
- Official schema `1.3` records `production_threshold=0.03` separately and
  requires `route_ablation.highest_passing_candidate=0.03` during model-free
  replay and release verification.
- Target, stress, and formal evidence retains full-precision scores and binds
  exact route identity, one first-stage retrieval per query, exact reranker
  calls/pair counts, `rrf_k=60`, pinned local-only model revisions, CPU/FP32,
  semantic-view SHA-256, merge-policy version, PRIMARY-score semantics,
  decision-code hashes, source hashes, a clean committed revision, and zero
  provider counters.
- `severance-policy-027` now has the exact empty-route, positive-hit, strict
  below-`0.03` threshold-refusal contract. Its equality boundary admits
  generation and therefore correctly fails that target contract.
- The release verifier rejects schema, provenance, evidence, route, hash, and
  committed-revision mismatches without constructing retrieval models.
- Failure output uses the distinct
  `eval/diagnostics/severance_retrieval_pivot_v0.3.6_no_go.json` path. The old
  NO-GO file is unchanged and still resolves to Git blob
  `2cdb13b36d98b5ebfbfcd2cec877e571f3ab2dd4` at commit `9890c785`.
- Diagnostic and official READMEs describe the pivot contract and keep the old
  NO-GO result explicitly historical. No official or pivot diagnostic artifact
  was generated in Task 5.

## GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_refusal_policy.py tests\test_answerer.py tests\test_reliability.py tests\test_portfolio_demo_regression.py tests\test_provider_crosscheck.py tests\test_pipeline.py tests\test_severance_refusal_policy.py -q -p no:cacheprovider
322 passed in 6.51s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 8.32s

.venv\Scripts\ruff.exe check src\rag\retrieval\pipeline.py src\rag\severance_refusal_policy.py src\rag\release_verification.py eval\run_severance_refusal_policy.py tests\test_pipeline.py tests\test_severance_refusal_policy.py tests\test_release_verification.py
All checks passed!

git diff --check
```

The combined policy/pipeline/release run reported `292 passed, 2 failed`
before the additional hardening cases were added. Both failures are the
pre-existing public Git allowlist mismatch: the tracked v0.3.6 pivot
plan/reports/evidence are not yet in `release/public-files.txt`.

## Plan-owned handoff

- Task 6 must run the fresh authoritative acceptance on committed code with
  `--device cpu`, offline model caches, and zero providers. It alone may export
  the official artifact or the new pivot NO-GO diagnostic.
- Task 7 owns `release/public-files.txt`, the final public file set, and the two
  currently failing public-tree allowlist tests. Task 5 intentionally did not
  edit release packaging or claim release-verifier completion.

## Review-fix round 1

Review-fix implementation commit:
`95d8f57dba5839df8922f42ab0be08cbeadbbaee`.

### Root causes and RED evidence

- The first schema `1.3` implementation treated six directly named files as
  the complete decision-code closure. It omitted dynamic runner dependencies,
  index/retrieval/fusion code, configuration and factory wiring, model/device
  resolution, reliability helpers, the release wrapper, and the runtime lock.
  Five representative post-run mutation cases plus a missing/untracked case
  failed at the incomplete provenance schema: `6 failed`.
- Six unsafe output cases reached the model preflight because
  `--diagnostics-output` accepted arbitrary historical, official, source,
  relative, traversal, and absolute paths: `6 failed`.
- Four cases that aliased the approved output as a source input also reached
  model preflight: `4 failed`.
- A hardlink alias from the approved pivot filename to a source artifact
  bypassed resolved-path comparison and reached model preflight: `1 failed`.

### Fix

- `decision_code_sha256` now uses an explicit 36-entry manifest covering the
  authoritative configuration, factory/provider-isolation imports, models,
  indexing/ingestion/retrieval/fusion/reranking, evaluator and reliability
  helpers, evidence/replay logic, release verifier and wrapper, `pyproject.toml`,
  `uv.lock`, and the legal-term dictionary.
- Release verification requires every manifest member and source artifact to
  exist, remain Git-tracked, have no staged/unstaged/untracked state, match its
  recorded hash, and match the recorded committed revision. Representative
  threshold-wiring, fusion, device/model, evaluator, and wrapper mutations are
  rejected without model execution.
- The runner resolves the diagnostics output before offline preflight and
  permits only the distinct pivot diagnostic path. It rejects normalized-path
  and filesystem-identity collisions with historical, official, target,
  stress, formal, and corpus-snapshot artifacts before any write or model
  construction.

### GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "nonapproved_diagnostics_output or approved_output_aliased_as_a_source or invalidates_representative_dependency_changes or missing_or_untracked_relevant_files"
16 passed, 117 deselected in 36.96s

.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "nonapproved_diagnostics_output or approved_output_aliased_as_a_source or hardlink_alias_collision"
11 passed, 123 deselected in 0.98s

.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_refusal_policy.py tests\test_answerer.py tests\test_reliability.py tests\test_portfolio_demo_regression.py tests\test_provider_crosscheck.py tests\test_pipeline.py tests\test_severance_refusal_policy.py -q -p no:cacheprovider
338 passed in 54.71s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 9.28s

.venv\Scripts\ruff.exe check src\rag\severance_refusal_policy.py src\rag\release_verification.py eval\run_severance_refusal_policy.py tests\test_severance_refusal_policy.py
All checks passed!

git diff --check
```

The immutable historical NO-GO still matches Git blob
`2cdb13b36d98b5ebfbfcd2cec877e571f3ab2dd4`; no model, provider, network,
acceptance, official-export, or pivot-diagnostic action occurred.

## Review-fix round 2

Review-fix implementation commit:
`765e372c61d5fbe43e3d7c829d5bc6d7661652e2`.

### Root causes and RED evidence

- The 36-entry manifest was broader than its original six-file predecessor,
  but it still had no executable derivation from the authoritative import
  graph. The new recursive static-import check initially failed with exactly
  the six confirmed omissions: `src/rag/portfolio_demo_regression.py` plus the
  `rag`, `generation`, `indexing`, `ingestion`, and `retrieval` package
  `__init__.py` files. A synthetic newly imported local module and an
  unallowlisted dynamic import also failed closed before model execution.
- Resolved-path equality could self-approve a path that traversed a symlink or
  Windows junction. The initial alias-focused run reported `3 failed, 2
  skipped`: traversal back to the approved filename, a working Windows
  junction, and the platform-independent reparse seam all reached the RED
  condition; the two real symlink variants skipped only because Windows denied
  symlink creation privileges.
- Windows `Path` equality case-folded a differently cased lexical path. The
  dedicated case-alias test reached preflight (`1 failed`) until the comparison
  was changed to exact lexical parts.

### Fix

- `DECISION_CODE_PATHS` now has 42 entries, including all six reviewed runtime
  omissions. A recursive AST walker starts from the authoritative evaluator,
  release-verifier module, and release wrapper; follows local imports and
  package-initializer execution semantics; and rejects any discovered local
  file missing from the explicit manifest. Dynamic imports are fail-closed,
  with an intentionally empty, documented exception allowlist.
- Both the evaluator and release verifier execute the closure validation. The
  release verifier still checks every manifest file's recorded hash, tracked
  and clean working-tree state, and committed-revision bytes; a future local
  import cannot silently escape those bindings.
- Diagnostics output is compared lexically before resolution, including exact
  path-part casing. The output and every lexical parent are inspected with
  `lstat`; symlinks and Windows reparse points (including junctions) are
  rejected before preflight, model construction, deletion, or writing.
  Existing hardlink and protected-artifact identity checks remain in force.
- Tests no longer duplicate the production manifest as their sole curated
  oracle. They exercise the real derived closure, a synthetic omitted import,
  dynamic-import rejection, traversal, case aliasing, output and parent
  symlinks, a real Windows junction, a portable reparse seam, and untouched
  arbitrary targets.

### GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "static_local_import_closure"
3 passed, 139 deselected in 1.12s

.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "rejects_nonapproved_diagnostics_output or through_symlink or through_windows_junction or alias_stat_seam or hardlink_alias_collision or aliased_as_a_source"
15 passed, 2 skipped, 126 deselected in 1.20s

.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_refusal_policy.py tests\test_answerer.py tests\test_reliability.py tests\test_portfolio_demo_regression.py tests\test_provider_crosscheck.py tests\test_pipeline.py tests\test_severance_refusal_policy.py -q -p no:cacheprovider
345 passed, 2 skipped in 51.47s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 7.95s

.venv\Scripts\python.exe -m ruff check eval/run_severance_refusal_policy.py src/rag/release_verification.py src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
All checks passed!

git diff --check
```

The historical NO-GO file and the blob at commit `9890c785` both hash to
`2cdb13b36d98b5ebfbfcd2cec877e571f3ab2dd4`. The pivot diagnostic and official
artifact are absent. This review fix performed no model inference/download,
provider call, network access, acceptance run, artifact export, deployment, or
secret access. The Task 6/Task 7 handoff above is unchanged.

## Review-fix round 3

Review-fix implementation commit:
`6a4806dd7048125cc1ef8b25263f2c327bcdaab9`.

### Root cause and RED evidence

The round-2 detector recognized only three call spellings. It did not resolve
bindings, so ordinary renames of `importlib`, `import_module`, `builtins`, or
`__import__`, assigned callable aliases, direct or nested `getattr`, and
`eval`/`exec`/`compile` indirection bypassed the fail-closed policy. It also
rejected clearly user-defined functions named `import_module` or `__import__`.

The first table-driven RED run reported `20 failed, 2 passed`. The failures
covered all four reviewed import forms, renamed and assigned aliases, direct
and nested `getattr`, `__builtins__`, all three dynamic-code builtins, and the
two user-defined-name false positives. A follow-up traversal audit produced
`2 failed, 50 passed`, exposing a parameter-annotation bypass and lambda-local
shadowing false positive before those branches were implemented.

### Fix

- A scoped, binding-aware AST visitor now resolves direct and renamed imports
  from `importlib` and `builtins`, assigned callable aliases, builtin `getattr`
  aliases, literal attribute/subscript access, and nested `__call__` lookup.
- Calls resolving to `__import__`, `import_module`, `eval`, `exec`, or
  `compile` fail closed. The visitor also inspects evaluated defaults,
  decorators, assignment/function annotations, and lambda/function/class
  scopes while allowing clearly user-bound similarly named functions and
  unrelated object attributes.
- The importer-wide allowlist was removed. There is deliberately no dynamic
  exception path; any future exception must resolve and bind an exact target
  rather than exempt an entire decision-relevant file.
- The authoritative evaluator and release verifier continue to invoke this
  closure validator before accepting or replaying evidence. The current
  authoritative import graph remains valid without an exception.

### GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "static_local_import_closure"
30 passed, 140 deselected in 1.63s

.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_refusal_policy.py tests\test_answerer.py tests\test_reliability.py tests\test_portfolio_demo_regression.py tests\test_provider_crosscheck.py tests\test_pipeline.py tests\test_severance_refusal_policy.py -q -p no:cacheprovider
372 passed, 2 skipped in 52.98s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 7.95s

.venv\Scripts\python.exe -m ruff check eval/run_severance_refusal_policy.py src/rag/release_verification.py src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
All checks passed!

git diff --check
```

No model inference/download, provider call, network access, acceptance run,
artifact export, deployment, or secret access occurred. The historical and
artifact-integrity checks and the Task 6/Task 7 handoff remain unchanged.

## Review-fix round 4

Review-fix implementation commit:
`b8a87be91f4bbb72b34cec2881b11b62e612f5a3`.

### Root cause and RED evidence

The round-3 call-taint visitor still depended on source order and only rejected
a dangerous binding when it could trace a later call. It therefore lost taint
across forward module/enclosing bindings, class-to-method lookup, branch and
try joins, conditional expressions, and destructuring. It also treated
`del eval` and a value-less `eval: object` annotation as safe runtime bindings,
while rejecting two harmless user functions defined after the nested function
that referenced them.

The exact review table plus acquisition-without-call cases reported `17
failed`. All nine reviewed unsafe programs escaped, both reviewed harmless
forward-bound programs were rejected, and six unused dynamic-API acquisition
forms were accepted. A further self-audit added and RED-proved direct access to
builtin `eval` before a later module definition (`1 failed, 6 passed`).

### Simplified policy

- Decision code now has a no-acquisition/no-access contract instead of a
  call-only taint contract. Importing `importlib`, importing `builtins`, or
  importing their dynamic APIs is rejected immediately, even when the acquired
  object is never called. References resolving to builtin `eval`, `exec`,
  `compile`, or `__import__` are likewise rejected at access time.
- Builtin `getattr` is allowed only for a literal attribute that is proven not
  to acquire a dynamic API; non-literal acquisition and access through
  `__builtins__` fail closed. Clearly user-bound functions and object
  attributes with similar names remain allowed.
- Python's `symtable` supplies module/function/class/free/global resolution. A
  separate runtime-binding pass joins branch, loop, try/handler/finally, and
  with-statement outcomes; handles destructuring and conditional expressions;
  models `Delete` as unbound; and does not treat a value-less annotation as a
  runtime binding. Method lookup skips the class namespace, as Python does.
- Same-scope access uses bindings available before the reference. Nested
  global/free access uses the enclosing scope's final binding set, which admits
  the two exact harmless forward definitions without admitting direct builtin
  access before a later definition.
- There is still no importer-wide allowlist. Errors name the source line and
  the unproven or forbidden acquisition.

The authoritative evaluator and release verifier continue to run this public
closure validator before evidence acceptance/replay.

### GREEN evidence

```text
.venv\Scripts\python.exe -m pytest tests\test_severance_refusal_policy.py -q -p no:cacheprovider -k "static_local_import_closure"
48 passed, 140 deselected in 1.42s

.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_refusal_policy.py tests\test_answerer.py tests\test_reliability.py tests\test_portfolio_demo_regression.py tests\test_provider_crosscheck.py tests\test_pipeline.py tests\test_severance_refusal_policy.py -q -p no:cacheprovider
390 passed, 2 skipped in 72.62s

.venv\Scripts\python.exe -m pytest tests\test_release_verification.py -q -p no:cacheprovider -k "not test_release_verifier_recomputes_committed_evidence and not test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions"
105 passed, 2 deselected in 10.76s

.venv\Scripts\python.exe -m ruff check eval/run_severance_refusal_policy.py src/rag/release_verification.py src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
All checks passed!

git diff --check
```

No model inference/download, provider call, network access, acceptance run,
artifact export, deployment, or secret access occurred. The historical and
artifact-integrity checks and the Task 6/Task 7 handoff remain unchanged.
