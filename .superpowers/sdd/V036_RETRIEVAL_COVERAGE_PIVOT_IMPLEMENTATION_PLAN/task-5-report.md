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
