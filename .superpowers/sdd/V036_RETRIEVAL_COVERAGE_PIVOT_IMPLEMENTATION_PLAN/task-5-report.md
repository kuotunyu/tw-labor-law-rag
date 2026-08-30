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
