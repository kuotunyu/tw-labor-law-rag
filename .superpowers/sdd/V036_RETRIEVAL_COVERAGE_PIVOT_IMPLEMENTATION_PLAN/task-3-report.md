# Task 3 report — exact-route two-view reranking integration

Implementation commit: this report is included with the implementation commit.

## RED evidence

Added the Task 3 pipeline tests, then ran:

```text
.venv\\Scripts\\python.exe -m pytest tests\\test_pipeline.py -q
```

Result: `5 failed, 71 passed`.

- The exact singleton severance test failed because the pipeline called the
  legacy cut-before-rerank method instead of full-pool `rerank_all()`.
- The empty-candidate test failed because the legacy path called the reranker
  even though there were zero candidate pairs to score.
- The three cached, offline `severance-policy-010`, `-014`, and `-015` tests
  failed because the primary/secondary full-ranking path did not exist.

## GREEN evidence

```text
.venv\\Scripts\\python.exe -m pytest tests\\test_pipeline.py -q
76 passed

.venv\\Scripts\\python.exe -m pytest tests\\test_severance_refusal_policy.py tests\\test_answerer.py -q
123 passed

.venv\\Scripts\\ruff.exe check src\\rag\\retrieval\\pipeline.py tests\\test_pipeline.py
All checks passed!
```

`git diff --check` was clean before commit.

## Files

- `src/rag/retrieval/pipeline.py`
- `tests/test_pipeline.py`
- `tests/fixtures/v036_severance_retrieval_stage_replay.json`

## Behavior locked

- First-stage retrieval remains one call using the existing planned query and
  candidate pool.
- An exact singleton `severance_comparison` plan reranks the full non-empty
  pool twice (primary query then the deterministic old-regime view), merges
  the validated permutations, and only then cuts to final Top 5.
- `RetrievalResult.top_score` captures the primary ranking's first score
  before merge. Secondary scores affect order only.
- Non-exact route shapes retain the legacy one-reranker call. Disabled and
  empty-candidate paths issue zero reranker calls and score zero pairs.
- The provenance-bound offline replay fixtures for qids `010`, `014`, and
  `015` record candidate-pool and final ranks, demonstrate both required
  authorities in Top 5, assert two calls of at most 20 pairs, and verify that
  neither qids nor article identifiers enter either reranker query.

## Review-fix round 1

### Root causes

- The first Task 3 coverage created the same six synthetic chunks and rankings
  for every qid. The `qid` parameter did not change the retrieval evidence, so
  the test could not demonstrate the documented `010`/`014` rank differences.
- `top_k_retrieve` was passed through unchanged and a retriever that returned
  more candidates than requested was not rejected. Either condition could
  score more than the release's twenty-pair ceiling.

### RED evidence

After adding the candidate-cap boundary tests and before modifying production
code, `pytest tests\\test_pipeline.py -q` reported `2 failed, 77 passed`:

- `top_k_retrieve=21` did not fail at pipeline construction.
- A retriever-returned pool of 21 reached the reranker instead of failing
  before any scoring call.

### Replay fixture provenance and scope

`tests/fixtures/v036_severance_retrieval_stage_replay.json` is a checked-in
deterministic control-flow replay fixture, SHA-256
`f3cf753cb90680a083591f7a8a5a783c573dbc5db613ebe06cd9a889fc545b2d`.
It is bound in-test to the retained v0.3.6 NO-GO artifact at source revision
`9890c78538176c5338f6a31232a615f8d970fdd2`, its artifact SHA-256
`ebed566b3778f6674ef55641564861ef3c17bb899197bb40e3ba7f81bb6e010c`,
the current target-dataset SHA-256
`cf9d5ae3a9ae05ccb944b22fb878118670f7272895d6a8d7c383928eaf19b40c`,
the retained corpus hash, and the pinned embedding/reranker revisions.

It contains distinct, opaque per-qid candidate pools and complete primary and
secondary ranking permutations for `010`, `014`, and `015`; tests calculate
and assert each fixture's candidate, primary, secondary, and final stage
ranks. Candidate text is content-free and the test asserts that qids and
article identifiers never enter reranker queries, so the production path is
not source-ID steered.

This fixture is not a fresh model-cache trace, calibration run, or acceptance
artifact. It only protects deterministic pipeline control flow. The fresh
CPU/FP32 acceptance evidence remains Task 6's responsibility.

An attempted local-only extraction from a retained run was stopped after its
cross-encoder worker did not yield a recordable trace in the command window.
It made no tracked-artifact changes, produced no fixture data, and no further
model inference was run for this fix.

### GREEN evidence

- A valid 20-candidate exact route makes one retrieval and two 20-pair
  reranker calls.
- `top_k_retrieve > 20` fails at construction before retrieval.
- A retriever that violates the 20-candidate contract fails after its one
  retrieval but before any reranker call.
