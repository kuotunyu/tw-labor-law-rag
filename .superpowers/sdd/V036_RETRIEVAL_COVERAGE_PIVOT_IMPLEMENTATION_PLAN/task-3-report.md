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
- Offline cached candidate/ranking fixtures for qids `010`, `014`, and `015`
  record candidate-pool and final ranks, demonstrate both required
  authorities in Top 5, assert two calls of at most 20 pairs, and verify that
  neither qids nor article identifiers enter either reranker query.
