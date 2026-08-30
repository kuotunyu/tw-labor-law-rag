# Task 2 report — deterministic multi-view planning and merge primitives

Implementation commit: `104ed8c7f880c8cad149feaef8acee408004ff2e`

## RED evidence

- Added the exact-singleton route test, then ran
  `.venv\\Scripts\\python.exe -m pytest tests/test_pipeline.py -q`:
  `1 failed, 57 passed` because `QueryPlan` did not expose
  `rerank_only_views`.
- Added the pure interleave and full-rerank score-count tests, then ran the
  same focused command: `10 failed, 58 passed` because neither
  `interleave_reranker_rankings` nor `Reranker.rerank_all` existed.
- Mutation checks against weakened ID validation failed as expected for a
  non-string ID (`AttributeError` rather than the required `ValueError`) and
  for a foreign ID (`KeyError` rather than fail-closed validation).

## GREEN evidence

Ran after restoring the final implementation:

```text
.venv\\Scripts\\python.exe -m pytest tests/test_pipeline.py tests/test_embedding_cache.py -q
83 passed

.venv\\Scripts\\ruff.exe check src\\rag\\retrieval\\pipeline.py src\\rag\\retrieval\\reranker.py tests\\test_pipeline.py
All checks passed!
```

`git diff --check` was also clean before the implementation commit.

## Files

- `src/rag/retrieval/pipeline.py`
- `src/rag/retrieval/reranker.py`
- `tests/test_pipeline.py`

## Decisions

- `QueryPlan.rerank_only_views` is a frozen tuple. The approved old-regime
  view is emitted only when routes equal exactly
  `("severance_comparison",)`; collision and unrelated plans receive `()`.
- `Reranker.rerank_all()` returns a complete, stable reranked permutation and
  rejects malformed candidate IDs and score-count mismatches. The legacy
  `rerank()` API delegates to it, then applies the caller's final cut.
- `interleave_reranker_rankings()` validates input, primary, and secondary
  IDs as exact unique permutations, then interleaves each depth primary-first.
  Cross-ranking repeats are deduplicated, while emitted chunks retain their
  primary-query score and may therefore be non-monotonic.

## Caveats

- This task provides planner and reranker primitives only. `RetrievalPipeline.run`
  intentionally still performs its existing one-pass rerank; exact-route
  two-view invocation, primary top-score capture, and final Top-K cutting are
  reserved for Task 3.
- Tests use in-memory chunks and an injected score function only; no model was
  loaded or downloaded, and no network call was made.
