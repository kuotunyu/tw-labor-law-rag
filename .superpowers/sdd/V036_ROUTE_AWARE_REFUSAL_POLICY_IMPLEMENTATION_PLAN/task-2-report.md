# Task 2 report: pure fail-closed retrieval refusal policy

## Implementation

Added `rag.retrieval.refusal_policy` with the exact pure policy contract:

- validates boolean flags, tuple route shape, and all score/threshold inputs;
- rejects non-finite values and values outside `[0, 1]`;
- returns `no_hits` before considering reranker or score;
- bypasses score refusal when reranking is disabled;
- uses the severance threshold only for the exact singleton route tuple;
- uses the global threshold for empty, unknown, duplicate, and multi-route tuples;
- refuses only when `top_score < effective_threshold` (equality passes).

## Files

- `src/rag/retrieval/refusal_policy.py`
- `tests/test_refusal_policy.py`

## Verification

### RED evidence

Before creating the production module, ran:

```powershell
uv run pytest tests/test_refusal_policy.py -q -p no:cacheprovider
```

Result: collection failed with `ModuleNotFoundError: No module named 'rag.retrieval.refusal_policy'`, confirming the test exercised the missing feature.

### GREEN evidence

After implementation, ran:

```powershell
uv run pytest tests/test_refusal_policy.py -q -p no:cacheprovider
```

Result: `27 passed in 0.04s`.

```powershell
uv run ruff check src/rag/retrieval/refusal_policy.py tests/test_refusal_policy.py
```

Result: `All checks passed!`.

Also ran `git diff --check` before commit; it reported no whitespace errors.

## Self-review

The implementation is pure and has no provider, Qdrant, deployment, UI, or runtime integration dependency. Validation occurs before decision branching, route matching is exact and fail-closed, and the decision object exposes the requested `refused` property. Tests cover the complete requested decision table plus invalid score/threshold and input-shape cases.
