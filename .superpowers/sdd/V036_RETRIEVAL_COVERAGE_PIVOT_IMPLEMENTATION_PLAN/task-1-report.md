# Task 1 report — Lock the pivot contract

Base: `82a9809323e9c9e486ef1b66441bd56c262c5672` on
`codex/v036-route-aware-refusal`.

## RED evidence

Command:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider -k 'exact_outcomes or exact_singleton or 027_requires or 027_rejects'
```

Result: `4 failed, 91 deselected in 1.79s`.

- Dataset case objects had no `expected_outcome` attribute.
- A positive route plus `off_hours_employer_message` incorrectly passed.
- Case `027` emitted neither `expected_outcome` nor an outcome-contract result.

## GREEN evidence

Focused RED tests after the implementation:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider -k 'exact_outcomes or exact_singleton or 027_requires or 027_rejects'
```

Result: `4 passed, 91 deselected in 0.90s`.

Focused policy suite:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
```

Result: `95 passed in 4.21s`.

## Ruff evidence

```powershell
uv run ruff check src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
```

Result: `All checks passed!`.

## Additional RED/GREEN evidence

The initial scorer permitted a nonempty non-severance route for `027`; the
following test was added before tightening that exact contract:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider -k '027_rejects_a_nonempty_route_tuple'
```

RED result: `1 failed, 95 deselected in 1.03s` (`route_contract_passed` was
incorrectly `True`).

GREEN result after the minimal exact-empty-route check:
`1 passed, 95 deselected in 0.84s`.

Final focused module rerun: `96 passed in 4.45s`.

Final Ruff rerun: `All checks passed!`.

## Dataset preservation evidence

Compared with the base dataset revision: 30 base rows and 30 current rows;
order and every non-outcome field (question, case type, answerability, source,
route, and style) are preserved for `30/30` rows. Outcome mapping preserves
`29/30` meanings; only `severance-policy-027` changes from `generation` to
`threshold`. Final exact outcome counts are `generation=28`, `no_hits=1`, and
`threshold=1`.

`027` requires empty routes, a positive hit count, a score below global `0.03`,
and refusal stage `threshold`; a `no_hits` result fails its outcome contract.
The fifteen positives require exactly `("severance_comparison",)`. Collision
rows retain their prior required/prohibited subset behavior.
