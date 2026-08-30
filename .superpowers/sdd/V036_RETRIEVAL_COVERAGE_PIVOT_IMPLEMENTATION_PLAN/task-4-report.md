# Task 4 report — remove the unsupported production threshold override

Implementation commit: recorded with this report.

## RED evidence

Before changing production code, the focused Task 4 suite was run with the
replacement global-threshold tests:

```text
.venv\\Scripts\\python.exe -m pytest tests\\test_config.py tests\\test_refusal_policy.py tests\\test_answerer.py tests\\test_reliability.py tests\\test_portfolio_demo_regression.py tests\\test_provider_crosscheck.py -q -p no:cacheprovider
```

Result: `41 failed, 84 passed`.

The failures established that the old field was still exposed by `Settings`,
required by the shared decision, read by factory and runner wiring, and used
to choose the exact singleton severance admission threshold.

## Implementation

- `Settings` now exposes only `rerank_score_threshold`; the retired
  severance-specific setting and its validation are removed.
- `Answerer` and the factory no longer accept or wire a route threshold.
- The shared retrieval refusal decision always compares the primary top score
  with the one validated global threshold. It continues to validate and carry
  deterministic route tuples as retrieval evidence, but routes cannot select
  an admission threshold.
- The reliability, portfolio, and provider runners use the shared global
  decision and retain route observations in their retrieval output.
- The provider pre-admission test still proves the refusal decision stops the
  runner before `build_llm`, so neither provider construction nor a provider
  request can occur on that path.
- The dataset README now describes the retained global `0.03` policy.

## GREEN evidence

```text
.venv\\Scripts\\python.exe -m pytest tests\\test_config.py tests\\test_refusal_policy.py tests\\test_answerer.py tests\\test_reliability.py tests\\test_portfolio_demo_regression.py tests\\test_provider_crosscheck.py -q -p no:cacheprovider
125 passed in 1.00s

.venv\\Scripts\\ruff.exe check src\\rag\\config.py src\\rag\\factory.py src\\rag\\generation\\answerer.py src\\rag\\retrieval\\refusal_policy.py eval\\run_reliability_eval.py eval\\run_portfolio_demo_regression.py eval\\run_provider_crosscheck.py tests\\test_config.py tests\\test_refusal_policy.py tests\\test_answerer.py tests\\test_reliability.py tests\\test_portfolio_demo_regression.py tests\\test_provider_crosscheck.py
All checks passed!

git diff --check
```

## Deferred compatibility boundary

`tests/test_severance_refusal_policy.py` was run to identify the integration
boundary with the existing pre-pivot evaluator. It reports `42 failed,
54 passed`, all because its evaluation-only candidate sweep still invokes the
removed shared-policy route threshold. That sweep is explicitly Task 5's
route-ablation work and is intentionally not preserved through production
runtime plumbing in Task 4. Task 5 must replace it with its evaluation-only
ablation decision and update the corresponding evaluator tests.
