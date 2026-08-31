# v0.3.6 Windows minimal child-environment implementation plan

## Goal

Make the private Windows runtime contract preserve the minimum identity and
profile variables required by local PyTorch/Transformers loading without
forwarding ambient secrets or unrelated environment state.

## Task 1: Add the failing contract test

In `tests/test_v036_authoritative_bootstrap.py`:

1. supply deterministic Windows identity/profile variables to the parent;
2. make the child probe call `getpass.getuser()`;
3. require the expected username, exact copied names, exact offline controls,
   accepted runtime locale, zero sentinels, and zero unexpected names;
4. confirm the test fails because the current allowlist omits identity/profile.

## Task 2: Implement the narrow allowlist change

Extend the private Windows process-variable tuple with `USERNAME` and
`USERPROFILE`.  Do not copy `os.environ`, add provider variables, add a public
CLI, or change evaluator/retrieval behavior.

## Task 3: Verify and review

Run:

- the focused RED/GREEN regression;
- the full authoritative-bootstrap tests;
- the bootstrap plus artifact-contract suites;
- Ruff on the touched Python test;
- privacy, scope, Git-cleanliness, and cache-root checks.

Obtain an independent code review.  Commit the exact candidate only when all
gates pass.

## Task 4: Fresh preflight and calibration gate

On the new commit, run a fresh model-free preflight using a new external audit
root and the exact committed allowlist.  Only after a PASS may one new
CPU/FP32, offline, provider-free 30/60/40 calibration launch occur.
