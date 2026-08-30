# Evaluation diagnostics

This directory contains durable, content-free evidence for evaluation runs that
did not satisfy their release gates. These files are diagnostics, not official
release artifacts, and are intentionally excluded from
`release/public-files.txt`.

`severance_refusal_policy_v0.3.6_no_go.json` is the immutable audit record for
the superseded route-threshold design. The pivot runner never overwrites it;
its original bytes remain bound to commit `9890c78538176c5338f6a31232a615f8d970fdd2`.

A failed retrieval-coverage pivot is written instead to
`severance_retrieval_pivot_v0.3.6_no_go.json`. Its schema is `1.3`, records the
fixed production threshold `0.03` separately from the evaluation-only
`route_ablation`, and is a NO-GO unless that ablation's highest passing
candidate is also `0.03`. It retains raw unrounded content-free target,
stress, and formal evidence; exact route checks; first-stage and reranker call
counts; CPU/FP32 and local-only model provenance; `rrf_k`; semantic-view and
merge-policy bindings; exact revision/environment bindings; and zero-provider
counters. The revision binding covers every tracked Python file plus the
declared configuration/data inputs and must match the recorded Git tree,
current `HEAD`, index metadata, and checkout bytes. The environment binding is
created before project imports by the committed `python -I -S` bootstrap using
an explicit environment outside the repository; it records only privacy-safe
interpreter/platform, frozen no-development lock selection, and exact installed
distribution facts.

The pivot diagnostic can be replayed with
`rag.severance_refusal_policy.replay_no_go_evidence` without loading retrieval
models or rerunning the 130 queries. A NO-GO diagnostic never authorizes an
official artifact or any production-threshold change. Authoritative Task 7
replay first passes the diagnostic to
`scripts/v036_authoritative_bootstrap.py --mode verify-artifact` under the same
explicit external environment; the verifier is read-only and rejects source,
environment, artifact, or inventory drift before importing project code.
