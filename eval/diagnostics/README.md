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
official artifact or any production-threshold change. The bootstrap's public
`verify-artifact` mode is reserved for an accepted official artifact; a pivot
diagnostic is instead retained as model-free failure evidence and replayed by
the policy function under the same verified source/environment bindings.

## v0.3.6 pivot calibration result

The one authorized CPU/FP32 calibration on candidate `19bda93` completed all
30 target, 60 stress, and 40 formal observations and produced
`severance_retrieval_pivot_v0.3.6_no_go.json`. It did not produce an official
artifact.

The retrieval pivot fixed the earlier positive-source misses: all 15 positive
cases used the exact singleton route, retrieved both required authorities in
Top 5, and remained generation-eligible. Stress and formal gates also passed:
stress retained 17/20 direct unanswerable refusals with 0/40 answerable false
refusals; formal reached Hit@5 `1.0`, MRR@10 `0.9388888888888888`, and 0/30
answerable false refusals.

The remaining target failure is `severance-policy-023`. Its dataset contract
requires `no_hits`, while the fresh pipeline returned positive candidates and
correctly stopped at the unchanged `0.03` threshold. The target result is
therefore 29/30, every route-ablation candidate fails the target gate, and
`highest_passing_candidate` remains `null`. This evidence does not authorize
weakening the threshold or changing production behavior.
