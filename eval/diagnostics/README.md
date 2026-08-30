# Evaluation diagnostics

This directory contains durable, content-free evidence for evaluation runs that
did not satisfy their release gates. These files are diagnostics, not official
release artifacts, and are intentionally excluded from
`release/public-files.txt`.

`severance_refusal_policy_v0.3.6_no_go.json` is written only when the Task 6
candidate sweep has no passing threshold or selects a value other than the
approved `0.015`. Its strict allowlist excludes question/answer text, retrieved
content, endpoints, credentials, private paths, and account data. It retains
the raw unrounded content-free target observations, fresh stress/formal guard
rows, all seven candidate aggregates, failed gates, hashes, retrieval
configuration, exact execution device, clean candidate revision, and zero
provider counters.

The diagnostic schema is `1.0` and is distinct from accepted official schema
`1.2`. It can be replayed with
`rag.severance_refusal_policy.replay_no_go_evidence` without loading retrieval
models or rerunning the 130 queries. A NO-GO diagnostic never authorizes a
production threshold or official artifact.
