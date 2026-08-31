# v0.3.6 Windows minimal child-environment design

**Status:** Approved by delegated project authority  
**Date:** 2026-08-31

## Context

The first authoritative CPU/FP32 launch stopped before embedding-model import.
The privacy-minimized Windows child environment preserved `USERPROFILE` for
local model discovery but omitted `USERNAME`.  Python's `getpass.getuser()`
therefore exhausted its environment lookup and attempted the unavailable POSIX
`pwd` fallback.  No evaluation observation or release artifact was produced.

This was an orchestration-contract failure, not a retrieval-quality failure.

## Decision

Define and regression-test one exact Windows child-environment allowlist for
the private v0.3.6 runtime smoke and future authoritative launch:

- `PROCESSOR_ARCHITECTURE`
- `SYSTEMROOT`
- `USERNAME`
- `USERPROFILE`
- `WINDIR`

The child environment starts from an empty map.  Only names present in the
parent are copied, then the existing fixed offline controls are added.  The
contract does not copy the ambient environment wholesale.

`USERNAME` is required so `getpass.getuser()` remains on its normal Windows
path.  `USERPROFILE` is required for deterministic discovery of already-local
model files.  Neither value may be serialized into reports or artifacts.

## Security boundary

The regression probe must prove all of the following in a fresh subprocess:

1. `getpass.getuser()` resolves to the explicitly supplied `USERNAME`.
2. The exact Windows allowlist and exact offline controls are present.
3. Runtime-injected `LC_CTYPE=C.UTF-8` is tolerated on affected POSIX test
   runtimes, while ambient locale values are not forwarded.
4. Provider, Hugging Face, Qdrant, and unrelated sentinel secrets are absent.
5. No unlisted parent variable reaches the child.

The authoritative operator must reproduce this empty-map allowlist instead of
inventing a separate environment policy.  This is a private execution
contract; no new public CLI mode or provider access is introduced.

## Release consequences

The failed single launch remains immutable audit evidence.  A new calibration
is permitted only after:

1. the regression test is RED before the fix and GREEN after it;
2. the complete bootstrap/artifact contract suite passes;
3. an independent review finds no P0-P2 issue;
4. a fresh model-free preflight passes on the new clean candidate.

The next calibration remains one launch, CPU/FP32, offline, provider-free, and
must not reuse any previous run directory or result.
