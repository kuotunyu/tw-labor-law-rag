# Blue-Green Qdrant Maintenance Design

## Status and decision

This design covers the manual maintenance path for the public BYOK demo. It
does not change the query pipeline, model revisions, public prompts, answer
thresholds, Hugging Face hardware, or the visitor-owned provider-key boundary.

The selected approach is a fail-closed blue-green rebuild. A candidate pair of
Qdrant collections is created under a new base name, validated in isolation,
and only then made active by changing the Hugging Face `COLLECTION_NAME`
variable. The currently active collections are never recreated, overwritten,
or deleted by the maintenance command.

The existing `v0.3.4` annotated tag remains immutable. Its GitHub Release will
describe the tagged runtime change and link to the later, additive targeted
regression evidence on `main`; the tag is not moved or force-updated.

## Goals

1. Provide a manual, reproducible way to determine whether the 15-law corpus
   differs from the committed audited snapshot.
2. Build `fixed` and `structure` candidate collections without touching the
   active pair.
3. Prove that the local normalized corpus and source archives match the
   committed `release/corpus_snapshot.json` before any cloud write.
4. Validate point counts, vector dimensions, and citation provenance fields
   before cutover.
5. Produce a redacted maintenance receipt containing no credential, endpoint,
   law text, question, answer, or provider payload.
6. Preserve a simple rollback path and require immediate revocation of the
   temporary writer key after the operator finishes.

## Non-goals

- No scheduled workflow, monitor, cron job, or unattended mutation.
- No paid Qdrant tier, Hugging Face hardware, persistent Space storage, or GPU.
- No in-place rebuild of `labor_laws_fixed` or `labor_laws_structure`.
- No automatic deletion of old, partial, or candidate collections.
- No automatic modification of `release/corpus_snapshot.json`.
- No Gemini or OpenAI request and no use of an owner LLM key.
- No automatic Hugging Face cutover or visibility change.

## Alternatives considered

### Publish only

Publishing the existing tag without adding a maintenance path is the smallest
change, but it leaves the operator dependent on the destructive
`scripts/build_index.py` behavior for future cloud refreshes. It is acceptable
for a frozen demo but does not meet the provenance-maintenance goal.

### Rebuild the active collections in place

The existing index command calls `recreate_collection`, which deletes an
existing collection before creating it again. A failure between deletion and
the final upsert could take the public demo offline and remove the rollback
source. This option is rejected.

### Automated scheduled refresh

A scheduled job could detect changes quickly, but it would introduce
unattended external writes, secret custody, failure handling, and possible
resource use. The owner explicitly requested no monitoring schedule. This
option is rejected.

## Architecture

### 1. Audited local corpus gate

The operator first runs the existing official-source download. The maintenance
command hashes the cached `acts` and `regulations` ZIP archives, loads the 15
normalized JSON laws, and rebuilds the same canonical snapshot representation
used by `rag.corpus_audit`.

The rebuilt snapshot must exactly match the committed snapshot except for the
operator-supplied observation date. In particular, the following values must
match before a cloud client is created:

- both official source SHA-256 values;
- the exact 15-law name set;
- 884 non-deleted articles in total;
- every law's article count and canonical content SHA-256;
- every law's official URL, amendment date, and effective date.

If any field differs, the command exits with a redacted drift summary and tells
the operator to review the legal changes in a separate release task. It does
not accept a flag that silently blesses or rewrites a changed snapshot.

### 2. Pure maintenance contracts

`src/rag/qdrant_maintenance.py` owns validation and redacted receipt building.
It has no network client and exposes small, testable interfaces:

- `candidate_collections(base: str) -> dict[str, str]` returns the fixed and
  structure collection names.
- `validate_candidate_base(active_base: str, candidate_base: str) -> None`
  rejects invalid, reserved, or active names.
- `validate_snapshot_match(committed: Mapping, local: Mapping) -> None` fails
  on source, law, metadata, count, or content drift.
- `validate_candidate_payloads(strategy: str, payloads: Sequence[Mapping],
  expected_count: int) -> None` enforces count and provenance requirements.
- `build_maintenance_receipt(...) -> dict` returns only allowlisted metadata.

Candidate base names use lowercase ASCII letters, digits, `_`, and `-`, begin
with an alphanumeric character, and are 3-64 characters long. They must differ
from the active base. The recommended form is
`labor_laws_YYYYMMDD_<short-snapshot-sha>`.

### 3. Non-destructive Qdrant adapter

`VectorStore` gains a `create_collection` operation that fails if the target
already exists. Existing local-development `recreate_collection` behavior is
retained for backward compatibility, but the maintenance command never calls
it.

The blue-green command checks that neither candidate collection exists before
creating either one. It has no delete API and no overwrite mode. If a partial
build remains after a failure, the command reports the candidate names and
stops. Cleanup is a separate, explicitly authorized operator action.

### 4. Manual command boundary

`scripts/rebuild_qdrant_blue_green.py` is dry-run by default. Dry-run performs
the local corpus gate, computes the candidate collection names and expected
point counts, verifies the pinned embedding model is locally available, and
prints a redacted plan without connecting to Qdrant.

Cloud mutation requires all of the following:

- `--execute`;
- `--candidate-base <name>`;
- `--confirm-candidate-base <same-name>`;
- `QDRANT_URL` in the process environment;
- `QDRANT_WRITER_API_KEY` in the process environment;
- `COLLECTION_NAME` identifying the active base;
- both pinned model snapshots already available locally.

The writer key is deliberately named differently from the runtime
`QDRANT_API_KEY`. The command does not read Gemini, OpenAI, Hugging Face, W&B,
or runtime Qdrant secrets. It never prints an endpoint or any credential.

For each strategy, the command chunks the audited corpus, embeds the chunks,
creates a new collection, upserts the points, verifies the exact point count,
scrolls payload metadata without vectors, and validates provenance. Only after
both strategies pass does it write a receipt under the ignored
`eval/runs/qdrant-maintenance/` directory.

### 5. Cutover and rollback

The command does not change Hugging Face. After a successful receipt, the
operator performs a separate private acceptance:

1. Keep the public Space running on the old base.
2. Change `COLLECTION_NAME` to the candidate base and restart the Space.
3. Require `RUNNING`, domain `READY`, HTTP health 200, expected model options,
   and one owner-entered BYOK query per provider if provider behavior changed.
4. Confirm logs contain no key, endpoint, question, answer, or provider body.
5. If acceptance fails, restore the previous `COLLECTION_NAME` and restart.
6. Revoke the temporary writer key after success or failure.

The old pair remains available for rollback. Deleting it is explicitly outside
this design and requires a later, separately confirmed destructive operation.

## Error handling

- Corpus or source drift stops before model loading or Qdrant connection.
- A missing pinned model snapshot stops before any Qdrant connection.
- An invalid or active candidate base stops before any Qdrant connection.
- An existing candidate collection stops before any create or upsert.
- A partial create/upsert failure leaves the active pair unchanged and reports
  only candidate names and error class.
- Count, vector, or provenance mismatch marks the candidate invalid and blocks
  cutover; it does not delete the candidate automatically.
- Receipts are written atomically through a temporary file and renamed only
  after both collections pass.

## Security and privacy

The maintenance receipt may contain only schema version, UTC completion time,
candidate and active collection names, strategies, point counts, vector
dimension, corpus snapshot SHA-256, source archive SHA-256 values, and pinned
embedding model identifier/revision. It must not contain endpoints,
credentials, local absolute paths, law text, questions, answers, prompts,
provider metadata, or exception messages copied from remote responses.

All logs use fixed operation labels and sanitized exception class names. The
temporary writer key is supplied only through the process environment, never
through a command-line argument, source file, receipt, or chat.

## Testing and acceptance

Tests use fakes and isolated local Qdrant instances; they do not call Qdrant
Cloud or a generation provider.

Required automated coverage:

1. Candidate names reject the active base, unsafe characters, and unequal
   confirmation values.
2. Snapshot validation rejects source, law, metadata, count, and content drift.
3. Dry-run creates no Qdrant client and writes no receipt.
4. Execute mode refuses missing credentials and missing cached model snapshots
   before cloud access.
5. Existing candidate collections block all mutation.
6. Fixed and structure builds use create-only semantics and exact count checks.
7. Payload validation requires `doc_title`, `article_label`, `source_url`,
   `last_amended`, and `effective_date`, without recording law content.
8. Receipts contain only the documented field allowlist and portable paths.
9. Failure paths never invoke collection deletion.
10. The full test suite, Ruff, Bandit, dependency audit, and release verifier
    pass with zero public-boundary or privacy findings.

## Release publication

The GitHub Release for `v0.3.4` uses the existing annotated tag. Release notes
state that the tag ships deterministic wage-arrears query expansion and does
not change providers, thresholds, Qdrant data, or historical metrics. They link
to the additive targeted regression evidence on `main` rather than claiming
that the later evidence files are present in the tag archive.

No tag is moved, recreated, force-pushed, or deleted.
