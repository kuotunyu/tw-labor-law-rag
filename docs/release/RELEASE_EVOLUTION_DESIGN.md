# Release Evolution Foundation Design

**Status:** Approved for specification on 2026-08-28
**Target branch:** `feat/v0.2-release-evolution-foundation`
**Repository:** `tw-labor-law-rag`

## Context

The v0.1.0 release verifier was designed for a one-commit source snapshot. It
currently enumerates commits with `git rev-list --all`, caps reachable history
at one commit, and requires every reachable commit tree to equal the current
publication allowlist.

The canonical local repository intentionally retains the amended-away v0.1.0
commit under the non-publishable recovery ref
`refs/archive/amended-away-before-public-snapshot-20260828`. The public refs
(`main`, `origin/main`, and `v0.1.0`) still resolve to the single public commit
`61385a24155a97af6b5d0201f9d81a1ad3eaf379`, but `git rev-list --all` also
follows the recovery ref. Consequently, the exact v0.1.0 tree reports:

- Ruff: pass
- unit and artifact tests: 127 pass
- release verifier and its integration test: fail with
  `reachable history has 2 commits; maximum is 1`

Deleting the recovery ref would make the current checkout green but would not
solve the architectural problem: the next legitimate public commit would
violate the same one-commit ceiling. The verifier needs a precise public-ref
boundary and a history policy that supports normal repository evolution.

## Goals

1. Preserve local recovery refs and exclude non-publishable namespaces from
   public-history claims.
2. Audit commits reachable from standard publishable refs:
   `refs/heads/*`, `refs/tags/*`, and `refs/remotes/*`.
3. Allow normal multi-commit public history without weakening the current-tree
   publication allowlist.
4. Fail closed if any publishable historical commit contains a forbidden path,
   secret-like text, private machine data, or an unreviewed binary.
5. Keep the clean reviewer path offline: no model loading, provider calls,
   Qdrant, Docker, GPU, or network service.
6. Keep `main`, `origin/main`, and tag `v0.1.0` unchanged while this work is
   developed and reviewed on a local feature branch.

## Non-goals

- Do not delete, rewrite, push, or rename the recovery ref.
- Do not push, merge, tag, or publish the feature branch in this work unit.
- Do not change retrieval, reranking, generation, refusal thresholds, UI, or
  evaluation metrics.
- Do not implement narrative-query rewriting or threshold recalibration in
  this work unit; those require a separate evidence-first design.
- Do not claim that local-only namespaces can never be pushed by custom Git
  plumbing. The contract is about refs selected by the verifier and ordinary
  Git publication semantics.

## Approaches Considered

### A. Publishable-ref-aware history verification (selected)

Enumerate only standard publishable refs, retain the current exact allowlist,
and scan every publishable historical tree. This preserves recovery evidence,
supports normal commits, and strengthens privacy verification beyond the
v0.1.0 path-only history check.

### B. Delete the recovery ref

This is a short-lived workaround. It discards an explicit recovery pointer and
the next public commit fails the one-commit ceiling again.

### C. Squash every release back to one commit

This preserves the current verifier but forces recurring history rewrites,
obscures engineering evolution, and increases operational risk. It is not a
sustainable public repository workflow.

## Publication Boundary

### Publishable refs

The verifier will derive public commits with the semantic equivalent of:

```text
git rev-list --branches --tags --remotes
```

The resulting commit IDs are deduplicated. This includes feature branches,
release branches, annotated and lightweight tags, and remote-tracking refs.
It deliberately excludes local-only namespaces such as:

- `refs/archive/*`
- `refs/stash`
- `refs/notes/*`
- Git worktree administration refs

A branch named `archive/foo` remains publishable because its full ref is
`refs/heads/archive/foo`; it must still be audited.

### Current tree

The checked-out commit remains subject to the existing strict contract:

- `git ls-files` must equal `release/public-files.txt` exactly.
- Every allowlisted file must exist.
- Public text must be UTF-8 and pass privacy/secret scanning.
- Allowlisted binaries must match reviewed SHA-256 entries in
  `release/manifest.json`.
- No excluded runtime, credential, raw evaluation, or private-note path may be
  tracked.

### Historical public trees

Every commit reachable from a publishable ref will be materialized in memory
with `git archive --format=tar <commit>`. The verifier will not check out an
old commit or write it into the working tree.

Each archive entry is represented as a path plus bytes and is passed through a
shared publication scanner:

1. Paths must belong to the current public allowlist or to the explicit
   append-only `publication.history.legacy_public_paths` manifest list.
2. Paths matching the project-wide forbidden-path rules fail regardless of any
   allowlist entry.
3. UTF-8 text is scanned with the same private-path, identity, secret-token,
   provider-payload, and forbidden-content rules used for the current tree.
4. Non-text blobs must match a reviewed SHA-256 in the manifest. The reviewed
   binary hash set is append-only so a previously published binary remains
   auditable after replacement or removal.
5. Author and committer identity must remain
   `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.

`legacy_public_paths` is empty for this work unit. It exists so a future release
can deliberately remove or rename a once-public file without making the old
public commit unverifiable. Adding a legacy path is a reviewed manifest change;
it cannot bypass forbidden-path or content scanning.

### Gitless source archives

The current GitHub source-archive behavior remains unchanged. When `.git`
metadata is absent, Git history checks report
`not_applicable_no_git_metadata`; the extracted current tree still must satisfy
the exact archive inventory and privacy checks.

## Components and Interfaces

### `src/rag/release_verification.py`

Introduce three focused internal units:

```python
@dataclass(frozen=True)
class PublicEntry:
    path: str
    data: bytes


def _publishable_commit_ids(project_root: Path) -> list[str]:
    """Return sorted unique commits reachable from heads, tags, and remotes."""


def _git_archive_entries(project_root: Path, commit: str) -> list[PublicEntry]:
    """Read one commit tree through an in-memory tar archive."""


def _verify_publishable_git_history(
    project_root: Path,
    current_public_paths: set[str],
    *,
    legacy_public_paths: set[str],
    reviewed_binary_hashes: set[str],
) -> int:
    """Validate every publishable commit and return the unique commit count."""
```

The existing current-tree scanner will be refactored only enough to share pure
path/byte validation with historical entries. Filesystem traversal stays in
the current-tree adapter; Git archive parsing stays in the history adapter.
Neither adapter may weaken the existing scanner categories or expose matched
secret values in error messages.

### `release/manifest.json`

Replace the one-release ceiling:

```json
{
  "publication": {
    "history": {
      "refs": ["heads", "tags", "remotes"],
      "legacy_public_paths": [],
      "reviewed_binary_sha256": [
        "ee95af1a9a8b92d2e1b8521da48e0a56fcc7447982122193bcc251ef95b86ecb"
      ]
    }
  }
}
```

The manifest records the selected namespaces for documentation and verifier
consistency. It no longer contains `max_commits`; safety comes from validating
every selected commit, not from limiting their number. The existing
`publication.reviewed_binaries` mapping remains the exact path-to-hash contract
for the current tree. `history.reviewed_binary_sha256` is the append-only set of
binary digests permitted anywhere in publishable history and must contain every
current reviewed-binary digest.

### Release documentation

Update these files to use the term **publishable history** and describe the
local recovery-ref boundary:

- `README.md`
- `README.en.md`
- `docs/release/PUBLICATION_BOUNDARY.md`
- `docs/release/CLAIM_MATRIX.md`
- `docs/release/REVIEWER_GUIDE.md`
- `release/public-files.txt`

This design document is itself part of the reviewed public inventory.

## Data Flow

```text
standard ref namespaces
        |
        v
git rev-list --branches --tags --remotes
        |
        v
sorted unique commit IDs
        |
        +--> identity check
        |
        +--> git archive --format=tar <commit>
                  |
                  v
           PublicEntry(path, bytes)
                  |
                  v
       shared path/content/binary scanner
                  |
                  v
      pass or privacy-safe failure category
```

The current worktree independently follows the existing exact
`git ls-files == public-files.txt` path before both results are combined in the
release report.

## Error Handling

- A Git command failure raises `ReleaseVerificationError` with the operation
  and commit ID, never captured blob content.
- A malformed or unsafe tar entry fails closed. Absolute paths, `..` traversal,
  links, devices, and unsupported entry types are rejected.
- Duplicate tar paths fail closed.
- Invalid UTF-8 in a file expected to be text is reported by path and category,
  not by file contents.
- Secret/privacy findings retain the existing redacted report format:
  repository-relative path, category, and location only.
- An empty publishable-ref set fails in a Git checkout; it is not silently
  treated as a source archive.
- Local recovery refs are neither opened nor reported by name in the public
  evidence summary.

## Test Strategy

All behavior changes use red-green-refactor TDD.

### Ref-selection tests

1. Create a temporary Git repository with one clean `main` commit.
2. Create a second, forbidden commit and retain it only under
   `refs/archive/recovery`.
3. Verify the archive-only commit is excluded and the public count is one.
4. Point `refs/heads/archive/recovery` at the same commit and verify it is
   included and rejected.
5. Verify annotated tags and remote-tracking refs are included and deduplicated.

### Historical-content tests

- A secret-like value added to an allowed text path and removed by the current
  commit must still fail through historical scanning.
- A forbidden path in a historical public commit must fail.
- A historical reviewed binary must pass by hash.
- An unreviewed historical binary must fail without echoing its bytes.
- Unsafe tar paths and link entries must fail closed.
- A current allowlist file absent from an older additive commit is permitted.
- A historical path absent from both current and legacy allowlists must fail.

### Regression gates

- Existing source-archive tests remain green.
- The real local recovery ref remains present.
- Ruff passes.
- `scripts/verify_release.py` passes from the feature worktree.
- The full pytest suite passes with zero failures.
- `uv build` succeeds and packaging/import smoke tests pass.
- `git status --short` is clean after verification.

## Security and Privacy Properties

- The design does not make local recovery data public; it defines why that ref
  is outside the publication graph.
- Standard branches, including branches whose short name begins with
  `archive/`, cannot evade verification.
- Removed secrets in allowed filenames remain detectable in publishable
  history.
- Scanner findings never reproduce secret values.
- No remote operation is required to verify the repository.

## Rollout and Recovery

1. Work only in the isolated local feature worktree.
2. Commit the approved design before implementation planning.
3. Implement in small TDD commits.
4. Keep `main`, `origin/main`, `v0.1.0`, and `refs/archive/*` unchanged.
5. Stop before merge, push, or tag and present verified results for review.

If implementation is abandoned, remove only the feature worktree and feature
branch after explicit approval. The canonical checkout and recovery ref remain
unchanged throughout.

## Success Criteria

The work unit is complete when all of the following are true:

1. The existing local recovery ref still resolves to
   `6527cbd35ee559573d83f83d3019c3d67c190e46`.
2. Publishable refs remain rooted in the intended public history.
3. The release verifier passes without deleting or renaming local-only refs.
4. A public branch containing forbidden historical content is rejected.
5. A secret removed from the current tree but retained in publishable history
   is rejected with a redacted finding.
6. The current tracked set still equals the authoritative public allowlist.
7. Ruff, pytest, release verification, package build, and smoke imports all
   pass offline.
8. No remote operation or application behavior change occurs.

## Follow-up Work

After this foundation is complete, the next independent design will address
the measured narrative-query failure boundary: expand the stress set first,
then compare query rewriting and refusal-gate strategies using evidence rather
than changing the 0.03 threshold directly.
