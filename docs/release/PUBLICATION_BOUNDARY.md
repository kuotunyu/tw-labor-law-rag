# Publication, Privacy, and Secret Boundary

## Authoritative public set

[`release/public-files.txt`](../../release/public-files.txt) is the authoritative 94-file public set. `scripts/verify_release.py` checks that every allowlisted file exists, is sorted/deduplicated, and passes the privacy/secret patterns. In this public Git repository, the tracked set must equal that allowlist exactly and `manifest.json:publication.tracked_excluded` must be empty. Every reachable commit tree is checked against the same set. In a GitHub-generated source archive without `.git` metadata, the verifier rejects non-allowlisted files except conventional generated install/test/build paths and reports the Git-only check as `not_applicable_no_git_metadata` instead of pretending it ran.

The allowlist includes source, tests, CI/configuration, lockfile, package metadata, public documentation, privacy-reduced official evidence, and the two attributed OGDL samples. Internal `docs/superpowers/` records and `.claude/launch.json` are absent from both the public tree and every reachable public Git object.

## Official trace schemas

The 320-row ablation trace permits exactly:

`answerable`, `chunking`, `elapsed_ms`, `qid`, `rank`, `reranker`, `retrieval`, `top_score`.

The 40-row end-to-end trace permits exactly:

`answerable`, `cited_sources`, `elapsed_ms`, `judge`, `q_type`, `qid`, `refusal_stage`, `refused`, `top_score`.

Nested `judge` objects permit only `faithfulness` and `relevancy`. Nested citations permit only `doc` and `article`. Unknown fields fail closed.

The official result files also record the model identifiers needed to interpret historical evidence. Model identifiers such as `openai/gpt-5.1` are evaluation provenance; request IDs, headers, endpoints, token usage, credentials, or provider response payloads are not retained.

## Explicit exclusions

The following are retained locally when they exist and excluded from publication; nothing in the closure deletes original research evidence:

- `.env` and all real credentials
- `.venv/`, caches, generated build directories, and worktrees
- `storage/`, vector/BM25 indexes, embedding cache, and downloaded tokenizer cache
- `data/raw/`, full government dumps, and the other normalized corpus files
- `eval/runs/`, complete generated answers, judge reasons, raw summaries, and debug logs
- provider responses, prompts, request identifiers, token usage, headers, endpoints, and API metadata
- absolute/private machine paths and user identifiers
- local working notes: `INTERVIEW_PREP.md`, `STARTUP.md`, and `plan.md`
- machine/session files under `.claude/` other than no allowlisted entries

The full 15-instrument corpus is therefore outside the public set. The two files under `data/sample/` are deliberate exceptions covered by [OGDL_ATTRIBUTION.md](OGDL_ATTRIBUTION.md).

## Scans and data minimization

The verifier performs three complementary checks:

1. Exact JSON field allowlists for both official traces and their nested objects.
2. A publication scan for private filesystem paths, private-key blocks, known GitHub/Google/OpenAI/Anthropic token forms, non-placeholder key assignments, provider payload fields, personal identifiers, missing files, non-UTF-8 public text, and forbidden local paths.
3. Existing official-artifact tests that reject local paths and secret-like fields and recompute all arithmetic summaries.

Scanner findings contain only repository-relative path, category, and location. Matched values are never copied into the report. `.env.example` may contain blank or obvious placeholder variables; non-placeholder assignments fail. Binary files cannot be meaningfully regex-scanned, so every allowlisted binary must instead match a manually reviewed SHA-256 recorded in `release/manifest.json`; an unreviewed or changed binary fails closed.

## Current audit result

At public-release verification time, both official trace schemas pass with zero issues and the publication scan reports zero issues. The release verifier returns these counts under `privacy.official_trace_issues` and `privacy.public_scan_issues`. This result must be regenerated on the exact release tree before any GO statement; it is not a permanent guarantee for future commits.
