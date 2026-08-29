# Wage-Arrears Query Expansion Design

**Status:** Approved for implementation on 2026-08-30

**Target branch:** `feat/v0.3.4-wage-arrears`

**Target release:** `v0.3.4`

## Context

The formal 40-question evaluation has one answerable question that the end-to-end
system refuses: `eval-10`, which asks whether a worker may leave without notice
when the employer has withheld wages. The relevant source is Labor Standards Act
Article 14, but seven of the eight recorded retrieval configurations do not place
that article in the top five. The report identifies a vocabulary gap between the
colloquial wording "拖欠薪水" and the statute wording "不依勞動契約給付工作報酬".

`v0.3.3` already has a narrow deterministic query-expansion layer. It appends legal
retrieval terms only when all cue groups for a measured scenario match. Retrieval
and reranking see the expanded string, while answer generation continues to see
the user's original question. This design extends that existing mechanism instead
of adding a provider call, changing the global refusal threshold, or rebuilding
the vector indexes.

## Goals

1. Bridge the measured wage-arrears vocabulary gap for questions about a worker
   ending employment immediately or without notice.
2. Keep the expansion deterministic, local, fast, and free of provider cost.
3. Require independent wage-nonpayment and worker-termination cues so unrelated
   wage, resignation, or employer-dismissal questions remain unchanged.
4. Preserve the original question for answer generation and all existing query
   expansion behavior.
5. Add an auditable contract without rewriting the committed formal metrics.

## Non-goals

- Do not change the reranker threshold, retrieval depth, model revisions, prompts,
  generation providers, BYOK policy, UI, Qdrant collections, or Space settings.
- Do not call Gemini or OpenAI and do not add any owner-funded provider key.
- Do not introduce general-purpose synonym rewriting, HyDE, or an LLM query
  rewriter.
- Do not claim that the formal Hit@5, MRR, or false-refusal metrics improved unless
  a separately versioned full evaluation produces new committed evidence.
- Do not perform Qdrant backup, restore, corpus refresh, or live deployment in this
  work unit.

## Approaches Considered

### A. Narrow deterministic expansion (selected)

Require both a wage-nonpayment cue and a worker immediate-termination cue, then
append Article 14 terminology. This directly targets the documented failure,
reuses the existing architecture, has no provider cost, and is straightforward to
cover with positive and collision tests.

### B. Provider-based query rewriting

Ask a language model to rewrite every colloquial question into legal terminology.
This could generalize beyond the measured case, but it adds latency, visitor-key
use, provider failure modes, and a substantially larger evaluation burden. It is
incompatible with the zero-extra-cost objective for retrieval.

### C. Broad synonym normalization

Maintain a general dictionary that rewrites wage and termination vocabulary
independently. This is cheaper than provider rewriting but has a larger collision
surface: a salary question and an unrelated resignation question could be pushed
toward Article 14. The narrower all-cue rule is safer for the current evidence.

## Trigger Contract

`src/rag/retrieval/pipeline.py` will define two explicit cue groups. The initial
reviewed vocabulary is deliberately finite:

```python
WAGE_NONPAYMENT_CUES = (
    "欠薪",
    "沒發薪",
    "沒有發薪",
    "未發薪",
    "沒付薪",
    "沒有付薪",
    "未付薪",
    "拖欠工資",
    "積欠工資",
    "沒付工資",
    "沒有付工資",
    "未付工資",
    "未給付工資",
    "未給付工作報酬",
    "沒有付 salary",
    "沒付 salary",
    "unpaid salary",
    "wage arrears",
)

WORKER_IMMEDIATE_TERMINATION_CUES = (
    "直接離職",
    "立即離職",
    "馬上離職",
    "立刻離職",
    "立即終止",
    "直接終止",
    "直接 resign",
    "immediately resign",
    "resign without notice",
)
```

The short `欠薪` cue intentionally covers compounds such as `拖欠薪水` and
`連續欠薪`. The English-mixed phrases cover the existing reliability-stress
wording without treating the generic word `salary` as nonpayment. Generic
`不經預告` is likewise insufficient by itself because it also appears in
employer-dismissal questions.

The rule activates only when at least one cue from each group is present after
`casefold()`. A cue from only one group must not alter the query. Employer-initiated
dismissal wording must not count as worker termination merely because it mentions
notice.

On a match, the pipeline appends this fixed retrieval aid exactly once:

```text
勞動基準法 第十四條 不依勞動契約給付工作報酬 勞工得不經預告終止契約
```

The new rule is appended after the existing off-hours and severance rules. This
preserves the stable order of all previously shipped expansions. If a question
matches more than one rule, each legal-term block appears once in that order.

## Components and Data Flow

No public API or data model changes are required.

```text
original user question
        |
        +-------------------------------> answer generator
        |
        v
casefold + all-cue matching
        |
        v
original question + fixed Article 14 terms
        |
        +--> BM25/vector retrieval --> reranker
```

`RetrievalPipeline.run()` continues to derive one search string through the pure
`_retrieval_query()` helper. The retriever and optional reranker receive that
search string. The pipeline result does not store the expanded query, and the
answerer continues to build its prompt from the original caller-supplied question.

## Error Handling and Safety

- Matching is a pure in-process string operation and introduces no I/O or new
  runtime exception path.
- A nonmatching query is returned byte-for-byte unchanged.
- Expansion terms contain no user data, secrets, endpoint information, or dynamic
  provider content.
- The rule must not log the user question or the expanded search string.
- Existing refusal behavior remains fail-closed: if retrieval still does not
  provide sufficient evidence, the threshold and generation layer behave exactly
  as before.

## Testing Strategy

Development follows red-green-refactor in `tests/test_pipeline.py`:

1. Add failing positive tests for `eval-10` and the two existing reliability
   stress phrasings (`stress-010` and `stress-038`).
2. Assert the retriever and reranker receive the original question followed by
   the fixed Article 14 terms.
3. Add collision tests proving that wage nonpayment alone, immediate resignation
   alone, employer dismissal, and unrelated salary/resignation wording remain
   unchanged.
4. Extend the stable-order test so all matching rule blocks appear once in the
   documented order.
5. Run the targeted pipeline tests, then the full offline test suite, Ruff, lock
   verification, security checks, release verifier, and source build.

The pre-change isolated-worktree baseline is `479 passed` with three existing
third-party jieba deprecation warnings. Tests and release verification must not
call a provider, start Docker, mutate Qdrant, or expose API keys.

If the audited corpus, fixed model snapshots, and an isolated local Qdrant are
already available, a no-provider retrieval check may additionally record whether
Article 14 enters the top five for the target wording. That check is supporting
evidence only; without a complete versioned evaluation run, public release metrics
remain the existing baseline.

## Documentation and Release Claims

Implementation will update the Chinese and English README release notes, the main
architecture decision record, and the evaluation-report limitation text. The
claim will be limited to the deterministic routing contract: matching wage-arrears
and worker-termination questions receive Article 14 retrieval terms while the
generator receives the original question. Documentation must not state that the
historical `eval-10` outcome or aggregate quality metrics changed unless new
offline evidence is committed and verified.

## Acceptance Criteria

- All reviewed positive phrasings trigger exactly one Article 14 term block.
- Each reviewed single-group and collision phrasing remains unchanged.
- Existing off-hours and severance expansion tests continue to pass.
- The answer-generation input contract remains the original question.
- No provider call, Qdrant mutation, paid infrastructure, secret change, or UI
  change is introduced.
- The complete offline verification chain passes before any completion or release
  claim.
