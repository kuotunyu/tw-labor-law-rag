# v0.3.6 Route-Aware Refusal Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Every
> checkbox is a separate verification boundary; do not batch red/green tests.

**Goal:** Remove the one measured severance-comparison direct false refusal
while preserving the global refusal policy, formal retrieval quality,
unanswerable coverage, free private deployment, and visitor-funded BYOK.

**Architecture:** Retrieval first creates a deterministic, ephemeral
`QueryPlan`. The pipeline exposes only normalized route identifiers in
`RetrievalResult`. A pure refusal-policy function selects the effective
threshold and is called by production plus every evaluation boundary. A
thirty-case offline calibration chooses the highest candidate that satisfies
all target and guard gates. The accepted artifact is content-free and bound by
the release verifier before the existing private free Space is updated.

**Tech Stack:** Python 3.11, dataclasses, `typing.Literal`, Pydantic Settings,
pytest, Qdrant local mode for offline evidence, pinned BGE-M3 and
bge-reranker-v2-m3 revisions, Ruff, Bandit, pip-audit, GitHub Actions, Hugging
Face private Space on `cpu-basic`.

## Global Constraints

- Follow strict test-driven development: add one failing test, run it and
  observe the expected failure, implement the smallest change, rerun it, then
  run the adjacent suite.
- Use `apply_patch` for source, test, dataset, manifest, and documentation
  edits. Formatting or lock-file regeneration may use the project tools.
- Never print, commit, log, or publish credentials, endpoints, private Space
  URLs, account identifiers, questions, legal text, provider payloads, or local
  absolute paths in official artifacts.
- Never construct an LLM adapter or make Gemini/OpenAI requests during
  calibration, deployment acceptance, or release verification.
- Do not create a Qdrant writer key, transition key, collection, alias, point,
  or snapshot. Keep the current Qdrant Free Tier read-only collections and the
  single v0.3.4 runtime reader.
- Keep the Space private on free `cpu-basic`; do not change hardware, replicas,
  visibility, variables, secrets, or owner-funded model policy.
- Do not overwrite the immutable v0.3.1 reliability evidence or v0.3.5
  portfolio evidence. v0.3.6 creates a new policy artifact and reaggregates
  their committed content-free scores under the new policy.
- Treat every acceptance failure as NO-GO. Do not lower a gate, choose a weaker
  candidate, edit evidence by hand, or tag a partial release.
- Work only on `codex/v036-route-aware-refusal`; never force-push `main`.

---

### Task 1: Add deterministic query planning and route observability

**Files:**

- Modify: `src/rag/retrieval/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing query-plan contract tests**

Add tests that import `QueryPlan`, `plan_retrieval_query`, and
`_retrieval_query`. Cover each single route, no route, all three routes in the
existing expansion order, case-folded English cues, and collision questions
that contain only some cue groups. Assert that the compatibility wrapper
returns exactly `plan.search_query` and that neither route labels nor expanded
terms enter the original question.

Use these exact route identifiers and order:

```python
RouteName: TypeAlias = Literal[
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
]
```

Run:

```powershell
uv run pytest tests/test_pipeline.py -q -p no:cacheprovider
```

Expected: FAIL because the new query-planning symbols do not exist.

- [ ] **Step 2: Implement the smallest query planner**

Add this public shape to `pipeline.py` and reuse the current cue groups without
changing their matching semantics:

```python
from typing import Literal, TypeAlias

RouteName: TypeAlias = Literal[
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
]

@dataclass(frozen=True)
class QueryPlan:
    search_query: str
    routes: tuple[RouteName, ...]

def plan_retrieval_query(query: str) -> QueryPlan:
    folded = query.casefold()
    expansions: list[str] = []
    routes: list[RouteName] = []
    # Evaluate the three existing cue contracts in their current order.
    return QueryPlan(search_query=" ".join((query, *expansions)), routes=tuple(routes))

def _retrieval_query(query: str) -> str:
    return plan_retrieval_query(query).search_query
```

Extend the compatible retrieval result without changing its first three
positional fields:

```python
@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    candidates: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0
    applied_routes: tuple[str, ...] = ()
```

`RetrievalPipeline.run()` must call the planner once, use
`plan.search_query` for retrieval and reranking, and return
`applied_routes=plan.routes`.

- [ ] **Step 3: Prove planner and pipeline behavior**

Run:

```powershell
uv run pytest tests/test_pipeline.py tests/test_retriever.py -q -p no:cacheprovider
uv run ruff check src/rag/retrieval/pipeline.py tests/test_pipeline.py
```

Expected: PASS; existing `_retrieval_query` callers remain compatible.

- [ ] **Step 4: Commit the query-planning boundary**

```powershell
git add src/rag/retrieval/pipeline.py tests/test_pipeline.py
git diff --cached --check
git commit -m "feat: expose deterministic retrieval routes"
```

---

### Task 2: Implement the pure fail-closed refusal policy

**Files:**

- Create: `src/rag/retrieval/refusal_policy.py`
- Create: `tests/test_refusal_policy.py`

- [ ] **Step 1: Write a failing decision table**

Parameterize tests for:

- no hits before every other condition;
- hits with no reranker;
- severance-only score below, equal to, and above its threshold;
- empty, unknown, duplicate, and multi-route inputs falling back to global
  `0.03`;
- invalid negative, greater-than-one, NaN, and infinite scores or thresholds.

Run:

```powershell
uv run pytest tests/test_refusal_policy.py -q -p no:cacheprovider
```

Expected: FAIL because the policy module does not exist.

- [ ] **Step 2: Add the exact pure policy contract**

Implement:

```python
import math
from dataclasses import dataclass

from rag.evaluation import RefusalStage

SEVERANCE_COMPARISON_ROUTES = ("severance_comparison",)

def _validated_unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return normalized

@dataclass(frozen=True)
class RetrievalRefusalDecision:
    refusal_stage: RefusalStage | None
    effective_threshold: float | None

    @property
    def refused(self) -> bool:
        return self.refusal_stage is not None

def decide_retrieval_refusal(
    *,
    has_hits: bool,
    reranker_enabled: bool,
    applied_routes: tuple[str, ...],
    top_score: float,
    global_threshold: float,
    severance_comparison_threshold: float,
) -> RetrievalRefusalDecision:
    if type(has_hits) is not bool or type(reranker_enabled) is not bool:
        raise ValueError("hit and reranker flags must be booleans")
    if not isinstance(applied_routes, tuple) or not all(
        isinstance(route, str) and route.strip() for route in applied_routes
    ):
        raise ValueError("applied_routes must be a tuple of non-blank strings")
    score = _validated_unit_interval(top_score, name="top_score")
    global_value = _validated_unit_interval(
        global_threshold, name="global_threshold"
    )
    severance_value = _validated_unit_interval(
        severance_comparison_threshold,
        name="severance_comparison_threshold",
    )
    if not has_hits:
        return RetrievalRefusalDecision("no_hits", None)
    if not reranker_enabled:
        return RetrievalRefusalDecision(None, None)
    threshold = (
        severance_value
        if applied_routes == SEVERANCE_COMPARISON_ROUTES
        else global_value
    )
    stage: RefusalStage | None = "threshold" if score < threshold else None
    return RetrievalRefusalDecision(stage, threshold)
```

Validation happens before the decision and accepts only finite values in
`[0, 1]`. `has_hits` and `reranker_enabled` must be actual booleans;
`applied_routes` must be a tuple of non-blank strings. The route override is
used only when the tuple equals `SEVERANCE_COMPARISON_ROUTES` exactly. Every
other tuple uses the global threshold. `top_score < effective_threshold`
refuses; equality allows generation. No reranker returns
`effective_threshold=None` and no score refusal. No hits returns `no_hits`.

- [ ] **Step 3: Run policy tests and lint**

```powershell
uv run pytest tests/test_refusal_policy.py -q -p no:cacheprovider
uv run ruff check src/rag/retrieval/refusal_policy.py tests/test_refusal_policy.py
```

Expected: PASS for the full table.

- [ ] **Step 4: Commit the reusable policy**

```powershell
git add src/rag/retrieval/refusal_policy.py tests/test_refusal_policy.py
git diff --cached --check
git commit -m "feat: add fail-closed retrieval refusal policy"
```

---

### Task 3: Integrate the shared policy into runtime configuration

**Files:**

- Modify: `src/rag/config.py`
- Modify: `src/rag/generation/answerer.py`
- Modify: `src/rag/factory.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_answerer.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add failing Settings and answerer boundary tests**

Assert the typed setting defaults to `0.015`, accepts every finite value in
`[0, 1]`, and rejects negative, greater-than-one, NaN, and infinite input.
Answerer tests must prove:

- `0.014999` with severance-only routes refuses without calling the LLM;
- `0.015` with severance-only routes calls the LLM;
- `0.02` with empty, unknown, duplicate, or multi-route input still refuses at
  global `0.03`;
- no reranker preserves the existing no-threshold behavior;
- no hits still refuses at `no_hits`.

Run:

```powershell
uv run pytest tests/test_config.py tests/test_answerer.py tests/test_api.py -q -p no:cacheprovider
```

Expected: FAIL on the missing setting and unchanged direct comparison.

- [ ] **Step 2: Add the bounded provisional setting**

In `Settings` add:

```python
severance_comparison_score_threshold: float = Field(
    default=0.015,
    ge=0.0,
    le=1.0,
    allow_inf_nan=False,
)
```

The value is provisional until Task 6 proves that `0.015` is the highest
passing candidate. If calibration selects another value, stop and review the
approved design instead of silently changing this default.

- [ ] **Step 3: Route production decisions through the pure policy**

Preserve positional constructor compatibility by adding a keyword-only
argument after `temperature`:

```python
def __init__(
    self,
    pipeline: RetrievalPipeline,
    llm: LLMAdapter | RoutedLLM,
    refusal_threshold: float = 0.0,
    temperature: float = 0.0,
    *,
    severance_comparison_threshold: float | None = None,
):
    self.severance_comparison_threshold = (
        refusal_threshold
        if severance_comparison_threshold is None
        else severance_comparison_threshold
    )
```

Replace both retrieval-layer branches in `answer()` with one
`decide_retrieval_refusal` call. Pass `bool(retrieval.hits)`, whether the
pipeline has a reranker, `retrieval.applied_routes`, the unrounded top score,
and both thresholds. If the returned stage is non-null, pass that stage to
`_refuse`; otherwise generate exactly as before.

`build_answerer()` passes the new Settings value explicitly. Do not add any API
or UI field.

- [ ] **Step 4: Verify runtime compatibility**

```powershell
uv run pytest tests/test_config.py tests/test_answerer.py tests/test_factory.py tests/test_api.py tests/test_ui_fixture_api.py -q -p no:cacheprovider
uv run ruff check src/rag/config.py src/rag/generation/answerer.py src/rag/factory.py tests/test_config.py tests/test_answerer.py tests/test_api.py
```

Expected: PASS; response schemas and provider-call behavior are unchanged.

- [ ] **Step 5: Commit runtime integration**

```powershell
git add src/rag/config.py src/rag/generation/answerer.py src/rag/factory.py tests/test_config.py tests/test_answerer.py tests/test_api.py
git diff --cached --check
git commit -m "feat: apply route-aware refusal at runtime"
```

---

### Task 4: Remove reconstructed decisions from evaluation runners

**Files:**

- Modify: `src/rag/reliability.py`
- Modify: `eval/run_reliability_eval.py`
- Modify: `eval/run_portfolio_demo_regression.py`
- Modify: `eval/run_provider_crosscheck.py`
- Modify: `tests/test_reliability.py`
- Modify: `tests/test_portfolio_demo_regression.py`
- Modify: `tests/test_provider_crosscheck.py`

- [ ] **Step 1: Add failing shared-boundary tests**

Patch `decide_retrieval_refusal` at each runner's import boundary and assert
that each production decision flows through the patched function once. Assert
that route labels come from `RetrievalResult.applied_routes`, not by searching
the expanded query. For the reliability trace, preserve the six public fields
exactly while accepting an already-computed decision.

Run:

```powershell
uv run pytest tests/test_reliability.py tests/test_portfolio_demo_regression.py tests/test_provider_crosscheck.py -q -p no:cacheprovider
```

Expected: FAIL because the runners still compare `top_score` directly and the
portfolio runner reconstructs route names.

- [ ] **Step 2: Make public trace reduction decision-agnostic**

Change only the input contract, not the published schema:

```python
def privacy_reduced_trace(
    row: Mapping[str, Any], *, threshold_refused: bool
) -> dict[str, Any]:
    if type(threshold_refused) is not bool:
        raise ValueError("threshold_refused must be a boolean")
```

Keep the current `_validated_row` call and output dictionary, replace only the
old `score < threshold_value` expression with the validated
`threshold_refused` argument, and retain the `PUBLIC_TRACE_FIELDS` assertion.

`run_reliability_eval._run_rows()` records `applied_routes` only in its private
raw row. When reducing a row, call the shared policy with both Settings
thresholds and pass `decision.refusal_stage == "threshold"` to
`privacy_reduced_trace`. Keep `compute_reliability_metrics()` and its historic
global threshold sweep unchanged because it is an analytical sweep, not a
runtime decision.

- [ ] **Step 3: Update portfolio and provider runners**

Delete `_applied_routes()` and its imports of private expansion constants.
Use `list(retrieval.applied_routes)` in portfolio results. Use the shared
decision object for `threshold_refused` and for provider-crosscheck admission.
Do not run the provider-crosscheck command; only its offline unit tests may run.

- [ ] **Step 4: Verify all evaluation boundaries**

```powershell
uv run pytest tests/test_reliability.py tests/test_reliability_dataset.py tests/test_portfolio_demo_regression.py tests/test_provider_crosscheck.py tests/test_wage_arrears_regression.py -q -p no:cacheprovider
uv run ruff check src/rag/reliability.py eval/run_reliability_eval.py eval/run_portfolio_demo_regression.py eval/run_provider_crosscheck.py tests/test_reliability.py tests/test_portfolio_demo_regression.py tests/test_provider_crosscheck.py
```

Expected: PASS; no runner-specific production comparison remains.

- [ ] **Step 5: Commit the shared evaluation boundary**

```powershell
git add src/rag/reliability.py eval/run_reliability_eval.py eval/run_portfolio_demo_regression.py eval/run_provider_crosscheck.py tests/test_reliability.py tests/test_portfolio_demo_regression.py tests/test_provider_crosscheck.py
git diff --cached --check
git commit -m "refactor: share refusal policy with evaluations"
```

---

### Task 5: Create and validate the reviewed thirty-case calibration contract

**Files:**

- Create: `eval/dataset/severance_refusal_policy_v0.3.6.jsonl`
- Create: `src/rag/severance_refusal_policy.py`
- Create: `tests/test_severance_refusal_policy.py`
- Modify: `eval/dataset/README.md`

- [ ] **Step 1: Write failing strict loader and scorer tests**

Define qids `severance-policy-001` through `severance-policy-030`. Require
exactly fifteen positives followed by fifteen collision negatives. Fail closed
on extra/missing fields, duplicate qids, blank text, duplicate sources/routes,
invalid booleans, bad candidate grids, non-finite scores, and content-bearing
official rows.

Each JSONL row uses exactly:

```json
{"qid":"severance-policy-001","question":"我同時有勞退舊制與新制年資，資遣費應如何分別計算？","case_type":"positive","answerable":true,"sources":[{"law":"勞工退休金條例","article":"第 12 條"},{"law":"勞動基準法","article":"第 17 條"}],"required_routes":["severance_comparison"],"prohibited_routes":[],"expect_generation":true,"style_tags":["statutory_chinese"]}
```

The committed dataset must contain a complete reviewed question in every row.
Positives must cover all style classes in the approved design. Negatives must
cover single regimes, ordinary termination, notice, wage arrears, generic
retirement, unrelated old/new wording, and partial cue collisions.

Run:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
```

Expected: FAIL because the dataset and validator do not exist.

- [ ] **Step 2: Implement strict content-free scoring**

In `src/rag/severance_refusal_policy.py`, define:

```python
CANDIDATE_THRESHOLDS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
EXPECTED_QIDS = tuple(f"severance-policy-{number:03d}" for number in range(1, 31))

@dataclass(frozen=True)
class SeverancePolicyCase:
    qid: str
    question: str
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    sources: tuple[dict[str, str], ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expect_generation: bool
    style_tags: tuple[str, ...]

def load_cases(path: Path) -> list[SeverancePolicyCase]:
    """Load and validate the exact thirty reviewed cases."""

def build_case_observation(
    case: SeverancePolicyCase,
    *,
    source_ranks: dict[str, int],
    applied_routes: tuple[str, ...],
    top_score: float,
) -> dict[str, Any]:
    """Return one validated content-free retrieval observation."""

def evaluate_candidate(
    observations: list[dict[str, Any]],
    *,
    candidate_threshold: float,
    global_threshold: float,
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute target and guard gates through the shared policy."""

def select_highest_passing_threshold(candidate_results: list[dict[str, Any]]) -> float:
    """Return the greatest candidate whose complete gate set passes."""

def build_official_artifact(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict content-free official schema."""
```

`select_highest_passing_threshold` sorts validated candidates numerically and
returns the greatest candidate whose target cases are 30/30, stress false
refusals are 0/40, stress unanswerable direct refusals are at least 17/20, and
formal Hit@5/MRR@10 plus zero direct false refusals meet the committed
baselines. It raises `RuntimeError` if none pass.

Official case rows contain only qid, case type, answerability, source ranks,
route names, unrounded decision inputs rounded to six decimals for output,
effective threshold, refusal decision, and contract booleans. They must never
contain `question`, legal content, endpoint, URL, credential, response, local
path, or account fields.

- [ ] **Step 3: Complete and manually review the dataset**

Check the positive/negative count, cue-group intent, source identities, and
style coverage. Add the dataset contract to `eval/dataset/README.md`.

- [ ] **Step 4: Run contract tests and privacy scans**

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
uv run ruff check src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
rg -n "API[_ -]?KEY|qdrant|huggingface|https?://|Users[/\\]|AI-Portfolio" eval/dataset/severance_refusal_policy_v0.3.6.jsonl
```

Expected: tests pass; the privacy scan returns no matches.

- [ ] **Step 5: Commit the reviewed calibration contract**

```powershell
git add eval/dataset/severance_refusal_policy_v0.3.6.jsonl eval/dataset/README.md src/rag/severance_refusal_policy.py tests/test_severance_refusal_policy.py
git diff --cached --check
git commit -m "test: add severance refusal calibration contract"
```

---

### Task 6: Build fresh offline calibration evidence and select the threshold

**Files:**

- Create: `eval/run_severance_refusal_policy.py`
- Create: `eval/official/severance_refusal_policy_v0.3.6.json`
- Modify: `eval/official/README.md`
- Modify: `tests/test_severance_refusal_policy.py`

- [ ] **Step 1: Add failing runner and deterministic rebuild tests**

Test argument parsing, forced `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`,
cached-model preflight, no LLM construction, strict empty work directory,
content-free export, deterministic result rebuilding after removing run time,
and NO-GO when any gate fails.

- [ ] **Step 2: Implement one offline retrieval pass**

The runner must reuse `_materialize_audited_corpus`, `_build_indexes`, the
committed corpus snapshot, and one pinned local pipeline. For each new case it
records source ranks, `retrieval.applied_routes`, and unrounded `top_score`.
Evaluate every candidate using `decide_retrieval_refusal` without rerunning
retrieval.

For guards, join the questions from the committed stress/formal datasets to
their existing content-free official scores by qid, call
`plan_retrieval_query(question).routes`, and reaggregate the committed ranks and
top scores through the same refusal policy. Do not edit the v0.3.1 artifacts.

The official artifact uses schema `1.0` and contains:

- dataset/corpus/source-artifact canonical SHA-256 values;
- exact pinned model names/revisions and retrieval settings;
- candidate grid and one selected threshold;
- per-candidate target, stress, and formal gates;
- thirty content-free case observations;
- zero provider adapters and zero provider requests;
- the candidate source Git SHA.

- [ ] **Step 3: Run the fresh offline calibration**

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
uv run python eval/run_severance_refusal_policy.py --offline --device auto --export-official
```

Expected: selected threshold is exactly `0.015`; target contracts are 30/30;
stress answerable direct false refusals are 0/40; stress direct unanswerable
coverage is at least 17/20; formal Hit@5/MRR@10 meet the committed values and
formal direct false refusals remain zero; provider adapters/requests are zero.

If the selected value is not `0.015` or any gate fails, stop with NO-GO. Do not
edit the artifact, setting, dataset, or gate to force acceptance.

- [ ] **Step 4: Verify deterministic and privacy-safe evidence**

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
uv run python -m json.tool eval/official/severance_refusal_policy_v0.3.6.json | Out-Null
rg -n "question|content|answer|endpoint|url|credential|secret|api_key|Users[/\\]|AI-Portfolio" eval/official/severance_refusal_policy_v0.3.6.json
```

Expected: tests and JSON parse pass; privacy scan returns no matches. Document
the artifact in `eval/official/README.md` without adding unverified claims.

- [ ] **Step 5: Commit accepted calibration evidence**

```powershell
git add eval/run_severance_refusal_policy.py eval/official/severance_refusal_policy_v0.3.6.json eval/official/README.md tests/test_severance_refusal_policy.py
git diff --cached --check
git commit -m "eval: bind v0.3.6 refusal calibration"
```

---

### Task 7: Bind v0.3.6 evidence, version, documentation, and publication scope

**Files:**

- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_release_verification.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `EVAL_REPORT.md`
- Modify: `docs/release/PORTFOLIO_CLAIM_EVIDENCE_MATRIX.md`
- Create: `docs/release/V036_RELEASE_NOTES.md`

- [ ] **Step 1: Add failing verifier and tamper tests**

Require schema, exact candidate grid, selected threshold `0.015`, dataset and
source hashes, model revisions, route list, 30/30 target results, stress/formal
guards, zero-provider counters, candidate source revision, privacy field
denylist, and full deterministic recomputation. Add one tamper test per major
binding: hash, route, score, selected candidate, aggregate, model revision,
provider counter, and source revision.

Change the release-version fixture and exact contract from `v0.3.5` to
`v0.3.6`. Do not change `formal_evidence_version`; the formal benchmark remains
the same committed baseline.

Run:

```powershell
uv run pytest tests/test_release_verification.py -q -p no:cacheprovider
```

Expected: FAIL until the verifier and manifest know the new artifact.

- [ ] **Step 2: Implement the verifier and manifest binding**

Add a dedicated `_verify_severance_refusal_policy` function. It reads the
dataset, stress/formal source datasets and artifacts, recomputes every summary
through production code, validates canonical hashes, and returns a compact
report. Call it from `verify_release()` and include its result in the report.

Compute every SHA from the staged file bytes or canonical text helper; never
copy a value from terminal history without verifying it again. Bind the new
setting in the manifest only after Task 6 selected `0.015`.

- [ ] **Step 3: Align release-facing documentation**

Set package/lock version to `0.3.6`. State only these new claims:

- route-aware threshold applies exclusively to exact severance-comparison
  routes;
- 30/30 target/collision contracts pass;
- stress direct false refusals are 0/40 and unanswerable direct coverage is at
  least 17/20;
- formal retrieval metrics do not regress;
- calibration uses zero provider calls;
- Qdrant, BYOK, privacy, and free hardware contracts are unchanged.

Explain that high-scoring out-of-corpus questions still rely on the
generation-layer citation refusal. Do not imply universal legal coverage or
legal advice.

- [ ] **Step 4: Update the exact public allowlist**

Add every newly created source, test, dataset, artifact, plan, design, release
note, and later deployment receipt in sorted order. The final v0.3.6 public
file count is `181`: the v0.3.5 count of `170`, the design, this plan, two policy
source modules, two policy test files, one dataset, one runner, one official
artifact, one release note, and one deployment receipt. Update the exact test
expectation only after `git ls-files` and the sorted allowlist both prove that
count.

- [ ] **Step 5: Regenerate lock and run verifier-focused gates**

```powershell
uv lock
uv lock --check
uv run pytest tests/test_release_verification.py tests/test_severance_refusal_policy.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
uv run ruff check src/rag/release_verification.py tests/test_release_verification.py
```

Expected: PASS with release version v0.3.6, selected threshold `0.015`, and
exact publication tracking.

- [ ] **Step 6: Commit the release contract**

```powershell
git add release/manifest.json release/public-files.txt src/rag/release_verification.py tests/test_release_verification.py pyproject.toml uv.lock README.md README.en.md EVAL_REPORT.md docs/release/PORTFOLIO_CLAIM_EVIDENCE_MATRIX.md docs/release/V036_RELEASE_NOTES.md
git diff --cached --check
git commit -m "docs: bind v0.3.6 route-aware refusal evidence"
```

---

### Task 8: Run complete local quality, privacy, security, and package gates

**Files:**

- Modify only files required by demonstrated failures; add a regression test
  before every fix.

- [ ] **Step 1: Run the complete test and static-analysis suite**

```powershell
uv lock --check
uv run ruff check .
uv run pytest -q -p no:cacheprovider
uv run bandit -q -r src ui scripts eval -x tests
uv run pip-audit
uv build
uv run python -c "import rag; print(rag.__file__)"
uv run labor-rag --help | Out-Null
uv run labor-rag-api --help | Out-Null
uv run python scripts/verify_release.py
```

Expected: every command exits zero. Record the actual pytest count in the
release notes; do not predict it.

- [ ] **Step 2: Run secret and publication-boundary checks**

```powershell
git grep -n -I -E "(AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,}|hf_[0-9A-Za-z]{20,}|QDRANT_API_KEY=.+|SESSION_SIGNING_SECRET=.+)"
git status --short
git diff --check origin/main...HEAD
```

Expected: secret scan returns no match; only intentional branch changes exist;
diff check passes.

- [ ] **Step 3: Fix only evidenced failures and rerun the owning gate**

Use `systematic-debugging` for unexpected behavior. Never weaken an assertion,
skip a test, suppress a security finding, or edit evidence to obtain green.

- [ ] **Step 4: Commit any demonstrated gate fix**

Use a narrow conventional commit whose subject describes the proved defect.
Skip this step if no fix was needed.

---

### Task 9: Deploy the exact candidate to the existing private free Space

**Files:**

- Create: `docs/release/V036_DEPLOYMENT_RECEIPT.md`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `tests/test_release_verification.py`

- [ ] **Step 1: Freeze and reverify the deployable source tree**

Require a clean tree, capture `git rev-parse HEAD`, build the exact Space
allowlist already enforced by the repository, and verify its archive inventory
against the committed publication/deployment contract. Do not include `.env`,
downloads, caches, runs, Git metadata, or local Qdrant files.

- [ ] **Step 2: Run signed-in Space preflight without changing settings**

Using the existing authenticated Hugging Face session, verify:

- Space identity matches the existing private demo;
- visibility is private;
- hardware is `cpu-basic` and billing tier is free;
- replicas and sleep policy are unchanged;
- required variable/secret names exist without reading or printing values;
- Qdrant collection base remains the approved v0.3.4 read-only pair;
- no owner provider key is configured.

If any condition differs, stop. Do not repair billing, secrets, hardware, or
Qdrant from this task.

- [ ] **Step 3: Upload the exact source allowlist and wait for readiness**

Use the existing deployment helper or Hugging Face API path already covered by
tests. Upload only the frozen candidate tree. Wait with bounded polling until
the Space reports `RUNNING` or a terminal failure. Do not use paid hardware.

- [ ] **Step 4: Run zero-provider acceptance**

Verify root health, private-access denial without authentication, API/session
policy, collection inventory, route-policy metadata, and retrieval-layer
admission using requests that cannot reach Gemini/OpenAI. Provider adapter and
request counters must remain zero. Do not paste a visitor API key into the
Space during acceptance.

- [ ] **Step 5: Write and bind the content-free deployment receipt**

Record date, candidate source SHA, resulting Space revision, private/free
status, unchanged collection base, BYOK policy, zero-provider acceptance, test
count, and rollback source revision from v0.3.5. Do not record the private URL,
endpoint, key, secret, account identity, or response content.

Add the receipt hash to `release/manifest.json`, ensure the final public file
count is `181`, and run:

```powershell
git add docs/release/V036_DEPLOYMENT_RECEIPT.md release/manifest.json release/public-files.txt tests/test_release_verification.py
git diff --cached --check
uv run pytest tests/test_release_verification.py tests/test_space_runner.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git commit -m "docs: record v0.3.6 private deployment"
```

- [ ] **Step 6: Re-deploy the receipt-bound commit if source SHA changed**

Because the receipt commit changes the candidate source SHA, upload the exact
receipt-bound commit once, repeat the same no-provider acceptance, and update
only the deployment revision if the platform assigns a new one. Finish with a
clean tree and a receipt whose source SHA equals `git rev-parse HEAD`. If the
receipt cannot self-bind without another source change, bind the pre-receipt
application commit and explicitly label it `deployed_application_revision`;
the release verifier must enforce that named contract.

---

### Task 10: Review, merge, tag, and publish v0.3.6

**Files:**

- Modify only release notes if final verified counts or revisions must be
  recorded before tagging.

- [ ] **Step 1: Rebase on current main and rerun risk-proportional gates**

```powershell
git fetch origin
git rebase origin/main
uv lock --check
uv run ruff check .
uv run pytest -q -p no:cacheprovider
uv run python scripts/verify_release.py
git status --short
```

Expected: clean, green, and still bound to the deployed candidate. If rebase
changes the application tree, redeploy and replace the receipt before review.

- [ ] **Step 2: Perform a code and evidence review**

Use `requesting-code-review`. Review route collision safety, equality behavior,
unknown/duplicate fallback, runtime/evaluation parity, artifact privacy,
manifest recomputation, no-provider proof, and deployment rollback. Resolve
every substantive finding with a failing regression test and rerun the owning
gate.

- [ ] **Step 3: Open and verify the pull request**

Push `codex/v036-route-aware-refusal`, open a PR summarizing the narrow defect,
accepted evidence, unchanged cost/security posture, and rollback. Wait for all
required GitHub checks. Do not merge while any required check is pending or
failed.

- [ ] **Step 4: Merge without rewriting main**

Use the repository's normal merge method. Fetch the resulting `origin/main`,
verify the merge contains every candidate commit and that main CI passes.

- [ ] **Step 5: Prove tag availability and publish the immutable release**

```powershell
git fetch --tags origin
git rev-parse --verify refs/tags/v0.3.6
```

Expected before creation: the command fails because the tag does not exist. If
it exists, stop and inspect; never move or replace it.

Create annotated tag `v0.3.6` on the verified main commit, push the tag, wait
for tag CI, and create the GitHub Release from
`docs/release/V036_RELEASE_NOTES.md`.

- [ ] **Step 6: Post-release verification and rollback readiness**

Verify the tag SHA, GitHub release assets/notes, main CI, tag CI, release
verifier from a clean checkout, private/free Space status, current Space
revision, single read-only Qdrant key scope, and zero owner-provider secrets.
If runtime acceptance fails, restore the recorded v0.3.5 Space revision; do not
change Qdrant.

## Definition of Done

- The exact severance-comparison route uses the evidence-selected `0.015`
  threshold; every other route shape keeps global `0.03`.
- Production and every evaluation runner call the same pure refusal policy.
- Target/collision calibration is 30/30, stress direct false refusals are 0/40,
  stress direct unanswerable coverage is at least 17/20, and formal metrics do
  not regress.
- Official evidence contains no content or secrets and records zero provider
  calls.
- Full tests, lint, security, dependency, package, release, CI, and privacy
  gates pass.
- The private Space remains on free `cpu-basic`, visitor BYOK remains required,
  and Qdrant remains Free Tier with one read-only v0.3.4 key.
- v0.3.6 is merged, immutably tagged, released, and recoverable by restoring
  the recorded v0.3.5 Space revision.
