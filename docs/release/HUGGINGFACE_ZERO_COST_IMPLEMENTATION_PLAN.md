# Hugging Face Zero-Cost Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and release the existing BYOK RAG demo on Hugging Face `cpu-basic` with US$0 additional Hugging Face hardware cost.

**Architecture:** Keep the existing private Docker Space, remote read-only Qdrant collections, loopback FastAPI, and Streamlit BYOK UI. Use fail-closed hardware checks before and after every runtime mutation; if free CPU cannot pass acceptance, pause the Space and stop without upgrading hardware.

**Tech Stack:** Python 3.11, FastAPI, Streamlit, Hugging Face Docker Spaces, `huggingface_hub`, Qdrant Cloud, pytest, Ruff, uv.

## Global Constraints

- Requested hardware must remain exactly `cpu-basic`; current hardware may only be `None` while paused or `cpu-basic` while starting/running.
- Persistent storage must remain `None`, requested replicas must remain `1`, and the Space must remain private during acceptance.
- Never call `request_space_hardware`; never choose T4, CPU Upgrade, paid storage, or extra replicas.
- Space Secrets are restricted to `QDRANT_API_KEY` and `SESSION_SIGNING_SECRET`.
- Space Variables must not contain `GEMINI_API_KEY` or `OPENAI_API_KEY`.
- Visitors supply their own Gemini/OpenAI keys; never print, persist, upload, or paste those values into chat or shell history.
- Any hardware, privacy, permission, memory, or timeout failure pauses the Space and ends acceptance without a paid fallback.

---

## File Structure

- `docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md`: approved cost, architecture, and security policy.
- `docs/release/HUGGINGFACE_ZERO_COST_IMPLEMENTATION_PLAN.md`: executable acceptance plan and checkpoints.
- `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`: operational source of truth; update its hardware step from T4 to CPU Basic after an initial failing policy test.
- `release/public-files.txt`: exact tracked publication inventory; include both new release documents.
- `tests/test_release_verification.py`: enforces the exact tracked count and required release documents.
- `tests/test_byok_policy.py`: add the zero-cost deployment documentation assertions.
- `ui/app.py`: keep retrieval controls in the sidebar and render the approved main-content BYOK activation card.
- `tests/test_ui_byok_app.py`: exercise the real Streamlit app, including key gating, provider switching, clearing, and non-retention.

### Task 1: Preserve the Exact Publication Boundary

**Files:**
- Create: `docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md`
- Create: `docs/release/HUGGINGFACE_ZERO_COST_IMPLEMENTATION_PLAN.md`
- Modify: `release/public-files.txt`
- Modify: `tests/test_release_verification.py`

**Interfaces:**
- Consumes: `git ls-files` and `release/public-files.txt` as exact path sets.
- Produces: a 110-file public tree with no `docs/superpowers/` paths.

- [ ] **Step 1: Move planning documents into the existing release-document boundary**

Use `apply_patch` so the approved design and this implementation plan live under `docs/release/`, and remove the temporary `docs/superpowers/` copy from the current tree.

- [ ] **Step 2: Add both paths to the sorted allowlist**

Insert these exact lines between `docs/release/CLAIM_MATRIX.md` and `docs/release/OGDL_ATTRIBUTION.md`:

```text
docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md
docs/release/HUGGINGFACE_ZERO_COST_IMPLEMENTATION_PLAN.md
```

- [ ] **Step 3: Update the publication test**

Change the exact tracked count from `108` to `110`, and assert both new paths are tracked while retaining `docs/superpowers/` in `forbidden_prefixes`.

- [ ] **Step 4: Verify the boundary test**

Run:

```powershell
uv run pytest tests/test_release_verification.py::test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions -q
```

Expected: one test passes and `git ls-files` exactly equals `release/public-files.txt`.

- [ ] **Step 5: Commit**

```powershell
git add docs/release release/public-files.txt tests/test_release_verification.py
git commit -m "docs: plan zero-cost Space acceptance"
```

### Task 2: Encode the Zero-Cost Runbook Policy

**Files:**
- Modify: `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`
- Test: `tests/test_byok_policy.py`

**Interfaces:**
- Consumes: the approved constraints from `docs/release/HUGGINGFACE_ZERO_COST_DESIGN.md`.
- Produces: documentation that permits only `cpu-basic` and explicitly forbids paid fallback.

- [ ] **Step 1: Write the failing policy assertions**

Add a test that reads `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md` and asserts it contains `cpu-basic`, `US$0`, and a sentence forbidding automatic paid-hardware fallback; assert the old T4 recommendation is absent.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
uv run pytest tests/test_byok_policy.py -q
```

Expected: failure because the runbook currently recommends `t4-small`.

- [ ] **Step 3: Replace the hardware instruction**

Specify private Docker Space, `cpu-basic`, default free-tier sleep, no persistent storage, one replica, and a fail-closed pause when CPU acceptance fails. Remove the custom 3600-second sleep and T4 recommendation.

- [ ] **Step 4: Run the focused tests**

```powershell
uv run pytest tests/test_byok_policy.py tests/test_release_verification.py::test_public_git_tree_exactly_matches_allowlist_and_has_no_exclusions -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md tests/test_byok_policy.py
git commit -m "docs: enforce free CPU Space deployment"
```

### Task 3: Run the Fail-Closed Hugging Face Preflight

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: authenticated `HfApi` access to `steven0226/tw-labor-law-rag-demo`.
- Produces: sanitized evidence for visibility, revision, stage, hardware, storage, replicas, secret names, and variable names.

- [ ] **Step 1: Read current runtime and repository metadata**

Use `HfApi.get_space_runtime`, `space_info`, `get_space_variables`, and `get_space_secrets`. Print names and non-secret configuration only; never print secret values.

- [ ] **Step 2: Enforce the preflight predicates**

Require: private `true`; requested hardware `cpu-basic`; current hardware `None` or `cpu-basic`; storage `None`; one requested replica; secret-name set exactly `{QDRANT_API_KEY, SESSION_SIGNING_SECRET}`; no owner LLM keys in secrets or variables.

- [ ] **Step 3: Stop on a predicate failure**

Call `HfApi.pause_space('steven0226/tw-labor-law-rag-demo')`, verify stage `PAUSED`, and report only the failed predicate. Do not change hardware.

### Task 4: Start and Observe the Free CPU Runtime

**Files:**
- No repository changes unless a separately approved CPU-failure design is created.

**Interfaces:**
- Consumes: a passing Task 3 preflight.
- Produces: a private `RUNNING` CPU Basic Space or a safely paused failure report.

- [ ] **Step 1: Restart the paused Space without a hardware request**

Call only:

```python
HfApi().restart_space("steven0226/tw-labor-law-rag-demo")
```

- [ ] **Step 2: Poll runtime state at 15-second intervals**

For up to 15 minutes, require requested/current hardware to remain `cpu-basic` or `None` during transitions. Treat `RUNTIME_ERROR`, unexpected hardware, or the timeout as failure.

- [ ] **Step 3: Verify the private application endpoint**

Open `https://steven0226-tw-labor-law-rag-demo.hf.space` in the signed-in browser. Require the Streamlit page to load and show both Gemini and OpenAI provider choices.

- [ ] **Step 4: Apply the fail-closed branch if startup fails**

Pause the Space, capture only sanitized error categories from the Space Logs tab, and stop. Do not request paid hardware or modify the retrieval models.

### Task 5: Perform Private BYOK and Qdrant Acceptance

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: the running private UI, two Qdrant read-only collections, and visitor-owned provider keys entered directly into the UI.
- Produces: sanitized pass/fail evidence without credentials, questions, answers, provider bodies, or session tokens.

- [ ] **Step 1: Verify the keyless guard**

Without a provider key, confirm the UI does not submit a query and displays a safe prompt to enter the selected provider's key.

- [ ] **Step 2: Verify invalid-key normalization**

Enter a disposable invalid string in the password field, submit one question, and confirm the UI shows only a normalized provider failure without echoing the string or upstream body.

- [ ] **Step 3: Verify one Gemini query**

The owner enters a dedicated Gemini key directly into the Space password field. Submit one query and verify `requested_provider=gemini`, `provider=gemini`, model `gemini-3.5-flash-lite`, and no fallback. Clear the field immediately.

- [ ] **Step 4: Verify one OpenAI query**

The owner enters a dedicated OpenAI key directly into the Space password field. Submit one query and verify `requested_provider=openai`, `provider=openai`, model `gpt-5.6-luna`, and no fallback. Clear the field immediately.

- [ ] **Step 5: Verify resource controls and Qdrant invariants**

Confirm the UI reports a 20-query session limit; use the existing automated tests for the 21st-request rejection, global concurrency `2`, timeout `60`, provider-key isolation, and Qdrant write blocking. Verify the collection counts remain `labor_laws_fixed=481` and `labor_laws_structure=884` through sanitized application/previous permission evidence.

- [ ] **Step 6: Scan logs**

Confirm logs do not contain either test key, any recorded 8-character key fragment, question/answer bodies, provider response bodies, or the Qdrant endpoint. If exposure is found, pause immediately and rotate the affected key outside chat.

### Task 6: Run Regression Gates and Update the Pull Request

**Files:**
- Modify only if a test exposes a separately approved defect.

**Interfaces:**
- Consumes: Tasks 1–5 results.
- Produces: a clean branch, passing release gates, synchronized remote branch, and an evidence-only PR update.

- [ ] **Step 1: Run locked dependency validation and all tests**

```powershell
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_release.py
```

Expected: all tests and verification gates pass with no network/provider calls from `verify_release.py`.

- [ ] **Step 2: Record immutable evidence**

Record the branch commit, GitHub CI result, Space revision, `cpu-basic`, private visibility, collection counts, and sanitized acceptance results. Do not record credentials, endpoint, session token, questions, or answers.

- [ ] **Step 3: Push the branch and update PR #2**

```powershell
git push origin codex/byok-huggingface-deployment
```

Post the zero-cost and acceptance evidence to `https://github.com/kuotunyu/tw-labor-law-rag/pull/2`. Keep the PR draft and the Space private until all acceptance steps pass.

- [ ] **Step 4: Preserve the no-cost end state**

Leave requested hardware `cpu-basic`, storage `None`, replicas `1`, and default free-tier sleep. If acceptance is incomplete, pause the Space. Public visibility and merging `main` remain separate explicit approval gates.

### Task 7: Move BYOK Activation Into the Main Content

**Files:**
- Modify: `tests/test_ui_byok_app.py`
- Modify: `ui/app.py`

**Interfaces:**
- Consumes: `/models` records with `provider`, `model`, `requires_api_key`, and `/session` with `token`, `query_limit`.
- Produces: a main-content `開始安全問答` card, `st.segmented_control` provider selection, one masked `visitor_provider_key` field, provider-switch clearing, and the existing query header contract.

- [ ] **Step 1: Extend the real Streamlit integration test**

Add assertions to `test_streamlit_byok_flow_keeps_visitor_key_out_of_rendered_history` that the first run exposes one segmented provider selector with literal values `gemini` and `openai`, renders `開始安全問答`, and keeps the chat input disabled before a Key is entered. After entering a literal visitor Key, require the ready status and enabled chat input. Switch the segmented selector to `openai` and require the masked field to be empty and chat input disabled again. Re-enter a Key, submit, and retain the existing assertions that the Key appears only in `X-Provider-Api-Key`, never in query JSON, history, or rendered copy. Click `清除 API Key` and require an empty field and disabled chat input.

The production regression this catches is moving the field back below the sidebar fold, retaining a Gemini Key after switching to OpenAI, or accidentally placing the Key into visible/history state.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
uv run pytest tests/test_ui_byok_app.py -q
```

Expected: FAIL because the current app has no segmented control or main-content activation card, and switching provider retains provider-specific Key state.

- [ ] **Step 3: Implement the minimal activation card**

In `ui/app.py`, keep only chunking, retrieval mode, reranker, and explanatory copy inside `st.sidebar`. Add provider display helpers for `Gemini` and `OpenAI`, then render this structure in the main content immediately below the page caption:

```python
with st.container(border=True):
    st.subheader("🔐 開始安全問答")
    st.caption("選擇模型並貼上你自己的 API Key；本站不使用站長的模型額度。")
    selected_provider = st.segmented_control(
        "回答模型",
        available_providers,
        default=default_provider,
        format_func=provider_label,
        selection_mode="single",
        key="selected_provider",
    )
```

Before creating the password input, compare `st.session_state["provider_key_provider"]` with `selected_provider`; when they differ, set `st.session_state["visitor_provider_key"] = ""` and update the owner field. Render `st.text_input(..., type="password", key="visitor_provider_key")`, a `清除 API Key` button, accurate ready/missing status, the session query limit, and the approved three-part security copy. Do not add custom JavaScript, external fonts, owner provider credentials, or persistence.

- [ ] **Step 4: Run focused tests and lint**

```powershell
uv run pytest tests/test_ui_byok_app.py tests/test_ui_api_client.py tests/test_byok_policy.py -q
uv run ruff check ui/app.py tests/test_ui_byok_app.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the tested interface**

```powershell
git add ui/app.py tests/test_ui_byok_app.py
git commit -m "feat: surface secure BYOK activation"
```

### Task 8: Verify and Redeploy the Private Zero-Cost Space

**Files:**
- No additional repository files unless bounded visual verification finds a defect covered by a new failing test.

**Interfaces:**
- Consumes: Task 7 commit and the existing exact 110-file public allowlist.
- Produces: desktop and narrow-screen evidence, a passing full gate, synchronized PR branch, and a private free-CPU Space revision.

- [ ] **Step 1: Run full local verification**

```powershell
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_release.py
```

Expected: zero failures, `status=pass`, 110 publication files, and zero privacy findings.

- [ ] **Step 2: Inspect desktop and narrow layouts once**

Run the Streamlit app against the existing test/local API, capture one desktop and one narrow screenshot, and check that the activation card appears before chat, labels do not clip, native controls stack without horizontal overflow, and the API Key remains masked. If defects exist, add one failing behavior test where possible, fix them in one batch, and perform at most one confirmation pass.

- [ ] **Step 3: Push and wait for CI**

```powershell
git push origin codex/byok-huggingface-deployment
```

Require the pull-request CI for the new HEAD to complete successfully. Keep PR #2 draft.

- [ ] **Step 4: Deploy only the exact allowlisted tree**

Create a normal Hugging Face Space repository commit without force-pushing or rewriting Space history. Verify before and after: private `true`, requested/current hardware `cpu-basic`, storage `None`, requested replicas `1`, `DEVICE=cpu`, and no owner LLM key variable. Do not request hardware, storage, or extra replicas.

- [ ] **Step 5: Confirm the private live UI**

Require Space stage `RUNNING`, domain stage `READY`, and a masked main-content BYOK card showing both `gemini-3.5-flash-lite` and `gpt-5.6-luna`. Real provider-key acceptance remains a separate owner-entered step; never paste a provider Key into shell, logs, source, or chat.
