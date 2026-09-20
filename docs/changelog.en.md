# Changelog

[繁體中文](changelog.md) | [English](changelog.en.md)

Per-version notes, moved unchanged from [README.en.md](../README.en.md). The formal evaluation numbers are still the `v0.1.0` results (see [EVAL_REPORT.md](../EVAL_REPORT.md)); none of the releases below rewrote them. The standalone v0.3.5 release notes are in [V035_RELEASE_NOTES.md](release/V035_RELEASE_NOTES.md).

## v0.3.5 portfolio readiness

This release turns the private BYOK demo into a reviewer-first journey: the landing view explains the verifiable capabilities and cost boundary before an invited reviewer selects Gemini or OpenAI, enters a dedicated key in a masked field, and inspects staged progress, citations, and expandable debug evidence. The Space stays private on free `cpu-basic`, holds no owner LLM key, and performs no cross-provider fallback.

It adds a ten-case, fully offline, content-free demonstration regression: all six answerable source contracts pass, all ten routing/retrieval-stage decision contracts pass, and provider calls remain zero. It also binds a 15-instrument/884-article content-free SHA-256 baseline to a manual official-source audit that reports law/source fields plus added, removed, or changed article labels. These compact proofs do not replace the 40-question formal baseline, 60-case reliability suite, or archived provider judgments.

## v0.3.4 wage-arrears/immediate-exit retrieval hardening

Only questions matching both reviewed wage-nonpayment and worker immediate-exit cue groups receive fixed Labor Standards Act Article 14 retrieval terms. BM25, dense retrieval, and the reranker see the expanded query; generation still receives the visitor's original question.

This release adds no provider call, 0.03 threshold change, Qdrant rebuild, or historical metric rewrite. The `v0.1.0` formal baseline and `v0.3.1` reliability evidence keep their original evidence versions; the v0.3.4 public claim is limited to the unit-testable deterministic routing contract.

## v0.3.3 new/old-regime severance retrieval hardening

This is the `v0.3.3` source-only runtime and deployment release. When a question contains severance, new-regime, old-regime, and calculation/comparison cues together, the retrieval pipeline deterministically appends legal search terms for the Labor Pension Act, Labor Standards Act, service years, average wage, and the six-month cap. The expansion is used only by BM25, dense retrieval, and the reranker; the generator still receives the visitor's original question so retrieval assistance cannot rewrite the user's intent.

All four cue groups are required, so ordinary severance, retirement, or single-regime questions are not broadly rewritten. The `v0.1.0` formal model-quality baseline, `v0.3.1` reliability evidence, and `v0.3.2` provider safety cross-check retain their original evidence versions; this release did not use new provider calls to rewrite historical metrics.

## v0.3.2 provider safety cross-check: reliability, provenance, and dual-model runtime

This is the `v0.3.2` source-only runtime and deployment release. The public API/UI defaults to Gemini `gemini-3.5-flash-lite`. When OpenAI is also configured on the server, a user may select `gpt-5.6-luna` per request. The model names can be overridden independently with server-side `GEMINI_GENERATION_MODEL` and `OPENAI_GENERATION_MODEL`. When its key is configured, `LLM_PROVIDER=gemini` controls the default for a request that omits a provider; otherwise the API uses the other configured public provider. `LLM_FALLBACK_ENABLED=true` permits fallback. `GEMINI_API_KEY` and `OPENAI_API_KEY` remain only in the API server environment: the UI neither accepts, stores, nor displays them.

The fallback boundary is fixed: only an operational failure of the primary provider—such as transport failure, rate limiting, a 5xx service response, or an empty response—may trigger at most one attempt through the other configured public provider. Retrieval-layer refusal does not call a generator. A model refusal based on the retrieved law, a provider safety block, or a policy rejection never falls back. The formal evaluation path continues to bind directly to one generator and one judge provider with runtime fallback off, so routing changes cannot silently change the evaluated configuration.

The Streamlit sidebar's **Answer model** selector shows only configured Gemini/OpenAI entries returned by API `/models`; the selected provider is sent with each `/query`. In a query response, `requested_provider` records the requested route, `provider` and `model` are metadata for the model that actually generated the answer, `fallback_used`/`fallback_from` describe rerouting, and `generation_called=false` means retrieval refused before generation. The UI displays requested and actual models separately and warns when fallback occurred. Live provider smoke tests require local server-side secrets and are outside public offline CI.

The `v0.1.0` formal model-quality metrics remain historical results produced by the generator and judge models recorded in `release/manifest.json`; this runtime release has not replaced or independently re-judged those values. It did rerun retrieval and threshold behavior against both the 60-question stress suite and the 40-question formal set as a regression guard, without calling a provider.

The private BYOK Space notes moved to [private-demo.en.md](private-demo.en.md).
