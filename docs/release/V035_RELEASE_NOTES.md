# v0.3.5 — Portfolio readiness

## Highlights

- Reworked the private Streamlit demo into a reviewer-first BYOK journey with clear capability, privacy, cost, citation, and refusal boundaries.
- Added a ten-case deterministic offline portfolio regression. All six answerable source contracts and all ten routing/retrieval-stage contracts pass with zero provider calls.
- Added a content-free article-level freshness artifact for 15 labor instruments and 884 non-deleted articles. The manual audit reports named law/source and article-label changes without publishing legal text.
- Preserved the v0.3.4 Qdrant candidate pair and the existing retrieval configuration: structure-aware/fixed collections, hybrid BM25+dense retrieval, reranker, RRF `k=60`, and threshold `0.03`.

## Evidence boundary

The 40-question formal benchmark, 60-case reliability suite, and archived provider judgments are unchanged. The compact portfolio regression demonstrates source rank, deterministic routes, retrieval-stage refusal, and correct passage to the later citation-completeness boundary; it does not execute or score generated prose. The official artifact is content-free and calls no Gemini, OpenAI, or paid hardware.

The article snapshot contains only law names, article labels, and SHA-256 fingerprints. It contains no legal text, chapters, URLs, credentials, endpoints, provider payloads, or local paths. Freshness remains an attended manual operation: no scheduler, heartbeat, cron job, or automatic writer exists.

## Private demo and cost posture

- Hugging Face Space remains private, on one free `cpu-basic` replica, with no persistent paid storage or accelerator.
- Visitors provide their own Gemini or OpenAI API key. The Space has no owner LLM key and cross-provider fallback is disabled.
- Qdrant remains Free Tier. Runtime access is collection-scoped and read-only; temporary writer and transition keys are revoked after attended maintenance.
- The private Space deployment receipt is bound before the immutable tag is created. The public repository does not disclose its URL, endpoint, key suffixes, session tokens, or account identity.

## Known limitations

- The system covers the audited 15-instrument corpus; it is not a general legal database or legal advice service.
- The `0.03` reranker threshold is calibrated evidence, not a universal answerability classifier. The reliability suite retains one measured direct false refusal among 40 answerable stress cases.
- Two semantically near unanswerable portfolio cases correctly reach the later LLM/citation-completeness boundary; the offline artifact does not claim to have judged a final model refusal.
- The full corpus, model weights, private indexes, raw provider runs, generated answers, and judge reasons are outside the public source release.

## Verification and rollback

Run `uv run python scripts/verify_release.py` to recompute the public evidence and publication boundary without loading models, contacting providers, or starting Qdrant. A failed private deployment stays private and is rolled back by restoring the previous Space revision and collection base; old collections are not deleted automatically. No paid tier is enabled as a fallback.
