# Private demo

[繁體中文](private-demo.md) | [English](private-demo.en.md)

How the hosted demo works, moved unchanged from [README.en.md](../README.en.md). The full operating procedure is in the [BYOK Hugging Face runbook](deployment/BYOK_HUGGINGFACE_RUNBOOK.md).

## Private BYOK Docker Space (invitation only)

**Demo status:** the private Space is running for the owner and invited reviewers; its entry point is not listed publicly.

The private Space uses BYOK (Bring Your Own Key). An invited reviewer selects Gemini `gemini-3.5-flash-lite` or OpenAI `gpt-5.6-luna` and enters a dedicated key in a masked field. The key exists only in the current Streamlit session, one loopback request header, and one request-scoped provider client. It is never written to files, chat history, shared settings, or cross-request caches. The Space has no owner `GEMINI_API_KEY` or `OPENAI_API_KEY` and performs no cross-provider fallback, so invited users cannot spend the owner's model-token balance.

The Space receives a collection-scoped read-only Qdrant key. A temporary write/manage key is revoked immediately after the two collections are built locally. Startup scrolls payloads read-only and rebuilds the structure/fixed BM25 indexes in memory; private `data/raw/` and `storage/bm25_*.json` artifacts are not shipped. Defaults are 20 queries per demo session, two concurrent queries globally, a 60-second provider timeout, and at most 1,000 unexpired anonymous sessions. Key isolation, read-only access, and free `cpu-basic` acceptance have passed. See the [BYOK Hugging Face runbook](deployment/BYOK_HUGGINGFACE_RUNBOOK.md).

Manual index maintenance (attended blue-green Qdrant rebuild) is described in the Chinese page: [private-demo.md](private-demo.md#人工更新-qdrant-法規索引).
