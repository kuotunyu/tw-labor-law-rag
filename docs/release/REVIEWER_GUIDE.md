# Clean Reviewer Guide

This path verifies the public source-only portfolio release without calling a model/provider, downloading model weights, starting Qdrant or Docker, using a GPU, or entering the FastAPI lifespan.

## Preconditions

- A clean checkout of `main` or a GitHub-generated source archive
- Python 3.11
- `uv`
- Enough disk space for the locked Python dependencies

Dependency installation may access the package indexes recorded by the project. The verification commands after installation require no runtime network service. Do not provide API keys; a clean checkout should have no `.env`.

## Required commands

```powershell
uv sync --locked
uv lock --check
uv run ruff check .
uv run python scripts/verify_release.py
uv run pytest -q
uv build --out-dir "$env:TEMP\labor-rag-review-build"
uv run python -W error::UserWarning -c "import sys; sys.path.insert(0, 'src'); import rag; import rag.api.main; print('FastAPI import: ok')"
uv run python -W error::UserWarning scripts/ask.py --help
git status --short
```

`scripts/ask.py --help` exits in argument parsing before settings, indexes, models, or providers are initialized. Importing `rag.api.main` constructs only module-level schemas and an empty state holder; the lifespan that creates heavy components is not entered.

`tests/test_packaging.py` performs the distribution-content check: it builds both sdist and wheel in a temporary pytest directory and verifies that `rag/indexing/dict/legal_terms.txt` is present. It also performs a warning-clean FastAPI import in a fresh subprocess.

## Expected evidence summary

### v0.3.2 provider safety cross-check

Gemini `gemini-3.5-flash-lite` and OpenAI `gpt-5.6-luna` both completed five requests. Gemini observed refusal accuracy `0.8`, citation success `1.0`, and estimated cost `US$0.0022620`; OpenAI observed refusal accuracy `1.0`, citation success `1.0`, and estimated cost `US$0.0026414`. This safety cross-check does not replace the `v0.1.0` formal evidence baseline or constitute a formal model-quality evaluation. The public trace is strictly content-free and contains no question/answer text, provider payload, or credentials.

The release verifier should exit zero and report:

- dataset: 40 questions, 30 answerable, 10 unanswerable
- canonical dataset SHA-256: `760e33eaa0821001d37ff974bc037043d019fc670b8f3621b6e713030274ca07`
- ablation: 8 configurations, 320 rows
- primary retrieval: Hit@5 `0.9666666666666667`, MRR@10 `0.9055555555555554`
- reliability stress: 60 questions, Hit@5 `0.95`, MRR@10 `0.9083333333333334`, 1 direct false refusal, 0.85 direct unanswerable coverage, decision `retain_0.03`
- end-to-end: 29 answered, 11 refused, 31 generation calls recorded
- refusal stages: 9 threshold, 2 LLM, 0 no-hits
- threshold score/stage contract: true for all 40 rows at gate `0.03`
- provider evidence: 29 archived numeric verdicts; faithfulness `4.896551724137931`, relevancy `5.0`
- provider cross-check: complete, five requests per provider under the authorized US$5.00 ceiling; Gemini refusal/citation `0.8`/`1.0`, cost `US$0.0022620`; OpenAI `1.0`/`1.0`, `US$0.0026414`
- OGDL source samples verified: 2
- full corpus snapshot: 2026-08-29, 15 laws, 884 non-deleted articles
- GitHub Action references: 2, both full commit SHAs
- publication inventory: exact match to `release/public-files.txt`, empty `tracked_excluded`, 0 unexpected archive files, and only manifest-reviewed binary hashes
- public Git history: all commits reachable from heads/tags/ordinary remotes pass identity and historical path/content/binary scanning; ephemeral `refs/remotes/pull/*` merge refs and local `refs/archive/*` recovery refs are excluded
- locked Ruff dependency and CI lint/tag gates: verified
- dependency audit: no known PyPI vulnerabilities; custom CUDA `torch` wheel is reported separately because it is not present on PyPI
- official trace issues: 0
- public scan issues: 0

Provider evidence values are re-aggregated historical verdicts, not regenerated judgments.

## Optional Git audit

```powershell
git branch --show-current
git rev-list --branches --tags --exclude=pull/* --remotes --count
git remote get-url origin
git config user.name
git config user.email
git log --format="%H %an <%ae> %cn <%ce> %s" --branches --tags --exclude=pull/* --remotes
git ls-files
```

The branch under review must be an intended publishable branch, `origin` is `https://github.com/kuotunyu/tw-labor-law-rag.git`, and every commit reachable from heads/tags/ordinary remotes must use `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for both author and committer. The verifier reports the audited namespaces and deduplicated commit count; ephemeral `refs/remotes/pull/*` synthetic merge refs and local `refs/archive/*` recovery refs are outside that publication graph.

## Interpretation

A green run proves internal consistency of the committed public contract and build/import path. It does not prove current legal correctness of the underlying laws, universal retrieval performance, deterministic provider judgments, or safety for legal reliance. The system is a portfolio artifact, not legal advice.
