"""Run a capped Gemini/OpenAI generation cross-check over audited retrieval.

The first phase calls exactly five questions per provider. Expansion starts
only after that provider has reported complete token usage within its US$5
authorization. Raw questions and answers stay under ignored ``eval/runs``;
``--export-official`` writes only the content-free public schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import _bootstrap  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lib  # noqa: E402
from run_reliability_eval import (  # noqa: E402
    _build_indexes,
    _materialize_audited_corpus,
)

from rag.config import Settings  # noqa: E402
from rag.factory import build_retrieval_pipeline  # noqa: E402
from rag.generation.answerer import Answerer  # noqa: E402
from rag.generation.llm import (  # noqa: E402
    ProviderOperationalError,
    ProviderPolicyError,
    build_llm,
)
from rag.generation.prompts import (  # noqa: E402
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag.provider_budget import BudgetLedger  # noqa: E402
from rag.provider_crosscheck import (  # noqa: E402
    BudgetSafetyError,
    compute_provider_metrics,
    generate_with_budget,
    privacy_reduced_provider_trace,
    resolve_private_run_dir,
    select_crosscheck_rows,
    validate_request_maxima,
)
from rag.retrieval.refusal_policy import decide_retrieval_refusal  # noqa: E402
from rag.retrieval.reranker import Reranker  # noqa: E402

DATASET = PROJECT_ROOT / "eval" / "dataset" / "reliability_stress_v0.3.1.jsonl"
RELIABILITY_TRACE = PROJECT_ROOT / "eval" / "official" / "reliability_trace.jsonl"
SNAPSHOT = PROJECT_ROOT / "release" / "corpus_snapshot.json"
OFFICIAL_DIR = PROJECT_ROOT / "eval" / "official"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
AUTHORIZED_CAP_USD = "5.00"
PROVIDERS = {
    "gemini": {
        "model": "gemini-3.5-flash-lite",
        "input_per_million": "0.30",
        "output_per_million": "2.50",
        "pricing_url": "https://ai.google.dev/gemini-api/docs/pricing",
    },
    "openai": {
        "model": "gpt-5.6-luna",
        "input_per_million": "0.20",
        "output_per_million": "1.20",
        "pricing_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    },
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _credential_settings(env_file: Path | None) -> Settings:
    settings = Settings(_env_file=env_file) if env_file else Settings()
    missing = [
        provider
        for provider, key in (
            ("gemini", settings.gemini_api_key),
            ("openai", settings.openai_api_key),
        )
        if not key.strip()
    ]
    if missing:
        raise RuntimeError("missing provider credentials: " + ", ".join(missing))
    return settings


def _raw_result(
    *,
    row: dict,
    provider: str,
    generation,
    retrieval,
    charge: Decimal,
    elapsed_ms: float,
) -> dict:
    refused = REFUSAL_PHRASE in generation.text
    sources = (
        []
        if refused
        else Answerer._parse_sources(generation.text, retrieval.hits)
    )
    return {
        "qid": row["qid"],
        "question": row["question"],
        "answerable": row["answerable"],
        "gold_answer": row["answer"],
        "gold_sources": row["sources"],
        "requested_provider": provider,
        "actual_provider": generation.provider,
        "model": generation.model,
        "answer": generation.text,
        "refused": refused,
        "citation_count": len(sources),
        "cited_sources": [
            {"doc": source["doc"], "article": source["article"]}
            for source in sources
        ],
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "estimated_cost_usd": str(charge),
        "elapsed_ms": elapsed_ms,
    }


def _require_generation_admission(
    retrieval,
    settings: Settings,
    *,
    qid: str,
    reranker_enabled: bool,
) -> None:
    decision = decide_retrieval_refusal(
        has_hits=bool(retrieval.hits),
        reranker_enabled=reranker_enabled,
        applied_routes=retrieval.applied_routes,
        top_score=retrieval.top_score,
        global_threshold=settings.rerank_score_threshold,
    )
    if decision.refusal_stage is not None:
        raise RuntimeError(f"selected row no longer reaches generation: {qid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--reliability-trace", type=Path, default=RELIABILITY_TRACE)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--initial-per-provider", type=int, choices=[5], default=5)
    parser.add_argument("--max-per-provider", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=20_000)
    parser.add_argument("--max-output-tokens", type=int, default=1_024)
    parser.add_argument("--gemini-cap-usd", default="0")
    parser.add_argument("--openai-cap-usd", default="0")
    parser.add_argument("--export-official", action="store_true")
    args = parser.parse_args()

    max_input_tokens, max_output_tokens = validate_request_maxima(
        args.max_input_tokens,
        args.max_output_tokens,
    )
    credentials = _credential_settings(args.env_file)
    caps = {
        "gemini": args.gemini_cap_usd,
        "openai": args.openai_cap_usd,
    }
    ledgers = {
        provider: BudgetLedger(
            cap_usd=caps[provider],
            authorized_cap_usd=AUTHORIZED_CAP_USD,
            input_per_million=config["input_per_million"],
            output_per_million=config["output_per_million"],
        )
        for provider, config in PROVIDERS.items()
    }
    for ledger in ledgers.values():
        ledger.can_start(
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

    dataset = lib.load_dataset(args.dataset)
    reliability = _load_jsonl(args.reliability_trace)
    initial, expansion = select_crosscheck_rows(
        dataset,
        reliability,
        initial_count=args.initial_per_provider,
        maximum_count=args.max_per_provider,
    )
    selected = [*initial, *expansion]

    run_dir = resolve_private_run_dir(
        args.work_dir,
        RUNS_DIR,
        f"{datetime.now():%Y%m%d-%H%M%S}-provider-crosscheck",
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"work directory must be absent or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    committed = json.loads(args.snapshot.read_text(encoding="utf-8"))
    corpus_dir = _materialize_audited_corpus(run_dir, committed)
    credential_overrides = {
        f"{provider}_api_key": getattr(credentials, f"{provider}_api_key")
        for provider in PROVIDERS
    }
    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        qdrant_path=str(run_dir / "qdrant"),
        storage_dir=run_dir / "storage",
        data_dir=run_dir / "data",
        collection_name="provider_crosscheck_laws",
        device=args.device,
        chunking_strategy="structure",
        retrieval_mode="hybrid",
        use_reranker=True,
        **credential_overrides,
    )
    embedder, store = _build_indexes(settings, corpus_dir)
    raw_rows: list[dict] = []
    public_rows: list[dict] = []
    statuses = {provider: {"status": "ready", "reason": None} for provider in PROVIDERS}
    try:
        reranker = Reranker(
            model_name=settings.reranker_model,
            model_revision=settings.reranker_model_revision,
            device=settings.device,
        )
        pipeline = build_retrieval_pipeline(
            settings,
            embedder,
            store,
            strategy="structure",
            mode="hybrid",
            use_reranker=True,
            reranker=reranker,
        )
        retrieval_by_qid = {}
        for index, row in enumerate(selected, start=1):
            retrieval = pipeline.run(row["question"])
            _require_generation_admission(
                retrieval,
                settings,
                qid=row["qid"],
                reranker_enabled=pipeline.reranker is not None,
            )
            retrieval_by_qid[row["qid"]] = retrieval
            print(f"[retrieve] {index}/{len(selected)} {row['qid']}", flush=True)

        adapters = {
            provider: build_llm(
                settings,
                provider=provider,
                model=config["model"],
                timeout_seconds=120.0,
            )
            for provider, config in PROVIDERS.items()
        }
        raw_path = run_dir / "provider_crosscheck_raw.jsonl"

        def run_phase(rows: list[dict], phase: str) -> None:
            for provider, adapter in adapters.items():
                if statuses[provider]["status"] == "stopped":
                    continue
                ledger = ledgers[provider]
                for row in rows:
                    retrieval = retrieval_by_qid[row["qid"]]
                    try:
                        generation, charge, elapsed_ms = generate_with_budget(
                            adapter,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=build_user_prompt(
                                row["question"],
                                retrieval.hits,
                            ),
                            ledger=ledger,
                            max_input_tokens=max_input_tokens,
                            max_output_tokens=max_output_tokens,
                        )
                    except BudgetSafetyError as exc:
                        statuses[provider] = {
                            "status": "stopped",
                            "reason": exc.reason_code,
                        }
                        break
                    except ProviderPolicyError:
                        statuses[provider] = {
                            "status": "stopped",
                            "reason": "provider_policy_rejection",
                        }
                        break
                    except ProviderOperationalError as exc:
                        statuses[provider] = {
                            "status": "stopped",
                            "reason": exc.reason_code,
                        }
                        break
                    raw = _raw_result(
                        row=row,
                        provider=provider,
                        generation=generation,
                        retrieval=retrieval,
                        charge=charge,
                        elapsed_ms=elapsed_ms,
                    )
                    reduced = privacy_reduced_provider_trace(raw)
                    raw_rows.append(raw)
                    public_rows.append(reduced)
                    _write_jsonl(raw_path, raw_rows)
                    print(
                        f"[{phase}] {provider} {row['qid']} "
                        f"cost=${charge} total=${ledger.spent_usd}",
                        flush=True,
                    )
                if statuses[provider]["status"] != "stopped":
                    statuses[provider] = {"status": "complete", "reason": None}

        run_phase(initial, "initial")
        for provider in statuses:
            if (
                statuses[provider]["status"] == "complete"
                and ledgers[provider].requests != len(initial)
            ):
                statuses[provider] = {
                    "status": "stopped",
                    "reason": "incomplete_initial_batch",
                }
        if expansion:
            run_phase(expansion, "expand")

        metrics = compute_provider_metrics(public_rows) if public_rows else {}
        results = {
            "schema_version": "1.0",
            "run_date": date.today().isoformat(),
            "dataset": {
                "path": "eval/dataset/reliability_stress_v0.3.1.jsonl",
                "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            },
            "corpus_snapshot": {
                "path": "release/corpus_snapshot.json",
                "snapshot_date": committed["snapshot_date"],
                "laws": committed["law_count"],
                "articles": committed["article_count"],
            },
            "selection": {
                "initial_per_provider": len(initial),
                "maximum_per_provider": args.max_per_provider,
                "generation_eligible_only": True,
            },
            "authorization": {
                "per_provider_cap_usd": AUTHORIZED_CAP_USD,
                "max_input_tokens_per_request": max_input_tokens,
                "max_output_tokens_per_request": max_output_tokens,
            },
            "pricing": {
                provider: {
                    "model": config["model"],
                    "input_per_million_usd": config["input_per_million"],
                    "output_per_million_usd": config["output_per_million"],
                    "source": config["pricing_url"],
                }
                for provider, config in PROVIDERS.items()
            },
            "provider_status": statuses,
            "provider_metrics": metrics,
            "budget_ledgers": {
                provider: ledger.public_summary()
                for provider, ledger in ledgers.items()
            },
            "privacy": {
                "public_trace_contains_question_or_answer": False,
                "public_trace_contains_provider_payload": False,
                "public_trace_contains_credentials": False,
                "raw_trace_path": "ignored eval/runs only",
            },
        }
        _write_json(run_dir / "results.json", results)
        completed = all(
            ledger.requests >= len(initial) for ledger in ledgers.values()
        )
        if args.export_official and completed:
            _write_json(OFFICIAL_DIR / "provider_crosscheck_results.json", results)
            _write_jsonl(
                OFFICIAL_DIR / "provider_crosscheck_trace.jsonl",
                public_rows,
            )
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        print(f"[done] raw artifacts: {run_dir}", flush=True)
        return 0 if completed else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
