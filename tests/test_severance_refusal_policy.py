import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag.severance_refusal_policy as policy
from eval import run_severance_refusal_policy as runner
from rag import factory
from rag.indexing import embedder as embedder_module
from rag.retrieval import reranker as reranker_module
from rag.retrieval.pipeline import plan_retrieval_query
from rag.severance_refusal_policy import (
    build_case_observation,
    build_official_artifact,
    evaluate_candidate,
    load_cases,
    select_highest_passing_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "eval/dataset/severance_refusal_policy_v0.3.6.jsonl"
RUNNER = PROJECT_ROOT / "eval/run_severance_refusal_policy.py"
STRESS_DATASET = PROJECT_ROOT / "eval/dataset/reliability_stress_v0.3.1.jsonl"
EXPECTED_QIDS = (
    "severance-policy-001",
    "severance-policy-002",
    "severance-policy-003",
    "severance-policy-004",
    "severance-policy-005",
    "severance-policy-006",
    "severance-policy-007",
    "severance-policy-008",
    "severance-policy-009",
    "severance-policy-010",
    "severance-policy-011",
    "severance-policy-012",
    "severance-policy-013",
    "severance-policy-014",
    "severance-policy-015",
    "severance-policy-016",
    "severance-policy-017",
    "severance-policy-018",
    "severance-policy-019",
    "severance-policy-020",
    "severance-policy-021",
    "severance-policy-022",
    "severance-policy-023",
    "severance-policy-024",
    "severance-policy-025",
    "severance-policy-026",
    "severance-policy-027",
    "severance-policy-028",
    "severance-policy-029",
    "severance-policy-030",
)
EXPECTED_CANDIDATES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
POSITIVE_STYLES = {
    "statutory_chinese",
    "colloquial_chinese",
    "code_switch",
    "punctuation",
    "long_narrative",
    "reversed_regime_order",
    "formula_wording",
    "cap_wording",
    "mixed_tenure",
}
COLLISION_STYLES = {
    "single_regime",
    "ordinary_termination",
    "notice_only",
    "wage_arrears",
    "generic_retirement",
    "unrelated_old_new",
    "partial_cue_collision",
}
SEVERANCE_SOURCES = (
    {"law": "勞工退休金條例", "article": "第 12 條"},
    {"law": "勞動基準法", "article": "第 17 條"},
)
SEVERANCE_SOURCE_KEYS = {
    "勞工退休金條例|第 12 條",
    "勞動基準法|第 17 條",
}
OBSERVATION_FIELDS = {
    "qid",
    "source_ranks",
    "applied_routes",
    "hit_count",
    "top_score",
}
OFFICIAL_CASE_FIELDS = {
    "qid",
    "case_type",
    "answerable",
    "source_ranks",
    "applied_routes",
    "hit_count",
    "top_score",
    "effective_threshold",
    "refused",
    "refusal_stage",
    "source_contract_passed",
    "route_contract_passed",
    "generation_expected",
    "generation_allowed",
    "generation_contract_passed",
    "passed",
}
FORMAL_RANKS = [
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    None,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
STRESS_ROUTES = {
    "stress-003": ["severance_comparison"],
    "stress-010": ["wage_arrears_termination"],
    "stress-037": ["severance_comparison"],
    "stress-038": ["wage_arrears_termination"],
}
FORMAL_ROUTES = {
    "eval-03": ["severance_comparison"],
    "eval-10": ["wage_arrears_termination"],
}


@pytest.fixture(scope="module")
def cases():
    return load_cases(DATASET)


def test_offline_runner_exposes_complete_cli_without_loading_models() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0, stderr
    for option in (
        "--dataset",
        "--stress-dataset",
        "--formal-dataset",
        "--snapshot",
        "--work-dir",
        "--offline",
        "--device",
        "--export-official",
    ):
        assert option in stdout


def test_offline_flag_precedes_every_hugging_face_import_snapshot(
    tmp_path,
) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """import builtins
import os

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'huggingface_hub', 'transformers', 'FlagEmbedding'}:
        if os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            raise RuntimeError('TRANSFORMERS_OFFLINE was not forced before import')
        if os.environ.get('HF_HUB_OFFLINE') != '1':
            raise RuntimeError('HF_HUB_OFFLINE was not forced before import')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("TRANSFORMERS_OFFLINE", None)
    environment.pop("HF_HUB_OFFLINE", None)
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--offline", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )


def test_runner_forces_offline_mode_before_cached_model_preflight(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    settings = SimpleNamespace(device="cpu")
    calls = []
    monkeypatch.setattr(runner, "Settings", lambda **kwargs: settings)
    monkeypatch.setattr(
        runner,
        "require_cached_models",
        lambda actual: calls.append(actual),
    )

    actual = runner._offline_preflight(SimpleNamespace(device="cpu"))

    assert actual is settings
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert calls == [settings]


def test_local_pipeline_forces_both_model_loaders_local_only_without_llm(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    settings = SimpleNamespace(
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_revision="b" * 40,
        device="cpu",
    )
    embedder = SimpleNamespace(close=lambda: None)
    store = SimpleNamespace(close=lambda: None)
    pipeline = object()

    monkeypatch.setattr(
        runner,
        "Settings",
        lambda **kwargs: calls.append(("settings", kwargs)) or settings,
    )
    monkeypatch.setattr(
        embedder_module,
        "resolve_device",
        lambda requested: calls.append(("resolve_device", requested)) or "cpu",
    )
    monkeypatch.setattr(
        runner,
        "_build_indexes",
        lambda actual_settings, corpus_dir, *, local_files_only: (
            calls.append(
                (
                    "indexes",
                    actual_settings,
                    corpus_dir,
                    local_files_only,
                )
            )
            or (embedder, store)
        ),
    )
    monkeypatch.setattr(
        reranker_module,
        "Reranker",
        lambda **kwargs: calls.append(("reranker", kwargs)) or object(),
    )
    monkeypatch.setattr(
        factory,
        "build_retrieval_pipeline",
        lambda *_args, **kwargs: calls.append(("retrieval", kwargs)) or pipeline,
    )
    monkeypatch.setattr(
        factory,
        "build_llm",
        lambda *_args, **_kwargs: pytest.fail("LLM construction is forbidden"),
    )

    result = runner._build_local_pipeline(
        SimpleNamespace(device="auto"), tmp_path / "work", tmp_path / "corpus"
    )

    assert result == (settings, embedder, store, pipeline)
    assert ("resolve_device", "auto") in calls
    assert next(call for call in calls if call[0] == "settings")[1]["device"] == "cpu"
    assert next(call for call in calls if call[0] == "indexes")[3] is True
    assert next(call for call in calls if call[0] == "reranker")[1][
        "local_files_only"
    ] is True
    assert [call[0] for call in calls].count("retrieval") == 1


def test_runner_rejects_nonempty_work_directory_before_retrieval(tmp_path) -> None:
    work_dir = tmp_path / "occupied"
    work_dir.mkdir()
    (work_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        runner._prepare_work_dir(work_dir)


def test_runner_retrieves_each_target_once_and_records_unrounded_observation(
    cases,
) -> None:
    by_question = {case.question: case for case in cases}
    calls = []

    def retrieve(question):
        calls.append(question)
        case = by_question[question]
        hits = [
            SimpleNamespace(
                payload={"doc_title": source["law"], "articles": [source["article"]]}
            )
            for source in case.sources
        ] or [SimpleNamespace(payload={"doc_title": "irrelevant", "articles": []})]
        return SimpleNamespace(
            hits=hits,
            applied_routes=plan_retrieval_query(question).routes,
            top_score=0.0150004,
        )

    observations = runner._run_target_cases(
        SimpleNamespace(run=retrieve), cases
    )

    assert calls == [case.question for case in cases]
    assert len(observations) == 30
    assert observations[0] == {
        "qid": "severance-policy-001",
        "source_ranks": {
            "勞動基準法|第 17 條": 2,
            "勞工退休金條例|第 12 條": 1,
        },
        "applied_routes": ["severance_comparison"],
        "hit_count": 2,
        "top_score": 0.0150004,
    }


def test_runner_builds_guard_rows_from_fresh_scores_and_authoritative_routes() -> None:
    row = json.loads(STRESS_DATASET.read_text(encoding="utf-8").splitlines()[2])
    expected_source = row["sources"][0]
    retrieval = SimpleNamespace(
        hits=[
            SimpleNamespace(payload={"doc_title": "other", "articles": []}),
            SimpleNamespace(
                payload={
                    "doc_title": expected_source["doc"],
                    "articles": [expected_source["article"]],
                }
            ),
        ],
        applied_routes=("severance_comparison",),
        top_score=0.0175004,
    )

    evidence = runner._run_guard_cases(
        SimpleNamespace(run=lambda _question: retrieval), [row]
    )

    assert evidence == [
        {
            "qid": "stress-003",
            "answerable": True,
            "rank": 2,
            "hit_count": 2,
            "top_score": 0.0175004,
            "applied_routes": ["severance_comparison"],
        }
    ]


def test_runner_rejects_guard_route_disagreement_instead_of_replacing_it() -> None:
    row = json.loads(STRESS_DATASET.read_text(encoding="utf-8").splitlines()[2])
    retrieval = SimpleNamespace(
        hits=[SimpleNamespace(payload={"doc_title": "other", "articles": []})],
        applied_routes=(),
        top_score=0.0175004,
    )

    with pytest.raises(RuntimeError, match="route mismatch.*stress-003"):
        runner._run_guard_cases(
            SimpleNamespace(run=lambda _question: retrieval), [row]
        )


def test_runner_sweeps_every_candidate_over_the_same_retrieval_evidence(
    monkeypatch,
) -> None:
    observations = [{"retrieval": "target"}]
    stress = [{"retrieval": "stress"}]
    formal = [{"retrieval": "formal"}]
    calls = []

    def evaluate(actual_observations, **kwargs):
        calls.append((actual_observations, kwargs))
        return {"candidate_threshold": kwargs["candidate_threshold"]}

    monkeypatch.setattr(runner, "evaluate_candidate", evaluate)

    results = runner._evaluate_candidates(observations, stress, formal)

    assert [result["candidate_threshold"] for result in results] == list(
        EXPECTED_CANDIDATES
    )
    assert len({id(call[0]) for call in calls}) == 1
    assert all(call[0] is observations for call in calls)
    assert all(call[1]["stress_rows"] is stress for call in calls)
    assert all(call[1]["formal_rows"] is formal for call in calls)
    assert all(call[1]["global_threshold"] == 0.03 for call in calls)


def test_runner_rebuilds_identical_content_free_accepted_artifact(cases) -> None:
    kwargs = {
        "observations": _observations(cases),
        "stress_rows": _stress_rows(),
        "formal_rows": _formal_rows(),
        "provenance": _provenance(),
    }

    first = runner._build_accepted_artifact(**kwargs)
    second = runner._build_accepted_artifact(**kwargs)

    assert first == second
    assert first["schema_version"] == "1.2"
    assert first["selected_threshold"] == 0.015
    serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for forbidden_key in (
        '"question"',
        '"content"',
        '"answer"',
        '"endpoint"',
        '"url"',
        '"credential"',
        '"secret"',
        '"api_key"',
    ):
        assert forbidden_key not in serialized


def test_runner_reports_no_go_and_builds_no_artifact_when_any_gate_fails(
    cases,
) -> None:
    observations = _observations(cases)
    observations[0] = {
        **observations[0],
        "source_ranks": {},
        "top_score": 0.0150004,
    }

    with pytest.raises(RuntimeError, match="NO-GO"):
        runner._build_accepted_artifact(
            observations=observations,
            stress_rows=_stress_rows(),
            formal_rows=_formal_rows(),
            provenance=_provenance(),
        )


def test_no_go_envelope_is_content_free_unrounded_and_replayable(cases) -> None:
    observations = _observations(cases)
    observations[0] = {
        **observations[0],
        "source_ranks": {},
        "top_score": 0.0150004,
    }
    candidates = _candidate_results(observations)

    envelope = policy.build_no_go_evidence(
        observations=observations,
        candidate_results=candidates,
        provenance=_provenance(),
        expected_threshold=0.015,
    )

    assert envelope["schema_version"] == "1.0"
    assert envelope["evidence_class"] == "non_release_no_go"
    assert envelope["outcome"] == "no_go"
    assert envelope["official_export_allowed"] is False
    assert envelope["selected_threshold"] is None
    assert envelope["target_observations"] == observations
    assert envelope["target_observations"][0]["top_score"] == 0.0150004
    assert len(envelope["guard_evidence"]["stress"]) == 60
    assert len(envelope["guard_evidence"]["formal"]) == 40
    assert len(envelope["candidates"]) == 7
    assert envelope["failed_gates"] == [
        {"candidate_threshold": 0.0, "gates": ["target"]},
        {"candidate_threshold": 0.005, "gates": ["target"]},
        {"candidate_threshold": 0.01, "gates": ["target"]},
        {"candidate_threshold": 0.015, "gates": ["target"]},
        {"candidate_threshold": 0.02, "gates": ["target", "stress"]},
        {"candidate_threshold": 0.025, "gates": ["target", "stress"]},
        {"candidate_threshold": 0.03, "gates": ["target", "stress"]},
    ]
    assert policy.replay_no_go_evidence(envelope) == envelope
    serialized = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
    for forbidden_key in (
        '"question"',
        '"content"',
        '"answer"',
        '"endpoint"',
        '"url"',
        '"credential"',
        '"secret"',
        '"api_key"',
    ):
        assert forbidden_key not in serialized


def test_no_go_replay_rejects_mutated_candidate_aggregate(cases) -> None:
    observations = _observations(cases)
    observations[0] = {
        **observations[0],
        "source_ranks": {},
        "top_score": 0.0150004,
    }
    envelope = policy.build_no_go_evidence(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=_provenance(),
        expected_threshold=0.015,
    )
    mutated = json.loads(json.dumps(envelope))
    mutated["candidates"][0]["target"]["passed_cases"] = 28

    with pytest.raises(ValueError, match="replay mismatch"):
        policy.replay_no_go_evidence(mutated)


def test_runner_public_json_export_is_deterministic(tmp_path, cases) -> None:
    artifact = runner._build_accepted_artifact(
        observations=_observations(cases),
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
        provenance=_provenance(),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    runner._write_public_json(first, artifact)
    runner._write_public_json(second, artifact)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert first.read_text(encoding="utf-8") == json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def test_runner_binds_exact_input_hashes_models_settings_and_zero_providers(
    tmp_path,
) -> None:
    dataset = tmp_path / "target.jsonl"
    stress = tmp_path / "stress.jsonl"
    formal = tmp_path / "formal.jsonl"
    snapshot = tmp_path / "snapshot.json"
    for path, payload in (
        (dataset, b"target\n"),
        (stress, b"stress\n"),
        (formal, b"formal\n"),
        (snapshot, b"snapshot\n"),
    ):
        path.write_bytes(payload)
    args = SimpleNamespace(
        dataset=dataset,
        stress_dataset=stress,
        formal_dataset=formal,
        snapshot=snapshot,
    )
    settings = SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        embedding_model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        top_k_retrieve=20,
        top_k_final=5,
        rrf_k=60,
        device="cpu",
    )

    provenance = runner._build_provenance(args, settings, "a" * 40)

    assert provenance == {
        "dataset_sha256": hashlib.sha256(b"target\n").hexdigest(),
        "corpus_snapshot_sha256": hashlib.sha256(b"snapshot\n").hexdigest(),
        "source_artifact_sha256": {
            "stress_dataset": hashlib.sha256(b"stress\n").hexdigest(),
            "formal_dataset": hashlib.sha256(b"formal\n").hexdigest(),
        },
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "retrieval_configuration": {
            "chunking": "structure",
            "retrieval": "hybrid",
            "reranker": True,
            "top_k_retrieve": 20,
            "top_k_final": 5,
            "rrf_k": 60,
        },
        "execution_device": "cpu",
        "code_revision": "a" * 40,
        "run_origin": "fresh_offline_retrieval",
        "provider_adapters": 0,
        "provider_requests": 0,
    }


def test_runner_refuses_to_cite_a_dirty_candidate_source_revision(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=" M tracked.py\n"),
    )

    with pytest.raises(RuntimeError, match="clean tracked and untracked tree"):
        runner._clean_git_revision()


def test_runner_main_uses_one_retrieval_only_pipeline_and_exports_after_acceptance(
    monkeypatch,
    tmp_path,
) -> None:
    target = tmp_path / "target.jsonl"
    stress = tmp_path / "stress.jsonl"
    formal = tmp_path / "formal.jsonl"
    snapshot = tmp_path / "snapshot.json"
    for path in (target, stress, formal):
        path.write_text("{}\n", encoding="utf-8")
    snapshot.write_text("{}\n", encoding="utf-8")
    work_dir = tmp_path / "run"
    official = tmp_path / "official.json"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pipeline = object()
    closed = []
    store = SimpleNamespace(close=lambda: closed.append("store"))
    embedder = SimpleNamespace(close=lambda: closed.append("embedder"))
    settings = SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        embedding_model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        top_k_retrieve=20,
        top_k_final=5,
        rrf_k=60,
        device="cpu",
    )
    target_cases = [object()]
    stress_rows = [{"qid": "stress"}]
    formal_rows = [{"qid": "formal"}]
    observations = [{"qid": "target"}]
    stress_evidence = [{"qid": "stress-evidence"}]
    formal_evidence = [{"qid": "formal-evidence"}]
    artifact = {"selected_threshold": 0.015}
    calls = []

    monkeypatch.setattr(runner, "OFFICIAL_RESULT", official, raising=False)
    monkeypatch.setattr(runner, "_offline_preflight", lambda _args: settings)
    monkeypatch.setattr(
        runner,
        "_materialize_audited_corpus",
        lambda actual_work, committed: (
            calls.append(("materialize", actual_work, committed)) or corpus
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_build_local_pipeline",
        lambda _args, _work, actual_corpus: (
            calls.append(("pipeline", actual_corpus))
            or (settings, embedder, store, pipeline)
        ),
        raising=False,
    )
    monkeypatch.setattr(runner, "load_cases", lambda _path: target_cases, raising=False)
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda path: stress_rows if path == stress else formal_rows,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_run_target_cases",
        lambda actual_pipeline, actual_cases: (
            calls.append(("target", actual_pipeline, actual_cases)) or observations
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_guard_cases",
        lambda actual_pipeline, rows: (
            calls.append(("guard", actual_pipeline, rows))
            or (stress_evidence if rows is stress_rows else formal_evidence)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_build_accepted_artifact",
        lambda **kwargs: calls.append(("artifact", kwargs)) or artifact,
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_candidates",
        lambda actual_observations, actual_stress, actual_formal: (
            calls.append(
                (
                    "candidates",
                    actual_observations,
                    actual_stress,
                    actual_formal,
                )
            )
            or [{"candidate_threshold": 0.015}]
        ),
    )
    monkeypatch.setattr(
        runner, "_clean_git_revision", lambda: "a" * 40, raising=False
    )

    exit_code = runner.main(
        [
            "--dataset",
            str(target),
            "--stress-dataset",
            str(stress),
            "--formal-dataset",
            str(formal),
            "--snapshot",
            str(snapshot),
            "--work-dir",
            str(work_dir),
            "--offline",
            "--device",
            "cpu",
            "--export-official",
        ]
    )

    assert exit_code == 0
    assert json.loads((work_dir / "results.json").read_text(encoding="utf-8")) == artifact
    assert official.read_bytes() == (work_dir / "results.json").read_bytes()
    assert closed == ["store", "embedder"]
    assert ("target", pipeline, target_cases) in calls
    assert ("guard", pipeline, stress_rows) in calls
    assert ("guard", pipeline, formal_rows) in calls


def test_runner_main_persists_replayable_no_go_without_official_export(
    monkeypatch,
    tmp_path,
    cases,
) -> None:
    target = tmp_path / "target.jsonl"
    stress = tmp_path / "stress.jsonl"
    formal = tmp_path / "formal.jsonl"
    snapshot = tmp_path / "snapshot.json"
    for path in (target, stress, formal):
        path.write_text("{}\n", encoding="utf-8")
    snapshot.write_text("{}\n", encoding="utf-8")
    work_dir = tmp_path / "run"
    diagnostic = tmp_path / "diagnostics" / "no_go.json"
    official = tmp_path / "official.json"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pipeline = object()
    settings = SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        embedding_model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        top_k_retrieve=20,
        top_k_final=5,
        rrf_k=60,
        device="cpu",
    )
    observations = _observations(cases)
    observations[0] = {**observations[0], "source_ranks": {}}
    stress_evidence = _stress_rows()
    formal_evidence = _formal_rows()
    store = SimpleNamespace(close=lambda: None)
    embedder = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(runner, "OFFICIAL_RESULT", official)
    monkeypatch.setattr(runner, "_offline_preflight", lambda _args: settings)
    monkeypatch.setattr(runner, "_clean_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "_materialize_audited_corpus",
        lambda _work, _snapshot: corpus,
    )
    monkeypatch.setattr(
        runner,
        "_build_local_pipeline",
        lambda _args, _work, _corpus: (
            settings,
            embedder,
            store,
            pipeline,
        ),
    )
    monkeypatch.setattr(runner, "load_cases", lambda _path: cases)
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda path: stress_evidence if path == stress else formal_evidence,
    )
    monkeypatch.setattr(
        runner, "_run_target_cases", lambda _pipeline, _cases: observations
    )
    monkeypatch.setattr(
        runner,
        "_run_guard_cases",
        lambda _pipeline, rows: rows,
    )

    exit_code = runner.main(
        [
            "--dataset",
            str(target),
            "--stress-dataset",
            str(stress),
            "--formal-dataset",
            str(formal),
            "--snapshot",
            str(snapshot),
            "--work-dir",
            str(work_dir),
            "--diagnostics-output",
            str(diagnostic),
            "--offline",
            "--device",
            "cpu",
            "--export-official",
        ]
    )

    assert exit_code == 1
    assert not official.exists()
    envelope = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert policy.replay_no_go_evidence(envelope) == envelope
    assert envelope["provenance"]["code_revision"] == "a" * 40


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _observations(cases):
    observations = []
    for index, case in enumerate(cases):
        score = 0.015 if index == 0 else 0.5
        if case.qid == "severance-policy-023":
            score = 0.0
        observations.append(
            build_case_observation(
                case,
                source_ranks={
                    _source_key(source): rank
                    for rank, source in enumerate(case.sources, start=1)
                },
                applied_routes=plan_retrieval_query(case.question).routes,
                top_score=score,
                hit_count=0 if case.qid == "severance-policy-023" else 5,
            )
        )
    return observations


def _stress_rows():
    rows = []
    for number in range(1, 61):
        qid = f"stress-{number:03d}"
        answerable = number <= 40
        if qid == "stress-037":
            score = 0.0175004
        elif qid == "stress-041":
            score = 0.0
        elif not answerable and number <= 57:
            score = 0.0000004
        else:
            score = 0.5
        rows.append(
            {
                "qid": qid,
                "answerable": answerable,
                "rank": 1 if answerable else None,
                "hit_count": 0 if qid == "stress-041" else 5,
                "top_score": score,
                "applied_routes": STRESS_ROUTES.get(qid, []),
            }
        )
    return rows


def _formal_rows():
    rows = [
        {
            "qid": f"eval-{number:02d}",
            "answerable": True,
            "rank": rank,
            "hit_count": 5,
            "top_score": 0.5000004 if number == 3 else 0.5,
            "applied_routes": FORMAL_ROUTES.get(f"eval-{number:02d}", []),
        }
        for number, rank in enumerate(FORMAL_RANKS, start=1)
    ]
    rows.extend(
        {
            "qid": f"eval-{number:02d}",
            "answerable": False,
            "rank": None,
            "hit_count": 0 if number == 31 else 5,
            "top_score": (
                0.0
                if number == 31
                else 0.0000004 if number <= 39 else 0.5
            ),
            "applied_routes": [],
        }
        for number in range(31, 41)
    )
    return rows


def _candidate_results(observations, *, stress_rows=None, formal_rows=None):
    stress = _stress_rows() if stress_rows is None else stress_rows
    formal = _formal_rows() if formal_rows is None else formal_rows
    return [
        evaluate_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=formal,
        )
        for threshold in EXPECTED_CANDIDATES
    ]


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance():
    return {
        "dataset_sha256": "a" * 64,
        "corpus_snapshot_sha256": "b" * 64,
        "source_artifact_sha256": {
            "stress_dataset": "c" * 64,
            "formal_dataset": "d" * 64,
        },
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "retrieval_configuration": {
            "chunking": "structure",
            "retrieval": "hybrid",
            "reranker": True,
            "top_k_retrieve": 20,
            "top_k_final": 5,
            "rrf_k": 60,
        },
        "execution_device": "cpu",
        "code_revision": "f" * 40,
        "run_origin": "fresh_offline_retrieval",
        "provider_adapters": 0,
        "provider_requests": 0,
    }


def test_reviewed_dataset_has_exact_order_contracts_and_style_coverage(cases):
    assert tuple(case.qid for case in cases) == EXPECTED_QIDS
    assert [case.case_type for case in cases] == ["positive"] * 15 + [
        "collision_negative"
    ] * 15
    assert all(case.sources == SEVERANCE_SOURCES for case in cases[:15])
    assert all(case.required_routes == ("severance_comparison",) for case in cases[:15])
    assert POSITIVE_STYLES <= {tag for case in cases[:15] for tag in case.style_tags}
    assert COLLISION_STYLES <= {tag for case in cases[15:] for tag in case.style_tags}
    assert all(len(case.question) >= 12 and case.question[-1] in "？?" for case in cases)


def test_reviewed_questions_match_committed_route_semantics(cases):
    expected = [("severance_comparison",)] * 15
    expected.extend([()] * 4)
    expected.extend([("wage_arrears_termination",)])
    expected.extend([()] * 9)
    expected.extend(
        [("severance_comparison", "wage_arrears_termination")]
    )
    assert [plan_retrieval_query(case.question).routes for case in cases] == expected


def _dataset_rows():
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update(extra="drift"), "fields"),
        (lambda rows: rows[0].pop("question"), "fields"),
        (lambda rows: rows[1].update(qid=rows[0]["qid"]), "duplicate qids"),
        (lambda rows: rows[0].update(question=" "), "question"),
        (
            lambda rows: rows[0].update(sources=[rows[0]["sources"][0]] * 2),
            "duplicate source",
        ),
        (
            lambda rows: rows[0].update(
                required_routes=["severance_comparison"] * 2
            ),
            "duplicates",
        ),
        (lambda rows: rows[0].update(answerable=1), "answerable"),
        (lambda rows: rows[0].update(expect_generation=1), "expect_generation"),
        (lambda rows: rows[0].update(case_type="collision_negative"), "ordering"),
        (lambda rows: rows.pop(), "qids"),
        (
            lambda rows: (
                rows[22].update(
                    answerable=True,
                    sources=[{"law": "勞動基準法", "article": "第 17 條"}],
                ),
                rows[24].update(answerable=False, sources=[]),
            ),
            "canonical contract",
        ),
    ],
)
def test_loader_fails_closed_on_dataset_drift(tmp_path, mutate, message):
    rows = _dataset_rows()
    mutate(rows)
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_observation_contains_only_raw_content_free_evidence(cases):
    observation = build_case_observation(
        cases[0],
        source_ranks={
            "勞工退休金條例|第 12 條": 1,
            "勞動基準法|第 17 條": 5,
        },
        applied_routes=("severance_comparison",),
        top_score=0.0150004,
        hit_count=5,
    )
    assert set(observation) == OBSERVATION_FIELDS
    assert observation["top_score"] == 0.0150004


def test_target_no_hit_observation_uses_no_hits_policy_semantics(cases):
    observations = _observations(cases)
    observations[0] = build_case_observation(
        cases[0],
        source_ranks={},
        applied_routes=("severance_comparison",),
        top_score=0.0,
        hit_count=0,
    )

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    case = result["cases"][0]
    assert case["hit_count"] == 0
    assert case["refusal_stage"] == "no_hits"
    assert case["effective_threshold"] is None
    assert case["generation_allowed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_ranks": {"未知法規|第 1 條": 1}}, "source_ranks"),
        ({"applied_routes": ["severance_comparison"]}, "applied_routes"),
        (
            {"applied_routes": ("severance_comparison", "severance_comparison")},
            "duplicates",
        ),
        ({"applied_routes": ("private-route",)}, "allowlist"),
        ({"top_score": math.nan}, "top_score"),
        ({"top_score": math.inf}, "top_score"),
    ],
)
def test_observation_rejects_invalid_content_free_inputs(cases, kwargs, message):
    valid = {
        "source_ranks": {
            "勞工退休金條例|第 12 條": 1,
            "勞動基準法|第 17 條": 2,
        },
        "applied_routes": ("severance_comparison",),
        "top_score": 0.1,
        "hit_count": 5,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_case_observation(cases[0], **valid)


def test_evaluator_recomputes_canonical_source_and_route_contracts(cases):
    observations = _observations(cases)
    observations[0] = {**observations[0], "source_ranks": {}}
    observations[1] = {**observations[1], "applied_routes": []}
    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )
    assert result["cases"][0]["source_contract_passed"] is False
    assert result["cases"][1]["route_contract_passed"] is False
    assert result["target"]["passed_cases"] == 28

    injected = [{**observations[0], "source_contract_passed": True}, *observations[1:]]
    with pytest.raises(ValueError, match="observation fields"):
        evaluate_candidate(
            injected,
            candidate_threshold=0.015,
            global_threshold=0.03,
            stress_rows=_stress_rows(),
            formal_rows=_formal_rows(),
        )


def test_collision_route_contract_allows_additional_non_prohibited_route(cases):
    observations = _observations(cases)
    observations[19] = {
        **observations[19],
        "applied_routes": [
            "wage_arrears_termination",
            "off_hours_employer_message",
        ],
    }

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][19]
    assert collision["route_contract_passed"] is True
    assert collision["effective_threshold"] == 0.03
    assert collision["passed"] is True


def test_collision_route_contract_rejects_any_prohibited_route(cases):
    observations = _observations(cases)
    observations[19] = {
        **observations[19],
        "applied_routes": [
            "wage_arrears_termination",
            "severance_comparison",
        ],
    }

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][19]
    assert collision["route_contract_passed"] is False
    assert collision["passed"] is False


def test_candidate_recomputes_target_stress_and_formal_gates(cases):
    result = evaluate_candidate(
        _observations(cases),
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )
    assert result["target"] == {
        "total": 30,
        "passed_cases": 30,
        "positive_routes": 15,
        "positive_sources_at_5": 15,
        "positive_generation_allowed": 15,
        "collision_contracts": 15,
        "passed": True,
    }
    assert result["stress"]["direct_false_refusals"] == 0
    assert result["stress"]["direct_unanswerable_refusals"] == 17
    assert result["formal"]["hit_at_5"] == 0.9666666666666667
    assert result["formal"]["mrr_at_10"] == 0.9055555555555554
    assert result["passed"] is True
    assert result["stress_evidence"][36] == {
        "qid": "stress-037",
        "answerable": True,
        "rank": 1,
        "hit_count": 5,
        "has_hits": True,
        "reranker_enabled": True,
        "top_score": 0.0175004,
        "applied_routes": ["severance_comparison"],
    }
    assert result["stress_evidence"][40] == {
        "qid": "stress-041",
        "answerable": False,
        "rank": None,
        "hit_count": 0,
        "has_hits": False,
        "reranker_enabled": True,
        "top_score": 0.0,
        "applied_routes": [],
    }


def test_candidate_uses_unrounded_score_and_shared_equality_behavior(cases):
    observations = _observations(cases)
    observations[0] = {**observations[0], "top_score": 0.0149996}
    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )
    first = result["cases"][0]
    assert round(first["top_score"], 6) == 0.015
    assert first["refused"] is True
    assert result["target"]["passed_cases"] == 29


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda stress, formal: stress[0].pop("hit_count"),
            "fields",
        ),
        (
            lambda stress, formal: stress[0].update(hit_count=True),
            "hit_count",
        ),
        (
            lambda stress, formal: formal[0].update(reranker_enabled=False),
            "fields",
        ),
        (
            lambda stress, formal: stress[0].update(
                score_precision="raw_unrounded"
            ),
            "fields",
        ),
        (
            lambda stress, formal: (
                stress[0].update(answerable=False, rank=None),
                stress[40].update(answerable=True, rank=1),
            ),
            "answerability",
        ),
        (
            lambda stress, formal: (
                stress[0].update(applied_routes=["severance_comparison"]),
                stress[2].update(applied_routes=[]),
            ),
            "route identity",
        ),
        (
            lambda stress, formal: formal[0].update(
                applied_routes=["endpoint.example.com"]
            ),
            "allowlist",
        ),
    ],
)
def test_guard_evidence_fields_and_identity_are_bound_to_qid(
    cases, mutate, message
):
    stress = _stress_rows()
    formal = _formal_rows()
    mutate(stress, formal)
    with pytest.raises(ValueError, match=message):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=0.015,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=formal,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("has_hits", False), ("reranker_enabled", False)],
)
def test_guard_input_rejects_caller_controlled_decision_flags(
    cases, field, value
):
    stress = _stress_rows()
    stress[36][field] = value

    with pytest.raises(ValueError, match="fields"):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=0.015,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=_formal_rows(),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda stress: stress[0].update(hit_count=-1), "hit_count"),
        (lambda stress: stress[0].update(hit_count=6), "hit_count"),
        (
            lambda stress: stress[0].update(
                hit_count=0,
                rank=None,
                top_score=0.1,
            ),
            "zero-hit",
        ),
        (
            lambda stress: stress[0].update(hit_count=0, top_score=0.0),
            "zero-hit",
        ),
        (
            lambda stress: stress[0].update(hit_count=1, rank=2),
            "rank.*hit_count",
        ),
        (
            lambda stress: stress[40].update(hit_count=5),
            "positive hit_count",
        ),
    ],
)
def test_guard_input_rejects_hit_count_score_and_rank_inconsistency(
    cases, mutate, message
):
    stress = _stress_rows()
    mutate(stress)

    with pytest.raises(ValueError, match=message):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=0.015,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=_formal_rows(),
        )


@pytest.mark.parametrize("guard_name", ["stress", "formal"])
def test_guard_evidence_rejects_four_decimal_source_substitution(cases, guard_name):
    stress = _stress_rows()
    formal = _formal_rows()
    rows = stress if guard_name == "stress" else formal
    for row in rows:
        if row["hit_count"] > 0 and row["top_score"] < 0.0001:
            row["top_score"] = 0.1000004
        row["top_score"] = round(row["top_score"], 4)

    with pytest.raises(ValueError, match="four-decimal"):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=0.015,
            global_threshold=0.03,
            stress_rows=stress,
            formal_rows=formal,
        )


@pytest.mark.parametrize("threshold", [math.nan, 0.017])
def test_candidate_rejects_bad_candidate_grid(cases, threshold):
    with pytest.raises(ValueError, match="candidate_threshold"):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=threshold,
            global_threshold=0.03,
            stress_rows=_stress_rows(),
            formal_rows=_formal_rows(),
        )


def test_candidate_rejects_noncanonical_global_threshold(cases):
    with pytest.raises(ValueError, match="global_threshold"):
        evaluate_candidate(
            _observations(cases),
            candidate_threshold=0.015,
            global_threshold=0.02,
            stress_rows=_stress_rows(),
            formal_rows=_formal_rows(),
        )


def test_evaluator_calls_shared_policy_with_exact_raw_boundary(monkeypatch, cases):
    calls = []
    real_policy = policy.decide_retrieval_refusal

    def spy(**kwargs):
        calls.append(kwargs)
        return real_policy(**kwargs)

    monkeypatch.setattr(policy, "decide_retrieval_refusal", spy)
    evaluate_candidate(
        _observations(cases),
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )
    assert len(calls) == 130
    assert calls[66] == {
        "has_hits": True,
        "reranker_enabled": True,
        "applied_routes": ("severance_comparison",),
        "top_score": 0.0175004,
        "global_threshold": 0.03,
        "severance_comparison_threshold": 0.015,
    }
    assert calls[70] == {
        "has_hits": False,
        "reranker_enabled": True,
        "applied_routes": (),
        "top_score": 0.0,
        "global_threshold": 0.03,
        "severance_comparison_threshold": 0.015,
    }


def test_selector_replays_policy_and_returns_highest_complete_pass(
    monkeypatch, cases
):
    results = list(reversed(_candidate_results(_observations(cases))))
    calls = []
    real_policy = policy.decide_retrieval_refusal

    def spy(**kwargs):
        calls.append(kwargs)
        return real_policy(**kwargs)

    monkeypatch.setattr(policy, "decide_retrieval_refusal", spy)
    assert select_highest_passing_threshold(results) == 0.015
    assert len(calls) == 910
    assert {
        "has_hits": True,
        "reranker_enabled": True,
        "applied_routes": ("severance_comparison",),
        "top_score": 0.0175004,
        "global_threshold": 0.03,
        "severance_comparison_threshold": 0.015,
    } in calls
    assert {
        "has_hits": False,
        "reranker_enabled": True,
        "applied_routes": (),
        "top_score": 0.0,
        "global_threshold": 0.03,
        "severance_comparison_threshold": 0.015,
    } in calls


def _result_for(results, threshold):
    return next(row for row in results if row["candidate_threshold"] == threshold)


def test_selector_rejects_fabricated_passing_point_zero_three(cases):
    results = _candidate_results(_observations(cases))
    fabricated = _result_for(results, 0.03)
    fabricated["target"] = {
        **fabricated["target"],
        "passed_cases": 30,
        "positive_generation_allowed": 15,
        "passed": True,
    }
    fabricated["stress"] = {
        **fabricated["stress"],
        "direct_false_refusals": 0,
        "passed": True,
    }
    fabricated["passed"] = True
    with pytest.raises(ValueError, match="target aggregate"):
        select_highest_passing_threshold(results)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["cases"][0].update(effective_threshold=0.03),
            "case decision",
        ),
        (
            lambda result: result["cases"][0].update(refused=True),
            "case decision",
        ),
        (
            lambda result: result["cases"][0].update(applied_routes=[]),
            "case decision",
        ),
        (
            lambda result: result["target"].update(passed_cases=29, passed=False),
            "target aggregate",
        ),
        (
            lambda result: result["stress_evidence"][41].update(top_score=0.5),
            "stress aggregate",
        ),
        (
            lambda result: result["stress_evidence"][36].update(has_hits=False),
            "has_hits",
        ),
        (
            lambda result: result["formal_evidence"][0].update(
                reranker_enabled=False
            ),
            "reranker_enabled",
        ),
        (
            lambda result: result["formal"].update(hit_at_5=1.0),
            "formal aggregate",
        ),
    ],
)
def test_selector_rejects_case_decision_and_guard_mismatches(
    cases, mutate, message
):
    results = _candidate_results(_observations(cases))
    mutate(_result_for(results, 0.015))
    with pytest.raises(ValueError, match=message):
        select_highest_passing_threshold(results)


def test_selector_rejects_grid_tampering_and_no_go(cases):
    results = _candidate_results(_observations(cases))
    with pytest.raises(ValueError, match="candidate grid"):
        select_highest_passing_threshold(results[:-1])

    observations = _observations(cases)
    observations[0] = {**observations[0], "applied_routes": []}
    failing = _candidate_results(observations)
    with pytest.raises(RuntimeError, match="no candidate"):
        select_highest_passing_threshold(failing)


def test_selector_rejects_nonfinite_case_decision_inputs(cases):
    results = _candidate_results(_observations(cases))
    results[0]["cases"][0]["top_score"] = math.nan
    with pytest.raises(ValueError, match="top_score"):
        select_highest_passing_threshold(results)


def test_official_artifact_recomputes_and_publishes_only_safe_truth(cases):
    observations = _observations(cases)
    observations[0] = {**observations[0], "top_score": 0.0150004}
    results = _candidate_results(observations)
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=results,
        provenance=_provenance(),
    )
    assert tuple(artifact) == (
        "schema_version",
        "provenance",
        "candidate_thresholds",
        "global_threshold",
        "selected_threshold",
        "guard_evidence",
        "guard_evidence_binding_sha256",
        "candidates",
        "cases",
    )
    assert artifact["schema_version"] == "1.2"
    assert artifact["candidate_thresholds"] == list(EXPECTED_CANDIDATES)
    assert artifact["global_threshold"] == 0.03
    assert artifact["selected_threshold"] == 0.015
    assert all("cases" not in result for result in artifact["candidates"])
    assert all("stress_evidence" not in result for result in artifact["candidates"])
    assert all("formal_evidence" not in result for result in artifact["candidates"])
    assert artifact["guard_evidence"] == {
        "stress": results[0]["stress_evidence"],
        "formal": results[0]["formal_evidence"],
    }
    assert artifact["guard_evidence"]["stress"][36]["top_score"] == 0.0175004
    assert artifact["guard_evidence"]["formal"][2]["top_score"] == 0.5000004
    assert artifact["guard_evidence_binding_sha256"] == _canonical_sha256(
        {
            "guard_evidence": artifact["guard_evidence"],
            "provenance": artifact["provenance"],
        }
    )
    assert set(artifact["cases"][0]) == OFFICIAL_CASE_FIELDS
    assert artifact["cases"][0]["top_score"] == 0.015

    selected = next(
        candidate
        for candidate in artifact["candidates"]
        if candidate["candidate_threshold"] == artifact["selected_threshold"]
    )
    stress_decisions = [
        policy.decide_retrieval_refusal(
            has_hits=row["has_hits"],
            reranker_enabled=row["reranker_enabled"],
            applied_routes=tuple(row["applied_routes"]),
            top_score=row["top_score"],
            global_threshold=artifact["global_threshold"],
            severance_comparison_threshold=artifact["selected_threshold"],
        ).refused
        for row in artifact["guard_evidence"]["stress"]
    ]
    assert sum(
        refused and row["answerable"]
        for refused, row in zip(
            stress_decisions, artifact["guard_evidence"]["stress"], strict=True
        )
    ) == selected["stress"]["direct_false_refusals"]
    assert sum(
        refused and not row["answerable"]
        for refused, row in zip(
            stress_decisions, artifact["guard_evidence"]["stress"], strict=True
        )
    ) == selected["stress"]["direct_unanswerable_refusals"]
    serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
    assert all(case.question not in serialized for case in cases)


def test_official_artifact_rejects_hidden_self_hashed_guard_evidence(cases):
    observations = _observations(cases)
    results = _candidate_results(observations)
    provenance = _provenance()
    provenance["source_artifact_sha256"]["stress_raw_evidence"] = "0" * 64
    provenance["source_artifact_sha256"]["formal_raw_evidence"] = "1" * 64
    with pytest.raises(ValueError, match="source_artifact_sha256 fields"):
        build_official_artifact(
            observations=observations,
            candidate_results=results,
            provenance=provenance,
        )


def test_official_artifact_binding_tracks_the_actual_published_guard_rows(cases):
    observations = _observations(cases)
    baseline = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=_provenance(),
    )
    changed_stress = _stress_rows()
    changed_stress[36]["top_score"] = 0.0175008
    changed = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(
            observations,
            stress_rows=changed_stress,
        ),
        provenance=_provenance(),
    )

    assert changed["guard_evidence"]["stress"][36]["top_score"] == 0.0175008
    assert (
        changed["guard_evidence_binding_sha256"]
        != baseline["guard_evidence_binding_sha256"]
    )


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("nested_source_key", "source_ranks"),
        ("route_value", "allowlist"),
        ("relative_path", "embedding_model"),
        ("schemeless_endpoint", "embedding_model"),
        ("credential_value", "embedding_revision"),
        ("invalid_run_origin", "run_origin"),
        ("reranker_disabled", "approved primitives"),
        ("account_field", "provenance fields"),
        ("nested_endpoint_key", "retrieval_configuration fields"),
    ],
)
def test_official_artifact_rejects_nested_privacy_attacks(cases, attack, message):
    observations = _observations(cases)
    results = _candidate_results(observations)
    provenance = _provenance()
    if attack == "nested_source_key":
        observations[0] = {**observations[0], "source_ranks": {"question": 1}}
    elif attack == "route_value":
        observations[0] = {
            **observations[0],
            "applied_routes": ["https://endpoint.invalid"],
        }
    elif attack == "relative_path":
        provenance["embedding_model"] = "private/models/bge-m3"
    elif attack == "schemeless_endpoint":
        provenance["embedding_model"] = "api.internal.example"
    elif attack == "credential_value":
        provenance["embedding_revision"] = "sk-private-credential"
    elif attack == "invalid_run_origin":
        provenance["run_origin"] = "imported_public_trace"
    elif attack == "reranker_disabled":
        provenance["retrieval_configuration"]["reranker"] = False
    elif attack == "account_field":
        provenance["account"] = "person@example.com"
    else:
        provenance["retrieval_configuration"]["endpoint"] = "api.internal.example"
    with pytest.raises(ValueError, match=message):
        build_official_artifact(
            observations=observations,
            candidate_results=results,
            provenance=provenance,
        )
