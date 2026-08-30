import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag.release_verification as release_verification
import rag.severance_refusal_policy as policy
from eval import run_severance_refusal_policy as runner
from rag import factory
from rag.generation import llm as llm_module
from rag.generation import router as router_module
from rag.indexing import embedder as embedder_module
from rag.retrieval import reranker as reranker_module
from rag.retrieval.pipeline import RetrievalPipeline, plan_retrieval_query
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
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
}
OFFICIAL_CASE_FIELDS = {
    "qid",
    "case_type",
    "answerable",
    "source_ranks",
    "applied_routes",
    "hit_count",
    "top_score",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
    "effective_threshold",
    "refused",
    "refusal_stage",
    "source_contract_passed",
    "route_contract_passed",
    "expected_outcome",
    "generation_allowed",
    "outcome_contract_passed",
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


def test_runner_rejects_non_cpu_device_before_model_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("non-CPU input must fail before model preflight"),
    )

    with pytest.raises(SystemExit):
        runner.main(["--offline", "--device", "cuda"])


@pytest.mark.parametrize(
    "unsafe_output",
    [
        "historical",
        "official",
        "source",
        "relative_arbitrary",
        "traversal_collision",
        "traversal_to_approved",
        "case_alias",
        "absolute_arbitrary",
    ],
)
def test_runner_rejects_nonapproved_diagnostics_output_before_models_or_writes(
    monkeypatch, tmp_path, unsafe_output
) -> None:
    outputs = {
        "historical": PROJECT_ROOT
        / "eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json",
        "official": runner.OFFICIAL_RESULT,
        "source": runner.DEFAULT_DATASET,
        "relative_arbitrary": Path("eval/diagnostics/arbitrary.json"),
        "traversal_collision": Path(
            "eval/diagnostics/../official/severance_refusal_policy_v0.3.6.json"
        ),
        "traversal_to_approved": Path(
            "eval/official/../diagnostics/"
            "severance_retrieval_pivot_v0.3.6_no_go.json"
        ),
        "case_alias": PROJECT_ROOT
        / "Eval/diagnostics/severance_retrieval_pivot_v0.3.6_no_go.json",
        "absolute_arbitrary": tmp_path / "arbitrary.json",
    }
    protected = outputs[unsafe_output]
    before = protected.read_bytes() if protected.is_file() else None
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("unsafe output must fail before model preflight"),
    )
    monkeypatch.setattr(
        runner,
        "_write_public_json",
        lambda *_args: pytest.fail("unsafe output must fail before artifact writes"),
    )

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--offline",
                "--diagnostics-output",
                str(protected),
            ]
        )

    if before is None:
        assert not protected.exists()
    else:
        assert protected.read_bytes() == before


@pytest.mark.parametrize(
    "source_option",
    ["--dataset", "--stress-dataset", "--formal-dataset", "--snapshot"],
)
def test_runner_rejects_approved_output_aliased_as_a_source_before_models(
    monkeypatch, source_option
) -> None:
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("output/source collision must fail before models"),
    )

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--offline",
                "--diagnostics-output",
                str(runner.DIAGNOSTIC_RESULT),
                source_option,
                str(runner.DIAGNOSTIC_RESULT),
            ]
        )


def test_runner_rejects_hardlink_alias_collision_before_models(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "source.jsonl"
    approved = tmp_path / "pivot-no-go.json"
    source.write_text("source evidence\n", encoding="utf-8")
    os.link(source, approved)
    monkeypatch.setattr(runner, "DIAGNOSTIC_RESULT", approved)
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("hardlink collision must fail before models"),
    )

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--offline",
                "--dataset",
                str(source),
                "--diagnostics-output",
                str(approved),
            ]
        )

    assert source.read_text(encoding="utf-8") == "source evidence\n"


@pytest.mark.parametrize("alias_component", ["output", "parent"])
def test_runner_rejects_approved_path_through_symlink_before_models(
    monkeypatch, tmp_path, alias_component
) -> None:
    repository = tmp_path / "repository"
    eval_dir = repository / "eval"
    eval_dir.mkdir(parents=True)
    arbitrary_target = tmp_path / "arbitrary-target"
    arbitrary_target.mkdir()
    diagnostics = eval_dir / "diagnostics"
    approved = diagnostics / "severance_retrieval_pivot_v0.3.6_no_go.json"
    try:
        if alias_component == "parent":
            diagnostics.symlink_to(arbitrary_target, target_is_directory=True)
            untouched = arbitrary_target
        else:
            diagnostics.mkdir()
            target_file = arbitrary_target / "unrelated.json"
            target_file.write_text("untouched\n", encoding="utf-8")
            approved.symlink_to(target_file)
            untouched = target_file
    except OSError as exc:
        pytest.skip(f"OS denied symlink creation: {exc}")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(runner, "DIAGNOSTIC_RESULT", approved)
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("symlinked output must fail before models"),
    )
    before = (
        tuple(arbitrary_target.iterdir())
        if untouched.is_dir()
        else untouched.read_bytes()
    )

    with pytest.raises(SystemExit):
        runner.main(["--offline", "--diagnostics-output", str(approved)])

    after = (
        tuple(arbitrary_target.iterdir())
        if untouched.is_dir()
        else untouched.read_bytes()
    )
    assert after == before


def test_runner_rejects_approved_path_through_windows_junction_before_models(
    monkeypatch, tmp_path
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows junctions are unavailable on this platform")
    repository = tmp_path / "repository"
    eval_dir = repository / "eval"
    eval_dir.mkdir(parents=True)
    arbitrary_target = tmp_path / "junction-target"
    arbitrary_target.mkdir()
    diagnostics = eval_dir / "diagnostics"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(diagnostics), str(arbitrary_target)],
        check=False,
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("OS denied junction creation")
    approved = diagnostics / "severance_retrieval_pivot_v0.3.6_no_go.json"
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(runner, "DIAGNOSTIC_RESULT", approved)
    monkeypatch.setattr(
        runner,
        "_offline_preflight",
        lambda _args: pytest.fail("junction output must fail before models"),
    )

    with pytest.raises(SystemExit):
        runner.main(["--offline", "--diagnostics-output", str(approved)])

    assert tuple(arbitrary_target.iterdir()) == ()


def test_alias_stat_seam_recognizes_windows_reparse_points() -> None:
    detector = getattr(runner, "_stat_indicates_alias", None)
    assert callable(detector), "runner must expose the platform-independent alias seam"

    assert detector(
        SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    ) is True


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
        routes = plan_retrieval_query(question).routes
        reranker_calls = 2 if routes == ("severance_comparison",) else 1
        return SimpleNamespace(
            hits=hits,
            candidates=hits,
            applied_routes=routes,
            top_score=0.0150004,
            first_stage_retrieval_calls=1,
            reranker_calls=reranker_calls,
            reranker_scored_pairs=(len(hits),) * reranker_calls,
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
        "candidate_count": 2,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": 2,
        "reranker_scored_pairs": [2, 2],
    }


def test_runner_rejects_target_route_disagreement_instead_of_recording_it(
    cases,
) -> None:
    case = cases[0]
    retrieval = SimpleNamespace(
        hits=[SimpleNamespace(payload={"doc_title": "other", "articles": []})],
        applied_routes=(),
        top_score=0.5000004,
    )

    with pytest.raises(
        RuntimeError, match="route mismatch.*severance-policy-001"
    ):
        runner._run_target_cases(
            SimpleNamespace(run=lambda _question: retrieval), [case]
        )


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
        candidates=[object(), object()],
        first_stage_retrieval_calls=1,
        reranker_calls=2,
        reranker_scored_pairs=(2, 2),
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
            "candidate_count": 2,
            "route_plan_matched": True,
            "first_stage_retrieval_calls": 1,
            "reranker_calls": 2,
            "reranker_scored_pairs": [2, 2],
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


def test_runner_route_ablation_sweeps_same_retrieval_evidence(
    monkeypatch,
) -> None:
    observations = [{"retrieval": "target"}]
    stress = [{"retrieval": "stress"}]
    formal = [{"retrieval": "formal"}]
    calls = []

    def evaluate(actual_observations, **kwargs):
        calls.append((actual_observations, kwargs))
        return {"candidate_threshold": kwargs["candidate_threshold"]}

    monkeypatch.setattr(runner, "evaluate_route_ablation_candidate", evaluate)

    results = runner._evaluate_route_ablation(observations, stress, formal)

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
    assert first["schema_version"] == "1.3"
    assert first["production_threshold"] == 0.03
    assert first["route_ablation"]["highest_passing_candidate"] == 0.03
    assert "selected_threshold" not in first
    assert "global_threshold" not in first
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
    )

    assert envelope["schema_version"] == "1.3"
    assert envelope["evidence_class"] == "non_release_pivot_no_go"
    assert envelope["outcome"] == "no_go"
    assert envelope["official_export_allowed"] is False
    assert envelope["production_threshold"] == 0.03
    assert envelope["route_ablation"]["highest_passing_candidate"] is None
    assert envelope["target_observations"] == observations
    assert envelope["target_observations"][0]["top_score"] == 0.0150004
    assert len(envelope["guard_evidence"]["stress"]) == 60
    assert len(envelope["guard_evidence"]["formal"]) == 40
    assert len(envelope["route_ablation"]["candidates"]) == 7
    assert envelope["failed_gates"] == [
        {"candidate_threshold": 0.0, "gates": ["target"]},
        {"candidate_threshold": 0.005, "gates": ["target"]},
        {"candidate_threshold": 0.01, "gates": ["target"]},
        {"candidate_threshold": 0.015, "gates": ["target"]},
        {"candidate_threshold": 0.02, "gates": ["target"]},
        {"candidate_threshold": 0.025, "gates": ["target"]},
        {"candidate_threshold": 0.03, "gates": ["target", "route_ablation"]},
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
    )
    mutated = json.loads(json.dumps(envelope))
    mutated["route_ablation"]["candidates"][0]["target"]["passed_cases"] = 28

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
        "decision_code_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for name, path in policy.DECISION_CODE_PATHS.items()
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
        "precision_mode": "fp32",
        "local_files_only": True,
        "semantic_view_sha256": (
            "3faf051810ad233ee2da982155c4c4d8127f00d2aea56351fa19f87bee0d49a6"
        ),
        "merge_policy_version": "primary_first_deduplicating_interleave_v1",
        "primary_score_semantics": "full_precision_primary_query_top_score",
        "source_tree_clean": True,
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
    artifact = {
        "production_threshold": 0.03,
        "route_ablation": {"highest_passing_candidate": 0.03},
    }
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
        "_evaluate_route_ablation",
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
    monkeypatch.setattr(runner, "DIAGNOSTIC_RESULT", diagnostic)
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


def test_runner_main_no_go_keeps_real_retrieval_factory_provider_free(
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
    observations = _observations(cases)
    observations[0] = {**observations[0], "source_ranks": {}}
    stress_evidence = _stress_rows()
    formal_evidence = _formal_rows()
    closed = []
    embedder = SimpleNamespace(close=lambda: closed.append("embedder"))
    store = SimpleNamespace(close=lambda: closed.append("store"))
    seen_pipelines = []

    def forbidden_provider_construction(*_args, **_kwargs):
        pytest.fail("LLM/provider construction is forbidden in calibration main")

    for name in (
        "AnthropicAdapter",
        "OpenAIAdapter",
        "GeminiAdapter",
        "OllamaAdapter",
        "build_llm",
    ):
        monkeypatch.setattr(llm_module, name, forbidden_provider_construction)
    monkeypatch.setattr(
        router_module, "RoutedLLM", forbidden_provider_construction
    )
    monkeypatch.setattr(factory, "build_llm", forbidden_provider_construction)
    monkeypatch.setattr(factory, "build_answerer", forbidden_provider_construction)
    monkeypatch.setattr(factory, "RoutedLLM", forbidden_provider_construction)
    monkeypatch.setattr(runner, "OFFICIAL_RESULT", official)
    monkeypatch.setattr(runner, "DIAGNOSTIC_RESULT", diagnostic)
    monkeypatch.setattr(runner, "require_cached_models", lambda _settings: None)
    monkeypatch.setattr(runner, "_clean_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "_materialize_audited_corpus",
        lambda _work, _snapshot: corpus,
    )
    monkeypatch.setattr(
        runner,
        "_build_indexes",
        lambda _settings, _corpus, *, local_files_only: (embedder, store),
    )
    monkeypatch.setattr(
        reranker_module,
        "Reranker",
        lambda **_kwargs: SimpleNamespace(rerank=lambda *_args, **_kwargs: []),
    )
    monkeypatch.setattr(
        factory.BM25Index,
        "load",
        lambda _path: SimpleNamespace(search=lambda *_args, **_kwargs: []),
    )
    monkeypatch.setattr(runner, "load_cases", lambda _path: cases)
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda path: stress_evidence if path == stress else formal_evidence,
    )

    def collect_targets(pipeline, actual_cases):
        assert isinstance(pipeline, RetrievalPipeline)
        seen_pipelines.append(pipeline)
        assert actual_cases is cases
        return observations

    def collect_guards(pipeline, rows):
        assert isinstance(pipeline, RetrievalPipeline)
        seen_pipelines.append(pipeline)
        return rows

    monkeypatch.setattr(runner, "_run_target_cases", collect_targets)
    monkeypatch.setattr(runner, "_run_guard_cases", collect_guards)

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
    assert len(seen_pipelines) == 3
    assert len({id(pipeline) for pipeline in seen_pipelines}) == 1
    assert closed == ["store", "embedder"]
    assert not official.exists()
    assert policy.replay_no_go_evidence(
        json.loads(diagnostic.read_text(encoding="utf-8"))
    )


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _observations(cases):
    observations = []
    for index, case in enumerate(cases):
        score = 0.0300004 if index == 0 else 0.5
        if case.qid == "severance-policy-023":
            score = 0.0
        elif case.qid == "severance-policy-027":
            score = 0.029
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
                candidate_count=0 if case.qid == "severance-policy-023" else 20,
                route_plan_matched=True,
                first_stage_retrieval_calls=1,
                reranker_calls=(
                    0
                    if case.qid == "severance-policy-023"
                    else 2
                    if plan_retrieval_query(case.question).routes
                    == ("severance_comparison",)
                    else 1
                ),
                reranker_scored_pairs=(
                    ()
                    if case.qid == "severance-policy-023"
                    else (20, 20)
                    if plan_retrieval_query(case.question).routes
                    == ("severance_comparison",)
                    else (20,)
                ),
            )
        )
    return observations


def _stress_rows():
    rows = []
    for number in range(1, 61):
        qid = f"stress-{number:03d}"
        answerable = number <= 40
        if qid == "stress-037":
            score = 0.3114268601959007
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
                "candidate_count": 0 if qid == "stress-041" else 20,
                "route_plan_matched": True,
                "first_stage_retrieval_calls": 1,
                "reranker_calls": (
                    0 if qid == "stress-041" else 2 if qid in {"stress-003", "stress-037"} else 1
                ),
                "reranker_scored_pairs": (
                    []
                    if qid == "stress-041"
                    else [20, 20]
                    if qid in {"stress-003", "stress-037"}
                    else [20]
                ),
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
            "candidate_count": 20,
            "route_plan_matched": True,
            "first_stage_retrieval_calls": 1,
            "reranker_calls": 2 if number == 3 else 1,
            "reranker_scored_pairs": [20, 20] if number == 3 else [20],
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
            "candidate_count": 0 if number == 31 else 20,
            "route_plan_matched": True,
            "first_stage_retrieval_calls": 1,
            "reranker_calls": 0 if number == 31 else 1,
            "reranker_scored_pairs": [] if number == 31 else [20],
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
        "decision_code_sha256": {
            name: f"{index % 16:x}" * 64
            for index, name in enumerate(policy.DECISION_CODE_PATHS)
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
        "precision_mode": "fp32",
        "local_files_only": True,
        "semantic_view_sha256": (
            "3faf051810ad233ee2da982155c4c4d8127f00d2aea56351fa19f87bee0d49a6"
        ),
        "merge_policy_version": "primary_first_deduplicating_interleave_v1",
        "primary_score_semantics": "full_precision_primary_query_top_score",
        "source_tree_clean": True,
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


def test_reviewed_dataset_uses_exact_outcomes_with_only_027_reclassified(cases):
    outcomes = {case.qid: case.expected_outcome for case in cases}

    assert set(outcomes.values()) == {"generation", "no_hits", "threshold"}
    assert outcomes["severance-policy-023"] == "no_hits"
    assert outcomes["severance-policy-024"] == "generation"
    assert outcomes["severance-policy-027"] == "threshold"
    assert all(
        outcome == "generation"
        for qid, outcome in outcomes.items()
        if qid not in {"severance-policy-023", "severance-policy-027"}
    )


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
        (lambda rows: rows[0].update(expected_outcome=True), "expected_outcome"),
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
        candidate_count=20,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=2,
        reranker_scored_pairs=(20, 20),
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
        candidate_count=0,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=0,
        reranker_scored_pairs=(),
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
        "candidate_count": 20,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": 2,
        "reranker_scored_pairs": (20, 20),
    }
    valid.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_case_observation(cases[0], **valid)


def test_evaluator_recomputes_canonical_source_and_route_contracts(cases):
    observations = _observations(cases)
    observations[0] = {**observations[0], "source_ranks": {}}
    observations[1] = {
        **observations[1],
        "applied_routes": [],
        "reranker_calls": 1,
        "reranker_scored_pairs": [20],
    }
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


def test_positive_route_contract_requires_the_exact_singleton_route(cases):
    observations = _observations(cases)
    observations[0] = {
        **observations[0],
        "applied_routes": [
            "severance_comparison",
            "off_hours_employer_message",
        ],
        "reranker_calls": 1,
        "reranker_scored_pairs": [20],
    }

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    assert result["cases"][0]["route_contract_passed"] is False
    assert result["cases"][0]["passed"] is False


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


def test_027_requires_a_positive_hit_threshold_refusal_without_generation(cases):
    observations = _observations(cases)
    observations[26] = build_case_observation(
        cases[26],
        source_ranks={},
        applied_routes=(),
        top_score=0.029,
        hit_count=5,
        candidate_count=20,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=1,
        reranker_scored_pairs=(20,),
    )

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][26]
    assert collision["route_contract_passed"] is True
    assert collision["refusal_stage"] == "threshold"
    assert collision["generation_allowed"] is False
    assert collision["expected_outcome"] == "threshold"
    assert collision["outcome_contract_passed"] is True
    assert collision["passed"] is True


def test_027_uses_the_strict_production_threshold_boundary(cases):
    observations = _observations(cases)
    observations[26] = build_case_observation(
        cases[26],
        source_ranks={},
        applied_routes=(),
        top_score=0.03,
        hit_count=5,
        candidate_count=20,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=1,
        reranker_scored_pairs=(20,),
    )

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][26]
    assert collision["effective_threshold"] == 0.03
    assert collision["refused"] is False
    assert collision["refusal_stage"] is None
    assert collision["generation_allowed"] is True
    assert collision["outcome_contract_passed"] is False
    assert collision["passed"] is False


def test_027_rejects_a_nonempty_route_tuple(cases):
    observations = _observations(cases)
    observations[26] = build_case_observation(
        cases[26],
        source_ranks={},
        applied_routes=("off_hours_employer_message",),
        top_score=0.029,
        hit_count=5,
        candidate_count=20,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=1,
        reranker_scored_pairs=(20,),
    )

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][26]
    assert collision["route_contract_passed"] is False
    assert collision["passed"] is False


def test_027_rejects_no_hits_instead_of_its_required_threshold_refusal(cases):
    observations = _observations(cases)
    observations[26] = build_case_observation(
        cases[26],
        source_ranks={},
        applied_routes=(),
        top_score=0.0,
        hit_count=0,
        candidate_count=0,
        route_plan_matched=True,
        first_stage_retrieval_calls=1,
        reranker_calls=0,
        reranker_scored_pairs=(),
    )

    result = evaluate_candidate(
        observations,
        candidate_threshold=0.015,
        global_threshold=0.03,
        stress_rows=_stress_rows(),
        formal_rows=_formal_rows(),
    )

    collision = result["cases"][26]
    assert collision["refusal_stage"] == "no_hits"
    assert collision["outcome_contract_passed"] is False
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
        "top_score": 0.3114268601959007,
        "applied_routes": ["severance_comparison"],
        "candidate_count": 20,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": 2,
        "reranker_scored_pairs": [20, 20],
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
        "candidate_count": 0,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": 0,
        "reranker_scored_pairs": [],
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
        "top_score": 0.3114268601959007,
        "global_threshold": 0.015,
    }
    assert calls[70] == {
        "has_hits": False,
        "reranker_enabled": True,
        "applied_routes": (),
        "top_score": 0.0,
        "global_threshold": 0.03,
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
    assert select_highest_passing_threshold(results) == 0.03
    assert len(calls) == 910
    assert {
        "has_hits": True,
        "reranker_enabled": True,
        "applied_routes": ("severance_comparison",),
        "top_score": 0.3114268601959007,
        "global_threshold": 0.015,
    } in calls
    assert {
        "has_hits": False,
        "reranker_enabled": True,
        "applied_routes": (),
        "top_score": 0.0,
        "global_threshold": 0.03,
    } in calls


def _result_for(results, threshold):
    return next(row for row in results if row["candidate_threshold"] == threshold)


def test_selector_rejects_fabricated_failing_point_zero_three(cases):
    results = _candidate_results(_observations(cases))
    fabricated = _result_for(results, 0.03)
    fabricated["target"] = {
        **fabricated["target"],
        "passed_cases": 29,
        "positive_generation_allowed": 14,
        "passed": False,
    }
    fabricated["stress"] = {
        **fabricated["stress"],
        "direct_false_refusals": 1,
        "passed": False,
    }
    fabricated["passed"] = False
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
            lambda result: result["cases"][0].update(
                applied_routes=[],
                reranker_calls=1,
                reranker_scored_pairs=[20],
            ),
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
    observations[0] = {
        **observations[0],
        "applied_routes": [],
        "reranker_calls": 1,
        "reranker_scored_pairs": [20],
    }
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
    observations[0] = {**observations[0], "top_score": 0.0300004}
    results = _candidate_results(observations)
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=results,
        provenance=_provenance(),
    )
    assert tuple(artifact) == (
        "schema_version",
        "provenance",
        "production_threshold",
        "route_ablation",
        "guard_evidence",
        "guard_evidence_binding_sha256",
        "target_evidence_binding_sha256",
        "cases",
    )
    assert artifact["schema_version"] == "1.3"
    assert artifact["production_threshold"] == 0.03
    assert artifact["route_ablation"]["candidate_thresholds"] == list(
        EXPECTED_CANDIDATES
    )
    assert artifact["route_ablation"]["highest_passing_candidate"] == 0.03
    assert all(
        "cases" not in result
        for result in artifact["route_ablation"]["candidates"]
    )
    assert all(
        "stress_evidence" not in result
        for result in artifact["route_ablation"]["candidates"]
    )
    assert all(
        "formal_evidence" not in result
        for result in artifact["route_ablation"]["candidates"]
    )
    assert artifact["guard_evidence"] == {
        "stress": results[0]["stress_evidence"],
        "formal": results[0]["formal_evidence"],
    }
    assert artifact["guard_evidence"]["stress"][36]["top_score"] == (
        0.3114268601959007
    )
    assert artifact["guard_evidence"]["formal"][2]["top_score"] == 0.5000004
    assert artifact["guard_evidence_binding_sha256"] == _canonical_sha256(
        {
            "guard_evidence": artifact["guard_evidence"],
            "provenance": artifact["provenance"],
        }
    )
    assert artifact["target_evidence_binding_sha256"] == _canonical_sha256(
        {"cases": artifact["cases"], "provenance": artifact["provenance"]}
    )
    assert set(artifact["cases"][0]) == OFFICIAL_CASE_FIELDS
    assert artifact["cases"][0]["top_score"] == 0.0300004

    selected = next(
        candidate
        for candidate in artifact["route_ablation"]["candidates"]
        if candidate["candidate_threshold"]
        == artifact["route_ablation"]["highest_passing_candidate"]
    )
    stress_decisions = [
        policy.decide_retrieval_refusal(
            has_hits=row["has_hits"],
            reranker_enabled=row["reranker_enabled"],
            applied_routes=tuple(row["applied_routes"]),
            top_score=row["top_score"],
            global_threshold=artifact["production_threshold"],
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


def test_official_artifact_replays_without_retrieval_or_model_execution(
    monkeypatch, cases
):
    observations = _observations(cases)
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=_provenance(),
    )

    monkeypatch.setattr(
        runner,
        "_build_local_pipeline",
        lambda *_args, **_kwargs: pytest.fail("replay must not construct models"),
    )

    assert policy.replay_official_artifact(artifact) == artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.update(schema_version="1.2"), "schema_version"),
        (
            lambda artifact: artifact.update(production_threshold=0.02),
            "production_threshold",
        ),
        (
            lambda artifact: artifact["route_ablation"].update(
                highest_passing_candidate=0.025
            ),
            "route ablation",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                execution_device="cuda"
            ),
            "execution_device",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                precision_mode="fp16"
            ),
            "precision_mode",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                local_files_only=False
            ),
            "local_files_only",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                semantic_view_sha256="0" * 64
            ),
            "semantic_view_sha256",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                merge_policy_version="primary_first_v0"
            ),
            "merge_policy_version",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                primary_score_semantics="rounded_top_score"
            ),
            "primary_score_semantics",
        ),
        (
            lambda artifact: artifact["provenance"].update(
                source_tree_clean=False
            ),
            "source_tree_clean",
        ),
        (
            lambda artifact: artifact["provenance"][
                "retrieval_configuration"
            ].update(rrf_k=61),
            "retrieval_configuration",
        ),
        (
            lambda artifact: artifact["provenance"]["decision_code_sha256"].update(
                severance_policy="z" * 64
            ),
            "severance_policy must be a lowercase",
        ),
        (
            lambda artifact: artifact["provenance"].update(provider_requests=1),
            "provider_requests",
        ),
        (
            lambda artifact: artifact["cases"][0].update(
                route_plan_matched=False
            ),
            "route_plan_matched",
        ),
        (
            lambda artifact: artifact["cases"][0].update(
                first_stage_retrieval_calls=2
            ),
            "first_stage_retrieval_calls",
        ),
        (
            lambda artifact: artifact["cases"][0].update(reranker_calls=1),
            "reranker_calls",
        ),
        (
            lambda artifact: artifact["cases"][0].update(
                reranker_scored_pairs=[21, 21]
            ),
            "reranker_scored_pairs",
        ),
        (
            lambda artifact: artifact["cases"][0].update(top_score=0.03),
            "replay mismatch",
        ),
    ],
)
def test_official_replay_rejects_schema_provenance_gate_and_evidence_tampering(
    monkeypatch, cases, mutate, message
):
    observations = _observations(cases)
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=_provenance(),
    )
    mutate(artifact)
    monkeypatch.setattr(
        runner,
        "_build_local_pipeline",
        lambda *_args, **_kwargs: pytest.fail("tamper checks must not construct models"),
    )

    with pytest.raises(ValueError, match=message):
        policy.replay_official_artifact(artifact)


def test_static_local_import_closure_covers_authoritative_execution_roots() -> None:
    validator = getattr(policy, "validate_decision_import_closure", None)
    assert callable(validator), "policy must validate its static local import closure"

    discovered = validator(PROJECT_ROOT)

    assert {
        "src/rag/portfolio_demo_regression.py",
        "src/rag/__init__.py",
        "src/rag/generation/__init__.py",
        "src/rag/indexing/__init__.py",
        "src/rag/ingestion/__init__.py",
        "src/rag/retrieval/__init__.py",
    } <= discovered


def test_static_local_import_closure_rejects_new_omitted_dependency(tmp_path) -> None:
    validator = getattr(policy, "validate_decision_import_closure", None)
    assert callable(validator), "policy must validate its static local import closure"
    runner_path = tmp_path / "runner.py"
    package = tmp_path / "src/rag"
    package.mkdir(parents=True)
    runner_path.write_text("from rag import newly_imported\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "newly_imported.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = {
        "runner": "runner.py",
        "rag_package": "src/rag/__init__.py",
    }

    with pytest.raises(ValueError, match="src/rag/newly_imported.py"):
        validator(tmp_path, roots=("runner.py",), manifest=manifest)


def test_static_local_import_closure_rejects_unallowlisted_dynamic_import(
    tmp_path,
) -> None:
    validator = getattr(policy, "validate_decision_import_closure", None)
    assert callable(validator), "policy must validate its static local import closure"
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(
        "import importlib\nimportlib.import_module('rag.hidden')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dynamic import.*runner.py"):
        validator(
            tmp_path,
            roots=("runner.py",),
            manifest={"runner": "runner.py"},
        )


@pytest.mark.parametrize(
    "source",
    [
        "import importlib as loader\nloader.import_module('rag.hidden')\n",
        "from importlib import import_module as loader\nloader('rag.hidden')\n",
        "import builtins as loader\nloader.__import__('rag.hidden')\n",
        "from builtins import __import__ as loader\nloader('rag.hidden')\n",
        "import importlib as il\nload = il.import_module\nload('rag.hidden')\n",
        "import builtins as bi\nload = bi.__import__\nload('rag.hidden')\n",
        "import importlib as il\ngetattr(il, 'import_module')('rag.hidden')\n",
        "import builtins as bi\ngetattr(bi, '__import__')('rag.hidden')\n",
        "getattr(__builtins__, '__import__')('rag.hidden')\n",
        (
            "import importlib\n"
            "getattr(getattr(importlib, 'import_module'), '__call__')"
            "('rag.hidden')\n"
        ),
        (
            "from builtins import getattr as pick\n"
            "import importlib as il\n"
            "pick(il, 'import_module')('rag.hidden')\n"
        ),
        "__import__('rag.hidden')\n",
        "import_module('rag.hidden')\n",
        "eval(\"__import__('rag.hidden')\")\n",
        "exec(\"import rag.hidden\")\n",
        "compile(\"import rag.hidden\", '<dynamic>', 'exec')\n",
        "from builtins import eval as run\nrun(\"__import__('rag.hidden')\")\n",
        "run = exec\nrun(\"import rag.hidden\")\n",
        "import builtins as bi\nbi.exec(\"import rag.hidden\")\n",
        (
            "import builtins as bi\n"
            "getattr(bi, 'compile')(\"import rag.hidden\", '<dynamic>', 'exec')\n"
        ),
        "value: eval(\"__import__('rag.hidden')\") = None\n",
        (
            "def evaluate(value: eval(\"__import__('rag.hidden')\")):\n"
            "    return value\n"
        ),
    ],
    ids=(
        "renamed-importlib-module",
        "renamed-importlib-function",
        "renamed-builtins-module",
        "renamed-builtins-function",
        "assigned-importlib-function",
        "assigned-builtins-function",
        "getattr-importlib-alias",
        "getattr-builtins-alias",
        "getattr-dunder-builtins",
        "nested-getattr",
        "renamed-getattr",
        "bare-dunder-import",
        "bare-import-module",
        "eval",
        "exec",
        "compile",
        "renamed-eval",
        "assigned-exec",
        "builtins-exec-attribute",
        "getattr-builtins-compile",
        "annotated-assignment-eval",
        "parameter-annotation-eval",
    ),
)
def test_static_local_import_closure_rejects_dynamic_aliases_and_code_execution(
    tmp_path, source
) -> None:
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="dynamic import or code execution"):
        policy.validate_decision_import_closure(
            tmp_path,
            roots=("runner.py",),
            manifest={"runner": "runner.py"},
        )


@pytest.mark.parametrize(
    "source",
    [
        (
            "def import_module(name):\n"
            "    return name\n"
            "import_module('harmless')\n"
        ),
        (
            "def __import__(name):\n"
            "    return name\n"
            "__import__('harmless')\n"
        ),
        (
            "class Helper:\n"
            "    def import_module(self, name):\n"
            "        return name\n"
            "helper = Helper()\n"
            "helper.import_module('harmless')\n"
        ),
        (
            "class Helper:\n"
            "    def compile(self, value):\n"
            "        return value\n"
            "helper = Helper()\n"
            "helper.compile('harmless')\n"
        ),
        "(lambda eval: eval('harmless'))(lambda value: value)\n",
    ],
    ids=(
        "user-import-module",
        "user-dunder-import",
        "user-import-module-attribute",
        "user-compile-attribute",
        "lambda-argument-eval",
    ),
)
def test_static_local_import_closure_allows_clearly_user_bound_similar_names(
    tmp_path, source
) -> None:
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(source, encoding="utf-8")

    assert policy.validate_decision_import_closure(
        tmp_path,
        roots=("runner.py",),
        manifest={"runner": "runner.py"},
    ) == frozenset({"runner.py"})


@pytest.mark.parametrize(
    "changed_dependency",
    [
        "retrieval_refusal_policy",
        "retrieval_fusion",
        "index_embedder",
        "runner_severance",
        "release_verifier_wrapper",
    ],
)
def test_release_verifier_invalidates_representative_dependency_changes(
    tmp_path, cases, changed_dependency
):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_paths = (
        "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
        "eval/dataset/reliability_stress_v0.3.1.jsonl",
        "eval/dataset/eval_set.jsonl",
        "release/corpus_snapshot.json",
        *policy.DECISION_CODE_PATHS.values(),
    )
    for relative_path in source_paths:
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative_path, destination)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    observations = _observations(cases)
    provenance = _provenance()
    provenance.update(
        dataset_sha256=hashlib.sha256(
            (repository / "eval/dataset/severance_refusal_policy_v0.3.6.jsonl").read_bytes()
        ).hexdigest(),
        corpus_snapshot_sha256=hashlib.sha256(
            (repository / "release/corpus_snapshot.json").read_bytes()
        ).hexdigest(),
        source_artifact_sha256={
            "stress_dataset": hashlib.sha256(
                (repository / "eval/dataset/reliability_stress_v0.3.1.jsonl").read_bytes()
            ).hexdigest(),
            "formal_dataset": hashlib.sha256(
                (repository / "eval/dataset/eval_set.jsonl").read_bytes()
            ).hexdigest(),
        },
        decision_code_sha256={
            name: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for name, path in policy.DECISION_CODE_PATHS.items()
        },
        code_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip(),
    )
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=provenance,
    )
    path = repository / "severance-policy.json"
    runner._write_public_json(path, artifact)

    summary = release_verification._verify_severance_refusal_policy_artifact(
        repository, path
    )

    assert summary == {
        "schema_version": "1.3",
        "production_threshold": 0.03,
        "highest_passing_candidate": 0.03,
        "target_cases": 30,
        "stress_cases": 60,
        "formal_cases": 40,
        "execution_device": "cpu",
        "precision_mode": "fp32",
        "provider_adapters": 0,
        "provider_requests": 0,
    }

    changed_path = repository / policy.DECISION_CODE_PATHS[changed_dependency]
    changed_path.write_bytes(changed_path.read_bytes() + b"\n")
    provenance["decision_code_sha256"][changed_dependency] = hashlib.sha256(
        changed_path.read_bytes()
    ).hexdigest()
    mismatched = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=provenance,
    )
    runner._write_public_json(path, mismatched)
    with pytest.raises(
        release_verification.ReleaseVerificationError,
        match=f"committed revision.*{changed_dependency}",
    ):
        release_verification._verify_severance_refusal_policy_artifact(
            repository, path
        )


def test_release_verifier_rejects_missing_or_untracked_relevant_files(
    tmp_path, cases
):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_paths = (
        "eval/dataset/severance_refusal_policy_v0.3.6.jsonl",
        "eval/dataset/reliability_stress_v0.3.1.jsonl",
        "eval/dataset/eval_set.jsonl",
        "release/corpus_snapshot.json",
        *policy.DECISION_CODE_PATHS.values(),
    )
    for relative_path in source_paths:
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative_path, destination)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    observations = _observations(cases)
    provenance = _provenance()
    provenance.update(
        dataset_sha256=hashlib.sha256(
            (repository / "eval/dataset/severance_refusal_policy_v0.3.6.jsonl").read_bytes()
        ).hexdigest(),
        corpus_snapshot_sha256=hashlib.sha256(
            (repository / "release/corpus_snapshot.json").read_bytes()
        ).hexdigest(),
        source_artifact_sha256={
            "stress_dataset": hashlib.sha256(
                (repository / "eval/dataset/reliability_stress_v0.3.1.jsonl").read_bytes()
            ).hexdigest(),
            "formal_dataset": hashlib.sha256(
                (repository / "eval/dataset/eval_set.jsonl").read_bytes()
            ).hexdigest(),
        },
        decision_code_sha256={
            name: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for name, relative in policy.DECISION_CODE_PATHS.items()
        },
        code_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip(),
    )
    artifact = build_official_artifact(
        observations=observations,
        candidate_results=_candidate_results(observations),
        provenance=provenance,
    )
    artifact_path = repository / "severance-policy.json"
    runner._write_public_json(artifact_path, artifact)
    relevant = repository / policy.DECISION_CODE_PATHS["release_verifier_wrapper"]
    subprocess.run(
        [
            "git",
            "rm",
            "--cached",
            policy.DECISION_CODE_PATHS["release_verifier_wrapper"],
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    with pytest.raises(
        release_verification.ReleaseVerificationError,
        match="committed revision.*release_verifier_wrapper",
    ):
        release_verification._verify_severance_refusal_policy_artifact(
            repository, artifact_path
        )

    relevant.unlink()
    with pytest.raises(
        release_verification.ReleaseVerificationError,
        match="missing.*release_verifier_wrapper",
    ):
        release_verification._verify_severance_refusal_policy_artifact(
            repository, artifact_path
        )


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
