"""Strict, content-free scoring for the severance refusal calibration."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import symtable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag.retrieval.pipeline import (
    MULTI_VIEW_MERGE_POLICY_VERSION,
    SEVERANCE_SEMANTIC_VIEW_SHA256,
)
from rag.retrieval.refusal_policy import decide_retrieval_refusal

CANDIDATE_THRESHOLDS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
DECISION_CODE_PATHS = {
    # Runtime configuration, data contracts, and construction wiring.
    "config": "src/rag/config.py",
    "corpus_audit": "src/rag/corpus_audit.py",
    "evaluation": "src/rag/evaluation.py",
    "factory": "src/rag/factory.py",
    "models": "src/rag/models.py",
    "portfolio_demo_regression": "src/rag/portfolio_demo_regression.py",
    "reliability": "src/rag/reliability.py",
    "wage_arrears_regression": "src/rag/wage_arrears_regression.py",
    # Factory imports whose import-time behavior is part of provider isolation.
    "generation_answerer": "src/rag/generation/answerer.py",
    "generation_package": "src/rag/generation/__init__.py",
    "generation_llm": "src/rag/generation/llm.py",
    "generation_prompts": "src/rag/generation/prompts.py",
    "generation_router": "src/rag/generation/router.py",
    # Complete local indexing and retrieval implementation used by acceptance.
    "index_bm25": "src/rag/indexing/bm25_index.py",
    "index_embedder": "src/rag/indexing/embedder.py",
    "index_legal_terms": "src/rag/indexing/dict/legal_terms.txt",
    "index_tokenizer": "src/rag/indexing/tokenizer.py",
    "index_vector_store": "src/rag/indexing/vector_store.py",
    "indexing_package": "src/rag/indexing/__init__.py",
    "ingestion_chunkers": "src/rag/ingestion/chunkers.py",
    "ingestion_cleaner": "src/rag/ingestion/cleaner.py",
    "ingestion_loader": "src/rag/ingestion/loader.py",
    "ingestion_package": "src/rag/ingestion/__init__.py",
    "rag_package": "src/rag/__init__.py",
    "retrieval_fusion": "src/rag/retrieval/fusion.py",
    "retrieval_pipeline": "src/rag/retrieval/pipeline.py",
    "retrieval_refusal_policy": "src/rag/retrieval/refusal_policy.py",
    "retrieval_reranker": "src/rag/retrieval/reranker.py",
    "retrieval_retriever": "src/rag/retrieval/retriever.py",
    "retrieval_package": "src/rag/retrieval/__init__.py",
    # Evidence construction, replay, and release verification entry points.
    "severance_policy": "src/rag/severance_refusal_policy.py",
    "runner_bootstrap": "eval/_bootstrap.py",
    "runner_lib": "eval/lib.py",
    "runner_reliability": "eval/run_reliability_eval.py",
    "runner_severance": "eval/run_severance_refusal_policy.py",
    "script_audit_corpus": "scripts/audit_corpus.py",
    "script_bootstrap": "scripts/_bootstrap.py",
    "script_download_corpus": "scripts/download_corpus.py",
    "release_verifier": "src/rag/release_verification.py",
    "release_verifier_wrapper": "scripts/verify_release.py",
    # Python dependency resolution can alter CPU/FP32/model execution semantics.
    "project_configuration": "pyproject.toml",
    "runtime_lock": "uv.lock",
}
DECISION_IMPORT_ROOTS = (
    "eval/run_severance_refusal_policy.py",
    "src/rag/release_verification.py",
    "scripts/verify_release.py",
)
EXPECTED_QIDS = tuple(
    f"severance-policy-{number:03d}" for number in range(1, 31)
)
FORMAL_HIT_AT_5_BASELINE = 0.9666666666666667
FORMAL_MRR_AT_10_BASELINE = 0.9055555555555554
_GLOBAL_THRESHOLD = 0.03
_SCHEMA_VERSION = "1.3"
_PRECISION_MODE = "fp32"
_SEMANTIC_VIEW_SHA256 = SEVERANCE_SEMANTIC_VIEW_SHA256
_MERGE_POLICY_VERSION = MULTI_VIEW_MERGE_POLICY_VERSION
_PRIMARY_SCORE_SEMANTICS = "full_precision_primary_query_top_score"

_DATASET_FIELDS = {
    "qid",
    "question",
    "case_type",
    "answerable",
    "sources",
    "required_routes",
    "prohibited_routes",
    "expected_outcome",
    "style_tags",
}
_OBSERVATION_FIELDS = {
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
_CASE_RESULT_FIELDS = {
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
_GUARD_INPUT_FIELDS = {
    "qid",
    "answerable",
    "rank",
    "hit_count",
    "top_score",
    "applied_routes",
    "candidate_count",
    "route_plan_matched",
    "first_stage_retrieval_calls",
    "reranker_calls",
    "reranker_scored_pairs",
}
_GUARD_EVIDENCE_FIELDS = _GUARD_INPUT_FIELDS | {
    "has_hits",
    "reranker_enabled",
}
_CANDIDATE_FIELDS = {
    "candidate_threshold",
    "global_threshold",
    "target",
    "stress",
    "formal",
    "cases",
    "stress_evidence",
    "formal_evidence",
    "passed",
}
_PROVENANCE_FIELDS = {
    "dataset_sha256",
    "corpus_snapshot_sha256",
    "source_artifact_sha256",
    "decision_code_sha256",
    "embedding_model",
    "embedding_revision",
    "reranker_model",
    "reranker_revision",
    "retrieval_configuration",
    "execution_device",
    "precision_mode",
    "local_files_only",
    "semantic_view_sha256",
    "merge_policy_version",
    "primary_score_semantics",
    "source_tree_clean",
    "code_revision",
    "run_origin",
    "provider_adapters",
    "provider_requests",
}
_SOURCE_ARTIFACT_FIELDS = {
    "stress_dataset",
    "formal_dataset",
}
_TOP_K_FINAL = 5
_RETRIEVAL_CONFIGURATION = {
    "chunking": "structure",
    "retrieval": "hybrid",
    "reranker": True,
    "top_k_retrieve": 20,
    "top_k_final": _TOP_K_FINAL,
    "rrf_k": 60,
}
_EMBEDDING_MODEL = "BAAI/bge-m3"
_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_RUN_ORIGIN = "fresh_offline_retrieval"
_KNOWN_ROUTES = {
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
}
_SEVERANCE_ROUTE = ("severance_comparison",)
_WAGE_ROUTE = ("wage_arrears_termination",)
_MULTI_ROUTE = ("severance_comparison", "wage_arrears_termination")
_PENSION_12 = "勞工退休金條例|第 12 條"
_LABOR_17 = "勞動基準法|第 17 條"
_LABOR_11 = "勞動基準法|第 11 條"
_LABOR_16 = "勞動基準法|第 16 條"
_LABOR_14 = "勞動基準法|第 14 條"
_PENSION_24 = "勞工退休金條例|第 24 條"
_LABOR_54 = "勞動基準法|第 54 條"
_LABOR_30 = "勞動基準法|第 30 條"
_CANONICAL_SOURCE_KEYS = {
    _PENSION_12,
    _LABOR_17,
    _LABOR_11,
    _LABOR_16,
    _LABOR_14,
    _PENSION_24,
    _LABOR_54,
    _LABOR_30,
}
_POSITIVE_STYLES = {
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
_COLLISION_STYLES = {
    "single_regime",
    "ordinary_termination",
    "notice_only",
    "wage_arrears",
    "generic_retirement",
    "unrelated_old_new",
    "partial_cue_collision",
}


@dataclass(frozen=True)
class SeverancePolicyCase:
    qid: str
    question: str
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    sources: tuple[dict[str, str], ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expected_outcome: Literal["generation", "no_hits", "threshold"]
    style_tags: tuple[str, ...]


@dataclass(frozen=True)
class _CaseContract:
    case_type: Literal["positive", "collision_negative"]
    answerable: bool
    source_keys: tuple[str, ...]
    required_routes: tuple[str, ...]
    prohibited_routes: tuple[str, ...]
    expected_outcome: Literal["generation", "no_hits", "threshold"]


def _collision(
    source_keys: tuple[str, ...],
    *,
    answerable: bool = True,
    required_routes: tuple[str, ...] = (),
    prohibited_routes: tuple[str, ...] = _SEVERANCE_ROUTE,
    expected_outcome: Literal["generation", "no_hits", "threshold"] = "generation",
) -> _CaseContract:
    return _CaseContract(
        case_type="collision_negative",
        answerable=answerable,
        source_keys=source_keys,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        expected_outcome=expected_outcome,
    )


_POSITIVE_CONTRACT = _CaseContract(
    case_type="positive",
    answerable=True,
    source_keys=(_PENSION_12, _LABOR_17),
    required_routes=_SEVERANCE_ROUTE,
    prohibited_routes=(),
    expected_outcome="generation",
)
_CASE_CONTRACTS = {
    **{qid: _POSITIVE_CONTRACT for qid in EXPECTED_QIDS[:15]},
    "severance-policy-016": _collision((_PENSION_12,)),
    "severance-policy-017": _collision((_LABOR_17,)),
    "severance-policy-018": _collision((_LABOR_11,)),
    "severance-policy-019": _collision((_LABOR_16,)),
    "severance-policy-020": _collision(
        (_LABOR_14,), required_routes=_WAGE_ROUTE
    ),
    "severance-policy-021": _collision((_PENSION_24,)),
    "severance-policy-022": _collision((_LABOR_54,)),
    "severance-policy-023": _collision(
        (), answerable=False, expected_outcome="no_hits"
    ),
    "severance-policy-024": _collision((), answerable=False),
    "severance-policy-025": _collision((_PENSION_12,)),
    "severance-policy-026": _collision((_LABOR_17,)),
    "severance-policy-027": _collision(
        (), answerable=False, expected_outcome="threshold"
    ),
    "severance-policy-028": _collision((_LABOR_16,)),
    "severance-policy-029": _collision((_LABOR_30,)),
    "severance-policy-030": _collision(
        (_LABOR_14, _PENSION_12, _LABOR_17),
        required_routes=_MULTI_ROUTE,
        prohibited_routes=(),
    ),
}
_STRESS_QIDS = tuple(f"stress-{number:03d}" for number in range(1, 61))
_FORMAL_QIDS = tuple(f"eval-{number:02d}" for number in range(1, 41))
_STRESS_ROUTES = {
    "stress-003": _SEVERANCE_ROUTE,
    "stress-010": _WAGE_ROUTE,
    "stress-037": _SEVERANCE_ROUTE,
    "stress-038": _WAGE_ROUTE,
}
_FORMAL_ROUTES = {
    "eval-03": _SEVERANCE_ROUTE,
    "eval-10": _WAGE_ROUTE,
}


def _relative_project_path(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _module_name_for_path(project_root: Path, path: Path) -> tuple[str, ...]:
    source_root = project_root / "src"
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        relative = path.relative_to(project_root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _resolve_local_module(
    project_root: Path, importing_path: Path, module: str
) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    candidates = []
    if len(parts) == 1:
        candidates.extend(
            (
                importing_path.parent / f"{module}.py",
                importing_path.parent / module / "__init__.py",
            )
        )
    for base in (project_root / "src", project_root):
        module_path = base.joinpath(*parts)
        candidates.extend((module_path.with_suffix(".py"), module_path / "__init__.py"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _package_initializers(project_root: Path, path: Path) -> tuple[Path, ...]:
    source_root = project_root / "src"
    try:
        path.relative_to(source_root)
        package_root = source_root
    except ValueError:
        package_root = project_root
    initializers = []
    parent = path.parent
    while parent != package_root and package_root in parent.parents:
        initializer = parent / "__init__.py"
        if initializer.is_file():
            initializers.append(initializer)
        parent = parent.parent
    return tuple(initializers)


def _imported_local_paths(
    project_root: Path, importing_path: Path, tree: ast.AST
) -> set[Path]:
    imported: set[Path] = set()
    current_module = _module_name_for_path(project_root, importing_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = (
                    current_module
                    if importing_path.name == "__init__.py"
                    else current_module[:-1]
                )
                keep = len(package) - node.level + 1
                prefix = package[: max(keep, 0)]
                base = ".".join((*prefix, *(node.module or "").split(".")))
            else:
                base = node.module or ""
            modules = [base]
            modules.extend(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
                if alias.name != "*"
            )
        else:
            continue
        for module in modules:
            resolved = _resolve_local_module(project_root, importing_path, module)
            if resolved is not None:
                imported.add(resolved)
    return imported


_PROVEN_SAFE = "safe"
_PROVEN_CLASS = "class"
_PROVEN_SYS_MODULE = "sys-module"
_LOCAL_FUNCTION_PREFIX = "local-function:"
_UNKNOWN_BINDING = "unknown"
_UNBOUND = "unbound"
_DYNAMIC_API_NAMES = {
    "__import__",
    "import_module",
    "eval",
    "exec",
    "compile",
}
_DYNAMIC_BUILTINS_IMPORTS = {*_DYNAMIC_API_NAMES, "getattr", "*"}
_DYNAMIC_NAMESPACE_NAMES = {"globals", "locals", "vars"}


def _local_function_binding(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> str:
    name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "lambda"
    return (
        f"{_LOCAL_FUNCTION_PREFIX}{node.lineno}:{node.col_offset}:{name}"
    )


def _bindings_are_proven_safe(bindings: set[str]) -> bool:
    return bool(bindings) and all(
        binding in {_PROVEN_SAFE, _PROVEN_CLASS, _PROVEN_SYS_MODULE}
        or binding.startswith(_LOCAL_FUNCTION_PREFIX)
        for binding in bindings
    )


def _join_binding_environments(
    environments: list[dict[str, set[str]]],
) -> dict[str, set[str]]:
    names = set().union(*(environment for environment in environments))
    return {
        name: set().union(
            *(environment.get(name, {_UNBOUND}) for environment in environments)
        )
        for name in names
    }


class _AccessScope:
    """Scope-wide runtime bindings paired with Python's symbol table."""

    def __init__(
        self,
        node: ast.AST,
        table: symtable.SymbolTable,
        parent: _AccessScope | None,
    ) -> None:
        self.node = node
        self.table = table
        self.parent = parent
        self._children = list(table.get_children())
        self._used_children: set[int] = set()
        self._child_scopes: dict[int, _AccessScope] = {}
        if parent is None:
            self.function_definitions: dict[
                str,
                tuple[
                    ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
                    _AccessScope,
                ],
            ] = {}
        else:
            self.function_definitions = parent.function_definitions
        self.final_bindings = self._final_bindings()

    def _is_local_target(self, name: str) -> bool:
        if self.table.get_type() == "module":
            return True
        try:
            symbol = self.table.lookup(name)
        except KeyError:
            return False
        return symbol.is_local() or symbol.is_parameter()

    def _bind_target(
        self,
        environment: dict[str, set[str]],
        target: ast.AST,
        bindings: set[str],
    ) -> None:
        if isinstance(target, ast.Name) and self._is_local_target(target.id):
            environment[target.id] = set(bindings)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(environment, element, {_UNKNOWN_BINDING})
        elif isinstance(target, ast.Starred):
            self._bind_target(environment, target.value, {_UNKNOWN_BINDING})

    def _expression_bindings(
        self, node: ast.AST, environment: dict[str, set[str]]
    ) -> set[str]:
        if isinstance(node, ast.Constant):
            return {_PROVEN_SAFE}
        if isinstance(node, ast.Lambda):
            binding = _local_function_binding(node)
            self.function_definitions[binding] = (node, self)
            return {binding}
        if isinstance(node, ast.Name):
            return set(environment.get(node.id, {_UNKNOWN_BINDING}))
        if isinstance(node, ast.IfExp):
            return self._expression_bindings(
                node.body, environment
            ) | self._expression_bindings(node.orelse, environment)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            return {_PROVEN_SAFE}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if environment.get(node.func.id) == {_PROVEN_CLASS}:
                return {_PROVEN_SAFE}
        return {_UNKNOWN_BINDING}

    def _bind_assignment(
        self,
        environment: dict[str, set[str]],
        target: ast.AST,
        value: ast.AST,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            if len(target.elts) == len(value.elts):
                for element, item in zip(target.elts, value.elts, strict=True):
                    self._bind_assignment(environment, element, item)
                return
        self._bind_target(
            environment, target, self._expression_bindings(value, environment)
        )

    def _analyze_statements(
        self,
        statements: list[ast.stmt],
        initial: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        environment = {name: set(kinds) for name, kinds in initial.items()}
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                binding = _local_function_binding(statement)
                self.function_definitions[binding] = (statement, self)
                self._bind_target(
                    environment,
                    ast.Name(id=statement.name, ctx=ast.Store()),
                    {binding},
                )
            elif isinstance(statement, ast.ClassDef):
                self._bind_target(
                    environment,
                    ast.Name(id=statement.name, ctx=ast.Store()),
                    {_PROVEN_CLASS},
                )
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    self._bind_assignment(environment, target, statement.value)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                self._bind_assignment(environment, statement.target, statement.value)
            elif isinstance(statement, ast.AugAssign):
                self._bind_target(
                    environment, statement.target, {_UNKNOWN_BINDING}
                )
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                aliases = statement.names
                for alias in aliases:
                    name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                    if isinstance(statement, ast.Import):
                        module = alias.name.split(".", maxsplit=1)[0]
                        binding = (
                            _UNKNOWN_BINDING
                            if module in {"importlib", "builtins"}
                            else (
                                _PROVEN_SYS_MODULE
                                if module == "sys"
                                else _PROVEN_SAFE
                            )
                        )
                    else:
                        module = (statement.module or "").split(
                            ".", maxsplit=1
                        )[0]
                        binding = (
                            _UNKNOWN_BINDING
                            if module == "importlib"
                            or (
                                module == "builtins"
                                and alias.name in _DYNAMIC_BUILTINS_IMPORTS
                            )
                            else _PROVEN_SAFE
                        )
                    self._bind_target(
                        environment,
                        ast.Name(id=name, ctx=ast.Store()),
                        {binding},
                    )
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    self._bind_target(environment, target, {_UNBOUND})
            elif isinstance(statement, ast.If):
                environment = _join_binding_environments(
                    [
                        self._analyze_statements(statement.body, environment),
                        self._analyze_statements(statement.orelse, environment),
                    ]
                )
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                loop_environment = {
                    name: set(kinds) for name, kinds in environment.items()
                }
                self._bind_target(
                    loop_environment, statement.target, {_UNKNOWN_BINDING}
                )
                loop_environment = self._analyze_statements(
                    statement.body, loop_environment
                )
                environment = _join_binding_environments(
                    [
                        environment,
                        loop_environment,
                        self._analyze_statements(statement.orelse, environment),
                    ]
                )
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                normal = self._analyze_statements(statement.body, environment)
                normal = self._analyze_statements(statement.orelse, normal)
                paths = [normal]
                for handler in statement.handlers:
                    handler_environment = {
                        name: set(kinds) for name, kinds in environment.items()
                    }
                    if handler.name:
                        self._bind_target(
                            handler_environment,
                            ast.Name(id=handler.name, ctx=ast.Store()),
                            {_UNKNOWN_BINDING},
                        )
                    paths.append(
                        self._analyze_statements(handler.body, handler_environment)
                    )
                environment = self._analyze_statements(
                    statement.finalbody, _join_binding_environments(paths)
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                with_environment = {
                    name: set(kinds) for name, kinds in environment.items()
                }
                for item in statement.items:
                    if item.optional_vars is not None:
                        self._bind_target(
                            with_environment,
                            item.optional_vars,
                            {_UNKNOWN_BINDING},
                        )
                environment = self._analyze_statements(
                    statement.body, with_environment
                )
        return environment

    def _final_bindings(self) -> dict[str, set[str]]:
        environment = self._initial_bindings()
        return self._analyze_statements(self._statements(), environment)

    def _initial_bindings(self) -> dict[str, set[str]]:
        return {
            symbol.get_name(): {_PROVEN_SAFE}
            for symbol in self.table.get_symbols()
            if symbol.is_parameter()
        }

    def _statements(self) -> list[ast.stmt]:
        if isinstance(self.node, ast.Module):
            return self.node.body
        if isinstance(
            self.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return self.node.body
        return []

    def bindings_before(self, name: str, node: ast.AST) -> set[str]:
        return self.environment_before(node).get(name, {_UNBOUND})

    def environment_before(self, node: ast.AST) -> dict[str, set[str]]:
        line = getattr(node, "lineno", -1)
        column = getattr(node, "col_offset", -1)
        preceding = [
            statement
            for statement in self._statements()
            if getattr(statement, "end_lineno", statement.lineno) < line
            or (
                getattr(statement, "end_lineno", statement.lineno) == line
                and getattr(statement, "end_col_offset", -1) <= column
            )
        ]
        return self._analyze_statements(preceding, self._initial_bindings())

    def child(self, node: ast.AST) -> _AccessScope:
        cached = self._child_scopes.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            expected_type, expected_name = "function", node.name
        elif isinstance(node, ast.Lambda):
            expected_type, expected_name = "function", "lambda"
        elif isinstance(node, ast.ClassDef):
            expected_type, expected_name = "class", node.name
        else:
            raise ValueError(f"unsupported decision scope: {type(node).__name__}")
        for index, child in enumerate(self._children):
            if index in self._used_children:
                continue
            if (
                child.get_type() == expected_type
                and child.get_name() == expected_name
                and child.get_lineno() == node.lineno
            ):
                self._used_children.add(index)
                resolved = _AccessScope(node, child, self)
                self._child_scopes[id(node)] = resolved
                return resolved
        raise ValueError(
            f"cannot resolve decision scope {expected_name!r} at line {node.lineno}"
        )


class _DynamicAccessVisitor(ast.NodeVisitor):
    """Reject acquisition or access to dynamic execution APIs."""

    def __init__(
        self,
        scope: _AccessScope,
        *,
        module_bindings: dict[str, set[str]] | None = None,
        import_time: bool = False,
        call_stack: frozenset[str] = frozenset(),
    ) -> None:
        self.scope = scope
        self.module_bindings = module_bindings
        self.import_time = import_time
        self.call_stack = call_stack
        self.reason: str | None = None

    def _reject(self, node: ast.AST, reason: str) -> None:
        if self.reason is None:
            self.reason = f"line {getattr(node, 'lineno', '?')}: {reason}"

    def _module_scope(self) -> _AccessScope:
        scope = self.scope
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def _enclosing_bindings(self, name: str) -> set[str]:
        scope = self.scope.parent
        while scope is not None:
            if scope.table.get_type() == "class":
                scope = scope.parent
                continue
            try:
                symbol = scope.table.lookup(name)
            except KeyError:
                scope = scope.parent
                continue
            if symbol.is_local() or symbol.is_parameter():
                return scope.final_bindings.get(name, {_UNBOUND})
            scope = scope.parent
        return {_UNBOUND}

    def _resolved_bindings(self, name: str, node: ast.AST) -> set[str]:
        try:
            symbol = self.scope.table.lookup(name)
        except KeyError:
            return {_UNBOUND}
        scope_type = self.scope.table.get_type()
        if scope_type == "module" or symbol.is_local() or symbol.is_parameter():
            return self.scope.bindings_before(name, node)
        if symbol.is_free() or symbol.is_nonlocal():
            return self._enclosing_bindings(name)
        if symbol.is_global():
            module = self._module_scope()
            if self.module_bindings is not None:
                return self.module_bindings.get(name, {_UNBOUND})
            return module.final_bindings.get(name, {_UNBOUND})
        return {_UNKNOWN_BINDING}

    def _name_is_proven_safe(self, name: str, node: ast.AST) -> bool:
        return _bindings_are_proven_safe(self._resolved_bindings(name, node))

    def _expression_is_proven_safe(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Constant, ast.Lambda)):
            return True
        if isinstance(node, ast.Name):
            return self._name_is_proven_safe(node.id, node)
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", maxsplit=1)[0] in {"importlib", "builtins"}:
                self._reject(node, f"forbidden dynamic API import {alias.name!r}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".", maxsplit=1)[0]
        imported = {alias.name for alias in node.names}
        if module == "importlib" or (
            module == "builtins" and imported & _DYNAMIC_BUILTINS_IMPORTS
        ) or (
            module == "sys" and "modules" in imported
        ):
            self._reject(
                node,
                f"forbidden dynamic API import from {(node.module or '')!r}",
            )

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        if node.id == "__builtins__":
            self._reject(node, "access to the builtin namespace is forbidden")
        elif node.id in _DYNAMIC_NAMESPACE_NAMES and not self._name_is_proven_safe(
            node.id, node
        ):
            self._reject(
                node,
                f"builtin namespace acquisition via {node.id!r} is forbidden",
            )
        elif node.id in _DYNAMIC_API_NAMES and not self._name_is_proven_safe(
            node.id, node
        ):
            self._reject(
                node,
                f"{node.id!r} does not resolve to a proven-safe user binding",
            )
        elif node.id == "getattr" and not self._name_is_proven_safe(
            node.id, node
        ):
            self._reject(node, "builtin getattr may only perform literal safe access")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "__dict__":
            self._reject(node, "dynamic namespace __dict__ access is forbidden")
        elif node.attr == "modules" and (
            isinstance(node.value, ast.Name)
            and _PROVEN_SYS_MODULE
            in self._resolved_bindings(node.value.id, node.value)
        ):
            self._reject(node, "sys.modules namespace access is forbidden")
        elif node.attr in _DYNAMIC_API_NAMES and not self._expression_is_proven_safe(
            node.value
        ):
            self._reject(
                node,
                f"cannot prove {node.attr!r} attribute acquisition is safe",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
            self._reject(node, "builtin namespace subscription is forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and not self._name_is_proven_safe("getattr", node.func)
        ):
            attribute = (
                node.args[1].value
                if len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                else None
            )
            if attribute is None:
                self._reject(node, "dynamic getattr attribute is not a literal string")
            elif attribute in _DYNAMIC_API_NAMES and (
                not node.args or not self._expression_is_proven_safe(node.args[0])
            ):
                self._reject(
                    node,
                    f"cannot prove getattr acquisition of {attribute!r} is safe",
                )
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        if self.import_time:
            if isinstance(node.func, ast.Name):
                callable_name = node.func.id
                bindings = self._resolved_bindings(node.func.id, node.func)
            elif isinstance(node.func, ast.Lambda):
                callable_name = "lambda"
                binding = _local_function_binding(node.func)
                self.scope.function_definitions[binding] = (node.func, self.scope)
                bindings = {binding}
            else:
                callable_name = "callable"
                bindings = set()
            function_bindings = sorted(
                binding
                for binding in bindings
                if binding.startswith(_LOCAL_FUNCTION_PREFIX)
            )
            for binding in function_bindings:
                if binding in self.call_stack:
                    continue
                definition = self.scope.function_definitions.get(binding)
                if definition is None:
                    self._reject(
                        node,
                        f"cannot resolve import-time local call {callable_name!r}",
                    )
                    break
                function, defining_scope = definition
                module_bindings = self.module_bindings
                if module_bindings is None:
                    module_bindings = self._module_scope().environment_before(node)
                called = _DynamicAccessVisitor(
                    defining_scope.child(function),
                    module_bindings=module_bindings,
                    import_time=True,
                    call_stack=self.call_stack | {binding},
                )
                if isinstance(function, ast.Lambda):
                    called.visit(function.body)
                else:
                    for statement in function.body:
                        called.visit(statement)
                if called.reason is not None:
                    self.reason = called.reason
                    break
        self.generic_visit(node)

    def _visit_function_header(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        )
        for argument in arguments:
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_nested_scope(self, node: ast.AST, body: list[ast.stmt]) -> None:
        child_visitor = _DynamicAccessVisitor(self.scope.child(node))
        for statement in body:
            child_visitor.visit(statement)
        if self.reason is None:
            self.reason = child_visitor.reason

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_header(node)
        if not self.import_time:
            self._visit_nested_scope(node, node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_header(node)
        if not self.import_time:
            self._visit_nested_scope(node, node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        if self.import_time:
            module_bindings = self.module_bindings
            if module_bindings is None:
                module_bindings = self._module_scope().environment_before(node)
            child_visitor = _DynamicAccessVisitor(
                self.scope.child(node),
                module_bindings=module_bindings,
                import_time=True,
                call_stack=self.call_stack,
            )
            for statement in node.body:
                child_visitor.visit(statement)
            if self.reason is None:
                self.reason = child_visitor.reason
        else:
            self._visit_nested_scope(node, node.body)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if not self.import_time:
            child_visitor = _DynamicAccessVisitor(self.scope.child(node))
            child_visitor.visit(node.body)
            if self.reason is None:
                self.reason = child_visitor.reason


def _dynamic_execution_reason(
    tree: ast.Module, source: str, filename: str
) -> str | None:
    # No file-level allowlist exists. A future exception must resolve and bind
    # an exact target rather than exempting an importer wholesale.
    table = symtable.symtable(source, filename, "exec")
    root_scope = _AccessScope(tree, table, None)
    visitor = _DynamicAccessVisitor(root_scope)
    visitor.visit(tree)
    if visitor.reason is not None:
        return visitor.reason
    import_time_visitor = _DynamicAccessVisitor(root_scope, import_time=True)
    import_time_visitor.visit(tree)
    return import_time_visitor.reason


def validate_decision_import_closure(
    project_root: Path,
    *,
    roots: tuple[str, ...] = DECISION_IMPORT_ROOTS,
    manifest: dict[str, str] = DECISION_CODE_PATHS,
) -> frozenset[str]:
    """Discover static local imports and reject manifest omissions."""

    root = project_root.resolve()
    pending = [root / relative for relative in roots]
    discovered: set[str] = set()
    while pending:
        path = pending.pop()
        if not path.is_file():
            raise ValueError(f"decision import closure file is missing: {path}")
        relative = _relative_project_path(root, path)
        if relative in discovered:
            continue
        discovered.add(relative)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        dynamic_reason = _dynamic_execution_reason(tree, source, relative)
        if dynamic_reason is not None:
            raise ValueError(
                "dynamic import or code execution in "
                f"{relative}: {dynamic_reason} is forbidden"
            )
        pending.extend(_package_initializers(root, path))
        pending.extend(_imported_local_paths(root, path, tree))
    omissions = discovered - set(manifest.values())
    if omissions:
        raise ValueError(
            "decision import closure is missing manifest entries: "
            + ", ".join(sorted(omissions))
        )
    return frozenset(discovered)


def _invalid(identity: object, message: str) -> ValueError:
    return ValueError(f"severance policy {identity}: {message}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _non_blank(value: object, *, field: str, identity: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(identity, f"{field} must be a non-blank string")
    return value.strip()


def _boolean(value: object, *, field: str, identity: object) -> bool:
    if type(value) is not bool:
        raise _invalid(identity, f"{field} must be boolean")
    return value


def _unit_interval(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be finite and between zero and one")
    return normalized


def _validated_global_threshold(value: object) -> float:
    normalized = _unit_interval(value, field="global_threshold")
    if normalized != _GLOBAL_THRESHOLD:
        raise ValueError("global_threshold must equal the committed 0.03")
    return normalized


def _strings(
    value: object,
    *,
    field: str,
    identity: object,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _invalid(identity, f"{field} must be a list")
    if not allow_empty and not value:
        raise _invalid(identity, f"{field} must not be empty")
    normalized = tuple(
        _non_blank(item, field=field, identity=identity) for item in value
    )
    if len(normalized) != len(set(normalized)):
        raise _invalid(identity, f"{field} contains duplicates")
    return normalized


def _routes(value: object, *, field: str, require_tuple: bool) -> tuple[str, ...]:
    expected_type = tuple if require_tuple else list
    if not isinstance(value, expected_type):
        raise ValueError(f"{field} must be a {expected_type.__name__}")
    if not all(isinstance(route, str) and route.strip() for route in value):
        raise ValueError(f"{field} must contain non-blank strings")
    normalized = tuple(route.strip() for route in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} contains duplicates")
    if not set(normalized) <= _KNOWN_ROUTES:
        raise ValueError(f"{field} values must belong to the route allowlist")
    return normalized


def _source_key(source: dict[str, str]) -> str:
    return f"{source['law']}|{source['article']}"


def _sources(value: object, *, identity: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise _invalid(identity, "sources must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {"law", "article"}:
            raise _invalid(identity, "source fields must equal law and article")
        normalized_source = {
            "law": _non_blank(
                source["law"], field="source law", identity=identity
            ),
            "article": _non_blank(
                source["article"], field="source article", identity=identity
            ),
        }
        key = _source_key(normalized_source)
        if key in seen:
            raise _invalid(identity, "sources contain a duplicate source")
        seen.add(key)
        normalized.append(normalized_source)
    return tuple(normalized)


def _parse_case(row: object, index: int) -> SeverancePolicyCase:
    identity: object = f"row {index}"
    if not isinstance(row, dict):
        raise _invalid(identity, "must be an object")
    identity = row.get("qid", identity)
    if set(row) != _DATASET_FIELDS:
        raise _invalid(identity, f"fields must equal {sorted(_DATASET_FIELDS)}")
    qid = _non_blank(row["qid"], field="qid", identity=identity)
    question = _non_blank(row["question"], field="question", identity=qid)
    case_type = row["case_type"]
    expected_type = "positive" if index <= 15 else "collision_negative"
    if case_type != expected_type:
        raise _invalid(qid, "case_type ordering must be fifteen positives then negatives")
    answerable = _boolean(row["answerable"], field="answerable", identity=qid)
    expected_outcome = row["expected_outcome"]
    if expected_outcome not in {"generation", "no_hits", "threshold"}:
        raise _invalid(qid, "expected_outcome must be a supported exact outcome")
    sources = _sources(row["sources"], identity=qid)
    required_routes = _strings(
        row["required_routes"],
        field="required_routes",
        identity=qid,
        allow_empty=True,
    )
    prohibited_routes = _strings(
        row["prohibited_routes"],
        field="prohibited_routes",
        identity=qid,
        allow_empty=True,
    )
    style_tags = _strings(
        row["style_tags"], field="style_tags", identity=qid, allow_empty=False
    )
    if set(required_routes) & set(prohibited_routes):
        raise _invalid(qid, "required_routes overlap prohibited_routes")
    if not set(required_routes + prohibited_routes) <= _KNOWN_ROUTES:
        raise _invalid(qid, "route contract contains an unknown route")
    contract = _CASE_CONTRACTS.get(qid)
    actual_contract = (
        case_type,
        answerable,
        tuple(_source_key(source) for source in sources),
        required_routes,
        prohibited_routes,
        expected_outcome,
    )
    expected_contract = (
        contract.case_type,
        contract.answerable,
        contract.source_keys,
        contract.required_routes,
        contract.prohibited_routes,
        contract.expected_outcome,
    ) if contract else None
    if actual_contract != expected_contract:
        raise _invalid(qid, "does not match its canonical contract")
    return SeverancePolicyCase(
        qid=qid,
        question=question,
        case_type=case_type,
        answerable=answerable,
        sources=sources,
        required_routes=required_routes,
        prohibited_routes=prohibited_routes,
        expected_outcome=expected_outcome,
        style_tags=style_tags,
    )


def load_cases(path: Path) -> list[SeverancePolicyCase]:
    """Load and validate the exact thirty reviewed cases."""

    rows: list[object] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, object_pairs_hook=_strict_object))
        except (json.JSONDecodeError, ValueError) as exc:
            raise _invalid(f"row {index}", "must be strict valid JSON") from exc
    cases = [_parse_case(row, index) for index, row in enumerate(rows, start=1)]
    qids = [case.qid for case in cases]
    if len(qids) != len(set(qids)):
        raise _invalid("dataset", "contains duplicate qids")
    if tuple(qids) != EXPECTED_QIDS:
        raise _invalid("dataset", "qids must be severance-policy-001 through -030")
    if not _POSITIVE_STYLES <= {
        tag for case in cases[:15] for tag in case.style_tags
    }:
        raise _invalid("dataset", "positive style coverage is incomplete")
    if not _COLLISION_STYLES <= {
        tag for case in cases[15:] for tag in case.style_tags
    }:
        raise _invalid("dataset", "collision style coverage is incomplete")
    return cases


def _validated_source_ranks(
    value: object, *, qid: str, allowed: tuple[str, ...]
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{qid}: source_ranks must be a dict")
    if not set(value) <= set(allowed):
        raise ValueError(f"{qid}: source_ranks keys are not canonical")
    normalized: dict[str, int] = {}
    for source, rank in value.items():
        if type(rank) is not int or rank < 1:
            raise ValueError(f"{qid}: source_ranks require positive integer ranks")
        normalized[source] = rank
    return dict(sorted(normalized.items()))


def _case_matches_contract(case: SeverancePolicyCase) -> _CaseContract:
    contract = _CASE_CONTRACTS.get(case.qid)
    if contract is None:
        raise ValueError("case qid is not canonical")
    actual = (
        case.case_type,
        case.answerable,
        tuple(_source_key(source) for source in case.sources),
        case.required_routes,
        case.prohibited_routes,
        case.expected_outcome,
    )
    expected = (
        contract.case_type,
        contract.answerable,
        contract.source_keys,
        contract.required_routes,
        contract.prohibited_routes,
        contract.expected_outcome,
    )
    if actual != expected:
        raise ValueError(f"{case.qid}: case does not match canonical contract")
    return contract


def build_case_observation(
    case: SeverancePolicyCase,
    *,
    source_ranks: dict[str, int],
    applied_routes: tuple[str, ...],
    top_score: float,
    hit_count: int,
    candidate_count: int,
    route_plan_matched: bool,
    first_stage_retrieval_calls: int,
    reranker_calls: int,
    reranker_scored_pairs: tuple[int, ...],
) -> dict[str, Any]:
    """Return one validated content-free retrieval observation."""

    if not isinstance(case, SeverancePolicyCase):
        raise ValueError("case must be a SeverancePolicyCase")
    contract = _case_matches_contract(case)
    ranks = _validated_source_ranks(
        source_ranks, qid=case.qid, allowed=contract.source_keys
    )
    routes = _routes(applied_routes, field="applied_routes", require_tuple=True)
    if type(hit_count) is not int or not 0 <= hit_count <= _TOP_K_FINAL:
        raise ValueError(
            f"{case.qid}: hit_count must be between zero and {_TOP_K_FINAL}"
        )
    score = _unit_interval(top_score, field="top_score")
    if hit_count == 0 and (ranks or score != 0.0):
        raise ValueError(
            f"{case.qid}: zero-hit observation requires no ranks and zero score"
        )
    if hit_count > 0 and score == 0.0:
        raise ValueError(
            f"{case.qid}: positive hit_count requires a positive score"
        )
    if any(rank > hit_count for rank in ranks.values()):
        raise ValueError(f"{case.qid}: source rank must not exceed hit_count")
    execution = _validated_execution_evidence(
        qid=case.qid,
        routes=routes,
        hit_count=hit_count,
        candidate_count=candidate_count,
        route_plan_matched=route_plan_matched,
        first_stage_retrieval_calls=first_stage_retrieval_calls,
        reranker_calls=reranker_calls,
        reranker_scored_pairs=reranker_scored_pairs,
        require_tuple=True,
    )
    return {
        "qid": case.qid,
        "source_ranks": ranks,
        "applied_routes": list(routes),
        "hit_count": hit_count,
        "top_score": score,
        **execution,
    }


def _validated_execution_evidence(
    *,
    qid: str,
    routes: tuple[str, ...],
    hit_count: int,
    candidate_count: object,
    route_plan_matched: object,
    first_stage_retrieval_calls: object,
    reranker_calls: object,
    reranker_scored_pairs: object,
    require_tuple: bool,
) -> dict[str, Any]:
    if type(candidate_count) is not int or not 0 <= candidate_count <= 20:
        raise ValueError(f"{qid}: candidate_count must be between zero and twenty")
    if hit_count > candidate_count or (candidate_count == 0) != (hit_count == 0):
        raise ValueError(f"{qid}: candidate_count and hit_count disagree")
    if route_plan_matched is not True:
        raise ValueError(f"{qid}: route_plan_matched must be true")
    if first_stage_retrieval_calls != 1 or type(first_stage_retrieval_calls) is not int:
        raise ValueError(f"{qid}: first_stage_retrieval_calls must equal one")
    expected_type = tuple if require_tuple else list
    if not isinstance(reranker_scored_pairs, expected_type):
        raise ValueError(
            f"{qid}: reranker_scored_pairs must be a {expected_type.__name__}"
        )
    pairs = tuple(reranker_scored_pairs)
    expected_calls = 0 if candidate_count == 0 else 2 if routes == _SEVERANCE_ROUTE else 1
    expected_pairs = () if expected_calls == 0 else (candidate_count,) * expected_calls
    if type(reranker_calls) is not int or reranker_calls != expected_calls:
        raise ValueError(f"{qid}: reranker_calls do not match the exact route contract")
    if pairs != expected_pairs:
        raise ValueError(f"{qid}: reranker_scored_pairs do not match candidate_count")
    return {
        "candidate_count": candidate_count,
        "route_plan_matched": True,
        "first_stage_retrieval_calls": 1,
        "reranker_calls": expected_calls,
        "reranker_scored_pairs": list(pairs),
    }


def _validated_observations(observations: object) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    if len(observations) != 30:
        raise ValueError("observations must contain thirty rows")
    normalized = []
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or set(row) != _OBSERVATION_FIELDS:
            raise ValueError(f"observation fields are invalid at row {index + 1}")
        qid = row["qid"]
        if qid != EXPECTED_QIDS[index]:
            raise ValueError("observations must contain the exact thirty qids in order")
        contract = _CASE_CONTRACTS[qid]
        normalized.append(
            {
                "qid": qid,
                "source_ranks": _validated_source_ranks(
                    row["source_ranks"], qid=qid, allowed=contract.source_keys
                ),
                "applied_routes": list(
                    _routes(
                        row["applied_routes"],
                        field="applied_routes",
                        require_tuple=False,
                    )
                ),
                "hit_count": row["hit_count"],
                "top_score": _unit_interval(row["top_score"], field="top_score"),
            }
        )
        hit_count = normalized[-1]["hit_count"]
        if type(hit_count) is not int or not 0 <= hit_count <= _TOP_K_FINAL:
            raise ValueError(
                f"{qid}: hit_count must be between zero and {_TOP_K_FINAL}"
            )
        if hit_count == 0 and (
            normalized[-1]["source_ranks"] or normalized[-1]["top_score"] != 0.0
        ):
            raise ValueError(
                f"{qid}: zero-hit observation requires no ranks and zero score"
            )
        if hit_count > 0 and normalized[-1]["top_score"] == 0.0:
            raise ValueError(f"{qid}: positive hit_count requires a positive score")
        if any(
            rank > hit_count for rank in normalized[-1]["source_ranks"].values()
        ):
            raise ValueError(f"{qid}: source rank must not exceed hit_count")
        normalized[-1].update(
            _validated_execution_evidence(
                qid=qid,
                routes=tuple(normalized[-1]["applied_routes"]),
                hit_count=hit_count,
                candidate_count=row["candidate_count"],
                route_plan_matched=row["route_plan_matched"],
                first_stage_retrieval_calls=row["first_stage_retrieval_calls"],
                reranker_calls=row["reranker_calls"],
                reranker_scored_pairs=row["reranker_scored_pairs"],
                require_tuple=False,
            )
        )
    return normalized


def _expected_guard_answerability(label: str, index: int) -> bool:
    return index < (40 if label == "stress" else 30)


def _validated_guard_rows(
    rows: object,
    *,
    label: str,
    published_evidence: bool = False,
) -> list[dict[str, Any]]:
    expected_qids = _STRESS_QIDS if label == "stress" else _FORMAL_QIDS
    expected_routes = _STRESS_ROUTES if label == "stress" else _FORMAL_ROUTES
    expected_fields = (
        _GUARD_EVIDENCE_FIELDS if published_evidence else _GUARD_INPUT_FIELDS
    )
    if not isinstance(rows, list) or len(rows) != len(expected_qids):
        raise ValueError(f"{label} rows must contain the exact committed qids")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"{label} rows have invalid fields at row {index + 1}")
        qid = row["qid"]
        if qid != expected_qids[index]:
            raise ValueError(f"{label} rows must contain the exact committed qids")
        answerable = _boolean(row["answerable"], field="answerable", identity=qid)
        if answerable is not _expected_guard_answerability(label, index):
            raise ValueError(f"{label} answerability is not canonical for {qid}")
        rank = row["rank"]
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError(f"{label} row {qid}: rank must be null or positive")
        if not answerable and rank is not None:
            raise ValueError(f"{label} row {qid}: unanswerable rank must be null")
        hit_count = row["hit_count"]
        if (
            type(hit_count) is not int
            or hit_count < 0
            or hit_count > _TOP_K_FINAL
        ):
            raise ValueError(
                f"{label} row {qid}: hit_count must be between zero and "
                f"{_TOP_K_FINAL}"
            )
        top_score = _unit_interval(row["top_score"], field="top_score")
        if hit_count == 0 and (top_score != 0.0 or rank is not None):
            raise ValueError(
                f"{label} row {qid}: zero-hit rows require zero score and null rank"
            )
        if hit_count > 0 and top_score == 0.0:
            raise ValueError(
                f"{label} row {qid}: positive hit_count requires a positive score"
            )
        if rank is not None and rank > hit_count:
            raise ValueError(f"{label} row {qid}: rank must not exceed hit_count")
        has_hits = hit_count > 0
        if published_evidence:
            if row["has_hits"] is not has_hits:
                raise ValueError(f"{label} row {qid}: has_hits derivation mismatch")
            if row["reranker_enabled"] is not True:
                raise ValueError(
                    f"{label} row {qid}: reranker_enabled must match configuration"
                )
        routes = _routes(
            row["applied_routes"], field="applied_routes", require_tuple=False
        )
        if routes != expected_routes.get(qid, ()):
            raise ValueError(f"{label} route identity is not canonical for {qid}")
        normalized.append(
            {
                "qid": qid,
                "answerable": answerable,
                "rank": rank,
                "hit_count": hit_count,
                "has_hits": has_hits,
                "reranker_enabled": True,
                "top_score": top_score,
                "applied_routes": list(routes),
            }
        )
        normalized[-1].update(
            _validated_execution_evidence(
                qid=qid,
                routes=routes,
                hit_count=hit_count,
                candidate_count=row["candidate_count"],
                route_plan_matched=row["route_plan_matched"],
                first_stage_retrieval_calls=row["first_stage_retrieval_calls"],
                reranker_calls=row["reranker_calls"],
                reranker_scored_pairs=row["reranker_scored_pairs"],
                require_tuple=False,
            )
        )
    if not any(row["top_score"] != round(row["top_score"], 4) for row in normalized):
        raise ValueError(
            f"{label} guard scores cannot be entirely four-decimal values"
        )
    return normalized


def _route_ablation_decision(
    *,
    has_hits: bool,
    routes: list[str],
    score: float,
    candidate: float,
    global_threshold: float,
):
    evaluation_threshold = (
        candidate if tuple(routes) == _SEVERANCE_ROUTE else global_threshold
    )
    return decide_retrieval_refusal(
        has_hits=has_hits,
        reranker_enabled=True,
        applied_routes=tuple(routes),
        top_score=score,
        global_threshold=evaluation_threshold,
    )


def _evaluate_target(
    observations: list[dict[str, Any]],
    *,
    candidate: float,
    global_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_results = []
    for row in observations:
        contract = _CASE_CONTRACTS[row["qid"]]
        source_contract = all(
            row["source_ranks"].get(source, 6) <= 5
            for source in contract.source_keys
        )
        applied_routes = set(row["applied_routes"])
        if contract.case_type == "positive":
            route_contract = tuple(row["applied_routes"]) == _SEVERANCE_ROUTE
        elif row["qid"] == "severance-policy-027":
            route_contract = tuple(row["applied_routes"]) == ()
        else:
            route_contract = set(contract.required_routes) <= applied_routes and not (
                set(contract.prohibited_routes) & applied_routes
            )
        decision = _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        )
        generation_allowed = not decision.refused
        actual_outcome = (
            "generation" if decision.refusal_stage is None else decision.refusal_stage
        )
        outcome_contract = actual_outcome == contract.expected_outcome
        passed = route_contract and outcome_contract and (
            source_contract or contract.case_type == "collision_negative"
        )
        case_results.append(
            {
                "qid": row["qid"],
                "case_type": contract.case_type,
                "answerable": contract.answerable,
                "source_ranks": dict(row["source_ranks"]),
                "applied_routes": list(row["applied_routes"]),
                "hit_count": row["hit_count"],
                "top_score": row["top_score"],
                "candidate_count": row["candidate_count"],
                "route_plan_matched": row["route_plan_matched"],
                "first_stage_retrieval_calls": row["first_stage_retrieval_calls"],
                "reranker_calls": row["reranker_calls"],
                "reranker_scored_pairs": list(row["reranker_scored_pairs"]),
                "effective_threshold": decision.effective_threshold,
                "refused": decision.refused,
                "refusal_stage": decision.refusal_stage,
                "source_contract_passed": source_contract,
                "route_contract_passed": route_contract,
                "expected_outcome": contract.expected_outcome,
                "generation_allowed": generation_allowed,
                "outcome_contract_passed": outcome_contract,
                "passed": passed,
            }
        )
    positives = case_results[:15]
    collisions = case_results[15:]
    passed_cases = sum(row["passed"] for row in case_results)
    summary = {
        "total": 30,
        "passed_cases": passed_cases,
        "positive_routes": sum(row["route_contract_passed"] for row in positives),
        "positive_sources_at_5": sum(
            row["source_contract_passed"] for row in positives
        ),
        "positive_generation_allowed": sum(
            row["generation_allowed"] for row in positives
        ),
        "collision_contracts": sum(
            row["route_contract_passed"]
            and row["outcome_contract_passed"]
            for row in collisions
        ),
        "passed": passed_cases == 30,
    }
    return case_results, summary


def _stress_summary(
    rows: list[dict[str, Any]], *, candidate: float, global_threshold: float
) -> dict[str, Any]:
    decisions = [
        _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        ).refused
        for row in rows
    ]
    false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    unanswerable_refusals = sum(
        refused and not row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    return {
        "questions": 60,
        "answerable": 40,
        "unanswerable": 20,
        "direct_false_refusals": false_refusals,
        "direct_unanswerable_refusals": unanswerable_refusals,
        "direct_unanswerable_coverage": unanswerable_refusals / 20,
        "passed": false_refusals == 0 and unanswerable_refusals >= 17,
    }


def _formal_summary(
    rows: list[dict[str, Any]], *, candidate: float, global_threshold: float
) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    decisions = [
        _route_ablation_decision(
            has_hits=row["hit_count"] > 0,
            routes=row["applied_routes"],
            score=row["top_score"],
            candidate=candidate,
            global_threshold=global_threshold,
        ).refused
        for row in rows
    ]
    false_refusals = sum(
        refused and row["answerable"]
        for row, refused in zip(rows, decisions, strict=True)
    )
    hit_at_5 = sum(
        row["rank"] is not None and row["rank"] <= 5 for row in answerable
    ) / 30
    mrr_at_10 = sum(
        1 / row["rank"]
        for row in answerable
        if row["rank"] is not None and row["rank"] <= 10
    ) / 30
    return {
        "questions": 40,
        "answerable": 30,
        "unanswerable": 10,
        "hit_at_5": hit_at_5,
        "mrr_at_10": mrr_at_10,
        "direct_false_refusals": false_refusals,
        "passed": hit_at_5 >= FORMAL_HIT_AT_5_BASELINE
        and mrr_at_10 >= FORMAL_MRR_AT_10_BASELINE
        and false_refusals == 0,
    }


def evaluate_route_ablation_candidate(
    observations: list[dict[str, Any]],
    *,
    candidate_threshold: float,
    global_threshold: float,
    stress_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute target and guard gates through the shared policy."""

    candidate = _unit_interval(candidate_threshold, field="candidate_threshold")
    if candidate not in CANDIDATE_THRESHOLDS:
        raise ValueError("candidate_threshold must belong to the committed grid")
    global_value = _validated_global_threshold(global_threshold)
    target_evidence = _validated_observations(observations)
    stress_evidence = _validated_guard_rows(stress_rows, label="stress")
    formal_evidence = _validated_guard_rows(formal_rows, label="formal")
    cases, target = _evaluate_target(
        target_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    stress = _stress_summary(
        stress_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    formal = _formal_summary(
        formal_evidence,
        candidate=candidate,
        global_threshold=global_value,
    )
    passed = target["passed"] and stress["passed"] and formal["passed"]
    return {
        "candidate_threshold": candidate,
        "global_threshold": global_value,
        "target": target,
        "stress": stress,
        "formal": formal,
        "cases": cases,
        "stress_evidence": stress_evidence,
        "formal_evidence": formal_evidence,
        "passed": passed,
    }


# Compatibility for callers predating the v0.3.6 retrieval-coverage pivot.
evaluate_candidate = evaluate_route_ablation_candidate


def _observations_from_case_results(cases: object) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) != 30:
        raise ValueError("candidate cases must contain thirty rows")
    observations = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != _CASE_RESULT_FIELDS:
            raise ValueError(f"candidate case fields are invalid at row {index + 1}")
        observations.append(
            {
                "qid": row["qid"],
                "source_ranks": row["source_ranks"],
                "applied_routes": row["applied_routes"],
                "hit_count": row["hit_count"],
                "top_score": row["top_score"],
                "candidate_count": row["candidate_count"],
                "route_plan_matched": row["route_plan_matched"],
                "first_stage_retrieval_calls": row["first_stage_retrieval_calls"],
                "reranker_calls": row["reranker_calls"],
                "reranker_scored_pairs": row["reranker_scored_pairs"],
            }
        )
    return _validated_observations(observations)


def _recompute_candidate(result: object) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != _CANDIDATE_FIELDS:
        raise ValueError("candidate result fields are invalid")
    candidate = _unit_interval(
        result["candidate_threshold"], field="candidate_threshold"
    )
    global_threshold = _validated_global_threshold(result["global_threshold"])
    observations = _observations_from_case_results(result["cases"])
    stress_evidence = _validated_guard_rows(
        result["stress_evidence"], label="stress", published_evidence=True
    )
    formal_evidence = _validated_guard_rows(
        result["formal_evidence"], label="formal", published_evidence=True
    )
    cases, target = _evaluate_target(
        observations,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    stress = _stress_summary(
        stress_evidence,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    formal = _formal_summary(
        formal_evidence,
        candidate=candidate,
        global_threshold=global_threshold,
    )
    passed = target["passed"] and stress["passed"] and formal["passed"]
    if result["cases"] != cases:
        raise ValueError(f"candidate {candidate}: case decision mismatch")
    if result["target"] != target:
        raise ValueError(f"candidate {candidate}: target aggregate mismatch")
    if result["stress"] != stress:
        raise ValueError(f"candidate {candidate}: stress aggregate mismatch")
    if result["formal"] != formal:
        raise ValueError(f"candidate {candidate}: formal aggregate mismatch")
    if type(result["passed"]) is not bool or result["passed"] is not passed:
        raise ValueError(f"candidate {candidate}: complete gate mismatch")
    return {
        "candidate_threshold": candidate,
        "global_threshold": global_threshold,
        "target": target,
        "stress": stress,
        "formal": formal,
        "cases": cases,
        "stress_evidence": stress_evidence,
        "formal_evidence": formal_evidence,
        "passed": passed,
    }


def _validated_candidate_results(
    candidate_results: object,
) -> list[dict[str, Any]]:
    if not isinstance(candidate_results, list):
        raise ValueError("candidate_results must be a list")
    results = [_recompute_candidate(result) for result in candidate_results]
    thresholds = [result["candidate_threshold"] for result in results]
    if len(thresholds) != len(set(thresholds)) or set(thresholds) != set(
        CANDIDATE_THRESHOLDS
    ):
        raise ValueError("candidate grid must equal the committed seven thresholds")
    first = results[0]
    first_observations = _observations_from_case_results(first["cases"])
    for result in results[1:]:
        if result["global_threshold"] != first["global_threshold"]:
            raise ValueError("candidate global threshold must be identical across grid")
        if _observations_from_case_results(result["cases"]) != first_observations:
            raise ValueError("candidate target evidence must be identical across grid")
        if result["stress_evidence"] != first["stress_evidence"]:
            raise ValueError("candidate stress evidence must be identical across grid")
        if result["formal_evidence"] != first["formal_evidence"]:
            raise ValueError("candidate formal evidence must be identical across grid")
    return sorted(results, key=lambda result: result["candidate_threshold"])


def _selected_threshold(results: list[dict[str, Any]]) -> float:
    passing = [result["candidate_threshold"] for result in results if result["passed"]]
    if not passing:
        raise RuntimeError("no candidate threshold satisfies the complete gate set")
    return max(passing)


def select_highest_passing_threshold(
    candidate_results: list[dict[str, Any]],
) -> float:
    """Return the greatest candidate whose complete gate set passes."""

    return _selected_threshold(_validated_candidate_results(candidate_results))


def _hex(value: object, *, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase {length}-character hex value")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_provenance(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError(f"provenance fields must equal {sorted(_PROVENANCE_FIELDS)}")
    source_hashes = provenance["source_artifact_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != _SOURCE_ARTIFACT_FIELDS:
        raise ValueError("source_artifact_sha256 fields are invalid")
    normalized_hashes = {
        field: _hex(source_hashes[field], field=field, length=64)
        for field in sorted(_SOURCE_ARTIFACT_FIELDS)
    }
    decision_code_hashes = provenance["decision_code_sha256"]
    if not isinstance(decision_code_hashes, dict) or set(
        decision_code_hashes
    ) != set(DECISION_CODE_PATHS):
        raise ValueError("decision_code_sha256 fields are invalid")
    normalized_code_hashes = {
        field: _hex(decision_code_hashes[field], field=field, length=64)
        for field in sorted(DECISION_CODE_PATHS)
    }
    exact_strings = {
        "embedding_model": _EMBEDDING_MODEL,
        "embedding_revision": _EMBEDDING_REVISION,
        "reranker_model": _RERANKER_MODEL,
        "reranker_revision": _RERANKER_REVISION,
    }
    for field, expected in exact_strings.items():
        if provenance[field] != expected:
            raise ValueError(f"{field} must equal the approved pinned value")
    configuration = provenance["retrieval_configuration"]
    if not isinstance(configuration, dict) or set(configuration) != set(
        _RETRIEVAL_CONFIGURATION
    ):
        raise ValueError("retrieval_configuration fields are invalid")
    if configuration != _RETRIEVAL_CONFIGURATION:
        raise ValueError("retrieval_configuration must equal the approved primitives")
    if provenance["run_origin"] != _RUN_ORIGIN:
        raise ValueError("run_origin must equal fresh_offline_retrieval")
    execution_device = provenance["execution_device"]
    if execution_device != "cpu":
        raise ValueError("execution_device must equal cpu for authoritative evidence")
    if provenance["precision_mode"] != _PRECISION_MODE:
        raise ValueError("precision_mode must equal fp32")
    if provenance["local_files_only"] is not True:
        raise ValueError("local_files_only must be true")
    if provenance["semantic_view_sha256"] != _SEMANTIC_VIEW_SHA256:
        raise ValueError("semantic_view_sha256 must equal the approved view hash")
    if provenance["merge_policy_version"] != _MERGE_POLICY_VERSION:
        raise ValueError("merge_policy_version must equal the approved version")
    if provenance["primary_score_semantics"] != _PRIMARY_SCORE_SEMANTICS:
        raise ValueError("primary_score_semantics must bind full-precision PRIMARY scores")
    if provenance["source_tree_clean"] is not True:
        raise ValueError("source_tree_clean must be true")
    for field in ("provider_adapters", "provider_requests"):
        if type(provenance[field]) is not int or provenance[field] != 0:
            raise ValueError(f"{field} must be zero")
    return {
        "dataset_sha256": _hex(
            provenance["dataset_sha256"], field="dataset_sha256", length=64
        ),
        "corpus_snapshot_sha256": _hex(
            provenance["corpus_snapshot_sha256"],
            field="corpus_snapshot_sha256",
            length=64,
        ),
        "source_artifact_sha256": normalized_hashes,
        "decision_code_sha256": normalized_code_hashes,
        **exact_strings,
        "retrieval_configuration": dict(_RETRIEVAL_CONFIGURATION),
        "execution_device": execution_device,
        "precision_mode": _PRECISION_MODE,
        "local_files_only": True,
        "semantic_view_sha256": _SEMANTIC_VIEW_SHA256,
        "merge_policy_version": _MERGE_POLICY_VERSION,
        "primary_score_semantics": _PRIMARY_SCORE_SEMANTICS,
        "source_tree_clean": True,
        "code_revision": _hex(
            provenance["code_revision"], field="code_revision", length=40
        ),
        "run_origin": _RUN_ORIGIN,
        "provider_adapters": 0,
        "provider_requests": 0,
    }


_PUBLIC_KEYS = {
    "schema_version",
    "provenance",
    "candidate_thresholds",
    "production_threshold",
    "route_ablation",
    "highest_passing_candidate",
    "guard_evidence",
    "guard_evidence_binding_sha256",
    "target_evidence_binding_sha256",
    "candidates",
    "cases",
    "evidence_class",
    "outcome",
    "official_export_allowed",
    "target_observations",
    "failed_gates",
    "gates",
    "candidate_threshold",
    "target",
    "stress",
    "formal",
    "passed",
    "total",
    "passed_cases",
    "positive_routes",
    "positive_sources_at_5",
    "positive_generation_allowed",
    "collision_contracts",
    "questions",
    "answerable",
    "unanswerable",
    "direct_false_refusals",
    "direct_unanswerable_refusals",
    "direct_unanswerable_coverage",
    "hit_at_5",
    "mrr_at_10",
    "qid",
    "case_type",
    "rank",
    "hit_count",
    "has_hits",
    "reranker_enabled",
    "source_ranks",
    "applied_routes",
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
    *_PROVENANCE_FIELDS,
    *_SOURCE_ARTIFACT_FIELDS,
    *DECISION_CODE_PATHS,
    *_RETRIEVAL_CONFIGURATION,
    *_CANONICAL_SOURCE_KEYS,
}
_PUBLIC_STRINGS = {
    _SCHEMA_VERSION,
    "non_release_pivot_no_go",
    "no_go",
    "target",
    "stress",
    "formal",
    "route_ablation",
    "positive",
    "collision_negative",
    "threshold",
    "no_hits",
    "generation",
    "structure",
    "hybrid",
    _EMBEDDING_MODEL,
    _EMBEDDING_REVISION,
    _RERANKER_MODEL,
    _RERANKER_REVISION,
    _RUN_ORIGIN,
    _PRECISION_MODE,
    _MERGE_POLICY_VERSION,
    _PRIMARY_SCORE_SEMANTICS,
    "cpu",
    *EXPECTED_QIDS,
    *_STRESS_QIDS,
    *_FORMAL_QIDS,
    *_KNOWN_ROUTES,
}


def _validate_public_tree(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str) or key not in _PUBLIC_KEYS:
                raise ValueError("public artifact contains a non-allowlisted key")
            _validate_public_tree(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_public_tree(nested)
    elif isinstance(value, str):
        is_hash = len(value) in {40, 64} and all(
            character in "0123456789abcdef" for character in value
        )
        if value not in _PUBLIC_STRINGS and not is_hash:
            raise ValueError("public artifact contains a non-allowlisted string value")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("public artifact contains a non-finite number")
    elif value is not None and type(value) not in {bool, int, float}:
        raise ValueError("public artifact contains an unsupported value")


def build_official_artifact(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict content-free official schema."""

    target_evidence = _validated_observations(observations)
    candidates = _validated_candidate_results(candidate_results)
    selected_threshold = _selected_threshold(candidates)
    selected = next(
        result
        for result in candidates
        if result["candidate_threshold"] == selected_threshold
    )
    if _observations_from_case_results(selected["cases"]) != target_evidence:
        raise ValueError("selected candidate cases do not match target observations")
    normalized_provenance = _validated_provenance(provenance)
    guard_evidence = {
        label: [
            {**row, "applied_routes": list(row["applied_routes"])}
            for row in selected[f"{label}_evidence"]
        ]
        for label in ("stress", "formal")
    }
    guard_evidence_binding = _canonical_sha256(
        {
            "guard_evidence": guard_evidence,
            "provenance": normalized_provenance,
        }
    )
    public_cases = [
        {
            **row,
            "source_ranks": dict(row["source_ranks"]),
            "applied_routes": list(row["applied_routes"]),
        }
        for row in selected["cases"]
    ]
    public_candidates = [
        {
            "candidate_threshold": result["candidate_threshold"],
            "target": dict(result["target"]),
            "stress": dict(result["stress"]),
            "formal": dict(result["formal"]),
            "passed": result["passed"],
        }
        for result in candidates
    ]
    target_evidence_binding = _canonical_sha256(
        {
            "cases": public_cases,
            "provenance": normalized_provenance,
        }
    )
    artifact = {
        "schema_version": _SCHEMA_VERSION,
        "provenance": normalized_provenance,
        "production_threshold": selected["global_threshold"],
        "route_ablation": {
            "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
            "highest_passing_candidate": selected_threshold,
            "candidates": public_candidates,
        },
        "guard_evidence": guard_evidence,
        "guard_evidence_binding_sha256": guard_evidence_binding,
        "target_evidence_binding_sha256": target_evidence_binding,
        "cases": public_cases,
    }
    _validate_public_tree(artifact)
    return artifact


_OFFICIAL_FIELDS = {
    "schema_version",
    "provenance",
    "production_threshold",
    "route_ablation",
    "guard_evidence",
    "guard_evidence_binding_sha256",
    "target_evidence_binding_sha256",
    "cases",
}
_ROUTE_ABLATION_FIELDS = {
    "candidate_thresholds",
    "highest_passing_candidate",
    "candidates",
}


def replay_official_artifact(artifact: object) -> dict[str, Any]:
    """Recompute schema 1.3 acceptance without retrieval or model execution."""

    if not isinstance(artifact, dict) or set(artifact) != _OFFICIAL_FIELDS:
        raise ValueError("official artifact fields are invalid")
    if artifact["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("schema_version must equal 1.3")
    if artifact["production_threshold"] != _GLOBAL_THRESHOLD:
        raise ValueError("production_threshold must equal 0.03")
    production_threshold = _validated_global_threshold(artifact["production_threshold"])
    route_ablation = artifact["route_ablation"]
    if (
        not isinstance(route_ablation, dict)
        or set(route_ablation) != _ROUTE_ABLATION_FIELDS
    ):
        raise ValueError("route ablation fields are invalid")
    if route_ablation["candidate_thresholds"] != list(CANDIDATE_THRESHOLDS):
        raise ValueError("route ablation candidate grid mismatch")
    if route_ablation["highest_passing_candidate"] != _GLOBAL_THRESHOLD:
        raise ValueError("route ablation highest passing candidate must equal 0.03")
    provenance = _validated_provenance(artifact["provenance"])
    observations = _observations_from_case_results(artifact["cases"])
    guard_evidence = artifact["guard_evidence"]
    if not isinstance(guard_evidence, dict) or set(guard_evidence) != {
        "stress",
        "formal",
    }:
        raise ValueError("official guard evidence fields are invalid")
    stress = _validated_guard_rows(
        guard_evidence["stress"], label="stress", published_evidence=True
    )
    formal = _validated_guard_rows(
        guard_evidence["formal"], label="formal", published_evidence=True
    )
    candidates = [
        evaluate_route_ablation_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=production_threshold,
            stress_rows=_guard_inputs(stress),
            formal_rows=_guard_inputs(formal),
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    rebuilt = build_official_artifact(
        observations=observations,
        candidate_results=candidates,
        provenance=provenance,
    )
    if artifact != rebuilt:
        raise ValueError("official artifact replay mismatch")
    return rebuilt


_NO_GO_FIELDS = {
    "schema_version",
    "evidence_class",
    "outcome",
    "official_export_allowed",
    "provenance",
    "production_threshold",
    "route_ablation",
    "target_observations",
    "guard_evidence",
    "failed_gates",
}


def _guard_inputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in _GUARD_INPUT_FIELDS
        }
        for row in rows
    ]


def _candidate_summaries(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_threshold": result["candidate_threshold"],
            "target": dict(result["target"]),
            "stress": dict(result["stress"]),
            "formal": dict(result["formal"]),
            "passed": result["passed"],
        }
        for result in candidates
    ]


def build_no_go_evidence(
    *,
    observations: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic, replayable evidence for a non-release NO-GO."""

    target_evidence = _validated_observations(observations)
    candidates = _validated_candidate_results(candidate_results)
    if _observations_from_case_results(candidates[0]["cases"]) != target_evidence:
        raise ValueError("candidate target evidence does not match observations")
    passing = [
        result["candidate_threshold"] for result in candidates if result["passed"]
    ]
    selected = max(passing) if passing else None
    if selected == _GLOBAL_THRESHOLD:
        raise ValueError("passing 0.03 requires the official artifact")
    normalized_provenance = _validated_provenance(provenance)
    guard_evidence = {
        label: [dict(row) for row in candidates[0][f"{label}_evidence"]]
        for label in ("stress", "formal")
    }
    failed_gates = []
    for result in candidates:
        gates = [
            label
            for label in ("target", "stress", "formal")
            if not result[label]["passed"]
        ]
        if result["candidate_threshold"] == _GLOBAL_THRESHOLD:
            gates.append("route_ablation")
        if gates:
            failed_gates.append(
                {
                    "candidate_threshold": result["candidate_threshold"],
                    "gates": gates,
                }
            )
    envelope = {
        "schema_version": _SCHEMA_VERSION,
        "evidence_class": "non_release_pivot_no_go",
        "outcome": "no_go",
        "official_export_allowed": False,
        "provenance": normalized_provenance,
        "production_threshold": candidates[0]["global_threshold"],
        "route_ablation": {
            "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
            "highest_passing_candidate": selected,
            "candidates": _candidate_summaries(candidates),
        },
        "target_observations": target_evidence,
        "guard_evidence": guard_evidence,
        "failed_gates": failed_gates,
    }
    _validate_public_tree(envelope)
    return envelope


def replay_no_go_evidence(envelope: object) -> dict[str, Any]:
    """Recompute a NO-GO envelope without retrieval or model construction."""

    if not isinstance(envelope, dict) or set(envelope) != _NO_GO_FIELDS:
        raise ValueError("NO-GO evidence fields are invalid")
    provenance = _validated_provenance(envelope["provenance"])
    observations = _validated_observations(envelope["target_observations"])
    guard_evidence = envelope["guard_evidence"]
    if not isinstance(guard_evidence, dict) or set(guard_evidence) != {
        "stress",
        "formal",
    }:
        raise ValueError("NO-GO guard evidence fields are invalid")
    stress = _validated_guard_rows(
        guard_evidence["stress"], label="stress", published_evidence=True
    )
    formal = _validated_guard_rows(
        guard_evidence["formal"], label="formal", published_evidence=True
    )
    candidates = [
        evaluate_route_ablation_candidate(
            observations,
            candidate_threshold=threshold,
            global_threshold=envelope["production_threshold"],
            stress_rows=_guard_inputs(stress),
            formal_rows=_guard_inputs(formal),
        )
        for threshold in CANDIDATE_THRESHOLDS
    ]
    rebuilt = build_no_go_evidence(
        observations=observations,
        candidate_results=candidates,
        provenance=provenance,
    )
    if envelope != rebuilt:
        raise ValueError("NO-GO evidence replay mismatch")
    return rebuilt
