"""Gold-blind isolation and coverage gate for paired offline experiments.

The gate validates two already committed Replay/Benchmark runs.  It does not
execute retrieval, inspect result quality, or choose a common-success subset.
The comparison plan is a pre-run, content-addressed contract; the same digest
is bound into both ``run_manifest_v1`` documents and their generation-zero
configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    STORE_DIRECTORY,
    BenchmarkRunCommitStore,
    CrashConsistencyError,
    stable_json_bytes,
)
from scholar_agent.evaluation.run_provenance import (
    GitProvenance,
    RunManifestV1,
    build_run_manifest,
    resolve_repo_path,
    validate_run_manifest,
    write_json,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


PLAN_CONTRACT = "comparison_plan_v1"
GATE_CONTRACT = "experiment_pairing_integrity_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "experiment_pairing_integrity_gate"

EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE_ERROR = 4

_OPAQUE_QUERY_RE = re.compile(r"^query:[0-9a-f]{64}$")
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "excluded"})
_SOURCE_STATUSES = frozenset(
    {"success", "failed", "cancelled", "skipped", "partial", "not_started"}
)
_FORBIDDEN_RECORD_FIELDS = frozenset(
    {"gold", "qrels", "case_id_raw", "target_paper", "quality_metrics"}
)


class ExperimentPairingError(RuntimeError):
    """The pairing input is invalid or cannot be represented safely."""


class ExperimentPairingNotEligible(ExperimentPairingError):
    """Legacy evidence lacks a pre-bound comparison contract."""


class QueryPopulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1)
    identities: list[str] = Field(min_length=1)
    stable_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_population(self) -> "QueryPopulation":
        if self.count != len(self.identities):
            raise ValueError("query count mismatch")
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("duplicate query identity")
        if any(not _OPAQUE_QUERY_RE.fullmatch(value) for value in self.identities):
            raise ValueError("query identity must be opaque")
        return self


class TreatmentChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pointer: str
    baseline_value: bool | int | float | str | None
    candidate_value: bool | int | float | str | None

    @model_validator(mode="after")
    def validate_exact_leaf(self) -> "TreatmentChange":
        _pointer_parts(self.pointer)
        if not self.pointer.startswith("/configuration/values/"):
            raise ValueError("treatment pointer is outside exact configuration values")
        if self.pointer in {"/configuration/values", "/configuration"}:
            raise ValueError("broad treatment pointer is forbidden")
        if self.baseline_value == self.candidate_value:
            raise ValueError("declared treatment must change")
        return self


class PredeclaredExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_identity: str
    reason: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_identity(self) -> "PredeclaredExclusion":
        if not _OPAQUE_QUERY_RE.fullmatch(self.query_identity):
            raise ValueError("exclusion query identity must be opaque")
        return self


class ComparisonPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    contract: Literal["comparison_plan_v1"] = PLAN_CONTRACT
    score_scope: Literal[
        "pairing_only_not_quality_or_official_score"
    ] = "pairing_only_not_quality_or_official_score"
    plan_id: str = Field(min_length=1, max_length=100)
    queries: QueryPopulation
    data_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roles: dict[str, Literal["baseline", "candidate"]]
    allowed_treatment_changes: list[TreatmentChange] = Field(min_length=1)
    common_execution_contract: dict[str, Any]
    common_execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predeclared_exclusions: list[PredeclaredExclusion] = Field(default_factory=list)
    population_policy: Literal[
        "all_queries_including_failed_cancelled_and_predeclared_exclusions"
    ] = "all_queries_including_failed_cancelled_and_predeclared_exclusions"
    evaluator_usage: Literal["identity_only_no_quality_metrics"] = (
        "identity_only_no_quality_metrics"
    )
    gold_accessed: Literal[False] = False
    quality_metrics_computed: Literal[False] = False

    @model_validator(mode="after")
    def validate_closed_plan(self) -> "ComparisonPlanV1":
        if self.roles != {"baseline": "baseline", "candidate": "candidate"}:
            raise ValueError("comparison roles are not canonical")
        pointers = [item.pointer for item in self.allowed_treatment_changes]
        if pointers != sorted(pointers) or len(pointers) != len(set(pointers)):
            raise ValueError("treatment pointers must be sorted and unique")
        exclusion_ids = [item.query_identity for item in self.predeclared_exclusions]
        if exclusion_ids != sorted(exclusion_ids) or len(exclusion_ids) != len(
            set(exclusion_ids)
        ):
            raise ValueError("exclusions must be sorted and unique")
        if not set(exclusion_ids).issubset(self.queries.identities):
            raise ValueError("exclusion is outside query population")
        if stable_hash(self.common_execution_contract) != (
            self.common_execution_contract_sha256
        ):
            raise ValueError("common execution contract digest mismatch")
        common = self.common_execution_contract
        if set(common) != {
            "dataset",
            "queries",
            "prompt",
            "configuration",
            "evaluator",
            "determinism",
            "execution_profile",
            "coverage_policy",
        }:
            raise ValueError("common execution contract fields are not closed")
        if common["dataset"].get("identity_summary_sha256") != (
            self.data_identity_sha256
        ):
            raise ValueError("plan dataset identity is not bound to common contract")
        common_queries = common["queries"]
        if any(
            common_queries.get(field) != getattr(self.queries, field)
            for field in ("count", "stable_identity_sha256", "order_sha256")
        ):
            raise ValueError("plan query identity is not bound to common contract")
        replay_digest = (
            common.get("configuration", {})
            .get("values", {})
            .get("replay_input_sha256")
        )
        if replay_digest != self.replay_input_sha256:
            raise ValueError("plan Replay identity is not bound to common contract")
        for treatment in self.allowed_treatment_changes:
            try:
                marker = _pointer_get(common, treatment.pointer)
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("treatment path is absent from common contract") from exc
            if marker != {"declared_treatment": treatment.pointer}:
                raise ValueError("treatment path is not an exact masked leaf")
        return self


def comparison_binding(plan_path: Path, role: str) -> dict[str, str]:
    """Return the exact generation-zero binding for ``BenchmarkRunCommitStore``."""

    plan = load_comparison_plan(plan_path)
    if role not in {"baseline", "candidate"}:
        raise ExperimentPairingError("comparison_role_invalid")
    return {
        "contract": PLAN_CONTRACT,
        "plan_sha256": sha256_file(plan_path),
        "role": role,
        "common_execution_contract_sha256": (
            plan.common_execution_contract_sha256
        ),
    }


def opaque_query_identity(value: str) -> str:
    normalized = str(value)
    if _OPAQUE_QUERY_RE.fullmatch(normalized):
        return normalized
    return "query:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_comparison_plan(path: Path) -> ComparisonPlanV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ComparisonPlanV1.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ExperimentPairingError("comparison_plan_invalid") from exc


def load_gate_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentPairingError("pairing_protocol_unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != GATE_CONTRACT
        or payload.get("comparison_plan_contract") != PLAN_CONTRACT
        or payload.get("score_scope")
        != "pairing_only_not_quality_or_official_score"
    ):
        raise ExperimentPairingError("pairing_protocol_version_incompatible")
    for section in ("frozen_baseline_eligibility", "evidence_registry_eligibility"):
        identity = payload.get(section)
        if not isinstance(identity, dict):
            raise ExperimentPairingError("pairing_protocol_identity_missing")
        file_path = resolve_repo_path(repository_root, str(identity.get("path") or ""))
        if not file_path.is_file() or sha256_file(file_path) != identity.get("sha256"):
            raise ExperimentPairingError("pairing_protocol_identity_drift")
    return payload


def write_comparison_plan(path: Path, plan: ComparisonPlanV1) -> None:
    write_json(path, plan.model_dump(mode="json"))


def build_common_execution_contract(
    manifest: RunManifestV1,
    treatments: Sequence[TreatmentChange],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "dataset": manifest.dataset.model_dump(mode="json"),
        "queries": manifest.queries.model_dump(mode="json"),
        "prompt": manifest.prompt.model_dump(mode="json"),
        "configuration": {
            "sources": manifest.configuration.sources,
            "budgets": manifest.configuration.budgets,
            "values": copy.deepcopy(manifest.configuration.values),
        },
        "evaluator": manifest.evaluator.model_dump(mode="json"),
        "determinism": manifest.determinism.model_dump(mode="json"),
        "execution_profile": "offline_committed_replay_v1",
        "coverage_policy": "all_planned_queries_no_common_success_filtering",
    }
    for item in treatments:
        _pointer_set(document, item.pointer, {"declared_treatment": item.pointer})
    return document


def build_comparison_plan(
    *,
    plan_id: str,
    baseline_manifest: RunManifestV1,
    query_identities: Sequence[str],
    data_identity_sha256: str,
    replay_input_sha256: str,
    treatments: Sequence[TreatmentChange],
    exclusions: Sequence[PredeclaredExclusion] = (),
) -> ComparisonPlanV1:
    ordered_treatments = sorted(treatments, key=lambda item: item.pointer)
    common = build_common_execution_contract(baseline_manifest, ordered_treatments)
    return ComparisonPlanV1(
        plan_id=plan_id,
        queries=QueryPopulation(
            count=len(query_identities),
            identities=list(query_identities),
            stable_identity_sha256=baseline_manifest.queries.stable_identity_sha256,
            order_sha256=baseline_manifest.queries.order_sha256,
        ),
        data_identity_sha256=data_identity_sha256,
        replay_input_sha256=replay_input_sha256,
        roles={"baseline": "baseline", "candidate": "candidate"},
        allowed_treatment_changes=ordered_treatments,
        common_execution_contract=common,
        common_execution_contract_sha256=stable_hash(common),
        predeclared_exclusions=sorted(
            exclusions, key=lambda item: item.query_identity
        ),
    )


def validate_pairing(
    plan_path: Path,
    baseline_manifest_path: Path,
    candidate_manifest_path: Path,
    *,
    repository_root: Path,
    controlled_fault: Literal["hidden_treatment", "asymmetric_coverage"] | None = None,
) -> dict[str, Any]:
    """Validate treatment isolation and symmetric query coverage without quality data."""

    root = repository_root.resolve()
    plan = load_comparison_plan(plan_path)
    plan_digest = sha256_file(plan_path)
    violations: list[dict[str, Any]] = []
    manifests: dict[str, RunManifestV1] = {}
    states: dict[str, Any] = {}
    duplicate_commits: dict[str, list[str]] = {}

    for role, path in (
        ("baseline", baseline_manifest_path),
        ("candidate", candidate_manifest_path),
    ):
        report = validate_run_manifest(path, repository_root=root)
        if report["status"] != "passed":
            violations.append(
                _violation(
                    "run_manifest_invalid",
                    role=role,
                    path="$",
                    expected="passed",
                    observed=report.get("violations", [])[:1],
                )
            )
            continue
        manifest = RunManifestV1.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        manifests[role] = manifest
        _validate_manifest_binding(
            manifest, role=role, plan=plan, plan_digest=plan_digest, violations=violations
        )
        run_directory = resolve_repo_path(root, manifest.output_directory)
        store = BenchmarkRunCommitStore(run_directory)
        try:
            state = store.load_latest()
        except CrashConsistencyError:
            violations.append(
                _violation(
                    "committed_lineage_invalid",
                    role=role,
                    path="$.lineage",
                    expected="complete valid generation chain",
                    observed="unavailable",
                )
            )
            continue
        states[role] = state
        if (
            state.run_id != manifest.run_id
            or len(state.records) != manifest.progress.completed_count
            or (
                manifest.progress.status == "completed"
                and state.status != "completed"
            )
        ):
            violations.append(
                _violation(
                    "manifest_generation_state_mismatch",
                    role=role,
                    path="$.progress",
                    expected={
                        "run_id": manifest.run_id,
                        "completed_count": manifest.progress.completed_count,
                        "status": manifest.progress.status,
                    },
                    observed={
                        "run_id": state.run_id,
                        "completed_count": len(state.records),
                        "status": state.status,
                    },
                )
            )
        duplicate_commits[role] = _duplicate_record_commits(store)
        for query_identity in duplicate_commits[role]:
            violations.append(
                _violation(
                    "duplicate_query_record",
                    role=role,
                    query_identity=query_identity,
                    path="$.generation_chain",
                    expected=1,
                    observed="more_than_once",
                )
            )
        expected_binding = comparison_binding(plan_path, role)
        if state.config.get("comparison") != expected_binding:
            violations.append(
                _violation(
                    "generation_zero_plan_binding_mismatch",
                    role=role,
                    path="$.config.comparison",
                    expected=expected_binding,
                    observed=state.config.get("comparison"),
                )
            )

    if set(manifests) == {"baseline", "candidate"}:
        baseline_common = build_common_execution_contract(
            manifests["baseline"], plan.allowed_treatment_changes
        )
        candidate_common = build_common_execution_contract(
            manifests["candidate"], plan.allowed_treatment_changes
        )
        if controlled_fault == "hidden_treatment":
            candidate_common["configuration"]["values"]["undeclared_mode"] = True
        for role, manifest, common in (
            ("baseline", manifests["baseline"], baseline_common),
            ("candidate", manifests["candidate"], candidate_common),
        ):
            _validate_treatment_values(manifest, role, plan, violations)
            difference = _first_difference(plan.common_execution_contract, common)
            if difference is not None:
                violations.append(
                    _violation(
                        "undeclared_execution_contract_drift",
                        role=role,
                        path=difference,
                        expected=_pointer_or_none(plan.common_execution_contract, difference),
                        observed=_pointer_or_none(common, difference),
                    )
                )

    coverage_status = "unavailable"
    pair_count = 0
    terminal_counts: dict[str, dict[str, int]] = {}
    if set(states) == {"baseline", "candidate"}:
        baseline_state = states["baseline"]
        candidate_state = states["candidate"]
        if controlled_fault == "asymmetric_coverage":
            candidate_records = list(candidate_state.records[:-1])
        else:
            candidate_records = list(candidate_state.records)
        baseline_records = list(baseline_state.records)
        baseline_expected = [
            opaque_query_identity(value) for value in baseline_state.expected_query_ids
        ]
        candidate_expected = [
            opaque_query_identity(value) for value in candidate_state.expected_query_ids
        ]
        if baseline_expected != plan.queries.identities:
            violations.append(
                _violation(
                    "baseline_query_population_mismatch",
                    role="baseline",
                    path="$.expected_query_ids",
                    expected=plan.queries.identities,
                    observed=baseline_expected,
                )
            )
        if candidate_expected != plan.queries.identities:
            violations.append(
                _violation(
                    "candidate_query_population_mismatch",
                    role="candidate",
                    path="$.expected_query_ids",
                    expected=plan.queries.identities,
                    observed=candidate_expected,
                )
            )
        baseline_by_id = _validated_records(baseline_records, "baseline", violations)
        candidate_by_id = _validated_records(candidate_records, "candidate", violations)
        if set(baseline_by_id) != set(candidate_by_id):
            difference = sorted(set(baseline_by_id) ^ set(candidate_by_id))[0]
            violations.append(
                _violation(
                    "asymmetric_query_coverage",
                    query_identity=difference,
                    path="$.records",
                    expected="present_once_on_both_sides",
                    observed={
                        "baseline": difference in baseline_by_id,
                        "candidate": difference in candidate_by_id,
                    },
                )
            )
        pair_count = len(set(baseline_by_id) & set(candidate_by_id))
        exclusions = {
            item.query_identity: item.reason for item in plan.predeclared_exclusions
        }
        for query_identity in plan.queries.identities:
            left = baseline_by_id.get(query_identity)
            right = candidate_by_id.get(query_identity)
            if left is None or right is None:
                continue
            _validate_paired_record(
                query_identity, left, right, exclusions=exclusions, violations=violations
            )
        terminal_counts = {
            "baseline": dict(sorted(Counter(
                str(item.get("status")) for item in baseline_records
            ).items())),
            "candidate": dict(sorted(Counter(
                str(item.get("status")) for item in candidate_records
            ).items())),
        }
        both_complete = (
            baseline_state.status == "completed"
            and candidate_state.status == "completed"
            and len(baseline_records) == plan.queries.count
            and len(candidate_records) == plan.queries.count
        )
        if both_complete:
            coverage_status = "complete"
        elif not violations and set(baseline_by_id) == set(candidate_by_id):
            coverage_status = "symmetric_intermediate"
        else:
            coverage_status = "asymmetric_or_invalid"

    if violations:
        status, exit_code = "violation", EXIT_VIOLATION
    elif coverage_status == "symmetric_intermediate":
        status, exit_code = "not_ready", EXIT_NOT_READY
    elif coverage_status == "complete":
        status, exit_code = "passed", EXIT_PASSED
    else:
        status, exit_code = "not_eligible", EXIT_NOT_READY
    return _report(
        status=status,
        exit_code=exit_code,
        plan_sha256=plan_digest,
        query_count=plan.queries.count,
        paired_query_count=pair_count,
        coverage_status=coverage_status,
        terminal_counts=terminal_counts,
        violations=violations,
        observation={
            "plan_binding_affects_execution": False,
            "result_payload_compared_for_quality": False,
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    )


def audit_frozen_eligibility(legacy_audit_path: Path) -> dict[str, Any]:
    payload = json.loads(legacy_audit_path.read_text(encoding="utf-8"))
    profiles = []
    for item in sorted(payload.get("profiles", []), key=lambda row: row["profile_id"]):
        profiles.append(
            {
                "profile_id": item["profile_id"],
                "status": "not_eligible",
                "missing_contracts": [
                    "comparison_plan_v1_prebinding",
                    "symmetric_query_pairing_records",
                    "experiment_pairing_integrity_v1",
                ],
                "historical_artifacts_modified": False,
            }
        )
    return _report(
        status="not_eligible",
        exit_code=EXIT_NOT_READY,
        plan_sha256=None,
        query_count=0,
        paired_query_count=0,
        coverage_status="legacy_contract_missing",
        terminal_counts={},
        violations=[],
        frozen_profiles=profiles,
        observation={
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    )


def audit_evidence_registry(
    registry_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.get("strategies", payload.get("entries", []))
    if not isinstance(entries, list):
        entries = []
    declared = [item for item in entries if isinstance(item, dict) and "pairing_evidence" in item]
    invalid = []
    root = repository_root.resolve() if repository_root is not None else registry_path.parent
    expected_fields = {
        "comparison_plan_path",
        "comparison_plan_sha256",
        "baseline_run_manifest_path",
        "baseline_run_manifest_sha256",
        "candidate_run_manifest_path",
        "candidate_run_manifest_sha256",
    }
    for item in declared:
        evidence = item.get("pairing_evidence")
        valid = isinstance(evidence, dict) and set(evidence) == expected_fields
        if valid:
            for prefix in (
                "comparison_plan",
                "baseline_run_manifest",
                "candidate_run_manifest",
            ):
                try:
                    registered = resolve_repo_path(root, str(evidence[f"{prefix}_path"]))
                except ValueError:
                    valid = False
                    break
                if (
                    not registered.is_file()
                    or sha256_file(registered) != evidence[f"{prefix}_sha256"]
                ):
                    valid = False
                    break
        if not valid:
            invalid.append(stable_hash(str(item.get("strategy_id", "unknown"))))
    status = "violation" if invalid else ("passed" if declared else "not_eligible")
    exit_code = (
        EXIT_VIOLATION if invalid else (EXIT_PASSED if declared else EXIT_NOT_READY)
    )
    return _report(
        status=status,
        exit_code=exit_code,
        plan_sha256=None,
        query_count=0,
        paired_query_count=0,
        coverage_status="registry_pairing_binding_audit",
        terminal_counts={},
        violations=[
            _violation(
                "registry_pairing_binding_invalid",
                path="$.pairing_evidence",
                expected="content-addressed plan and both run manifests",
                observed=value,
            )
            for value in invalid
        ],
        registry={
            "declared_pairing_entry_count": len(declared),
            "legacy_unbound": not declared,
            "evidence_conclusions_modified": False,
        },
        observation={
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    )


def build_local_fixture(
    root: Path,
    *,
    partial_count: int | None = None,
    baseline_statuses: Sequence[str] = ("succeeded", "failed", "succeeded"),
    candidate_statuses: Sequence[str] = ("succeeded", "failed", "succeeded"),
    candidate_query_order: Sequence[int] = (0, 1, 2),
    excluded_indexes: Sequence[int] = (),
    duplicate_candidate_index: int | None = None,
) -> tuple[Path, Path, Path]:
    """Build a deterministic gold-free pair through production persistence paths."""

    root.mkdir(parents=True, exist_ok=True)
    query_ids = [f"query:{hashlib.sha256(f'fixture-{i}'.encode()).hexdigest()}" for i in range(3)]
    query_rows = [
        {"query_id": value, "query": f"offline fixture query {index}"}
        for index, value in enumerate(query_ids)
    ]
    _write_jsonl(root / "inputs/queries.jsonl", query_rows)
    (root / "inputs/dataset.txt").write_text("offline-dataset-v1\n", encoding="utf-8")
    (root / "inputs/replay.json").write_text('{"offline":true}\n', encoding="utf-8")
    write_json(root / "inputs/prompt.json", {"planner": "current_rules_fixture_v1"})
    git_payload = {
        "commit": "a" * 40,
        "dirty_paths": ["third_party/paper-qa"],
        "allowed_dirty_paths": ["third_party/paper-qa"],
        "unexpected_dirty_paths": [],
    }
    git = GitProvenance(
        **git_payload,
        dirty=True,
        worktree_state_sha256=stable_hash(git_payload),
    )
    treatment = TreatmentChange(
        pointer="/configuration/values/treatment_mode",
        baseline_value="off",
        candidate_value="candidate_v1",
    )
    provisional_spec = _fixture_manifest_spec(root, "baseline", "off", 0, None)
    provisional = _build_provisional_manifest(root, provisional_spec, git)
    plan = build_comparison_plan(
        plan_id="offline-gold-blind-fixture",
        baseline_manifest=provisional,
        query_identities=query_ids,
        data_identity_sha256=provisional.dataset.identity_summary_sha256,
        replay_input_sha256=sha256_file(root / "inputs/replay.json"),
        treatments=[treatment],
        exclusions=[
            PredeclaredExclusion(
                query_identity=query_ids[index], reason="predeclared_fixture_exclusion"
            )
            for index in sorted(excluded_indexes)
        ],
    )
    plan_path = root / "inputs/comparison_plan.json"
    write_comparison_plan(plan_path, plan)

    manifest_paths = []
    limit = len(query_ids) if partial_count is None else partial_count
    for role, mode, statuses in (
        ("baseline", "off", baseline_statuses),
        ("candidate", "candidate_v1", candidate_statuses),
    ):
        run_dir = root / "runs" / role
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        expected_ids = (
            query_ids
            if role == "baseline"
            else [query_ids[index] for index in candidate_query_order]
        )
        config = _fixture_run_config(expected_ids, mode)
        store = BenchmarkRunCommitStore(run_dir)
        store.initialize(
            run_id=f"offline-pair-{role}",
            expected_query_ids=expected_ids,
            config=config,
            dataset_report={"identity": provisional.dataset.identity_summary_sha256},
            comparison_binding=comparison_binding(plan_path, role),
        )
        status_by_id = dict(zip(query_ids, statuses, strict=True))
        for query_identity in expected_ids[:limit]:
            index = query_ids.index(query_identity)
            store.commit_record(
                _fixture_record(
                    query_identity,
                    status_by_id[query_identity],
                    exclusion_reason=(
                        "predeclared_fixture_exclusion"
                        if index in excluded_indexes
                        else None
                    ),
                )
            )
        if role == "candidate" and duplicate_candidate_index is not None:
            identity = query_ids[duplicate_candidate_index]
            store.commit_record(
                _fixture_record(identity, status_by_id[identity])
            )
        if limit == len(query_ids):
            state = store.commit_completion({})
        else:
            state = store.load_latest()
        store.materialize_compatibility_view(state)
        spec = _fixture_manifest_spec(root, role, mode, limit, plan)
        manifest = build_run_manifest(spec, repository_root=root, git_provenance=git)
        path = root / f"{role}_run_manifest.json"
        write_json(path, manifest.model_dump(mode="json"))
        manifest_paths.append(path)
    return plan_path, manifest_paths[0], manifest_paths[1]


def _build_provisional_manifest(
    root: Path, spec: Mapping[str, Any], git: GitProvenance
) -> RunManifestV1:
    run_dir = root / "runs" / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", _fixture_run_config([], "off"))
    (run_dir / "results.jsonl").write_text("", encoding="utf-8")
    (run_dir / "failures.jsonl").write_text("", encoding="utf-8")
    write_json(run_dir / "dataset_report.json", {"identity": "provisional"})
    return build_run_manifest(spec, repository_root=root, git_provenance=git)


def _fixture_manifest_spec(
    root: Path,
    role: str,
    mode: str,
    completed_count: int,
    plan: ComparisonPlanV1 | None,
) -> dict[str, Any]:
    run_dir = root / "runs" / role
    outputs = [
        {
            "path": f"runs/{role}/{name}",
            "role": role_name,
            "format": "jsonl" if name.endswith("jsonl") else "json",
        }
        for name, role_name in (
            ("config.json", "run_configuration"),
            ("dataset_report.json", "dataset_identity"),
            ("failures.jsonl", "terminal_failures"),
            ("results.jsonl", "paired_records"),
        )
    ]
    excluded = []
    commit_root = run_dir / STORE_DIRECTORY
    if commit_root.is_dir():
        excluded = sorted(
            path.relative_to(run_dir).as_posix()
            for path in commit_root.rglob("*")
            if path.is_file()
        )
    spec: dict[str, Any] = {
        "run_id": f"offline-pair-{role}",
        "dataset": {
            "name": "gold_blind_pairing_fixture",
            "version": "1",
            "input_paths": ["inputs/dataset.txt", "inputs/replay.json"],
        },
        "queries": {
            "input_path": "inputs/queries.jsonl",
            "id_field": "query_id",
            "text_field": "query",
        },
        "prompt": {
            "manifest_path": "inputs/prompt.json",
            "versions": {"planner": "current_rules_fixture_v1"},
            "used": False,
        },
        "configuration": {
            "sources": ["arxiv", "openalex", "semantic_scholar", "pubmed"],
            "budgets": {"max_queries": 4, "top_k": 20},
            "values": {
                "treatment_mode": mode,
                "execution_profile": "offline_committed_replay_v1",
                "timeout_seconds": 10,
                "retry_count": 1,
                "normalization_version": "identity_v1",
                "replay_input_sha256": sha256_file(root / "inputs/replay.json"),
            },
        },
        "evaluator": {"name": "pairing_identity_only", "version": "1"},
        "determinism": {
            "random_seed": 0,
            "parameters": {"ordering": "stable", "concurrency": 1},
        },
        "progress": {
            "status": "completed" if completed_count == 3 else "partial",
            "expected_count": 3,
            "completed_count": completed_count,
            "record_output_path": f"runs/{role}/results.jsonl",
        },
        "lineage": {
            "checkpoint_id": f"offline-pair-{role}-checkpoint",
            "resume_index": 0,
            "parent": None,
        },
        "output_directory": f"runs/{role}",
        "output_inventory_excludes": excluded,
        "outputs": outputs,
        "metadata_bindings": _fixture_bindings(role),
    }
    if plan is not None:
        spec["comparison"] = {
            "plan_path": "inputs/comparison_plan.json",
            "role": role,
            "common_execution_contract_sha256": (
                plan.common_execution_contract_sha256
            ),
        }
    return spec


def _fixture_run_config(query_ids: Sequence[str], mode: str) -> dict[str, Any]:
    return {
        "case_ids": list(query_ids),
        "dataset": {"name": "gold_blind_pairing_fixture", "version": "1"},
        "prompt": {"versions": {"planner": "current_rules_fixture_v1"}},
        "configuration": {
            "sources": ["arxiv", "openalex", "semantic_scholar", "pubmed"],
            "budgets": {"max_queries": 4, "top_k": 20},
            "values": {"treatment_mode": mode},
        },
        "evaluator": {"name": "pairing_identity_only", "version": "1"},
    }


def _fixture_bindings(role: str) -> list[dict[str, str]]:
    return [
        {
            "artifact_path": f"runs/{role}/config.json",
            "artifact_json_pointer": artifact,
            "manifest_json_pointer": manifest,
        }
        for artifact, manifest in (
            ("/dataset/name", "/dataset/name"),
            ("/dataset/version", "/dataset/version"),
            ("/prompt/versions", "/prompt/versions"),
            ("/configuration/sources", "/configuration/sources"),
            ("/configuration/budgets", "/configuration/budgets"),
            ("/evaluator/name", "/evaluator/name"),
            ("/evaluator/version", "/evaluator/version"),
        )
    ]


def _fixture_record(
    query_identity: str,
    status: str,
    *,
    exclusion_reason: str | None = None,
) -> dict[str, Any]:
    source_status = "success" if status == "succeeded" else "failed"
    record = {
        "case_id": query_identity,
        "status": status,
        "source_terminals": [
            {"source": source, "status": source_status, "reason": None}
            for source in ("arxiv", "openalex", "semantic_scholar", "pubmed")
        ],
        "semantic_events": ["query_started", "query_terminal"],
        "normalized_result_identity_sha256": stable_hash(
            {"query_identity": query_identity, "fixture": "same-on-both-sides"}
        ),
    }
    if exclusion_reason is not None:
        record["exclusion_reason"] = exclusion_reason
    return record


def _validate_manifest_binding(
    manifest: RunManifestV1,
    *,
    role: str,
    plan: ComparisonPlanV1,
    plan_digest: str,
    violations: list[dict[str, Any]],
) -> None:
    binding = manifest.comparison
    if binding is None:
        violations.append(
            _violation(
                "comparison_plan_not_prebound",
                role=role,
                path="$.comparison",
                expected=plan_digest,
                observed=None,
            )
        )
        return
    expected = {
        "contract": PLAN_CONTRACT,
        "plan_sha256": plan_digest,
        "role": role,
        "common_execution_contract_sha256": plan.common_execution_contract_sha256,
    }
    observed = {
        "contract": binding.contract,
        "plan_sha256": binding.plan_sha256,
        "role": binding.role,
        "common_execution_contract_sha256": binding.common_execution_contract_sha256,
    }
    if observed != expected:
        violations.append(
            _violation(
                "comparison_plan_binding_mismatch",
                role=role,
                path="$.comparison",
                expected=expected,
                observed=observed,
            )
        )
    if (
        manifest.queries.count != plan.queries.count
        or manifest.queries.stable_identity_sha256
        != plan.queries.stable_identity_sha256
        or manifest.queries.order_sha256 != plan.queries.order_sha256
    ):
        violations.append(
            _violation(
                "manifest_query_population_mismatch",
                role=role,
                path="$.queries",
                expected=plan.queries.model_dump(mode="json"),
                observed={
                    "count": manifest.queries.count,
                    "stable_identity_sha256": manifest.queries.stable_identity_sha256,
                    "order_sha256": manifest.queries.order_sha256,
                },
            )
        )
    if manifest.dataset.identity_summary_sha256 != plan.data_identity_sha256:
        violations.append(
            _violation(
                "dataset_identity_mismatch",
                role=role,
                path="$.dataset.identity_summary_sha256",
                expected=plan.data_identity_sha256,
                observed=manifest.dataset.identity_summary_sha256,
            )
        )


def _validate_treatment_values(
    manifest: RunManifestV1,
    role: str,
    plan: ComparisonPlanV1,
    violations: list[dict[str, Any]],
) -> None:
    document = {
        "configuration": {
            "sources": manifest.configuration.sources,
            "budgets": manifest.configuration.budgets,
            "values": manifest.configuration.values,
        }
    }
    for change in plan.allowed_treatment_changes:
        expected = (
            change.baseline_value if role == "baseline" else change.candidate_value
        )
        try:
            observed = _pointer_get(document, change.pointer)
        except (KeyError, IndexError, TypeError):
            observed = "missing"
        if observed != expected:
            violations.append(
                _violation(
                    "declared_treatment_missing_or_drifted",
                    role=role,
                    path=change.pointer,
                    expected=expected,
                    observed=observed,
                )
            )


def _validated_records(
    records: Sequence[Mapping[str, Any]],
    role: str,
    violations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        forbidden = sorted(_FORBIDDEN_RECORD_FIELDS & set(row))
        raw_identity = str(row.get("case_id") or "")
        if not raw_identity:
            violations.append(
                _violation(
                    "query_identity_missing",
                    role=role,
                    path="$.records[].case_id",
                    expected="stable identity input",
                    observed=None,
                )
            )
            continue
        query_identity = opaque_query_identity(raw_identity)
        if forbidden:
            violations.append(
                _violation(
                    "forbidden_quality_or_identity_field",
                    role=role,
                    query_identity=query_identity,
                    path="$.records",
                    expected="gold-blind terminal metadata",
                    observed=forbidden,
                )
            )
        if query_identity in rows:
            violations.append(
                _violation(
                    "duplicate_query_record",
                    role=role,
                    query_identity=query_identity,
                    path="$.records",
                    expected=1,
                    observed=2,
                )
            )
        rows[query_identity] = dict(row)
    return rows


def _validate_paired_record(
    query_identity: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    exclusions: Mapping[str, str],
    violations: list[dict[str, Any]],
) -> None:
    left_status, right_status = str(baseline.get("status")), str(candidate.get("status"))
    if left_status not in _TERMINAL_STATUSES or right_status not in _TERMINAL_STATUSES:
        violations.append(
            _violation(
                "query_terminal_status_invalid",
                query_identity=query_identity,
                path="$.status",
                expected=sorted(_TERMINAL_STATUSES),
                observed={"baseline": left_status, "candidate": right_status},
            )
        )
    if left_status != right_status:
        violations.append(
            _violation(
                "asymmetric_terminal_status",
                query_identity=query_identity,
                path="$.status",
                expected=left_status,
                observed=right_status,
            )
        )
    excluded = query_identity in exclusions
    if excluded:
        expected = {"status": "excluded", "exclusion_reason": exclusions[query_identity]}
        for role, row in (("baseline", baseline), ("candidate", candidate)):
            observed = {
                "status": row.get("status"),
                "exclusion_reason": row.get("exclusion_reason"),
            }
            if observed != expected:
                violations.append(
                    _violation(
                        "predeclared_exclusion_not_applied_symmetrically",
                        role=role,
                        query_identity=query_identity,
                        path="$.exclusion_reason",
                        expected=expected,
                        observed=observed,
                    )
                )
        return
    if left_status == "excluded" or right_status == "excluded":
        violations.append(
            _violation(
                "post_hoc_exclusion",
                query_identity=query_identity,
                path="$.status",
                expected="not excluded",
                observed={"baseline": left_status, "candidate": right_status},
            )
        )
    left_sources = _normalize_source_terminals(baseline.get("source_terminals"))
    right_sources = _normalize_source_terminals(candidate.get("source_terminals"))
    if left_sources is None or right_sources is None:
        violations.append(
            _violation(
                "source_terminal_contract_missing",
                query_identity=query_identity,
                path="$.source_terminals",
                expected="one structured terminal per source",
                observed={"baseline": left_sources, "candidate": right_sources},
            )
        )
    elif left_sources != right_sources:
        difference = _first_difference(left_sources, right_sources) or "$"
        violations.append(
            _violation(
                "source_availability_or_terminal_drift",
                query_identity=query_identity,
                path=difference,
                expected=_pointer_or_none(left_sources, difference),
                observed=_pointer_or_none(right_sources, difference),
            )
        )


def _normalize_source_terminals(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    rows = []
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"source", "status", "reason"}:
            return None
        source, status = str(item["source"]), str(item["status"])
        if not source or source in seen or status not in _SOURCE_STATUSES:
            return None
        seen.add(source)
        rows.append({"source": source, "status": status, "reason": item["reason"]})
    return rows


def _duplicate_record_commits(store: BenchmarkRunCommitStore) -> list[str]:
    counts: Counter[str] = Counter()
    generations = store.root / "generations"
    if not generations.is_dir():
        return []
    for directory in sorted(generations.glob("generation-*")):
        if not (directory / "COMMITTED").is_file():
            continue
        try:
            delta = json.loads((directory / "delta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if delta.get("kind") == "record" and isinstance(delta.get("record"), dict):
            counts[opaque_query_identity(str(delta["record"].get("case_id")))] += 1
    return sorted(key for key, count in counts.items() if count != 1)


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/") or pointer.endswith("/"):
        raise ValueError("invalid JSON pointer")
    parts = pointer[1:].split("/")
    if not parts or any(not part for part in parts):
        raise ValueError("invalid JSON pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def _pointer_get(document: Any, pointer: str) -> Any:
    current = document
    for part in _pointer_parts(pointer):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _pointer_set(document: Any, pointer: str, value: Any) -> None:
    parts = _pointer_parts(pointer)
    current = document
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        if last not in current:
            raise KeyError(pointer)
        current[last] = value


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if set(left) != set(right):
            key = sorted(set(left) ^ set(right))[0]
            return f"{path}/{_escape_pointer(key)}"
        for key in sorted(left):
            difference = _first_difference(
                left[key], right[key], f"{path}/{_escape_pointer(str(key))}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}/length"
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(
                left_value, right_value, f"{path}/{index}"
            )
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def _pointer_or_none(document: Any, path: str) -> Any:
    if path == "$":
        return document
    if path.endswith("/length"):
        parent = path.removesuffix("/length")
        value = _pointer_or_none(document, parent)
        return len(value) if isinstance(value, list) else None
    try:
        return _pointer_get(document, path.removeprefix("$"))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _violation(
    invariant: str,
    *,
    path: str,
    expected: Any,
    observed: Any,
    role: str | None = None,
    query_identity: str | None = None,
) -> dict[str, Any]:
    return {
        "invariant": invariant,
        "role": role,
        "query_identity": query_identity,
        "configuration_path": path if path.startswith("$/configuration") else None,
        "first_difference_path": path,
        "expected_sha256": stable_hash(expected),
        "observed_sha256": stable_hash(observed),
    }


def _report(**values: Any) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_CONTRACT,
        "gate": GATE_NAME,
        "score_scope": "pairing_only_not_quality_or_official_score",
        **values,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(stable_json_bytes(dict(row), indent=None) for row in rows))


def deterministic_fixture_report(
    *,
    controlled_fault: Literal["hidden_treatment", "asymmetric_coverage"] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="spar-pairing-") as value:
        root = Path(value)
        plan, baseline, candidate = build_local_fixture(root)
        return validate_pairing(
            plan,
            baseline,
            candidate,
            repository_root=root,
            controlled_fault=controlled_fault,
        )
