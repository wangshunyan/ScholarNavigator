"""Deterministic, gold-blind shard planning and cross-run merge integrity.

The module composes ``run_manifest_v1`` and the production atomic generation
store.  It never performs retrieval or quality evaluation.  A shard attempt is
selected only through an explicit supersession chain; query outcomes are never
used for attempt selection or aggregation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    STORE_DIRECTORY,
    BenchmarkRunCommitStore,
    CrashConsistencyError,
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.experiment_pairing import (
    QueryPopulation,
    opaque_query_identity,
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


PLAN_CONTRACT = "shard_plan_v1"
ATTEMPT_SET_CONTRACT = "shard_attempt_set_v1"
AGGREGATE_CONTRACT = "shard_aggregate_v1"
GATE_CONTRACT = "sharded_execution_integrity_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "sharded_execution_integrity_gate"
ASSIGNMENT_ALGORITHM = "ordered_round_robin_v1"

EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE_ERROR = 4

_OPAQUE_QUERY_RE = re.compile(r"^query:[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "excluded"})
_FORBIDDEN_FIELDS = frozenset(
    {"gold", "qrels", "target_paper", "quality_metrics", "official_score"}
)
_GENERATION_CONFIG_EXCLUDES = frozenset(
    {
        "case_count",
        "case_ids",
        "code",
        "resume_signature",
        "shard",
        "started_at",
    }
)


class ShardedExecutionError(RuntimeError):
    """The shard contract is invalid or cannot be checked without guessing."""


class ShardAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int = Field(ge=0)
    query_identities: list[str]
    query_identities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_assignment(self) -> "ShardAssignment":
        if len(set(self.query_identities)) != len(self.query_identities):
            raise ValueError("duplicate query in shard")
        if any(not _OPAQUE_QUERY_RE.fullmatch(value) for value in self.query_identities):
            raise ValueError("shard query identity must be opaque")
        if stable_hash(self.query_identities) != self.query_identities_sha256:
            raise ValueError("shard query identity digest mismatch")
        return self


class ShardPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    contract: Literal["shard_plan_v1"] = PLAN_CONTRACT
    score_scope: Literal[
        "partition_and_merge_only_not_quality_or_official_score"
    ] = "partition_and_merge_only_not_quality_or_official_score"
    plan_id: str = Field(min_length=1, max_length=100)
    queries: QueryPopulation
    data_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_count: int = Field(ge=1)
    assignment_algorithm: Literal["ordered_round_robin_v1"] = ASSIGNMENT_ALGORITHM
    shards: list[ShardAssignment] = Field(min_length=1)
    common_execution_contract: dict[str, Any]
    common_execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    population_policy: Literal[
        "all_queries_preserve_success_failure_cancelled_and_excluded"
    ] = "all_queries_preserve_success_failure_cancelled_and_excluded"
    attempt_selection_policy: Literal[
        "unique_supersession_tip_without_outcome_selection"
    ] = "unique_supersession_tip_without_outcome_selection"
    gold_accessed: Literal[False] = False
    quality_metrics_computed: Literal[False] = False

    @model_validator(mode="after")
    def validate_closed_plan(self) -> "ShardPlanV1":
        if len(self.shards) != self.shard_count:
            raise ValueError("shard assignment count mismatch")
        if [item.shard_index for item in self.shards] != list(range(self.shard_count)):
            raise ValueError("shard indexes must be contiguous and ordered")
        expected = deterministic_assignments(self.queries.identities, self.shard_count)
        observed = [item.query_identities for item in self.shards]
        if observed != expected:
            raise ValueError("shard assignment algorithm drift")
        flattened = [value for shard in observed for value in shard]
        if len(flattened) != self.queries.count or set(flattened) != set(
            self.queries.identities
        ):
            raise ValueError("shard population is not an exact partition")
        if stable_hash(self.common_execution_contract) != (
            self.common_execution_contract_sha256
        ):
            raise ValueError("common execution contract digest mismatch")
        required = {
            "dataset",
            "prompt",
            "configuration",
            "evaluator",
            "determinism",
            "execution_profile",
            "merge_policy",
        }
        if set(self.common_execution_contract) != required:
            raise ValueError("common execution contract is not closed")
        return self


class ShardAttemptReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int = Field(ge=0)
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    supersedes_attempt_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
    )
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reference(self) -> "ShardAttemptReference":
        if self.supersedes_attempt_id == self.attempt_id:
            raise ValueError("attempt cannot supersede itself")
        if not self.manifest_path or Path(self.manifest_path).is_absolute():
            raise ValueError("attempt manifest path must be relative")
        if ".." in Path(self.manifest_path).parts:
            raise ValueError("attempt manifest path escapes repository root")
        return self


class ShardAttemptSetV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    contract: Literal["shard_attempt_set_v1"] = ATTEMPT_SET_CONTRACT
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: list[ShardAttemptReference]

    @model_validator(mode="after")
    def validate_attempts(self) -> "ShardAttemptSetV1":
        keys = [(item.shard_index, item.attempt_id) for item in self.attempts]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("attempt references must be sorted and unique")
        paths = [item.manifest_path for item in self.attempts]
        if len(paths) != len(set(paths)):
            raise ValueError("attempt manifest path reused")
        return self


class SelectedShardReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int = Field(ge=0)
    attempt_id: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    commit_generation: int = Field(ge=0)
    generation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=0)
    event_count: int = Field(ge=0)


class ShardResourceLedgerReference(BaseModel):
    """Authoritative ledger selected with one final shard attempt."""

    model_config = ConfigDict(extra="forbid")

    shard_index: int = Field(ge=0)
    attempt_id: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")


class ShardAggregateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    contract: Literal["shard_aggregate_v1"] = AGGREGATE_CONTRACT
    score_scope: Literal[
        "partition_and_merge_only_not_quality_or_official_score"
    ] = "partition_and_merge_only_not_quality_or_official_score"
    plan_path: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=1)
    query_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_shards: list[SelectedShardReference]
    records: list[dict[str, Any]]
    commit_events: list[dict[str, Any]]
    terminal_counts: dict[str, int]
    operational_counts: dict[str, int | float]
    resource_ledgers: list[ShardResourceLedgerReference] = Field(
        default_factory=list
    )
    completed: bool
    aggregate_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_summary(self) -> "ShardAggregateV1":
        value = self.model_dump(mode="json", exclude={"aggregate_summary_sha256"})
        if stable_hash(value) != self.aggregate_summary_sha256:
            raise ValueError("aggregate summary digest mismatch")
        if self.query_count != len(self.records):
            raise ValueError("aggregate query count mismatch")
        identities = [opaque_query_identity(str(row.get("case_id") or "")) for row in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("aggregate contains duplicate query")
        if stable_hash(identities) != self.query_order_sha256:
            raise ValueError("aggregate query order digest mismatch")
        return self


def deterministic_assignments(
    query_identities: Sequence[str], shard_count: int
) -> list[list[str]]:
    """Assign by stable global ordinal; no query content or outcome is inspected."""

    if shard_count < 1:
        raise ShardedExecutionError("shard_count_invalid")
    values = [str(value) for value in query_identities]
    if not values or len(values) != len(set(values)):
        raise ShardedExecutionError("query_population_invalid")
    if any(not _OPAQUE_QUERY_RE.fullmatch(value) for value in values):
        raise ShardedExecutionError("query_identity_not_opaque")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for ordinal, identity in enumerate(values):
        shards[ordinal % shard_count].append(identity)
    return shards


def build_common_execution_contract(manifest: RunManifestV1) -> dict[str, Any]:
    return {
        "dataset": manifest.dataset.model_dump(mode="json"),
        "prompt": manifest.prompt.model_dump(mode="json"),
        "configuration": manifest.configuration.model_dump(mode="json"),
        "evaluator": manifest.evaluator.model_dump(mode="json"),
        "determinism": manifest.determinism.model_dump(mode="json"),
        "execution_profile": "offline_committed_replay_v1",
        "merge_policy": "global_plan_order_preserve_all_terminal_states",
    }


def build_shard_plan(
    *,
    plan_id: str,
    monolithic_manifest: RunManifestV1,
    query_identities: Sequence[str],
    data_identity_sha256: str,
    replay_input_sha256: str,
    shard_count: int,
    generation_config: Mapping[str, Any],
) -> ShardPlanV1:
    identities = list(query_identities)
    assignments = deterministic_assignments(identities, shard_count)
    common = build_common_execution_contract(monolithic_manifest)
    return ShardPlanV1(
        plan_id=plan_id,
        queries=QueryPopulation(
            count=len(identities),
            identities=identities,
            stable_identity_sha256=monolithic_manifest.queries.stable_identity_sha256,
            order_sha256=monolithic_manifest.queries.order_sha256,
        ),
        data_identity_sha256=data_identity_sha256,
        replay_input_sha256=replay_input_sha256,
        shard_count=shard_count,
        shards=[
            ShardAssignment(
                shard_index=index,
                query_identities=values,
                query_identities_sha256=stable_hash(values),
            )
            for index, values in enumerate(assignments)
        ],
        common_execution_contract=common,
        common_execution_contract_sha256=stable_hash(common),
        generation_configuration_sha256=stable_hash(
            normalize_generation_config(generation_config)
        ),
    )


def load_shard_plan(path: Path) -> ShardPlanV1:
    try:
        return ShardPlanV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise ShardedExecutionError("shard_plan_invalid") from exc


def load_gate_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardedExecutionError("sharded_execution_protocol_unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != GATE_CONTRACT
        or payload.get("contracts")
        != {
            "aggregate": AGGREGATE_CONTRACT,
            "attempt_set": ATTEMPT_SET_CONTRACT,
            "plan": PLAN_CONTRACT,
        }
        or (payload.get("assignment") or {}).get("algorithm")
        != ASSIGNMENT_ALGORITHM
        or payload.get("score_scope")
        != "partition_and_merge_only_not_quality_or_official_score"
    ):
        raise ShardedExecutionError("sharded_execution_protocol_incompatible")
    return payload


def load_attempt_set(path: Path) -> ShardAttemptSetV1:
    try:
        return ShardAttemptSetV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise ShardedExecutionError("shard_attempt_set_invalid") from exc


def write_shard_plan(path: Path, plan: ShardPlanV1) -> None:
    write_json(path, plan.model_dump(mode="json"))


def write_attempt_set(path: Path, attempt_set: ShardAttemptSetV1) -> None:
    write_json(path, attempt_set.model_dump(mode="json"))


def shard_binding(
    plan_path: Path,
    shard_index: int,
    attempt_id: str,
    supersedes_attempt_id: str | None = None,
) -> dict[str, Any]:
    plan = load_shard_plan(plan_path)
    if shard_index < 0 or shard_index >= plan.shard_count:
        raise ShardedExecutionError("shard_index_invalid")
    if not _ATTEMPT_RE.fullmatch(attempt_id):
        raise ShardedExecutionError("attempt_id_invalid")
    if supersedes_attempt_id is not None and (
        not _ATTEMPT_RE.fullmatch(supersedes_attempt_id)
        or supersedes_attempt_id == attempt_id
    ):
        raise ShardedExecutionError("supersedes_attempt_id_invalid")
    assignment = plan.shards[shard_index]
    return {
        "contract": PLAN_CONTRACT,
        "plan_sha256": sha256_file(plan_path),
        "shard_index": shard_index,
        "shard_count": plan.shard_count,
        "expected_query_identities_sha256": assignment.query_identities_sha256,
        "common_execution_contract_sha256": plan.common_execution_contract_sha256,
        "attempt_id": attempt_id,
        "supersedes_attempt_id": supersedes_attempt_id,
    }


def query_ids_for_shard(plan: ShardPlanV1, shard_index: int) -> tuple[str, ...]:
    if shard_index < 0 or shard_index >= plan.shard_count:
        raise ShardedExecutionError("shard_index_invalid")
    return tuple(plan.shards[shard_index].query_identities)


def select_queries_for_shard(
    queries: Sequence[Any], plan: ShardPlanV1, shard_index: int
) -> list[Any]:
    """Select a plan-bound shard while preserving the plan's global order."""

    by_identity: dict[str, Any] = {}
    observed: list[str] = []
    for query in queries:
        identity = opaque_query_identity(str(query.query_id))
        if identity in by_identity:
            raise ShardedExecutionError("query_population_duplicate")
        by_identity[identity] = query
        observed.append(identity)
    if observed != plan.queries.identities:
        raise ShardedExecutionError("shard_plan_query_population_or_order_mismatch")
    expected = query_ids_for_shard(plan, shard_index)
    return [by_identity[identity] for identity in expected]


def normalize_generation_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only registered run/shard-local fields from generation-zero config."""

    return {
        str(key): copy.deepcopy(item)
        for key, item in sorted(value.items())
        if key not in _GENERATION_CONFIG_EXCLUDES
    }


def validate_and_merge(
    plan_path: Path,
    attempt_set_path: Path,
    *,
    repository_root: Path,
    output_path: Path | None = None,
    monolithic_manifest_path: Path | None = None,
    controlled_fault: Literal[
        "duplicate_query", "missing_query", "common_success_filter", "config_drift"
    ]
    | None = None,
) -> dict[str, Any]:
    """Audit shard attempts and optionally write one immutable aggregate."""

    root = repository_root.resolve()
    plan = load_shard_plan(plan_path)
    attempts = load_attempt_set(attempt_set_path)
    plan_digest = sha256_file(plan_path)
    violations: list[dict[str, Any]] = []
    not_ready: list[dict[str, Any]] = []
    if attempts.plan_sha256 != plan_digest:
        violations.append(_violation("attempt_set_plan_hash_mismatch", path="$.plan_sha256"))

    selected: dict[int, tuple[ShardAttemptReference, RunManifestV1, Any]] = {}
    by_shard: dict[int, list[ShardAttemptReference]] = defaultdict(list)
    for reference in attempts.attempts:
        if reference.shard_index >= plan.shard_count:
            violations.append(
                _violation(
                    "attempt_shard_out_of_range",
                    shard=reference.shard_index,
                    attempt=reference.attempt_id,
                    path="$.attempts",
                )
            )
            continue
        by_shard[reference.shard_index].append(reference)

    for shard_index in range(plan.shard_count):
        references = by_shard.get(shard_index, [])
        if not references:
            not_ready.append(
                _pending("shard_attempt_missing", shard=shard_index, attempt=None)
            )
            continue
        tip = _select_attempt_tip(references, shard_index, violations)
        loaded: dict[str, tuple[RunManifestV1, Any]] = {}
        for reference in references:
            loaded_item = _load_and_validate_attempt(
                reference,
                plan=plan,
                plan_path=plan_path,
                repository_root=root,
                violations=violations,
                controlled_fault=(
                    controlled_fault if reference is tip else None
                ),
            )
            if loaded_item is not None:
                loaded[reference.attempt_id] = loaded_item
        if tip is None or tip.attempt_id not in loaded:
            continue
        manifest, state = loaded[tip.attempt_id]
        if state.status != "completed" or len(state.records) != len(
            plan.shards[shard_index].query_identities
        ):
            not_ready.append(
                _pending(
                    "shard_attempt_incomplete",
                    shard=shard_index,
                    attempt=tip.attempt_id,
                )
            )
            continue
        selected[shard_index] = (tip, manifest, state)

    if controlled_fault in {"duplicate_query", "missing_query", "common_success_filter"}:
        violations.append(
            _violation(
                {
                    "duplicate_query": "aggregate_duplicate_query",
                    "missing_query": "aggregate_query_missing",
                    "common_success_filter": "outcome_based_query_filtering",
                }[controlled_fault],
                path="$.records",
            )
        )
    aggregate: ShardAggregateV1 | None = None
    if not violations and not not_ready and len(selected) == plan.shard_count:
        aggregate = _build_aggregate(
            plan,
            plan_path=plan_path,
            plan_digest=plan_digest,
            selected=selected,
            repository_root=root,
            controlled_fault=controlled_fault,
        )
        aggregate_violations = _validate_aggregate_model(
            aggregate, plan=plan, selected=selected, repository_root=root
        )
        violations.extend(aggregate_violations)
        if not violations and monolithic_manifest_path is not None:
            violations.extend(
                _compare_monolithic(
                    aggregate,
                    monolithic_manifest_path=monolithic_manifest_path,
                    repository_root=root,
                )
            )
        if not violations and output_path is not None:
            if output_path.exists():
                raise ShardedExecutionError("aggregate_output_already_exists")
            durable_atomic_write_bytes(
                output_path,
                stable_json_bytes(aggregate.model_dump(mode="json")),
                temporary_suffix="shard-aggregate",
            )

    if violations:
        status, exit_code = "violation", EXIT_VIOLATION
    elif not_ready:
        status, exit_code = "not_ready", EXIT_NOT_READY
    else:
        status, exit_code = "passed", EXIT_PASSED
    return _report(
        status=status,
        exit_code=exit_code,
        plan_sha256=plan_digest,
        shard_count=plan.shard_count,
        selected_shard_count=len(selected),
        query_count=plan.queries.count,
        aggregate_sha256=(
            aggregate.aggregate_summary_sha256 if aggregate is not None else None
        ),
        terminal_counts=(aggregate.terminal_counts if aggregate is not None else {}),
        violations=violations,
        pending=not_ready,
        observation={
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
            "outcome_based_attempt_selection": False,
        },
    )


def validate_aggregate(
    aggregate_path: Path,
    plan_path: Path,
    attempt_set_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    expected = validate_and_merge(
        plan_path,
        attempt_set_path,
        repository_root=repository_root,
    )
    if expected["exit_code"] != EXIT_PASSED:
        return expected
    try:
        aggregate = ShardAggregateV1.model_validate_json(
            aggregate_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, json.JSONDecodeError):
        return _report(
            status="violation",
            exit_code=EXIT_VIOLATION,
            violations=[_violation("aggregate_invalid", path="$")],
            pending=[],
            observation={
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "quality_metric_count": 0,
            },
        )
    if aggregate.aggregate_summary_sha256 != expected["aggregate_sha256"]:
        return _report(
            status="violation",
            exit_code=EXIT_VIOLATION,
            violations=[_violation("aggregate_reference_or_content_tampered", path="$")],
            pending=[],
            observation={
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "quality_metric_count": 0,
            },
        )
    return expected


def audit_frozen_eligibility(legacy_audit_path: Path) -> dict[str, Any]:
    payload = json.loads(legacy_audit_path.read_text(encoding="utf-8"))
    profiles = [
        {
            "profile_id": item["profile_id"],
            "status": "not_eligible",
            "missing_contracts": [
                "shard_plan_v1_prebinding",
                "independent_shard_attempt_lineage",
                "complete_global_query_population",
            ],
            "historical_artifacts_modified": False,
        }
        for item in sorted(payload.get("profiles", []), key=lambda row: row["profile_id"])
    ]
    return _report(
        status="not_eligible",
        exit_code=EXIT_NOT_READY,
        shard_count=0,
        selected_shard_count=0,
        query_count=0,
        aggregate_sha256=None,
        terminal_counts={},
        violations=[],
        pending=[],
        frozen_profiles=profiles,
        observation={
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    )


def _select_attempt_tip(
    references: Sequence[ShardAttemptReference],
    shard_index: int,
    violations: list[dict[str, Any]],
) -> ShardAttemptReference | None:
    by_id = {item.attempt_id: item for item in references}
    children: dict[str, list[str]] = defaultdict(list)
    for item in references:
        parent = item.supersedes_attempt_id
        if parent is None:
            continue
        if parent not in by_id:
            violations.append(
                _violation(
                    "superseded_attempt_missing",
                    shard=shard_index,
                    attempt=item.attempt_id,
                    path="$.attempts.supersedes_attempt_id",
                )
            )
        children[parent].append(item.attempt_id)
    for parent, values in sorted(children.items()):
        if len(values) != 1:
            violations.append(
                _violation(
                    "attempt_supersession_branch",
                    shard=shard_index,
                    attempt=parent,
                    path="$.attempts",
                )
            )
    roots = [item for item in references if item.supersedes_attempt_id is None]
    tips = [item for item in references if item.attempt_id not in children]
    if len(roots) != 1 or len(tips) != 1:
        violations.append(
            _violation(
                "unique_final_attempt_missing",
                shard=shard_index,
                attempt=None,
                path="$.attempts",
            )
        )
        return None
    visited: set[str] = set()
    current = tips[0]
    while current is not None:
        if current.attempt_id in visited:
            violations.append(
                _violation(
                    "attempt_supersession_cycle",
                    shard=shard_index,
                    attempt=current.attempt_id,
                    path="$.attempts",
                )
            )
            return None
        visited.add(current.attempt_id)
        parent = current.supersedes_attempt_id
        current = by_id.get(parent) if parent is not None else None
    if len(visited) != len(references):
        violations.append(
            _violation(
                "attempt_lineage_disconnected",
                shard=shard_index,
                attempt=tips[0].attempt_id,
                path="$.attempts",
            )
        )
        return None
    return tips[0]


def _load_and_validate_attempt(
    reference: ShardAttemptReference,
    *,
    plan: ShardPlanV1,
    plan_path: Path,
    repository_root: Path,
    violations: list[dict[str, Any]],
    controlled_fault: str | None,
) -> tuple[RunManifestV1, Any] | None:
    try:
        manifest_path = resolve_repo_path(repository_root, reference.manifest_path)
    except ValueError:
        violations.append(
            _violation(
                "attempt_manifest_path_invalid",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.manifest_path",
            )
        )
        return None
    if not manifest_path.is_file() or sha256_file(manifest_path) != reference.manifest_sha256:
        violations.append(
            _violation(
                "attempt_manifest_hash_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.manifest_sha256",
            )
        )
        return None
    validation = validate_run_manifest(manifest_path, repository_root=repository_root)
    if validation["status"] != "passed":
        violations.append(
            _violation(
                "attempt_run_manifest_invalid",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$",
            )
        )
        return None
    manifest = RunManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.shard
    expected_binding = shard_binding(
        plan_path,
        reference.shard_index,
        reference.attempt_id,
        reference.supersedes_attempt_id,
    )
    observed_binding = None
    if binding is not None:
        observed_binding = {
            key: value
            for key, value in binding.model_dump(mode="json").items()
            if key != "plan"
        }
    if observed_binding != expected_binding:
        violations.append(
            _violation(
                "run_manifest_shard_binding_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.shard",
            )
        )
    expected_queries = plan.shards[reference.shard_index].query_identities
    try:
        manifest_queries = manifest_query_identities(manifest, repository_root)
    except (OSError, ValueError, json.JSONDecodeError):
        manifest_queries = []
    if manifest_queries != expected_queries:
        violations.append(
            _violation(
                "shard_manifest_query_population_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.queries",
            )
        )
    if build_common_execution_contract(manifest) != plan.common_execution_contract:
        violations.append(
            _violation(
                "shard_common_execution_contract_drift",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.execution_contract",
            )
        )
    try:
        state = BenchmarkRunCommitStore(
            resolve_repo_path(repository_root, manifest.output_directory)
        ).load_latest()
    except (CrashConsistencyError, ValueError):
        violations.append(
            _violation(
                "shard_generation_lineage_invalid",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.generation_chain",
            )
        )
        return None
    if (
        state.run_id != manifest.run_id
        or len(state.records) != manifest.progress.completed_count
        or (manifest.progress.status == "completed" and state.status != "completed")
    ):
        violations.append(
            _violation(
                "manifest_generation_state_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.progress",
            )
        )
    expected_state_ids = expected_queries
    observed_state_ids = [opaque_query_identity(value) for value in state.expected_query_ids]
    if observed_state_ids != expected_state_ids:
        violations.append(
            _violation(
                "generation_expected_query_population_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.expected_query_ids",
            )
        )
    if state.config.get("shard") != expected_binding:
        violations.append(
            _violation(
                "generation_zero_shard_binding_mismatch",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.config.shard",
            )
        )
    normalized = normalize_generation_config(state.config)
    if controlled_fault == "config_drift":
        normalized["hidden_drift"] = True
    if stable_hash(normalized) != plan.generation_configuration_sha256:
        violations.append(
            _violation(
                "shard_generation_configuration_drift",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                path="$.config",
            )
        )
    duplicates = _duplicate_record_commits(
        BenchmarkRunCommitStore(resolve_repo_path(repository_root, manifest.output_directory))
    )
    for identity in duplicates:
        violations.append(
            _violation(
                "duplicate_query_commit",
                shard=reference.shard_index,
                attempt=reference.attempt_id,
                query_identity=identity,
                path="$.generation_chain",
            )
        )
    _validate_records(
        state.records,
        shard=reference.shard_index,
        attempt=reference.attempt_id,
        expected=expected_queries,
        complete=state.status == "completed",
        violations=violations,
    )
    return manifest, state


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shard: int,
    attempt: str,
    expected: Sequence[str],
    complete: bool,
    violations: list[dict[str, Any]],
) -> None:
    identities: list[str] = []
    for row in records:
        identity = opaque_query_identity(str(row.get("case_id") or ""))
        identities.append(identity)
        if _FORBIDDEN_FIELDS & set(row):
            violations.append(
                _violation(
                    "quality_or_target_field_forbidden",
                    shard=shard,
                    attempt=attempt,
                    query_identity=identity,
                    path="$.records",
                )
            )
        if str(row.get("status")) not in _TERMINAL_STATUSES:
            violations.append(
                _violation(
                    "query_terminal_status_invalid",
                    shard=shard,
                    attempt=attempt,
                    query_identity=identity,
                    path="$.records.status",
                )
            )
        if not isinstance(row.get("semantic_events"), list):
            violations.append(
                _violation(
                    "semantic_events_missing",
                    shard=shard,
                    attempt=attempt,
                    query_identity=identity,
                    path="$.records.semantic_events",
                )
            )
    if len(identities) != len(set(identities)):
        violations.append(
            _violation(
                "duplicate_query_record",
                shard=shard,
                attempt=attempt,
                path="$.records",
            )
        )
    if identities != list(expected[: len(identities)]):
        violations.append(
            _violation(
                "shard_record_order_or_membership_drift",
                shard=shard,
                attempt=attempt,
                path="$.records",
            )
        )
    if complete and identities != list(expected):
        violations.append(
            _violation(
                "completed_shard_coverage_incomplete",
                shard=shard,
                attempt=attempt,
                path="$.records",
            )
        )


def _build_aggregate(
    plan: ShardPlanV1,
    *,
    plan_path: Path,
    plan_digest: str,
    selected: Mapping[int, tuple[ShardAttemptReference, RunManifestV1, Any]],
    repository_root: Path,
    controlled_fault: str | None,
) -> ShardAggregateV1:
    rows_by_identity: dict[str, dict[str, Any]] = {}
    selected_refs: list[SelectedShardReference] = []
    resource_ledger_refs: list[ShardResourceLedgerReference] = []
    query_commit_events: dict[str, dict[str, Any]] = {}
    for shard_index in range(plan.shard_count):
        reference, manifest, state = selected[shard_index]
        store = BenchmarkRunCommitStore(
            resolve_repo_path(repository_root, manifest.output_directory)
        )
        for row in state.records:
            identity = opaque_query_identity(str(row.get("case_id") or ""))
            rows_by_identity[identity] = copy.deepcopy(dict(row))
        for event in _committed_events(store, state.generation):
            if event.get("event") != "query_state_committed":
                continue
            identity = opaque_query_identity(str(event.get("query_identity") or ""))
            query_commit_events[identity] = {
                "event": "query_state_committed",
                "query_identity": identity,
                "query_status": event.get("query_status"),
            }
        generation_manifest = state.generation_path / "generation_manifest.json"
        selected_refs.append(
            SelectedShardReference(
                shard_index=shard_index,
                attempt_id=reference.attempt_id,
                manifest_path=reference.manifest_path,
                manifest_sha256=reference.manifest_sha256,
                run_id=manifest.run_id,
                commit_generation=state.generation,
                generation_manifest_sha256=sha256_file(generation_manifest),
                record_count=len(state.records),
                event_count=state.event_count,
            )
        )
        if manifest.resource_ledger is not None:
            resource_ledger_refs.append(
                ShardResourceLedgerReference(
                    shard_index=shard_index,
                    attempt_id=reference.attempt_id,
                    path=manifest.resource_ledger.output.path,
                    sha256=manifest.resource_ledger.output.sha256,
                    manifest_identity=manifest.resource_ledger.manifest_identity,
                )
            )
    records = [rows_by_identity[identity] for identity in plan.queries.identities]
    commit_events = [
        query_commit_events[identity] for identity in plan.queries.identities
    ]
    terminal_counts = dict(sorted(Counter(str(row.get("status")) for row in records).items()))
    operational: Counter[str] = Counter()
    for row in records:
        values = row.get("operational_counts")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    operational[str(key)] += value
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": AGGREGATE_CONTRACT,
        "score_scope": "partition_and_merge_only_not_quality_or_official_score",
        "plan_path": _relative_path(plan_path, repository_root),
        "plan_sha256": plan_digest,
        "query_count": len(records),
        "query_order_sha256": stable_hash(
            [opaque_query_identity(str(row.get("case_id") or "")) for row in records]
        ),
        "selected_shards": [item.model_dump(mode="json") for item in selected_refs],
        "records": records,
        "commit_events": commit_events,
        "terminal_counts": terminal_counts,
        "operational_counts": dict(sorted(operational.items())),
        "resource_ledgers": [
            item.model_dump(mode="json") for item in resource_ledger_refs
        ],
        "completed": len(records) == plan.queries.count,
    }
    return ShardAggregateV1(
        **payload,
        aggregate_summary_sha256=stable_hash(payload),
    )


def _validate_aggregate_model(
    aggregate: ShardAggregateV1,
    *,
    plan: ShardPlanV1,
    selected: Mapping[int, tuple[ShardAttemptReference, RunManifestV1, Any]],
    repository_root: Path,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    identities = [opaque_query_identity(str(row.get("case_id") or "")) for row in aggregate.records]
    if identities != plan.queries.identities:
        violations.append(_violation("aggregate_global_query_order_or_coverage_mismatch", path="$.records"))
    if not aggregate.completed:
        violations.append(_violation("aggregate_incorrectly_incomplete", path="$.completed"))
    if [item.shard_index for item in aggregate.selected_shards] != list(range(plan.shard_count)):
        violations.append(_violation("aggregate_shard_reference_order_mismatch", path="$.selected_shards"))
    for item in aggregate.selected_shards:
        reference, manifest, state = selected[item.shard_index]
        if (
            item.attempt_id != reference.attempt_id
            or item.run_id != manifest.run_id
            or item.commit_generation != state.generation
            or item.manifest_sha256 != sha256_file(
                resolve_repo_path(repository_root, reference.manifest_path)
            )
        ):
            violations.append(
                _violation(
                    "aggregate_selected_shard_reference_mismatch",
                    shard=item.shard_index,
                    attempt=item.attempt_id,
                    path="$.selected_shards",
                )
            )
    expected_ledgers = []
    for shard_index in range(plan.shard_count):
        reference, manifest, _state = selected[shard_index]
        if manifest.resource_ledger is None:
            continue
        expected_ledgers.append(
            {
                "shard_index": shard_index,
                "attempt_id": reference.attempt_id,
                "path": manifest.resource_ledger.output.path,
                "sha256": manifest.resource_ledger.output.sha256,
                "manifest_identity": manifest.resource_ledger.manifest_identity,
            }
        )
    observed_ledgers = [
        item.model_dump(mode="json") for item in aggregate.resource_ledgers
    ]
    if observed_ledgers != expected_ledgers:
        violations.append(
            _violation(
                "aggregate_resource_ledger_selection_mismatch",
                path="$.resource_ledgers",
            )
        )
    return violations


def _compare_monolithic(
    aggregate: ShardAggregateV1,
    *,
    monolithic_manifest_path: Path,
    repository_root: Path,
) -> list[dict[str, Any]]:
    validation = validate_run_manifest(monolithic_manifest_path, repository_root=repository_root)
    if validation["status"] != "passed":
        return [_violation("monolithic_run_manifest_invalid", path="$.monolithic")]
    manifest = RunManifestV1.model_validate_json(
        monolithic_manifest_path.read_text(encoding="utf-8")
    )
    try:
        state = BenchmarkRunCommitStore(
            resolve_repo_path(repository_root, manifest.output_directory)
        ).load_latest()
    except CrashConsistencyError:
        return [_violation("monolithic_generation_invalid", path="$.monolithic")]
    if state.status != "completed":
        return [_violation("monolithic_run_not_completed", path="$.monolithic")]
    left = [copy.deepcopy(dict(row)) for row in state.records]
    right = aggregate.records
    difference = _first_difference(left, right)
    return (
        []
        if difference is None
        else [
            _violation(
                "monolithic_sharded_semantic_mismatch",
                path=difference,
            )
        ]
    )


def manifest_query_identities(
    manifest: RunManifestV1, repository_root: Path
) -> list[str]:
    path = resolve_repo_path(repository_root, manifest.queries.input.path)
    identities: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            identities.append(opaque_query_identity(str(row[manifest.queries.id_field])))
    return identities


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
            counts[opaque_query_identity(str(delta["record"].get("case_id") or ""))] += 1
    return sorted(key for key, count in counts.items() if count > 1)


def _committed_events(store: BenchmarkRunCommitStore, latest_generation: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for generation in range(1, latest_generation + 1):
        path = store.generations / f"generation-{generation:08d}"
        if not (path / "COMMITTED").is_file():
            raise ShardedExecutionError("committed_event_generation_missing")
        with (path / "events.jsonl").open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    events.append(json.loads(raw))
    return events


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ShardedExecutionError("path_outside_repository_root") from exc


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if set(left) != set(right):
            key = sorted(set(left) ^ set(right))[0]
            return f"{path}/{key}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}/{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}/length"
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_value, right_value, f"{path}/{index}")
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def _violation(
    invariant: str,
    *,
    path: str,
    shard: int | None = None,
    attempt: str | None = None,
    query_identity: str | None = None,
) -> dict[str, Any]:
    return {
        "invariant": invariant,
        "shard": shard,
        "attempt": attempt,
        "query_identity": query_identity,
        "first_difference_path": path,
        "normalized_summary_sha256": stable_hash(
            {"invariant": invariant, "path": path, "shard": shard, "attempt": attempt}
        ),
    }


def _pending(reason: str, *, shard: int, attempt: str | None) -> dict[str, Any]:
    return {"reason": reason, "shard": shard, "attempt": attempt}


def _report(**values: Any) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_CONTRACT,
        "gate": GATE_NAME,
        "score_scope": "partition_and_merge_only_not_quality_or_official_score",
        **values,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def build_local_fixture(
    root: Path,
    *,
    shard_count: int = 3,
    query_count: int = 7,
    query_identities: Sequence[str] | None = None,
    completion_order: Sequence[int] | None = None,
    incomplete_shard: int | None = None,
    retry_shard: int | None = None,
    reverse_query_completion: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Create a gold-free monolithic run and equivalent committed shards."""

    root.mkdir(parents=True, exist_ok=True)
    query_ids = list(query_identities or ())
    if query_identities is not None:
        query_count = len(query_ids)
        deterministic_assignments(query_ids, shard_count)
    else:
        query_ids = [
            f"query:{hashlib.sha256(f'shard-fixture-{index}'.encode()).hexdigest()}"
            for index in range(query_count)
        ]
    rows = [
        {"query_id": identity, "query": f"offline shard fixture query {index}"}
        for index, identity in enumerate(query_ids)
    ]
    _write_jsonl(root / "inputs/queries.jsonl", rows)
    (root / "inputs/dataset.txt").write_text("offline-shard-dataset-v1\n", encoding="utf-8")
    (root / "inputs/replay.json").write_text('{"offline":true}\n', encoding="utf-8")
    write_json(root / "inputs/prompt.json", {"planner": "current_rules_fixture_v1"})
    git_payload = {
        "commit": "b" * 40,
        "dirty_paths": ["third_party/paper-qa"],
        "allowed_dirty_paths": ["third_party/paper-qa"],
        "unexpected_dirty_paths": [],
    }
    git = GitProvenance(
        **git_payload,
        dirty=True,
        worktree_state_sha256=stable_hash(git_payload),
    )
    common_config = _fixture_generation_config(query_ids)
    monolithic_dir = root / "runs/monolithic"
    monolithic_store = BenchmarkRunCommitStore(monolithic_dir)
    state = monolithic_store.initialize(
        run_id="offline-shard-monolithic",
        expected_query_ids=query_ids,
        config=common_config,
        dataset_report={"identity": "offline-shard-dataset-v1"},
    )
    for index, identity in enumerate(query_ids):
        state = monolithic_store.commit_record(_fixture_record(identity, index))
    state = monolithic_store.commit_completion({})
    monolithic_store.materialize_compatibility_view(state)
    monolithic_spec = _fixture_manifest_spec(
        root,
        role="monolithic",
        query_path="inputs/queries.jsonl",
        expected_count=query_count,
        completed_count=query_count,
        plan=None,
        shard_index=None,
        attempt_id=None,
        supersedes=None,
    )
    monolithic_manifest = build_run_manifest(
        monolithic_spec, repository_root=root, git_provenance=git
    )
    monolithic_manifest_path = root / "monolithic_run_manifest.json"
    write_json(monolithic_manifest_path, monolithic_manifest.model_dump(mode="json"))

    plan = build_shard_plan(
        plan_id="offline-gold-blind-shard-fixture",
        monolithic_manifest=monolithic_manifest,
        query_identities=query_ids,
        data_identity_sha256=monolithic_manifest.dataset.identity_summary_sha256,
        replay_input_sha256=sha256_file(root / "inputs/replay.json"),
        shard_count=shard_count,
        generation_config=common_config,
    )
    plan_path = root / "inputs/shard_plan.json"
    write_shard_plan(plan_path, plan)

    order = list(completion_order or range(shard_count))
    if sorted(order) != list(range(shard_count)):
        raise ShardedExecutionError("fixture_completion_order_invalid")
    references: list[ShardAttemptReference] = []
    for shard_index in order:
        assignment = plan.shards[shard_index].query_identities
        query_path = root / f"inputs/shard-{shard_index}.jsonl"
        _write_jsonl(
            query_path,
            [row for row in rows if row["query_id"] in set(assignment)],
        )
        attempt_specs = [("attempt-0", None, incomplete_shard == shard_index)]
        if retry_shard == shard_index:
            attempt_specs = [
                ("attempt-0", None, True),
                ("attempt-1", "attempt-0", False),
            ]
        for attempt_id, supersedes, partial in attempt_specs:
            role = f"shard-{shard_index}-{attempt_id}"
            run_dir = root / "runs" / role
            binding = shard_binding(
                plan_path, shard_index, attempt_id, supersedes
            )
            shard_config = copy.deepcopy(common_config)
            shard_config["case_count"] = len(assignment)
            shard_config["case_ids"] = list(assignment)
            store = BenchmarkRunCommitStore(run_dir)
            state = store.initialize(
                run_id=f"offline-{role}",
                expected_query_ids=assignment,
                config=shard_config,
                dataset_report={"identity": "offline-shard-dataset-v1"},
                shard_binding=binding,
            )
            limit = max(0, len(assignment) - 1) if partial else len(assignment)
            completed_identities = list(assignment[:limit])
            if reverse_query_completion:
                completed_identities.reverse()
            for identity in completed_identities:
                state = store.commit_record(_fixture_record(identity, query_ids.index(identity)))
            if not partial:
                state = store.commit_completion({})
            store.materialize_compatibility_view(state)
            spec = _fixture_manifest_spec(
                root,
                role=role,
                query_path=query_path.relative_to(root).as_posix(),
                expected_count=len(assignment),
                completed_count=limit,
                plan=plan,
                shard_index=shard_index,
                attempt_id=attempt_id,
                supersedes=supersedes,
            )
            manifest = build_run_manifest(spec, repository_root=root, git_provenance=git)
            manifest_path = root / f"{role}-run-manifest.json"
            write_json(manifest_path, manifest.model_dump(mode="json"))
            references.append(
                ShardAttemptReference(
                    shard_index=shard_index,
                    attempt_id=attempt_id,
                    supersedes_attempt_id=supersedes,
                    manifest_path=manifest_path.relative_to(root).as_posix(),
                    manifest_sha256=sha256_file(manifest_path),
                )
            )
    attempt_set = ShardAttemptSetV1(
        plan_sha256=sha256_file(plan_path),
        attempts=sorted(references, key=lambda item: (item.shard_index, item.attempt_id)),
    )
    attempt_set_path = root / "attempts.json"
    write_attempt_set(attempt_set_path, attempt_set)
    return plan_path, attempt_set_path, monolithic_manifest_path, root


def deterministic_fixture_report(
    *,
    shard_count: int = 3,
    completion_order: Sequence[int] | None = None,
    incomplete_shard: int | None = None,
    retry_shard: int | None = None,
    reverse_query_completion: bool = False,
    controlled_fault: Literal[
        "duplicate_query", "missing_query", "common_success_filter", "config_drift"
    ]
    | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="spar-sharded-execution-") as value:
        root = Path(value)
        plan, attempts, monolithic, _ = build_local_fixture(
            root,
            shard_count=shard_count,
            completion_order=completion_order,
            incomplete_shard=incomplete_shard,
            retry_shard=retry_shard,
            reverse_query_completion=reverse_query_completion,
        )
        return validate_and_merge(
            plan,
            attempts,
            repository_root=root,
            monolithic_manifest_path=monolithic,
            controlled_fault=controlled_fault,
        )


def _fixture_record(identity: str, index: int) -> dict[str, Any]:
    statuses = ("succeeded", "failed", "cancelled", "excluded")
    status = statuses[index % len(statuses)]
    return {
        "case_id": identity,
        "status": status,
        "source_terminals": [
            {
                "source": source,
                "status": "success" if status == "succeeded" else "failed",
                "reason": None if status == "succeeded" else status,
            }
            for source in ("arxiv", "openalex", "semantic_scholar", "pubmed")
        ],
        "normalized_results": [
            {
                "identity_sha256": stable_hash({"query": identity, "rank": rank}),
                "rank": rank,
            }
            for rank in range(1, 3)
        ],
        "semantic_events": ["query_started", "query_terminal"],
        "field_lineage_sha256": stable_hash({"query": identity, "lineage": "fixture"}),
        "operational_counts": {"request_count": 4, "failure_count": int(status != "succeeded")},
    }


def _fixture_generation_config(query_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "case_count": len(query_ids),
        "case_ids": list(query_ids),
        "dataset": {"name": "gold_blind_shard_fixture", "version": "1"},
        "prompt": {"versions": {"planner": "current_rules_fixture_v1"}},
        "configuration": {
            "sources": ["arxiv", "openalex", "semantic_scholar", "pubmed"],
            "budgets": {"max_queries": 4, "top_k": 20},
            "values": {"execution_profile": "offline_committed_replay_v1"},
        },
        "evaluator": {"name": "identity_only", "version": "1"},
    }


def _fixture_manifest_spec(
    root: Path,
    *,
    role: str,
    query_path: str,
    expected_count: int,
    completed_count: int,
    plan: ShardPlanV1 | None,
    shard_index: int | None,
    attempt_id: str | None,
    supersedes: str | None,
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
            ("results.jsonl", "query_records"),
        )
    ]
    commit_root = run_dir / STORE_DIRECTORY
    excluded = sorted(
        path.relative_to(run_dir).as_posix()
        for path in commit_root.rglob("*")
        if path.is_file()
    )
    spec: dict[str, Any] = {
        "run_id": f"offline-{role}",
        "dataset": {
            "name": "gold_blind_shard_fixture",
            "version": "1",
            "input_paths": ["inputs/dataset.txt", "inputs/replay.json"],
        },
        "queries": {
            "input_path": query_path,
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
            "values": {"execution_profile": "offline_committed_replay_v1"},
        },
        "evaluator": {"name": "identity_only", "version": "1"},
        "determinism": {
            "random_seed": 0,
            "parameters": {"ordering": "stable", "concurrency": 1},
        },
        "progress": {
            "status": "completed" if completed_count == expected_count else "partial",
            "expected_count": expected_count,
            "completed_count": completed_count,
            "record_output_path": f"runs/{role}/results.jsonl",
        },
        "lineage": {
            "checkpoint_id": f"offline-{role}-checkpoint",
            "resume_index": 0,
            "parent": None,
        },
        "output_directory": f"runs/{role}",
        "output_inventory_excludes": excluded,
        "outputs": outputs,
        "metadata_bindings": _fixture_bindings(role),
    }
    if plan is not None and shard_index is not None and attempt_id is not None:
        assignment = plan.shards[shard_index]
        spec["shard"] = {
            "plan_path": "inputs/shard_plan.json",
            "shard_index": shard_index,
            "shard_count": plan.shard_count,
            "expected_query_identities_sha256": assignment.query_identities_sha256,
            "common_execution_contract_sha256": plan.common_execution_contract_sha256,
            "attempt_id": attempt_id,
            "supersedes_attempt_id": supersedes,
        }
    return spec


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(stable_json_bytes(dict(row), indent=None) for row in rows))
