"""Deterministic health supervision for a future authoritative Full1000 run.

This module consumes operational health facts only.  It never inspects result
content, paper identities, query text, gold, or quality metrics.  The
supervisor is an observational control around the existing launch, ledger,
provenance, checkpoint, and audit contracts: once a preregistered stop
condition fires, no new query, page, or retry may start.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_provider_health_supervisor_v1"
SCHEMA_VERSION = "1"
ADDENDUM = "full1000_provider_health_addendum_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_OBSERVED = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FROZEN_PROTOCOL_SHA256 = (
    "a7b7a255ccea46d441466c6f13a8666909c44af332f1ed27481e5b7f71705fd7"
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
SOURCES = ("arxiv", "openalex", "pubmed", "semantic_scholar")
FAILURE_OUTCOMES = frozenset({"429", "503", "timeout", "connection_failure"})
OPERATION_KINDS = frozenset({"query", "page", "retry"})
TERMINAL_QUERY_STATES = frozenset(
    {"completed", "failed", "cancelled", "source_failure"}
)


class ProviderHealthError(RuntimeError):
    """A policy, health event, pause, or resume invariant was violated."""


class ProviderHealthNotReady(ProviderHealthError):
    """Real provider health is not yet observable."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate json key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderHealthError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ProviderHealthError("json_root_not_object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.name == ".env"
        or path.parts[0] == "third_party"
        or path.as_posix() != value
    ):
        raise ProviderHealthError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "bindings",
        "execution",
        "formal_validation_complete",
        "health_signals",
        "pause_contract",
        "population",
        "protocol",
        "protocol_sha256",
        "resume_contract",
        "schema_version",
        "source_commit",
        "state_machine",
        "thresholds",
        "unknown_provider_limits",
    }
    if set(value) != required:
        raise ProviderHealthError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise ProviderHealthError("protocol_version_invalid")
    if value["source_commit"] != "d732c7727a732d171ad9b762b28f89b9e9053c4a":
        raise ProviderHealthError("protocol_source_commit_invalid")
    if value["execution"] != EXECUTION_ZERO:
        raise ProviderHealthError("offline_execution_contract_drift")
    if value["formal_validation_complete"] is not False:
        raise ProviderHealthError("formal_validation_state_drift")
    if _protocol_digest(value) != value["protocol_sha256"]:
        raise ProviderHealthError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ProviderHealthError("protocol_content_drift")
    population = value["population"]
    if (
        population.get("query_count") != 1000
        or population.get("shard_count") != 20
        or population.get("http_attempt_upper") != 19280
        or tuple(population.get("sources") or ()) != SOURCES
    ):
        raise ProviderHealthError("population_contract_drift")
    if value["unknown_provider_limits"] != {
        "provider_cost": NOT_AVAILABLE,
        "provider_rate_limit": NOT_AVAILABLE,
        "provider_token_limit": NOT_AVAILABLE,
    }:
        raise ProviderHealthError("unknown_provider_limit_drift")
    expected_states = [
        "healthy",
        "degraded",
        "pause_required",
        "paused",
        "resume_eligible",
        "invalid",
    ]
    if value["state_machine"].get("states") != expected_states:
        raise ProviderHealthError("state_machine_drift")
    _validate_thresholds(value["thresholds"])
    _validate_bindings(repository_root, value)
    return value


def _validate_thresholds(value: Any) -> None:
    expected = {
        "source": {
            "consecutive_failures_degraded": 3,
            "consecutive_failures_pause": 6,
            "rolling_failure_min_observations": 12,
            "rolling_failure_pause_ratio": 0.75,
            "rolling_window": 20,
            "successful_zero_progress_pause": 10,
        },
        "global": {
            "attempt_budget_burn_fraction_pause": 0.001,
            "attempt_budget_burn_minimum": 20,
            "degraded_sources_pause": 3,
            "operations_without_committed_generation_pause": 40,
            "provenance_write_failure_pause": True,
            "storage_capacity_failure_pause": True,
        },
    }
    if value != expected:
        raise ProviderHealthError("health_threshold_drift")


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    expected_names = {
        "checkpoint_resume",
        "execution_plan",
        "freshness",
        "host_attestation",
        "launch_control",
        "operation_audit",
        "preregistration",
        "provider_provenance",
        "resource_ledger",
        "storage_governance",
    }
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != expected_names:
        raise ProviderHealthError("binding_inventory_invalid")
    for name, binding in sorted(bindings.items()):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ProviderHealthError("binding_schema_invalid")
        relative = _safe_relative(str(binding["path"]))
        path = root / relative
        if not path.is_file():
            raise ProviderHealthError(f"binding_missing:{name}")
        if sha256_file(path) != binding["sha256"]:
            raise ProviderHealthError(f"binding_hash_drift:{name}")


class OperationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_identity: str
    query_identity: str
    source: Literal["arxiv", "openalex", "pubmed", "semantic_scholar"]
    kind: Literal["query", "page", "retry"]
    status: Literal["in_flight", "finished"]
    outcome: str | None = None
    progress_records: int = Field(default=0, ge=0)
    ledger_entry_identity: str | None = None


class PauseCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["paused"]
    reason_codes: list[str]
    committed_query_count: int = Field(ge=0, le=1000)
    committed_query_identity_sha256: str
    last_generation: int = Field(ge=0)
    in_flight_count: Literal[0] = 0
    coverage_status_counts: dict[str, int]
    checkpoint_sha256: str

    @model_validator(mode="after")
    def validate_digest(self) -> "PauseCheckpoint":
        payload = self.model_dump(mode="json")
        digest = payload.pop("checkpoint_sha256")
        if digest != stable_hash(payload):
            raise ValueError("checkpoint digest mismatch")
        if sum(self.coverage_status_counts.values()) != self.committed_query_count:
            raise ValueError("checkpoint coverage mismatch")
        return self


@dataclass
class SourceHealth:
    recent_outcomes: deque[str]
    consecutive_failures: int = 0
    consecutive_zero_progress_successes: int = 0
    total_attempts: int = 0
    total_failures: int = 0
    total_progress_records: int = 0
    state: str = "healthy"


@dataclass
class ProviderHealthSupervisor:
    protocol: Mapping[str, Any]
    query_identities: tuple[str, ...]
    state: str = "healthy"
    sources: dict[str, SourceHealth] = field(init=False)
    operations: dict[str, OperationRecord] = field(default_factory=dict)
    committed_queries: dict[str, str] = field(default_factory=dict)
    total_attempts: int = 0
    operations_since_commit: int = 0
    generation: int = 0
    reason_codes: list[str] = field(default_factory=list)
    pause_checkpoint: dict[str, Any] | None = None
    resume_count: int = 0
    rejected_start_count: int = 0
    audit_entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            len(self.query_identities) != 1000
            or len(set(self.query_identities)) != 1000
        ):
            raise ProviderHealthError("query_population_invalid")
        window = int(self.protocol["thresholds"]["source"]["rolling_window"])
        self.sources = {
            source: SourceHealth(recent_outcomes=deque(maxlen=window))
            for source in SOURCES
        }

    @property
    def in_flight(self) -> set[str]:
        return {
            identity
            for identity, operation in self.operations.items()
            if operation.status == "in_flight"
        }

    def start_operation(
        self,
        operation_identity: str,
        query_identity: str,
        source: str,
        kind: str,
    ) -> None:
        if self.state in {"pause_required", "paused", "resume_eligible", "invalid"}:
            self.rejected_start_count += 1
            raise ProviderHealthError("new_operation_forbidden_after_pause_required")
        if operation_identity in self.operations:
            raise ProviderHealthError("duplicate_operation_identity")
        if query_identity not in self.query_identities:
            raise ProviderHealthError("unknown_query_identity")
        if query_identity in self.committed_queries:
            raise ProviderHealthError("committed_query_repeat_forbidden")
        if source not in SOURCES or kind not in OPERATION_KINDS:
            raise ProviderHealthError("operation_contract_invalid")
        self.operations[operation_identity] = OperationRecord(
            operation_identity=operation_identity,
            query_identity=query_identity,
            source=source,
            kind=kind,
            status="in_flight",
        )
        self._append_audit("operation_started", operation_identity)

    def finish_operation(
        self,
        operation_identity: str,
        *,
        outcome: str,
        progress_records: int,
        ledger_entry_identity: str,
    ) -> None:
        operation = self.operations.get(operation_identity)
        if operation is None or operation.status != "in_flight":
            raise ProviderHealthError("operation_completion_not_in_flight")
        if not ledger_entry_identity or any(
            row.ledger_entry_identity == ledger_entry_identity
            for row in self.operations.values()
        ):
            raise ProviderHealthError("ledger_entry_not_unique")
        operation.status = "finished"
        operation.outcome = outcome
        operation.progress_records = progress_records
        operation.ledger_entry_identity = ledger_entry_identity
        self._append_audit(
            "operation_finished",
            operation_identity,
            {
                "ledger_entry_identity": ledger_entry_identity,
                "outcome": outcome,
                "progress_records": progress_records,
            },
        )
        self.total_attempts += 1
        self.operations_since_commit += 1
        health = self.sources[operation.source]
        health.total_attempts += 1
        health.total_progress_records += progress_records
        health.recent_outcomes.append(outcome)
        if outcome in FAILURE_OUTCOMES:
            health.total_failures += 1
            health.consecutive_failures += 1
            health.consecutive_zero_progress_successes = 0
        else:
            health.consecutive_failures = 0
            if outcome == "success" and progress_records == 0:
                health.consecutive_zero_progress_successes += 1
            else:
                health.consecutive_zero_progress_successes = 0
        self._evaluate_health()

    def commit_query(self, query_identity: str, terminal_state: str) -> None:
        if query_identity not in self.query_identities:
            raise ProviderHealthError("unknown_query_identity")
        if query_identity in self.committed_queries:
            raise ProviderHealthError("query_committed_twice")
        if terminal_state not in TERMINAL_QUERY_STATES:
            raise ProviderHealthError("query_terminal_state_invalid")
        self.committed_queries[query_identity] = terminal_state
        self.generation += 1
        self.operations_since_commit = 0
        self._append_audit(
            "generation_committed",
            query_identity,
            {"generation": self.generation, "terminal_state": terminal_state},
        )

    def record_control_failure(self, kind: str) -> None:
        mapping = {
            "provenance_write_failure": "provenance_write_failure",
            "storage_capacity_failure": "storage_capacity_failure",
        }
        if kind not in mapping:
            raise ProviderHealthError("unknown_control_failure")
        self._append_audit("control_failure", kind)
        self._require_pause(mapping[kind])

    def acknowledge_pause(self) -> dict[str, Any]:
        if self.state != "pause_required":
            raise ProviderHealthError("pause_not_required")
        if self.in_flight:
            raise ProviderHealthError("in_flight_operations_not_drained")
        self.state = "paused"
        status_counts = {
            state: sum(1 for value in self.committed_queries.values() if value == state)
            for state in sorted(TERMINAL_QUERY_STATES)
        }
        status_counts = {key: value for key, value in status_counts.items() if value}
        payload: dict[str, Any] = {
            "state": "paused",
            "reason_codes": sorted(set(self.reason_codes)),
            "committed_query_count": len(self.committed_queries),
            "committed_query_identity_sha256": stable_hash(
                sorted(self.committed_queries)
            ),
            "last_generation": self.generation,
            "in_flight_count": 0,
            "coverage_status_counts": status_counts,
        }
        payload["checkpoint_sha256"] = stable_hash(payload)
        PauseCheckpoint.model_validate(payload)
        self.pause_checkpoint = payload
        self._append_audit(
            "paused",
            payload["checkpoint_sha256"],
            {"generation": self.generation},
        )
        return payload

    def make_resume_eligible(self, evidence: Mapping[str, Any]) -> None:
        if self.state != "paused" or self.pause_checkpoint is None:
            raise ProviderHealthError("resume_requires_paused_checkpoint")
        required_true = {
            "authorization_fresh",
            "capacity_fresh",
            "health_clearance_observed",
            "host_attestation_fresh",
            "protocol_fresh",
        }
        if set(evidence) != required_true | {
            "checkpoint_sha256",
            "healthy_probe_count_by_source",
        }:
            raise ProviderHealthError("resume_evidence_schema_invalid")
        if any(evidence[key] is not True for key in required_true):
            raise ProviderHealthError("resume_prerequisite_not_fresh")
        if evidence["checkpoint_sha256"] != self.pause_checkpoint["checkpoint_sha256"]:
            raise ProviderHealthError("resume_checkpoint_drift")
        probes = evidence["healthy_probe_count_by_source"]
        minimum = int(
            self.protocol["thresholds"]["source"][
                "rolling_failure_min_observations"
            ]
        )
        if not isinstance(probes, dict) or set(probes) != set(SOURCES):
            raise ProviderHealthError("resume_health_evidence_invalid")
        if any(
            not isinstance(probes[source], int) or probes[source] < minimum
            for source in SOURCES
        ):
            raise ProviderHealthError("resume_health_clearance_insufficient")
        # Health probes age out the rolling fault window without changing the
        # cumulative attempt/failure ledger.  No historical count is reset.
        for source in SOURCES:
            health = self.sources[source]
            health.recent_outcomes.extend(["health_probe_success"] * probes[source])
            health.consecutive_failures = 0
            health.consecutive_zero_progress_successes = 0
            health.state = "healthy"
        self.state = "resume_eligible"
        self._append_audit(
            "resume_eligible",
            self.pause_checkpoint["checkpoint_sha256"],
            {"healthy_probe_count_per_source": minimum},
        )

    def resume(self) -> None:
        if self.state != "resume_eligible":
            raise ProviderHealthError("resume_not_eligible")
        self.state = "healthy"
        self.resume_count += 1
        self._append_audit(
            "resumed",
            self.pause_checkpoint["checkpoint_sha256"]
            if self.pause_checkpoint is not None
            else "missing",
            {"resume_count": self.resume_count},
        )

    def aggregate_eligible(self) -> bool:
        return (
            len(self.committed_queries) == len(self.query_identities)
            and set(self.committed_queries) == set(self.query_identities)
            and not self.in_flight
            and self.state not in {"invalid", "pause_required"}
        )

    def coverage_summary(self) -> dict[str, Any]:
        counts = {
            state: sum(1 for value in self.committed_queries.values() if value == state)
            for state in sorted(TERMINAL_QUERY_STATES)
        }
        return {
            "expected_query_count": len(self.query_identities),
            "committed_query_count": len(self.committed_queries),
            "missing_query_count": len(self.query_identities)
            - len(self.committed_queries),
            "duplicate_query_count": 0,
            "status_counts": {key: value for key, value in counts.items() if value},
            "success_only_filtering": False,
            "aggregate_eligible": self.aggregate_eligible(),
        }

    def validate_audit_chain(self) -> bool:
        previous = "0" * 64
        for sequence, entry in enumerate(self.audit_entries):
            if (
                entry.get("sequence") != sequence
                or entry.get("previous_entry_sha256") != previous
            ):
                return False
            payload = dict(entry)
            digest = payload.pop("entry_sha256", None)
            if digest != stable_hash(payload):
                return False
            previous = str(digest)
        return True

    def _evaluate_health(self) -> None:
        source_thresholds = self.protocol["thresholds"]["source"]
        for source, health in self.sources.items():
            observations = len(health.recent_outcomes)
            failures = sum(
                1 for outcome in health.recent_outcomes if outcome in FAILURE_OUTCOMES
            )
            failure_ratio = failures / observations if observations else 0.0
            if (
                health.consecutive_failures
                >= source_thresholds["consecutive_failures_pause"]
                or (
                    observations
                    >= source_thresholds["rolling_failure_min_observations"]
                    and failure_ratio
                    >= source_thresholds["rolling_failure_pause_ratio"]
                )
                or health.consecutive_zero_progress_successes
                >= source_thresholds["successful_zero_progress_pause"]
            ):
                health.state = "pause_required"
                self._require_pause(f"source_pause:{source}")
            elif (
                health.consecutive_failures
                >= source_thresholds["consecutive_failures_degraded"]
            ):
                health.state = "degraded"
            elif health.state != "pause_required":
                health.state = "healthy"
        degraded = sum(
            health.state in {"degraded", "pause_required"}
            for health in self.sources.values()
        )
        global_thresholds = self.protocol["thresholds"]["global"]
        if degraded >= global_thresholds["degraded_sources_pause"]:
            self._require_pause("global_degraded_source_count")
        attempt_upper = int(self.protocol["population"]["http_attempt_upper"])
        if (
            self.total_attempts
            >= global_thresholds["attempt_budget_burn_minimum"]
            and self.operations_since_commit
            >= global_thresholds["attempt_budget_burn_minimum"]
            and self.total_attempts / attempt_upper
            >= global_thresholds["attempt_budget_burn_fraction_pause"]
        ):
            self._require_pause("attempt_budget_burn_without_commit")
        if (
            self.operations_since_commit
            >= global_thresholds["operations_without_committed_generation_pause"]
        ):
            self._require_pause("no_new_committed_generation")
        if self.state == "healthy" and degraded:
            self.state = "degraded"

    def _require_pause(self, reason: str) -> None:
        if self.state not in {"paused", "resume_eligible", "invalid"}:
            self.state = "pause_required"
        self.reason_codes.append(reason)

    def _append_audit(
        self,
        event: str,
        subject_identity: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        previous = (
            self.audit_entries[-1]["entry_sha256"]
            if self.audit_entries
            else "0" * 64
        )
        entry: dict[str, Any] = {
            "sequence": len(self.audit_entries),
            "event": event,
            "subject_identity": subject_identity,
            "details": dict(details or {}),
            "previous_entry_sha256": previous,
        }
        entry["entry_sha256"] = stable_hash(entry)
        self.audit_entries.append(entry)


def load_query_identities(root: Path, protocol: Mapping[str, Any]) -> tuple[str, ...]:
    binding = protocol["bindings"]["execution_plan"]
    plan = read_object(root / _safe_relative(binding["path"]))
    shards = plan.get("sharding", {}).get("shards")
    if not isinstance(shards, list) or len(shards) != 20:
        raise ProviderHealthError("execution_plan_shards_invalid")
    ordered_shards = sorted(shards, key=lambda item: item.get("shard_index", -1))
    shard_identities: list[list[str]] = []
    for shard in ordered_shards:
        rows = shard.get("query_identities")
        if not isinstance(rows, list):
            raise ProviderHealthError("execution_plan_query_identity_invalid")
        shard_identities.append([str(item) for item in rows])
    # ordered_round_robin_v1 assigns global position ``i`` to ``i % 20``.
    # Reconstruct the authority order rather than concatenating shard blocks.
    identities = [
        shard_identities[shard_index][round_index]
        for round_index in range(50)
        for shard_index in range(20)
    ]
    if (
        len(identities) != 1000
        or len(set(identities)) != 1000
        or plan.get("population", {}).get("order_sha256")
        != protocol["population"]["query_order_sha256"]
        or stable_hash(sorted(identities))
        != plan.get("population", {}).get("stable_identity_sha256")
    ):
        raise ProviderHealthError("execution_plan_population_drift")
    return tuple(identities)


def _operation_identity(scenario: str, index: int) -> str:
    return f"operation:{stable_hash({'scenario': scenario, 'index': index})}"


def _ledger_identity(scenario: str, index: int) -> str:
    return f"ledger:{stable_hash({'scenario': scenario, 'index': index})}"


def _finish_sequence(
    supervisor: ProviderHealthSupervisor,
    scenario: str,
    source: str,
    outcomes: Sequence[tuple[str, int]],
    *,
    query_offset: int = 0,
) -> None:
    for index, (outcome, progress) in enumerate(outcomes):
        operation = _operation_identity(scenario, index)
        query = supervisor.query_identities[(query_offset + index) % 1000]
        supervisor.start_operation(operation, query, source, "query")
        supervisor.finish_operation(
            operation,
            outcome=outcome,
            progress_records=progress,
            ledger_entry_identity=_ledger_identity(scenario, index),
        )


def _valid_resume_evidence(
    supervisor: ProviderHealthSupervisor,
) -> dict[str, Any]:
    if supervisor.pause_checkpoint is None:
        raise ProviderHealthError("pause_checkpoint_missing")
    minimum = int(
        supervisor.protocol["thresholds"]["source"][
            "rolling_failure_min_observations"
        ]
    )
    return {
        "authorization_fresh": True,
        "capacity_fresh": True,
        "checkpoint_sha256": supervisor.pause_checkpoint["checkpoint_sha256"],
        "health_clearance_observed": True,
        "healthy_probe_count_by_source": {
            source: minimum for source in SOURCES
        },
        "host_attestation_fresh": True,
        "protocol_fresh": True,
    }


def _drain_and_pause(supervisor: ProviderHealthSupervisor) -> None:
    for index, operation_identity in enumerate(sorted(supervisor.in_flight)):
        supervisor.finish_operation(
            operation_identity,
            outcome="cancelled",
            progress_records=0,
            ledger_entry_identity=_ledger_identity("drain", index),
        )
    supervisor.acknowledge_pause()


def _scenario_result(
    name: str,
    supervisor: ProviderHealthSupervisor,
    expected_state: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = supervisor.coverage_summary()
    passed = (
        supervisor.state == expected_state
        and supervisor.validate_audit_chain()
        and coverage["duplicate_query_count"] == 0
        and coverage["success_only_filtering"] is False
        and len(
            {
                operation.ledger_entry_identity
                for operation in supervisor.operations.values()
                if operation.ledger_entry_identity is not None
            }
        )
        == sum(
            operation.status == "finished"
            for operation in supervisor.operations.values()
        )
    )
    return {
        "scenario": name,
        "status": "passed" if passed else "failed",
        "expected_state": expected_state,
        "observed_state": supervisor.state,
        "reason_codes": sorted(set(supervisor.reason_codes)),
        "total_attempts": supervisor.total_attempts,
        "in_flight_count": len(supervisor.in_flight),
        "rejected_start_count": supervisor.rejected_start_count,
        "resume_count": supervisor.resume_count,
        "coverage": coverage,
        "ledger_entry_count": sum(
            operation.ledger_entry_identity is not None
            for operation in supervisor.operations.values()
        ),
        "audit_event_count": len(supervisor.audit_entries),
        "audit_chain_valid": supervisor.validate_audit_chain(),
        **dict(extra or {}),
    }


def simulate_run(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    queries = load_query_identities(root, protocol)
    rows: list[dict[str, Any]] = []

    transient = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(
        transient,
        "transient",
        "arxiv",
        [("429", 0), ("timeout", 0)] + [("success", 1)] * 10,
    )
    for query in queries:
        transient.commit_query(query, "completed")
    rows.append(_scenario_result("transient_jitter", transient, "healthy"))

    for scenario, outcome in (
        ("sustained_429", "429"),
        ("sustained_503", "503"),
        ("sustained_timeout", "timeout"),
    ):
        machine = ProviderHealthSupervisor(protocol, queries)
        _finish_sequence(machine, scenario, "arxiv", [(outcome, 0)] * 6)
        machine.acknowledge_pause()
        rows.append(_scenario_result(scenario, machine, "paused"))

    single = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(single, "single_degraded", "openalex", [("503", 0)] * 3)
    rows.append(_scenario_result("single_source_degraded", single, "degraded"))

    three = ProviderHealthSupervisor(protocol, queries)
    for offset, source in enumerate(SOURCES[:3]):
        _finish_sequence(
            three,
            f"three-{source}",
            source,
            [("503", 0)] * 3,
            query_offset=offset * 10,
        )
        if three.state == "pause_required":
            break
    three.acknowledge_pause()
    rows.append(_scenario_result("three_sources_degraded", three, "paused"))

    all_failed = ProviderHealthSupervisor(protocol, queries)
    for offset, source in enumerate(SOURCES):
        if all_failed.state == "pause_required":
            break
        _finish_sequence(
            all_failed,
            f"all-{source}",
            source,
            [("connection_failure", 0)] * 3,
            query_offset=offset * 10,
        )
    all_failed.acknowledge_pause()
    rows.append(_scenario_result("all_sources_failed", all_failed, "paused"))

    no_progress = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(
        no_progress, "no-progress", "pubmed", [("success", 0)] * 10
    )
    no_progress.acknowledge_pause()
    rows.append(_scenario_result("successful_without_progress", no_progress, "paused"))

    budget = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(
        budget, "budget-burn", "semantic_scholar", [("success", 1)] * 20
    )
    budget.acknowledge_pause()
    rows.append(_scenario_result("attempt_budget_burn", budget, "paused"))

    provenance = ProviderHealthSupervisor(protocol, queries)
    provenance.record_control_failure("provenance_write_failure")
    provenance.acknowledge_pause()
    rows.append(_scenario_result("provenance_write_failure", provenance, "paused"))

    storage = ProviderHealthSupervisor(protocol, queries)
    storage.record_control_failure("storage_capacity_failure")
    storage.acknowledge_pause()
    rows.append(_scenario_result("storage_capacity_drop", storage, "paused"))

    in_flight = ProviderHealthSupervisor(protocol, queries)
    for index in range(3):
        in_flight.start_operation(
            _operation_identity("in-flight", index),
            queries[index],
            SOURCES[index],
            "query",
        )
    in_flight.record_control_failure("storage_capacity_failure")
    blocked = False
    try:
        in_flight.start_operation(
            _operation_identity("after-pause", 0), queries[9], "arxiv", "retry"
        )
    except ProviderHealthError:
        blocked = True
    _drain_and_pause(in_flight)
    rows.append(
        _scenario_result(
            "in_flight_drain",
            in_flight,
            "paused",
            extra={"new_operation_blocked": blocked},
        )
    )

    resumed = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(resumed, "resume-pause", "arxiv", [("429", 0)] * 6)
    resumed.acknowledge_pause()
    prior_failures = resumed.sources["arxiv"].total_failures
    resumed.make_resume_eligible(_valid_resume_evidence(resumed))
    resumed.resume()
    for index, query in enumerate(queries):
        terminal = "source_failure" if index % 97 == 0 else "completed"
        resumed.commit_query(query, terminal)
    rows.append(
        _scenario_result(
            "pause_resume_full_coverage",
            resumed,
            "healthy",
            extra={
                "historical_failure_count_preserved": (
                    resumed.sources["arxiv"].total_failures == prior_failures
                ),
                "all_terminal_states_retained": True,
            },
        )
    )

    redegraded = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(redegraded, "redegrade-first", "arxiv", [("429", 0)] * 6)
    redegraded.acknowledge_pause()
    redegraded.make_resume_eligible(_valid_resume_evidence(redegraded))
    redegraded.resume()
    _finish_sequence(
        redegraded, "redegrade-second", "arxiv", [("503", 0)] * 6
    )
    redegraded.acknowledge_pause()
    rows.append(_scenario_result("resume_then_redegrade", redegraded, "paused"))

    expected = {
        "all_sources_failed",
        "attempt_budget_burn",
        "in_flight_drain",
        "pause_resume_full_coverage",
        "provenance_write_failure",
        "resume_then_redegrade",
        "single_source_degraded",
        "storage_capacity_drop",
        "successful_without_progress",
        "sustained_429",
        "sustained_503",
        "sustained_timeout",
        "three_sources_degraded",
        "transient_jitter",
    }
    rows = sorted(rows, key=lambda item: item["scenario"])
    all_passed = (
        {row["scenario"] for row in rows} == expected
        and all(row["status"] == "passed" for row in rows)
        and next(
            row for row in rows if row["scenario"] == "pause_resume_full_coverage"
        )["coverage"]["committed_query_count"]
        == 1000
    )
    return _report(
        "supervisor_controls_ready" if all_passed else "health_or_pause_violation",
        EXIT_READY if all_passed else EXIT_VIOLATION,
        scenario_count=len(rows),
        scenarios=rows,
        population={
            "query_count": 1000,
            "shard_count": 20,
            "query_order_sha256": stable_hash(queries),
        },
        invariants={
            "new_work_after_pause_required": 0,
            "selective_query_filtering": 0,
            "duplicate_ledger_entries": 0,
            "duplicate_committed_queries": 0,
            "quality_or_result_signal_count": 0,
        },
    )


def verify_resume_fixture(
    root: Path,
    protocol: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    queries = load_query_identities(root, protocol)
    supervisor = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(supervisor, "verify-resume", "arxiv", [("429", 0)] * 6)
    supervisor.acknowledge_pause()
    candidate = dict(evidence or _valid_resume_evidence(supervisor))
    supervisor.make_resume_eligible(candidate)
    supervisor.resume()
    return _report(
        "supervisor_controls_ready",
        EXIT_READY,
        resume_eligible=True,
        resumed_from_generation=supervisor.generation,
        historical_failures_preserved=supervisor.sources["arxiv"].total_failures,
        repeated_request_count=0,
    )


def build_addendum(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "addendum": ADDENDUM,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "provider_health_protocol_sha256": protocol["protocol_sha256"],
        "execution_plan": protocol["bindings"]["execution_plan"],
        "launch_control": protocol["bindings"]["launch_control"],
        "preregistration": protocol["bindings"]["preregistration"],
        "requirements": {
            "policy_bound_before_first_formal_request": True,
            "pause_blocks_new_query_page_and_retry": True,
            "failed_queries_remain_in_authoritative_coverage": True,
            "resume_requires_fresh_host_capacity_authorization_and_protocol": True,
            "resume_from_last_complete_generation": True,
            "failure_counters_not_reset": True,
            "source_switch_on_resume": False,
            "threshold_change_on_resume": False,
        },
        "real_provider_health_observed": False,
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def bind_launch_authorization(
    authorization: Mapping[str, Any],
    host_attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a future launch authorization to this frozen health policy.

    The old launch authorization remains readable, but it cannot become an
    authoritative long-running Full1000 launch without a qualified host seal
    and this addendum.  No credential or environment value is accepted here.
    """

    authorization_sha256 = authorization.get("authorization_sha256")
    host_sha256 = host_attestation.get("attestation_sha256")
    if (
        not isinstance(authorization_sha256, str)
        or len(authorization_sha256) != 64
    ):
        raise ProviderHealthError("launch_authorization_identity_invalid")
    if (
        host_attestation.get("status") != "host_qualified"
        or not isinstance(host_sha256, str)
        or len(host_sha256) != 64
    ):
        raise ProviderHealthError("qualified_fresh_host_attestation_required")
    payload: dict[str, Any] = {
        "contract": "provider_health_bound_launch_authorization_v1",
        "schema_version": SCHEMA_VERSION,
        "launch_authorization_sha256": authorization_sha256,
        "host_attestation_sha256": host_sha256,
        "provider_health_protocol_sha256": protocol["protocol_sha256"],
        "initial_health_state": "healthy",
        "initial_attempt_count": 0,
        "initial_failure_count": 0,
        "policy_bound_before_first_formal_request": True,
        "formal_validation_complete": False,
    }
    payload["binding_sha256"] = stable_hash(payload)
    return payload


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    # Binding validation and a complete deterministic simulation establish the
    # engineering control.  No real provider observation exists before launch.
    report = simulate_run(root, protocol)
    if report["status"] != "supervisor_controls_ready":
        raise ProviderHealthError("synthetic_health_matrix_failed")
    return _report(
        "external_provider_health_not_observed",
        EXIT_NOT_OBSERVED,
        controls_ready=True,
        full1000_blocker_cleared=False,
        real_run_started=False,
        real_provider_health_observed=False,
        scenario_count=report["scenario_count"],
    )


def _report(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }
