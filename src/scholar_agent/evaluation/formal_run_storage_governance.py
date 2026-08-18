"""Deterministic storage governance for a future Full1000 formal run.

The module is an offline control plane.  It binds the frozen Full1000 plan to
explicit byte/file/inode quotas, models reserve/commit/release accounting, and
checks retention eligibility without touching a real Snapshot or provider.
Actual filesystem capacity is observational input and is never inferred from
compression, sparse allocation, or hoped-for cleanup.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_run_storage_governance_v1"
SCHEMA_VERSION = "1"
FROZEN_PROTOCOL_SHA256 = (
    "367950b348220a7c52d779611fb10d6da2211884a4e533c74b2befaa955d7bda"
)
PLAN_CONTRACT = "formal_run_storage_plan_v1"
ADDENDUM_CONTRACT = "full1000_storage_governance_addendum_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
NOT_AVAILABLE = "not_available"
_HEX64 = r"^[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40}$"


class StorageGovernanceError(RuntimeError):
    """A quota, retention, protocol, or storage-accounting invariant failed."""


class StorageCapacityNotReady(StorageGovernanceError):
    """Real primary/backup capacity is not proven for formal activation."""


class QuotaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(gt=0)
    max_files: int = Field(gt=0)


class StoragePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["formal_run_storage_plan_v1"] = PLAN_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol_sha256: str = Field(pattern=_HEX64)
    source_commit: str = Field(pattern=_COMMIT)
    execution_plan_sha256: str = Field(pattern=_HEX64)
    execution_plan_identity: str = Field(pattern=_HEX64)
    launch_control_sha256: str = Field(pattern=_HEX64)
    capture_addendum_sha256: str = Field(pattern=_HEX64)
    disaster_recovery_sha256: str = Field(pattern=_HEX64)
    query_count: Literal[1000] = 1000
    shard_count: Literal[20] = 20
    max_http_attempts: Literal[19280] = 19280
    selected_generation_upper: Literal[1040] = 1040
    quotas: dict[str, QuotaSpec]
    preflight: dict[str, Any]
    retention: dict[str, Any]
    unknown_capacity_fields: list[str]
    plan_sha256: str = Field(pattern=_HEX64)
    formal_validation_complete: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "StoragePlan":
        required = {
            "aggregate",
            "backup_chain",
            "committed_generation",
            "provider_response",
            "run",
            "shard",
        }
        if set(self.quotas) != required:
            raise ValueError("storage quota inventory mismatch")
        if self.unknown_capacity_fields != sorted(set(self.unknown_capacity_fields)):
            raise ValueError("unknown capacity fields must be sorted and unique")
        payload = self.model_dump(mode="json")
        digest = payload.pop("plan_sha256")
        if stable_hash(payload) != digest:
            raise ValueError("storage plan digest mismatch")
        return self


class CapacityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_bytes: int = Field(ge=0)
    available_inodes: int | Literal["not_available"]
    filesystem_quota_bytes: int | Literal["not_available"]
    target_kind: Literal["primary", "backup"]
    sparse_or_compression_credit_bytes: Literal[0] = 0
    future_cleanup_credit_bytes: Literal[0] = 0

    @model_validator(mode="after")
    def validate_values(self) -> "CapacityObservation":
        if isinstance(self.available_inodes, int) and self.available_inodes < 0:
            raise ValueError("available inodes cannot be negative")
        if isinstance(self.filesystem_quota_bytes, int) and self.filesystem_quota_bytes < 0:
            raise ValueError("filesystem quota cannot be negative")
        return self


class StorageTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_identity: str = Field(pattern=_HEX64)
    writer_identity: str = Field(pattern=_HEX64)
    artifact_type: str
    requested_bytes: int = Field(ge=0)
    requested_files: int = Field(ge=0)
    state: Literal["reserved", "committed", "released", "rejected"]
    reason_code: str | None = None


class StorageLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["formal_run_storage_ledger_v1"] = "formal_run_storage_ledger_v1"
    schema_version: Literal["1"] = SCHEMA_VERSION
    capacity_bytes: int = Field(ge=0)
    capacity_inodes: int = Field(ge=0)
    quota_bytes: int = Field(ge=0)
    committed_bytes: int = Field(ge=0)
    committed_files: int = Field(ge=0)
    reserved_bytes: int = Field(ge=0)
    reserved_files: int = Field(ge=0)
    transactions: list[StorageTransaction]
    ledger_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def validate_ledger(self) -> "StorageLedger":
        if self.committed_bytes + self.reserved_bytes > min(
            self.capacity_bytes, self.quota_bytes
        ):
            raise ValueError("byte conservation violated")
        if self.committed_files + self.reserved_files > self.capacity_inodes:
            raise ValueError("inode conservation violated")
        identities = [item.operation_identity for item in self.transactions]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate storage operation")
        payload = self.model_dump(mode="json")
        digest = payload.pop("ledger_sha256")
        if stable_hash(payload) != digest:
            raise ValueError("storage ledger digest mismatch")
        return self


@dataclass(frozen=True)
class CleanupCandidate:
    """Facts needed to decide whether one path may be removed."""

    relative_path: str
    kind: Literal["temporary", "generation", "other"]
    committed: bool
    backup_verified: bool
    generations_older_than_window: int
    is_current_resume: bool = False
    is_authoritative_ledger: bool = False
    is_raw_response: bool = False
    is_audit_chain: bool = False
    is_final_attempt: bool = False
    is_transparency_referenced: bool = False


class InjectedCapacityLedger:
    """Deterministic filesystem-capacity seam used by the pressure matrix."""

    def __init__(self, *, capacity_bytes: int, capacity_inodes: int, quota_bytes: int):
        if min(capacity_bytes, capacity_inodes, quota_bytes) < 0:
            raise StorageGovernanceError("negative_capacity")
        self.capacity_bytes = capacity_bytes
        self.capacity_inodes = capacity_inodes
        self.quota_bytes = quota_bytes
        self.committed_bytes = 0
        self.committed_files = 0
        self.reservations: dict[str, tuple[int, int, str, str]] = {}
        self.transactions: list[StorageTransaction] = []
        self.active_writer: str | None = None

    def acquire_writer(self, writer_identity: str) -> None:
        if self.active_writer not in {None, writer_identity}:
            raise StorageGovernanceError("concurrent_writer_rejected")
        self.active_writer = writer_identity

    def release_writer(self, writer_identity: str) -> None:
        if self.active_writer != writer_identity:
            raise StorageGovernanceError("writer_identity_mismatch")
        self.active_writer = None

    def reserve(
        self,
        *,
        operation_identity: str,
        writer_identity: str,
        artifact_type: str,
        requested_bytes: int,
        requested_files: int,
    ) -> None:
        if self.active_writer != writer_identity:
            raise StorageGovernanceError("writer_not_authorized")
        if operation_identity in self.reservations or any(
            row.operation_identity == operation_identity for row in self.transactions
        ):
            raise StorageGovernanceError("duplicate_storage_operation")
        if requested_bytes < 0 or requested_files < 0:
            raise StorageGovernanceError("negative_reservation")
        reserved_bytes = sum(value[0] for value in self.reservations.values())
        reserved_files = sum(value[1] for value in self.reservations.values())
        if (
            self.committed_bytes + reserved_bytes + requested_bytes
            > min(self.capacity_bytes, self.quota_bytes)
        ):
            raise StorageGovernanceError("storage_bytes_unavailable")
        if (
            self.committed_files + reserved_files + requested_files
            > self.capacity_inodes
        ):
            raise StorageGovernanceError("storage_inodes_unavailable")
        self.reservations[operation_identity] = (
            requested_bytes,
            requested_files,
            writer_identity,
            artifact_type,
        )
        self.transactions.append(
            StorageTransaction(
                operation_identity=operation_identity,
                writer_identity=writer_identity,
                artifact_type=artifact_type,
                requested_bytes=requested_bytes,
                requested_files=requested_files,
                state="reserved",
            )
        )

    def commit(self, operation_identity: str) -> None:
        try:
            byte_count, file_count, writer, artifact_type = self.reservations[
                operation_identity
            ]
        except KeyError as exc:
            raise StorageGovernanceError("reservation_missing") from exc
        # Capacity may drop after reservation.  Re-check before exposing a new
        # generation and preserve all prior committed bytes on failure.
        reserved_other_bytes = sum(
            value[0]
            for key, value in self.reservations.items()
            if key != operation_identity
        )
        reserved_other_files = sum(
            value[1]
            for key, value in self.reservations.items()
            if key != operation_identity
        )
        if (
            self.committed_bytes + reserved_other_bytes + byte_count
            > min(self.capacity_bytes, self.quota_bytes)
        ):
            self.release(operation_identity, reason_code="capacity_dropped_before_commit")
            raise StorageGovernanceError("capacity_dropped_before_commit")
        if (
            self.committed_files + reserved_other_files + file_count
            > self.capacity_inodes
        ):
            self.release(operation_identity, reason_code="inodes_dropped_before_commit")
            raise StorageGovernanceError("inodes_dropped_before_commit")
        del self.reservations[operation_identity]
        self.committed_bytes += byte_count
        self.committed_files += file_count
        self.transactions.append(
            StorageTransaction(
                operation_identity=_phase_identity(operation_identity, "commit"),
                writer_identity=writer,
                artifact_type=artifact_type,
                requested_bytes=byte_count,
                requested_files=file_count,
                state="committed",
            )
        )

    def release(self, operation_identity: str, *, reason_code: str = "released") -> None:
        try:
            byte_count, file_count, writer, artifact_type = self.reservations.pop(
                operation_identity
            )
        except KeyError as exc:
            raise StorageGovernanceError("reservation_missing") from exc
        self.transactions.append(
            StorageTransaction(
                operation_identity=_phase_identity(operation_identity, "release"),
                writer_identity=writer,
                artifact_type=artifact_type,
                requested_bytes=byte_count,
                requested_files=file_count,
                state="released",
                reason_code=reason_code,
            )
        )

    def shrink(self, *, capacity_bytes: int | None = None, capacity_inodes: int | None = None) -> None:
        if capacity_bytes is not None:
            self.capacity_bytes = capacity_bytes
        if capacity_inodes is not None:
            self.capacity_inodes = capacity_inodes

    def snapshot(self) -> StorageLedger:
        reserved_bytes = sum(value[0] for value in self.reservations.values())
        reserved_files = sum(value[1] for value in self.reservations.values())
        payload: dict[str, Any] = {
            "contract": "formal_run_storage_ledger_v1",
            "schema_version": SCHEMA_VERSION,
            "capacity_bytes": self.capacity_bytes,
            "capacity_inodes": self.capacity_inodes,
            "quota_bytes": self.quota_bytes,
            "committed_bytes": self.committed_bytes,
            "committed_files": self.committed_files,
            "reserved_bytes": reserved_bytes,
            "reserved_files": reserved_files,
            "transactions": [
                item.model_dump(mode="json") for item in self.transactions
            ],
        }
        payload["ledger_sha256"] = stable_hash(payload)
        return StorageLedger.model_validate(payload)


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StorageGovernanceError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise StorageGovernanceError("json_input_not_object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate json key")
        value[key] = item
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
    ):
        raise StorageGovernanceError("unsafe_relative_path")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = _read_object(path)
    required = {
        "bindings",
        "capacity",
        "capture_limit",
        "execution",
        "formal_validation_complete",
        "population",
        "protocol",
        "protocol_sha256",
        "quotas",
        "retention",
        "schema_version",
        "source_commit",
    }
    if set(value) != required:
        raise StorageGovernanceError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise StorageGovernanceError("protocol_version_invalid")
    if value["formal_validation_complete"] is not False:
        raise StorageGovernanceError("formal_validation_state_invalid")
    if value["protocol_sha256"] != _protocol_digest(value):
        raise StorageGovernanceError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise StorageGovernanceError("protocol_content_drift")
    population = value.get("population")
    if not isinstance(population, dict) or population != {
        "http_attempt_upper": 19280,
        "query_count": 1000,
        "selected_generation_upper": 1040,
        "shard_count": 20,
    }:
        raise StorageGovernanceError("population_binding_invalid")
    _validate_bindings(repository_root, value)
    return value


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise StorageGovernanceError("protocol_bindings_invalid")
    for name, raw in sorted(bindings.items()):
        if not isinstance(raw, dict):
            raise StorageGovernanceError("protocol_binding_invalid")
        relative = _safe_relative(str(raw.get("path", "")))
        expected = raw.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise StorageGovernanceError("protocol_binding_digest_invalid")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StorageGovernanceError(f"bound_input_mismatch:{name}")
    plan_binding = bindings["execution_plan"]
    plan = _read_object(root / str(plan_binding["path"]))
    resources = plan.get("resource_upper_bounds")
    if not isinstance(resources, dict) or (
        resources.get("http_request_attempt_upper") != 19280
        or resources.get("checkpoint_generation_selected_attempt_upper") != 1040
        or plan.get("population", {}).get("count") != 1000
        or plan.get("sharding", {}).get("shard_count") != 20
    ):
        raise StorageGovernanceError("execution_plan_resource_binding_invalid")


def build_storage_plan(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_bindings(root, protocol)
    bindings = protocol["bindings"]
    quotas = protocol["quotas"]
    plan_binding = bindings["execution_plan"]
    execution_plan = _read_object(root / str(plan_binding["path"]))
    payload: dict[str, Any] = {
        "contract": PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "execution_plan_sha256": plan_binding["sha256"],
        "execution_plan_identity": execution_plan["plan_sha256"],
        "launch_control_sha256": bindings["launch_control"]["sha256"],
        "capture_addendum_sha256": bindings["provider_capture_addendum"]["sha256"],
        "disaster_recovery_sha256": bindings["disaster_recovery"]["sha256"],
        "query_count": 1000,
        "shard_count": 20,
        "max_http_attempts": 19280,
        "selected_generation_upper": 1040,
        "quotas": quotas,
        "preflight": protocol["capacity"],
        "retention": protocol["retention"],
        "unknown_capacity_fields": sorted(
            str(item) for item in protocol["capacity"]["required_observations"]
        ),
        "formal_validation_complete": False,
    }
    payload["plan_sha256"] = stable_hash(payload)
    StoragePlan.model_validate(payload)
    return payload


def build_launch_addendum(
    root: Path, protocol: Mapping[str, Any], storage_plan: Mapping[str, Any]
) -> dict[str, Any]:
    StoragePlan.model_validate(storage_plan)
    payload: dict[str, Any] = {
        "addendum": ADDENDUM_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "launch_control": protocol["bindings"]["launch_control"],
        "storage_protocol_sha256": protocol["protocol_sha256"],
        "storage_plan_sha256": storage_plan["plan_sha256"],
        "activation_requirements": {
            "capture_enabled_explicitly": True,
            "empty_authoritative_output_root": True,
            "primary_and_backup_capacity_verified": True,
            "reserve_commit_release_ledger_enabled": True,
        },
        "legacy_launch_authorization_reusable": False,
        "mutation_policy": "does_not_modify_bound_launch_protocol_or_execution_plan",
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def observe_capacity(path: Path, target_kind: Literal["primary", "backup"]) -> CapacityObservation:
    try:
        stats = os.statvfs(path)
    except OSError as exc:
        raise StorageCapacityNotReady("capacity_target_unavailable") from exc
    available_inodes: int | Literal["not_available"] = (
        int(stats.f_favail) if stats.f_favail >= 0 else NOT_AVAILABLE
    )
    return CapacityObservation(
        available_bytes=int(stats.f_bavail) * int(stats.f_frsize),
        available_inodes=available_inodes,
        filesystem_quota_bytes=NOT_AVAILABLE,
        target_kind=target_kind,
    )


def verify_capacity(
    storage_plan: Mapping[str, Any],
    *,
    primary: CapacityObservation,
    backup: CapacityObservation,
) -> dict[str, Any]:
    plan = StoragePlan.model_validate(storage_plan)
    requirements = plan.preflight["requirements"]
    missing: list[str] = []
    violations: list[dict[str, Any]] = []
    observations = {"primary": primary, "backup": backup}
    for target, item in observations.items():
        required = requirements[target]
        if item.filesystem_quota_bytes == NOT_AVAILABLE:
            missing.append(f"{target}.filesystem_quota_bytes")
        elif item.filesystem_quota_bytes < required["bytes"]:
            violations.append(_violation("filesystem_quota_below_required", target))
        if item.available_bytes < required["bytes"]:
            violations.append(_violation("available_bytes_below_required", target))
        if item.available_inodes == NOT_AVAILABLE:
            missing.append(f"{target}.available_inodes")
        elif item.available_inodes < required["inodes"]:
            violations.append(_violation("available_inodes_below_required", target))
    if violations:
        return _report(
            "quota_or_retention_violation",
            EXIT_VIOLATION,
            violations=violations,
            missing_capacity_fields=sorted(missing),
        )
    if missing:
        return _report(
            "not_ready_capacity_unverified",
            EXIT_NOT_READY,
            missing_capacity_fields=sorted(missing),
            violations=[],
        )
    return _report(
        "storage_controls_ready",
        EXIT_READY,
        missing_capacity_fields=[],
        violations=[],
    )


def cleanup_eligibility(candidate: CleanupCandidate, *, retention_window: int) -> tuple[bool, str]:
    _safe_relative(candidate.relative_path)
    protected = {
        "current_resume": candidate.is_current_resume,
        "authoritative_ledger": candidate.is_authoritative_ledger,
        "raw_response": candidate.is_raw_response,
        "audit_chain": candidate.is_audit_chain,
        "final_attempt": candidate.is_final_attempt,
        "transparency_reference": candidate.is_transparency_referenced,
    }
    if any(protected.values()):
        return False, "protected_authoritative_artifact"
    if candidate.kind == "temporary":
        return (not candidate.committed, "uncommitted_temporary_only")
    if candidate.kind == "generation":
        if not candidate.committed:
            return True, "uncommitted_generation"
        if not candidate.backup_verified:
            return False, "backup_not_verified"
        if candidate.generations_older_than_window <= retention_window:
            return False, "inside_retention_window"
        return True, "verified_backup_outside_retention_window"
    return False, "artifact_type_not_cleanup_eligible"


def enforce_capture_limit(raw_bytes: bytes | None, *, enabled: bool, max_bytes: int) -> dict[str, Any]:
    """Decide whether exact raw bytes may enter parser/capture.

    Oversized bytes are never truncated and never passed to the parser.
    """

    if max_bytes <= 0:
        raise StorageGovernanceError("capture_limit_invalid")
    if not enabled:
        return {
            "capture_enabled": False,
            "accepted": raw_bytes is None,
            "attempt_terminal": raw_bytes is not None,
            "reason_code": "capture_not_enabled" if raw_bytes is not None else None,
            "captured_bytes": 0,
        }
    if raw_bytes is not None and len(raw_bytes) > max_bytes:
        return {
            "capture_enabled": True,
            "accepted": False,
            "attempt_terminal": True,
            "reason_code": "capture_size_exceeded",
            "captured_bytes": 0,
        }
    return {
        "capture_enabled": True,
        "accepted": True,
        "attempt_terminal": False,
        "reason_code": None,
        "captured_bytes": 0 if raw_bytes is None else len(raw_bytes),
    }


def simulate_pressure(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    plan = build_storage_plan(root, protocol)
    generation = plan["quotas"]["committed_generation"]
    writer = stable_hash({"writer": "fixture"})
    scenarios: dict[str, dict[str, Any]] = {}

    def identity(name: str) -> str:
        return stable_hash({"operation": name})

    # The logical 1000-query exercise uses small deterministic records; quotas
    # are validated independently against the frozen 19,280/1,040 maxima.
    normal = InjectedCapacityLedger(
        capacity_bytes=2_000_000,
        capacity_inodes=2_000,
        quota_bytes=2_000_000,
    )
    normal.acquire_writer(writer)
    for query_index in range(1000):
        operation = identity(f"query-{query_index}")
        normal.reserve(
            operation_identity=operation,
            writer_identity=writer,
            artifact_type="synthetic_query_generation",
            requested_bytes=1000,
            requested_files=1,
        )
        normal.commit(operation)
    normal.release_writer(writer)
    normal_snapshot = normal.snapshot()
    scenarios["normal_capacity"] = {
        "status": "passed",
        "query_count": 1000,
        "committed_bytes": normal_snapshot.committed_bytes,
        "committed_files": normal_snapshot.committed_files,
    }

    exact = InjectedCapacityLedger(
        capacity_bytes=generation["max_bytes"],
        capacity_inodes=generation["max_files"],
        quota_bytes=generation["max_bytes"],
    )
    exact.acquire_writer(writer)
    exact.reserve(
        operation_identity=identity("exact"),
        writer_identity=writer,
        artifact_type="committed_generation",
        requested_bytes=generation["max_bytes"],
        requested_files=generation["max_files"],
    )
    exact.commit(identity("exact"))
    scenarios["exact_boundary"] = {"status": "passed", "remaining_bytes": 0}

    scenarios["enospc"] = _rejection_scenario(
        capacity_bytes=99,
        capacity_inodes=2,
        requested_bytes=100,
        requested_files=1,
        expected="storage_bytes_unavailable",
    )
    scenarios["inode_exhaustion"] = _rejection_scenario(
        capacity_bytes=100,
        capacity_inodes=0,
        requested_bytes=1,
        requested_files=1,
        expected="storage_inodes_unavailable",
    )

    drop = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=2,
        quota_bytes=100,
    )
    drop.acquire_writer(writer)
    drop.reserve(
        operation_identity=identity("drop"),
        writer_identity=writer,
        artifact_type="committed_generation",
        requested_bytes=80,
        requested_files=1,
    )
    drop.shrink(capacity_bytes=50)
    before = drop.committed_bytes
    drop_reason = _expect_error(lambda: drop.commit(identity("drop")))
    scenarios["capacity_drop"] = {
        "status": "passed" if drop_reason == "capacity_dropped_before_commit" else "failed",
        "reason_code": drop_reason,
        "previous_generation_preserved": drop.committed_bytes == before,
    }

    limit = int(protocol["capture_limit"]["max_response_bytes"])
    capture = enforce_capture_limit(b"x" * (limit + 1), enabled=True, max_bytes=limit)
    scenarios["oversized_response"] = {
        "status": "passed"
        if capture["reason_code"] == "capture_size_exceeded"
        and capture["captured_bytes"] == 0
        else "failed",
        **capture,
    }

    backup_required = plan["preflight"]["requirements"]["backup"]["bytes"]
    backup_observation = CapacityObservation(
        available_bytes=backup_required - 1,
        available_inodes=plan["preflight"]["requirements"]["backup"]["inodes"],
        filesystem_quota_bytes=backup_required,
        target_kind="backup",
    )
    primary_required = plan["preflight"]["requirements"]["primary"]
    primary_observation = CapacityObservation(
        available_bytes=primary_required["bytes"],
        available_inodes=primary_required["inodes"],
        filesystem_quota_bytes=primary_required["bytes"],
        target_kind="primary",
    )
    backup_report = verify_capacity(
        plan, primary=primary_observation, backup=backup_observation
    )
    scenarios["backup_space_insufficient"] = {
        "status": "passed" if backup_report["exit_code"] == EXIT_VIOLATION else "failed",
        "reason_code": backup_report["violations"][0]["invariant"],
    }

    unsafe, reason = cleanup_eligibility(
        CleanupCandidate(
            relative_path="shards/0/raw/response.bin",
            kind="generation",
            committed=True,
            backup_verified=True,
            generations_older_than_window=99,
            is_raw_response=True,
        ),
        retention_window=int(plan["retention"]["superseded_generation_window"]),
    )
    scenarios["unsafe_cleanup"] = {
        "status": "passed" if not unsafe else "failed",
        "reason_code": reason,
    }

    writer_gate = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=2,
        quota_bytes=100,
    )
    writer_gate.acquire_writer(writer)
    second_writer = stable_hash({"writer": "other"})
    dual_reason = _expect_error(lambda: writer_gate.acquire_writer(second_writer))
    scenarios["dual_writer"] = {
        "status": "passed" if dual_reason == "concurrent_writer_rejected" else "failed",
        "reason_code": dual_reason,
    }

    expected = {
        "backup_space_insufficient",
        "capacity_drop",
        "dual_writer",
        "enospc",
        "exact_boundary",
        "inode_exhaustion",
        "normal_capacity",
        "oversized_response",
        "unsafe_cleanup",
    }
    all_passed = set(scenarios) == expected and all(
        item["status"] == "passed" for item in scenarios.values()
    )
    return _report(
        "storage_controls_ready" if all_passed else "quota_or_retention_violation",
        EXIT_READY if all_passed else EXIT_VIOLATION,
        scenario_count=len(scenarios),
        scenarios={key: scenarios[key] for key in sorted(scenarios)},
        closure={
            "query_count": 1000,
            "http_attempt_upper": 19280,
            "selected_generation_upper": 1040,
            "shard_count": 20,
        },
        storage_plan_sha256=plan["plan_sha256"],
    )


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    plan = build_storage_plan(root, protocol)
    return _report(
        "not_ready_capacity_unverified",
        EXIT_NOT_READY,
        missing_capacity_fields=plan["unknown_capacity_fields"],
        real_run_started=False,
        real_backup_created=False,
        storage_plan_sha256=plan["plan_sha256"],
        full1000_blocker_cleared=False,
    )


def _rejection_scenario(
    *,
    capacity_bytes: int,
    capacity_inodes: int,
    requested_bytes: int,
    requested_files: int,
    expected: str,
) -> dict[str, Any]:
    ledger = InjectedCapacityLedger(
        capacity_bytes=capacity_bytes,
        capacity_inodes=capacity_inodes,
        quota_bytes=capacity_bytes,
    )
    writer = stable_hash({"writer": "fixture"})
    ledger.acquire_writer(writer)
    before = (ledger.committed_bytes, ledger.committed_files)
    reason = _expect_error(
        lambda: ledger.reserve(
            operation_identity=stable_hash({"operation": expected}),
            writer_identity=writer,
            artifact_type="committed_generation",
            requested_bytes=requested_bytes,
            requested_files=requested_files,
        )
    )
    return {
        "status": "passed" if reason == expected else "failed",
        "reason_code": reason,
        "previous_generation_preserved": before
        == (ledger.committed_bytes, ledger.committed_files),
    }


def _expect_error(call: Any) -> str:
    try:
        call()
    except StorageGovernanceError as exc:
        return str(exc)
    return "no_error"


def _phase_identity(operation_identity: str, phase: str) -> str:
    return stable_hash({"operation_identity": operation_identity, "phase": phase})


def _violation(invariant: str, target: str) -> dict[str, Any]:
    return {"invariant": invariant, "target": target}


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


def run_deterministic_artifact_generation(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build plan/addendum plus simulation/readiness without real I/O pressure."""

    plan = build_storage_plan(root, protocol)
    addendum = build_launch_addendum(root, protocol, plan)
    with tempfile.TemporaryDirectory(prefix="formal-storage-simulation-") as temporary:
        simulation = simulate_pressure(Path(root), protocol)
        # The injected model writes no authority data; the temporary directory
        # exists to make cleanup expectations explicit and testable.
        if any(Path(temporary).iterdir()):
            raise StorageGovernanceError("simulation_left_temporary_files")
    readiness = audit_readiness(root, protocol)
    return plan, addendum, simulation, readiness
