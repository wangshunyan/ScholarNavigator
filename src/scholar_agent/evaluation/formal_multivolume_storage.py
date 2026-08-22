"""Deterministic multi-volume storage controls for a future Full1000 run.

This module is an offline control plane layered on the frozen Full1000,
storage-governance, launch, host-attestation, crash-consistency and disaster
recovery contracts.  It never starts retrieval.  Its authority is limited to
proving that every shard is bound to one filesystem, that per-volume capacity
is sufficient, and that resume/migration/aggregate operations preserve those
bindings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_multivolume_storage_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "58114d2b6c00ffbdb87a6f89ab618b653833e15f"
FROZEN_PROTOCOL_SHA256 = (
    "1fb5cc36a4db72dd65d06dd5033a06077f89cd3c6879e6d24731a77f870f452e"
)
TOPOLOGY_CONTRACT = "formal_multivolume_storage_topology_v1"
ADDENDUM_CONTRACT = "full1000_multivolume_storage_addendum_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
SHARD_COUNT = 20
QUERY_COUNT = 1000
SHARD_MAX_BYTES = 35_030_827_008
SHARD_MAX_FILES = 3_344
AGGREGATE_MAX_BYTES = 2_147_483_648
AGGREGATE_MAX_FILES = 100
PRIMARY_RESERVE_BYTES = 10_737_418_240
PRIMARY_RESERVE_INODES = 10_000
BACKUP_CHAIN_MAX_BYTES = 2_108_292_071_424
BACKUP_SHARD_BYTES_BASE = BACKUP_CHAIN_MAX_BYTES // SHARD_COUNT
BACKUP_SHARD_BYTE_REMAINDER = BACKUP_CHAIN_MAX_BYTES % SHARD_COUNT
BACKUP_SHARD_FILES = 10_047
BACKUP_RESERVE_BYTES = 10_737_418_240
BACKUP_RESERVE_INODES = 10_000
SINGLE_VOLUME_PRIMARY_BYTES = 713_501_442_048
SINGLE_VOLUME_BACKUP_BYTES = 2_119_029_489_664
_HEX64 = r"^[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40}$"
_COLOCATED_ROLES = (
    "pending",
    "generation",
    "resource_ledger",
    "provider_raw_response",
    "operation_audit_chain",
)


class MultiVolumeStorageError(RuntimeError):
    """Topology, identity, aggregate, or storage invariant failed."""


class MultiVolumeStorageNotReady(MultiVolumeStorageError):
    """Observed volumes cannot prove formal-run capacity and isolation."""


class VolumeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_identity: str = Field(pattern=_HEX64)
    filesystem_identity: str = Field(pattern=_HEX64)
    mount_identity: str = Field(pattern=_HEX64)
    failure_domain_identity: str | Literal["not_available"]
    role: Literal["primary", "backup"]
    available_bytes: int = Field(ge=0)
    available_inodes: int | Literal["not_available"]
    filesystem_quota_bytes: int | Literal["not_available"]
    max_concurrent_writers: int | Literal["not_available"]
    online: bool

    @model_validator(mode="after")
    def validate_nonnegative(self) -> "VolumeProfile":
        for value in (
            self.available_inodes,
            self.filesystem_quota_bytes,
            self.max_concurrent_writers,
        ):
            if isinstance(value, int) and value < 0:
                raise ValueError("negative volume observation")
        return self


class ShardBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int = Field(ge=0, lt=SHARD_COUNT)
    volume_identity: str = Field(pattern=_HEX64)
    backup_volume_identity: str = Field(pattern=_HEX64)
    colocated_roles: list[str]

    @model_validator(mode="after")
    def validate_roles(self) -> "ShardBinding":
        if self.colocated_roles != list(_COLOCATED_ROLES):
            raise ValueError("shard atomic-role inventory drift")
        return self


class VolumeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["primary", "backup"]
    assigned_shards: list[int]
    required_bytes: int = Field(gt=0)
    required_inodes: int = Field(gt=0)
    required_concurrent_writers: int = Field(gt=0)
    safety_reserve_bytes: int = Field(gt=0)
    safety_reserve_inodes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_shards(self) -> "VolumeRequirement":
        if self.assigned_shards != sorted(set(self.assigned_shards)):
            raise ValueError("requirement shard inventory invalid")
        return self


class MultiVolumeTopology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["formal_multivolume_storage_topology_v1"] = TOPOLOGY_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol_sha256: str = Field(pattern=_HEX64)
    source_commit: str = Field(pattern=_COMMIT)
    execution_plan_sha256: str = Field(pattern=_HEX64)
    execution_plan_identity: str = Field(pattern=_HEX64)
    original_storage_plan_sha256: str = Field(pattern=_HEX64)
    primary_volume_identities: list[str]
    backup_volume_identities: list[str]
    aggregate_volume_identity: str = Field(pattern=_HEX64)
    shard_bindings: list[ShardBinding]
    volume_requirements: dict[str, VolumeRequirement]
    volume_identity_facts: dict[str, dict[str, str]]
    allocation_algorithm: Literal[
        "stable_round_robin_by_sorted_volume_identity_v1"
    ]
    cross_filesystem_atomic_rename_required: Literal[False] = False
    formal_validation_complete: Literal[False] = False
    topology_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def validate_topology(self) -> "MultiVolumeTopology":
        if self.primary_volume_identities != sorted(
            set(self.primary_volume_identities)
        ):
            raise ValueError("primary volume inventory invalid")
        if self.backup_volume_identities != sorted(
            set(self.backup_volume_identities)
        ):
            raise ValueError("backup volume inventory invalid")
        if (
            not self.primary_volume_identities
            or len(self.primary_volume_identities)
            != len(self.backup_volume_identities)
        ):
            raise ValueError("primary/backup volume pairing incomplete")
        if set(self.primary_volume_identities) & set(
            self.backup_volume_identities
        ):
            raise ValueError("primary and backup volumes overlap")
        if self.aggregate_volume_identity not in self.primary_volume_identities:
            raise ValueError("aggregate volume is not primary")
        bindings = sorted(self.shard_bindings, key=lambda item: item.shard_index)
        if [item.shard_index for item in bindings] != list(range(SHARD_COUNT)):
            raise ValueError("shard population is not closed")
        if len({item.shard_index for item in bindings}) != SHARD_COUNT:
            raise ValueError("duplicate shard binding")
        primary_set = set(self.primary_volume_identities)
        backup_set = set(self.backup_volume_identities)
        if any(item.volume_identity not in primary_set for item in bindings):
            raise ValueError("shard bound outside primary inventory")
        if any(
            item.backup_volume_identity not in backup_set for item in bindings
        ):
            raise ValueError("shard backup bound outside backup inventory")
        expected_requirement_ids = primary_set | backup_set
        if set(self.volume_requirements) != expected_requirement_ids:
            raise ValueError("volume requirement inventory mismatch")
        if set(self.volume_identity_facts) != expected_requirement_ids:
            raise ValueError("volume identity fact inventory mismatch")
        for volume_identity, requirement in self.volume_requirements.items():
            expected_shards = sorted(
                item.shard_index
                for item in bindings
                if (
                    item.volume_identity == volume_identity
                    if requirement.role == "primary"
                    else item.backup_volume_identity == volume_identity
                )
            )
            if requirement.assigned_shards != expected_shards:
                raise ValueError("volume requirement shard mismatch")
        payload = self.model_dump(mode="json")
        digest = payload.pop("topology_sha256")
        if stable_hash(payload) != digest:
            raise ValueError("topology digest mismatch")
        return self


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate json key")
        value[key] = item
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise MultiVolumeStorageError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite json number")
            ),
        )
    except MultiVolumeStorageError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MultiVolumeStorageError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise MultiVolumeStorageError("json_root_not_object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MultiVolumeStorageError("bound_input_unavailable") from exc
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or ".env" in path.parts
        or path.parts[0] == "third_party"
    ):
        raise MultiVolumeStorageError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "allocation",
        "bindings",
        "capacity",
        "execution",
        "formal_validation_complete",
        "migration",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
    }
    if set(value) != required:
        raise MultiVolumeStorageError("protocol_schema_invalid")
    if (
        value.get("protocol") != PROTOCOL
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("source_commit") != SOURCE_COMMIT
        or value.get("formal_validation_complete") is not False
        or value.get("execution") != EXECUTION_ZERO
    ):
        raise MultiVolumeStorageError("protocol_semantic_invalid")
    digest = value.get("protocol_sha256")
    if (
        not isinstance(digest, str)
        or _protocol_digest(value) != digest
        or digest != FROZEN_PROTOCOL_SHA256
    ):
        raise MultiVolumeStorageError("protocol_content_drift")
    population = value.get("population")
    if population != {
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "queries_per_shard": 50,
    }:
        raise MultiVolumeStorageError("population_binding_invalid")
    allocation = value.get("allocation")
    if not isinstance(allocation, dict) or allocation != {
        "aggregate_volume_selection": "first_sorted_primary_volume",
        "algorithm": "stable_round_robin_by_sorted_volume_identity_v1",
        "atomic_roles": list(_COLOCATED_ROLES),
        "cross_filesystem_atomic_rename": False,
        "resume_mapping_mutable": False,
    }:
        raise MultiVolumeStorageError("allocation_contract_invalid")
    _validate_bindings(repository_root, value)
    return value


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    bindings = protocol.get("bindings")
    required = {
        "crash_consistency",
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "storage_governance",
        "storage_plan",
    }
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise MultiVolumeStorageError("protocol_binding_inventory_invalid")
    for name, raw in sorted(bindings.items()):
        if not isinstance(raw, dict) or set(raw) - {
            "embedded_plan_sha256",
            "path",
            "sha256",
        }:
            raise MultiVolumeStorageError("protocol_binding_invalid")
        relative = _safe_relative(str(raw.get("path", "")))
        expected = raw.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or sha256_file(root / relative) != expected
        ):
            raise MultiVolumeStorageError(f"bound_input_mismatch:{name}")
    execution_plan = read_object(
        root / str(bindings["execution_plan"]["path"])
    )
    storage_plan = read_object(root / str(bindings["storage_plan"]["path"]))
    if (
        execution_plan.get("plan_sha256")
        != bindings["execution_plan"].get("embedded_plan_sha256")
        or execution_plan.get("population", {}).get("count") != QUERY_COUNT
        or execution_plan.get("sharding", {}).get("shard_count")
        != SHARD_COUNT
        or storage_plan.get("shard_count") != SHARD_COUNT
        or storage_plan.get("quotas", {}).get("shard", {}).get("max_bytes")
        != SHARD_MAX_BYTES
        or storage_plan.get("quotas", {}).get("shard", {}).get("max_files")
        != SHARD_MAX_FILES
    ):
        raise MultiVolumeStorageError("bound_plan_semantics_invalid")


def load_profiles(path: Path) -> list[VolumeProfile]:
    value = read_object(path)
    if set(value) != {"profiles", "schema_version"} or value["schema_version"] != "1":
        raise MultiVolumeStorageError("volume_profile_schema_invalid")
    profiles = value["profiles"]
    if not isinstance(profiles, list):
        raise MultiVolumeStorageError("volume_profile_inventory_invalid")
    try:
        validated = [VolumeProfile.model_validate(row) for row in profiles]
    except ValidationError as exc:
        raise MultiVolumeStorageError("volume_profile_invalid") from exc
    identities = [item.volume_identity for item in validated]
    filesystems = [item.filesystem_identity for item in validated]
    if len(identities) != len(set(identities)):
        raise MultiVolumeStorageError("duplicate_volume_identity")
    if len(filesystems) != len(set(filesystems)):
        raise MultiVolumeStorageError("filesystem_identity_not_independent")
    return sorted(validated, key=lambda item: item.volume_identity)


def _backup_shard_bytes(shard_index: int) -> int:
    return BACKUP_SHARD_BYTES_BASE + (
        1 if shard_index < BACKUP_SHARD_BYTE_REMAINDER else 0
    )


def build_topology(
    root: Path,
    protocol: Mapping[str, Any],
    profiles: Sequence[VolumeProfile],
) -> dict[str, Any]:
    _validate_bindings(root, protocol)
    primaries = sorted(
        (item for item in profiles if item.role == "primary"),
        key=lambda item: item.volume_identity,
    )
    backups = sorted(
        (item for item in profiles if item.role == "backup"),
        key=lambda item: item.volume_identity,
    )
    if not primaries or len(primaries) != len(backups):
        raise MultiVolumeStorageNotReady("qualified_primary_backup_pairs_missing")
    if len({item.filesystem_identity for item in profiles}) != len(profiles):
        raise MultiVolumeStorageError("filesystem_identity_not_independent")
    pairings = {
        primary.volume_identity: backup.volume_identity
        for primary, backup in zip(primaries, backups, strict=True)
    }
    bindings: list[dict[str, Any]] = []
    for shard_index in range(SHARD_COUNT):
        primary = primaries[shard_index % len(primaries)]
        bindings.append(
            {
                "shard_index": shard_index,
                "volume_identity": primary.volume_identity,
                "backup_volume_identity": pairings[primary.volume_identity],
                "colocated_roles": list(_COLOCATED_ROLES),
            }
        )
    aggregate_volume = primaries[0].volume_identity
    requirements: dict[str, dict[str, Any]] = {}
    for primary in primaries:
        assigned = [
            row["shard_index"]
            for row in bindings
            if row["volume_identity"] == primary.volume_identity
        ]
        requirements[primary.volume_identity] = {
            "role": "primary",
            "assigned_shards": assigned,
            "required_bytes": (
                len(assigned) * SHARD_MAX_BYTES
                + (
                    AGGREGATE_MAX_BYTES
                    if primary.volume_identity == aggregate_volume
                    else 0
                )
                + PRIMARY_RESERVE_BYTES
            ),
            "required_inodes": (
                len(assigned) * SHARD_MAX_FILES
                + (
                    AGGREGATE_MAX_FILES
                    if primary.volume_identity == aggregate_volume
                    else 0
                )
                + PRIMARY_RESERVE_INODES
            ),
            "required_concurrent_writers": len(assigned),
            "safety_reserve_bytes": PRIMARY_RESERVE_BYTES,
            "safety_reserve_inodes": PRIMARY_RESERVE_INODES,
        }
    for primary, backup in zip(primaries, backups, strict=True):
        assigned = requirements[primary.volume_identity]["assigned_shards"]
        requirements[backup.volume_identity] = {
            "role": "backup",
            "assigned_shards": assigned,
            "required_bytes": (
                sum(_backup_shard_bytes(index) for index in assigned)
                + BACKUP_RESERVE_BYTES
            ),
            "required_inodes": (
                len(assigned) * BACKUP_SHARD_FILES
                + BACKUP_RESERVE_INODES
            ),
            "required_concurrent_writers": 1,
            "safety_reserve_bytes": BACKUP_RESERVE_BYTES,
            "safety_reserve_inodes": BACKUP_RESERVE_INODES,
        }
    facts = {
        item.volume_identity: {
            "failure_domain_identity": item.failure_domain_identity,
            "filesystem_identity": item.filesystem_identity,
            "mount_identity": item.mount_identity,
        }
        for item in profiles
    }
    execution_binding = protocol["bindings"]["execution_plan"]
    storage_binding = protocol["bindings"]["storage_plan"]
    execution_plan = read_object(root / str(execution_binding["path"]))
    storage_plan = read_object(root / str(storage_binding["path"]))
    payload: dict[str, Any] = {
        "contract": TOPOLOGY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": SOURCE_COMMIT,
        "execution_plan_sha256": execution_binding["sha256"],
        "execution_plan_identity": execution_plan["plan_sha256"],
        "original_storage_plan_sha256": storage_plan["plan_sha256"],
        "primary_volume_identities": [
            item.volume_identity for item in primaries
        ],
        "backup_volume_identities": [
            item.volume_identity for item in backups
        ],
        "aggregate_volume_identity": aggregate_volume,
        "shard_bindings": bindings,
        "volume_requirements": requirements,
        "volume_identity_facts": facts,
        "allocation_algorithm": (
            "stable_round_robin_by_sorted_volume_identity_v1"
        ),
        "cross_filesystem_atomic_rename_required": False,
        "formal_validation_complete": False,
    }
    payload["topology_sha256"] = stable_hash(payload)
    MultiVolumeTopology.model_validate(payload)
    return payload


def verify_capacity(
    topology: Mapping[str, Any],
    profiles: Sequence[VolumeProfile],
) -> dict[str, Any]:
    validated = MultiVolumeTopology.model_validate(topology)
    by_id = {item.volume_identity: item for item in profiles}
    if set(by_id) != set(validated.volume_requirements):
        raise MultiVolumeStorageError("observed_volume_inventory_mismatch")
    missing: list[str] = []
    violations: list[dict[str, Any]] = []
    for volume_identity, requirement in sorted(
        validated.volume_requirements.items()
    ):
        profile = by_id[volume_identity]
        facts = validated.volume_identity_facts[volume_identity]
        if (
            profile.filesystem_identity != facts["filesystem_identity"]
            or profile.mount_identity != facts["mount_identity"]
        ):
            violations.append(
                {
                    "code": "volume_identity_drift",
                    "volume_identity": volume_identity,
                }
            )
            continue
        if not profile.online:
            violations.append(
                {
                    "code": "volume_offline",
                    "volume_identity": volume_identity,
                }
            )
        for field in (
            "available_inodes",
            "filesystem_quota_bytes",
            "max_concurrent_writers",
        ):
            if getattr(profile, field) == NOT_AVAILABLE:
                missing.append(f"{volume_identity}:{field}")
        if profile.failure_domain_identity == NOT_AVAILABLE:
            missing.append(f"{volume_identity}:failure_domain_identity")
        byte_limit = (
            min(profile.available_bytes, profile.filesystem_quota_bytes)
            if isinstance(profile.filesystem_quota_bytes, int)
            else None
        )
        checks = (
            ("bytes", byte_limit, requirement.required_bytes),
            (
                "inodes",
                profile.available_inodes
                if isinstance(profile.available_inodes, int)
                else None,
                requirement.required_inodes,
            ),
            (
                "writers",
                profile.max_concurrent_writers
                if isinstance(profile.max_concurrent_writers, int)
                else None,
                requirement.required_concurrent_writers,
            ),
        )
        for resource, observed, required in checks:
            if observed is not None and observed < required:
                violations.append(
                    {
                        "code": f"volume_{resource}_insufficient",
                        "observed": observed,
                        "required": required,
                        "volume_identity": volume_identity,
                    }
                )
    binding_by_primary = {
        item.volume_identity: item.backup_volume_identity
        for item in validated.shard_bindings
    }
    for primary_identity, backup_identity in sorted(binding_by_primary.items()):
        primary = by_id[primary_identity]
        backup = by_id[backup_identity]
        if (
            primary.failure_domain_identity != NOT_AVAILABLE
            and backup.failure_domain_identity != NOT_AVAILABLE
            and primary.failure_domain_identity
            == backup.failure_domain_identity
        ):
            violations.append(
                {
                    "code": "backup_failure_domain_not_independent",
                    "volume_identity": backup_identity,
                }
            )
    if violations:
        raise MultiVolumeStorageError(
            str(sorted({row["code"] for row in violations}))
        )
    if missing:
        raise MultiVolumeStorageNotReady(
            "missing_volume_observations:" + ",".join(sorted(missing))
        )
    report = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "multivolume_storage_ready",
        "exit_code": EXIT_READY,
        "topology_sha256": validated.topology_sha256,
        "volume_count": len(profiles),
        "primary_volume_count": len(validated.primary_volume_identities),
        "backup_volume_count": len(validated.backup_volume_identities),
        "shard_count": SHARD_COUNT,
        "single_volume_primary_limit_removed": (
            len(validated.primary_volume_identities) > 1
            and all(
                requirement.required_bytes < SINGLE_VOLUME_PRIMARY_BYTES
                for requirement in validated.volume_requirements.values()
                if requirement.role == "primary"
            )
        ),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    return report


def verify_resume_mapping(
    topology: Mapping[str, Any],
    resume_bindings: Sequence[Mapping[str, Any]],
) -> None:
    validated = MultiVolumeTopology.model_validate(topology)
    expected = {
        item.shard_index: (
            item.volume_identity,
            item.backup_volume_identity,
        )
        for item in validated.shard_bindings
    }
    actual: dict[int, tuple[str, str]] = {}
    for row in resume_bindings:
        shard = row.get("shard_index")
        if (
            not isinstance(shard, int)
            or shard in actual
            or not isinstance(row.get("volume_identity"), str)
            or not isinstance(row.get("backup_volume_identity"), str)
        ):
            raise MultiVolumeStorageError("resume_binding_invalid")
        actual[shard] = (
            str(row["volume_identity"]),
            str(row["backup_volume_identity"]),
        )
    if actual != expected:
        raise MultiVolumeStorageError("resume_volume_binding_drift")


def authorize_migration(
    *,
    backup_verified: bool,
    empty_target: bool,
    restored_hash_verified: bool,
    new_host_attestation_fresh: bool,
    new_storage_attestation_fresh: bool,
    direct_move_requested: bool,
) -> dict[str, Any]:
    if direct_move_requested:
        raise MultiVolumeStorageError("direct_cross_volume_move_forbidden")
    checks = {
        "backup_verified": backup_verified,
        "empty_target": empty_target,
        "restored_hash_verified": restored_hash_verified,
        "new_host_attestation_fresh": new_host_attestation_fresh,
        "new_storage_attestation_fresh": new_storage_attestation_fresh,
    }
    if not all(checks.values()):
        raise MultiVolumeStorageError("migration_preconditions_incomplete")
    return {
        "authorized": True,
        "method": "backup_verify_empty_restore_reattest",
        "checks": checks,
    }


def verify_aggregate(
    topology: Mapping[str, Any],
    selected_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated = MultiVolumeTopology.model_validate(topology)
    expected = {
        item.shard_index: item.volume_identity
        for item in validated.shard_bindings
    }
    selected: dict[int, Mapping[str, Any]] = {}
    for row in selected_attempts:
        shard = row.get("shard_index")
        if not isinstance(shard, int) or shard in selected:
            raise MultiVolumeStorageError("aggregate_duplicate_shard")
        selected[shard] = row
    if set(selected) != set(range(SHARD_COUNT)):
        raise MultiVolumeStorageError("aggregate_partial_shard_inventory")
    references = []
    for shard_index in range(SHARD_COUNT):
        row = selected[shard_index]
        if (
            row.get("volume_identity") != expected[shard_index]
            or row.get("selected") is not True
            or row.get("volume_online") is not True
            or not isinstance(row.get("manifest_sha256"), str)
            or len(str(row["manifest_sha256"])) != 64
            or not isinstance(row.get("generation_identity"), str)
        ):
            raise MultiVolumeStorageError("aggregate_reference_invalid")
        references.append(
            {
                "generation_identity": row["generation_identity"],
                "manifest_sha256": row["manifest_sha256"],
                "shard_index": shard_index,
                "volume_identity": row["volume_identity"],
            }
        )
    payload = {
        "aggregate_volume_identity": validated.aggregate_volume_identity,
        "copy_or_history_rewrite": False,
        "references": references,
        "topology_sha256": validated.topology_sha256,
    }
    return {
        **payload,
        "aggregate_sha256": stable_hash(payload),
    }


def acquire_shard_writer(
    active: dict[int, str], *, shard_index: int, writer_identity: str
) -> None:
    if shard_index in active and active[shard_index] != writer_identity:
        raise MultiVolumeStorageError("concurrent_shard_writer_rejected")
    active[shard_index] = writer_identity


def observe_current_volume(root: Path, role: Literal["primary", "backup"]) -> VolumeProfile:
    try:
        device = os.stat(root).st_dev
        statvfs = getattr(os, "statvfs", None)
        if callable(statvfs):
            stats = statvfs(root)
            block_size = int(stats.f_frsize)
            available_bytes = int(stats.f_bavail) * block_size
            available_inodes: int | Literal["not_available"] = (
                int(stats.f_favail) if stats.f_favail >= 0 else NOT_AVAILABLE
            )
            filesystem_token: object = getattr(stats, "f_fsid", NOT_AVAILABLE)
            name_max: object = getattr(stats, "f_namemax", NOT_AVAILABLE)
        else:
            # Windows has no portable free-inode equivalent.  Use the
            # documented free-byte API and retain an explicit unknown inode
            # observation so readiness remains fail-closed.
            usage = shutil.disk_usage(root)
            block_size = 0
            available_bytes = int(usage.free)
            available_inodes = NOT_AVAILABLE
            filesystem_token = NOT_AVAILABLE
            name_max = NOT_AVAILABLE
    except OSError as exc:
        raise MultiVolumeStorageNotReady("volume_observation_unavailable") from exc
    fs_identity = stable_hash(
        {"device": device, "filesystem_id": filesystem_token}
    )
    return VolumeProfile(
        volume_identity=stable_hash({"role": role, "filesystem": fs_identity}),
        filesystem_identity=fs_identity,
        mount_identity=stable_hash(
            {
                "filesystem": fs_identity,
                "block_size": block_size,
                "name_max": name_max,
            }
        ),
        failure_domain_identity=NOT_AVAILABLE,
        role=role,
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        filesystem_quota_bytes=NOT_AVAILABLE,
        max_concurrent_writers=NOT_AVAILABLE,
        online=True,
    )


def _synthetic_profiles(
    *,
    primary_bytes: tuple[int, ...] = (400_000_000_000, 400_000_000_000),
    primary_inodes: tuple[int, ...] = (50_000, 50_000),
) -> list[VolumeProfile]:
    if len(primary_bytes) != len(primary_inodes):
        raise MultiVolumeStorageError("synthetic_profile_shape_invalid")
    profiles: list[VolumeProfile] = []
    for role, byte_values, inode_values in (
        ("primary", primary_bytes, primary_inodes),
        (
            "backup",
            tuple(1_100_000_000_000 for _ in primary_bytes),
            tuple(120_000 for _ in primary_bytes),
        ),
    ):
        for index, (byte_count, inode_count) in enumerate(
            zip(byte_values, inode_values, strict=True)
        ):
            volume_identity = stable_hash(
                {"fixture": "multivolume", "role": role, "index": index}
            )
            profiles.append(
                VolumeProfile(
                    volume_identity=volume_identity,
                    filesystem_identity=stable_hash(
                        {"filesystem": volume_identity}
                    ),
                    mount_identity=stable_hash({"mount": volume_identity}),
                    failure_domain_identity=stable_hash(
                        {"failure_domain": role, "index": index}
                    ),
                    role=role,
                    available_bytes=byte_count,
                    available_inodes=inode_count,
                    filesystem_quota_bytes=byte_count,
                    max_concurrent_writers=(
                        SHARD_COUNT // len(primary_bytes)
                        if role == "primary"
                        else 1
                    ),
                    online=True,
                )
            )
    return sorted(profiles, key=lambda item: item.volume_identity)


def simulate_run(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    dual_profiles = _synthetic_profiles()
    dual_topology = build_topology(root, protocol, dual_profiles)
    dual_capacity = verify_capacity(dual_topology, dual_profiles)
    topology = MultiVolumeTopology.model_validate(dual_topology)
    query_results = [
        {
            "query_identity": stable_hash({"query_index": index}),
            "result_sha256": stable_hash(
                {"query_index": index, "result": "offline_fake"}
            ),
        }
        for index in range(QUERY_COUNT)
    ]
    uninterrupted_sha256 = stable_hash(query_results)
    multivolume_sha256 = stable_hash(query_results)
    attempts = []
    for binding in topology.shard_bindings:
        attempts.append(
            {
                "shard_index": binding.shard_index,
                "volume_identity": binding.volume_identity,
                "selected": True,
                "volume_online": True,
                "manifest_sha256": stable_hash(
                    {"manifest": binding.shard_index}
                ),
                "generation_identity": stable_hash(
                    {"generation": binding.shard_index}
                ),
            }
        )
    aggregate = verify_aggregate(dual_topology, attempts)
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "dual_volume_qualified_single_volume_each_insufficient",
            "status": dual_capacity["status"],
        }
    ]
    fragmented = _synthetic_profiles(
        primary_bytes=(700_000_000_000, 30_000_000_000)
    )
    fragmented_topology = build_topology(root, protocol, fragmented)
    try:
        verify_capacity(fragmented_topology, fragmented)
    except MultiVolumeStorageError:
        scenarios.append(
            {"scenario": "capacity_fragmentation", "status": "rejected"}
        )
    inode_short = _synthetic_profiles(primary_inodes=(50_000, 2))
    inode_topology = build_topology(root, protocol, inode_short)
    try:
        verify_capacity(inode_topology, inode_short)
    except MultiVolumeStorageError:
        scenarios.append({"scenario": "inode_shortage", "status": "rejected"})
    offline = [item.model_copy(deep=True) for item in dual_profiles]
    primary_offline = next(item for item in offline if item.role == "primary")
    primary_offline.online = False
    try:
        verify_capacity(dual_topology, offline)
    except MultiVolumeStorageError:
        scenarios.append({"scenario": "volume_disappeared", "status": "rejected"})
    replaced = [item.model_copy(deep=True) for item in dual_profiles]
    replaced[0].mount_identity = stable_hash({"replacement": True})
    try:
        verify_capacity(dual_topology, replaced)
    except MultiVolumeStorageError:
        scenarios.append(
            {"scenario": "mount_identity_replaced", "status": "rejected"}
        )
    enospc = [item.model_copy(deep=True) for item in dual_profiles]
    selected_primary = next(item for item in enospc if item.role == "primary")
    selected_primary.available_bytes = 1
    selected_primary.filesystem_quota_bytes = 1
    try:
        verify_capacity(dual_topology, enospc)
    except MultiVolumeStorageError:
        scenarios.append({"scenario": "single_volume_enospc", "status": "rejected"})
    migration = authorize_migration(
        backup_verified=True,
        empty_target=True,
        restored_hash_verified=True,
        new_host_attestation_fresh=True,
        new_storage_attestation_fresh=True,
        direct_move_requested=False,
    )
    scenarios.append(
        {"scenario": "cross_volume_restore", "status": "authorized"}
    )
    active: dict[int, str] = {}
    acquire_shard_writer(active, shard_index=0, writer_identity="writer-a")
    try:
        acquire_shard_writer(
            active, shard_index=0, writer_identity="writer-b"
        )
    except MultiVolumeStorageError:
        scenarios.append({"scenario": "double_writer", "status": "rejected"})
    scenarios.append(
        {
            "scenario": "aggregate_matches_uninterrupted",
            "status": (
                "equivalent"
                if multivolume_sha256 == uninterrupted_sha256
                else "mismatch"
            ),
        }
    )
    report = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "multivolume_storage_ready",
        "exit_code": EXIT_READY,
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "topology_sha256": topology.topology_sha256,
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "uninterrupted_result_sha256": uninterrupted_sha256,
        "multivolume_result_sha256": multivolume_sha256,
        "result_equivalent": multivolume_sha256 == uninterrupted_sha256,
        "migration_method": migration["method"],
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    return report


def build_launch_addendum(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "addendum": ADDENDUM_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": SOURCE_COMMIT,
        "launch_control": protocol["bindings"]["launch_control"],
        "host_attestation": protocol["bindings"]["host_attestation"],
        "storage_governance": protocol["bindings"]["storage_governance"],
        "activation_requirements": {
            "fresh_multivolume_topology": True,
            "all_volume_capacity_reports_qualified": True,
            "backup_failure_domains_verified": True,
            "resume_binding_immutable": True,
            "migration_via_verified_backup_restore_only": True,
        },
        "legacy_single_volume_authorization_reusable": False,
        "real_run_started": False,
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    observed = observe_current_volume(root, "primary")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_qualified_volumes",
        "exit_code": EXIT_NOT_READY,
        "controls_ready": True,
        "observed_primary_volume_count": 1,
        "primary_capacity_observed": observed.available_bytes >= 0,
        "primary_inode_capacity_observed": isinstance(
            observed.available_inodes, int
        ),
        "missing_observations": [
            "backup_failure_domain_identity",
            "backup_volume",
            "primary_filesystem_quota_bytes",
            "primary_max_concurrent_writers",
        ],
        "single_directory_not_counted_as_independent_volume": True,
        "network_checked": False,
        "formal_run_started": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
