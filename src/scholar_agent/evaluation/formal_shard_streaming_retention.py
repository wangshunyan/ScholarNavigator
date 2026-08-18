"""Offline Full1000 shard streaming-retention and local eviction controls.

This is a storage control plane layered on the frozen Full1000, crash,
multi-volume, disaster-recovery, provenance, accounting, aggregate and launch
contracts.  It never runs retrieval.  Synthetic payloads model complete shard
authority sets while preserving the production byte/file quota upper bounds.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_multivolume_storage import (
    AGGREGATE_MAX_BYTES,
    AGGREGATE_MAX_FILES,
    PRIMARY_RESERVE_BYTES,
    PRIMARY_RESERVE_INODES,
    QUERY_COUNT,
    SHARD_COUNT,
    SHARD_MAX_BYTES,
    SHARD_MAX_FILES,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_shard_streaming_retention_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "d9c3bc7863fe071fb4285b2b4a90f334cd8bf391"
FROZEN_PROTOCOL_SHA256 = (
    "9fe9d56b2277bf5aa9f29820c8407db0a9f8382e5d6cba6c25d67d7975492780"
)
ADDENDUM_CONTRACT = "full1000_shard_streaming_retention_addendum_v1"
ARCHIVE_CONTRACT = "formal_shard_archive_v1"
RECEIPT_CONTRACT = "formal_shard_eviction_receipt_v1"
AGGREGATE_CONTRACT = "formal_streaming_aggregate_v1"
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
REQUIRED_ARCHIVE_ROLES = (
    "generation",
    "manifest",
    "operation_audit_chain",
    "provider_raw_response",
    "resource_ledger",
    "semantic_events",
    "top20_results",
)
ALLOWED_WINDOWS = (1, 2, 4)
CURRENT_PRIMARY_AVAILABLE_BYTES = 587_336_777_728
BACKUP_REQUIRED_BYTES = 2_119_029_489_664
BACKUP_REQUIRED_INODES = 210_940


class ShardRetentionError(RuntimeError):
    """A retention, archive, recovery, or aggregate invariant failed."""


class ShardRetentionNotReady(ShardRetentionError):
    """Primary capacity may qualify, but a required backup fact is absent."""


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
        raw = path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise ShardRetentionError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite json number")
            ),
        )
    except ShardRetentionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ShardRetentionError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ShardRetentionError("json_root_not_object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ShardRetentionError("bound_input_unavailable") from exc
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
        raise ShardRetentionError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "archive",
        "bindings",
        "capacity",
        "execution",
        "formal_validation_complete",
        "policy",
        "population",
        "protocol",
        "protocol_sha256",
        "release",
        "schema_version",
        "source_commit",
    }
    if set(value) != required:
        raise ShardRetentionError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256
        or _protocol_digest(value) != FROZEN_PROTOCOL_SHA256
    ):
        raise ShardRetentionError("protocol_identity_invalid")
    policy = value["policy"]
    population = value["population"]
    capacity = value["capacity"]
    archive = value["archive"]
    if (
        policy != {
            "active_shard_window": 4,
            "allowed_active_shard_windows": [1, 2, 4],
            "default_enabled": False,
            "runtime_window_adjustment_allowed": False,
            "window_limits_residency_only": True,
        }
        or population
        != {
            "http_attempt_upper": 19280,
            "queries_per_shard": 50,
            "query_count": 1000,
            "shard_count": 20,
        }
        or capacity["shard_max_bytes"] != SHARD_MAX_BYTES
        or capacity["shard_max_files"] != SHARD_MAX_FILES
        or capacity["aggregate_bytes"] != AGGREGATE_MAX_BYTES
        or capacity["aggregate_files"] != AGGREGATE_MAX_FILES
        or capacity["safety_reserve_bytes"] != PRIMARY_RESERVE_BYTES
        or capacity["safety_reserve_inodes"] != PRIMARY_RESERVE_INODES
        or capacity["compression_credit_bytes"] != 0
        or capacity["future_cleanup_credit_bytes"] != 0
        or archive["required_roles"] != list(REQUIRED_ARCHIVE_ROLES)
        or archive["staging_bytes"] != SHARD_MAX_BYTES
        or archive["staging_files"] != SHARD_MAX_FILES
    ):
        raise ShardRetentionError("protocol_policy_invalid")
    bindings = value["bindings"]
    expected_bindings = {
        "crash_consistency",
        "disaster_recovery",
        "evidence_revocation",
        "evidence_transparency",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "multivolume_storage",
        "provider_ingest",
        "resource_accounting",
        "storage_governance",
    }
    if set(bindings) != expected_bindings:
        raise ShardRetentionError("protocol_binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ShardRetentionError("protocol_binding_invalid")
        relative = _safe_relative(str(row["path"]))
        bound = repository_root / relative
        if not bound.is_file() or sha256_file(bound) != row["sha256"]:
            raise ShardRetentionError("protocol_binding_hash_drift")
    return value


def capacity_requirement(window: int) -> dict[str, int]:
    if window not in ALLOWED_WINDOWS:
        raise ShardRetentionError("active_shard_window_invalid")
    return {
        "active_shard_window": window,
        "resident_shard_bytes": window * SHARD_MAX_BYTES,
        "resident_shard_files": window * SHARD_MAX_FILES,
        "archive_staging_bytes": SHARD_MAX_BYTES,
        "archive_staging_files": SHARD_MAX_FILES,
        "aggregate_bytes": AGGREGATE_MAX_BYTES,
        "aggregate_files": AGGREGATE_MAX_FILES,
        "safety_reserve_bytes": PRIMARY_RESERVE_BYTES,
        "safety_reserve_inodes": PRIMARY_RESERVE_INODES,
        "required_primary_bytes": (
            (window + 1) * SHARD_MAX_BYTES
            + AGGREGATE_MAX_BYTES
            + PRIMARY_RESERVE_BYTES
        ),
        "required_primary_inodes": (
            (window + 1) * SHARD_MAX_FILES
            + AGGREGATE_MAX_FILES
            + PRIMARY_RESERVE_INODES
        ),
    }


def build_addendum(protocol: Mapping[str, Any], *, window: int | None = None) -> dict[str, Any]:
    selected = int(protocol["policy"]["active_shard_window"] if window is None else window)
    if selected != protocol["policy"]["active_shard_window"]:
        raise ShardRetentionError("runtime_window_adjustment_forbidden")
    requirements = {
        str(item): capacity_requirement(item) for item in ALLOWED_WINDOWS
    }
    payload: dict[str, Any] = {
        "contract": ADDENDUM_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": SOURCE_COMMIT,
        "active_shard_window": selected,
        "allowed_active_shard_windows": list(ALLOWED_WINDOWS),
        "default_enabled": False,
        "runtime_window_adjustment_allowed": False,
        "window_capacity_requirements": requirements,
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "http_attempt_upper": 19280,
        "request_or_result_semantics_changed": False,
        "local_release_requires_verified_backup_and_restore": True,
        "backup_required_bytes": BACKUP_REQUIRED_BYTES,
        "backup_required_inodes": BACKUP_REQUIRED_INODES,
        "bindings": {
            key: row["sha256"] for key, row in sorted(protocol["bindings"].items())
        },
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def validate_addendum(addendum: Mapping[str, Any]) -> None:
    required = {
        "active_shard_window",
        "addendum_sha256",
        "allowed_active_shard_windows",
        "backup_required_bytes",
        "backup_required_inodes",
        "bindings",
        "contract",
        "default_enabled",
        "formal_validation_complete",
        "http_attempt_upper",
        "local_release_requires_verified_backup_and_restore",
        "protocol_sha256",
        "query_count",
        "request_or_result_semantics_changed",
        "runtime_window_adjustment_allowed",
        "schema_version",
        "shard_count",
        "source_commit",
        "window_capacity_requirements",
    }
    if set(addendum) != required:
        raise ShardRetentionError("addendum_schema_invalid")
    if (
        addendum["contract"] != ADDENDUM_CONTRACT
        or addendum["schema_version"] != SCHEMA_VERSION
        or addendum["source_commit"] != SOURCE_COMMIT
        or addendum["protocol_sha256"] != FROZEN_PROTOCOL_SHA256
        or addendum["active_shard_window"] != 4
        or addendum["allowed_active_shard_windows"] != list(ALLOWED_WINDOWS)
        or addendum["default_enabled"] is not False
        or addendum["runtime_window_adjustment_allowed"] is not False
        or addendum["request_or_result_semantics_changed"] is not False
        or addendum["local_release_requires_verified_backup_and_restore"] is not True
        or addendum["query_count"] != QUERY_COUNT
        or addendum["shard_count"] != SHARD_COUNT
        or addendum["http_attempt_upper"] != 19280
        or addendum["backup_required_bytes"] != BACKUP_REQUIRED_BYTES
        or addendum["backup_required_inodes"] != BACKUP_REQUIRED_INODES
        or addendum["formal_validation_complete"] is not False
        or addendum["window_capacity_requirements"]
        != {
            str(window): capacity_requirement(window)
            for window in ALLOWED_WINDOWS
        }
        or stable_hash(
            {
                key: value
                for key, value in addendum.items()
                if key != "addendum_sha256"
            }
        )
        != addendum["addendum_sha256"]
    ):
        raise ShardRetentionError("addendum_identity_invalid")


def verify_capacity(
    addendum: Mapping[str, Any],
    *,
    primary_available_bytes: int,
    primary_available_inodes: int | str,
    primary_quota_bytes: int | str,
    backup_available_bytes: int | str,
    backup_available_inodes: int | str,
    backup_quota_bytes: int | str,
    backup_failure_domain_independent: bool | str,
) -> dict[str, Any]:
    validate_addendum(addendum)
    window = int(addendum["active_shard_window"])
    requirement = capacity_requirement(window)
    if primary_available_bytes < 0:
        raise ShardRetentionError("primary_capacity_invalid")
    missing: list[str] = []
    if primary_available_inodes == NOT_AVAILABLE:
        missing.append("primary.available_inodes")
    if primary_quota_bytes == NOT_AVAILABLE:
        missing.append("primary.filesystem_quota_bytes")
    if backup_available_bytes == NOT_AVAILABLE:
        missing.append("backup.available_bytes")
    if backup_available_inodes == NOT_AVAILABLE:
        missing.append("backup.available_inodes")
    if backup_quota_bytes == NOT_AVAILABLE:
        missing.append("backup.filesystem_quota_bytes")
    if backup_failure_domain_independent == NOT_AVAILABLE:
        missing.append("backup.failure_domain_independent")
    primary_limit = (
        min(primary_available_bytes, int(primary_quota_bytes))
        if isinstance(primary_quota_bytes, int)
        else primary_available_bytes
    )
    primary_qualified = primary_limit >= requirement["required_primary_bytes"]
    if isinstance(primary_available_inodes, int):
        primary_qualified = (
            primary_qualified
            and primary_available_inodes >= requirement["required_primary_inodes"]
        )
    if not primary_qualified:
        raise ShardRetentionError("primary_window_capacity_insufficient")
    backup_qualified = False
    if not any(item.startswith("backup.") for item in missing):
        backup_limit = min(int(backup_available_bytes), int(backup_quota_bytes))
        backup_qualified = (
            backup_limit >= int(addendum["backup_required_bytes"])
            and int(backup_available_inodes) >= int(addendum["backup_required_inodes"])
            and backup_failure_domain_independent is True
        )
        if not backup_qualified:
            raise ShardRetentionError("backup_capacity_or_failure_domain_insufficient")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "streaming_retention_ready"
            if not missing and backup_qualified
            else "not_ready_missing_qualified_backup"
        ),
        "exit_code": EXIT_READY if not missing and backup_qualified else EXIT_NOT_READY,
        "active_shard_window": window,
        "primary_available_bytes": primary_available_bytes,
        "primary_required_bytes": requirement["required_primary_bytes"],
        "primary_qualified": primary_qualified,
        "backup_qualified": backup_qualified,
        "missing_observations": sorted(missing),
        "full1000_blocker_cleared": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


@dataclass
class ShardAuthority:
    shard_index: int
    attempt_identity: str
    generation_identity: str
    query_results: list[dict[str, str]]
    files: dict[str, bytes]
    selected_final_attempt_unique: bool = True
    committed_chain_verified: bool = True
    active_generation: bool = False
    resume_point: bool = False
    local_present: bool = True
    writer_identity: str | None = None

    @property
    def authority_sha256(self) -> str:
        return stable_hash(
            {
                "attempt_identity": self.attempt_identity,
                "files": {
                    key: hashlib.sha256(value).hexdigest()
                    for key, value in sorted(self.files.items())
                },
                "generation_identity": self.generation_identity,
                "query_results": self.query_results,
                "shard_index": self.shard_index,
            }
        )


def build_fixture_authority(shard_index: int) -> ShardAuthority:
    if shard_index not in range(SHARD_COUNT):
        raise ShardRetentionError("shard_index_invalid")
    results = [
        {
            "query_identity": stable_hash({"query_index": shard_index * 50 + offset}),
            "result_sha256": stable_hash(
                {"query_index": shard_index * 50 + offset, "result": "offline_fake"}
            ),
        }
        for offset in range(50)
    ]
    files = {
        role: canonical_json(
            {
                "role": role,
                "shard_index": shard_index,
                "synthetic_only": True,
            }
        )
        for role in REQUIRED_ARCHIVE_ROLES
    }
    return ShardAuthority(
        shard_index=shard_index,
        attempt_identity=stable_hash({"attempt": shard_index, "selected": True}),
        generation_identity=stable_hash({"generation": shard_index}),
        query_results=results,
        files=files,
    )


def create_archive(
    authority: ShardAuthority,
    *,
    parent_archive_sha256: str | None,
    backup_qualified: bool,
    restore_drill_verified: bool,
) -> dict[str, Any]:
    if not authority.local_present:
        raise ShardRetentionError("local_authority_missing")
    if not authority.selected_final_attempt_unique:
        raise ShardRetentionError("selected_final_attempt_not_unique")
    if not authority.committed_chain_verified:
        raise ShardRetentionError("committed_chain_invalid")
    if not backup_qualified:
        raise ShardRetentionNotReady("qualified_backup_unavailable")
    inventory = [
        {
            "role": role,
            "sha256": hashlib.sha256(authority.files[role]).hexdigest(),
            "size": len(authority.files[role]),
        }
        for role in REQUIRED_ARCHIVE_ROLES
    ]
    payload: dict[str, Any] = {
        "contract": ARCHIVE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "shard_index": authority.shard_index,
        "attempt_identity": authority.attempt_identity,
        "generation_identity": authority.generation_identity,
        "authority_sha256": authority.authority_sha256,
        "parent_archive_sha256": parent_archive_sha256,
        "files": inventory,
        "query_results": authority.query_results,
        "backup_qualified": True,
        "restore_drill_verified": restore_drill_verified,
        "formal_validation_complete": False,
    }
    payload["archive_sha256"] = stable_hash(payload)
    return payload


def verify_archive(
    archive: Mapping[str, Any],
    *,
    authority: ShardAuthority | None = None,
    known_archives: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    expected_keys = {
        "archive_sha256",
        "attempt_identity",
        "authority_sha256",
        "backup_qualified",
        "contract",
        "files",
        "formal_validation_complete",
        "generation_identity",
        "parent_archive_sha256",
        "query_results",
        "restore_drill_verified",
        "schema_version",
        "shard_index",
    }
    if set(archive) != expected_keys:
        raise ShardRetentionError("archive_schema_invalid")
    if (
        archive["contract"] != ARCHIVE_CONTRACT
        or archive["schema_version"] != SCHEMA_VERSION
        or archive["formal_validation_complete"] is not False
        or archive["backup_qualified"] is not True
        or archive["restore_drill_verified"] is not True
        or stable_hash({k: v for k, v in archive.items() if k != "archive_sha256"})
        != archive["archive_sha256"]
    ):
        raise ShardRetentionError("archive_integrity_invalid")
    files = archive["files"]
    if not isinstance(files, list):
        raise ShardRetentionError("archive_inventory_invalid")
    roles = [row.get("role") for row in files if isinstance(row, dict)]
    if roles != list(REQUIRED_ARCHIVE_ROLES) or len(roles) != len(set(roles)):
        raise ShardRetentionError("archive_inventory_invalid")
    if authority is not None:
        if archive["authority_sha256"] != authority.authority_sha256:
            raise ShardRetentionError("archive_authority_mismatch")
        for row in files:
            role = str(row["role"])
            content = authority.files[role]
            if (
                row.get("size") != len(content)
                or row.get("sha256") != hashlib.sha256(content).hexdigest()
            ):
                raise ShardRetentionError("archive_file_hash_mismatch")
    parent = archive["parent_archive_sha256"]
    if parent is not None:
        if known_archives is None or parent not in known_archives:
            raise ShardRetentionError("archive_parent_missing")
        seen = {str(archive["archive_sha256"])}
        current = parent
        while current is not None:
            if current in seen:
                raise ShardRetentionError("archive_parent_cycle")
            seen.add(current)
            row = known_archives.get(current)
            if row is None:
                raise ShardRetentionError("archive_parent_missing")
            current = row.get("parent_archive_sha256")


class EvictionLedger:
    """Append-only receipt chain; only a completed receipt authorizes absence."""

    def __init__(self) -> None:
        self.receipts: list[dict[str, Any]] = []

    def append(
        self,
        *,
        shard_index: int,
        authority_sha256: str,
        archive_sha256: str,
        state: str,
    ) -> dict[str, Any]:
        if state not in {"eviction_started", "eviction_completed"}:
            raise ShardRetentionError("eviction_state_invalid")
        previous = self.receipts[-1]["receipt_sha256"] if self.receipts else None
        payload: dict[str, Any] = {
            "contract": RECEIPT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "sequence": len(self.receipts),
            "shard_index": shard_index,
            "authority_sha256": authority_sha256,
            "archive_sha256": archive_sha256,
            "state": state,
            "previous_receipt_sha256": previous,
        }
        payload["receipt_sha256"] = stable_hash(payload)
        self.receipts.append(payload)
        return payload

    def verify(self) -> None:
        previous = None
        for sequence, row in enumerate(self.receipts):
            if (
                row.get("sequence") != sequence
                or row.get("previous_receipt_sha256") != previous
                or stable_hash({k: v for k, v in row.items() if k != "receipt_sha256"})
                != row.get("receipt_sha256")
            ):
                raise ShardRetentionError("eviction_receipt_chain_invalid")
            previous = row["receipt_sha256"]


def evict_local(
    authority: ShardAuthority,
    archive: Mapping[str, Any],
    ledger: EvictionLedger,
    *,
    known_archives: Mapping[str, Mapping[str, Any]] | None = None,
    transparency_path_referenced: bool = False,
    revocation_path_referenced: bool = False,
    fault_after_start: bool = False,
) -> dict[str, Any]:
    verify_archive(
        archive,
        authority=authority,
        known_archives=known_archives,
    )
    if (
        authority.active_generation
        or authority.resume_point
        or transparency_path_referenced
        or revocation_path_referenced
    ):
        raise ShardRetentionError("local_release_forbidden_reference")
    ledger.append(
        shard_index=authority.shard_index,
        authority_sha256=authority.authority_sha256,
        archive_sha256=str(archive["archive_sha256"]),
        state="eviction_started",
    )
    if fault_after_start:
        raise ShardRetentionError("eviction_interrupted_authority_preserved")
    authority.local_present = False
    completed = ledger.append(
        shard_index=authority.shard_index,
        authority_sha256=authority.authority_sha256,
        archive_sha256=str(archive["archive_sha256"]),
        state="eviction_completed",
    )
    ledger.verify()
    return completed


def restore_authority(
    archive: Mapping[str, Any],
    *,
    known_archives: Mapping[str, Mapping[str, Any]] | None = None,
) -> ShardAuthority:
    verify_archive(archive, known_archives=known_archives)
    shard_index = int(archive["shard_index"])
    authority = build_fixture_authority(shard_index)
    if (
        authority.authority_sha256 != archive["authority_sha256"]
        or authority.query_results != archive["query_results"]
    ):
        raise ShardRetentionError("archive_restore_mismatch")
    return authority


def acquire_writer(authority: ShardAuthority, writer_identity: str) -> None:
    if authority.writer_identity not in {None, writer_identity}:
        raise ShardRetentionError("concurrent_shard_writer_rejected")
    authority.writer_identity = writer_identity


def aggregate_mixed(
    references: Sequence[Mapping[str, Any]],
    *,
    archives: Mapping[str, Mapping[str, Any]],
    local: Mapping[int, ShardAuthority],
) -> dict[str, Any]:
    if len(references) != SHARD_COUNT:
        raise ShardRetentionError("aggregate_partial_shard_inventory")
    by_shard: dict[int, Mapping[str, Any]] = {}
    query_results: list[dict[str, str]] = []
    normalized: list[dict[str, Any]] = []
    for row in references:
        shard = row.get("shard_index")
        if not isinstance(shard, int) or shard in by_shard:
            raise ShardRetentionError("aggregate_duplicate_shard")
        by_shard[shard] = row
    if set(by_shard) != set(range(SHARD_COUNT)):
        raise ShardRetentionError("aggregate_partial_shard_inventory")
    for shard in range(SHARD_COUNT):
        row = by_shard[shard]
        location = row.get("location")
        if location == "local":
            authority = local.get(shard)
            if authority is None or not authority.local_present:
                raise ShardRetentionError("aggregate_local_authority_missing")
            digest = authority.authority_sha256
            results = authority.query_results
        elif location == "archive":
            archive_id = str(row.get("archive_sha256") or "")
            archive = archives.get(archive_id)
            if archive is None:
                raise ShardRetentionError("aggregate_archive_missing")
            verify_archive(archive, known_archives=archives)
            digest = str(archive["authority_sha256"])
            results = list(archive["query_results"])
        else:
            raise ShardRetentionError("aggregate_location_invalid")
        if row.get("authority_sha256") != digest:
            raise ShardRetentionError("aggregate_authority_hash_mismatch")
        query_results.extend(results)
        normalized.append(
            {
                "authority_sha256": digest,
                "location": location,
                "shard_index": shard,
            }
        )
    if len(query_results) != QUERY_COUNT:
        raise ShardRetentionError("aggregate_query_coverage_invalid")
    payload = {
        "contract": AGGREGATE_CONTRACT,
        "references": normalized,
        "query_results_sha256": stable_hash(query_results),
        "query_count": len(query_results),
        "history_rewritten": False,
    }
    return {**payload, "aggregate_sha256": stable_hash(payload)}


def _simulate_window(window: int) -> dict[str, Any]:
    requirement = capacity_requirement(window)
    active: dict[int, ShardAuthority] = {}
    archives: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    ledger = EvictionLedger()
    adapter_calls: dict[str, int] = {}
    parent: str | None = None
    peak = 0

    def archive_and_release(shard: int) -> None:
        nonlocal parent
        authority = active[shard]
        archive = create_archive(
            authority,
            parent_archive_sha256=parent,
            backup_qualified=True,
            restore_drill_verified=True,
        )
        verify_archive(archive, authority=authority, known_archives=archives)
        archives[str(archive["archive_sha256"])] = archive
        parent = str(archive["archive_sha256"])
        evict_local(
            authority,
            archive,
            ledger,
            known_archives=archives,
        )
        del active[shard]
        references.append(
            {
                "shard_index": shard,
                "location": "archive",
                "archive_sha256": archive["archive_sha256"],
                "authority_sha256": archive["authority_sha256"],
            }
        )

    for shard in range(SHARD_COUNT):
        authority = build_fixture_authority(shard)
        active[shard] = authority
        peak = max(peak, len(active))
        for row in authority.query_results:
            query_identity = row["query_identity"]
            adapter_calls[query_identity] = adapter_calls.get(query_identity, 0) + 1
        if len(active) == window:
            archive_and_release(min(active))
    while active:
        archive_and_release(min(active))
    aggregate = aggregate_mixed(references, archives=archives, local=active)
    uninterrupted = [
        row
        for shard in range(SHARD_COUNT)
        for row in build_fixture_authority(shard).query_results
    ]
    return {
        "active_shard_window": window,
        "required_primary_bytes": requirement["required_primary_bytes"],
        "primary_capacity_qualified": (
            CURRENT_PRIMARY_AVAILABLE_BYTES >= requirement["required_primary_bytes"]
        ),
        "peak_resident_shards": peak,
        "archive_count": len(archives),
        "eviction_receipt_count": len(ledger.receipts),
        "adapter_call_count": sum(adapter_calls.values()),
        "duplicate_request_count": sum(
            count - 1 for count in adapter_calls.values() if count > 1
        ),
        "resource_ledger_conserved": (
            len(adapter_calls) == QUERY_COUNT
            and sum(adapter_calls.values()) == QUERY_COUNT
        ),
        "aggregate_query_count": aggregate["query_count"],
        "aggregate_matches_uninterrupted": (
            aggregate["query_results_sha256"] == stable_hash(uninterrupted)
        ),
    }


def _expect_error(call: Any) -> str:
    try:
        call()
    except ShardRetentionError as exc:
        return str(exc)
    return "no_error"


def simulate_streaming(protocol: Mapping[str, Any]) -> dict[str, Any]:
    windows = {str(item): _simulate_window(item) for item in ALLOWED_WINDOWS}
    authority = build_fixture_authority(0)
    archive = create_archive(
        authority,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    interrupted_ledger = EvictionLedger()
    interruption = _expect_error(
        lambda: evict_local(
            authority,
            archive,
            interrupted_ledger,
            fault_after_start=True,
        )
    )
    tampered = copy.deepcopy(archive)
    tampered["files"][0]["sha256"] = "0" * 64
    unavailable = _expect_error(
        lambda: create_archive(
            build_fixture_authority(1),
            parent_archive_sha256=None,
            backup_qualified=False,
            restore_drill_verified=True,
        )
    )
    active = build_fixture_authority(2)
    active.active_generation = True
    active_archive = create_archive(
        active,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    early = _expect_error(
        lambda: evict_local(active, active_archive, EvictionLedger())
    )
    restored = restore_authority(archive)
    writer = build_fixture_authority(3)
    acquire_writer(writer, stable_hash({"writer": "one"}))
    double_writer = _expect_error(
        lambda: acquire_writer(writer, stable_hash({"writer": "two"}))
    )
    scenarios = {
        "archive_tamper": {
            "status": "passed"
            if _expect_error(lambda: verify_archive(tampered, authority=authority))
            == "archive_integrity_invalid"
            else "failed"
        },
        "backup_unavailable": {
            "status": "passed"
            if unavailable == "qualified_backup_unavailable"
            else "failed"
        },
        "delete_interruption": {
            "status": "passed"
            if interruption == "eviction_interrupted_authority_preserved"
            and authority.local_present
            and not any(
                row["state"] == "eviction_completed"
                for row in interrupted_ledger.receipts
            )
            else "failed"
        },
        "double_writer": {
            "status": "passed"
            if double_writer == "concurrent_shard_writer_rejected"
            else "failed"
        },
        "early_release": {
            "status": "passed"
            if early == "local_release_forbidden_reference"
            else "failed"
        },
        "restore_released_shard": {
            "status": "passed"
            if restored.authority_sha256 == archive["authority_sha256"]
            else "failed",
            "additional_adapter_calls": 0,
        },
    }
    for window, report in windows.items():
        scenarios[f"window_{window}"] = {
            "status": "passed"
            if report["primary_capacity_qualified"]
            and report["duplicate_request_count"] == 0
            and report["resource_ledger_conserved"]
            and report["aggregate_matches_uninterrupted"]
            else "failed"
        }
    all_passed = all(row["status"] == "passed" for row in scenarios.values())
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "streaming_retention_ready"
            if all_passed
            else "retention_or_recovery_violation"
        ),
        "exit_code": EXIT_READY if all_passed else EXIT_VIOLATION,
        "scenario_count": len(scenarios),
        "scenarios": {key: scenarios[key] for key in sorted(scenarios)},
        "window_reports": windows,
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "http_attempt_upper": 19280,
        "synthetic_only": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    addendum = build_addendum(protocol)
    windows = {
        str(item): {
            **capacity_requirement(item),
            "current_primary_available_bytes": CURRENT_PRIMARY_AVAILABLE_BYTES,
            "current_primary_qualified": (
                CURRENT_PRIMARY_AVAILABLE_BYTES
                >= capacity_requirement(item)["required_primary_bytes"]
            ),
        }
        for item in ALLOWED_WINDOWS
    }
    selected = capacity_requirement(int(addendum["active_shard_window"]))
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_qualified_backup",
        "exit_code": EXIT_NOT_READY,
        "active_shard_window": addendum["active_shard_window"],
        "current_primary_available_bytes": CURRENT_PRIMARY_AVAILABLE_BYTES,
        "current_primary_required_bytes": selected["required_primary_bytes"],
        "current_primary_qualified": (
            CURRENT_PRIMARY_AVAILABLE_BYTES >= selected["required_primary_bytes"]
        ),
        "window_capacity": windows,
        "backup_available_bytes": NOT_AVAILABLE,
        "backup_available_inodes": NOT_AVAILABLE,
        "backup_failure_domain_independent": NOT_AVAILABLE,
        "missing_backup_fields": [
            "backup.available_bytes",
            "backup.available_inodes",
            "backup.failure_domain_independent",
            "backup.filesystem_quota_bytes",
        ],
        "controls_default_enabled": False,
        "formal_run_started": False,
        "full1000_blocker_cleared": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
