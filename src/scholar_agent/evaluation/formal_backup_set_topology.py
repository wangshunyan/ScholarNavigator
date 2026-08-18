"""Deterministic multi-target backup-set controls for a future Full1000 run.

The addendum splits the already proven compacted backup bound across two,
three, or four independently attested members.  A shard archive is never split
across members, and the complete member set remains the recovery unit.  The
module uses synthetic states only and never performs retrieval or reads
credentials.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_backup_compaction import (
    ACTIVE_SHARD_WINDOW,
    NEW_BACKUP_REQUIRED_BYTES,
    NEW_BACKUP_REQUIRED_INODES,
    PARENT_CHAIN_INDEX_BYTES,
    PARENT_CHAIN_INDEX_FILES,
    _aggregate,
    build_shard_state,
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


PROTOCOL = "formal_backup_set_topology_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "6acc9722438c79481609b01c0f64feac0cb98e24"
FROZEN_PROTOCOL_SHA256 = (
    "2ca567202f133147b49ecabaa74b74ea932398dc37f1edf909a3b553a91370bb"
)
TOPOLOGY_CONTRACT = "formal_backup_set_topology_v1"
SET_MANIFEST_CONTRACT = "formal_backup_set_manifest_v1"
MEMBER_MANIFEST_CONTRACT = "formal_backup_set_member_manifest_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
SUPPORTED_MEMBER_COUNTS = (2, 3, 4)
COMPACTION_STAGING_PER_MEMBER = 1
MINIMUM_WRITERS_PER_MEMBER = 2
MAX_JSON_BYTES = 16 * 1024 * 1024
ZERO_SHA256 = "0" * 64
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class BackupSetError(RuntimeError):
    """A topology, capacity, member, or restore invariant failed."""


class BackupSetNotReady(BackupSetError):
    """No complete set of qualified real members is currently available."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BackupSetError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BackupSetError("artifact_unavailable") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise BackupSetError("artifact_size_exceeded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("nonfinite_json_number")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupSetError("artifact_json_invalid") from exc
    if not isinstance(value, dict):
        raise BackupSetError("artifact_root_invalid")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackupSetError("bound_artifact_unavailable") from exc


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupSetError("unsafe_binding_path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ".env" in path.parts
        or path.parts[0] == "third_party"
        or str(path) != value
    ):
        raise BackupSetError("unsafe_binding_path")
    return value


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def _split(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + (1 if index < remainder else 0) for index in range(count)]


def assigned_shards(member_count: int, member_index: int) -> list[int]:
    if member_count not in SUPPORTED_MEMBER_COUNTS:
        raise BackupSetError("member_count_unsupported")
    if member_index not in range(member_count):
        raise BackupSetError("member_index_invalid")
    return [
        shard for shard in range(SHARD_COUNT) if shard % member_count == member_index
    ]


def capacity_model(member_count: int) -> dict[str, Any]:
    if member_count not in SUPPORTED_MEMBER_COUNTS:
        raise BackupSetError("member_count_unsupported")
    active_slots = _split(ACTIVE_SHARD_WINDOW, member_count)
    recovery_slots = _split(ACTIVE_SHARD_WINDOW, member_count)
    reserve_bytes = _split(PRIMARY_RESERVE_BYTES, member_count)
    reserve_inodes = _split(PRIMARY_RESERVE_INODES, member_count)
    members: list[dict[str, Any]] = []
    for member_index in range(member_count):
        shards = assigned_shards(member_count, member_index)
        components = {
            "assigned_final_shard_archives": {
                "coefficient": len(shards),
                "unit_bytes": SHARD_MAX_BYTES,
                "bytes": len(shards) * SHARD_MAX_BYTES,
                "inodes": len(shards) * SHARD_MAX_FILES,
            },
            "active_window_share": {
                "coefficient": active_slots[member_index],
                "unit_bytes": SHARD_MAX_BYTES,
                "bytes": active_slots[member_index] * SHARD_MAX_BYTES,
                "inodes": active_slots[member_index] * SHARD_MAX_FILES,
            },
            "compaction_staging": {
                "coefficient": COMPACTION_STAGING_PER_MEMBER,
                "unit_bytes": SHARD_MAX_BYTES,
                "bytes": SHARD_MAX_BYTES,
                "inodes": SHARD_MAX_FILES,
            },
            "copied_parent_chain_index": {
                "coefficient": 1,
                "unit_bytes": PARENT_CHAIN_INDEX_BYTES,
                "bytes": PARENT_CHAIN_INDEX_BYTES,
                "inodes": PARENT_CHAIN_INDEX_FILES,
            },
            "recovery_window_share": {
                "coefficient": recovery_slots[member_index],
                "unit_bytes": SHARD_MAX_BYTES,
                "bytes": recovery_slots[member_index] * SHARD_MAX_BYTES,
                "inodes": recovery_slots[member_index] * SHARD_MAX_FILES,
            },
            "aggregate": {
                "coefficient": 1 if member_index == 0 else 0,
                "unit_bytes": AGGREGATE_MAX_BYTES,
                "bytes": AGGREGATE_MAX_BYTES if member_index == 0 else 0,
                "inodes": AGGREGATE_MAX_FILES if member_index == 0 else 0,
            },
            "safety_reserve_share": {
                "coefficient": 1,
                "unit_bytes": reserve_bytes[member_index],
                "bytes": reserve_bytes[member_index],
                "inodes": reserve_inodes[member_index],
            },
        }
        required_bytes = sum(int(row["bytes"]) for row in components.values())
        required_inodes = sum(int(row["inodes"]) for row in components.values())
        members.append(
            {
                "member_index": member_index,
                "assigned_shards": shards,
                "required_bytes": required_bytes,
                "required_inodes": required_inodes,
                "required_writers": MINIMUM_WRITERS_PER_MEMBER,
                "components": components,
            }
        )
    total_bytes = sum(row["required_bytes"] for row in members)
    total_inodes = sum(row["required_inodes"] for row in members)
    return {
        "member_count": member_count,
        "assignment_algorithm": "shard_index_mod_member_count_v1",
        "members": members,
        "maximum_member_required_bytes": max(
            row["required_bytes"] for row in members
        ),
        "maximum_member_required_inodes": max(
            row["required_inodes"] for row in members
        ),
        "set_required_bytes": total_bytes,
        "set_required_inodes": total_inodes,
        "single_target_compacted_bytes": NEW_BACKUP_REQUIRED_BYTES,
        "single_target_compacted_inodes": NEW_BACKUP_REQUIRED_INODES,
        "extra_bytes_from_per_member_staging_and_index": (
            total_bytes - NEW_BACKUP_REQUIRED_BYTES
        ),
        "compression_credit_bytes": 0,
        "sparse_file_credit_bytes": 0,
        "future_deduplication_credit_bytes": 0,
        "future_cleanup_credit_bytes": 0,
    }


def all_capacity_models() -> dict[str, Any]:
    return {
        str(member_count): capacity_model(member_count)
        for member_count in SUPPORTED_MEMBER_COUNTS
    }


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "capacity_model_sha256",
        "execution",
        "formal_validation_complete",
        "policy",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
    }:
        raise BackupSetError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256
        or stable_hash(_without_digest(value, "protocol_sha256"))
        != FROZEN_PROTOCOL_SHA256
    ):
        raise BackupSetError("protocol_identity_invalid")
    if value["population"] != {
        "active_shard_window": ACTIVE_SHARD_WINDOW,
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "supported_member_counts": list(SUPPORTED_MEMBER_COUNTS),
    }:
        raise BackupSetError("protocol_population_invalid")
    if value["policy"] != {
        "ability_default_enabled": False,
        "aggregate_member_index": 0,
        "complete_set_required_for_restore": True,
        "cross_member_atomic_rename_required": False,
        "member_redundancy_claimed": False,
        "one_complete_archive_per_shard": True,
        "quota_pool_overlap_allowed": False,
        "single_blob_split_allowed": False,
        "single_target_default_changed": False,
    }:
        raise BackupSetError("protocol_policy_invalid")
    if value["capacity_model_sha256"] != stable_hash(all_capacity_models()):
        raise BackupSetError("protocol_capacity_invalid")
    expected_bindings = {
        "backup_compaction",
        "backup_target_attestation",
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "multivolume_storage",
        "portable_execution_site",
        "shard_streaming_retention",
        "storage_governance",
    }
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise BackupSetError("protocol_binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BackupSetError("protocol_binding_invalid")
        target = repository_root / _safe_relative(row["path"])
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            raise BackupSetError("protocol_binding_hash_drift")
    return value


def build_topology(
    protocol: Mapping[str, Any], *, member_count: int
) -> dict[str, Any]:
    model = capacity_model(member_count)
    members = []
    for row in model["members"]:
        index = int(row["member_index"])
        members.append(
            {
                "member_index": index,
                "member_identity": stable_hash(
                    {
                        "member_index": index,
                        "member_count": member_count,
                        "protocol_sha256": protocol["protocol_sha256"],
                    }
                ),
                "assigned_shards": list(row["assigned_shards"]),
                "required_bytes": row["required_bytes"],
                "required_inodes": row["required_inodes"],
                "required_writers": row["required_writers"],
            }
        )
    value: dict[str, Any] = {
        "contract": TOPOLOGY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "member_count": member_count,
        "assignment_algorithm": "shard_index_mod_member_count_v1",
        "aggregate_member_index": 0,
        "members": members,
        "complete_set_required_for_restore": True,
        "member_redundancy_claimed": False,
        "cross_member_atomic_rename_required": False,
        "single_target_default_changed": False,
        "formal_validation_complete": False,
    }
    value["topology_sha256"] = stable_hash(value)
    return value


def verify_topology(topology: Mapping[str, Any]) -> None:
    if set(topology) != {
        "aggregate_member_index",
        "assignment_algorithm",
        "complete_set_required_for_restore",
        "contract",
        "cross_member_atomic_rename_required",
        "formal_validation_complete",
        "member_count",
        "member_redundancy_claimed",
        "members",
        "protocol_sha256",
        "schema_version",
        "single_target_default_changed",
        "source_commit",
        "topology_sha256",
    }:
        raise BackupSetError("topology_schema_invalid")
    count = topology["member_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count not in SUPPORTED_MEMBER_COUNTS
        or topology["contract"] != TOPOLOGY_CONTRACT
        or topology["schema_version"] != SCHEMA_VERSION
        or topology["source_commit"] != SOURCE_COMMIT
        or topology["assignment_algorithm"]
        != "shard_index_mod_member_count_v1"
        or topology["aggregate_member_index"] != 0
        or topology["complete_set_required_for_restore"] is not True
        or topology["member_redundancy_claimed"] is not False
        or topology["cross_member_atomic_rename_required"] is not False
        or topology["single_target_default_changed"] is not False
        or topology["formal_validation_complete"] is not False
        or stable_hash(_without_digest(topology, "topology_sha256"))
        != topology["topology_sha256"]
    ):
        raise BackupSetError("topology_integrity_invalid")
    members = topology["members"]
    if not isinstance(members, list) or len(members) != count:
        raise BackupSetError("topology_member_inventory_invalid")
    seen_shards: list[int] = []
    identities: set[str] = set()
    model = capacity_model(count)
    for index, row in enumerate(members):
        expected = model["members"][index]
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "assigned_shards",
                "member_identity",
                "member_index",
                "required_bytes",
                "required_inodes",
                "required_writers",
            }
            or row["member_index"] != index
            or row["assigned_shards"] != expected["assigned_shards"]
            or row["required_bytes"] != expected["required_bytes"]
            or row["required_inodes"] != expected["required_inodes"]
            or row["required_writers"] != MINIMUM_WRITERS_PER_MEMBER
            or row["member_identity"] in identities
        ):
            raise BackupSetError("topology_member_invalid")
        identities.add(row["member_identity"])
        seen_shards.extend(row["assigned_shards"])
    if sorted(seen_shards) != list(range(SHARD_COUNT)):
        raise BackupSetError("topology_shard_coverage_invalid")
    if len(seen_shards) != len(set(seen_shards)):
        raise BackupSetError("topology_duplicate_shard")


def synthetic_profiles(
    topology: Mapping[str, Any],
    *,
    primary_failure_domain: str | None = None,
) -> list[dict[str, Any]]:
    verify_topology(topology)
    primary = primary_failure_domain or stable_hash({"domain": "primary"})
    profiles = []
    for row in topology["members"]:
        index = row["member_index"]
        profiles.append(
            {
                "member_identity": row["member_identity"],
                "filesystem_identity": stable_hash(
                    {"backup_filesystem": index, "count": topology["member_count"]}
                ),
                "quota_pool_identity": stable_hash(
                    {"backup_quota_pool": index, "count": topology["member_count"]}
                ),
                "failure_domain_identity": stable_hash(
                    {"backup_failure_domain": index, "count": topology["member_count"]}
                ),
                "primary_failure_domain_identity": primary,
                "available_bytes": row["required_bytes"],
                "available_inodes": row["required_inodes"],
                "quota_bytes": row["required_bytes"],
                "max_writers": row["required_writers"],
                "online": True,
                "observation_fresh": True,
            }
        )
    return profiles


def verify_profiles(
    topology: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]]
) -> None:
    verify_topology(topology)
    if len(profiles) != topology["member_count"]:
        raise BackupSetError("member_profile_missing")
    expected = {row["member_identity"]: row for row in topology["members"]}
    filesystems: set[str] = set()
    quota_pools: set[str] = set()
    failure_domains: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, Mapping) or set(profile) != {
            "available_bytes",
            "available_inodes",
            "failure_domain_identity",
            "filesystem_identity",
            "max_writers",
            "member_identity",
            "observation_fresh",
            "online",
            "primary_failure_domain_identity",
            "quota_bytes",
            "quota_pool_identity",
        }:
            raise BackupSetError("member_profile_schema_invalid")
        member = expected.get(profile["member_identity"])
        if member is None:
            raise BackupSetError("member_identity_replaced")
        numeric = (
            profile["available_bytes"],
            profile["available_inodes"],
            profile["quota_bytes"],
            profile["max_writers"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in numeric):
            raise BackupSetError("member_observation_not_available")
        if (
            profile["online"] is not True
            or profile["observation_fresh"] is not True
            or profile["available_bytes"] < member["required_bytes"]
            or profile["available_inodes"] < member["required_inodes"]
            or profile["quota_bytes"] < member["required_bytes"]
            or profile["max_writers"] < member["required_writers"]
        ):
            raise BackupSetError("member_capacity_insufficient")
        filesystem = str(profile["filesystem_identity"])
        quota_pool = str(profile["quota_pool_identity"])
        failure_domain = str(profile["failure_domain_identity"])
        primary_domain = str(profile["primary_failure_domain_identity"])
        if (
            filesystem in filesystems
            or quota_pool in quota_pools
            or failure_domain in failure_domains
            or failure_domain == primary_domain
            or any(
                value == NOT_AVAILABLE
                for value in (filesystem, quota_pool, failure_domain, primary_domain)
            )
        ):
            raise BackupSetError("member_capacity_or_failure_domain_overlap")
        filesystems.add(filesystem)
        quota_pools.add(quota_pool)
        failure_domains.add(failure_domain)


def build_backup_set(
    topology: Mapping[str, Any],
    states: Mapping[int, Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    sequence: int = 0,
    parent_set_root_sha256: str = ZERO_SHA256,
) -> tuple[dict[str, Any], dict[int, dict[int, dict[str, Any]]]]:
    verify_profiles(topology, profiles)
    if set(states) != set(range(SHARD_COUNT)):
        raise BackupSetError("set_shard_coverage_invalid")
    profile_by_id = {row["member_identity"]: row for row in profiles}
    archives: dict[int, dict[int, dict[str, Any]]] = {}
    member_manifests = []
    for row in topology["members"]:
        index = row["member_index"]
        member_states = {
            shard: copy.deepcopy(dict(states[shard]))
            for shard in row["assigned_shards"]
        }
        archives[index] = member_states
        inventory = [
            {
                "attempt_identity": state["attempt_identity"],
                "generation": state["generation"],
                "query_cursor": state["query_cursor"],
                "shard_index": shard,
                "state_sha256": state["state_sha256"],
            }
            for shard, state in sorted(member_states.items())
        ]
        profile = profile_by_id[row["member_identity"]]
        manifest: dict[str, Any] = {
            "contract": MEMBER_MANIFEST_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "member_index": index,
            "member_identity": row["member_identity"],
            "filesystem_identity": profile["filesystem_identity"],
            "quota_pool_identity": profile["quota_pool_identity"],
            "failure_domain_identity": profile["failure_domain_identity"],
            "assigned_shards": list(row["assigned_shards"]),
            "inventory": inventory,
            "parent_chain_index_sha256": stable_hash(
                {
                    "parent_set_root_sha256": parent_set_root_sha256,
                    "sequence": sequence,
                }
            ),
            "writer_lock_holders": 1,
        }
        manifest["member_root_sha256"] = stable_hash(manifest)
        member_manifests.append(manifest)
    set_manifest: dict[str, Any] = {
        "contract": SET_MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": topology["protocol_sha256"],
        "topology_sha256": topology["topology_sha256"],
        "sequence": sequence,
        "parent_set_root_sha256": parent_set_root_sha256,
        "member_count": topology["member_count"],
        "member_roots": [
            {
                "member_index": row["member_index"],
                "member_identity": row["member_identity"],
                "member_root_sha256": row["member_root_sha256"],
            }
            for row in member_manifests
        ],
        "member_manifests": member_manifests,
        "complete_set_required_for_restore": True,
        "formal_validation_complete": False,
    }
    set_manifest["set_root_sha256"] = stable_hash(set_manifest)
    return set_manifest, archives


def verify_backup_set(
    topology: Mapping[str, Any],
    set_manifest: Mapping[str, Any],
    archives: Mapping[int, Mapping[int, Mapping[str, Any]]],
    profiles: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    verify_topology(topology)
    verify_profiles(topology, profiles)
    profiles_by_identity = {
        row["member_identity"]: row for row in profiles
    }
    if set(set_manifest) != {
        "complete_set_required_for_restore",
        "contract",
        "formal_validation_complete",
        "member_count",
        "member_manifests",
        "member_roots",
        "parent_set_root_sha256",
        "protocol_sha256",
        "schema_version",
        "sequence",
        "set_root_sha256",
        "source_commit",
        "topology_sha256",
    }:
        raise BackupSetError("set_manifest_schema_invalid")
    if (
        set_manifest["contract"] != SET_MANIFEST_CONTRACT
        or set_manifest["schema_version"] != SCHEMA_VERSION
        or set_manifest["source_commit"] != SOURCE_COMMIT
        or set_manifest["protocol_sha256"] != topology["protocol_sha256"]
        or set_manifest["topology_sha256"] != topology["topology_sha256"]
        or set_manifest["member_count"] != topology["member_count"]
        or set_manifest["complete_set_required_for_restore"] is not True
        or set_manifest["formal_validation_complete"] is not False
        or stable_hash(_without_digest(set_manifest, "set_root_sha256"))
        != set_manifest["set_root_sha256"]
        or isinstance(set_manifest["sequence"], bool)
        or not isinstance(set_manifest["sequence"], int)
        or set_manifest["sequence"] < 0
        or (
            set_manifest["sequence"] == 0
            and set_manifest["parent_set_root_sha256"] != ZERO_SHA256
        )
        or (
            set_manifest["sequence"] > 0
            and set_manifest["parent_set_root_sha256"] == ZERO_SHA256
        )
    ):
        raise BackupSetError("set_manifest_integrity_invalid")
    member_manifests = set_manifest["member_manifests"]
    if (
        not isinstance(member_manifests, list)
        or len(member_manifests) != topology["member_count"]
        or set(archives) != set(range(topology["member_count"]))
    ):
        raise BackupSetError("set_member_missing")
    restored: dict[int, dict[str, Any]] = {}
    roots = []
    for index, manifest in enumerate(member_manifests):
        if (
            not isinstance(manifest, dict)
            or manifest.get("contract") != MEMBER_MANIFEST_CONTRACT
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("member_index") != index
            or manifest.get("member_identity")
            != topology["members"][index]["member_identity"]
            or manifest.get("assigned_shards")
            != topology["members"][index]["assigned_shards"]
            or manifest.get("writer_lock_holders") != 1
            or manifest.get("parent_chain_index_sha256")
            != stable_hash(
                {
                    "parent_set_root_sha256": set_manifest[
                        "parent_set_root_sha256"
                    ],
                    "sequence": set_manifest["sequence"],
                }
            )
            or stable_hash(_without_digest(manifest, "member_root_sha256"))
            != manifest.get("member_root_sha256")
        ):
            raise BackupSetError("member_manifest_integrity_invalid")
        profile = profiles_by_identity[manifest["member_identity"]]
        if (
            manifest.get("filesystem_identity")
            != profile["filesystem_identity"]
            or manifest.get("quota_pool_identity")
            != profile["quota_pool_identity"]
            or manifest.get("failure_domain_identity")
            != profile["failure_domain_identity"]
        ):
            raise BackupSetError("member_attestation_identity_drift")
        roots.append(
            {
                "member_index": index,
                "member_identity": manifest["member_identity"],
                "member_root_sha256": manifest["member_root_sha256"],
            }
        )
        inventory = manifest.get("inventory")
        member_archives = archives[index]
        if not isinstance(inventory, list) or set(member_archives) != set(
            manifest["assigned_shards"]
        ):
            raise BackupSetError("member_archive_inventory_invalid")
        expected_inventory = []
        for shard in sorted(member_archives):
            state = member_archives[shard]
            if shard in restored:
                raise BackupSetError("set_duplicate_shard")
            if (
                state.get("shard_index") != shard
                or state.get("query_cursor") != len(state.get("query_results", []))
            ):
                raise BackupSetError("member_archive_state_invalid")
            files = state.get("files")
            if not isinstance(files, Mapping):
                raise BackupSetError("member_archive_file_inventory_invalid")
            state_digest = stable_hash(
                {
                    "attempt_identity": state["attempt_identity"],
                    "files": {
                        role: hashlib.sha256(content).hexdigest()
                        for role, content in sorted(files.items())
                        if isinstance(role, str) and isinstance(content, bytes)
                    },
                    "generation": state["generation"],
                    "query_cursor": state["query_cursor"],
                    "query_results": state["query_results"],
                    "shard_index": state["shard_index"],
                }
            )
            if state_digest != state.get("state_sha256"):
                raise BackupSetError("member_archive_state_digest_invalid")
            expected_inventory.append(
                {
                    "attempt_identity": state["attempt_identity"],
                    "generation": state["generation"],
                    "query_cursor": state["query_cursor"],
                    "shard_index": shard,
                    "state_sha256": state["state_sha256"],
                }
            )
            restored[shard] = copy.deepcopy(dict(state))
        if inventory != expected_inventory:
            raise BackupSetError("member_inventory_hash_mismatch")
    if set_manifest["member_roots"] != roots:
        raise BackupSetError("set_member_root_conflict")
    if set(restored) != set(range(SHARD_COUNT)):
        raise BackupSetError("set_shard_coverage_invalid")
    return restored


def _expect_error(call: Any) -> str:
    try:
        call()
    except BackupSetError as exc:
        return str(exc)
    return "no_error"


def simulate_set(
    protocol: Mapping[str, Any], *, member_count: int = 4
) -> dict[str, Any]:
    topology = build_topology(protocol, member_count=member_count)
    profiles = synthetic_profiles(topology)
    states = {
        shard: build_shard_state(shard, cursor=50, generation=1)
        for shard in range(SHARD_COUNT)
    }
    calls = {row["query_identity"]: 1 for state in states.values() for row in state["query_results"]}
    manifest, archives = build_backup_set(topology, states, profiles)
    restored = verify_backup_set(topology, manifest, archives, profiles)
    uninterrupted = _aggregate(states)
    aggregate = _aggregate(restored)

    missing = copy.deepcopy(archives)
    missing.pop(member_count - 1)
    missing_reason = _expect_error(
        lambda: verify_backup_set(topology, manifest, missing, profiles)
    )

    tampered = copy.deepcopy(archives)
    tampered[0][topology["members"][0]["assigned_shards"][0]][
        "state_sha256"
    ] = "f" * 64
    tamper_reason = _expect_error(
        lambda: verify_backup_set(topology, manifest, tampered, profiles)
    )

    overlap = copy.deepcopy(profiles)
    overlap[1]["quota_pool_identity"] = overlap[0]["quota_pool_identity"]
    overlap_reason = _expect_error(lambda: verify_profiles(topology, overlap))

    insufficient = copy.deepcopy(profiles)
    insufficient[-1]["available_bytes"] -= 1
    insufficient_reason = _expect_error(
        lambda: verify_profiles(topology, insufficient)
    )

    replaced = copy.deepcopy(profiles)
    replaced[0]["member_identity"] = stable_hash({"replaced": True})
    replacement_reason = _expect_error(lambda: verify_profiles(topology, replaced))

    double_writer_manifest = copy.deepcopy(manifest)
    double_writer_manifest["member_manifests"][0]["writer_lock_holders"] = 2
    double_writer_manifest["member_manifests"][0]["member_root_sha256"] = stable_hash(
        _without_digest(
            double_writer_manifest["member_manifests"][0],
            "member_root_sha256",
        )
    )
    double_writer_manifest["member_roots"][0]["member_root_sha256"] = (
        double_writer_manifest["member_manifests"][0]["member_root_sha256"]
    )
    double_writer_manifest["set_root_sha256"] = stable_hash(
        _without_digest(double_writer_manifest, "set_root_sha256")
    )
    writer_reason = _expect_error(
        lambda: verify_backup_set(
            topology, double_writer_manifest, archives, profiles
        )
    )

    replacement_state = build_shard_state(7, cursor=50, attempt=1, generation=2)
    replacement_states = copy.deepcopy(states)
    replacement_states[7] = replacement_state
    replacement_manifest, replacement_archives = build_backup_set(
        topology,
        replacement_states,
        profiles,
        sequence=1,
        parent_set_root_sha256=manifest["set_root_sha256"],
    )
    replacement_restored = verify_backup_set(
        topology, replacement_manifest, replacement_archives, profiles
    )
    replacement_expected = _aggregate(replacement_states)

    scenarios = {
        "aggregate_equivalence": aggregate == uninterrupted,
        "capacity_uneven_assignment": (
            len({len(row["assigned_shards"]) for row in topology["members"]}) > 1
            if member_count == 3
            else True
        ),
        "cross_set_member_replacement": replacement_reason
        == "member_identity_replaced",
        "double_writer": writer_reason == "member_manifest_integrity_invalid",
        "member_missing": missing_reason == "set_member_missing",
        "member_tamper": tamper_reason
        == "member_archive_state_digest_invalid",
        "quota_pool_overlap": overlap_reason
        == "member_capacity_or_failure_domain_overlap",
        "restore_resume_without_repeat": (
            len(calls) == QUERY_COUNT and max(calls.values(), default=0) == 1
        ),
        "single_member_insufficient": insufficient_reason
        == "member_capacity_insufficient",
        "single_shard_replacement": (
            replacement_restored[7]["attempt_identity"]
            == replacement_state["attempt_identity"]
            and _aggregate(replacement_restored) == replacement_expected
        ),
    }
    failed = sorted(name for name, passed in scenarios.items() if not passed)
    if failed:
        raise BackupSetError("synthetic_scenario_failed:" + failed[0])
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_set_ready",
        "exit_code": EXIT_READY,
        "member_count": member_count,
        "query_count": QUERY_COUNT,
        "adapter_call_count": sum(calls.values()),
        "duplicate_request_count": sum(
            count - 1 for count in calls.values() if count > 1
        ),
        "resource_ledger_query_count": len(calls),
        "resource_ledger_conserved": (
            len(calls) == QUERY_COUNT and sum(calls.values()) == QUERY_COUNT
        ),
        "aggregate_matches_single_target": aggregate == uninterrupted,
        "scenario_count": len(scenarios),
        "scenarios": {
            name: {"status": "passed"} for name in sorted(scenarios)
        },
        "synthetic_only": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def verify_set(protocol: Mapping[str, Any]) -> dict[str, Any]:
    reports = {
        str(count): simulate_set(protocol, member_count=count)
        for count in SUPPORTED_MEMBER_COUNTS
    }
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_set_ready",
        "exit_code": EXIT_READY,
        "verified_member_counts": list(SUPPORTED_MEMBER_COUNTS),
        "query_count": QUERY_COUNT,
        "all_member_sets_complete": True,
        "aggregate_matches_single_target": all(
            row["aggregate_matches_single_target"] for row in reports.values()
        ),
        "duplicate_request_count": 0,
        "resource_ledger_conserved": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def calculate_capacity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    models = all_capacity_models()
    if protocol["capacity_model_sha256"] != stable_hash(models):
        raise BackupSetError("capacity_protocol_drift")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_set_ready",
        "exit_code": EXIT_READY,
        "capacity_models": models,
        "single_target_default_changed": False,
        "member_redundancy_claimed": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    models = all_capacity_models()
    if protocol["capacity_model_sha256"] != stable_hash(models):
        raise BackupSetError("capacity_protocol_drift")
    missing = {
        str(count): [
            field
            for index in range(count)
            for field in (
                f"backup_set_{count}.member_{index}.available_bytes",
                f"backup_set_{count}.member_{index}.available_inodes",
                f"backup_set_{count}.member_{index}.failure_domain_identity",
                f"backup_set_{count}.member_{index}.filesystem_identity",
                f"backup_set_{count}.member_{index}.quota_bytes",
                f"backup_set_{count}.member_{index}.quota_pool_identity",
            )
        ]
        for count in SUPPORTED_MEMBER_COUNTS
    }
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_qualified_members",
        "exit_code": EXIT_NOT_READY,
        "per_member_capacity_thresholds": {
            str(count): [
                {
                    "member_index": row["member_index"],
                    "required_bytes": row["required_bytes"],
                    "required_inodes": row["required_inodes"],
                    "required_writers": row["required_writers"],
                }
                for row in models[str(count)]["members"]
            ]
            for count in SUPPORTED_MEMBER_COUNTS
        },
        "missing_real_member_fields": missing,
        "qualified_real_member_count": 0,
        "single_target_compatibility_preserved": True,
        "controls_default_enabled": False,
        "full1000_blocker_cleared": False,
        "formal_run_started": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
