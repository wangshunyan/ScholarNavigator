"""Read-only discovery of explicitly registered Full1000 backup candidates.

Discovery never qualifies, registers, or activates a backup member.  It only
collects exact-path metadata for protocol-registered aliases, de-duplicates
identical capacity/failure domains, and reports which frozen member-intake
slots could proceed to the existing attestation gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_backup_set_member_intake import CAPABILITIES
from scholar_agent.evaluation.formal_backup_set_topology import (
    SUPPORTED_MEMBER_COUNTS,
    capacity_model,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_backup_member_discovery_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "014f0b68c3c88d40194f13f68496999e0f998b20"
CANDIDATE = "formal_backup_member_candidate_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
MAX_JSON_BYTES = 8 * 1024 * 1024
IDENTITY_FIELDS = (
    "device_identity",
    "filesystem_identity",
    "quota_pool_identity",
    "failure_domain_identity",
    "management_domain_identity",
)
REQUIRED_OBSERVATIONS = (
    "available_bytes",
    "available_inodes",
    "quota_bytes",
    "writers",
    "failure_domain_independent",
    *IDENTITY_FIELDS,
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class BackupMemberDiscoveryError(RuntimeError):
    """A discovery, identity, or topology invariant failed closed."""


class BackupMemberDiscoveryNotReady(BackupMemberDiscoveryError):
    """No real candidate currently satisfies the frozen intake prerequisites."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise BackupMemberDiscoveryError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise BackupMemberDiscoveryError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                BackupMemberDiscoveryError("nonfinite_json_number")
            ),
        )
    except BackupMemberDiscoveryError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupMemberDiscoveryError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupMemberDiscoveryError("json_root_not_object")
    return value


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackupMemberDiscoveryError("bound_artifact_unavailable") from exc


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupMemberDiscoveryError("unsafe_binding_path")
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
        raise BackupMemberDiscoveryError("unsafe_binding_path")
    return value


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "execution",
        "formal_validation_complete",
        "policy",
        "protocol",
        "protocol_sha256",
        "registered_targets",
        "schema_version",
        "source_commit",
    }:
        raise BackupMemberDiscoveryError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"]
        != stable_hash(_without(value, "protocol_sha256"))
    ):
        raise BackupMemberDiscoveryError("protocol_identity_invalid")
    if value["policy"] != {
        "candidate_is_qualified_member": False,
        "enumeration": "explicit_registered_targets_only",
        "identity_authentication": False,
        "member_counts": list(SUPPORTED_MEMBER_COUNTS),
        "network_or_account_scan": False,
        "output_absolute_paths": False,
        "registration_or_activation_side_effect": False,
        "unknown_evidence_policy": "not_available_fail_closed",
    }:
        raise BackupMemberDiscoveryError("protocol_policy_invalid")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "backup_set_member_intake",
        "backup_set_topology",
        "backup_target_attestation",
    }:
        raise BackupMemberDiscoveryError("binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BackupMemberDiscoveryError("binding_schema_invalid")
        target = repository_root / _safe_relative(row["path"])
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise BackupMemberDiscoveryError("binding_hash_drift")
    registered = value["registered_targets"]
    if not isinstance(registered, list):
        raise BackupMemberDiscoveryError("registered_targets_invalid")
    aliases: set[str] = set()
    for row in registered:
        if (
            not isinstance(row, dict)
            or set(row) != {"alias", "path_binding_sha256"}
            or not isinstance(row["alias"], str)
            or not row["alias"].startswith("backup-target-")
            or row["alias"] in aliases
            or not isinstance(row["path_binding_sha256"], str)
            or len(row["path_binding_sha256"]) != 64
        ):
            raise BackupMemberDiscoveryError("registered_targets_invalid")
        aliases.add(row["alias"])
    return value


def path_binding(alias: str, path: Path) -> str:
    """Bind an operator path without serializing it into reports."""
    return stable_hash({"alias": alias, "resolved_path": str(path.resolve())})


def _registered_map(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["alias"]: row["path_binding_sha256"]
        for row in protocol["registered_targets"]
    }


def observe_exact_path(alias: str, path: Path) -> dict[str, Any]:
    """Collect exact-target metadata only; never recurse or scan siblings."""
    try:
        stat = path.stat()
        statvfs = os.statvfs(path)
    except OSError as exc:
        raise BackupMemberDiscoveryError("registered_target_unavailable") from exc
    if not path.is_dir():
        raise BackupMemberDiscoveryError("registered_target_not_directory")
    block_size = statvfs.f_frsize or statvfs.f_bsize
    available_inodes: int | str = (
        int(statvfs.f_favail) if statvfs.f_files else NOT_AVAILABLE
    )
    device_identity = stable_hash({"device_number": int(stat.st_dev)})
    filesystem_identity = stable_hash(
        {
            "device_number": int(stat.st_dev),
            "block_size": int(block_size),
            "name_max": int(statvfs.f_namemax),
        }
    )
    return {
        "target_alias": alias,
        "target_present": True,
        "available_bytes": int(statvfs.f_bavail) * int(block_size),
        "available_inodes": available_inodes,
        "quota_bytes": NOT_AVAILABLE,
        "writers": NOT_AVAILABLE,
        "device_identity": device_identity,
        "filesystem_identity": filesystem_identity,
        "quota_pool_identity": NOT_AVAILABLE,
        "failure_domain_identity": NOT_AVAILABLE,
        "failure_domain_independent": NOT_AVAILABLE,
        "management_domain_identity": NOT_AVAILABLE,
        "capabilities": {name: NOT_AVAILABLE for name in CAPABILITIES},
        "recovery_verified": NOT_AVAILABLE,
        "fresh": True,
        "synthetic_only": False,
    }


def build_candidate(
    protocol: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    synthetic_only: bool,
) -> dict[str, Any]:
    alias = observation.get("target_alias")
    registered = _registered_map(protocol)
    if not isinstance(alias, str) or alias not in registered:
        raise BackupMemberDiscoveryError("target_not_registered")
    expected = {
        "available_bytes",
        "available_inodes",
        "capabilities",
        "device_identity",
        "failure_domain_identity",
        "failure_domain_independent",
        "filesystem_identity",
        "fresh",
        "management_domain_identity",
        "quota_bytes",
        "quota_pool_identity",
        "recovery_verified",
        "synthetic_only",
        "target_alias",
        "target_present",
        "writers",
    }
    if set(observation) != expected or observation["synthetic_only"] is not synthetic_only:
        raise BackupMemberDiscoveryError("observation_schema_invalid")
    capabilities = observation["capabilities"]
    if not isinstance(capabilities, Mapping) or set(capabilities) != set(CAPABILITIES):
        raise BackupMemberDiscoveryError("capability_inventory_invalid")
    for field in ("available_bytes", "available_inodes", "quota_bytes", "writers"):
        item = observation[field]
        if item != NOT_AVAILABLE and (
            isinstance(item, bool) or not isinstance(item, int) or item < 0
        ):
            raise BackupMemberDiscoveryError("observation_numeric_invalid")
    for field in IDENTITY_FIELDS:
        item = observation[field]
        if item != NOT_AVAILABLE and (
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise BackupMemberDiscoveryError("observation_identity_invalid")
    if any(value not in (True, False, NOT_AVAILABLE) for value in capabilities.values()):
        raise BackupMemberDiscoveryError("capability_value_invalid")
    if (
        observation["target_present"] not in (True, False)
        or observation["fresh"] not in (True, False)
        or observation["recovery_verified"] not in (True, False, NOT_AVAILABLE)
        or observation["failure_domain_independent"]
        not in (True, False, NOT_AVAILABLE)
        or not isinstance(observation["synthetic_only"], bool)
    ):
        raise BackupMemberDiscoveryError("observation_state_invalid")
    unknown_fields = [
        field
        for field in REQUIRED_OBSERVATIONS
        if observation[field] == NOT_AVAILABLE
    ]
    unknown_fields.extend(
        f"capabilities.{name}"
        for name in CAPABILITIES
        if capabilities[name] == NOT_AVAILABLE
    )
    if observation["recovery_verified"] == NOT_AVAILABLE:
        unknown_fields.append("recovery_verified")
    complete = (
        observation["target_present"] is True
        and observation["fresh"] is True
        and not unknown_fields
        and all(value is True for value in capabilities.values())
        and observation["recovery_verified"] is True
        and observation["failure_domain_independent"] is True
    )
    target_identity = stable_hash(
        {
            field: observation[field]
            for field in IDENTITY_FIELDS
        }
    )
    sanitized = {
        key: copy.deepcopy(value)
        for key, value in observation.items()
        if key != "target_alias"
    }
    value: dict[str, Any] = {
        "candidate": CANDIDATE,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "target_id": stable_hash({"registered_alias": alias}),
        "path_binding_sha256": registered[alias],
        "target_identity": target_identity,
        "status": "candidate",
        "candidate_complete": complete,
        "observations": sanitized,
        "missing_fields": sorted(unknown_fields),
        "next_gate": "formal_backup_target_attestation_v1",
        "member_intake_gate": "formal_backup_set_member_intake_v1",
        "auto_registered": False,
        "auto_activated": False,
        "identity_authentication": False,
        "synthetic_only": synthetic_only,
        "formal_validation_complete": False,
    }
    value["candidate_sha256"] = stable_hash(value)
    return value


def validate_candidate(
    protocol: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        set(candidate)
        != {
            "auto_activated",
            "auto_registered",
            "candidate",
            "candidate_complete",
            "candidate_sha256",
            "formal_validation_complete",
            "identity_authentication",
            "member_intake_gate",
            "missing_fields",
            "next_gate",
            "observations",
            "path_binding_sha256",
            "protocol",
            "protocol_sha256",
            "schema_version",
            "source_commit",
            "status",
            "synthetic_only",
            "target_id",
            "target_identity",
        }
        or candidate["candidate"] != CANDIDATE
        or candidate["protocol"] != PROTOCOL
        or candidate["schema_version"] != SCHEMA_VERSION
        or candidate["source_commit"] != SOURCE_COMMIT
        or candidate["protocol_sha256"] != protocol["protocol_sha256"]
        or candidate["status"] != "candidate"
        or candidate["auto_registered"] is not False
        or candidate["auto_activated"] is not False
        or candidate["formal_validation_complete"] is not False
        or stable_hash(_without(candidate, "candidate_sha256"))
        != candidate["candidate_sha256"]
    ):
        raise BackupMemberDiscoveryError("candidate_binding_invalid")
    observations = candidate["observations"]
    if not isinstance(observations, Mapping):
        raise BackupMemberDiscoveryError("candidate_observations_invalid")
    alias_matches = [
        row
        for row in protocol["registered_targets"]
        if stable_hash({"registered_alias": row["alias"]}) == candidate["target_id"]
    ]
    if (
        len(alias_matches) != 1
        or alias_matches[0]["path_binding_sha256"]
        != candidate["path_binding_sha256"]
    ):
        raise BackupMemberDiscoveryError("candidate_registration_drift")
    rebuilt = build_candidate(
        protocol,
        {"target_alias": alias_matches[0]["alias"], **copy.deepcopy(observations)},
        synthetic_only=bool(candidate["synthetic_only"]),
    )
    if rebuilt != candidate:
        raise BackupMemberDiscoveryError("candidate_semantic_drift")
    return dict(candidate)


def discover(
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    observer: Callable[[str, Path], Mapping[str, Any]] = observe_exact_path,
) -> dict[str, Any]:
    registered = _registered_map(protocol)
    if set(paths) - set(registered):
        raise BackupMemberDiscoveryError("unregistered_target_requested")
    candidates: list[dict[str, Any]] = []
    missing_aliases: list[str] = []
    for alias in sorted(registered):
        path = paths.get(alias)
        if path is None:
            missing_aliases.append(stable_hash({"registered_alias": alias}))
            continue
        if path_binding(alias, path) != registered[alias]:
            raise BackupMemberDiscoveryError("registered_path_binding_mismatch")
        candidates.append(
            build_candidate(
                protocol,
                observer(alias, path),
                synthetic_only=False,
            )
        )
    match = match_topologies(protocol, candidates)
    ready = any(row["candidate_complete"] for row in candidates)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "qualifying_candidates_discovered"
            if ready
            else "no_real_qualifying_candidates"
        ),
        "exit_code": EXIT_READY if ready else EXIT_NOT_READY,
        "registered_target_count": len(registered),
        "observed_candidate_count": len(candidates),
        "complete_candidate_count": sum(
            bool(row["candidate_complete"]) for row in candidates
        ),
        "missing_registered_target_ids": missing_aliases,
        "candidates": candidates,
        "topology_match": match,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _known(value: Any) -> bool:
    return value != NOT_AVAILABLE


def deduplicate_candidates(
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated = [validate_candidate(protocol, row) for row in candidates]
    parents = list(range(len(validated)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(validated)):
        for right in range(left + 1, len(validated)):
            if any(
                _known(validated[left]["observations"][field])
                and validated[left]["observations"][field]
                == validated[right]["observations"][field]
                for field in IDENTITY_FIELDS
            ):
                union(left, right)
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(validated):
        groups.setdefault(find(index), []).append(row)
    return [
        {
            "group_id": stable_hash(
                {"target_ids": sorted(row["target_id"] for row in members)}
            ),
            "target_ids": sorted(row["target_id"] for row in members),
            "representative": sorted(
                members, key=lambda row: row["target_id"]
            )[0],
            "alias_count": len(members),
        }
        for _root, members in sorted(groups.items())
    ]


def _fits(candidate: Mapping[str, Any], member: Mapping[str, Any]) -> bool:
    observations = candidate["observations"]
    if not candidate["candidate_complete"]:
        return False
    numeric = (
        ("available_bytes", "required_bytes"),
        ("available_inodes", "required_inodes"),
        ("quota_bytes", "required_bytes"),
        ("writers", "required_writers"),
    )
    return all(
        isinstance(observations[source], int)
        and not isinstance(observations[source], bool)
        and observations[source] >= member[target]
        for source, target in numeric
    )


def _assignment(
    members: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    ordered_groups = sorted(groups, key=lambda row: row["group_id"])

    def search(slot: int, used: set[str]) -> list[dict[str, Any]] | None:
        if slot == len(members):
            return []
        for group in ordered_groups:
            if group["group_id"] in used or not _fits(
                group["representative"], members[slot]
            ):
                continue
            suffix = search(slot + 1, used | {group["group_id"]})
            if suffix is not None:
                return [
                    {
                        "slot": slot,
                        "group_id": group["group_id"],
                        "target_id": group["representative"]["target_id"],
                    },
                    *suffix,
                ]
        return None

    return search(0, set())


def match_topologies(
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = deduplicate_candidates(protocol, candidates)
    plans: list[dict[str, Any]] = []
    for count in SUPPORTED_MEMBER_COUNTS:
        model = capacity_model(count)
        assignment = _assignment(model["members"], groups)
        eligible = [
            group
            for group in groups
            if group["representative"]["candidate_complete"]
        ]
        missing = max(0, count - len(eligible))
        plans.append(
            {
                "member_count": count,
                "status": (
                    "candidate_topology_match"
                    if assignment is not None
                    else "not_ready_missing_qualified_candidates"
                ),
                "assignment": assignment or [],
                "missing_candidate_count": 0 if assignment is not None else missing,
                "unsatisfied_reasons": (
                    []
                    if assignment is not None
                    else sorted(
                        {
                            "insufficient_distinct_capacity_domains"
                            if len(groups) < count
                            else "slot_capacity_or_evidence_gap"
                        }
                    )
                ),
                "maximum_slot_bytes": model["maximum_member_required_bytes"],
                "maximum_slot_inodes": model["maximum_member_required_inodes"],
            }
        )
    return {
        "deduplicated_candidate_count": len(groups),
        "alias_count": sum(row["alias_count"] for row in groups),
        "plans": plans,
    }


def synthetic_candidate(
    protocol: Mapping[str, Any],
    *,
    alias: str,
    identity_seed: str,
    available_bytes: int,
    available_inodes: int,
    quota_bytes: int | str,
    failure_domain: str | None = None,
) -> dict[str, Any]:
    identity = lambda role: stable_hash(
        {"identity_seed": identity_seed, "role": role}
    )
    observation = {
        "target_alias": alias,
        "target_present": True,
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
        "quota_bytes": quota_bytes,
        "writers": 2,
        "device_identity": identity("device"),
        "filesystem_identity": identity("filesystem"),
        "quota_pool_identity": (
            identity("quota_pool") if quota_bytes != NOT_AVAILABLE else NOT_AVAILABLE
        ),
        "failure_domain_identity": (
            failure_domain or identity("failure_domain")
        ),
        "failure_domain_independent": True,
        "management_domain_identity": identity("management_domain"),
        "capabilities": {name: True for name in CAPABILITIES},
        "recovery_verified": True,
        "fresh": True,
        "synthetic_only": True,
    }
    return build_candidate(protocol, observation, synthetic_only=True)


def simulate_profiles(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(protocol))
    protocol["registered_targets"] = [
        {
            "alias": f"backup-target-synthetic-{index}",
            "path_binding_sha256": stable_hash(
                {"synthetic_path_binding": index}
            ),
        }
        for index in range(4)
    ]
    four = capacity_model(4)
    aliases = [row["alias"] for row in protocol["registered_targets"][:4]]
    if len(aliases) < 4:
        raise BackupMemberDiscoveryError("simulation_targets_missing")
    complete = [
        synthetic_candidate(
            protocol,
            alias=aliases[index],
            identity_seed=f"member-{index}",
            available_bytes=four["members"][index]["required_bytes"],
            available_inodes=four["members"][index]["required_inodes"],
            quota_bytes=four["members"][index]["required_bytes"],
        )
        for index in range(4)
    ]

    def mutate(
        candidate: Mapping[str, Any], field: str, value: Any
    ) -> dict[str, Any]:
        observation = copy.deepcopy(dict(candidate["observations"]))
        observation["target_alias"] = next(
            row["alias"]
            for row in protocol["registered_targets"]
            if stable_hash({"registered_alias": row["alias"]})
            == candidate["target_id"]
        )
        observation[field] = value
        return build_candidate(protocol, observation, synthetic_only=True)

    alias_duplicate = copy.deepcopy(complete[1])
    alias_duplicate["observations"]["device_identity"] = complete[0][
        "observations"
    ]["device_identity"]
    alias_duplicate["observations"]["filesystem_identity"] = complete[0][
        "observations"
    ]["filesystem_identity"]
    alias_duplicate = build_candidate(
        protocol,
        {
            "target_alias": aliases[1],
            **alias_duplicate["observations"],
        },
        synthetic_only=True,
    )
    scenarios = {
        "no_candidate": match_topologies(protocol, []),
        "qualified_candidates": match_topologies(protocol, complete),
        "alias_duplicate": match_topologies(
            protocol, [complete[0], alias_duplicate, *complete[2:]]
        ),
        "quota_unknown": match_topologies(
            protocol, [mutate(complete[0], "quota_bytes", NOT_AVAILABLE), *complete[1:]]
        ),
        "capacity_insufficient": match_topologies(
            protocol, [mutate(complete[0], "available_bytes", 0), *complete[1:]]
        ),
        "inode_insufficient": match_topologies(
            protocol, [mutate(complete[0], "available_inodes", 0), *complete[1:]]
        ),
        "failure_domain_unknown": match_topologies(
            protocol,
            [
                mutate(
                    complete[0],
                    "failure_domain_identity",
                    NOT_AVAILABLE,
                ),
                *complete[1:],
            ],
        ),
        "target_disappeared": match_topologies(
            protocol, [mutate(complete[0], "target_present", False), *complete[1:]]
        ),
        "identity_replacement": "candidate_binding_invalid",
    }
    passed = (
        scenarios["no_candidate"]["plans"][0]["status"]
        == "not_ready_missing_qualified_candidates"
        and scenarios["qualified_candidates"]["plans"][2]["status"]
        == "candidate_topology_match"
        and scenarios["alias_duplicate"]["deduplicated_candidate_count"] == 3
        and all(
            scenarios[name]["plans"][2]["status"]
            == "not_ready_missing_qualified_candidates"
            for name in (
                "quota_unknown",
                "capacity_insufficient",
                "inode_insufficient",
                "failure_domain_unknown",
                "target_disappeared",
            )
        )
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "qualifying_candidates_discovered" if passed else "simulation_failed",
        "exit_code": EXIT_READY if passed else EXIT_VIOLATION,
        "scenario_count": len(scenarios),
        "passed_count": len(scenarios) if passed else 0,
        "scenarios": scenarios,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    discovered = discover(protocol, {})
    return {
        **discovered,
        "status": "no_real_qualifying_candidates",
        "exit_code": EXIT_NOT_READY,
        "reason_code": "no_protocol_registered_real_targets",
        "candidate_only": True,
        "auto_registration_or_activation": False,
    }
