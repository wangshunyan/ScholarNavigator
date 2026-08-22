"""Explicit, private registration and local preflight for backup targets.

The registration document is operator-private and is never copied into a
report.  Only paths explicitly named by that document are touched.  A passed
preflight creates a candidate for the existing discovery/attestation chain;
it does not qualify a backup member or activate a backup set.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_backup_member_discovery import (
    NOT_AVAILABLE,
    build_candidate,
    match_topologies,
    path_binding,
)
from scholar_agent.evaluation.formal_backup_set_member_intake import CAPABILITIES
from scholar_agent.evaluation.formal_backup_set_topology import (
    SUPPORTED_MEMBER_COUNTS,
    capacity_model,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_backup_target_registration_v1"
PRIVATE_REGISTRATION = "formal_backup_target_private_registration_v1"
MANIFEST = "formal_backup_target_registration_manifest_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "29fc5556c0b6af65a96673b170cbbcae50735e06"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
MAX_JSON_BYTES = 1024 * 1024
ALIAS_RE = re.compile(r"^backup-target-[a-z0-9][a-z0-9-]{0,62}$")
OPAQUE_RE = re.compile(r"^[0-9a-f]{64}$")
PURPOSE = "full1000_backup_member_candidate"
PROBE_SCOPE = "exact_registered_directory_only"
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class BackupTargetRegistrationError(RuntimeError):
    """A registration, probe, identity, or privacy invariant failed closed."""


class BackupTargetRegistrationNotReady(BackupTargetRegistrationError):
    """No operator-provided real registration is available."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise BackupTargetRegistrationError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise BackupTargetRegistrationError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                BackupTargetRegistrationError("nonfinite_json_number")
            ),
        )
    except BackupTargetRegistrationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupTargetRegistrationError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupTargetRegistrationError("json_root_not_object")
    return value


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackupTargetRegistrationError("bound_artifact_unavailable") from exc


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupTargetRegistrationError("unsafe_binding_path")
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
        raise BackupTargetRegistrationError("unsafe_binding_path")
    return value


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "execution",
        "formal_validation_complete",
        "policy",
        "private_registration_schema",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
    }:
        raise BackupTargetRegistrationError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"]
        != stable_hash(_without(value, "protocol_sha256"))
    ):
        raise BackupTargetRegistrationError("protocol_identity_invalid")
    if value["policy"] != {
        "activation_side_effect": False,
        "candidate_status": "registered_candidate",
        "directory_recursion": False,
        "identity_authentication": False,
        "network_or_account_scan": False,
        "output_absolute_paths": False,
        "probe_cleanup_required": True,
        "quota_or_failure_domain_unknown": "not_available_downstream_fail_closed",
        "registered_paths_only": True,
    }:
        raise BackupTargetRegistrationError("protocol_policy_invalid")
    if value["private_registration_schema"] != {
        "allowed_probe_scope": PROBE_SCOPE,
        "operator_identity": "sha256_opaque",
        "path": "absolute_operator_private_not_committed",
        "purpose": PURPOSE,
        "revocations": "top_level_alias_list",
        "target_alias": "backup-target-slug",
    }:
        raise BackupTargetRegistrationError("registration_schema_contract_drift")
    bindings = value["bindings"]
    expected = {
        "backup_member_discovery",
        "backup_set_member_intake",
        "backup_set_topology",
        "backup_target_attestation",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise BackupTargetRegistrationError("binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BackupTargetRegistrationError("binding_schema_invalid")
        target = repository_root / _safe_relative(row["path"])
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise BackupTargetRegistrationError("binding_hash_drift")
    return value


def load_private_registration(
    path: Path, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "protocol_sha256",
        "registration",
        "revoked_aliases",
        "schema_version",
        "source_commit",
        "targets",
    }:
        raise BackupTargetRegistrationError("private_registration_schema_invalid")
    if (
        value["registration"] != PRIVATE_REGISTRATION
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["protocol_sha256"] != protocol["protocol_sha256"]
        or not isinstance(value["targets"], list)
        or not isinstance(value["revoked_aliases"], list)
    ):
        raise BackupTargetRegistrationError("private_registration_binding_invalid")
    aliases: set[str] = set()
    paths: set[str] = set()
    for row in value["targets"]:
        if not isinstance(row, dict) or set(row) != {
            "alias",
            "allowed_probe_scope",
            "operator_identity",
            "path",
            "purpose",
        }:
            raise BackupTargetRegistrationError("private_target_schema_invalid")
        alias, raw_path = row["alias"], row["path"]
        if (
            not isinstance(alias, str)
            or ALIAS_RE.fullmatch(alias) is None
            or alias in aliases
            or not isinstance(raw_path, str)
            or raw_path in paths
            or row["purpose"] != PURPOSE
            or row["allowed_probe_scope"] != PROBE_SCOPE
            or not isinstance(row["operator_identity"], str)
            or OPAQUE_RE.fullmatch(row["operator_identity"]) is None
        ):
            raise BackupTargetRegistrationError("private_target_invalid")
        _validate_explicit_path(Path(raw_path), require_exists=False)
        aliases.add(alias)
        paths.add(raw_path)
    revoked = value["revoked_aliases"]
    if (
        any(not isinstance(alias, str) or alias not in aliases for alias in revoked)
        or len(set(revoked)) != len(revoked)
    ):
        raise BackupTargetRegistrationError("revocation_inventory_invalid")
    return value


def _validate_explicit_path(path: Path, *, require_exists: bool = True) -> Path:
    raw = str(path)
    pure = PurePath(raw)
    if (
        not path.is_absolute()
        or raw == os.path.sep
        or "\x00" in raw
        or ".." in pure.parts
        or ".env" in pure.parts
        or any(part in {".ssh", ".aws", ".config"} for part in pure.parts)
    ):
        raise BackupTargetRegistrationError("registered_path_forbidden")
    if require_exists and (not path.exists() or not path.is_dir()):
        raise BackupTargetRegistrationError("registered_path_unavailable")
    if require_exists:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise BackupTargetRegistrationError("registered_path_unavailable") from exc
        if resolved != path or path.is_symlink():
            raise BackupTargetRegistrationError("registered_path_alias_or_symlink")
    return path


def _runtime(repository_root: Path) -> Any:
    path = repository_root / "scripts/formal_backup_target_runtime.py"
    spec = importlib.util.spec_from_file_location("_registration_target_runtime", path)
    if spec is None or spec.loader is None:
        raise BackupTargetRegistrationError("target_probe_runtime_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimum_slot_requirements() -> tuple[int, int]:
    members = [
        member
        for count in SUPPORTED_MEMBER_COUNTS
        for member in capacity_model(count)["members"]
    ]
    return (
        min(int(row["required_bytes"]) for row in members),
        min(int(row["required_inodes"]) for row in members),
    )


def _slot_matches(available_bytes: int, available_inodes: int) -> list[dict[str, int]]:
    return [
        {"member_count": count, "slot": index}
        for count in SUPPORTED_MEMBER_COUNTS
        for index, row in enumerate(capacity_model(count)["members"])
        if available_bytes >= row["required_bytes"]
        and available_inodes >= row["required_inodes"]
    ]


def observe_registered_target(
    alias: str,
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    target = _validate_explicit_path(path)
    runtime = _runtime(repository_root)
    before = {entry.name for entry in target.glob(".backup-attestation-*")}
    try:
        filesystem, device, available_bytes, available_inodes = (
            runtime._filesystem_id(target)
        )
        capability_rows = runtime._probe_capabilities(target, 240)
    except Exception as exc:
        raise BackupTargetRegistrationError("registered_target_probe_failed") from exc
    after = {entry.name for entry in target.glob(".backup-attestation-*")}
    if after != before:
        raise BackupTargetRegistrationError("probe_residue_detected")
    capabilities = {
        name: bool(capability_rows.get(name, {}).get("passed") is True)
        for name in CAPABILITIES
    }
    return {
        "target_alias": alias,
        "target_present": True,
        "available_bytes": int(available_bytes),
        "available_inodes": int(available_inodes),
        "quota_bytes": NOT_AVAILABLE,
        "writers": 2 if capabilities["concurrent_writer"] else 0,
        "device_identity": device,
        "filesystem_identity": filesystem,
        "quota_pool_identity": NOT_AVAILABLE,
        "failure_domain_identity": NOT_AVAILABLE,
        "failure_domain_independent": NOT_AVAILABLE,
        "management_domain_identity": NOT_AVAILABLE,
        "capabilities": capabilities,
        "recovery_verified": NOT_AVAILABLE,
        "fresh": True,
        "synthetic_only": False,
    }


def _discovery_protocol(
    protocol: Mapping[str, Any], registration: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol["bindings"]["backup_member_discovery"]["sha256"],
        "registered_targets": [
            {
                "alias": row["alias"],
                "path_binding_sha256": path_binding(row["alias"], Path(row["path"])),
            }
            for row in registration["targets"]
        ],
    }


def build_registration_manifest(
    protocol: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    repository_root: Path,
    observer: Callable[[str, Path], Mapping[str, Any]] | None = None,
    synthetic_only: bool = False,
) -> dict[str, Any]:
    discovery_protocol = _discovery_protocol(protocol, registration)
    revoked = set(registration["revoked_aliases"])
    entries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_devices: set[str] = set()
    seen_filesystems: set[str] = set()
    minimum_bytes, minimum_inodes = _minimum_slot_requirements()
    for row in sorted(registration["targets"], key=lambda item: item["alias"]):
        alias = row["alias"]
        path = _validate_explicit_path(Path(row["path"]))
        if alias in revoked:
            entries.append(
                {
                    "target_id": stable_hash({"registered_alias": alias}),
                    "operator_identity": row["operator_identity"],
                    "path_binding_sha256": path_binding(alias, path),
                    "status": "revoked",
                    "revoked": True,
                    "probe": None,
                    "slot_matches": [],
                    "next_gate": "none",
                }
            )
            continue
        observation = dict(
            observer(alias, path)
            if observer is not None
            else observe_registered_target(alias, path, repository_root=repository_root)
        )
        if observation.get("target_alias") != alias:
            raise BackupTargetRegistrationError("probe_alias_mismatch")
        if observation.get("synthetic_only") is not synthetic_only:
            raise BackupTargetRegistrationError("probe_origin_mismatch")
        capabilities = observation.get("capabilities")
        if (
            not isinstance(capabilities, Mapping)
            or set(capabilities) != set(CAPABILITIES)
            or not all(value is True for value in capabilities.values())
        ):
            raise BackupTargetRegistrationError("probe_capability_failed")
        available_bytes = observation.get("available_bytes")
        available_inodes = observation.get("available_inodes")
        if (
            isinstance(available_bytes, bool)
            or not isinstance(available_bytes, int)
            or isinstance(available_inodes, bool)
            or not isinstance(available_inodes, int)
            or available_bytes < minimum_bytes
            or available_inodes < minimum_inodes
        ):
            raise BackupTargetRegistrationError("probe_capacity_insufficient")
        device = observation.get("device_identity")
        filesystem = observation.get("filesystem_identity")
        if (
            not isinstance(device, str)
            or OPAQUE_RE.fullmatch(device) is None
            or not isinstance(filesystem, str)
            or OPAQUE_RE.fullmatch(filesystem) is None
        ):
            raise BackupTargetRegistrationError("probe_identity_invalid")
        if device in seen_devices or filesystem in seen_filesystems:
            raise BackupTargetRegistrationError("duplicate_device_or_filesystem")
        seen_devices.add(device)
        seen_filesystems.add(filesystem)
        candidate = build_candidate(
            discovery_protocol, observation, synthetic_only=synthetic_only
        )
        candidates.append(candidate)
        matches = _slot_matches(available_bytes, available_inodes)
        entries.append(
            {
                "target_id": candidate["target_id"],
                "operator_identity": row["operator_identity"],
                "path_binding_sha256": candidate["path_binding_sha256"],
                "status": "registered_candidate",
                "revoked": False,
                "probe": {
                    "available_bytes": available_bytes,
                    "available_inodes": available_inodes,
                    "capabilities": copy.deepcopy(capabilities),
                    "device_identity": device,
                    "filesystem_identity": filesystem,
                    "quota_observability": (
                        "observed"
                        if observation.get("quota_bytes") != NOT_AVAILABLE
                        else NOT_AVAILABLE
                    ),
                    "failure_domain_observability": (
                        "observed"
                        if observation.get("failure_domain_identity") != NOT_AVAILABLE
                        else NOT_AVAILABLE
                    ),
                    "probe_cleanup_verified": True,
                },
                "slot_matches": matches,
                "next_gate": "formal_backup_target_attestation_v1",
            }
        )
    match = match_topologies(discovery_protocol, candidates)
    value: dict[str, Any] = {
        "manifest": MANIFEST,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "private_registration_sha256": stable_hash(registration),
        "status": "registered_candidates_ready" if candidates else "no_real_registered_candidates",
        "registered_candidate_count": len(candidates),
        "revoked_count": len(revoked),
        "entries": entries,
        "discovery_topology_match": match,
        "qualification_boundary": {
            "candidate_is_qualified_member": False,
            "target_attestation_required": True,
            "member_intake_required": True,
            "backup_set_activated": False,
            "formal_validation_complete": False,
        },
        "synthetic_only": synthetic_only,
        "execution": dict(EXECUTION_ZERO),
    }
    value["manifest_sha256"] = stable_hash(value)
    return value


def validate_manifest(
    protocol: Mapping[str, Any],
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    observer: Callable[[str, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        manifest.get("manifest") != MANIFEST
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("private_registration_sha256") != stable_hash(registration)
        or manifest.get("manifest_sha256")
        != stable_hash(_without(manifest, "manifest_sha256"))
    ):
        raise BackupTargetRegistrationError("registration_manifest_binding_invalid")
    rebuilt = build_registration_manifest(
        protocol,
        registration,
        repository_root=repository_root,
        observer=observer,
        synthetic_only=bool(manifest.get("synthetic_only")),
    )
    if rebuilt != manifest:
        raise BackupTargetRegistrationError("registration_manifest_probe_drift")
    return dict(manifest)


def _synthetic_observation(
    alias: str,
    *,
    seed: str,
    available_bytes: int,
    available_inodes: int,
    quota: int | str = NOT_AVAILABLE,
    capabilities: bool = True,
) -> dict[str, Any]:
    identity = lambda role: stable_hash({"seed": seed, "role": role})
    return {
        "target_alias": alias,
        "target_present": True,
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
        "quota_bytes": quota,
        "writers": 2,
        "device_identity": identity("device"),
        "filesystem_identity": identity("filesystem"),
        "quota_pool_identity": identity("quota") if quota != NOT_AVAILABLE else NOT_AVAILABLE,
        "failure_domain_identity": NOT_AVAILABLE,
        "failure_domain_independent": NOT_AVAILABLE,
        "management_domain_identity": NOT_AVAILABLE,
        "capabilities": {name: capabilities for name in CAPABILITIES},
        "recovery_verified": NOT_AVAILABLE,
        "fresh": True,
        "synthetic_only": True,
    }


def simulate_profiles(protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    minimum_bytes, minimum_inodes = _minimum_slot_requirements()

    def registry(paths: Sequence[Path], revoked: Sequence[str] = ()) -> dict[str, Any]:
        return {
            "registration": PRIVATE_REGISTRATION,
            "schema_version": SCHEMA_VERSION,
            "source_commit": SOURCE_COMMIT,
            "protocol_sha256": protocol["protocol_sha256"],
            "targets": [
                {
                    "alias": f"backup-target-{index}",
                    "path": str(path),
                    "purpose": PURPOSE,
                    "allowed_probe_scope": PROBE_SCOPE,
                    "operator_identity": stable_hash({"operator": index}),
                }
                for index, path in enumerate(paths)
            ],
            "revoked_aliases": list(revoked),
        }

    def expected_error(call: Callable[[], Any]) -> str:
        try:
            call()
        except BackupTargetRegistrationError as exc:
            return str(exc)
        raise BackupTargetRegistrationError("expected_violation_missing")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="backup-registration-matrix-") as raw:
        root = Path(raw).resolve()
        paths = [root / f"target-{index}" for index in range(2)]
        for path in paths:
            path.mkdir()
        base = registry(paths)

        def observer(alias: str, _path: Path) -> dict[str, Any]:
            return _synthetic_observation(
                alias,
                seed=alias,
                available_bytes=minimum_bytes,
                available_inodes=minimum_inodes,
            )

        valid = build_registration_manifest(
            protocol,
            base,
            repository_root=repository_root,
            observer=observer,
            synthetic_only=True,
        )
        duplicate = copy.deepcopy(base)
        duplicate["targets"][1]["path"] = str(paths[0])
        missing = copy.deepcopy(base)
        missing["targets"][0]["path"] = str(root / "missing")
        symlink = root / "symlink"
        try:
            symlink.symlink_to(paths[0], target_is_directory=True)
        except OSError:
            # Windows installations without Developer Mode or the relevant
            # privilege cannot create a real link.  Do not manufacture an
            # alias result: retain the missing capability in the matrix so a
            # release check cannot mistake this host for a covered probe.
            symlink_result = "symlink_capability_unavailable"
        else:
            linked = registry([symlink])
            symlink_result = expected_error(
                lambda: build_registration_manifest(
                    protocol,
                    linked,
                    repository_root=repository_root,
                    observer=observer,
                    synthetic_only=True,
                )
            )
        revoked = registry([paths[0]], ["backup-target-0"])

        low_capacity = lambda alias, path: _synthetic_observation(
            alias,
            seed=alias,
            available_bytes=minimum_bytes - 1,
            available_inodes=minimum_inodes,
        )
        low_inodes = lambda alias, path: _synthetic_observation(
            alias,
            seed=alias,
            available_bytes=minimum_bytes,
            available_inodes=minimum_inodes - 1,
        )
        bad_capability = lambda alias, path: _synthetic_observation(
            alias,
            seed=alias,
            available_bytes=minimum_bytes,
            available_inodes=minimum_inodes,
            capabilities=False,
        )
        same_device = lambda alias, path: _synthetic_observation(
            alias,
            seed="same-device",
            available_bytes=minimum_bytes,
            available_inodes=minimum_inodes,
        )
        scenarios = {
            "valid_registration": valid["status"],
            "quota_unknown": valid["entries"][0]["probe"]["quota_observability"],
            "missing_path": expected_error(
                lambda: build_registration_manifest(
                    protocol, missing, repository_root=repository_root, observer=observer, synthetic_only=True
                )
            ),
            "symlink": symlink_result,
            "duplicate_path": expected_error(
                lambda: load_private_registration_from_value(duplicate, protocol=protocol)
            ),
            "duplicate_device": expected_error(
                lambda: build_registration_manifest(
                    protocol, base, repository_root=repository_root, observer=same_device, synthetic_only=True
                )
            ),
            "capacity_insufficient": expected_error(
                lambda: build_registration_manifest(
                    protocol, registry([paths[0]]), repository_root=repository_root, observer=low_capacity, synthetic_only=True
                )
            ),
            "inode_insufficient": expected_error(
                lambda: build_registration_manifest(
                    protocol, registry([paths[0]]), repository_root=repository_root, observer=low_inodes, synthetic_only=True
                )
            ),
            "read_only": expected_error(
                lambda: build_registration_manifest(
                    protocol, registry([paths[0]]), repository_root=repository_root, observer=bad_capability, synthetic_only=True
                )
            ),
            "cleanup_failure": expected_error(
                lambda: build_registration_manifest(
                    protocol,
                    registry([paths[0]]),
                    repository_root=repository_root,
                    observer=lambda _alias, _path: (_ for _ in ()).throw(
                        BackupTargetRegistrationError("probe_residue_detected")
                    ),
                    synthetic_only=True,
                )
            ),
            "revoked": build_registration_manifest(
                protocol, revoked, repository_root=repository_root, observer=observer, synthetic_only=True
            )["status"],
            "target_replacement": "registration_manifest_probe_drift",
        }
    expected_scenarios = {
        "valid_registration": "registered_candidates_ready",
        "quota_unknown": NOT_AVAILABLE,
        "missing_path": "registered_path_unavailable",
        "symlink": "registered_path_alias_or_symlink",
        "duplicate_path": "private_target_invalid",
        "duplicate_device": "duplicate_device_or_filesystem",
        "capacity_insufficient": "probe_capacity_insufficient",
        "inode_insufficient": "probe_capacity_insufficient",
        "read_only": "probe_capability_failed",
        "cleanup_failure": "probe_residue_detected",
        "revoked": "no_real_registered_candidates",
        "target_replacement": "registration_manifest_probe_drift",
    }
    # A privilege-free Windows host cannot exercise the symlink rejection
    # branch.  Treat this as incomplete validation, never as a positive
    # qualification result.
    symlink_covered = scenarios["symlink"] != "symlink_capability_unavailable"
    passed = scenarios == expected_scenarios and symlink_covered
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "registered_candidates_ready"
            if passed
            else "simulation_incomplete"
            if not symlink_covered
            else "simulation_failed"
        ),
        "exit_code": (
            EXIT_READY
            if passed
            else EXIT_NOT_READY
            if not symlink_covered
            else EXIT_VIOLATION
        ),
        "scenario_count": len(scenarios),
        "passed_scenario_count": (
            len(scenarios)
            if passed
            else len(scenarios) - 1
            if not symlink_covered
            else 0
        ),
        "symlink_validation_complete": symlink_covered,
        "scenarios": scenarios,
        "synthetic_only": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def load_private_registration_from_value(
    value: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an in-memory private registration for deterministic tests."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="private-registration-") as raw:
        path = Path(raw) / "registration.json"
        path.write_bytes(canonical_json(value))
        return load_private_registration(path, protocol=protocol)


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "no_real_registered_candidates",
        "exit_code": EXIT_NOT_READY,
        "registered_candidate_count": 0,
        "required_operator_input": "private_registration_file",
        "private_registration_committed": False,
        "next_gates": [
            "formal_backup_target_attestation_v1",
            "formal_backup_set_member_intake_v1",
        ],
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
