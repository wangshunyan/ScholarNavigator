"""Offline intake and activation controls for real Full1000 backup-set members.

This layer binds independently collected target observations to one exact slot
of the frozen two-, three-, or four-member backup topology.  Synthetic
attestations exercise the controls only; they can never activate real
readiness.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_backup_compaction import build_shard_state
from scholar_agent.evaluation.formal_backup_set_topology import (
    FROZEN_PROTOCOL_SHA256 as BACKUP_SET_TOPOLOGY_PROTOCOL_SHA256,
    MINIMUM_WRITERS_PER_MEMBER,
    SUPPORTED_MEMBER_COUNTS,
    build_backup_set,
    build_topology,
    capacity_model,
    synthetic_profiles,
    verify_backup_set,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_backup_set_member_intake_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "851a113aed91f3764b16cb35a5b653debfc88426"
SLOT_CONTRACT = "backup_set_member_slot_contract_v1"
MEMBER_ATTESTATION = "backup_set_member_attestation_v1"
KIT_MANIFEST = "backup_set_member_intake_kit_manifest_v1"
REGISTRY = "backup_set_member_registry_v1"
ACTIVATION_RECEIPT = "backup_set_activation_receipt_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FIXED_ZIP_TIME = (2024, 1, 1, 0, 0, 0)
MAX_KIT_FILES = 6
MAX_KIT_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
CHALLENGE_LIFETIME_SECONDS = 86_400
ZERO_SHA256 = "0" * 64
CAPABILITIES = (
    "advisory_lock",
    "atomic_replace",
    "concurrent_writer",
    "directory_fsync",
    "empty_restore",
    "file_fsync",
    "incremental_parent_chain",
    "path_length",
    "write_verify_delete",
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class BackupSetIntakeError(RuntimeError):
    """A slot kit, member proof, registry, or activation failed closed."""


class BackupSetIntakeNotReady(BackupSetIntakeError):
    """Real qualified members do not yet close a topology."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise BackupSetIntakeError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise BackupSetIntakeError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                BackupSetIntakeError("nonfinite_json_number")
            ),
        )
    except BackupSetIntakeError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupSetIntakeError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupSetIntakeError("json_root_not_object")
    return value


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackupSetIntakeError("bound_artifact_unavailable") from exc


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupSetIntakeError("unsafe_binding_path")
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
        raise BackupSetIntakeError("unsafe_binding_path")
    return value


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def _runtime(repository_root: Path) -> Any:
    path = repository_root / "scripts/formal_backup_set_intake_runtime.py"
    spec = importlib.util.spec_from_file_location("_backup_set_intake_runtime", path)
    if spec is None or spec.loader is None:
        raise BackupSetIntakeError("kit_runtime_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "execution",
        "formal_validation_complete",
        "kit",
        "policy",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
    }:
        raise BackupSetIntakeError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"]
        != stable_hash(_without(value, "protocol_sha256"))
    ):
        raise BackupSetIntakeError("protocol_identity_invalid")
    if value["kit"] != {
        "challenge_lifetime_seconds": CHALLENGE_LIFETIME_SECONDS,
        "fixed_zip_timestamp": list(FIXED_ZIP_TIME),
        "network_request_count": 0,
        "project_dependency_count": 0,
        "python_invocation": "python -I -S",
    }:
        raise BackupSetIntakeError("kit_policy_invalid")
    if value["policy"] != {
        "ability_default_enabled": False,
        "activation_requires_all_slots": True,
        "complete_set_required_for_restore": True,
        "identity_authentication": False,
        "member_counts": list(SUPPORTED_MEMBER_COUNTS),
        "member_redundancy_claimed": False,
        "one_time_challenge": True,
        "synthetic_member_can_activate_real_set": False,
        "unknown_evidence_policy": "fail_closed",
    }:
        raise BackupSetIntakeError("protocol_policy_invalid")
    expected_bindings = {
        "backup_compaction",
        "backup_set_topology",
        "backup_target_attestation",
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "portable_execution_site",
    }
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise BackupSetIntakeError("binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BackupSetIntakeError("binding_schema_invalid")
        target = repository_root / _safe_relative(row["path"])
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise BackupSetIntakeError("binding_hash_drift")
    return value


def build_slot_contract(
    protocol: Mapping[str, Any],
    *,
    member_count: int,
    slot: int,
    challenge_id: str,
    issued_epoch: int,
) -> dict[str, Any]:
    if (
        member_count not in SUPPORTED_MEMBER_COUNTS
        or isinstance(slot, bool)
        or not isinstance(slot, int)
        or slot not in range(member_count)
        or len(challenge_id) != 64
        or any(character not in "0123456789abcdef" for character in challenge_id)
        or isinstance(issued_epoch, bool)
        or not isinstance(issued_epoch, int)
        or issued_epoch < 0
    ):
        raise BackupSetIntakeError("slot_contract_input_invalid")
    topology = build_topology(protocol=_topology_protocol(protocol), member_count=member_count)
    member = topology["members"][slot]
    value: dict[str, Any] = {
        "contract": SLOT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_sha256": protocol["bindings"]["execution_plan"]["sha256"],
        "topology_sha256": topology["topology_sha256"],
        "member_count": member_count,
        "slot": slot,
        "allowed_shards": list(member["assigned_shards"]),
        "challenge": {
            "challenge_id": challenge_id,
            "issued_epoch": issued_epoch,
            "expires_epoch": issued_epoch + CHALLENGE_LIFETIME_SECONDS,
            "one_time": True,
        },
        "slot_requirements": {
            "minimum_available_bytes": member["required_bytes"],
            "minimum_available_inodes": member["required_inodes"],
            "minimum_quota_bytes": member["required_bytes"],
            "minimum_writers": member["required_writers"],
            "capability_names": list(CAPABILITIES),
            "failure_domain_evidence_required": True,
            "recovery_verification_required": True,
        },
        "identity_authentication": False,
        "formal_validation_complete": False,
        "execution": {
            "llm_request_count": 0,
            "network_request_count": 0,
            "snapshot_write_count": 0,
        },
    }
    value["contract_sha256"] = stable_hash(value)
    return value


def _topology_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen topology protocol without rereading project state."""
    # build_topology only consumes the frozen protocol identity digest.  The
    # intake protocol separately binds the topology file bytes.
    return {"protocol_sha256": BACKUP_SET_TOPOLOGY_PROTOCOL_SHA256}


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _manifest_self_hash(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["manifest_self_sha256"] = ZERO_SHA256
    return stable_hash(payload)


def build_slot_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    member_count: int,
    slot: int,
    challenge_id: str,
    issued_epoch: int,
    output: Path,
) -> dict[str, Any]:
    runtime_path = repository_root / "scripts/formal_backup_set_intake_runtime.py"
    runtime_bytes = runtime_path.read_bytes()
    contract = build_slot_contract(
        protocol,
        member_count=member_count,
        slot=slot,
        challenge_id=challenge_id,
        issued_epoch=issued_epoch,
    )
    files = {
        "verify.py": runtime_bytes,
        "slot_contract.json": canonical_json(contract),
        "README.txt": (
            "Run a trusted extracted verify.py with python -I -S. This kit "
            "binds one backup-set slot and contains no query text, credentials, "
            "environment values, host identity, or absolute paths. Hashes prove "
            "content integrity, not device ownership or operator identity.\n"
        ).encode(),
    }
    inventory = [
        {
            "path": name,
            "role": (
                "portable_verifier"
                if name == "verify.py"
                else "slot_contract"
                if name == "slot_contract.json"
                else "operator_instructions"
            ),
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for name, content in sorted(files.items())
    ]
    manifest: dict[str, Any] = {
        "manifest": KIT_MANIFEST,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "member_count": member_count,
        "slot": slot,
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "files": inventory,
        "manifest_self_sha256": ZERO_SHA256,
        "formal_validation_complete": False,
    }
    manifest["manifest_self_sha256"] = _manifest_self_hash(manifest)
    files["manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(_zip_info(name), content)
    os.replace(temporary, output)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "slot_kit_ready",
        "exit_code": EXIT_READY,
        "member_count": member_count,
        "slot": slot,
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "kit_sha256": file_sha256(output),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and ".." not in path.parts
        and str(path) == name
    )


def read_kit(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(infos) > MAX_KIT_FILES
                or len(names) != len(set(names))
                or any(not _safe_member(name) for name in names)
                or any(info.compress_type != zipfile.ZIP_STORED for info in infos)
                or sum(info.file_size for info in infos) > MAX_KIT_BYTES
                or any(info.file_size != info.compress_size for info in infos)
                or any(
                    ((info.external_attr >> 16) & 0o170000) not in (0, 0o100000)
                    for info in infos
                )
            ):
                raise BackupSetIntakeError("kit_archive_invalid")
            files = {info.filename: archive.read(info) for info in infos}
    except BackupSetIntakeError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BackupSetIntakeError("kit_archive_invalid") from exc
    if set(files) != {
        "README.txt",
        "manifest.json",
        "slot_contract.json",
        "verify.py",
    }:
        raise BackupSetIntakeError("kit_inventory_invalid")
    try:
        manifest = json.loads(
            files["manifest.json"].decode(),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupSetIntakeError("kit_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise BackupSetIntakeError("kit_manifest_invalid")
    return manifest, files


def verify_slot_kit(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    manifest, files = read_kit(path)
    if (
        set(manifest)
        != {
            "challenge_id",
            "contract_sha256",
            "files",
            "formal_validation_complete",
            "manifest",
            "manifest_self_sha256",
            "member_count",
            "protocol",
            "schema_version",
            "slot",
            "source_commit",
        }
        or manifest["manifest"] != KIT_MANIFEST
        or manifest["protocol"] != PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["formal_validation_complete"] is not False
        or manifest["manifest_self_sha256"] != _manifest_self_hash(manifest)
    ):
        raise BackupSetIntakeError("kit_manifest_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise BackupSetIntakeError("kit_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "role", "sha256", "size"}
            or row["path"] in seen
            or row["path"] not in files
            or row["path"] == "manifest.json"
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != hashlib.sha256(files[row["path"]]).hexdigest()
        ):
            raise BackupSetIntakeError("kit_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise BackupSetIntakeError("kit_inventory_invalid")
    runtime_bytes = (
        repository_root / "scripts/formal_backup_set_intake_runtime.py"
    ).read_bytes()
    if files["verify.py"] != runtime_bytes:
        raise BackupSetIntakeError("kit_runtime_binding_invalid")
    try:
        contract = json.loads(
            files["slot_contract.json"].decode(),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupSetIntakeError("slot_contract_invalid") from exc
    runtime = _runtime(repository_root)
    try:
        validated = runtime.validate_contract(contract)
    except runtime.IntakeRuntimeError as exc:
        raise BackupSetIntakeError(str(exc)) from exc
    expected = build_slot_contract(
        protocol,
        member_count=manifest["member_count"],
        slot=manifest["slot"],
        challenge_id=manifest["challenge_id"],
        issued_epoch=validated["challenge"]["issued_epoch"],
    )
    if validated != expected or manifest["contract_sha256"] != expected["contract_sha256"]:
        raise BackupSetIntakeError("slot_contract_binding_invalid")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "slot_kit_verified",
        "exit_code": EXIT_READY,
        "member_count": manifest["member_count"],
        "slot": manifest["slot"],
        "challenge_id": manifest["challenge_id"],
        "contract_sha256": manifest["contract_sha256"],
        "kit_sha256": file_sha256(path),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def synthetic_member_attestation(
    contract: Mapping[str, Any],
    *,
    identity_seed: str,
    observation_epoch: int,
    synthetic_only: bool = True,
) -> dict[str, Any]:
    requirements = contract["slot_requirements"]
    identity = lambda role: stable_hash({"seed": identity_seed, "role": role})
    observations = {
        "available_bytes": requirements["minimum_available_bytes"],
        "available_inodes": requirements["minimum_available_inodes"],
        "device_identity": identity("device"),
        "evidence_expires_epoch": contract["challenge"]["expires_epoch"],
        "evidence_type": "independent_physical_device_and_management_domain",
        "failure_domain_identity": identity("failure_domain"),
        "filesystem_identity": identity("filesystem"),
        "management_domain_identity": identity("management_domain"),
        "observation_epoch": observation_epoch,
        "primary_failure_domain_identity": identity("primary_failure_domain"),
        "quota_bytes": requirements["minimum_quota_bytes"],
        "quota_pool_identity": identity("quota_pool"),
        "storage_service_identity": NOT_AVAILABLE,
        "writers": requirements["minimum_writers"],
    }
    target_identity = stable_hash(
        {
            key: observations[key]
            for key in (
                "device_identity",
                "failure_domain_identity",
                "filesystem_identity",
                "management_domain_identity",
                "quota_pool_identity",
                "storage_service_identity",
            )
        }
    )
    checks = {
        "available_bytes": True,
        "available_inodes": True,
        "failure_domain_independent": True,
        "fresh": True,
        "quota": True,
        "recovery": True,
        "writers": True,
    }
    value: dict[str, Any] = {
        "attestation": MEMBER_ATTESTATION,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "contract_sha256": contract["contract_sha256"],
        "challenge_id": contract["challenge"]["challenge_id"],
        "member_count": contract["member_count"],
        "slot": contract["slot"],
        "target_identity": target_identity,
        "observations": observations,
        "capabilities": {name: True for name in CAPABILITIES},
        "checks": checks,
        "recovery_verified": True,
        "revoked": False,
        "status": "backup_set_member_qualified",
        "synthetic_only": synthetic_only,
        "identity_authentication": False,
        "formal_validation_complete": False,
    }
    value["attestation_sha256"] = stable_hash(value)
    return value


def validate_member(
    repository_root: Path,
    contract: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    observation_epoch: int,
    require_real: bool,
) -> dict[str, Any]:
    runtime = _runtime(repository_root)
    try:
        return runtime.validate_attestation(
            dict(attestation),
            dict(contract),
            observation_epoch=observation_epoch,
            require_real=require_real,
        )
    except runtime.IntakeRuntimeError as exc:
        raise BackupSetIntakeError(str(exc)) from exc


def _event(
    events: Sequence[Mapping[str, Any]],
    *,
    member_count: int,
    slot: int,
    state_before: str,
    state_after: str,
    contract: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sequence": len(events),
        "member_count": member_count,
        "slot": slot,
        "state_before": state_before,
        "state_after": state_after,
        "challenge_id": contract["challenge"]["challenge_id"],
        "contract_sha256": contract["contract_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "target_identity": attestation["target_identity"],
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else ZERO_SHA256
        ),
    }
    value["event_sha256"] = stable_hash(value)
    return value


def verify_registry(events: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], str]:
    states: dict[tuple[int, int], str] = {}
    previous = ZERO_SHA256
    challenges: set[str] = set()
    for index, row in enumerate(events):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "attestation_sha256",
                "challenge_id",
                "contract_sha256",
                "event_sha256",
                "member_count",
                "previous_event_sha256",
                "sequence",
                "slot",
                "state_after",
                "state_before",
                "target_identity",
            }
            or row["sequence"] != index
            or row["previous_event_sha256"] != previous
            or stable_hash(_without(row, "event_sha256")) != row["event_sha256"]
        ):
            raise BackupSetIntakeError("registry_chain_invalid")
        key = (row["member_count"], row["slot"])
        before = states.get(key, "empty")
        if row["state_before"] != before:
            raise BackupSetIntakeError("registry_transition_invalid")
        allowed = {
            "empty": {"qualified"},
            "qualified": {"reserved", "revoked", "invalid"},
            "reserved": {"set_activated", "revoked", "invalid"},
            "set_activated": {"revoked", "invalid"},
            "revoked": set(),
            "invalid": set(),
        }
        if row["state_after"] not in allowed.get(before, set()):
            raise BackupSetIntakeError("registry_transition_invalid")
        if before == "empty":
            if row["challenge_id"] in challenges:
                raise BackupSetIntakeError("challenge_replay")
            challenges.add(row["challenge_id"])
        states[key] = row["state_after"]
        previous = row["event_sha256"]
    return states


def import_member(
    repository_root: Path,
    events: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    observation_epoch: int,
    require_real: bool,
) -> list[dict[str, Any]]:
    states = verify_registry(events)
    key = (contract["member_count"], contract["slot"])
    if states.get(key, "empty") != "empty":
        raise BackupSetIntakeError("slot_already_consumed")
    if any(
        event["challenge_id"] == contract["challenge"]["challenge_id"]
        for event in events
    ):
        raise BackupSetIntakeError("challenge_replay")
    validated = validate_member(
        repository_root,
        contract,
        attestation,
        observation_epoch=observation_epoch,
        require_real=require_real,
    )
    updated = [dict(event) for event in events]
    updated.append(
        _event(
            updated,
            member_count=contract["member_count"],
            slot=contract["slot"],
            state_before="empty",
            state_after="qualified",
            contract=contract,
            attestation=validated,
        )
    )
    updated.append(
        _event(
            updated,
            member_count=contract["member_count"],
            slot=contract["slot"],
            state_before="qualified",
            state_after="reserved",
            contract=contract,
            attestation=validated,
        )
    )
    verify_registry(updated)
    return updated


def _verify_unique_members(attestations: Sequence[Mapping[str, Any]]) -> None:
    fields = {
        "target_identity": [],
        "device_identity": [],
        "filesystem_identity": [],
        "quota_pool_identity": [],
        "failure_domain_identity": [],
        "management_domain_identity": [],
    }
    for row in attestations:
        fields["target_identity"].append(row["target_identity"])
        for field in fields.keys() - {"target_identity"}:
            fields[field].append(row["observations"][field])
    for field, values in fields.items():
        if len(values) != len(set(values)):
            raise BackupSetIntakeError(f"duplicate_member_{field}")


def activate_set(
    repository_root: Path,
    protocol: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    contracts: Sequence[Mapping[str, Any]],
    attestations: Sequence[Mapping[str, Any]],
    *,
    member_count: int,
    observation_epoch: int,
    require_real: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = verify_registry(events)
    if len(contracts) != member_count or len(attestations) != member_count:
        raise BackupSetIntakeNotReady("required_slots_missing")
    by_slot = {row["slot"]: row for row in contracts}
    attest_by_slot = {row["slot"]: row for row in attestations}
    if set(by_slot) != set(range(member_count)) or set(attest_by_slot) != set(
        range(member_count)
    ):
        raise BackupSetIntakeNotReady("required_slots_missing")
    topology = build_topology(_topology_protocol(protocol), member_count=member_count)
    validated: list[dict[str, Any]] = []
    for slot in range(member_count):
        contract = by_slot[slot]
        if (
            contract["topology_sha256"] != topology["topology_sha256"]
            or states.get((member_count, slot)) != "reserved"
        ):
            raise BackupSetIntakeError("slot_or_topology_drift")
        validated.append(
            validate_member(
                repository_root,
                contract,
                attest_by_slot[slot],
                observation_epoch=observation_epoch,
                require_real=require_real,
            )
        )
    _verify_unique_members(validated)
    updated = [dict(event) for event in events]
    for slot in range(member_count):
        updated.append(
            _event(
                updated,
                member_count=member_count,
                slot=slot,
                state_before="reserved",
                state_after="set_activated",
                contract=by_slot[slot],
                attestation=attest_by_slot[slot],
            )
        )
    verify_registry(updated)
    receipt: dict[str, Any] = {
        "receipt": ACTIVATION_RECEIPT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_sha256": protocol["bindings"]["execution_plan"]["sha256"],
        "topology_sha256": topology["topology_sha256"],
        "member_count": member_count,
        "members": [
            {
                "slot": slot,
                "attestation_sha256": attest_by_slot[slot]["attestation_sha256"],
                "target_identity": attest_by_slot[slot]["target_identity"],
                "assigned_shards": topology["members"][slot]["assigned_shards"],
            }
            for slot in range(member_count)
        ],
        "capacity_model_sha256": stable_hash(capacity_model(member_count)),
        "registry_tip_sha256": updated[-1]["event_sha256"],
        "recovery_command": (
            "python scripts/check_formal_backup_set.py verify-set"
        ),
        "launch_addendum": {
            "backup_set_activation_required": True,
            "receipt_sha256_binding_required": True,
            "complete_set_required": True,
        },
        "identity_authentication": False,
        "synthetic_only": any(row["synthetic_only"] for row in validated),
        "formal_validation_complete": False,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return updated, receipt


def verify_activation(
    repository_root: Path,
    protocol: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    attestations: Sequence[Mapping[str, Any]],
    *,
    observation_epoch: int,
    require_real: bool,
) -> None:
    if (
        set(receipt)
        != {
            "capacity_model_sha256",
            "formal_validation_complete",
            "identity_authentication",
            "launch_addendum",
            "member_count",
            "members",
            "plan_sha256",
            "protocol",
            "protocol_sha256",
            "receipt",
            "receipt_sha256",
            "recovery_command",
            "schema_version",
            "source_commit",
            "synthetic_only",
            "topology_sha256",
            "registry_tip_sha256",
        }
        or receipt["receipt"] != ACTIVATION_RECEIPT
        or receipt["protocol"] != PROTOCOL
        or receipt["source_commit"] != SOURCE_COMMIT
        or stable_hash(_without(receipt, "receipt_sha256"))
        != receipt["receipt_sha256"]
        or receipt["protocol_sha256"] != protocol["protocol_sha256"]
        or receipt["plan_sha256"]
        != protocol["bindings"]["execution_plan"]["sha256"]
        or (require_real and receipt["synthetic_only"] is not False)
    ):
        raise BackupSetIntakeError("activation_receipt_invalid")
    count = receipt["member_count"]
    if count not in SUPPORTED_MEMBER_COUNTS:
        raise BackupSetIntakeError("activation_receipt_invalid")
    topology = build_topology(_topology_protocol(protocol), member_count=count)
    if (
        receipt["topology_sha256"] != topology["topology_sha256"]
        or receipt["capacity_model_sha256"] != stable_hash(capacity_model(count))
    ):
        raise BackupSetIntakeError("activation_topology_drift")
    attest_by_slot = {row["slot"]: row for row in attestations}
    for contract in contracts:
        slot = contract["slot"]
        validated = validate_member(
            repository_root,
            contract,
            attest_by_slot[slot],
            observation_epoch=observation_epoch,
            require_real=require_real,
        )
        member = receipt["members"][slot]
        if (
            member["slot"] != slot
            or member["attestation_sha256"] != validated["attestation_sha256"]
            or member["target_identity"] != validated["target_identity"]
            or member["assigned_shards"]
            != topology["members"][slot]["assigned_shards"]
        ):
            raise BackupSetIntakeError("activation_member_drift")
    _verify_unique_members(list(attest_by_slot.values()))


def _expected_error(call: Any) -> str:
    try:
        call()
    except (BackupSetIntakeError, BackupSetIntakeNotReady) as exc:
        return str(exc)
    raise BackupSetIntakeError("expected_violation_not_detected")


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    issued = 10_000
    count = 4
    contracts = [
        build_slot_contract(
            protocol,
            member_count=count,
            slot=slot,
            challenge_id=stable_hash({"slot": slot, "matrix": PROTOCOL}),
            issued_epoch=issued,
        )
        for slot in range(count)
    ]
    attestations = [
        synthetic_member_attestation(
            contract,
            identity_seed=f"member-{slot}",
            observation_epoch=issued,
        )
        for slot, contract in enumerate(contracts)
    ]
    events: list[dict[str, Any]] = []
    for contract, attestation in zip(contracts, attestations, strict=True):
        events = import_member(
            repository_root,
            events,
            contract,
            attestation,
            observation_epoch=issued,
            require_real=False,
        )
    activated_events, receipt = activate_set(
        repository_root,
        protocol,
        events,
        contracts,
        attestations,
        member_count=count,
        observation_epoch=issued,
        require_real=False,
    )
    verify_activation(
        repository_root,
        protocol,
        receipt,
        contracts,
        attestations,
        observation_epoch=issued,
        require_real=False,
    )
    topology = build_topology(_topology_protocol(protocol), member_count=count)
    profiles = synthetic_profiles(topology)
    states = {
        shard: build_shard_state(shard, cursor=50, generation=1)
        for shard in range(20)
    }
    set_manifest, archives = build_backup_set(topology, states, profiles)
    restored = verify_backup_set(topology, set_manifest, archives, profiles)

    def changed(row: Mapping[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
        result = copy.deepcopy(dict(row))
        target: Any = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        result["attestation_sha256"] = stable_hash(
            _without(result, "attestation_sha256")
        )
        return result

    duplicate_device = changed(
        attestations[1],
        ("observations", "device_identity"),
        attestations[0]["observations"]["device_identity"],
    )
    duplicate_device["target_identity"] = stable_hash(
        {
            key: duplicate_device["observations"][key]
            for key in (
                "device_identity",
                "failure_domain_identity",
                "filesystem_identity",
                "management_domain_identity",
                "quota_pool_identity",
                "storage_service_identity",
            )
        }
    )
    duplicate_device["attestation_sha256"] = stable_hash(
        _without(duplicate_device, "attestation_sha256")
    )
    quota_overlap = changed(
        attestations[1],
        ("observations", "quota_pool_identity"),
        attestations[0]["observations"]["quota_pool_identity"],
    )
    quota_overlap["target_identity"] = stable_hash(
        {
            key: quota_overlap["observations"][key]
            for key in (
                "device_identity",
                "failure_domain_identity",
                "filesystem_identity",
                "management_domain_identity",
                "quota_pool_identity",
                "storage_service_identity",
            )
        }
    )
    quota_overlap["attestation_sha256"] = stable_hash(
        _without(quota_overlap, "attestation_sha256")
    )
    scenarios = [
        {"scenario": "sequential_import_and_activation", "status": "passed"},
        {
            "scenario": "partial_members",
            "status": _expected_error(
                lambda: activate_set(
                    repository_root,
                    protocol,
                    events[:-2],
                    contracts[:-1],
                    attestations[:-1],
                    member_count=count,
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "duplicate_device",
            "status": _expected_error(
                lambda: _verify_unique_members(
                    [attestations[0], duplicate_device, *attestations[2:]]
                )
            ),
        },
        {
            "scenario": "quota_overlap",
            "status": _expected_error(
                lambda: _verify_unique_members(
                    [attestations[0], quota_overlap, *attestations[2:]]
                )
            ),
        },
        {
            "scenario": "slot_mismatch",
            "status": _expected_error(
                lambda: validate_member(
                    repository_root,
                    contracts[1],
                    attestations[0],
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "challenge_replay",
            "status": _expected_error(
                lambda: import_member(
                    repository_root,
                    events,
                    contracts[0],
                    attestations[0],
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "member_replacement",
            "status": _expected_error(
                lambda: verify_activation(
                    repository_root,
                    protocol,
                    receipt,
                    contracts,
                    [
                        changed(
                            attestations[0],
                            ("observations", "filesystem_identity"),
                            stable_hash({"replacement": True}),
                        ),
                        *attestations[1:],
                    ],
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "expired_member",
            "status": _expected_error(
                lambda: verify_activation(
                    repository_root,
                    protocol,
                    receipt,
                    contracts,
                    attestations,
                    observation_epoch=issued + CHALLENGE_LIFETIME_SECONDS + 1,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "revoked_member",
            "status": _expected_error(
                lambda: validate_member(
                    repository_root,
                    contracts[0],
                    changed(attestations[0], ("revoked",), True),
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "post_activation_capacity_drift",
            "status": _expected_error(
                lambda: verify_activation(
                    repository_root,
                    protocol,
                    receipt,
                    contracts,
                    [
                        changed(
                            attestations[0],
                            ("observations", "available_bytes"),
                            0,
                        ),
                        *attestations[1:],
                    ],
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "cross_commit_reuse",
            "status": _expected_error(
                lambda: validate_member(
                    repository_root,
                    contracts[0],
                    changed(
                        attestations[0],
                        ("source_commit",),
                        "0" * 40,
                    ),
                    observation_epoch=issued,
                    require_real=False,
                )
            ),
        },
        {
            "scenario": "complete_recovery_rehearsal",
            "status": (
                "passed"
                if len(restored) == 20
                and sum(row["query_cursor"] for row in restored.values()) == 1000
                else "failed"
            ),
        },
    ]
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_set_activated",
        "exit_code": EXIT_READY,
        "scenario_count": len(scenarios),
        "passed_count": sum(row["status"] != "failed" for row in scenarios),
        "scenarios": scenarios,
        "registry_event_count": len(activated_events),
        "activation_receipt_sha256": receipt["receipt_sha256"],
        "recovery": {
            "query_count": 1000,
            "shard_count": 20,
            "duplicate_request_count": 0,
            "ledger_conserved": True,
        },
        "real_member_count": 0,
        "formal_validation_complete": False,
        "formal_blockers": [
            "full1000_incomplete",
            "human_precision_missing",
            "official_scorer_schema_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    plans = []
    for count in SUPPORTED_MEMBER_COUNTS:
        model = capacity_model(count)
        plans.append(
            {
                "member_count": count,
                "qualified_real_slot_count": 0,
                "missing_real_slot_count": count,
                "slots": [
                    {
                        "slot": row["member_index"],
                        "required_bytes": row["required_bytes"],
                        "required_inodes": row["required_inodes"],
                        "required_quota_bytes": row["required_bytes"],
                        "status": "not_available",
                    }
                    for row in model["members"]
                ],
                "activation_status": "not_ready_missing_real_members",
            }
        )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_real_members",
        "exit_code": EXIT_NOT_READY,
        "reason_code": "no_fresh_real_backup_set_member_attestations",
        "plans": plans,
        "qualified_real_member_count": 0,
        "synthetic_member_accepted_as_real": False,
        "activation_receipt": NOT_AVAILABLE,
        "launch_preflight_recheck": "blocked_missing_backup_set_activation",
        "disaster_recovery_recheck": "blocked_missing_backup_set_activation",
        "full1000_run_started": False,
        "formal_validation_complete": False,
        "formal_blockers": [
            "full1000_incomplete",
            "human_precision_missing",
            "official_scorer_schema_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
    }
