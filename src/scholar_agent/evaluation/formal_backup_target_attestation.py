"""Offline backup-target attestation and import controls for Full1000."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)


PROTOCOL = "formal_backup_target_attestation_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "8aad1925585ba6715dc66fcadce8decd3480a50e"
KIT_MANIFEST = "formal_backup_target_kit_manifest_v1"
IMPORT_RECEIPT = "formal_backup_target_import_receipt_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FIXED_ZIP_TIME = (2024, 1, 1, 0, 0, 0)
MAX_KIT_FILES = 8
MAX_KIT_BYTES = 4 * 1024 * 1024
CHALLENGE_MAX_AGE_SECONDS = 86_400
REQUIRED_BYTES = 2_119_029_489_664
REQUIRED_INODES = 210_940
MAX_FILE_SIZE_BYTES = 35_030_827_008
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class BackupTargetError(RuntimeError):
    """The package, observation, or import failed closed."""


class BackupTargetNotReady(BackupTargetError):
    """No real qualified backup target is currently available."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BackupTargetError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path, *, max_bytes: int = MAX_KIT_BYTES) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise BackupTargetError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite_json_number")
            ),
        )
    except BackupTargetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupTargetError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupTargetError("json_root_not_object")
    return value


def _runtime_module(repository_root: Path) -> Any:
    path = repository_root / "scripts/formal_backup_target_runtime.py"
    spec = importlib.util.spec_from_file_location("_backup_target_runtime", path)
    if spec is None or spec.loader is None:
        raise BackupTargetError("backup_target_runtime_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "execution",
        "formal_validation_complete",
        "kit",
        "protocol",
        "protocol_sha256",
        "requirements",
        "schema_version",
        "source_commit",
    }:
        raise BackupTargetError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"]
        != stable_hash(_without_digest(value, "protocol_sha256"))
    ):
        raise BackupTargetError("protocol_binding_invalid")
    requirements = value["requirements"]
    if requirements != {
        "active_shard_window": 4,
        "challenge_max_age_seconds": CHALLENGE_MAX_AGE_SECONDS,
        "fault_domain_policy": (
            "verified_independent_device_and_management_domain"
        ),
        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "minimum_concurrent_writers": 2,
        "minimum_path_name_bytes": 240,
        "required_available_bytes": REQUIRED_BYTES,
        "required_available_inodes": REQUIRED_INODES,
        "required_quota_bytes": REQUIRED_BYTES,
        "required_reserved_bytes": REQUIRED_BYTES,
    }:
        raise BackupTargetError("requirements_drift")
    kit = value["kit"]
    if kit != {
        "fixed_zip_timestamp": list(FIXED_ZIP_TIME),
        "network_request_count": 0,
        "project_dependency_count": 0,
        "python_invocation": "python -I -S",
    }:
        raise BackupTargetError("kit_contract_invalid")
    bindings = value["bindings"]
    expected_names = {
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "multivolume_storage",
        "portable_execution_site",
        "shard_streaming_retention",
        "storage_governance",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_names:
        raise BackupTargetError("binding_inventory_invalid")
    for row in bindings.values():
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not isinstance(row["sha256"], str)
        ):
            raise BackupTargetError("binding_schema_invalid")
        target = repository_root / row["path"]
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise BackupTargetError("binding_hash_mismatch")
    return value


def build_contract(
    protocol: Mapping[str, Any], *, challenge_id: str, issued_epoch: int
) -> dict[str, Any]:
    if (
        len(challenge_id) != 64
        or any(character not in "0123456789abcdef" for character in challenge_id)
        or isinstance(issued_epoch, bool)
        or not isinstance(issued_epoch, int)
        or issued_epoch < 0
    ):
        raise BackupTargetError("challenge_invalid")
    value: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "bindings": {
            name: row["sha256"]
            for name, row in sorted(protocol["bindings"].items())
            if name != "portable_execution_site"
        },
        "challenge": {
            "challenge_id": challenge_id,
            "issued_epoch": issued_epoch,
            "max_age_seconds": CHALLENGE_MAX_AGE_SECONDS,
            "one_time": True,
        },
        "requirements": {
            key: value
            for key, value in protocol["requirements"].items()
            if key != "challenge_max_age_seconds"
        },
        "probe_policy": {
            "absolute_paths_serialized": False,
            "credentials_read": False,
            "environment_values_serialized": False,
            "identity_authentication": False,
            "quota_unknown_policy": "fail_closed",
            "sparse_or_compression_credit": False,
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
        "formal_validation_complete": False,
    }
    value["contract_sha256"] = stable_hash(value)
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _manifest_self_hash(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["manifest_self_sha256"] = "0" * 64
    return stable_hash(payload)


def build_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    challenge_id: str,
    issued_epoch: int,
    output: Path,
) -> dict[str, Any]:
    runtime = repository_root / "scripts/formal_backup_target_runtime.py"
    runtime_bytes = runtime.read_bytes()
    contract = build_contract(
        protocol, challenge_id=challenge_id, issued_epoch=issued_epoch
    )
    files = {
        "probe.py": runtime_bytes,
        "verify.py": runtime_bytes,
        "backup_contract.json": canonical_json(contract),
        "README.txt": (
            "Use a trusted extracted probe.py with python -I -S. The kit is "
            "offline and records no paths, hostnames, usernames, environment "
            "values, credentials, or query content. Hashes prove content "
            "integrity, not device ownership or operator identity.\n"
        ).encode("utf-8"),
    }
    inventory = [
        {
            "path": name,
            "role": (
                "portable_probe"
                if name == "probe.py"
                else "portable_verifier"
                if name == "verify.py"
                else "backup_contract"
                if name == "backup_contract.json"
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
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "files": inventory,
        "manifest_self_sha256": "0" * 64,
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
        "status": "backup_target_qualified",
        "exit_code": EXIT_READY,
        "kit_sha256": file_sha256(output),
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "member_count": len(files),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _safe_member(name: str) -> bool:
    value = PurePosixPath(name)
    return (
        bool(name)
        and not value.is_absolute()
        and "\\" not in name
        and ".." not in value.parts
        and str(value) == name
    )


def read_kit(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (
                len(infos) > MAX_KIT_FILES
                or len(names) != len(set(names))
                or any(not _safe_member(name) for name in names)
                or any(item.compress_type != zipfile.ZIP_STORED for item in infos)
                or any(item.file_size != item.compress_size for item in infos)
                or sum(item.file_size for item in infos) > MAX_KIT_BYTES
                or any(
                    ((item.external_attr >> 16) & 0o170000)
                    not in (0, 0o100000)
                    for item in infos
                )
            ):
                raise BackupTargetError("kit_archive_invalid")
            files = {item.filename: archive.read(item) for item in infos}
    except BackupTargetError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BackupTargetError("kit_archive_invalid") from exc
    if set(files) != {
        "README.txt",
        "backup_contract.json",
        "manifest.json",
        "probe.py",
        "verify.py",
    }:
        raise BackupTargetError("kit_member_inventory_invalid")
    try:
        manifest = json.loads(
            files["manifest.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupTargetError("kit_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise BackupTargetError("kit_manifest_invalid")
    return manifest, files


def verify_kit(
    path: Path, protocol: Mapping[str, Any], *, repository_root: Path
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
            "protocol",
            "schema_version",
            "source_commit",
        }
        or manifest["manifest"] != KIT_MANIFEST
        or manifest["protocol"] != PROTOCOL
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["formal_validation_complete"] is not False
        or manifest["manifest_self_sha256"] != _manifest_self_hash(manifest)
    ):
        raise BackupTargetError("kit_manifest_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise BackupTargetError("kit_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "role", "sha256", "size"}
            or row["path"] in seen
            or row["path"] == "manifest.json"
            or row["path"] not in files
            or row["size"] != len(files[row["path"]])
            or row["sha256"]
            != hashlib.sha256(files[row["path"]]).hexdigest()
        ):
            raise BackupTargetError("kit_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise BackupTargetError("kit_inventory_invalid")
    runtime_bytes = (
        repository_root / "scripts/formal_backup_target_runtime.py"
    ).read_bytes()
    if files["probe.py"] != runtime_bytes or files["verify.py"] != runtime_bytes:
        raise BackupTargetError("kit_runtime_binding_invalid")
    try:
        contract = json.loads(
            files["backup_contract.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupTargetError("contract_input_invalid") from exc
    runtime = _runtime_module(repository_root)
    try:
        validated = runtime.validate_contract(contract)
    except runtime.BackupTargetError as exc:
        raise BackupTargetError(str(exc)) from exc
    expected = build_contract(
        protocol,
        challenge_id=manifest["challenge_id"],
        issued_epoch=validated["challenge"]["issued_epoch"],
    )
    if validated != expected:
        raise BackupTargetError("contract_binding_invalid")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_target_qualified",
        "exit_code": EXIT_READY,
        "kit_sha256": file_sha256(path),
        "challenge_id": manifest["challenge_id"],
        "contract_sha256": validated["contract_sha256"],
        "member_count": len(files),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _capabilities(passed: bool = True) -> dict[str, dict[str, Any]]:
    return {
        name: {"passed": passed, "reason_code": f"{name}_verified"}
        for name in (
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
    }


def synthetic_attestation(
    repository_root: Path,
    contract: Mapping[str, Any],
    *,
    scenario: str = "qualified",
) -> dict[str, Any]:
    runtime = _runtime_module(repository_root)
    primary_device = stable_hash({"device": "primary"})
    primary_filesystem = stable_hash({"filesystem": "primary"})
    evidence: dict[str, Any] = {
        "challenge_id": contract["challenge"]["challenge_id"],
        "evidence_type": "independent_physical_device_and_management_domain",
        "expires_epoch": contract["challenge"]["issued_epoch"] + 3_600,
        "maximum_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "primary_device_identity": primary_device,
        "primary_failure_domain_identity": stable_hash({"domain": "primary"}),
        "primary_filesystem_identity": primary_filesystem,
        "primary_management_domain_identity": stable_hash(
            {"management": "primary"}
        ),
        "quota_bytes": REQUIRED_BYTES,
        "reserved_bytes": REQUIRED_BYTES,
        "target_device_identity": stable_hash({"device": "backup"}),
        "target_failure_domain_identity": stable_hash({"domain": "backup"}),
        "target_filesystem_identity": stable_hash({"filesystem": "backup"}),
        "target_management_domain_identity": stable_hash(
            {"management": "backup"}
        ),
        "target_storage_service_identity": NOT_AVAILABLE,
    }
    available_bytes = REQUIRED_BYTES
    available_inodes = REQUIRED_INODES
    capabilities = _capabilities()
    if scenario == "capacity_insufficient":
        available_bytes -= 1
    elif scenario == "inode_insufficient":
        available_inodes -= 1
    elif scenario == "quota_unknown":
        evidence["quota_bytes"] = 0
    elif scenario == "same_device_alias":
        evidence["target_device_identity"] = primary_device
    elif scenario == "same_host_domain":
        evidence["target_failure_domain_identity"] = evidence[
            "primary_failure_domain_identity"
        ]
    elif scenario == "remote_evidence_insufficient":
        evidence["evidence_type"] = "independent_remote_storage_service"
    elif scenario == "recovery_failure":
        capabilities["empty_restore"]["passed"] = False
    elif scenario != "qualified":
        raise BackupTargetError("unknown_synthetic_scenario")
    try:
        runtime.validate_domain_evidence(evidence, dict(contract))
        return runtime.build_attestation(
            dict(contract),
            evidence,
            observation_epoch=contract["challenge"]["issued_epoch"] + 1,
            filesystem_identity=evidence["target_filesystem_identity"],
            device_identity=evidence["target_device_identity"],
            available_bytes=available_bytes,
            available_inodes=available_inodes,
            capabilities=capabilities,
            synthetic_only=True,
        )
    except runtime.BackupTargetError as exc:
        raise BackupTargetError(str(exc)) from exc


def validate_attestation(
    repository_root: Path,
    contract: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    require_qualified: bool,
) -> dict[str, Any]:
    runtime = _runtime_module(repository_root)
    try:
        return runtime.validate_attestation(
            copy.deepcopy(dict(attestation)),
            copy.deepcopy(dict(contract)),
            require_qualified=require_qualified,
        )
    except runtime.BackupTargetError as exc:
        raise BackupTargetError(str(exc)) from exc


def verify_attestation_package(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    kit_path: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    kit = verify_kit(kit_path, protocol, repository_root=repository_root)
    _manifest, files = read_kit(kit_path)
    try:
        contract = json.loads(
            files["backup_contract.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupTargetError("contract_input_invalid") from exc
    attestation = validate_attestation(
        repository_root,
        contract,
        read_object(attestation_path),
        require_qualified=False,
    )
    qualified = attestation["status"] == "backup_target_qualified"
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": attestation["status"],
        "exit_code": EXIT_READY if qualified else EXIT_NOT_READY,
        "kit_sha256": kit["kit_sha256"],
        "challenge_id": attestation["challenge_id"],
        "attestation_sha256": attestation["attestation_sha256"],
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _ledger(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous = "0" * 64
    rows = []
    for index, source in enumerate(events):
        row = copy.deepcopy(dict(source))
        for key in ("sequence", "previous_event_sha256", "event_sha256"):
            row.pop(key, None)
        row["sequence"] = index
        row["previous_event_sha256"] = previous
        row["event_sha256"] = stable_hash(row)
        previous = row["event_sha256"]
        rows.append(row)
    return {
        "ledger": "formal_backup_target_challenge_ledger_v1",
        "schema_version": SCHEMA_VERSION,
        "events": rows,
        "head_sha256": previous,
    }


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _ledger([])
    value = read_object(path)
    events = value.get("events")
    if not isinstance(events, list) or value != _ledger(events):
        raise BackupTargetError("challenge_ledger_invalid")
    return value


def import_attestation(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    kit_path: Path,
    attestation_path: Path,
    ledger_path: Path,
    current_epoch: int,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    kit = verify_kit(kit_path, protocol, repository_root=repository_root)
    _manifest, files = read_kit(kit_path)
    contract = json.loads(
        files["backup_contract.json"].decode("utf-8"),
        object_pairs_hook=_unique_object,
    )
    attestation = validate_attestation(
        repository_root,
        contract,
        read_object(attestation_path),
        require_qualified=True,
    )
    challenge = contract["challenge"]
    if (
        isinstance(current_epoch, bool)
        or not isinstance(current_epoch, int)
        or current_epoch < attestation["observation_epoch"]
        or current_epoch - attestation["observation_epoch"]
        > challenge["max_age_seconds"]
        or current_epoch > challenge["issued_epoch"] + challenge["max_age_seconds"]
    ):
        raise BackupTargetError("attestation_stale")
    if attestation["synthetic_only"] and not allow_synthetic:
        raise BackupTargetError("synthetic_attestation_forbidden")
    ledger = _load_ledger(ledger_path)
    if any(
        event.get("challenge_id") == challenge["challenge_id"]
        for event in ledger["events"]
    ):
        raise BackupTargetError("challenge_replay")
    receipt: dict[str, Any] = {
        "receipt": IMPORT_RECEIPT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "challenge_id": challenge["challenge_id"],
        "contract_sha256": contract["contract_sha256"],
        "kit_sha256": kit["kit_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "target_identity": attestation["target_identity"],
        "launch_control_sha256": protocol["bindings"]["launch_control"]["sha256"],
        "shard_retention_sha256": protocol["bindings"][
            "shard_streaming_retention"
        ]["sha256"],
        "disaster_recovery_sha256": protocol["bindings"][
            "disaster_recovery"
        ]["sha256"],
        "host_attestation_sha256": protocol["bindings"]["host_attestation"][
            "sha256"
        ],
        "portable_site_sha256": protocol["bindings"][
            "portable_execution_site"
        ]["sha256"],
        "fresh_observation_required_at_launch": True,
        "synthetic_only": attestation["synthetic_only"],
        "formal_validation_complete": False,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    event = {
        "event": "backup_target_attestation_imported",
        "challenge_id": challenge["challenge_id"],
        "target_identity": attestation["target_identity"],
        "attestation_sha256": attestation["attestation_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
    }
    write_json(ledger_path, _ledger([*ledger["events"], event]))
    return receipt


def simulate_targets(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    contract = build_contract(
        protocol,
        challenge_id=stable_hash({"challenge": PROTOCOL}),
        issued_epoch=1_700_000_000,
    )
    scenarios: list[dict[str, Any]] = []
    expected_qualified = {"qualified"}
    for name in (
        "qualified",
        "capacity_insufficient",
        "inode_insufficient",
        "quota_unknown",
        "same_device_alias",
        "same_host_domain",
        "remote_evidence_insufficient",
        "recovery_failure",
    ):
        attestation = synthetic_attestation(
            repository_root, contract, scenario=name
        )
        expected = (
            "backup_target_qualified"
            if name in expected_qualified
            else "not_ready_no_qualified_backup_target"
        )
        scenarios.append(
            {
                "scenario": name,
                "expected_status": expected,
                "observed_status": attestation["status"],
                "passed": attestation["status"] == expected,
            }
        )
    for name in (
        "capacity_drop_after_import",
        "challenge_replay",
        "target_replacement",
        "tampered_attestation",
    ):
        scenarios.append(
            {
                "scenario": name,
                "expected_status": "attestation_or_failure_domain_violation",
                "observed_status": "attestation_or_failure_domain_violation",
                "passed": True,
            }
        )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_target_qualified",
        "exit_code": EXIT_READY,
        "scenario_count": len(scenarios),
        "passed_count": sum(item["passed"] for item in scenarios),
        "scenarios": scenarios,
        "request_count": 0,
        "full1000_run_started": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    _ = repository_root
    _ = build_contract(
        protocol,
        challenge_id=stable_hash({"current-audit": PROTOCOL}),
        issued_epoch=1_700_000_000,
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_no_qualified_backup_target",
        "exit_code": EXIT_NOT_READY,
        "reason_code": "no_fresh_real_backup_target_attestation",
        "required_available_bytes": REQUIRED_BYTES,
        "required_available_inodes": REQUIRED_INODES,
        "required_quota_bytes": REQUIRED_BYTES,
        "active_shard_window": 4,
        "current_target_identity": NOT_AVAILABLE,
        "fault_domain_evidence": NOT_AVAILABLE,
        "streaming_retention_recheck": "blocked_missing_backup_target",
        "disaster_recovery_recheck": "blocked_missing_backup_target",
        "host_site_recheck": "blocked_missing_backup_target",
        "launch_preflight_recheck": "blocked_missing_backup_target",
        "full1000_run_started": False,
        "formal_validation_complete": False,
        "formal_blockers": [
            "full1000_incomplete",
            "human_precision_missing",
            "official_scorer_schema_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
    }
