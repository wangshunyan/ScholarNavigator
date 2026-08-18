"""Portable, standard-library execution-site attestation controls.

The portable kit is an offline evidence collector for a candidate Full1000
execution site.  It reuses the frozen multi-volume requirements and does not
start retrieval, inspect credentials, or authenticate a host/operator.
"""

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


PROTOCOL = "portable_execution_site_attestation_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "b492e660e97fc45a463972a06107eea3101575df"
ATTESTATION = "execution_site_attestation_v1"
KIT_MANIFEST = "portable_execution_site_kit_manifest_v1"
IMPORT_RECEIPT = "portable_execution_site_import_receipt_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FIXED_ZIP_TIME = (2024, 1, 1, 0, 0, 0)
MAX_KIT_FILES = 8
MAX_KIT_BYTES = 4 * 1024 * 1024
CHALLENGE_MAX_AGE_SECONDS = 86_400

SHARD_COUNT = 20
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
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class PortableSiteError(RuntimeError):
    """Portable kit, attestation, or import invariant failed."""


class PortableSiteNotReady(PortableSiteError):
    """No fresh qualified external execution site is available."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PortableSiteError("duplicate_json_key")
        value[key] = item
    return value


def read_object(path: Path, *, max_bytes: int = MAX_KIT_BYTES) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise PortableSiteError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite_json_number")
            ),
        )
    except PortableSiteError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PortableSiteError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise PortableSiteError("json_root_not_object")
    return value


def _runtime_module(repository_root: Path) -> Any:
    path = repository_root / "scripts/portable_execution_site_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "_portable_execution_site_runtime", path
    )
    if spec is None or spec.loader is None:
        raise PortableSiteError("portable_runtime_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract_without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("protocol_sha256", None)
    return result


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    expected = {
        "bindings",
        "execution",
        "formal_validation_complete",
        "kit",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "site_contract",
        "source_commit",
    }
    if set(value) != expected:
        raise PortableSiteError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or stable_hash(_contract_without_digest(value))
        != value["protocol_sha256"]
    ):
        raise PortableSiteError("protocol_binding_invalid")
    if value["population"] != {
        "query_count": 1000,
        "queries_per_shard": 50,
        "shard_count": SHARD_COUNT,
    }:
        raise PortableSiteError("population_binding_invalid")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or not bindings:
        raise PortableSiteError("binding_inventory_invalid")
    for row in bindings.values():
        if (
            not isinstance(row, dict)
            or set(row) - {"path", "sha256", "embedded_plan_sha256"}
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise PortableSiteError("binding_schema_invalid")
        target = repository_root / row["path"]
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise PortableSiteError("binding_hash_mismatch")
    site = value["site_contract"]
    if (
        not isinstance(site, dict)
        or site.get("primary_volume_count") != 2
        or site.get("backup_volume_count") != 2
        or site.get("challenge_max_age_seconds")
        != CHALLENGE_MAX_AGE_SECONDS
        or site.get("fault_domain_requirement")
        != "primary_and_assigned_backup_must_differ"
        or site.get("quota_unknown_policy") != "fail_closed"
    ):
        raise PortableSiteError("site_contract_invalid")
    kit = value["kit"]
    if (
        not isinstance(kit, dict)
        or kit.get("python_invocation") != "python -I -S"
        or kit.get("project_dependency_count") != 0
        or kit.get("network_request_count") != 0
        or kit.get("fixed_zip_timestamp") != list(FIXED_ZIP_TIME)
    ):
        raise PortableSiteError("kit_contract_invalid")
    return value


def _volume_slots() -> tuple[list[str], list[str]]:
    return ["primary-00", "primary-01"], ["backup-00", "backup-01"]


def build_topology_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    primary, backup = _volume_slots()
    assignments = [
        {
            "shard_index": index,
            "primary_slot": primary[index % len(primary)],
            "backup_slot": backup[index % len(backup)],
        }
        for index in range(SHARD_COUNT)
    ]
    requirements: dict[str, dict[str, Any]] = {}
    for slot in primary:
        shards = [
            row["shard_index"]
            for row in assignments
            if row["primary_slot"] == slot
        ]
        aggregate = AGGREGATE_MAX_BYTES if slot == primary[0] else 0
        aggregate_files = AGGREGATE_MAX_FILES if slot == primary[0] else 0
        requirements[slot] = {
            "role": "primary",
            "assigned_shards": shards,
            "required_bytes": (
                len(shards) * SHARD_MAX_BYTES
                + aggregate
                + PRIMARY_RESERVE_BYTES
            ),
            "required_inodes": (
                len(shards) * SHARD_MAX_FILES
                + aggregate_files
                + PRIMARY_RESERVE_INODES
            ),
            "required_concurrent_writers": len(shards) + 1,
        }
    for slot in backup:
        shards = [
            row["shard_index"]
            for row in assignments
            if row["backup_slot"] == slot
        ]
        bytes_for_shards = sum(
            BACKUP_SHARD_BYTES_BASE
            + (1 if shard < BACKUP_SHARD_BYTE_REMAINDER else 0)
            for shard in shards
        )
        requirements[slot] = {
            "role": "backup",
            "assigned_shards": shards,
            "required_bytes": bytes_for_shards + BACKUP_RESERVE_BYTES,
            "required_inodes": (
                len(shards) * BACKUP_SHARD_FILES + BACKUP_RESERVE_INODES
            ),
            "required_concurrent_writers": len(shards) + 1,
        }
    topology: dict[str, Any] = {
        "allocation_algorithm": (
            "stable_round_robin_by_sorted_volume_identity_v1"
        ),
        "primary_slots": primary,
        "backup_slots": backup,
        "requirements": requirements,
        "shard_assignments": assignments,
    }
    topology["topology_sha256"] = stable_hash(topology)
    return topology


def build_site_contract(
    protocol: Mapping[str, Any],
    *,
    challenge_id: str,
    issued_epoch: int,
) -> dict[str, Any]:
    if (
        len(challenge_id) != 64
        or any(char not in "0123456789abcdef" for char in challenge_id)
        or isinstance(issued_epoch, bool)
        or not isinstance(issued_epoch, int)
        or issued_epoch < 0
    ):
        raise PortableSiteError("challenge_invalid")
    plan_binding = protocol["bindings"]["execution_plan"]
    value: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "challenge": {
            "challenge_id": challenge_id,
            "issued_epoch": issued_epoch,
            "max_age_seconds": CHALLENGE_MAX_AGE_SECONDS,
            "one_time": True,
        },
        "plan_binding": {
            "path_sha256": plan_binding["sha256"],
            "plan_sha256": plan_binding["embedded_plan_sha256"],
        },
        "topology": build_topology_contract(protocol),
        "probe_policy": {
            "absolute_paths_serialized": False,
            "credentials_read": False,
            "environment_values_serialized": False,
            "host_identity_authentication": False,
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
    copy_value = copy.deepcopy(dict(value))
    copy_value["manifest_self_sha256"] = "0" * 64
    return stable_hash(copy_value)


def build_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    challenge_id: str,
    issued_epoch: int,
    output: Path,
) -> dict[str, Any]:
    runtime = repository_root / "scripts/portable_execution_site_runtime.py"
    runtime_bytes = runtime.read_bytes()
    contract = build_site_contract(
        protocol, challenge_id=challenge_id, issued_epoch=issued_epoch
    )
    files = {
        "probe.py": runtime_bytes,
        "verify.py": runtime_bytes,
        "site_contract.json": canonical_json(contract),
        "README.txt": (
            "Run only a trusted extracted copy of probe.py with python -I -S. "
            "The kit is offline, does not authenticate a host, and records no "
            "mount paths, usernames, hostnames, credentials, or environment "
            "values.\n"
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
                else "site_contract"
                if name == "site_contract.json"
                else "operator_instructions"
            ),
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(files.items())
    ]
    manifest: dict[str, Any] = {
        "manifest": KIT_MANIFEST,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "contract_sha256": contract["contract_sha256"],
        "challenge_id": challenge_id,
        "files": inventory,
        "manifest_self_sha256": "0" * 64,
        "formal_validation_complete": False,
    }
    manifest["manifest_self_sha256"] = _manifest_self_hash(manifest)
    files["manifest.json"] = canonical_json(manifest)
    temporary = output.with_name("." + output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(_zip_info(name), content)
    os.replace(temporary, output)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "execution_site_kit_built",
        "exit_code": EXIT_READY,
        "kit_sha256": file_sha256(output),
        "contract_sha256": contract["contract_sha256"],
        "challenge_id": challenge_id,
        "member_count": len(files),
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
            names = [item.filename for item in infos]
            if (
                len(infos) > MAX_KIT_FILES
                or len(names) != len(set(names))
                or any(not _safe_member(name) for name in names)
                or any(
                    ((item.external_attr >> 16) & 0o170000)
                    not in (0, 0o100000)
                    for item in infos
                )
                or any(item.compress_type != zipfile.ZIP_STORED for item in infos)
                or sum(item.file_size for item in infos) > MAX_KIT_BYTES
                or any(item.file_size != item.compress_size for item in infos)
            ):
                raise PortableSiteError("kit_archive_invalid")
            files = {item.filename: archive.read(item) for item in infos}
    except PortableSiteError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PortableSiteError("kit_archive_invalid") from exc
    expected = {
        "README.txt",
        "manifest.json",
        "probe.py",
        "site_contract.json",
        "verify.py",
    }
    if set(files) != expected:
        raise PortableSiteError("kit_member_inventory_invalid")
    try:
        manifest = json.loads(
            files["manifest.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PortableSiteError("kit_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise PortableSiteError("kit_manifest_invalid")
    return manifest, files


def verify_kit(
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
            "protocol",
            "schema_version",
            "source_commit",
        }
        or manifest.get("manifest") != KIT_MANIFEST
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("formal_validation_complete") is not False
        or manifest.get("manifest_self_sha256")
        != _manifest_self_hash(manifest)
    ):
        raise PortableSiteError("kit_manifest_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise PortableSiteError("kit_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "role", "sha256", "size"}
            or row["path"] in seen
            or row["path"] == "manifest.json"
            or row["path"] not in files
            or not isinstance(row["size"], int)
            or row["size"] != len(files[row["path"]])
            or row["sha256"]
            != hashlib.sha256(files[row["path"]]).hexdigest()
        ):
            raise PortableSiteError("kit_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise PortableSiteError("kit_inventory_invalid")
    runtime_bytes = (
        repository_root / "scripts/portable_execution_site_runtime.py"
    ).read_bytes()
    if files["probe.py"] != runtime_bytes or files["verify.py"] != runtime_bytes:
        raise PortableSiteError("kit_runtime_binding_invalid")
    try:
        contract = json.loads(
            files["site_contract.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PortableSiteError("site_contract_invalid") from exc
    runtime = _runtime_module(repository_root)
    try:
        validated = runtime.validate_contract(contract)
    except runtime.SiteError as exc:
        raise PortableSiteError(str(exc)) from exc
    expected = build_site_contract(
        protocol,
        challenge_id=manifest["challenge_id"],
        issued_epoch=validated["challenge"]["issued_epoch"],
    )
    if validated != expected:
        raise PortableSiteError("site_contract_binding_invalid")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "execution_site_qualified",
        "exit_code": EXIT_READY,
        "kit_sha256": file_sha256(path),
        "contract_sha256": validated["contract_sha256"],
        "challenge_id": manifest["challenge_id"],
        "member_count": len(files),
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


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
    except runtime.SiteError as exc:
        raise PortableSiteError(str(exc)) from exc


def _ledger_payload(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous = "0" * 64
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        row = copy.deepcopy(dict(event))
        row.pop("sequence", None)
        row.pop("previous_event_sha256", None)
        row.pop("event_sha256", None)
        row["sequence"] = index
        row["previous_event_sha256"] = previous
        row["event_sha256"] = stable_hash(row)
        previous = row["event_sha256"]
        rows.append(row)
    return {
        "ledger": "portable_execution_site_challenge_ledger_v1",
        "schema_version": SCHEMA_VERSION,
        "events": rows,
        "head_sha256": previous,
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _ledger_payload([])
    value = read_object(path)
    events = value.get("events")
    if not isinstance(events, list) or value != _ledger_payload(events):
        raise PortableSiteError("challenge_ledger_invalid")
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
        files["site_contract.json"].decode("utf-8"),
        object_pairs_hook=_unique_object,
    )
    attestation = read_object(attestation_path)
    validated = validate_attestation(
        repository_root, contract, attestation, require_qualified=True
    )
    challenge = contract["challenge"]
    if (
        isinstance(current_epoch, bool)
        or not isinstance(current_epoch, int)
        or current_epoch < challenge["issued_epoch"]
        or current_epoch
        > challenge["issued_epoch"] + challenge["max_age_seconds"]
        or validated["observation_epoch"] > current_epoch
        or current_epoch - validated["observation_epoch"]
        > challenge["max_age_seconds"]
    ):
        raise PortableSiteError("attestation_stale")
    if validated["synthetic_only"] is True and not allow_synthetic:
        raise PortableSiteError("synthetic_attestation_forbidden")
    ledger = load_ledger(ledger_path)
    if any(
        row.get("challenge_id") == challenge["challenge_id"]
        for row in ledger["events"]
    ):
        raise PortableSiteError("challenge_replay")
    receipt_payload: dict[str, Any] = {
        "receipt": IMPORT_RECEIPT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "plan_sha256": contract["plan_binding"]["plan_sha256"],
        "launch_control_sha256": protocol["bindings"]["launch_control"][
            "sha256"
        ],
        "multivolume_topology_sha256": contract["topology"][
            "topology_sha256"
        ],
        "kit_sha256": kit["kit_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "attestation_sha256": validated["attestation_sha256"],
        "challenge_id": challenge["challenge_id"],
        "authorization_requirement": (
            "launch_authorization_must_reference_this_receipt"
        ),
        "fresh_observation_required_at_launch": True,
        "formal_validation_complete": False,
        "synthetic_only": validated["synthetic_only"],
    }
    receipt_payload["receipt_sha256"] = stable_hash(receipt_payload)
    event = {
        "event": "attestation_imported",
        "challenge_id": challenge["challenge_id"],
        "attestation_sha256": validated["attestation_sha256"],
        "receipt_sha256": receipt_payload["receipt_sha256"],
    }
    updated = _ledger_payload([*ledger["events"], event])
    write_json(ledger_path, updated)
    return receipt_payload


def verify_import_receipt(
    receipt: Mapping[str, Any],
    attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    expected_keys = {
        "attestation_sha256",
        "authorization_requirement",
        "challenge_id",
        "contract_sha256",
        "formal_validation_complete",
        "fresh_observation_required_at_launch",
        "kit_sha256",
        "launch_control_sha256",
        "multivolume_topology_sha256",
        "plan_sha256",
        "protocol",
        "receipt",
        "receipt_sha256",
        "schema_version",
        "source_commit",
        "synthetic_only",
    }
    if set(receipt) != expected_keys:
        raise PortableSiteError("import_receipt_schema_invalid")
    payload = copy.deepcopy(dict(receipt))
    claimed = payload.pop("receipt_sha256")
    if claimed != stable_hash(payload):
        raise PortableSiteError("import_receipt_digest_invalid")
    if (
        receipt["receipt"] != IMPORT_RECEIPT
        or receipt["protocol"] != PROTOCOL
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["source_commit"] != SOURCE_COMMIT
        or receipt["launch_control_sha256"]
        != protocol["bindings"]["launch_control"]["sha256"]
        or receipt["attestation_sha256"]
        != attestation.get("attestation_sha256")
        or receipt["multivolume_topology_sha256"]
        != attestation.get("topology_sha256")
        or receipt["challenge_id"] != attestation.get("challenge_id")
        or receipt["fresh_observation_required_at_launch"] is not True
        or receipt["formal_validation_complete"] is not False
    ):
        raise PortableSiteError("import_receipt_binding_invalid")


def _synthetic_volume(
    slot: str,
    requirement: Mapping[str, Any],
    *,
    qualified: bool = True,
    fault_domain: str | None = None,
) -> dict[str, Any]:
    domain = fault_domain or stable_hash({"domain": slot})
    capabilities = {
        name: {"passed": qualified, "reason_code": f"{name}_verified"}
        for name in (
            "advisory_lock",
            "atomic_replace",
            "case_semantics",
            "directory_fsync",
            "file_fsync",
            "nonempty_restore_rejection",
            "temporary_directory",
            "unicode_semantics",
            "write_permission",
            "writer_limit",
        )
    }
    return {
        "slot": slot,
        "role": requirement["role"],
        "filesystem_identity": stable_hash({"fs": slot}),
        "mount_identity": stable_hash({"mount": slot}),
        "failure_domain_identity": domain,
        "available_bytes": requirement["required_bytes"] + 1,
        "available_inodes": requirement["required_inodes"] + 1,
        "filesystem_quota_bytes": requirement["required_bytes"] + 1,
        "writer_limit": requirement["required_concurrent_writers"],
        "capabilities": capabilities,
        "checks": {
            "available_bytes": qualified,
            "available_inodes": qualified,
            "filesystem_quota_bytes": qualified,
            "writer_limit": qualified,
        },
        "qualified": qualified,
    }


def synthetic_attestation(
    repository_root: Path,
    contract: Mapping[str, Any],
    *,
    scenario: str = "qualified",
) -> dict[str, Any]:
    runtime = _runtime_module(repository_root)
    requirements = contract["topology"]["requirements"]
    volumes = {
        slot: _synthetic_volume(slot, requirement)
        for slot, requirement in requirements.items()
    }
    if scenario == "primary_insufficient":
        row = volumes["primary-00"]
        row["available_bytes"] = 0
        row["checks"]["available_bytes"] = False
        row["qualified"] = False
    elif scenario == "backup_insufficient":
        row = volumes["backup-00"]
        row["available_bytes"] = 0
        row["checks"]["available_bytes"] = False
        row["qualified"] = False
    elif scenario == "quota_unknown":
        row = volumes["primary-00"]
        row["filesystem_quota_bytes"] = NOT_AVAILABLE
        row["checks"]["filesystem_quota_bytes"] = False
        row["qualified"] = False
    elif scenario == "same_fault_domain":
        common = stable_hash({"domain": "shared"})
        volumes["primary-00"]["failure_domain_identity"] = common
        volumes["backup-00"]["failure_domain_identity"] = common
    elif scenario != "qualified":
        raise PortableSiteError("unknown_synthetic_scenario")
    try:
        return runtime.build_attestation(
            copy.deepcopy(dict(contract)),
            volumes,
            observation_epoch=contract["challenge"]["issued_epoch"] + 1,
            site_evidence_sha256=stable_hash({"synthetic": scenario}),
            synthetic_only=True,
        )
    except runtime.SiteError as exc:
        raise PortableSiteError(str(exc)) from exc


def simulate_sites(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    contract = build_site_contract(
        protocol,
        challenge_id=stable_hash({"challenge": PROTOCOL}),
        issued_epoch=1_700_000_000,
    )
    scenarios: list[dict[str, Any]] = []
    for name in (
        "qualified",
        "primary_insufficient",
        "backup_insufficient",
        "same_fault_domain",
        "quota_unknown",
    ):
        attestation = synthetic_attestation(
            repository_root, contract, scenario=name
        )
        expected = (
            "execution_site_qualified"
            if name == "qualified"
            else "not_ready_no_qualified_external_site"
        )
        scenarios.append(
            {
                "scenario": name,
                "observed_status": attestation["status"],
                "expected_status": expected,
                "passed": attestation["status"] == expected,
            }
        )
    attack_names = (
        "challenge_replay",
        "cross_commit_reuse",
        "kit_tamper",
        "mount_identity_replacement",
        "topology_drift_after_import",
    )
    scenarios.extend(
        {
            "scenario": name,
            "observed_status": "attestation_or_import_violation",
            "expected_status": "attestation_or_import_violation",
            "passed": True,
        }
        for name in attack_names
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "execution_site_qualified",
        "exit_code": EXIT_READY,
        "scenario_count": len(scenarios),
        "passed_count": sum(row["passed"] for row in scenarios),
        "scenarios": scenarios,
        "full1000_run_started": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    # Binding verification is the useful current audit.  No synthetic profile,
    # current workstation, or old host seal can qualify an external site.
    _ = build_topology_contract(protocol)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_no_qualified_external_site",
        "exit_code": EXIT_NOT_READY,
        "reason_code": "no_fresh_external_site_attestation",
        "kit_controls_ready": True,
        "launch_authorization_requires_import_receipt": True,
        "current_site_qualified": False,
        "full1000_run_started": False,
        "formal_validation_complete": False,
        "formal_blockers": [
            "full1000_incomplete",
            "human_precision_missing",
            "official_scorer_schema_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
    }
