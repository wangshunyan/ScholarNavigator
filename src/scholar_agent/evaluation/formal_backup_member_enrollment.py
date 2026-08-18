"""End-to-end offline enrollment for Full1000 backup-set members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from scholar_agent.evaluation.formal_backup_set_member_intake import (
    build_slot_contract,
    validate_member,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_backup_member_enrollment_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "699ef5a0669ff9ca606b7df0e2f3ffcce09e5a9a"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
EXECUTION_ZERO = {"network_request_count": 0, "llm_request_count": 0,
                  "snapshot_write_count": 0}
MAX_KIT_BYTES = 2 * 1024 * 1024


class BackupMemberEnrollmentError(RuntimeError):
    pass


class BackupMemberEnrollmentNotReady(BackupMemberEnrollmentError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _unique(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise BackupMemberEnrollmentError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_KIT_BYTES:
            raise BackupMemberEnrollmentError("json_size_limit")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               BackupMemberEnrollmentError("nonfinite_json_number")))
    except BackupMemberEnrollmentError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupMemberEnrollmentError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupMemberEnrollmentError("json_root_not_object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupMemberEnrollmentError("binding_path_invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise BackupMemberEnrollmentError("binding_path_invalid")
    return value


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {"bindings", "execution", "formal_validation_complete", "kit",
                      "policy", "protocol", "protocol_sha256", "schema_version",
                      "source_commit"}:
        raise BackupMemberEnrollmentError("protocol_schema_invalid")
    payload = dict(value); claimed = payload.pop("protocol_sha256")
    if (value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION
            or value["source_commit"] != SOURCE_COMMIT
            or claimed != stable_hash(payload)
            or value["execution"] != {**EXECUTION_ZERO, "gold_or_qrels_loaded": False,
                                      "quality_metric_count": 0}
            or value["formal_validation_complete"] is not False):
        raise BackupMemberEnrollmentError("protocol_identity_invalid")
    if value["kit"] != {
        "fixed_zip_timestamp": [2024, 1, 1, 0, 0, 0],
        "project_dependency_count": 0,
        "python_invocation": "python -I -S",
        "supported_member_counts": [2, 3, 4],
    } or value["policy"] != {
        "activation_side_effect": False,
        "automatic_path_scan": False,
        "candidate_status": "member_candidate_ready_for_intake",
        "challenge_one_time": True,
        "explicit_operator_path_only": True,
        "identity_authentication": False,
        "output_absolute_paths": False,
        "unknown_evidence_policy": "fail_closed",
    }:
        raise BackupMemberEnrollmentError("protocol_policy_invalid")
    expected = {"backup_member_discovery", "backup_set_member_intake",
                "backup_set_topology", "backup_target_attestation",
                "backup_target_registration", "execution_plan", "intake_runtime",
                "target_runtime"}
    if not isinstance(value["bindings"], dict) or set(value["bindings"]) != expected:
        raise BackupMemberEnrollmentError("binding_inventory_invalid")
    for binding in value["bindings"].values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise BackupMemberEnrollmentError("binding_schema_invalid")
        relative = _safe_relative(binding["path"])
        target = repository_root / relative
        if file_sha256(target) != binding["sha256"]:
            raise BackupMemberEnrollmentError("binding_digest_mismatch")
    return value


def _runtime(repository_root: Path) -> Any:
    path = repository_root / "scripts/formal_backup_member_enrollment_runtime.py"
    spec = importlib.util.spec_from_file_location("formal_backup_member_enrollment_runtime", path)
    if spec is None or spec.loader is None:
        raise BackupMemberEnrollmentError("runtime_unavailable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _intake_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    binding = protocol["bindings"]["backup_set_member_intake"]
    return {
        "protocol_sha256": "c5355780bb65e7b37f3bc4f398d62f6ef3791d998ce7803df959531e91aa0f23",
        "bindings": {
            "execution_plan": {"sha256": protocol["bindings"]["execution_plan"]["sha256"]}
        },
    }


def build_contract(protocol: Mapping[str, Any], *, member_count: int, slot: int,
                   challenge_id: str, issued_epoch: int,
                   repository_root: Path) -> dict[str, Any]:
    slot_contract = build_slot_contract(_intake_protocol(protocol), member_count=member_count,
                                        slot=slot, challenge_id=challenge_id,
                                        issued_epoch=issued_epoch)
    value: dict[str, Any] = {
        "contract": "backup_member_enrollment_contract_v1",
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "bindings": {key: row["sha256"] for key, row in protocol["bindings"].items()},
        "slot_contract": slot_contract,
        "policy": {
            "activation_side_effect": False, "automatic_scan": False,
            "candidate_status": "member_candidate_ready_for_intake",
            "challenge_one_time": True, "explicit_path_only": True,
            "path_serialized": False, "synthetic_can_be_real": False,
            "unknown_evidence_policy": "fail_closed",
        },
        "identity_authentication": False, "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    value["contract_sha256"] = stable_hash(value)
    _runtime(repository_root).validate_contract(copy.deepcopy(value))
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (2024, 1, 1, 0, 0, 0)); info.create_system = 3
    info.external_attr = 0o100644 << 16; info.compress_type = zipfile.ZIP_STORED
    return info


def _manifest_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value); payload.pop("manifest_self_sha256", None)
    return stable_hash(payload)


def build_kit(repository_root: Path, protocol: Mapping[str, Any], *, member_count: int,
              slot: int, challenge_id: str, issued_epoch: int, output: Path) -> dict[str, Any]:
    contract = build_contract(protocol, member_count=member_count, slot=slot,
                              challenge_id=challenge_id, issued_epoch=issued_epoch,
                              repository_root=repository_root)
    files = {
        "contract.json": canonical_json(contract),
        "enroll.py": (repository_root / "scripts/formal_backup_member_enrollment_runtime.py").read_bytes(),
        "target_probe_reference.py": (repository_root / "scripts/formal_backup_target_runtime.py").read_bytes(),
        "member_intake_reference.py": (repository_root / "scripts/formal_backup_set_intake_runtime.py").read_bytes(),
    }
    manifest: dict[str, Any] = {
        "bundle": "formal_backup_member_enrollment_kit_v1",
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT, "member_count": member_count, "slot": slot,
        "contract_sha256": contract["contract_sha256"],
        "inventory": [{"path": name, "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in sorted(files.items())],
        "contains_prefilled_path": False, "project_dependency_count": 0,
        "python_invocation": "python -I -S", "manifest_self_sha256": "",
    }
    manifest["manifest_self_sha256"] = _manifest_hash(manifest)
    files["manifest.json"] = canonical_json(manifest)
    if output.exists():
        raise BackupMemberEnrollmentError("output_already_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(files.items()): archive.writestr(_zip_info(name), data)
    return {"protocol": PROTOCOL, "status": "member_enrollment_kit_built",
            "exit_code": EXIT_READY, "member_count": member_count, "slot": slot,
            "kit_sha256": file_sha256(output), "contract_sha256": contract["contract_sha256"],
            "execution": dict(EXECUTION_ZERO), "formal_validation_complete": False}


def read_kit(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        if path.stat().st_size > MAX_KIT_BYTES:
            raise BackupMemberEnrollmentError("kit_size_limit")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist(); names = [info.filename for info in infos]
            expected = {"contract.json", "enroll.py", "target_probe_reference.py",
                        "member_intake_reference.py", "manifest.json"}
            if set(names) != expected or len(names) != len(set(names)):
                raise BackupMemberEnrollmentError("kit_inventory_invalid")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise BackupMemberEnrollmentError("kit_path_invalid")
            files = {name: archive.read(name) for name in names}
    except BackupMemberEnrollmentError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise BackupMemberEnrollmentError("kit_invalid") from exc
    try: manifest = json.loads(files["manifest.json"].decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupMemberEnrollmentError("kit_manifest_invalid") from exc
    if not isinstance(manifest, dict): raise BackupMemberEnrollmentError("kit_manifest_invalid")
    return manifest, files


def verify_kit(path: Path, protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    manifest, files = read_kit(path)
    if (manifest.get("manifest_self_sha256") != _manifest_hash(manifest)
            or manifest.get("protocol") != PROTOCOL or manifest.get("source_commit") != SOURCE_COMMIT):
        raise BackupMemberEnrollmentError("kit_manifest_invalid")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise BackupMemberEnrollmentError("kit_inventory_invalid")
    expected_rows = [{"path": name, "size": len(files[name]),
                     "sha256": hashlib.sha256(files[name]).hexdigest()}
                    for name in sorted(files) if name != "manifest.json"]
    if inventory != expected_rows:
        raise BackupMemberEnrollmentError("kit_inventory_invalid")
    if (files["enroll.py"] != (repository_root / "scripts/formal_backup_member_enrollment_runtime.py").read_bytes()
            or files["target_probe_reference.py"] != (repository_root / "scripts/formal_backup_target_runtime.py").read_bytes()
            or files["member_intake_reference.py"] != (repository_root / "scripts/formal_backup_set_intake_runtime.py").read_bytes()):
        raise BackupMemberEnrollmentError("kit_runtime_drift")
    try: contract = json.loads(files["contract.json"].decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupMemberEnrollmentError("kit_contract_invalid") from exc
    _runtime(repository_root).validate_contract(contract)
    if contract["contract_sha256"] != manifest["contract_sha256"]:
        raise BackupMemberEnrollmentError("kit_contract_binding_invalid")
    return {"protocol": PROTOCOL, "status": "member_enrollment_kit_verified",
            "exit_code": EXIT_READY, "member_count": manifest["member_count"],
            "slot": manifest["slot"], "contract_sha256": contract["contract_sha256"],
            "kit_sha256": file_sha256(path), "execution": dict(EXECUTION_ZERO),
            "formal_validation_complete": False}


def contract_from_kit(path: Path, protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    verify_kit(path, protocol, repository_root=repository_root)
    _manifest, files = read_kit(path)
    return json.loads(files["contract.json"].decode("utf-8"))


def run_enrollment(repository_root: Path, contract: Mapping[str, Any], target: Path,
                   evidence: Mapping[str, Any], *, observation_epoch: int,
                   synthetic_only: bool = False,
                   observed: Mapping[str, Any] | None = None,
                   consumed_challenges: set[str] | None = None,
                   occupied_identities: set[str] | None = None) -> dict[str, Any]:
    runtime = _runtime(repository_root)
    try:
        challenge = contract["slot_contract"]["challenge"]["challenge_id"]
        if consumed_challenges is not None and challenge in consumed_challenges:
            raise BackupMemberEnrollmentError("challenge_replayed")
        occupied = occupied_identities or set()
        observed_device = observed.get("device_identity") if observed else evidence.get("target_device_identity")
        if observed_device in occupied or evidence.get("quota_pool_identity") in occupied:
            raise BackupMemberEnrollmentError("target_identity_already_occupied")
        package = runtime.enroll(copy.deepcopy(dict(contract)), target,
                                 copy.deepcopy(dict(evidence)),
                                 observation_epoch=observation_epoch,
                                 synthetic_only=synthetic_only,
                                 observed=copy.deepcopy(dict(observed)) if observed else None)
        validate_member(repository_root, contract["slot_contract"], package["member_attestation"],
                        observation_epoch=observation_epoch, require_real=not synthetic_only)
        if consumed_challenges is not None:
            consumed_challenges.add(challenge)
        return package
    except Exception as exc:
        if isinstance(exc, BackupMemberEnrollmentError): raise
        raise BackupMemberEnrollmentError(str(exc).split(":", 1)[0]) from exc


def verify_member_package(repository_root: Path, contract: Mapping[str, Any],
                          package: Mapping[str, Any], *, observation_epoch: int,
                          require_real: bool) -> dict[str, Any]:
    try:
        validated = _runtime(repository_root).validate_package(
            copy.deepcopy(dict(package)), copy.deepcopy(dict(contract)), require_real=require_real)
        validate_member(repository_root, contract["slot_contract"], validated["member_attestation"],
                        observation_epoch=observation_epoch, require_real=require_real)
        return validated
    except Exception as exc:
        raise BackupMemberEnrollmentError(str(exc).split(":", 1)[0]) from exc


def _fixture(protocol: Mapping[str, Any], repository_root: Path, target: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    challenge = stable_hash({"protocol": PROTOCOL, "scenario": "matrix"})
    contract = build_contract(protocol, member_count=4, slot=0, challenge_id=challenge,
                              issued_epoch=10_000, repository_root=repository_root)
    req = contract["slot_contract"]["slot_requirements"]
    identity = lambda role: stable_hash({"role": role, "scenario": "matrix"})
    observed = {"filesystem_identity": identity("filesystem"), "device_identity": identity("device"),
                "available_bytes": req["minimum_available_bytes"],
                "available_inodes": req["minimum_available_inodes"],
                "capabilities": {name: True for name in _runtime(repository_root).CAPABILITIES}}
    evidence = {
        "challenge_id": challenge, "evidence_type": "independent_physical_device_and_management_domain",
        "expires_epoch": 10_000 + 86_400, "maximum_file_size_bytes": 35_030_827_008,
        "primary_device_identity": identity("primary-device"),
        "primary_failure_domain_identity": identity("primary-failure"),
        "primary_filesystem_identity": identity("primary-filesystem"),
        "primary_management_domain_identity": identity("primary-management"),
        "quota_bytes": req["minimum_quota_bytes"], "quota_pool_identity": identity("quota"),
        "recovery_verified": True, "reserved_bytes": req["minimum_available_bytes"],
        "revoked": False, "storage_service_identity": "not_available",
        "target_device_identity": observed["device_identity"],
        "target_failure_domain_identity": identity("target-failure"),
        "target_filesystem_identity": observed["filesystem_identity"],
        "target_management_domain_identity": identity("target-management"),
    }
    return contract, evidence, observed


def simulate_matrix(protocol: Mapping[str, Any], *, repository_root: Path,
                    temporary_root: Path) -> dict[str, Any]:
    target = (temporary_root / "target").resolve(); target.mkdir(parents=True)
    contract, evidence, observed = _fixture(protocol, repository_root, target)
    consumed: set[str] = set()
    valid = run_enrollment(repository_root, contract, target, evidence,
                           observation_epoch=10_100, synthetic_only=True, observed=observed,
                           consumed_challenges=consumed)
    verify_member_package(repository_root, contract, valid, observation_epoch=10_100,
                          require_real=False)
    scenarios: list[dict[str, Any]] = [{"scenario": "legal_enrollment", "passed": True}]
    def rejected(name: str, mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]) -> None:
        c, e, o = copy.deepcopy(contract), copy.deepcopy(evidence), copy.deepcopy(observed)
        mutate(c, e, o)
        try: run_enrollment(repository_root, c, target, e, observation_epoch=10_100,
                            synthetic_only=True, observed=o)
        except BackupMemberEnrollmentError: scenarios.append({"scenario": name, "passed": True})
        else: scenarios.append({"scenario": name, "passed": False})
    rejected("quota_unknown", lambda c,e,o: e.__setitem__("quota_bytes", "not_available"))
    rejected("capacity_insufficient", lambda c,e,o: o.__setitem__("available_bytes", 0))
    rejected("inode_insufficient", lambda c,e,o: o.__setitem__("available_inodes", 0))
    rejected("failure_domain_insufficient", lambda c,e,o: e.__setitem__("target_failure_domain_identity", e["primary_failure_domain_identity"]))
    rejected("attestation_identity_tamper", lambda c,e,o: e.__setitem__("target_device_identity", stable_hash({"tamper": 1})))
    rejected("slot_mismatch", lambda c,e,o: c["slot_contract"].__setitem__("slot", 1))
    rejected("target_drift", lambda c,e,o: o.__setitem__("device_identity", stable_hash({"drift": 1})))
    rejected("revoked", lambda c,e,o: e.__setitem__("revoked", True))
    rejected("permission_failure", lambda c,e,o: o["capabilities"].__setitem__("write_verify_delete", False))
    try:
        run_enrollment(repository_root, contract, target, evidence, observation_epoch=10_100,
                       synthetic_only=True, observed=observed,
                       consumed_challenges=consumed)
    except BackupMemberEnrollmentError:
        scenarios.append({"scenario": "challenge_replay", "passed": True})
    else:
        scenarios.append({"scenario": "challenge_replay", "passed": False})
    try:
        run_enrollment(repository_root, contract, target, evidence, observation_epoch=10_100,
                       synthetic_only=True, observed=observed,
                       occupied_identities={evidence["quota_pool_identity"]})
    except BackupMemberEnrollmentError:
        scenarios.append({"scenario": "quota_pool_overlap", "passed": True})
    else:
        scenarios.append({"scenario": "quota_pool_overlap", "passed": False})
    missing = temporary_root / "missing"
    try: run_enrollment(repository_root, contract, missing, evidence, observation_epoch=10_100,
                        synthetic_only=True, observed=observed)
    except BackupMemberEnrollmentError: scenarios.append({"scenario": "path_missing", "passed": True})
    alias = temporary_root / "alias"; alias.symlink_to(target, target_is_directory=True)
    try: run_enrollment(repository_root, contract, alias, evidence, observation_epoch=10_100,
                        synthetic_only=True, observed=observed)
    except BackupMemberEnrollmentError: scenarios.append({"scenario": "same_device_alias", "passed": True})
    passed = all(row["passed"] for row in scenarios)
    return {"protocol": PROTOCOL, "schema_version": SCHEMA_VERSION,
            "status": "member_package_ready" if passed else "enrollment_or_attestation_violation",
            "exit_code": EXIT_READY if passed else EXIT_VIOLATION,
            "scenario_count": len(scenarios), "passed_count": sum(row["passed"] for row in scenarios),
            "scenarios": scenarios, "request_identity_change_count": 0,
            "activation_side_effect_count": 0, "execution": dict(EXECUTION_ZERO),
            "formal_validation_complete": False}


def audit_readiness() -> dict[str, Any]:
    plans = [{"member_count": count, "enrolled_slots": 0, "missing_slots": count,
              "status": "no_real_enrolled_members"} for count in (2, 3, 4)]
    return {"protocol": PROTOCOL, "schema_version": SCHEMA_VERSION,
            "status": "no_real_enrolled_members", "exit_code": EXIT_NOT_READY,
            "real_enrolled_member_count": 0, "plans": plans,
            "next_gate": "formal_backup_set_member_intake_v1",
            "full1000_blocker_cleared": False, "execution": dict(EXECUTION_ZERO),
            "formal_validation_complete": False}
