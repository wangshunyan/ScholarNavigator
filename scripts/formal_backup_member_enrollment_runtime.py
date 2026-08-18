#!/usr/bin/env python3
"""Pure-standard-library runtime for a backup-member enrollment kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


PROTOCOL = "formal_backup_member_enrollment_v1"
CONTRACT = "backup_member_enrollment_contract_v1"
PACKAGE = "backup_member_candidate_package_v1"
INTAKE_PROTOCOL = "formal_backup_set_member_intake_v1"
INTAKE_ATTESTATION = "backup_set_member_attestation_v1"
INTAKE_SOURCE_COMMIT = "851a113aed91f3764b16cb35a5b653debfc88426"
SOURCE_COMMIT = "699ef5a0669ff9ca606b7df0e2f3ffcce09e5a9a"
SCHEMA_VERSION = "1"
NOT_AVAILABLE = "not_available"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CAPABILITIES = (
    "advisory_lock", "atomic_replace", "concurrent_writer",
    "directory_fsync", "empty_restore", "file_fsync",
    "incremental_parent_chain", "path_length", "write_verify_delete",
)
EVIDENCE_TYPES = {
    "independent_physical_device_and_management_domain",
    "independent_remote_storage_service",
}


class EnrollmentError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise EnrollmentError("invalid_arguments")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _unique(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise EnrollmentError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise EnrollmentError("json_size_limit")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               EnrollmentError("nonfinite_json_number")))
    except EnrollmentError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EnrollmentError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise EnrollmentError("json_root_not_object")
    return value


def write_object(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise EnrollmentError("output_write_failed") from exc


def _exact(value: object, keys: set[str], reason: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EnrollmentError(reason)
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and bool(SHA_RE.fullmatch(value))


def validate_contract(value: object) -> dict[str, object]:
    contract = _exact(value, {
        "bindings", "contract", "contract_sha256", "execution",
        "formal_validation_complete", "identity_authentication", "policy",
        "protocol", "schema_version", "slot_contract", "source_commit",
    }, "contract_schema_invalid")
    payload = dict(contract)
    claimed = payload.pop("contract_sha256")
    if (contract["contract"] != CONTRACT or contract["protocol"] != PROTOCOL
            or contract["schema_version"] != SCHEMA_VERSION
            or contract["source_commit"] != SOURCE_COMMIT
            or not _sha(claimed) or stable_hash(payload) != claimed
            or contract["identity_authentication"] is not False
            or contract["formal_validation_complete"] is not False
            or contract["execution"] != {"network_request_count": 0,
                "llm_request_count": 0, "snapshot_write_count": 0}):
        raise EnrollmentError("contract_identity_invalid")
    bindings = _exact(contract["bindings"], {
        "backup_member_discovery", "backup_set_member_intake",
        "backup_set_topology", "backup_target_attestation",
        "backup_target_registration", "execution_plan", "intake_runtime",
        "target_runtime",
    }, "binding_inventory_invalid")
    if any(not _sha(value) for value in bindings.values()):
        raise EnrollmentError("binding_digest_invalid")
    policy = _exact(contract["policy"], {
        "activation_side_effect", "automatic_scan", "candidate_status",
        "challenge_one_time", "explicit_path_only", "path_serialized",
        "synthetic_can_be_real", "unknown_evidence_policy",
    }, "policy_invalid")
    if policy != {
        "activation_side_effect": False, "automatic_scan": False,
        "candidate_status": "member_candidate_ready_for_intake",
        "challenge_one_time": True, "explicit_path_only": True,
        "path_serialized": False, "synthetic_can_be_real": False,
        "unknown_evidence_policy": "fail_closed",
    }:
        raise EnrollmentError("policy_invalid")
    slot = _exact(contract["slot_contract"], {
        "allowed_shards", "challenge", "contract", "contract_sha256",
        "execution", "formal_validation_complete", "identity_authentication",
        "member_count", "plan_sha256", "protocol", "protocol_sha256",
        "schema_version", "slot", "slot_requirements", "source_commit",
        "topology_sha256",
    }, "slot_contract_invalid")
    if (slot["protocol"] != INTAKE_PROTOCOL
            or slot["source_commit"] != INTAKE_SOURCE_COMMIT
            or slot["member_count"] not in (2, 3, 4)
            or not isinstance(slot["slot"], int)
            or slot["slot"] not in range(slot["member_count"])):
        raise EnrollmentError("slot_contract_invalid")
    slot_payload = dict(slot)
    slot_claimed = slot_payload.pop("contract_sha256")
    if not _sha(slot_claimed) or stable_hash(slot_payload) != slot_claimed:
        raise EnrollmentError("slot_contract_digest_invalid")
    return contract


def validate_evidence(value: object, contract: dict[str, object]) -> dict[str, object]:
    evidence = _exact(value, {
        "challenge_id", "evidence_type", "expires_epoch", "maximum_file_size_bytes",
        "primary_device_identity", "primary_failure_domain_identity",
        "primary_filesystem_identity", "primary_management_domain_identity",
        "quota_bytes", "quota_pool_identity", "recovery_verified", "reserved_bytes",
        "revoked", "storage_service_identity", "target_device_identity",
        "target_failure_domain_identity", "target_filesystem_identity",
        "target_management_domain_identity",
    }, "domain_evidence_schema_invalid")
    challenge = contract["slot_contract"]["challenge"]
    identity_fields = [key for key in evidence if key.endswith("_identity")]
    if (evidence["challenge_id"] != challenge["challenge_id"]
            or evidence["evidence_type"] not in EVIDENCE_TYPES
            or evidence["recovery_verified"] is not True
            or evidence["revoked"] is not False
            or any(value != NOT_AVAILABLE and not _sha(value)
                   for key, value in evidence.items() if key in identity_fields)
            or any(isinstance(evidence[key], bool) or not isinstance(evidence[key], int)
                   or evidence[key] < 0 for key in (
                       "expires_epoch", "maximum_file_size_bytes", "quota_bytes",
                       "reserved_bytes"))):
        raise EnrollmentError("domain_evidence_invalid")
    return evidence


def _filesystem(path: Path) -> tuple[str, str, int, int]:
    stats = os.statvfs(path)
    file_stats = path.stat()
    filesystem = stable_hash({"device": int(file_stats.st_dev),
        "filesystem_id": str(getattr(stats, "f_fsid", NOT_AVAILABLE)),
        "fragment_size": int(stats.f_frsize)})
    device = stable_hash({"device": int(file_stats.st_dev)})
    return (filesystem, device, int(stats.f_bavail) * int(stats.f_frsize),
            int(stats.f_favail))


def _probe(path: Path) -> dict[str, bool]:
    results = {name: False for name in CAPABILITIES}
    try:
        with tempfile.TemporaryDirectory(prefix=".backup-enrollment-", dir=path) as raw:
            root = Path(raw)
            source, target = root / "source", root / "target"
            with source.open("wb") as handle:
                handle.write(b"new"); handle.flush(); os.fsync(handle.fileno())
            results["file_fsync"] = True
            target.write_bytes(b"old"); os.replace(source, target)
            results["atomic_replace"] = target.read_bytes() == b"new"
            fd = os.open(root, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            results["directory_fsync"] = True
            a, b = (root / "lock").open("a+b"), (root / "lock").open("a+b")
            try:
                if fcntl is not None:
                    fcntl.flock(a.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try: fcntl.flock(b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: results["advisory_lock"] = True
            finally: a.close(); b.close()
            (root / "writer-a").write_bytes(b"a"); (root / "writer-b").write_bytes(b"b")
            results["concurrent_writer"] = True
            payload = root / "payload"; payload.write_bytes(b"probe")
            ok = hashlib.sha256(payload.read_bytes()).digest() == hashlib.sha256(b"probe").digest()
            payload.unlink(); results["write_verify_delete"] = ok and not payload.exists()
            parent, child = root / "parent", root / "child"
            parent.write_bytes(b"parent"); child.write_bytes(hashlib.sha256(parent.read_bytes()).digest())
            results["incremental_parent_chain"] = child.read_bytes() == hashlib.sha256(parent.read_bytes()).digest()
            restore = root / "restore"; restore.mkdir()
            results["empty_restore"] = next(restore.iterdir(), None) is None
            long_path = root / ("p" * 240); long_path.write_bytes(b"x")
            results["path_length"] = long_path.exists()
    except (OSError, ValueError):
        pass
    return results


def enroll(contract: dict[str, object], path: Path, evidence: dict[str, object],
           *, observation_epoch: int, synthetic_only: bool = False,
           observed: dict[str, object] | None = None) -> dict[str, object]:
    contract = validate_contract(contract)
    evidence = validate_evidence(evidence, contract)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or path.is_symlink() or resolved != path.absolute():
            raise EnrollmentError("target_path_invalid")
        before = {item.name for item in resolved.glob(".backup-enrollment-*")}
        if observed is None:
            filesystem, device, available_bytes, available_inodes = _filesystem(resolved)
            capabilities = _probe(resolved)
        else:
            filesystem = observed["filesystem_identity"]
            device = observed["device_identity"]
            available_bytes = observed["available_bytes"]
            available_inodes = observed["available_inodes"]
            capabilities = observed["capabilities"]
        after = {item.name for item in resolved.glob(".backup-enrollment-*")}
    except EnrollmentError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EnrollmentError("target_probe_failed") from exc
    if before != after:
        raise EnrollmentError("probe_residue_detected")
    if (evidence["target_filesystem_identity"] != filesystem
            or evidence["target_device_identity"] != device):
        raise EnrollmentError("target_identity_drift")
    slot = contract["slot_contract"]
    req = slot["slot_requirements"]
    observations = {
        "available_bytes": available_bytes, "available_inodes": available_inodes,
        "device_identity": device, "evidence_expires_epoch": evidence["expires_epoch"],
        "evidence_type": evidence["evidence_type"],
        "failure_domain_identity": evidence["target_failure_domain_identity"],
        "filesystem_identity": filesystem,
        "management_domain_identity": evidence["target_management_domain_identity"],
        "observation_epoch": observation_epoch,
        "primary_failure_domain_identity": evidence["primary_failure_domain_identity"],
        "quota_bytes": evidence["quota_bytes"],
        "quota_pool_identity": evidence["quota_pool_identity"],
        "storage_service_identity": evidence["storage_service_identity"],
        "writers": 2 if capabilities.get("concurrent_writer") is True else 0,
    }
    checks = {
        "available_bytes": available_bytes >= req["minimum_available_bytes"],
        "available_inodes": available_inodes >= req["minimum_available_inodes"],
        "failure_domain_independent": evidence["target_failure_domain_identity"] != evidence["primary_failure_domain_identity"],
        "fresh": slot["challenge"]["issued_epoch"] <= observation_epoch <= evidence["expires_epoch"] <= slot["challenge"]["expires_epoch"],
        "quota": evidence["quota_bytes"] >= req["minimum_quota_bytes"],
        "recovery": evidence["recovery_verified"] is True,
        "writers": (2 if capabilities.get("concurrent_writer") is True else 0) >= req["minimum_writers"],
    }
    if (evidence["reserved_bytes"] < req["minimum_available_bytes"]
            or evidence["maximum_file_size_bytes"] <= 0
            or set(capabilities) != set(CAPABILITIES)
            or any(value is not True for value in capabilities.values())
            or not all(checks.values())):
        raise EnrollmentError("target_not_qualified")
    target_identity = stable_hash({key: observations[key] for key in (
        "device_identity", "failure_domain_identity", "filesystem_identity",
        "management_domain_identity", "quota_pool_identity", "storage_service_identity")})
    attestation: dict[str, object] = {
        "attestation": INTAKE_ATTESTATION, "schema_version": SCHEMA_VERSION,
        "protocol": INTAKE_PROTOCOL, "source_commit": INTAKE_SOURCE_COMMIT,
        "contract_sha256": slot["contract_sha256"],
        "challenge_id": slot["challenge"]["challenge_id"],
        "member_count": slot["member_count"], "slot": slot["slot"],
        "target_identity": target_identity, "observations": observations,
        "capabilities": {name: True for name in CAPABILITIES}, "checks": checks,
        "recovery_verified": True, "revoked": False,
        "status": "backup_set_member_qualified", "synthetic_only": synthetic_only,
        "identity_authentication": False, "formal_validation_complete": False,
    }
    attestation["attestation_sha256"] = stable_hash(attestation)
    package: dict[str, object] = {
        "package": PACKAGE, "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL, "source_commit": SOURCE_COMMIT,
        "contract_sha256": contract["contract_sha256"],
        "challenge_id": slot["challenge"]["challenge_id"],
        "member_count": slot["member_count"], "slot": slot["slot"],
        "path_binding_sha256": stable_hash({"path": str(resolved), "target_identity": target_identity}),
        "target_identity": target_identity, "capability_summary": {name: True for name in CAPABILITIES},
        "member_attestation": attestation,
        "status": "member_candidate_ready_for_intake", "synthetic_only": synthetic_only,
        "activation_side_effect": False, "identity_authentication": False,
        "formal_validation_complete": False,
    }
    package["package_sha256"] = stable_hash(package)
    return package


def validate_package(value: object, contract: dict[str, object], *, require_real: bool) -> dict[str, object]:
    package = _exact(value, {
        "activation_side_effect", "capability_summary", "challenge_id",
        "contract_sha256", "formal_validation_complete", "identity_authentication",
        "member_attestation", "member_count", "package", "package_sha256",
        "path_binding_sha256", "protocol", "schema_version", "slot", "source_commit",
        "status", "synthetic_only", "target_identity",
    }, "member_package_schema_invalid")
    payload = dict(package); claimed = payload.pop("package_sha256")
    slot = validate_contract(contract)["slot_contract"]
    if (not _sha(claimed) or stable_hash(payload) != claimed
            or package["package"] != PACKAGE or package["protocol"] != PROTOCOL
            or package["source_commit"] != SOURCE_COMMIT
            or package["contract_sha256"] != contract["contract_sha256"]
            or package["challenge_id"] != slot["challenge"]["challenge_id"]
            or package["member_count"] != slot["member_count"] or package["slot"] != slot["slot"]
            or package["status"] != "member_candidate_ready_for_intake"
            or package["activation_side_effect"] is not False
            or package["identity_authentication"] is not False
            or package["formal_validation_complete"] is not False
            or (require_real and package["synthetic_only"] is not False)):
        raise EnrollmentError("member_package_binding_invalid")
    attestation = package["member_attestation"]
    if not isinstance(attestation, dict):
        raise EnrollmentError("member_attestation_binding_invalid")
    attestation_payload = dict(attestation)
    attestation_claimed = attestation_payload.pop("attestation_sha256", None)
    if (set(attestation) != {
            "attestation", "attestation_sha256", "capabilities", "challenge_id",
            "checks", "contract_sha256", "formal_validation_complete",
            "identity_authentication", "member_count", "observations", "protocol",
            "recovery_verified", "revoked", "schema_version", "slot", "source_commit",
            "status", "synthetic_only", "target_identity"}
            or not _sha(attestation_claimed)
            or stable_hash(attestation_payload) != attestation_claimed
            or attestation.get("protocol") != INTAKE_PROTOCOL
            or attestation.get("source_commit") != INTAKE_SOURCE_COMMIT
            or attestation.get("contract_sha256") != slot["contract_sha256"]
            or attestation.get("challenge_id") != slot["challenge"]["challenge_id"]
            or attestation.get("status") != "backup_set_member_qualified"
            or attestation.get("revoked") is not False
            or attestation.get("recovery_verified") is not True
            or attestation.get("target_identity") != package["target_identity"]
            or attestation.get("synthetic_only") != package["synthetic_only"]):
        raise EnrollmentError("member_attestation_binding_invalid")
    return package


def main(argv: list[str] | None = None) -> int:
    parser = Parser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-contract"); verify.add_argument("--contract", required=True)
    enroll_cmd = commands.add_parser("enroll")
    enroll_cmd.add_argument("--contract", required=True); enroll_cmd.add_argument("--target", required=True)
    enroll_cmd.add_argument("--evidence", required=True); enroll_cmd.add_argument("--observation-epoch", type=int, required=True)
    enroll_cmd.add_argument("--output", required=True)
    package_cmd = commands.add_parser("verify-package")
    package_cmd.add_argument("--contract", required=True); package_cmd.add_argument("--package", required=True)
    try:
        args = parser.parse_args(argv); contract = validate_contract(read_object(Path(args.contract)))
        if args.command == "verify-contract":
            result = {"protocol": PROTOCOL, "status": "kit_contract_verified", "exit_code": 0}
        elif args.command == "enroll":
            result = enroll(contract, Path(args.target), read_object(Path(args.evidence)), observation_epoch=args.observation_epoch)
            write_object(Path(args.output), result)
        else:
            result = validate_package(read_object(Path(args.package)), contract, require_real=False)
        sys.stdout.buffer.write(canonical_bytes(result)); return 0
    except EnrollmentError as exc:
        sys.stdout.buffer.write(canonical_bytes({"protocol": PROTOCOL,
            "status": "enrollment_or_attestation_violation", "exit_code": 2,
            "reason_code": str(exc).split(":", 1)[0]})); return 2
    except Exception:
        sys.stdout.buffer.write(canonical_bytes({"protocol": PROTOCOL,
            "status": "enrollment_or_attestation_violation", "exit_code": 2,
            "reason_code": "runtime_input_invalid"})); return 2


if __name__ == "__main__":
    raise SystemExit(main())
