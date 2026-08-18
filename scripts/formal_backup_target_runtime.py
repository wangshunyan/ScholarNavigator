#!/usr/bin/env python3
"""Standard-library probe for a Full1000 backup target.

The script is copied byte-for-byte into the portable qualification kit.  It
uses operator-supplied paths only while probing and never serializes paths,
hostnames, usernames, environment values, credentials, or query content.
"""

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
except ImportError:  # pragma: no cover - non-POSIX candidates fail closed.
    fcntl = None


PROTOCOL = "formal_backup_target_attestation_v1"
ATTESTATION = "backup_target_attestation_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "8aad1925585ba6715dc66fcadce8decd3480a50e"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EVIDENCE_TYPES = {
    "independent_physical_device_and_management_domain",
    "independent_remote_storage_service",
}
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


class BackupTargetError(RuntimeError):
    """The contract, observation, or attestation failed closed."""


class UsageError(BackupTargetError):
    """CLI arguments are invalid."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise BackupTargetError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise BackupTargetError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                BackupTargetError("nonfinite_json_number")
            ),
        )
    except BackupTargetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupTargetError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise BackupTargetError("json_root_not_object")
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
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BackupTargetError("output_write_failed") from exc


def _exact_object(
    value: object, keys: set[str], reason: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BackupTargetError(reason)
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def validate_contract(value: object) -> dict[str, object]:
    contract = _exact_object(
        value,
        {
            "bindings",
            "challenge",
            "contract_sha256",
            "execution",
            "formal_validation_complete",
            "probe_policy",
            "protocol",
            "requirements",
            "schema_version",
            "source_commit",
        },
        "contract_schema_invalid",
    )
    payload = dict(contract)
    claimed = payload.pop("contract_sha256")
    if (
        contract["protocol"] != PROTOCOL
        or contract["schema_version"] != SCHEMA_VERSION
        or contract["source_commit"] != SOURCE_COMMIT
        or contract["formal_validation_complete"] is not False
        or not _sha(claimed)
        or stable_hash(payload) != claimed
        or contract["execution"]
        != {
            "llm_request_count": 0,
            "network_request_count": 0,
            "snapshot_write_count": 0,
        }
    ):
        raise BackupTargetError("contract_identity_invalid")
    challenge = _exact_object(
        contract["challenge"],
        {"challenge_id", "issued_epoch", "max_age_seconds", "one_time"},
        "challenge_contract_invalid",
    )
    if (
        not _sha(challenge["challenge_id"])
        or isinstance(challenge["issued_epoch"], bool)
        or not isinstance(challenge["issued_epoch"], int)
        or challenge["issued_epoch"] < 0
        or challenge["max_age_seconds"] != 86_400
        or challenge["one_time"] is not True
    ):
        raise BackupTargetError("challenge_contract_invalid")
    requirements = _exact_object(
        contract["requirements"],
        {
            "active_shard_window",
            "fault_domain_policy",
            "max_file_size_bytes",
            "minimum_concurrent_writers",
            "minimum_path_name_bytes",
            "required_available_bytes",
            "required_available_inodes",
            "required_quota_bytes",
            "required_reserved_bytes",
        },
        "requirements_invalid",
    )
    if (
        requirements["active_shard_window"] != 4
        or requirements["required_available_bytes"] != 2_119_029_489_664
        or requirements["required_available_inodes"] != 210_940
        or requirements["required_quota_bytes"] != 2_119_029_489_664
        or requirements["required_reserved_bytes"] != 2_119_029_489_664
        or requirements["max_file_size_bytes"] != 35_030_827_008
        or requirements["minimum_concurrent_writers"] != 2
        or requirements["minimum_path_name_bytes"] != 240
        or requirements["fault_domain_policy"]
        != "verified_independent_device_and_management_domain"
    ):
        raise BackupTargetError("requirements_drift")
    bindings = _exact_object(
        contract["bindings"],
        {
            "disaster_recovery",
            "execution_plan",
            "host_attestation",
            "launch_control",
            "multivolume_storage",
            "shard_streaming_retention",
            "storage_governance",
        },
        "binding_inventory_invalid",
    )
    if any(not _sha(item) for item in bindings.values()):
        raise BackupTargetError("binding_digest_invalid")
    policy = _exact_object(
        contract["probe_policy"],
        {
            "absolute_paths_serialized",
            "credentials_read",
            "environment_values_serialized",
            "identity_authentication",
            "quota_unknown_policy",
            "sparse_or_compression_credit",
        },
        "probe_policy_invalid",
    )
    if policy != {
        "absolute_paths_serialized": False,
        "credentials_read": False,
        "environment_values_serialized": False,
        "identity_authentication": False,
        "quota_unknown_policy": "fail_closed",
        "sparse_or_compression_credit": False,
    }:
        raise BackupTargetError("probe_policy_invalid")
    return contract


def validate_domain_evidence(
    value: object, contract: dict[str, object]
) -> dict[str, object]:
    evidence = _exact_object(
        value,
        {
            "challenge_id",
            "evidence_type",
            "expires_epoch",
            "maximum_file_size_bytes",
            "primary_device_identity",
            "primary_failure_domain_identity",
            "primary_filesystem_identity",
            "primary_management_domain_identity",
            "quota_bytes",
            "reserved_bytes",
            "target_device_identity",
            "target_failure_domain_identity",
            "target_filesystem_identity",
            "target_management_domain_identity",
            "target_storage_service_identity",
        },
        "domain_evidence_schema_invalid",
    )
    if (
        evidence["challenge_id"] != contract["challenge"]["challenge_id"]
        or evidence["evidence_type"] not in EVIDENCE_TYPES
        or isinstance(evidence["expires_epoch"], bool)
        or not isinstance(evidence["expires_epoch"], int)
        or any(
            isinstance(evidence[key], bool)
            or not isinstance(evidence[key], int)
            or evidence[key] < 0
            for key in (
                "maximum_file_size_bytes",
                "quota_bytes",
                "reserved_bytes",
            )
        )
        or any(
            value != NOT_AVAILABLE and not _sha(value)
            for key, value in evidence.items()
            if key.endswith("_identity")
        )
    ):
        raise BackupTargetError("domain_evidence_invalid")
    return evidence


def _capability(passed: bool, reason: str) -> dict[str, object]:
    return {"passed": bool(passed), "reason_code": reason}


def _probe_capabilities(path: Path, minimum_path_name_bytes: int) -> dict[str, object]:
    results: dict[str, dict[str, object]] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=".backup-attestation-", dir=path
        ) as raw:
            root = Path(raw)
            source = root / "source"
            target = root / "target"
            with source.open("wb") as handle:
                handle.write(b"new")
                handle.flush()
                os.fsync(handle.fileno())
            results["file_fsync"] = _capability(True, "file_fsync_verified")
            target.write_bytes(b"old")
            os.replace(source, target)
            results["atomic_replace"] = _capability(
                target.read_bytes() == b"new", "atomic_replace_verified"
            )
            descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            results["directory_fsync"] = _capability(
                True, "directory_fsync_verified"
            )
            lock_a = (root / "lock").open("a+b")
            lock_b = (root / "lock").open("a+b")
            rejected = False
            try:
                if fcntl is not None:
                    fcntl.flock(
                        lock_a.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    try:
                        fcntl.flock(
                            lock_b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        rejected = True
            finally:
                lock_a.close()
                lock_b.close()
            results["advisory_lock"] = _capability(
                rejected, "advisory_lock_contention_verified"
            )
            writer_a = (root / "writer-a").open("wb")
            writer_b = (root / "writer-b").open("wb")
            writer_a.close()
            writer_b.close()
            results["concurrent_writer"] = _capability(
                True, "concurrent_writer_verified"
            )
            payload = root / "payload"
            payload.write_bytes(b"backup-target-probe")
            verified = (
                hashlib.sha256(payload.read_bytes()).hexdigest()
                == hashlib.sha256(b"backup-target-probe").hexdigest()
            )
            payload.unlink()
            results["write_verify_delete"] = _capability(
                verified and not payload.exists(), "write_verify_delete_verified"
            )
            parent = root / "parent"
            child = root / "child"
            parent.write_bytes(b"parent")
            child.write_bytes(hashlib.sha256(parent.read_bytes()).digest())
            results["incremental_parent_chain"] = _capability(
                child.read_bytes() == hashlib.sha256(parent.read_bytes()).digest(),
                "incremental_parent_chain_verified",
            )
            restore = root / "restore"
            restore.mkdir()
            results["empty_restore"] = _capability(
                next(restore.iterdir(), None) is None,
                "empty_restore_verified",
            )
            name = "p" * minimum_path_name_bytes
            long_path = root / name
            long_path.write_bytes(b"x")
            results["path_length"] = _capability(
                long_path.exists(), "path_length_verified"
            )
    except (OSError, ValueError):
        for name in CAPABILITIES:
            results.setdefault(name, _capability(False, f"{name}_failed"))
    return {key: results[key] for key in sorted(results)}


def _filesystem_id(path: Path) -> tuple[str, str, int, int]:
    try:
        stats = os.statvfs(path)
        file_stats = path.stat()
    except OSError as exc:
        raise BackupTargetError("target_observation_failed") from exc
    filesystem = stable_hash(
        {
            "device": int(file_stats.st_dev),
            "filesystem_id": str(getattr(stats, "f_fsid", NOT_AVAILABLE)),
            "fragment_size": int(stats.f_frsize),
        }
    )
    device = stable_hash({"device": int(file_stats.st_dev)})
    available_bytes = int(stats.f_bavail) * int(stats.f_frsize)
    available_inodes = int(stats.f_favail)
    return filesystem, device, available_bytes, available_inodes


def build_attestation(
    contract: dict[str, object],
    evidence: dict[str, object],
    *,
    observation_epoch: int,
    filesystem_identity: str,
    device_identity: str,
    available_bytes: int,
    available_inodes: int,
    capabilities: dict[str, dict[str, object]],
    synthetic_only: bool,
) -> dict[str, object]:
    requirements = contract["requirements"]
    evidence_fresh = (
        observation_epoch >= contract["challenge"]["issued_epoch"]
        and observation_epoch <= evidence["expires_epoch"]
        and observation_epoch
        <= contract["challenge"]["issued_epoch"]
        + contract["challenge"]["max_age_seconds"]
    )
    identities_match = (
        evidence["target_filesystem_identity"] == filesystem_identity
        and evidence["target_device_identity"] == device_identity
    )
    identities_distinct = (
        evidence["primary_device_identity"] != device_identity
        and evidence["primary_failure_domain_identity"]
        != evidence["target_failure_domain_identity"]
        and evidence["primary_management_domain_identity"]
        != evidence["target_management_domain_identity"]
        and evidence["primary_filesystem_identity"] != filesystem_identity
    )
    remote_complete = (
        evidence["evidence_type"]
        != "independent_remote_storage_service"
        or evidence["target_storage_service_identity"] != NOT_AVAILABLE
    )
    checks = {
        "available_bytes": available_bytes
        >= requirements["required_available_bytes"],
        "available_inodes": available_inodes
        >= requirements["required_available_inodes"],
        "capabilities": all(
            item["passed"] is True for item in capabilities.values()
        ),
        "domain_evidence_fresh": evidence_fresh,
        "failure_domain_independent": identities_distinct and remote_complete,
        "identity_binding": identities_match,
        "maximum_file_size": evidence["maximum_file_size_bytes"]
        >= requirements["max_file_size_bytes"],
        "quota": evidence["quota_bytes"] >= requirements["required_quota_bytes"],
        "reserved_space": evidence["reserved_bytes"]
        >= requirements["required_reserved_bytes"],
    }
    qualified = all(checks.values())
    payload: dict[str, object] = {
        "attestation": ATTESTATION,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "challenge_id": contract["challenge"]["challenge_id"],
        "contract_sha256": contract["contract_sha256"],
        "observation_epoch": observation_epoch,
        "target_identity": stable_hash(
            {
                "device_identity": device_identity,
                "failure_domain_identity": evidence[
                    "target_failure_domain_identity"
                ],
                "filesystem_identity": filesystem_identity,
                "management_domain_identity": evidence[
                    "target_management_domain_identity"
                ],
                "storage_service_identity": evidence[
                    "target_storage_service_identity"
                ],
            }
        ),
        "observations": {
            "available_bytes": available_bytes,
            "available_inodes": available_inodes,
            "device_identity": device_identity,
            "evidence_expires_epoch": evidence["expires_epoch"],
            "evidence_type": evidence["evidence_type"],
            "failure_domain_identity": evidence[
                "target_failure_domain_identity"
            ],
            "filesystem_identity": filesystem_identity,
            "management_domain_identity": evidence[
                "target_management_domain_identity"
            ],
            "maximum_file_size_bytes": evidence["maximum_file_size_bytes"],
            "primary_device_identity": evidence["primary_device_identity"],
            "primary_failure_domain_identity": evidence[
                "primary_failure_domain_identity"
            ],
            "primary_filesystem_identity": evidence[
                "primary_filesystem_identity"
            ],
            "primary_management_domain_identity": evidence[
                "primary_management_domain_identity"
            ],
            "quota_bytes": evidence["quota_bytes"],
            "reserved_bytes": evidence["reserved_bytes"],
            "storage_service_identity": evidence[
                "target_storage_service_identity"
            ],
        },
        "capabilities": capabilities,
        "checks": checks,
        "status": (
            "backup_target_qualified"
            if qualified
            else "not_ready_no_qualified_backup_target"
        ),
        "identity_authentication": False,
        "synthetic_only": synthetic_only,
        "formal_validation_complete": False,
    }
    payload["attestation_sha256"] = stable_hash(payload)
    return payload


def validate_attestation(
    value: object,
    contract: dict[str, object],
    *,
    require_qualified: bool,
) -> dict[str, object]:
    attestation = _exact_object(
        value,
        {
            "attestation",
            "attestation_sha256",
            "capabilities",
            "challenge_id",
            "checks",
            "contract_sha256",
            "formal_validation_complete",
            "identity_authentication",
            "observation_epoch",
            "observations",
            "protocol",
            "schema_version",
            "source_commit",
            "status",
            "synthetic_only",
            "target_identity",
        },
        "attestation_schema_invalid",
    )
    payload = dict(attestation)
    claimed = payload.pop("attestation_sha256")
    if not _sha(claimed) or stable_hash(payload) != claimed:
        raise BackupTargetError("attestation_digest_invalid")
    if (
        attestation["attestation"] != ATTESTATION
        or attestation["protocol"] != PROTOCOL
        or attestation["schema_version"] != SCHEMA_VERSION
        or attestation["source_commit"] != SOURCE_COMMIT
        or attestation["challenge_id"]
        != contract["challenge"]["challenge_id"]
        or attestation["contract_sha256"] != contract["contract_sha256"]
        or attestation["identity_authentication"] is not False
        or attestation["formal_validation_complete"] is not False
        or not isinstance(attestation["synthetic_only"], bool)
        or not _sha(attestation["target_identity"])
    ):
        raise BackupTargetError("attestation_binding_invalid")
    capabilities = _exact_object(
        attestation["capabilities"],
        set(CAPABILITIES),
        "attestation_capabilities_invalid",
    )
    for item in capabilities.values():
        row = _exact_object(
            item,
            {"passed", "reason_code"},
            "attestation_capability_invalid",
        )
        if (
            not isinstance(row["passed"], bool)
            or not isinstance(row["reason_code"], str)
            or not row["reason_code"]
        ):
            raise BackupTargetError("attestation_capability_invalid")
    checks = _exact_object(
        attestation["checks"],
        {
            "available_bytes",
            "available_inodes",
            "capabilities",
            "domain_evidence_fresh",
            "failure_domain_independent",
            "identity_binding",
            "maximum_file_size",
            "quota",
            "reserved_space",
        },
        "attestation_checks_invalid",
    )
    if any(not isinstance(item, bool) for item in checks.values()):
        raise BackupTargetError("attestation_checks_invalid")
    observations = _exact_object(
        attestation["observations"],
        {
            "available_bytes",
            "available_inodes",
            "device_identity",
            "evidence_expires_epoch",
            "evidence_type",
            "failure_domain_identity",
            "filesystem_identity",
            "management_domain_identity",
            "maximum_file_size_bytes",
            "primary_device_identity",
            "primary_failure_domain_identity",
            "primary_filesystem_identity",
            "primary_management_domain_identity",
            "quota_bytes",
            "reserved_bytes",
            "storage_service_identity",
        },
        "attestation_observations_invalid",
    )
    if (
        observations["evidence_type"] not in EVIDENCE_TYPES
        or any(
            not _sha(observations[key])
            for key in (
                "device_identity",
                "failure_domain_identity",
                "filesystem_identity",
                "management_domain_identity",
                "primary_device_identity",
                "primary_failure_domain_identity",
                "primary_filesystem_identity",
                "primary_management_domain_identity",
            )
        )
        or (
            observations["storage_service_identity"] != NOT_AVAILABLE
            and not _sha(observations["storage_service_identity"])
        )
        or any(
            isinstance(observations[key], bool)
            or not isinstance(observations[key], int)
            or observations[key] < 0
            for key in (
                "available_bytes",
                "available_inodes",
                "evidence_expires_epoch",
                "maximum_file_size_bytes",
                "quota_bytes",
                "reserved_bytes",
            )
        )
    ):
        raise BackupTargetError("attestation_observations_invalid")
    requirements = contract["requirements"]
    expected_target_identity = stable_hash(
        {
            "device_identity": observations["device_identity"],
            "failure_domain_identity": observations["failure_domain_identity"],
            "filesystem_identity": observations["filesystem_identity"],
            "management_domain_identity": observations[
                "management_domain_identity"
            ],
            "storage_service_identity": observations[
                "storage_service_identity"
            ],
        }
    )
    expected_checks = {
        "available_bytes": observations["available_bytes"]
        >= requirements["required_available_bytes"],
        "available_inodes": observations["available_inodes"]
        >= requirements["required_available_inodes"],
        "capabilities": all(
            item["passed"] is True for item in capabilities.values()
        ),
        "domain_evidence_fresh": (
            attestation["observation_epoch"]
            >= contract["challenge"]["issued_epoch"]
            and attestation["observation_epoch"]
            <= observations["evidence_expires_epoch"]
            and attestation["observation_epoch"]
            <= contract["challenge"]["issued_epoch"]
            + contract["challenge"]["max_age_seconds"]
        ),
        "failure_domain_independent": (
            observations["primary_device_identity"]
            != observations["device_identity"]
            and observations["primary_failure_domain_identity"]
            != observations["failure_domain_identity"]
            and observations["primary_filesystem_identity"]
            != observations["filesystem_identity"]
            and observations["primary_management_domain_identity"]
            != observations["management_domain_identity"]
            and (
                observations["evidence_type"]
                != "independent_remote_storage_service"
                or observations["storage_service_identity"] != NOT_AVAILABLE
            )
        ),
        "identity_binding": True,
        "maximum_file_size": observations["maximum_file_size_bytes"]
        >= requirements["max_file_size_bytes"],
        "quota": observations["quota_bytes"]
        >= requirements["required_quota_bytes"],
        "reserved_space": observations["reserved_bytes"]
        >= requirements["required_reserved_bytes"],
    }
    if (
        checks != expected_checks
        or attestation["target_identity"] != expected_target_identity
    ):
        raise BackupTargetError("attestation_semantic_inconsistent")
    qualified = all(checks.values()) and all(
        item["passed"] is True for item in capabilities.values()
    )
    expected_status = (
        "backup_target_qualified"
        if qualified
        else "not_ready_no_qualified_backup_target"
    )
    if attestation["status"] != expected_status:
        raise BackupTargetError("attestation_status_inconsistent")
    if require_qualified and not qualified:
        raise BackupTargetError("attestation_not_qualified")
    return attestation


def probe(
    contract_path: Path,
    target_path: Path,
    evidence_path: Path,
    *,
    observation_epoch: int,
    output: Path,
) -> dict[str, object]:
    contract = validate_contract(read_object(contract_path))
    evidence = validate_domain_evidence(read_object(evidence_path), contract)
    if not target_path.is_dir():
        raise BackupTargetError("target_path_unavailable")
    filesystem, device, available_bytes, available_inodes = _filesystem_id(
        target_path
    )
    capabilities = _probe_capabilities(
        target_path, contract["requirements"]["minimum_path_name_bytes"]
    )
    attestation = build_attestation(
        contract,
        evidence,
        observation_epoch=observation_epoch,
        filesystem_identity=filesystem,
        device_identity=device,
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        capabilities=capabilities,
        synthetic_only=False,
    )
    write_object(output, attestation)
    return attestation


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify_contract = commands.add_parser("verify-contract")
    verify_contract.add_argument("--contract", required=True)
    probe_command = commands.add_parser("probe")
    probe_command.add_argument("--contract", required=True)
    probe_command.add_argument("--target", required=True)
    probe_command.add_argument("--domain-evidence", required=True)
    probe_command.add_argument("--observation-epoch", required=True, type=int)
    probe_command.add_argument("--output", required=True)
    verify = commands.add_parser("verify-attestation")
    verify.add_argument("--contract", required=True)
    verify.add_argument("--attestation", required=True)
    return parser


def _result(status: str, exit_code: int, reason: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
    }
    if reason:
        result["reason_code"] = reason.split(":", 1)[0]
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "verify-contract":
            contract = validate_contract(read_object(Path(args.contract)))
            report = {
                **_result("backup_target_qualified", EXIT_QUALIFIED),
                "contract_sha256": contract["contract_sha256"],
            }
        elif args.command == "probe":
            attestation = probe(
                Path(args.contract),
                Path(args.target),
                Path(args.domain_evidence),
                observation_epoch=args.observation_epoch,
                output=Path(args.output),
            )
            report = {
                **_result(
                    attestation["status"],
                    (
                        EXIT_QUALIFIED
                        if attestation["status"] == "backup_target_qualified"
                        else EXIT_NOT_READY
                    ),
                ),
                "attestation_sha256": attestation["attestation_sha256"],
            }
        elif args.command == "verify-attestation":
            contract = validate_contract(read_object(Path(args.contract)))
            attestation = validate_attestation(
                read_object(Path(args.attestation)),
                contract,
                require_qualified=False,
            )
            report = {
                **_result(
                    attestation["status"],
                    (
                        EXIT_QUALIFIED
                        if attestation["status"] == "backup_target_qualified"
                        else EXIT_NOT_READY
                    ),
                ),
                "attestation_sha256": attestation["attestation_sha256"],
            }
        else:  # pragma: no cover
            raise UsageError("unsupported_command")
    except UsageError as exc:
        report = _result("usage_error", EXIT_USAGE, str(exc))
    except (
        BackupTargetError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        report = _result(
            "attestation_or_failure_domain_violation",
            EXIT_VIOLATION,
            (
                str(exc)
                if isinstance(exc, BackupTargetError)
                else "controlled_input_or_filesystem_violation"
            ),
        )
    sys.stdout.buffer.write(canonical_bytes(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
