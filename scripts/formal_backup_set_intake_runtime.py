#!/usr/bin/env python3
"""Standard-library verifier for a Full1000 backup-set member slot kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


PROTOCOL = "formal_backup_set_member_intake_v1"
CONTRACT = "backup_set_member_slot_contract_v1"
ATTESTATION = "backup_set_member_attestation_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "851a113aed91f3764b16cb35a5b653debfc88426"
NOT_AVAILABLE = "not_available"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
EVIDENCE_TYPES = {
    "independent_physical_device_and_management_domain",
    "independent_remote_storage_service",
}


class IntakeRuntimeError(RuntimeError):
    """An offline contract or attestation failed closed."""


class UsageError(IntakeRuntimeError):
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
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise IntakeRuntimeError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise IntakeRuntimeError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                IntakeRuntimeError("nonfinite_json_number")
            ),
        )
    except IntakeRuntimeError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise IntakeRuntimeError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise IntakeRuntimeError("json_root_not_object")
    return value


def write_object(path: Path, value: dict[str, object]) -> None:
    try:
        path.write_bytes(canonical_bytes(value))
    except OSError as exc:
        raise IntakeRuntimeError("output_write_failed") from exc


def _exact(value: object, keys: set[str], reason: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise IntakeRuntimeError(reason)
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_contract(value: object) -> dict[str, object]:
    contract = _exact(
        value,
        {
            "allowed_shards",
            "challenge",
            "contract",
            "contract_sha256",
            "execution",
            "formal_validation_complete",
            "identity_authentication",
            "member_count",
            "plan_sha256",
            "protocol",
            "protocol_sha256",
            "schema_version",
            "slot",
            "slot_requirements",
            "source_commit",
            "topology_sha256",
        },
        "contract_schema_invalid",
    )
    payload = dict(contract)
    claimed = payload.pop("contract_sha256")
    if (
        contract["contract"] != CONTRACT
        or contract["protocol"] != PROTOCOL
        or contract["schema_version"] != SCHEMA_VERSION
        or contract["source_commit"] != SOURCE_COMMIT
        or contract["formal_validation_complete"] is not False
        or contract["identity_authentication"] is not False
        or not _sha(claimed)
        or stable_hash(payload) != claimed
        or not _sha(contract["protocol_sha256"])
        or not _sha(contract["plan_sha256"])
        or not _sha(contract["topology_sha256"])
        or contract["execution"]
        != {
            "llm_request_count": 0,
            "network_request_count": 0,
            "snapshot_write_count": 0,
        }
    ):
        raise IntakeRuntimeError("contract_identity_invalid")
    member_count = contract["member_count"]
    slot = contract["slot"]
    if (
        member_count not in (2, 3, 4)
        or not _nonnegative_int(slot)
        or slot >= member_count
    ):
        raise IntakeRuntimeError("contract_slot_invalid")
    shards = contract["allowed_shards"]
    if (
        not isinstance(shards, list)
        or any(not _nonnegative_int(item) or item >= 20 for item in shards)
        or len(shards) != len(set(shards))
        or shards != [item for item in range(20) if item % member_count == slot]
    ):
        raise IntakeRuntimeError("contract_shards_invalid")
    challenge = _exact(
        contract["challenge"],
        {"challenge_id", "expires_epoch", "issued_epoch", "one_time"},
        "challenge_schema_invalid",
    )
    if (
        not _sha(challenge["challenge_id"])
        or not _nonnegative_int(challenge["issued_epoch"])
        or not _nonnegative_int(challenge["expires_epoch"])
        or challenge["expires_epoch"] <= challenge["issued_epoch"]
        or challenge["one_time"] is not True
    ):
        raise IntakeRuntimeError("challenge_invalid")
    requirements = _exact(
        contract["slot_requirements"],
        {
            "capability_names",
            "failure_domain_evidence_required",
            "minimum_available_bytes",
            "minimum_available_inodes",
            "minimum_quota_bytes",
            "minimum_writers",
            "recovery_verification_required",
        },
        "slot_requirements_invalid",
    )
    if (
        requirements["capability_names"] != list(CAPABILITIES)
        or requirements["failure_domain_evidence_required"] is not True
        or requirements["recovery_verification_required"] is not True
        or any(
            not _nonnegative_int(requirements[key]) or requirements[key] <= 0
            for key in (
                "minimum_available_bytes",
                "minimum_available_inodes",
                "minimum_quota_bytes",
                "minimum_writers",
            )
        )
        or requirements["minimum_quota_bytes"]
        != requirements["minimum_available_bytes"]
    ):
        raise IntakeRuntimeError("slot_requirements_invalid")
    return contract


def validate_attestation(
    value: object,
    contract: dict[str, object],
    *,
    observation_epoch: int,
    require_real: bool,
) -> dict[str, object]:
    attestation = _exact(
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
            "member_count",
            "observations",
            "protocol",
            "recovery_verified",
            "revoked",
            "schema_version",
            "slot",
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
        raise IntakeRuntimeError("attestation_digest_invalid")
    if (
        attestation["attestation"] != ATTESTATION
        or attestation["protocol"] != PROTOCOL
        or attestation["schema_version"] != SCHEMA_VERSION
        or attestation["source_commit"] != SOURCE_COMMIT
        or attestation["contract_sha256"] != contract["contract_sha256"]
        or attestation["challenge_id"]
        != contract["challenge"]["challenge_id"]
        or attestation["member_count"] != contract["member_count"]
        or attestation["slot"] != contract["slot"]
        or attestation["formal_validation_complete"] is not False
        or attestation["identity_authentication"] is not False
        or attestation["revoked"] is not False
        or attestation["recovery_verified"] is not True
        or not isinstance(attestation["synthetic_only"], bool)
        or (require_real and attestation["synthetic_only"] is not False)
        or not _sha(attestation["target_identity"])
    ):
        raise IntakeRuntimeError("attestation_binding_invalid")
    capabilities = _exact(
        attestation["capabilities"],
        set(CAPABILITIES),
        "capabilities_invalid",
    )
    if any(value is not True for value in capabilities.values()):
        raise IntakeRuntimeError("capability_not_qualified")
    observations = _exact(
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
            "observation_epoch",
            "primary_failure_domain_identity",
            "quota_bytes",
            "quota_pool_identity",
            "storage_service_identity",
            "writers",
        },
        "observations_invalid",
    )
    identities = (
        "device_identity",
        "failure_domain_identity",
        "filesystem_identity",
        "management_domain_identity",
        "primary_failure_domain_identity",
        "quota_pool_identity",
    )
    if (
        observations["evidence_type"] not in EVIDENCE_TYPES
        or any(not _sha(observations[key]) for key in identities)
        or (
            observations["storage_service_identity"] != NOT_AVAILABLE
            and not _sha(observations["storage_service_identity"])
        )
        or any(
            not _nonnegative_int(observations[key])
            for key in (
                "available_bytes",
                "available_inodes",
                "evidence_expires_epoch",
                "observation_epoch",
                "quota_bytes",
                "writers",
            )
        )
        or observation_epoch < observations["observation_epoch"]
        or observation_epoch > observations["evidence_expires_epoch"]
        or observations["observation_epoch"]
        < contract["challenge"]["issued_epoch"]
        or observations["evidence_expires_epoch"]
        > contract["challenge"]["expires_epoch"]
    ):
        raise IntakeRuntimeError("observations_invalid")
    requirements = contract["slot_requirements"]
    expected_checks = {
        "available_bytes": observations["available_bytes"]
        >= requirements["minimum_available_bytes"],
        "available_inodes": observations["available_inodes"]
        >= requirements["minimum_available_inodes"],
        "failure_domain_independent": observations["failure_domain_identity"]
        != observations["primary_failure_domain_identity"],
        "fresh": observation_epoch <= observations["evidence_expires_epoch"],
        "quota": observations["quota_bytes"]
        >= requirements["minimum_quota_bytes"],
        "recovery": attestation["recovery_verified"] is True,
        "writers": observations["writers"] >= requirements["minimum_writers"],
    }
    checks = _exact(
        attestation["checks"], set(expected_checks), "checks_invalid"
    )
    if checks != expected_checks or not all(expected_checks.values()):
        raise IntakeRuntimeError("attestation_not_qualified")
    expected_identity = stable_hash(
        {
            "device_identity": observations["device_identity"],
            "failure_domain_identity": observations["failure_domain_identity"],
            "filesystem_identity": observations["filesystem_identity"],
            "management_domain_identity": observations["management_domain_identity"],
            "quota_pool_identity": observations["quota_pool_identity"],
            "storage_service_identity": observations["storage_service_identity"],
        }
    )
    if (
        attestation["target_identity"] != expected_identity
        or attestation["status"] != "backup_set_member_qualified"
    ):
        raise IntakeRuntimeError("attestation_semantic_invalid")
    return attestation


def _result(status: str, exit_code: int, reason: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
    }
    if reason:
        value["reason_code"] = reason.split(":", 1)[0]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    verify_contract = command.add_parser("verify-contract")
    verify_contract.add_argument("--contract", required=True)
    verify = command.add_parser("verify-member")
    verify.add_argument("--contract", required=True)
    verify.add_argument("--attestation", required=True)
    verify.add_argument("--observation-epoch", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        contract = validate_contract(read_object(Path(args.contract)))
        if args.command == "verify-contract":
            result = _result("slot_kit_verified", EXIT_READY)
            result["contract_sha256"] = contract["contract_sha256"]
        elif args.command == "verify-member":
            attestation = validate_attestation(
                read_object(Path(args.attestation)),
                contract,
                observation_epoch=args.observation_epoch,
                require_real=False,
            )
            result = _result("backup_set_member_qualified", EXIT_READY)
            result["attestation_sha256"] = attestation["attestation_sha256"]
        else:
            raise UsageError("unsupported_command")
    except UsageError as exc:
        result = _result("usage_error", EXIT_USAGE, str(exc))
    except (
        IntakeRuntimeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        result = _result(
            "member_or_activation_violation",
            EXIT_VIOLATION,
            str(exc)
            if isinstance(exc, IntakeRuntimeError)
            else "controlled_input_violation",
        )
    sys.stdout.buffer.write(canonical_bytes(result))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
