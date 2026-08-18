#!/usr/bin/env python3
"""Standard-library probe and verifier for portable execution-site kits.

This file is copied byte-for-byte into a portable kit.  It intentionally has
no project imports and is suitable for ``python -I -S``.  Paths supplied by an
operator are used only while probing and are never serialized.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import unicodedata
from pathlib import Path


PROTOCOL = "portable_execution_site_attestation_v1"
ATTESTATION = "execution_site_attestation_v1"
SCHEMA_VERSION = "1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CAPABILITIES = (
    "advisory_lock",
    "atomic_replace",
    "case_semantics",
    "directory_fsync",
    "file_fsync",
    "nonempty_restore_rejection",
    "temporary_directory",
    "unicode_semantics",
    "write_permission",
)


class SiteError(RuntimeError):
    """Fail-closed portable probe or verifier error."""


class UsageError(SiteError):
    """Invalid command-line use."""


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


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_hash(value: object) -> str:
    return digest_bytes(canonical_bytes(value))


def _unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise SiteError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise SiteError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                SiteError("nonfinite_json_number")
            ),
        )
    except SiteError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SiteError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise SiteError("json_root_not_object")
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise SiteError("output_write_failed") from exc


def _exact_object(
    value: object, keys: set[str], reason: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SiteError(reason)
    return value


def validate_contract(value: object) -> dict[str, object]:
    contract = _exact_object(
        value,
        {
            "challenge",
            "execution",
            "formal_validation_complete",
            "plan_binding",
            "probe_policy",
            "protocol",
            "schema_version",
            "source_commit",
            "topology",
            "contract_sha256",
        },
        "contract_schema_invalid",
    )
    if (
        contract["protocol"] != PROTOCOL
        or contract["schema_version"] != SCHEMA_VERSION
        or contract["formal_validation_complete"] is not False
        or not isinstance(contract["source_commit"], str)
        or not COMMIT_RE.fullmatch(contract["source_commit"])
    ):
        raise SiteError("contract_semantic_invalid")
    payload = dict(contract)
    claimed = payload.pop("contract_sha256")
    if not isinstance(claimed, str) or stable_hash(payload) != claimed:
        raise SiteError("contract_digest_invalid")
    execution = contract["execution"]
    if execution != {
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }:
        raise SiteError("execution_boundary_invalid")
    challenge = _exact_object(
        contract["challenge"],
        {"challenge_id", "issued_epoch", "max_age_seconds", "one_time"},
        "challenge_contract_invalid",
    )
    if (
        not isinstance(challenge["challenge_id"], str)
        or not SHA256_RE.fullmatch(challenge["challenge_id"])
        or not isinstance(challenge["issued_epoch"], int)
        or isinstance(challenge["issued_epoch"], bool)
        or challenge["issued_epoch"] < 0
        or challenge["max_age_seconds"] != 86400
        or challenge["one_time"] is not True
    ):
        raise SiteError("challenge_contract_invalid")
    topology = _exact_object(
        contract["topology"],
        {
            "allocation_algorithm",
            "backup_slots",
            "primary_slots",
            "requirements",
            "shard_assignments",
            "topology_sha256",
        },
        "topology_contract_invalid",
    )
    topology_payload = dict(topology)
    topology_digest = topology_payload.pop("topology_sha256")
    if not isinstance(topology_digest, str) or stable_hash(topology_payload) != topology_digest:
        raise SiteError("topology_digest_invalid")
    primary = topology["primary_slots"]
    backup = topology["backup_slots"]
    assignments = topology["shard_assignments"]
    requirements = topology["requirements"]
    if (
        not isinstance(primary, list)
        or not isinstance(backup, list)
        or not primary
        or len(primary) != len(backup)
        or primary != sorted(set(primary))
        or backup != sorted(set(backup))
        or set(primary) & set(backup)
        or not isinstance(assignments, list)
        or len(assignments) != 20
        or not isinstance(requirements, dict)
        or set(requirements) != set(primary) | set(backup)
    ):
        raise SiteError("topology_inventory_invalid")
    for row in assignments:
        item = _exact_object(
            row,
            {"backup_slot", "primary_slot", "shard_index"},
            "shard_assignment_invalid",
        )
        if (
            not isinstance(item["shard_index"], int)
            or isinstance(item["shard_index"], bool)
            or item["shard_index"] < 0
            or item["shard_index"] >= 20
            or item["primary_slot"] not in primary
            or item["backup_slot"] not in backup
        ):
            raise SiteError("shard_assignment_invalid")
    if sorted(row["shard_index"] for row in assignments) != list(range(20)):
        raise SiteError("shard_assignment_not_closed")
    for slot, raw in requirements.items():
        item = _exact_object(
            raw,
            {
                "assigned_shards",
                "required_bytes",
                "required_concurrent_writers",
                "required_inodes",
                "role",
            },
            "volume_requirement_invalid",
        )
        if (
            item["role"]
            not in (("primary",) if slot in primary else ("backup",))
            or not isinstance(item["assigned_shards"], list)
            or item["assigned_shards"] != sorted(set(item["assigned_shards"]))
            or any(
                isinstance(item[key], bool)
                or not isinstance(item[key], int)
                or item[key] <= 0
                for key in (
                    "required_bytes",
                    "required_concurrent_writers",
                    "required_inodes",
                )
            )
        ):
            raise SiteError("volume_requirement_invalid")
    return contract


def _parse_volume_rows(values: list[str]) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise UsageError("invalid_volume_binding")
        slot, raw_path = value.split("=", 1)
        if (
            not slot
            or slot in rows
            or not re.fullmatch(r"(?:primary|backup)-[0-9]{2}", slot)
            or not raw_path
        ):
            raise UsageError("invalid_volume_binding")
        rows[slot] = Path(raw_path)
    return rows


def _site_evidence(
    path: Path | None, contract: dict[str, object]
) -> tuple[dict[str, dict[str, object]], str]:
    if path is None:
        return {}, NOT_AVAILABLE
    raw = path.read_bytes()
    value = read_object(path)
    document = _exact_object(
        value,
        {"challenge_id", "evidence_type", "volumes"},
        "site_evidence_schema_invalid",
    )
    if (
        document["challenge_id"] != contract["challenge"]["challenge_id"]
        or document["evidence_type"] != "portable_site_operator_observation_v1"
        or not isinstance(document["volumes"], dict)
    ):
        raise SiteError("site_evidence_binding_invalid")
    result: dict[str, dict[str, object]] = {}
    expected_slots = set(contract["topology"]["requirements"])
    if set(document["volumes"]) != expected_slots:
        raise SiteError("site_evidence_volume_inventory_invalid")
    for slot, raw_row in document["volumes"].items():
        row = _exact_object(
            raw_row,
            {"failure_domain_identity", "filesystem_quota_bytes"},
            "site_evidence_entry_invalid",
        )
        if (
            not isinstance(row["failure_domain_identity"], str)
            or not SHA256_RE.fullmatch(row["failure_domain_identity"])
            or isinstance(row["filesystem_quota_bytes"], bool)
            or not isinstance(row["filesystem_quota_bytes"], int)
            or row["filesystem_quota_bytes"] < 0
        ):
            raise SiteError("site_evidence_entry_invalid")
        result[str(slot)] = dict(row)
    return result, digest_bytes(raw)


def _capability_result(passed: bool, code: str) -> dict[str, object]:
    return {"passed": bool(passed), "reason_code": code}


def _probe_capabilities(path: Path, writer_count: int) -> dict[str, object]:
    results: dict[str, object] = {}
    try:
        with tempfile.TemporaryDirectory(prefix=".portable-site-", dir=path) as raw:
            root = Path(raw)
            source = root / "source"
            target = root / "target"
            with source.open("wb") as handle:
                handle.write(b"new")
                handle.flush()
                os.fsync(handle.fileno())
            results["file_fsync"] = _capability_result(
                True, "file_fsync_verified"
            )
            target.write_bytes(b"old")
            os.replace(source, target)
            results["atomic_replace"] = _capability_result(
                target.read_bytes() == b"new", "atomic_replace_verified"
            )
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            results["directory_fsync"] = _capability_result(
                True, "directory_fsync_verified"
            )
            first = (root / "lock").open("a+b")
            second = (root / "lock").open("a+b")
            try:
                fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                rejected = False
                try:
                    fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    rejected = True
                results["advisory_lock"] = _capability_result(
                    rejected, "lock_contention_verified"
                )
            finally:
                first.close()
                second.close()
            marker = root / "permission"
            marker.write_bytes(b"x")
            marker.unlink()
            results["write_permission"] = _capability_result(
                True, "write_cleanup_verified"
            )
            case_path = root / "CaseProbe"
            case_path.write_bytes(b"x")
            results["case_semantics"] = _capability_result(
                True, "case_semantics_observed"
            )
            composed = root / "unicode-\u00e9"
            composed.write_bytes(b"x")
            _ = (root / unicodedata.normalize("NFD", composed.name)).exists()
            results["unicode_semantics"] = _capability_result(
                True, "unicode_semantics_observed"
            )
            restore = root / "restore"
            restore.mkdir()
            (restore / "committed").write_bytes(b"x")
            results["nonempty_restore_rejection"] = _capability_result(
                next(restore.iterdir(), None) is not None,
                "nonempty_restore_rejected",
            )
            handles = []
            try:
                for index in range(writer_count):
                    handles.append((root / f"writer-{index:03d}").open("wb"))
                writers_ok = len(handles) == writer_count
            finally:
                for handle in handles:
                    handle.close()
            results["temporary_directory"] = _capability_result(
                root.stat().st_dev == path.stat().st_dev,
                "same_filesystem_temporary_directory_verified",
            )
            results["writer_limit"] = _capability_result(
                writers_ok, "writer_limit_verified"
            )
    except (OSError, ValueError):
        for name in (*CAPABILITIES, "writer_limit"):
            results.setdefault(name, _capability_result(False, f"{name}_failed"))
    return {key: results[key] for key in sorted(results)}


def _probe_volume(
    slot: str,
    path: Path,
    requirement: dict[str, object],
    evidence: dict[str, object] | None,
) -> dict[str, object]:
    if not path.is_dir():
        raise SiteError("volume_path_unavailable")
    try:
        stats = os.statvfs(path)
        file_stats = path.stat()
    except OSError as exc:
        raise SiteError("volume_observation_failed") from exc
    filesystem_identity = stable_hash(
        {
            "device": int(file_stats.st_dev),
            "filesystem_id": str(getattr(stats, "f_fsid", NOT_AVAILABLE)),
            "fragment_size": int(stats.f_frsize),
        }
    )
    mount_identity = stable_hash(
        {
            "filesystem_identity": filesystem_identity,
            "name_max": int(stats.f_namemax),
        }
    )
    available_bytes = int(stats.f_bavail) * int(stats.f_frsize)
    available_inodes: object = (
        int(stats.f_favail) if int(stats.f_favail) >= 0 else NOT_AVAILABLE
    )
    quota: object = (
        evidence["filesystem_quota_bytes"] if evidence else NOT_AVAILABLE
    )
    fault_domain: object = (
        evidence["failure_domain_identity"] if evidence else NOT_AVAILABLE
    )
    capabilities = _probe_capabilities(
        path, int(requirement["required_concurrent_writers"])
    )
    checks = {
        "available_bytes": available_bytes
        >= int(requirement["required_bytes"]),
        "available_inodes": isinstance(available_inodes, int)
        and available_inodes >= int(requirement["required_inodes"]),
        "filesystem_quota_bytes": isinstance(quota, int)
        and quota >= int(requirement["required_bytes"]),
        "writer_limit": capabilities["writer_limit"]["passed"] is True,
    }
    qualified = all(checks.values()) and all(
        item["passed"] is True for item in capabilities.values()
    )
    return {
        "slot": slot,
        "role": requirement["role"],
        "filesystem_identity": filesystem_identity,
        "mount_identity": mount_identity,
        "failure_domain_identity": fault_domain,
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
        "filesystem_quota_bytes": quota,
        "writer_limit": int(requirement["required_concurrent_writers"]),
        "capabilities": capabilities,
        "checks": checks,
        "qualified": qualified,
    }


def build_attestation(
    contract: dict[str, object],
    volumes: dict[str, dict[str, object]],
    *,
    observation_epoch: int,
    site_evidence_sha256: str,
    synthetic_only: bool,
) -> dict[str, object]:
    topology = contract["topology"]
    expected_slots = set(topology["requirements"])
    if set(volumes) != expected_slots:
        raise SiteError("attestation_volume_inventory_invalid")
    fault_domains_ok = True
    for assignment in topology["shard_assignments"]:
        primary = volumes[assignment["primary_slot"]]
        backup = volumes[assignment["backup_slot"]]
        if (
            primary["failure_domain_identity"] == NOT_AVAILABLE
            or backup["failure_domain_identity"] == NOT_AVAILABLE
            or primary["failure_domain_identity"]
            == backup["failure_domain_identity"]
        ):
            fault_domains_ok = False
            break
    qualified = (
        all(item["qualified"] is True for item in volumes.values())
        and fault_domains_ok
        and len(
            {
                item["filesystem_identity"]
                for item in volumes.values()
            }
        )
        == len(volumes)
        and site_evidence_sha256 != NOT_AVAILABLE
    )
    payload: dict[str, object] = {
        "attestation": ATTESTATION,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": contract["source_commit"],
        "plan_sha256": contract["plan_binding"]["plan_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "topology_sha256": topology["topology_sha256"],
        "challenge_id": contract["challenge"]["challenge_id"],
        "observation_epoch": observation_epoch,
        "site_evidence_sha256": site_evidence_sha256,
        "runtime": {
            "architecture": platform.machine() or NOT_AVAILABLE,
            "os_family": platform.system() or NOT_AVAILABLE,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "volumes": [volumes[key] for key in sorted(volumes)],
        "failure_domains_verified": fault_domains_ok,
        "status": (
            "execution_site_qualified"
            if qualified
            else "not_ready_no_qualified_external_site"
        ),
        "synthetic_only": synthetic_only,
        "identity_authentication": False,
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
            "challenge_id",
            "contract_sha256",
            "failure_domains_verified",
            "formal_validation_complete",
            "identity_authentication",
            "observation_epoch",
            "plan_sha256",
            "protocol",
            "runtime",
            "schema_version",
            "site_evidence_sha256",
            "source_commit",
            "status",
            "synthetic_only",
            "topology_sha256",
            "volumes",
        },
        "attestation_schema_invalid",
    )
    payload = dict(attestation)
    claimed = payload.pop("attestation_sha256")
    if not isinstance(claimed, str) or stable_hash(payload) != claimed:
        raise SiteError("attestation_digest_invalid")
    if (
        attestation["attestation"] != ATTESTATION
        or attestation["protocol"] != PROTOCOL
        or attestation["schema_version"] != SCHEMA_VERSION
        or attestation["formal_validation_complete"] is not False
        or attestation["identity_authentication"] is not False
        or attestation["source_commit"] != contract["source_commit"]
        or attestation["plan_sha256"]
        != contract["plan_binding"]["plan_sha256"]
        or attestation["contract_sha256"] != contract["contract_sha256"]
        or attestation["topology_sha256"]
        != contract["topology"]["topology_sha256"]
        or attestation["challenge_id"]
        != contract["challenge"]["challenge_id"]
        or isinstance(attestation["observation_epoch"], bool)
        or not isinstance(attestation["observation_epoch"], int)
        or attestation["observation_epoch"]
        < contract["challenge"]["issued_epoch"]
        or attestation["observation_epoch"]
        > (
            contract["challenge"]["issued_epoch"]
            + contract["challenge"]["max_age_seconds"]
        )
        or not isinstance(attestation["synthetic_only"], bool)
        or not isinstance(attestation["site_evidence_sha256"], str)
        or (
            attestation["site_evidence_sha256"] != NOT_AVAILABLE
            and not SHA256_RE.fullmatch(attestation["site_evidence_sha256"])
        )
    ):
        raise SiteError("attestation_binding_invalid")
    runtime = _exact_object(
        attestation["runtime"],
        {
            "architecture",
            "os_family",
            "python_implementation",
            "python_version",
        },
        "attestation_runtime_invalid",
    )
    if any(not isinstance(runtime[key], str) or not runtime[key] for key in runtime):
        raise SiteError("attestation_runtime_invalid")
    volumes = attestation["volumes"]
    if not isinstance(volumes, list):
        raise SiteError("attestation_volume_inventory_invalid")
    by_slot: dict[str, dict[str, object]] = {}
    for raw in volumes:
        row = _exact_object(
            raw,
            {
                "available_bytes",
                "available_inodes",
                "capabilities",
                "checks",
                "failure_domain_identity",
                "filesystem_identity",
                "filesystem_quota_bytes",
                "mount_identity",
                "qualified",
                "role",
                "slot",
                "writer_limit",
            },
            "attestation_volume_entry_invalid",
        )
        if not isinstance(row.get("slot"), str):
            raise SiteError("attestation_volume_entry_invalid")
        if row["slot"] in by_slot:
            raise SiteError("duplicate_attestation_volume")
        capabilities = _exact_object(
            row["capabilities"],
            set(CAPABILITIES) | {"writer_limit"},
            "attestation_capabilities_invalid",
        )
        for name, item in capabilities.items():
            capability = _exact_object(
                item,
                {"passed", "reason_code"},
                "attestation_capability_invalid",
            )
            if (
                not isinstance(capability["passed"], bool)
                or not isinstance(capability["reason_code"], str)
                or not capability["reason_code"]
            ):
                raise SiteError("attestation_capability_invalid")
        checks = _exact_object(
            row["checks"],
            {
                "available_bytes",
                "available_inodes",
                "filesystem_quota_bytes",
                "writer_limit",
            },
            "attestation_checks_invalid",
        )
        if any(not isinstance(value, bool) for value in checks.values()):
            raise SiteError("attestation_checks_invalid")
        if (
            row["role"] not in {"primary", "backup"}
            or not isinstance(row["filesystem_identity"], str)
            or not SHA256_RE.fullmatch(row["filesystem_identity"])
            or not isinstance(row["mount_identity"], str)
            or not SHA256_RE.fullmatch(row["mount_identity"])
            or not isinstance(row["failure_domain_identity"], str)
            or (
                row["failure_domain_identity"] != NOT_AVAILABLE
                and not SHA256_RE.fullmatch(row["failure_domain_identity"])
            )
            or not isinstance(row["qualified"], bool)
            or row["qualified"]
            is not (
                all(checks.values())
                and all(item["passed"] for item in capabilities.values())
            )
        ):
            raise SiteError("attestation_volume_entry_invalid")
        by_slot[row["slot"]] = row
    requirements = contract["topology"]["requirements"]
    if set(by_slot) != set(requirements):
        raise SiteError("attestation_volume_inventory_invalid")
    derived_qualified = True
    for slot, requirement in requirements.items():
        row = by_slot[slot]
        writer_limit = row.get("writer_limit")
        if (
            row.get("role") != requirement["role"]
            or row.get("qualified") is not True
            or not isinstance(row.get("available_bytes"), int)
            or row["available_bytes"] < requirement["required_bytes"]
            or not isinstance(row.get("available_inodes"), int)
            or row["available_inodes"] < requirement["required_inodes"]
            or not isinstance(row.get("filesystem_quota_bytes"), int)
            or row["filesystem_quota_bytes"] < requirement["required_bytes"]
            or isinstance(writer_limit, bool)
            or not isinstance(writer_limit, int)
            or writer_limit < requirement["required_concurrent_writers"]
            or row.get("failure_domain_identity") == NOT_AVAILABLE
        ):
            derived_qualified = False
    if attestation["failure_domains_verified"] is not True:
        derived_qualified = False
    expected_fault_domains = True
    for assignment in contract["topology"]["shard_assignments"]:
        primary = by_slot[assignment["primary_slot"]]
        backup = by_slot[assignment["backup_slot"]]
        if (
            primary["failure_domain_identity"] == NOT_AVAILABLE
            or backup["failure_domain_identity"] == NOT_AVAILABLE
            or primary["failure_domain_identity"]
            == backup["failure_domain_identity"]
        ):
            expected_fault_domains = False
            break
    if attestation["failure_domains_verified"] is not expected_fault_domains:
        raise SiteError("attestation_failure_domain_status_invalid")
    if len(
        {
            row.get("filesystem_identity")
            for row in by_slot.values()
        }
    ) != len(by_slot):
        derived_qualified = False
    expected_status = (
        "execution_site_qualified"
        if derived_qualified
        else "not_ready_no_qualified_external_site"
    )
    if attestation["status"] != expected_status:
        raise SiteError("attestation_status_inconsistent")
    if require_qualified and expected_status != "execution_site_qualified":
        raise SiteError("attestation_not_qualified")
    return attestation


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--contract", required=True)
    probe.add_argument("--volume", action="append", default=[])
    probe.add_argument("--site-evidence")
    probe.add_argument("--observation-epoch", required=True, type=int)
    probe.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--contract", required=True)
    verify.add_argument("--attestation", required=True)
    return parser


def _result(status: str, code: int, **values: object) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": code,
        "formal_validation_complete": False,
        **values,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        contract = validate_contract(read_object(Path(args.contract)))
        if args.command == "probe":
            volume_paths = _parse_volume_rows(args.volume)
            requirements = contract["topology"]["requirements"]
            if set(volume_paths) != set(requirements):
                raise SiteError("volume_binding_inventory_invalid")
            evidence, evidence_sha256 = _site_evidence(
                Path(args.site_evidence) if args.site_evidence else None,
                contract,
            )
            volumes = {
                slot: _probe_volume(
                    slot,
                    volume_paths[slot],
                    requirements[slot],
                    evidence.get(slot),
                )
                for slot in sorted(volume_paths)
            }
            attestation = build_attestation(
                contract,
                volumes,
                observation_epoch=args.observation_epoch,
                site_evidence_sha256=evidence_sha256,
                synthetic_only=False,
            )
            write_object(Path(args.output), attestation)
            report = _result(
                attestation["status"],
                EXIT_QUALIFIED
                if attestation["status"] == "execution_site_qualified"
                else EXIT_NOT_READY,
                attestation_sha256=attestation["attestation_sha256"],
            )
        else:
            attestation = validate_attestation(
                read_object(Path(args.attestation)),
                contract,
                require_qualified=False,
            )
            report = _result(
                attestation["status"],
                EXIT_QUALIFIED
                if attestation["status"] == "execution_site_qualified"
                else EXIT_NOT_READY,
                attestation_sha256=attestation["attestation_sha256"],
            )
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except (SiteError, OSError, TypeError, ValueError) as exc:
        report = _result(
            "attestation_or_import_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc) if isinstance(exc, SiteError) else "controlled_io_failure"
            ),
        )
    sys.stdout.buffer.write(canonical_bytes(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
