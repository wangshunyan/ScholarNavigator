"""Offline host attestation for a future authoritative Full1000 execution.

The attestation is an observational, fail-closed launch prerequisite.  It
probes small temporary files on explicitly selected filesystems, records only
redacted capability facts, and binds those facts to the frozen Full1000
contracts.  It neither authenticates a host nor starts retrieval.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None  # type: ignore[assignment]

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_execution_host_attestation_v1"
SCHEMA_VERSION = "1"
ATTESTATION_CONTRACT = "host_attestation_manifest_v1"
LAUNCH_ADDENDUM = "full1000_host_attestation_addendum_v1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FROZEN_PROTOCOL_SHA256 = (
    "2328b088f19f0e45eeec7c0d75c6cd983c32cb3d8783ba77613d83e0543a920e"
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
_HEX64 = r"^[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40}$"
_WINDOWS_LOCK_CONFLICT_ERRNOS = frozenset(
    value
    for value in (
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", None),
        getattr(errno, "EWOULDBLOCK", None),
    )
    if value is not None
)


class HostAttestationError(RuntimeError):
    """The protocol, attestation, or launch binding is invalid."""


class HostAttestationNotReady(HostAttestationError):
    """The observed host is incomplete or insufficient for activation."""


class CapabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed", "not_available"]
    reason_code: str
    observed: Any = None
    required: Any = None


class HostAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["host_attestation_manifest_v1"] = ATTESTATION_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol: Literal["formal_execution_host_attestation_v1"] = PROTOCOL
    protocol_sha256: str = Field(pattern=_HEX64)
    source_commit: str = Field(pattern=_COMMIT)
    observed_head: str = Field(pattern=_COMMIT)
    host_scope_identity: str = Field(pattern=_HEX64)
    authoritative_output_root_identity: str = Field(pattern=_HEX64)
    primary_target_identity: str = Field(pattern=_HEX64)
    backup_target_identity: str | Literal["not_available"]
    runtime: dict[str, str]
    capabilities: dict[str, CapabilityResult]
    storage_requirements: dict[str, dict[str, int]]
    binding_sha256s: dict[str, str]
    status: Literal["host_qualified", "not_ready_unverified_or_insufficient_host"]
    missing_observations: list[str]
    failed_capabilities: list[str]
    identity_authentication: Literal[False] = False
    formal_validation_complete: Literal[False] = False
    execution: dict[str, Any]
    attestation_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def validate_attestation(self) -> "HostAttestation":
        if self.missing_observations != sorted(set(self.missing_observations)):
            raise ValueError("missing observations must be sorted and unique")
        if self.failed_capabilities != sorted(set(self.failed_capabilities)):
            raise ValueError("failed capabilities must be sorted and unique")
        if set(self.binding_sha256s) != {
            "crash_consistency",
            "disaster_recovery",
            "execution_plan",
            "launch_control",
            "runtime_hermeticity",
            "storage_governance",
            "storage_plan",
        }:
            raise ValueError("attestation binding inventory mismatch")
        if self.execution != EXECUTION_ZERO:
            raise ValueError("offline execution contract drift")
        expected_status = (
            "host_qualified"
            if not self.missing_observations and not self.failed_capabilities
            else "not_ready_unverified_or_insufficient_host"
        )
        if self.status != expected_status:
            raise ValueError("attestation status inconsistent with capability facts")
        payload = self.model_dump(mode="json")
        digest = payload.pop("attestation_sha256")
        if stable_hash(payload) != digest:
            raise ValueError("attestation digest mismatch")
        _assert_redacted(payload)
        return self


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate json key")
        result[key] = value
    return result


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HostAttestationError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise HostAttestationError("json_input_not_object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or path.name == ".env"
        or path.parts[0] == "third_party"
    ):
        raise HostAttestationError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = _read_object(path)
    required = {
        "bindings",
        "capability_requirements",
        "execution",
        "formal_validation_complete",
        "host_identity",
        "launch_binding",
        "probe",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "storage_requirements",
    }
    if set(value) != required:
        raise HostAttestationError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise HostAttestationError("protocol_version_invalid")
    if value["source_commit"] != "e47c5393cdd9f67d38ab8e38749664ca0a3310a1":
        raise HostAttestationError("protocol_source_commit_invalid")
    if value["formal_validation_complete"] is not False:
        raise HostAttestationError("formal_validation_state_invalid")
    if value["execution"] != EXECUTION_ZERO:
        raise HostAttestationError("offline_execution_contract_drift")
    if value["protocol_sha256"] != _protocol_digest(value):
        raise HostAttestationError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise HostAttestationError("protocol_content_drift")
    expected_storage = {
        "backup": {"bytes": 2_119_029_489_664, "inodes": 210_940},
        "primary": {"bytes": 713_501_442_048, "inodes": 76_980},
    }
    if value["storage_requirements"] != expected_storage:
        raise HostAttestationError("storage_requirement_drift")
    requirements = value["capability_requirements"]
    if (
        requirements.get("minimum_nofile_soft") != 256
        or requirements.get("minimum_process_soft") != 64
        or requirements.get("minimum_path_max") != 1024
    ):
        raise HostAttestationError("resource_requirement_drift")
    if (
        value["probe"].get("available_bytes_lower_bound_granularity")
        != 1_073_741_824
        or value["probe"].get("available_inodes_lower_bound_granularity")
        != 100_000
    ):
        raise HostAttestationError("capacity_observation_granularity_drift")
    _validate_bindings(repository_root, value)
    return value


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "crash_consistency",
        "disaster_recovery",
        "execution_plan",
        "launch_control",
        "runtime_hermeticity",
        "storage_governance",
        "storage_plan",
    }:
        raise HostAttestationError("protocol_binding_inventory_invalid")
    for name, raw in sorted(bindings.items()):
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise HostAttestationError("protocol_binding_invalid")
        relative = _safe_relative(str(raw["path"]))
        expected = str(raw["sha256"])
        path = root / relative
        if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            raise HostAttestationError(f"bound_input_mismatch:{name}")


def _authoritative_output_root_identity(
    root: Path, protocol: Mapping[str, Any]
) -> str:
    launch_path = root / _safe_relative(
        str(protocol["bindings"]["launch_control"]["path"])
    )
    launch = _read_object(launch_path)
    output = launch.get("output")
    if not isinstance(output, dict):
        raise HostAttestationError("launch_output_contract_invalid")
    relative = _safe_relative(str(output.get("authoritative_run_root", "")))
    return stable_hash(
        {
            "launch_control_sha256": protocol["bindings"]["launch_control"]["sha256"],
            "authoritative_run_root": relative,
        }
    )


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if completed.returncode != 0:
        raise HostAttestationNotReady("git_identity_unavailable")
    value = completed.stdout.strip()
    if len(value) != 40:
        raise HostAttestationNotReady("git_identity_invalid")
    return value


def _filesystem_type(path: Path) -> str:
    if os.name == "nt":
        return NOT_AVAILABLE
    if platform.system() == "Darwin":
        mount_binary = Path("/sbin/mount")
        if not mount_binary.is_file():
            return NOT_AVAILABLE
        try:
            completed = subprocess.run(
                [str(mount_binary)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/sbin"},
            )
        except (OSError, subprocess.SubprocessError):
            return NOT_AVAILABLE
        resolved = path.resolve()
        matches: list[tuple[int, str]] = []
        for line in completed.stdout.splitlines():
            if " on " not in line or " (" not in line:
                continue
            _device, remainder = line.split(" on ", 1)
            mountpoint, options = remainder.rsplit(" (", 1)
            try:
                resolved.relative_to(Path(mountpoint))
            except ValueError:
                continue
            filesystem_type = options.split(",", 1)[0].rstrip(")").strip()
            if filesystem_type:
                matches.append((len(mountpoint), filesystem_type.casefold()))
        return max(matches)[1] if matches else NOT_AVAILABLE
    stat_binary = shutil.which("stat", path="/usr/bin:/bin")
    if stat_binary is None:
        return NOT_AVAILABLE
    arguments = [stat_binary, "-f", "-c", "%T", str(path)]
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError):
        return NOT_AVAILABLE
    value = completed.stdout.strip().casefold()
    return value if completed.returncode == 0 and value else NOT_AVAILABLE


def _limit_value(kind: int) -> int | Literal["not_available"]:
    if resource is None:
        return NOT_AVAILABLE
    try:
        soft, _hard = resource.getrlimit(kind)
    except (OSError, ValueError):
        return NOT_AVAILABLE
    if soft == resource.RLIM_INFINITY:
        return sys.maxsize
    return int(soft) if soft >= 0 else NOT_AVAILABLE


def _host_scope_identity() -> str:
    # Raw host labels never leave this function.  This digest is a reuse guard,
    # not host authentication and not a publisher identity.
    return stable_hash(
        {
            "node": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
        }
    )


def _target_facts(
    path: Path,
    role: str,
    *,
    byte_granularity: int,
    inode_granularity: int,
) -> dict[str, Any]:
    if byte_granularity <= 0 or inode_granularity <= 0:
        raise HostAttestationError("capacity_observation_granularity_invalid")
    statvfs = getattr(os, "statvfs", None)
    stats = statvfs(path) if callable(statvfs) else None
    raw_inodes: int | Literal["not_available"] = (
        int(stats.f_favail) if stats is not None and stats.f_favail >= 0 else NOT_AVAILABLE
    )
    inode_value: int | Literal["not_available"] = (
        raw_inodes - (raw_inodes % inode_granularity)
        if isinstance(raw_inodes, int)
        else NOT_AVAILABLE
    )
    raw_bytes = (
        int(stats.f_bavail) * int(stats.f_frsize)
        if stats is not None
        else int(shutil.disk_usage(path).free)
    )
    available_bytes = raw_bytes - (raw_bytes % byte_granularity)
    file_stats = path.stat()
    filesystem_type = _filesystem_type(path)
    fsid = getattr(stats, "f_fsid", NOT_AVAILABLE) if stats is not None else NOT_AVAILABLE
    fault_domain_identity = stable_hash(
        {
            "device": int(file_stats.st_dev),
            "filesystem_id": str(fsid),
            "filesystem_type": filesystem_type,
        }
    )
    return {
        "role": role,
        "available_bytes": available_bytes,
        "available_inodes": inode_value,
        "filesystem_quota_bytes": NOT_AVAILABLE,
        "filesystem_type": filesystem_type,
        "fault_domain_identity": fault_domain_identity,
        "target_identity": stable_hash(
            {
                "fault_domain_identity": fault_domain_identity,
                "role": role,
            }
        ),
    }


def _capability(
    status: Literal["passed", "failed", "not_available"],
    reason_code: str,
    *,
    observed: Any = None,
    required: Any = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "observed": observed,
        "required": required,
    }


def _test_atomic_filesystem(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=".host-attestation-", dir=root) as raw:
        temporary = Path(raw)
        source = temporary / "source.json"
        target = temporary / "target.json"
        lock_path = temporary / "writer.lock"

        try:
            with source.open("wb") as handle:
                handle.write(b'{"generation":2}\n')
                handle.flush()
                os.fsync(handle.fileno())
            results["file_fsync"] = _capability("passed", "file_fsync_verified")
        except OSError:
            results["file_fsync"] = _capability("failed", "file_fsync_failed")

        target.write_bytes(b'{"generation":1}\n')
        try:
            os.replace(source, target)
            replaced = target.read_bytes() == b'{"generation":2}\n'
            results["atomic_replace"] = _capability(
                "passed" if replaced else "failed",
                "same_filesystem_atomic_replace_verified"
                if replaced
                else "atomic_replace_content_mismatch",
            )
        except OSError:
            results["atomic_replace"] = _capability(
                "failed", "same_filesystem_atomic_replace_failed"
            )

        if os.name == "nt":
            results["directory_fsync"] = _capability(
                "not_available", "directory_fsync_not_supported"
            )
        else:
            try:
                directory_fd = os.open(temporary, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                results["directory_fsync"] = _capability(
                    "passed", "directory_fsync_verified"
                )
            except OSError:
                results["directory_fsync"] = _capability(
                    "failed", "directory_fsync_failed"
                )

        first = lock_path.open("a+b")
        second = lock_path.open("a+b")
        try:
            if os.name == "nt":
                first.write(b"\0")
                first.flush()
                first.seek(0)
                msvcrt.locking(first.fileno(), msvcrt.LK_NBLCK, 1)
                second.seek(0)
                try:
                    msvcrt.locking(second.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in _WINDOWS_LOCK_CONFLICT_ERRNOS:
                        results["advisory_lock"] = _capability(
                            "passed", "lock_contention_rejected"
                        )
                    else:
                        raise
                else:
                    results["advisory_lock"] = _capability(
                        "failed", "lock_contention_not_rejected"
                    )
            else:
                fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    results["advisory_lock"] = _capability(
                        "passed", "lock_contention_rejected"
                    )
                else:
                    results["advisory_lock"] = _capability(
                        "failed", "lock_contention_not_rejected"
                    )
        except OSError:
            results["advisory_lock"] = _capability(
                "failed", "advisory_lock_unavailable"
            )
        finally:
            if os.name == "nt":
                try:
                    first.seek(0)
                    msvcrt.locking(first.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            first.close()
            second.close()

        case_name = temporary / "CaseProbe"
        case_name.write_bytes(b"x")
        case_sensitive = not (temporary / "caseprobe").exists()
        results["case_semantics"] = _capability(
            "passed",
            "case_semantics_observed",
            observed={"case_sensitive": case_sensitive},
        )

        composed = "unicode-\u00e9"
        decomposed = unicodedata.normalize("NFD", composed)
        (temporary / composed).write_bytes(b"x")
        normalization_sensitive = not (temporary / decomposed).exists()
        results["unicode_semantics"] = _capability(
            "passed",
            "unicode_semantics_observed",
            observed={"normalization_sensitive": normalization_sensitive},
        )

        nonempty = temporary / "restore-target"
        nonempty.mkdir()
        (nonempty / "committed").write_bytes(b"x")
        try:
            _require_empty_restore_target(nonempty)
        except HostAttestationError:
            results["nonempty_restore_rejection"] = _capability(
                "passed", "nonempty_restore_rejected"
            )
        else:
            results["nonempty_restore_rejection"] = _capability(
                "failed", "nonempty_restore_accepted"
            )

        permission_root = temporary / "permission-probe"
        permission_root.mkdir()
        try:
            marker = permission_root / "marker"
            marker.write_bytes(b"x")
            marker.unlink()
            results["write_permission"] = _capability(
                "passed", "temporary_write_and_cleanup_verified"
            )
        except OSError:
            results["write_permission"] = _capability(
                "failed", "temporary_write_or_cleanup_failed"
            )

        results["temporary_directory"] = _capability(
            "passed",
            "same_filesystem_temporary_directory_verified",
            observed={
                "same_device": temporary.stat().st_dev == root.stat().st_dev,
                "cleanup_policy": "automatic",
            },
        )
    return results


def _require_empty_restore_target(path: Path) -> None:
    if path.exists() and (not path.is_dir() or next(path.iterdir(), None) is not None):
        raise HostAttestationError("restore_target_not_empty")


def _storage_capabilities(
    protocol: Mapping[str, Any],
    *,
    primary: Mapping[str, Any],
    backup: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    capabilities: dict[str, dict[str, Any]] = {}
    requirements = protocol["storage_requirements"]
    for role, facts in (("primary", primary), ("backup", backup)):
        required = requirements[role]
        if facts is None:
            for field in ("available_bytes", "available_inodes", "filesystem_quota_bytes"):
                capabilities[f"{role}_{field}"] = _capability(
                    "not_available", f"{role}_{field}_not_available", required=required[
                        "bytes" if field != "available_inodes" else "inodes"
                    ]
                )
            capabilities[f"{role}_filesystem_type"] = _capability(
                "not_available", f"{role}_filesystem_type_not_available"
            )
            continue
        for field, minimum in (
            ("available_bytes", required["bytes"]),
            ("available_inodes", required["inodes"]),
            ("filesystem_quota_bytes", required["bytes"]),
        ):
            observed = facts[field]
            if observed == NOT_AVAILABLE:
                status = "not_available"
                reason = f"{role}_{field}_not_available"
            elif int(observed) < minimum:
                status = "failed"
                reason = f"{role}_{field}_below_required"
            else:
                status = "passed"
                reason = f"{role}_{field}_verified"
            capabilities[f"{role}_{field}"] = _capability(
                status, reason, observed=observed, required=minimum
            )
        filesystem_type = facts["filesystem_type"]
        capabilities[f"{role}_filesystem_type"] = _capability(
            "not_available" if filesystem_type == NOT_AVAILABLE else "passed",
            f"{role}_filesystem_type_"
            + ("not_available" if filesystem_type == NOT_AVAILABLE else "observed"),
            observed=filesystem_type,
        )
    if backup is None:
        capabilities["independent_backup_fault_domain"] = _capability(
            "not_available", "backup_fault_domain_not_available"
        )
    else:
        distinct = (
            primary["fault_domain_identity"] != backup["fault_domain_identity"]
        )
        capabilities["independent_backup_fault_domain"] = _capability(
            "passed" if distinct else "failed",
            "independent_fault_domain_verified"
            if distinct
            else "primary_backup_share_fault_domain",
            observed={"distinct": distinct},
            required={"distinct": True},
        )
    return capabilities


def _resource_capabilities(protocol: Mapping[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    required = protocol["capability_requirements"]
    nofile_kind = getattr(resource, "RLIMIT_NOFILE", None) if resource is not None else None
    nofile = _limit_value(nofile_kind) if nofile_kind is not None else NOT_AVAILABLE
    nproc_kind = getattr(resource, "RLIMIT_NPROC", None) if resource is not None else None
    nproc = _limit_value(nproc_kind) if nproc_kind is not None else NOT_AVAILABLE
    try:
        path_max = int(os.pathconf(path, "PC_PATH_MAX"))
    except (AttributeError, OSError, ValueError):
        path_max = NOT_AVAILABLE

    def minimum_result(
        name: str, observed: int | Literal["not_available"], minimum: int
    ) -> dict[str, Any]:
        if observed == NOT_AVAILABLE:
            return _capability(
                "not_available", f"{name}_not_available", required=minimum
            )
        return _capability(
            "passed" if observed >= minimum else "failed",
            f"{name}_verified" if observed >= minimum else f"{name}_below_required",
            observed=observed,
            required=minimum,
        )

    return {
        "nofile_soft_limit": minimum_result(
            "nofile_soft_limit", nofile, int(required["minimum_nofile_soft"])
        ),
        "process_soft_limit": minimum_result(
            "process_soft_limit", nproc, int(required["minimum_process_soft"])
        ),
        "path_max": minimum_result(
            "path_max", path_max, int(required["minimum_path_max"])
        ),
    }


def _assemble_attestation(
    protocol: Mapping[str, Any],
    *,
    observed_head: str,
    host_scope_identity: str,
    authoritative_output_root_identity: str,
    primary_target_identity: str,
    backup_target_identity: str | Literal["not_available"],
    runtime: Mapping[str, str],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = sorted(
        name
        for name, item in capabilities.items()
        if item.get("status") == "not_available"
    )
    failed = sorted(
        name for name, item in capabilities.items() if item.get("status") == "failed"
    )
    status = (
        "host_qualified"
        if not missing and not failed
        else "not_ready_unverified_or_insufficient_host"
    )
    payload: dict[str, Any] = {
        "contract": ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "observed_head": observed_head,
        "host_scope_identity": host_scope_identity,
        "authoritative_output_root_identity": authoritative_output_root_identity,
        "primary_target_identity": primary_target_identity,
        "backup_target_identity": backup_target_identity,
        "runtime": dict(runtime),
        "capabilities": {
            key: dict(capabilities[key]) for key in sorted(capabilities)
        },
        "storage_requirements": protocol["storage_requirements"],
        "binding_sha256s": {
            key: protocol["bindings"][key]["sha256"]
            for key in sorted(protocol["bindings"])
        },
        "status": status,
        "missing_observations": missing,
        "failed_capabilities": failed,
        "identity_authentication": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    payload["attestation_sha256"] = stable_hash(payload)
    HostAttestation.model_validate(payload)
    return payload


def probe_host(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    primary_root: Path,
    backup_root: Path | None,
    observed_head: str | None = None,
    host_scope_identity: str | None = None,
) -> dict[str, Any]:
    _validate_bindings(repository_root, protocol)
    if not primary_root.is_dir():
        raise HostAttestationNotReady("primary_target_unavailable")
    byte_granularity = int(
        protocol["probe"]["available_bytes_lower_bound_granularity"]
    )
    inode_granularity = int(
        protocol["probe"]["available_inodes_lower_bound_granularity"]
    )
    primary = _target_facts(
        primary_root,
        "primary",
        byte_granularity=byte_granularity,
        inode_granularity=inode_granularity,
    )
    backup = (
        _target_facts(
            backup_root,
            "backup",
            byte_granularity=byte_granularity,
            inode_granularity=inode_granularity,
        )
        if backup_root is not None and backup_root.is_dir()
        else None
    )
    capabilities = _storage_capabilities(
        protocol, primary=primary, backup=backup
    )
    capabilities.update(_resource_capabilities(protocol, primary_root))
    try:
        capabilities.update(_test_atomic_filesystem(primary_root))
    except (OSError, HostAttestationError):
        for name in protocol["probe"]["filesystem_capabilities"]:
            capabilities.setdefault(
                name, _capability("failed", f"{name}_probe_failed")
            )
    runtime = {
        "architecture": platform.machine() or NOT_AVAILABLE,
        "filesystem_type_primary": str(primary["filesystem_type"]),
        "filesystem_type_backup": (
            str(backup["filesystem_type"]) if backup is not None else NOT_AVAILABLE
        ),
        "os_family": platform.system() or NOT_AVAILABLE,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    return _assemble_attestation(
        protocol,
        observed_head=observed_head or _git_head(repository_root),
        host_scope_identity=host_scope_identity or _host_scope_identity(),
        authoritative_output_root_identity=_authoritative_output_root_identity(
            repository_root, protocol
        ),
        primary_target_identity=str(primary["target_identity"]),
        backup_target_identity=(
            str(backup["target_identity"]) if backup is not None else NOT_AVAILABLE
        ),
        runtime=runtime,
        capabilities=capabilities,
    )


def validate_attestation(
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    expected_head: str | None = None,
    expected_host_scope: str | None = None,
    expected_output_root_identity: str | None = None,
    expected_primary_target: str | None = None,
    expected_backup_target: str | None = None,
    require_qualified: bool = False,
) -> HostAttestation:
    try:
        attestation = HostAttestation.model_validate(value)
    except ValidationError as exc:
        raise HostAttestationError("attestation_schema_or_digest_invalid") from exc
    if (
        attestation.protocol_sha256 != protocol["protocol_sha256"]
        or attestation.source_commit != protocol["source_commit"]
        or attestation.storage_requirements != protocol["storage_requirements"]
    ):
        raise HostAttestationError("attestation_protocol_binding_drift")
    expected_bindings = {
        key: protocol["bindings"][key]["sha256"] for key in sorted(protocol["bindings"])
    }
    if attestation.binding_sha256s != expected_bindings:
        raise HostAttestationError("attestation_bound_input_drift")
    _validate_capability_semantics(attestation, protocol)
    context_checks = {
        "attestation_commit_drift": (
            expected_head,
            attestation.observed_head,
        ),
        "attestation_host_scope_drift": (
            expected_host_scope,
            attestation.host_scope_identity,
        ),
        "attestation_primary_target_drift": (
            expected_primary_target,
            attestation.primary_target_identity,
        ),
        "attestation_output_root_drift": (
            expected_output_root_identity,
            attestation.authoritative_output_root_identity,
        ),
        "attestation_backup_target_drift": (
            expected_backup_target,
            attestation.backup_target_identity,
        ),
    }
    for reason, (expected, actual) in context_checks.items():
        if expected is not None and expected != actual:
            raise HostAttestationError(reason)
    if require_qualified and attestation.status != "host_qualified":
        raise HostAttestationNotReady("host_attestation_not_qualified")
    return attestation


def _required_capability_names(protocol: Mapping[str, Any]) -> set[str]:
    return {
        "advisory_lock",
        "atomic_replace",
        "backup_available_bytes",
        "backup_available_inodes",
        "backup_filesystem_quota_bytes",
        "backup_filesystem_type",
        "case_semantics",
        "directory_fsync",
        "file_fsync",
        "independent_backup_fault_domain",
        "nofile_soft_limit",
        "nonempty_restore_rejection",
        "path_max",
        "primary_available_bytes",
        "primary_available_inodes",
        "primary_filesystem_quota_bytes",
        "primary_filesystem_type",
        "process_soft_limit",
        "temporary_directory",
        "unicode_semantics",
        "write_permission",
    }


def _validate_capability_semantics(
    attestation: HostAttestation, protocol: Mapping[str, Any]
) -> None:
    if set(attestation.capabilities) != _required_capability_names(protocol):
        raise HostAttestationError("attestation_capability_inventory_drift")
    minimums = {
        "primary_available_bytes": protocol["storage_requirements"]["primary"]["bytes"],
        "primary_available_inodes": protocol["storage_requirements"]["primary"]["inodes"],
        "primary_filesystem_quota_bytes": protocol["storage_requirements"]["primary"]["bytes"],
        "backup_available_bytes": protocol["storage_requirements"]["backup"]["bytes"],
        "backup_available_inodes": protocol["storage_requirements"]["backup"]["inodes"],
        "backup_filesystem_quota_bytes": protocol["storage_requirements"]["backup"]["bytes"],
        "nofile_soft_limit": protocol["capability_requirements"]["minimum_nofile_soft"],
        "process_soft_limit": protocol["capability_requirements"]["minimum_process_soft"],
        "path_max": protocol["capability_requirements"]["minimum_path_max"],
    }
    for name, minimum in minimums.items():
        capability = attestation.capabilities[name]
        if capability.status == "passed" and (
            not isinstance(capability.observed, int)
            or isinstance(capability.observed, bool)
            or capability.observed < minimum
            or capability.required != minimum
        ):
            raise HostAttestationError("attestation_capability_claim_invalid")
    fault_domain = attestation.capabilities["independent_backup_fault_domain"]
    if fault_domain.status == "passed" and fault_domain.observed != {"distinct": True}:
        raise HostAttestationError("attestation_fault_domain_claim_invalid")
    if attestation.status == "host_qualified":
        if attestation.backup_target_identity == NOT_AVAILABLE:
            raise HostAttestationError("qualified_attestation_backup_target_missing")
        if any(value == NOT_AVAILABLE for value in attestation.runtime.values()):
            raise HostAttestationError("qualified_attestation_runtime_incomplete")


def validate_attestation_freshness(
    attestation: Mapping[str, Any],
    current_observation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> HostAttestation:
    sealed = validate_attestation(attestation, protocol)
    current = validate_attestation(current_observation, protocol)
    if sealed.attestation_sha256 != current.attestation_sha256:
        raise HostAttestationError("attestation_observation_drift")
    return sealed


def build_launch_addendum(
    protocol: Mapping[str, Any], attestation: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "addendum": LAUNCH_ADDENDUM,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "host_protocol_sha256": protocol["protocol_sha256"],
        "launch_control": protocol["bindings"]["launch_control"],
        "storage_governance": protocol["bindings"]["storage_governance"],
        "launch_authorization_requirements": {
            "attestation_status": "host_qualified",
            "attestation_fresh_for_exact_head": True,
            "host_scope_identity_match": True,
            "authoritative_output_root_identity_match": True,
            "primary_target_identity_match": True,
            "backup_target_identity_match": True,
            "legacy_authorization_reusable": False,
        },
        "attestation_sha256": (
            attestation.get("attestation_sha256") if attestation else NOT_AVAILABLE
        ),
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def bind_launch_authorization(
    authorization: Mapping[str, Any],
    attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    current_head: str,
    host_scope_identity: str,
    authoritative_output_root_identity: str,
    primary_target_identity: str,
    backup_target_identity: str,
    current_probe_sha256: str,
) -> dict[str, Any]:
    validated = validate_attestation(
        attestation,
        protocol,
        expected_head=current_head,
        expected_host_scope=host_scope_identity,
        expected_output_root_identity=authoritative_output_root_identity,
        expected_primary_target=primary_target_identity,
        expected_backup_target=backup_target_identity,
        require_qualified=True,
    )
    if current_probe_sha256 != validated.attestation_sha256:
        raise HostAttestationError("attestation_observation_drift")
    authorization_digest = authorization.get("authorization_sha256")
    if not isinstance(authorization_digest, str) or len(authorization_digest) != 64:
        raise HostAttestationError("launch_authorization_identity_invalid")
    payload: dict[str, Any] = {
        "contract": "host_bound_launch_authorization_v1",
        "schema_version": SCHEMA_VERSION,
        "launch_authorization_sha256": authorization_digest,
        "host_attestation_sha256": validated.attestation_sha256,
        "host_protocol_sha256": protocol["protocol_sha256"],
        "observed_head": current_head,
        "host_scope_identity": host_scope_identity,
        "authoritative_output_root_identity": authoritative_output_root_identity,
        "primary_target_identity": primary_target_identity,
        "backup_target_identity": backup_target_identity,
        "formal_validation_complete": False,
    }
    payload["host_bound_authorization_sha256"] = stable_hash(payload)
    return payload


def _qualified_profile(protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = protocol["storage_requirements"]
    capabilities = {
        "primary_available_bytes": _capability(
            "passed",
            "primary_available_bytes_verified",
            observed=required["primary"]["bytes"],
            required=required["primary"]["bytes"],
        ),
        "primary_available_inodes": _capability(
            "passed",
            "primary_available_inodes_verified",
            observed=required["primary"]["inodes"],
            required=required["primary"]["inodes"],
        ),
        "primary_filesystem_quota_bytes": _capability(
            "passed",
            "primary_filesystem_quota_bytes_verified",
            observed=required["primary"]["bytes"],
            required=required["primary"]["bytes"],
        ),
        "backup_available_bytes": _capability(
            "passed",
            "backup_available_bytes_verified",
            observed=required["backup"]["bytes"],
            required=required["backup"]["bytes"],
        ),
        "backup_available_inodes": _capability(
            "passed",
            "backup_available_inodes_verified",
            observed=required["backup"]["inodes"],
            required=required["backup"]["inodes"],
        ),
        "backup_filesystem_quota_bytes": _capability(
            "passed",
            "backup_filesystem_quota_bytes_verified",
            observed=required["backup"]["bytes"],
            required=required["backup"]["bytes"],
        ),
        "primary_filesystem_type": _capability(
            "passed", "primary_filesystem_type_observed", observed="fixturefs"
        ),
        "backup_filesystem_type": _capability(
            "passed", "backup_filesystem_type_observed", observed="backupfs"
        ),
        "independent_backup_fault_domain": _capability(
            "passed",
            "independent_fault_domain_verified",
            observed={"distinct": True},
            required={"distinct": True},
        ),
        "nofile_soft_limit": _capability(
            "passed", "nofile_soft_limit_verified", observed=256, required=256
        ),
        "process_soft_limit": _capability(
            "passed", "process_soft_limit_verified", observed=64, required=64
        ),
        "path_max": _capability(
            "passed", "path_max_verified", observed=1024, required=1024
        ),
    }
    for name in protocol["probe"]["filesystem_capabilities"]:
        capabilities[name] = _capability("passed", f"{name}_verified")
    return _assemble_attestation(
        protocol,
        observed_head=protocol["source_commit"],
        host_scope_identity=stable_hash({"fixture": "host-a"}),
        authoritative_output_root_identity=stable_hash(
            {"fixture": "authoritative-output-root"}
        ),
        primary_target_identity=stable_hash({"fixture": "primary-a"}),
        backup_target_identity=stable_hash({"fixture": "backup-a"}),
        runtime={
            "architecture": "fixture",
            "filesystem_type_primary": "fixturefs",
            "filesystem_type_backup": "backupfs",
            "os_family": "fixture",
            "python_implementation": "CPython",
            "python_version": "3.fixture",
        },
        capabilities=capabilities,
    )


def simulate_profiles(protocol: Mapping[str, Any]) -> dict[str, Any]:
    base = _qualified_profile(protocol)
    scenarios: dict[str, dict[str, Any]] = {
        "fully_qualified": {"value": base, "expected": "host_qualified"}
    }
    mutations = {
        "capacity_insufficient": (
            "primary_available_bytes",
            _capability(
                "failed",
                "primary_available_bytes_below_required",
                observed=1,
                required=713_501_442_048,
            ),
        ),
        "inode_insufficient": (
            "primary_available_inodes",
            _capability(
                "failed",
                "primary_available_inodes_below_required",
                observed=1,
                required=76_980,
            ),
        ),
        "directory_fsync_unavailable": (
            "directory_fsync",
            _capability("failed", "directory_fsync_failed"),
        ),
        "non_atomic_replace": (
            "atomic_replace",
            _capability("failed", "same_filesystem_atomic_replace_failed"),
        ),
        "advisory_lock_unavailable": (
            "advisory_lock",
            _capability("failed", "advisory_lock_unavailable"),
        ),
        "nofile_insufficient": (
            "nofile_soft_limit",
            _capability(
                "failed", "nofile_soft_limit_below_required", observed=32, required=256
            ),
        ),
        "shared_fault_domain": (
            "independent_backup_fault_domain",
            _capability(
                "failed",
                "primary_backup_share_fault_domain",
                observed={"distinct": False},
                required={"distinct": True},
            ),
        ),
        "observation_missing": (
            "backup_filesystem_quota_bytes",
            _capability(
                "not_available",
                "backup_filesystem_quota_bytes_not_available",
                required=2_119_029_489_664,
            ),
        ),
    }
    for scenario, (name, replacement) in mutations.items():
        value = copy.deepcopy(base)
        value["capabilities"][name] = replacement
        value.pop("attestation_sha256")
        missing = sorted(
            key
            for key, item in value["capabilities"].items()
            if item["status"] == "not_available"
        )
        failed = sorted(
            key
            for key, item in value["capabilities"].items()
            if item["status"] == "failed"
        )
        value["missing_observations"] = missing
        value["failed_capabilities"] = failed
        value["status"] = "not_ready_unverified_or_insufficient_host"
        value["attestation_sha256"] = stable_hash(value)
        scenarios[scenario] = {
            "value": value,
            "expected": "not_ready_unverified_or_insufficient_host",
        }

    results: dict[str, Any] = {}
    for name, scenario in sorted(scenarios.items()):
        validated = validate_attestation(scenario["value"], protocol)
        results[name] = {
            "status": "passed"
            if validated.status == scenario["expected"]
            else "failed",
            "observed_host_status": validated.status,
        }

    tampered = copy.deepcopy(base)
    tampered["host_scope_identity"] = "0" * 64
    try:
        validate_attestation(tampered, protocol)
    except HostAttestationError:
        results["attestation_tamper"] = {
            "status": "passed",
            "observed_host_status": "integrity_violation",
        }
    else:
        results["attestation_tamper"] = {
            "status": "failed",
            "observed_host_status": "accepted",
        }

    passed = all(item["status"] == "passed" for item in results.values())
    return _report(
        "host_controls_ready" if passed else "host_capability_violation",
        EXIT_QUALIFIED if passed else EXIT_VIOLATION,
        scenario_count=len(results),
        scenarios=results,
    )


def audit_readiness(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    attestation_path: Path,
) -> dict[str, Any]:
    value = _read_object(attestation_path)
    attestation = validate_attestation(value, protocol)
    return _report(
        "host_qualified"
        if attestation.status == "host_qualified"
        else "not_ready_unverified_or_insufficient_host",
        EXIT_QUALIFIED if attestation.status == "host_qualified" else EXIT_NOT_READY,
        attestation_sha256=attestation.attestation_sha256,
        observed_head=attestation.observed_head,
        missing_observations=attestation.missing_observations,
        failed_capabilities=attestation.failed_capabilities,
        full1000_blocker_cleared=False,
        real_run_started=False,
    )


def _assert_redacted(value: Any, *, field: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_redacted(child, field=str(key))
        return
    if isinstance(value, list):
        for child in value:
            _assert_redacted(child, field=field)
        return
    if not isinstance(value, str):
        return
    lowered = value.casefold()
    if value.startswith(("/Users/", "/home/", "file://")):
        raise ValueError("absolute path leak")
    if any(
        token in lowered
        for token in (".env", "api_key", "authorization:", "bearer ", "hostname")
    ):
        raise ValueError("sensitive host value leak")


def _report(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }
