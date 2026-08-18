from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_set_member_intake import CAPABILITIES
from scholar_agent.evaluation.formal_backup_set_topology import capacity_model
from scholar_agent.evaluation.formal_backup_target_registration import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    NOT_AVAILABLE,
    PRIVATE_REGISTRATION,
    PROBE_SCOPE,
    PURPOSE,
    SOURCE_COMMIT,
    BackupTargetRegistrationError,
    audit_readiness,
    build_registration_manifest,
    canonical_json,
    load_private_registration,
    load_protocol,
    simulate_profiles,
    validate_manifest,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_backup_target_registration_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_target_registration.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _registration(
    protocol: dict[str, object], paths: list[Path], *, revoked: list[str] | None = None
) -> dict[str, object]:
    return {
        "registration": PRIVATE_REGISTRATION,
        "schema_version": "1",
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
        "revoked_aliases": revoked or [],
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def _observation(
    alias: str,
    _path: Path,
    *,
    seed: str | None = None,
    available_bytes: int | None = None,
    available_inodes: int | None = None,
    capability: bool = True,
) -> dict[str, object]:
    member = capacity_model(4)["members"][3]
    identity = lambda role: stable_hash({"seed": seed or alias, "role": role})
    return {
        "target_alias": alias,
        "target_present": True,
        "available_bytes": member["required_bytes"] if available_bytes is None else available_bytes,
        "available_inodes": member["required_inodes"] if available_inodes is None else available_inodes,
        "quota_bytes": NOT_AVAILABLE,
        "writers": 2,
        "device_identity": identity("device"),
        "filesystem_identity": identity("filesystem"),
        "quota_pool_identity": NOT_AVAILABLE,
        "failure_domain_identity": NOT_AVAILABLE,
        "failure_domain_independent": NOT_AVAILABLE,
        "management_domain_identity": NOT_AVAILABLE,
        "capabilities": {name: capability for name in CAPABILITIES},
        "recovery_verified": NOT_AVAILABLE,
        "fresh": True,
        "synthetic_only": True,
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_current_audit_has_no_private_registration_and_is_not_ready(
    protocol: dict[str, object],
) -> None:
    first = audit_readiness(protocol)
    assert first == audit_readiness(protocol)
    assert first["exit_code"] == EXIT_NOT_READY
    assert first["registered_candidate_count"] == 0
    assert first["private_registration_committed"] is False


def test_private_registration_is_strict_and_never_embedded(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    private = _registration(protocol, [target])
    registration_path = tmp_path / "private.json"
    _write(registration_path, private)
    loaded = load_private_registration(registration_path, protocol=protocol)
    manifest = build_registration_manifest(
        protocol,
        loaded,
        repository_root=ROOT,
        observer=_observation,
        synthetic_only=True,
    )
    serialized = canonical_json(manifest).decode()
    assert manifest["status"] == "registered_candidates_ready"
    assert manifest["registered_candidate_count"] == 1
    assert str(target) not in serialized
    assert "registered_candidate" in serialized
    assert manifest["qualification_boundary"] == {
        "candidate_is_qualified_member": False,
        "target_attestation_required": True,
        "member_intake_required": True,
        "backup_set_activated": False,
        "formal_validation_complete": False,
    }


def test_quota_and_failure_domain_remain_downstream_gaps(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    manifest = build_registration_manifest(
        protocol,
        _registration(protocol, [target]),
        repository_root=ROOT,
        observer=_observation,
        synthetic_only=True,
    )
    probe = manifest["entries"][0]["probe"]
    assert probe["quota_observability"] == NOT_AVAILABLE
    assert probe["failure_domain_observability"] == NOT_AVAILABLE
    assert manifest["discovery_topology_match"]["plans"][0]["status"] == "not_ready_missing_qualified_candidates"


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_missing_and_symlink_paths_fail_closed(
    protocol: dict[str, object], tmp_path: Path, kind: str
) -> None:
    real = (tmp_path / "real").resolve()
    real.mkdir()
    target = (tmp_path / "missing").resolve()
    if kind == "symlink":
        target.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        BackupTargetRegistrationError,
        match="registered_path_unavailable" if kind == "missing" else "registered_path_alias_or_symlink",
    ):
        build_registration_manifest(
            protocol,
            _registration(protocol, [target]),
            repository_root=ROOT,
            observer=_observation,
            synthetic_only=True,
        )


def test_duplicate_path_and_duplicate_device_fail(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    targets = [(tmp_path / name).resolve() for name in ("a", "b")]
    for target in targets:
        target.mkdir()
    private = _registration(protocol, targets)
    private["targets"][1]["path"] = private["targets"][0]["path"]
    private_path = tmp_path / "private.json"
    _write(private_path, private)
    with pytest.raises(BackupTargetRegistrationError, match="private_target_invalid"):
        load_private_registration(private_path, protocol=protocol)
    with pytest.raises(BackupTargetRegistrationError, match="duplicate_device_or_filesystem"):
        build_registration_manifest(
            protocol,
            _registration(protocol, targets),
            repository_root=ROOT,
            observer=lambda alias, path: _observation(alias, path, seed="same"),
            synthetic_only=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("available_bytes", capacity_model(4)["members"][3]["required_bytes"] - 1),
        ("available_inodes", capacity_model(4)["members"][3]["required_inodes"] - 1),
    ],
)
def test_capacity_and_inode_minima_are_not_lowered(
    protocol: dict[str, object], tmp_path: Path, field: str, value: int
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    kwargs = {field: value}
    with pytest.raises(BackupTargetRegistrationError, match="probe_capacity_insufficient"):
        build_registration_manifest(
            protocol,
            _registration(protocol, [target]),
            repository_root=ROOT,
            observer=lambda alias, path: _observation(alias, path, **kwargs),
            synthetic_only=True,
        )


def test_probe_or_cleanup_failure_fails_closed(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    with pytest.raises(BackupTargetRegistrationError, match="probe_capability_failed"):
        build_registration_manifest(
            protocol,
            _registration(protocol, [target]),
            repository_root=ROOT,
            observer=lambda alias, path: _observation(alias, path, capability=False),
            synthetic_only=True,
        )


def test_revocation_prevents_candidate_use(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    manifest = build_registration_manifest(
        protocol,
        _registration(protocol, [target], revoked=["backup-target-0"]),
        repository_root=ROOT,
        observer=_observation,
        synthetic_only=True,
    )
    assert manifest["registered_candidate_count"] == 0
    assert manifest["entries"][0]["status"] == "revoked"


def test_manifest_detects_path_device_and_commit_drift(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    registration = _registration(protocol, [target])
    manifest = build_registration_manifest(
        protocol,
        registration,
        repository_root=ROOT,
        observer=_observation,
        synthetic_only=True,
    )
    assert validate_manifest(
        protocol,
        registration,
        manifest,
        repository_root=ROOT,
        observer=_observation,
    ) == manifest
    changed = lambda alias, path: _observation(alias, path, seed="replacement")
    with pytest.raises(BackupTargetRegistrationError, match="registration_manifest_probe_drift"):
        validate_manifest(
            protocol,
            registration,
            manifest,
            repository_root=ROOT,
            observer=changed,
        )
    tampered = copy.deepcopy(manifest)
    tampered["source_commit"] = "0" * 40
    tampered["manifest_sha256"] = stable_hash({key: value for key, value in tampered.items() if key != "manifest_sha256"})
    with pytest.raises(BackupTargetRegistrationError, match="registration_manifest_binding_invalid"):
        validate_manifest(
            protocol,
            registration,
            tampered,
            repository_root=ROOT,
            observer=_observation,
        )


def test_live_probe_cleans_temporary_files_and_never_serializes_path(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    before = set(target.iterdir())
    try:
        manifest = build_registration_manifest(
            protocol,
            _registration(protocol, [target]),
            repository_root=ROOT,
            synthetic_only=False,
        )
    except BackupTargetRegistrationError as exc:
        if str(exc) == "probe_capacity_insufficient":
            pytest.skip("test host does not meet the frozen minimum slot capacity")
        raise
    assert set(target.iterdir()) == before
    assert str(target) not in canonical_json(manifest).decode()
    assert manifest["entries"][0]["probe"]["probe_cleanup_verified"] is True


def test_profile_matrix_and_report_are_byte_deterministic(
    protocol: dict[str, object]
) -> None:
    first = simulate_profiles(protocol, repository_root=ROOT)
    second = simulate_profiles(protocol, repository_root=ROOT)
    assert first["exit_code"] == EXIT_READY
    assert first["scenario_count"] == 12
    assert canonical_json(first) == canonical_json(second)


def test_cli_readiness_simulation_and_private_path_redaction(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    readiness = _run_cli("audit-readiness")
    assert readiness.returncode == EXIT_NOT_READY
    assert readiness.stderr == b""
    assert json.loads(readiness.stdout)["registered_candidate_count"] == 0
    simulation = _run_cli("simulate-profiles")
    assert simulation.returncode == EXIT_READY
    assert simulation.stderr == b""
    target = (tmp_path / "missing-private-target").resolve()
    private_path = tmp_path / "private.json"
    _write(private_path, _registration(protocol, [target]))
    failed = _run_cli("register-dry-run", "--registration", str(private_path))
    assert failed.returncode == EXIT_VIOLATION
    assert failed.stderr == b""
    assert str(target).encode() not in failed.stdout
    assert json.loads(failed.stdout)["reason_code"] == "registered_path_unavailable"


def test_cli_usage_is_exit_four() -> None:
    result = _run_cli("register-dry-run")
    assert result.returncode == EXIT_USAGE
    assert result.stderr == b""
