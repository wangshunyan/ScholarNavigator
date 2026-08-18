from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_member_enrollment import (
    EXIT_NOT_READY, EXIT_READY, BackupMemberEnrollmentError, audit_readiness,
    build_contract, build_kit, canonical_json, contract_from_kit, load_protocol,
    run_enrollment, simulate_matrix, verify_kit, verify_member_package,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_backup_member_enrollment_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_member_enrollment.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _kit(protocol: dict[str, object], tmp_path: Path, *, count: int = 4, slot: int = 0) -> Path:
    path = tmp_path / "kit.zip"
    build_kit(ROOT, protocol, member_count=count, slot=slot,
              challenge_id=stable_hash({"test": "challenge", "slot": slot}),
              issued_epoch=10_000, output=path)
    return path


def _fixture(protocol: dict[str, object], tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    target = (tmp_path / "target").resolve(); target.mkdir()
    contract = build_contract(protocol, member_count=4, slot=0,
                              challenge_id=stable_hash({"fixture": 1}), issued_epoch=10_000,
                              repository_root=ROOT)
    req = contract["slot_contract"]["slot_requirements"]
    identity = lambda name: stable_hash({"identity": name})
    observed = {"filesystem_identity": identity("fs"), "device_identity": identity("device"),
                "available_bytes": req["minimum_available_bytes"],
                "available_inodes": req["minimum_available_inodes"],
                "capabilities": {name: True for name in (
                    "advisory_lock", "atomic_replace", "concurrent_writer", "directory_fsync",
                    "empty_restore", "file_fsync", "incremental_parent_chain", "path_length",
                    "write_verify_delete")}}
    evidence = {"challenge_id": contract["slot_contract"]["challenge"]["challenge_id"],
                "evidence_type": "independent_physical_device_and_management_domain",
                "expires_epoch": 96_400, "maximum_file_size_bytes": 35_030_827_008,
                "primary_device_identity": identity("primary-device"),
                "primary_failure_domain_identity": identity("primary-domain"),
                "primary_filesystem_identity": identity("primary-fs"),
                "primary_management_domain_identity": identity("primary-management"),
                "quota_bytes": req["minimum_quota_bytes"], "quota_pool_identity": identity("quota"),
                "recovery_verified": True, "reserved_bytes": req["minimum_available_bytes"],
                "revoked": False, "storage_service_identity": "not_available",
                "target_device_identity": observed["device_identity"],
                "target_failure_domain_identity": identity("target-domain"),
                "target_filesystem_identity": observed["filesystem_identity"],
                "target_management_domain_identity": identity("target-management")}
    return contract, target, evidence, observed


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT,
                          env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")},
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def test_protocol_and_current_readiness_are_closed(protocol: dict[str, object]) -> None:
    assert protocol["policy"]["candidate_status"] == "member_candidate_ready_for_intake"
    report = audit_readiness()
    assert report == audit_readiness()
    assert report["exit_code"] == EXIT_NOT_READY
    assert [(row["enrolled_slots"], row["missing_slots"]) for row in report["plans"]] == [(0,2),(0,3),(0,4)]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_each_topology_slot_uses_existing_intake_contract(protocol: dict[str, object], count: int) -> None:
    for slot in range(count):
        contract = build_contract(protocol, member_count=count, slot=slot,
                                  challenge_id=stable_hash({"count": count, "slot": slot}),
                                  issued_epoch=10_000, repository_root=ROOT)
        assert contract["slot_contract"]["member_count"] == count
        assert contract["slot_contract"]["slot"] == slot
        assert contract["slot_contract"]["slot_requirements"]["minimum_quota_bytes"] == contract["slot_contract"]["slot_requirements"]["minimum_available_bytes"]


def test_kit_is_byte_deterministic_and_path_free(protocol: dict[str, object], tmp_path: Path) -> None:
    first = _kit(protocol, tmp_path / "a")
    second = _kit(protocol, tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
    verify_kit(first, protocol, repository_root=ROOT)
    raw = first.read_bytes()
    assert str(tmp_path).encode() not in raw
    assert b".env" not in raw


def test_two_no_repository_isolated_environments(protocol: dict[str, object], tmp_path: Path) -> None:
    kit = _kit(protocol, tmp_path / "source")
    outputs = []
    for name in ("site-a", "site-b"):
        site = tmp_path / name; site.mkdir()
        with zipfile.ZipFile(kit) as archive:
            archive.extract("enroll.py", site); archive.extract("contract.json", site)
        completed = subprocess.run([sys.executable, "-I", "-S", str(site / "enroll.py"),
                                    "verify-contract", "--contract", str(site / "contract.json")],
                                   cwd=site, env={"PATH": os.environ.get("PATH", ""),
                                   "HOME": str(site / "home"), "TMPDIR": str(site / "tmp")},
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert completed.returncode == 0 and completed.stderr == b""
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_valid_enrollment_is_directly_accepted_by_member_intake(protocol: dict[str, object], tmp_path: Path) -> None:
    contract, target, evidence, observed = _fixture(protocol, tmp_path)
    package = run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                             synthetic_only=True, observed=observed)
    validated = verify_member_package(ROOT, contract, package, observation_epoch=10_100,
                                      require_real=False)
    assert validated["status"] == "member_candidate_ready_for_intake"
    assert validated["activation_side_effect"] is False
    serialized = canonical_json(validated).decode()
    assert str(target) not in serialized


@pytest.mark.parametrize("mutation,reason", [
    (lambda e,o,c: e.__setitem__("quota_bytes", "not_available"), "domain_evidence_invalid"),
    (lambda e,o,c: o.__setitem__("available_bytes", 0), "target_not_qualified"),
    (lambda e,o,c: o.__setitem__("available_inodes", 0), "target_not_qualified"),
    (lambda e,o,c: e.__setitem__("target_failure_domain_identity", e["primary_failure_domain_identity"]), "target_not_qualified"),
    (lambda e,o,c: e.__setitem__("revoked", True), "domain_evidence_invalid"),
    (lambda e,o,c: o.__setitem__("device_identity", stable_hash({"drift": 1})), "target_identity_drift"),
])
def test_fail_closed_profiles(protocol: dict[str, object], tmp_path: Path, mutation, reason: str) -> None:
    contract, target, evidence, observed = _fixture(protocol, tmp_path)
    mutation(evidence, observed, contract)
    with pytest.raises(BackupMemberEnrollmentError, match=reason):
        run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                       synthetic_only=True, observed=observed)


def test_challenge_replay_and_occupied_identity_fail(protocol: dict[str, object], tmp_path: Path) -> None:
    contract, target, evidence, observed = _fixture(protocol, tmp_path)
    consumed: set[str] = set()
    run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                   synthetic_only=True, observed=observed, consumed_challenges=consumed)
    with pytest.raises(BackupMemberEnrollmentError, match="challenge_replayed"):
        run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                       synthetic_only=True, observed=observed, consumed_challenges=consumed)
    with pytest.raises(BackupMemberEnrollmentError, match="target_identity_already_occupied"):
        run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                       synthetic_only=True, observed=observed,
                       occupied_identities={evidence["quota_pool_identity"]})


def test_symlink_and_missing_paths_fail(protocol: dict[str, object], tmp_path: Path) -> None:
    contract, target, evidence, observed = _fixture(protocol, tmp_path)
    alias = tmp_path / "alias"; alias.symlink_to(target, target_is_directory=True)
    for path in (alias, tmp_path / "missing"):
        with pytest.raises(BackupMemberEnrollmentError):
            run_enrollment(ROOT, contract, path, evidence, observation_epoch=10_100,
                           synthetic_only=True, observed=observed)


def test_package_tamper_and_slot_mismatch_fail(protocol: dict[str, object], tmp_path: Path) -> None:
    contract, target, evidence, observed = _fixture(protocol, tmp_path)
    package = run_enrollment(ROOT, contract, target, evidence, observation_epoch=10_100,
                             synthetic_only=True, observed=observed)
    tampered = copy.deepcopy(package); tampered["slot"] = 1
    with pytest.raises(BackupMemberEnrollmentError):
        verify_member_package(ROOT, contract, tampered, observation_epoch=10_100, require_real=False)


def test_simulation_matrix_and_cli_are_deterministic(protocol: dict[str, object], tmp_path: Path) -> None:
    first = simulate_matrix(protocol, repository_root=ROOT, temporary_root=tmp_path / "one")
    second = simulate_matrix(protocol, repository_root=ROOT, temporary_root=tmp_path / "two")
    assert first == second
    assert first["exit_code"] == EXIT_READY and first["passed_count"] == first["scenario_count"] == 14
    cli_first = _run_cli("simulate-matrix"); cli_second = _run_cli("simulate-matrix")
    assert cli_first.returncode == cli_second.returncode == 0
    assert cli_first.stdout == cli_second.stdout and cli_first.stderr == cli_second.stderr == b""
    readiness = _run_cli("audit-readiness")
    assert readiness.returncode == 3 and json.loads(readiness.stdout)["real_enrolled_member_count"] == 0


def test_kit_tamper_and_cross_slot_reuse_fail(protocol: dict[str, object], tmp_path: Path) -> None:
    kit = _kit(protocol, tmp_path / "a")
    contract = contract_from_kit(kit, protocol, repository_root=ROOT)
    other = build_contract(protocol, member_count=4, slot=1,
                           challenge_id=contract["slot_contract"]["challenge"]["challenge_id"],
                           issued_epoch=10_000, repository_root=ROOT)
    assert other["contract_sha256"] != contract["contract_sha256"]
    with zipfile.ZipFile(kit, "a") as archive:
        archive.writestr("extra", b"tamper")
    with pytest.raises(BackupMemberEnrollmentError): verify_kit(kit, protocol, repository_root=ROOT)
