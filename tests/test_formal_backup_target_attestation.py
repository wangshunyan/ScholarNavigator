from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_target_attestation import (
    EXIT_NOT_READY,
    EXIT_READY,
    BackupTargetError,
    audit_readiness,
    build_contract,
    build_kit,
    canonical_json,
    import_attestation,
    load_protocol,
    simulate_targets,
    stable_hash,
    synthetic_attestation,
    validate_attestation,
    verify_attestation_package,
    verify_kit,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/formal_backup_target_attestation_v1_protocol.json"
)
CLI = ROOT / "scripts/check_formal_backup_target.py"
RUNTIME = ROOT / "scripts/formal_backup_target_runtime.py"
CHALLENGE = hashlib.sha256(b"backup-target-test-challenge").hexdigest()
ISSUED = 1_700_000_000


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture()
def contract(protocol: dict[str, object]) -> dict[str, object]:
    return build_contract(
        protocol, challenge_id=CHALLENGE, issued_epoch=ISSUED
    )


def _run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            if os.environ.get(name):
                environment[name] = os.environ[name]
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_protocol_binds_existing_capacity_without_reduction(
    protocol: dict[str, object],
) -> None:
    assert protocol["requirements"]["active_shard_window"] == 4
    assert protocol["requirements"]["required_available_bytes"] == 2_119_029_489_664
    assert protocol["requirements"]["required_available_inodes"] == 210_940
    assert protocol["requirements"]["required_quota_bytes"] == 2_119_029_489_664
    assert set(protocol["bindings"]) == {
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "multivolume_storage",
        "portable_execution_site",
        "shard_streaming_retention",
        "storage_governance",
    }


def test_kit_is_deterministic_and_standard_library_only(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=first,
    )
    two = build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=second,
    )
    assert first.read_bytes() == second.read_bytes()
    assert one == two
    assert verify_kit(first, protocol, repository_root=ROOT)["exit_code"] == 0
    with zipfile.ZipFile(first) as archive:
        assert archive.read("probe.py") == archive.read("verify.py") == RUNTIME.read_bytes()
        assert b"scholar_agent" not in archive.read("probe.py")


def test_two_no_repository_environments_verify_contract_identically(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    outputs = []
    for name in ("site-a", "site-b"):
        site = tmp_path / name
        site.mkdir()
        with zipfile.ZipFile(kit) as archive:
            archive.extractall(site)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(site / "verify.py"),
                "verify-contract",
                "--contract",
                str(site / "backup_contract.json"),
            ],
            cwd=site,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(site / "home"),
                "TMPDIR": str(site / "tmp"),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == EXIT_READY
        assert result.stderr == b""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "scenario",
    [
        "capacity_insufficient",
        "inode_insufficient",
        "quota_unknown",
        "same_device_alias",
        "same_host_domain",
        "remote_evidence_insufficient",
        "recovery_failure",
    ],
)
def test_unqualified_profiles_fail_closed(
    contract: dict[str, object], scenario: str
) -> None:
    attestation = synthetic_attestation(ROOT, contract, scenario=scenario)
    assert attestation["status"] == "not_ready_no_qualified_backup_target"
    with pytest.raises(BackupTargetError, match="attestation_not_qualified"):
        validate_attestation(
            ROOT, contract, attestation, require_qualified=True
        )


def test_qualified_import_is_bound_and_challenge_is_one_time(
    tmp_path: Path, protocol: dict[str, object], contract: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    attestation_path = tmp_path / "attestation.json"
    ledger = tmp_path / "ledger.json"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    write_json(attestation_path, synthetic_attestation(ROOT, contract))
    receipt = import_attestation(
        ROOT,
        protocol,
        kit_path=kit,
        attestation_path=attestation_path,
        ledger_path=ledger,
        current_epoch=ISSUED + 2,
        allow_synthetic=True,
    )
    assert receipt["synthetic_only"] is True
    assert receipt["fresh_observation_required_at_launch"] is True
    assert verify_attestation_package(
        ROOT,
        protocol,
        kit_path=kit,
        attestation_path=attestation_path,
    )["exit_code"] == EXIT_READY
    with pytest.raises(BackupTargetError, match="challenge_replay"):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation_path,
            ledger_path=ledger,
            current_epoch=ISSUED + 2,
            allow_synthetic=True,
        )


def test_attestation_tamper_and_target_replacement_are_rejected(
    contract: dict[str, object]
) -> None:
    attestation = synthetic_attestation(ROOT, contract)
    tampered = copy.deepcopy(attestation)
    tampered["observations"]["available_bytes"] += 1
    with pytest.raises(BackupTargetError, match="attestation_digest_invalid"):
        validate_attestation(ROOT, contract, tampered, require_qualified=True)
    replaced = copy.deepcopy(attestation)
    replaced["observations"]["filesystem_identity"] = stable_hash(
        {"filesystem": "replacement"}
    )
    payload = dict(replaced)
    payload.pop("attestation_sha256")
    replaced["attestation_sha256"] = stable_hash(payload)
    with pytest.raises(
        BackupTargetError, match="attestation_semantic_inconsistent"
    ):
        validate_attestation(ROOT, contract, replaced, require_qualified=True)


def test_capacity_drop_and_stale_observation_are_rejected(
    tmp_path: Path, protocol: dict[str, object], contract: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    attestation_path = tmp_path / "attestation.json"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    dropped = synthetic_attestation(ROOT, contract)
    dropped["observations"]["available_bytes"] -= 1
    dropped["checks"]["available_bytes"] = False
    dropped["status"] = "not_ready_no_qualified_backup_target"
    payload = dict(dropped)
    payload.pop("attestation_sha256")
    dropped["attestation_sha256"] = stable_hash(payload)
    write_json(attestation_path, dropped)
    with pytest.raises(BackupTargetError, match="attestation_not_qualified"):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation_path,
            ledger_path=tmp_path / "ledger-a.json",
            current_epoch=ISSUED + 2,
            allow_synthetic=True,
        )
    write_json(attestation_path, synthetic_attestation(ROOT, contract))
    with pytest.raises(BackupTargetError, match="attestation_stale"):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation_path,
            ledger_path=tmp_path / "ledger-b.json",
            current_epoch=ISSUED + 86_401,
            allow_synthetic=True,
        )


def test_synthetic_import_cannot_qualify_real_launch(
    tmp_path: Path, protocol: dict[str, object], contract: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    attestation = tmp_path / "attestation.json"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    write_json(attestation, synthetic_attestation(ROOT, contract))
    with pytest.raises(
        BackupTargetError, match="synthetic_attestation_forbidden"
    ):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation,
            ledger_path=tmp_path / "ledger.json",
            current_epoch=ISSUED + 2,
        )


def test_simulation_and_current_readiness_are_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_targets(ROOT, protocol)
    second = simulate_targets(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["passed_count"] == first["scenario_count"] == 12
    readiness = audit_readiness(ROOT, protocol)
    assert readiness["exit_code"] == EXIT_NOT_READY
    assert readiness["fault_domain_evidence"] == "not_available"
    assert readiness["formal_validation_complete"] is False


def test_cli_outputs_are_deterministic_and_readiness_is_blocked() -> None:
    first = _run_cli("simulate-targets")
    second = _run_cli("simulate-targets")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    audit = _run_cli("audit-readiness")
    assert audit.returncode == EXIT_NOT_READY
    assert audit.stderr == b""
    assert json.loads(audit.stdout)["status"] == (
        "not_ready_no_qualified_backup_target"
    )


def test_kit_tamper_and_protocol_drift_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    with zipfile.ZipFile(kit, "a") as archive:
        archive.writestr("extra.txt", b"extra")
    with pytest.raises(BackupTargetError, match="kit_member_inventory_invalid"):
        verify_kit(kit, protocol, repository_root=ROOT)
    drifted = copy.deepcopy(protocol)
    drifted["requirements"]["required_available_bytes"] -= 1
    payload = copy.deepcopy(drifted)
    payload.pop("protocol_sha256")
    drifted["protocol_sha256"] = stable_hash(payload)
    drifted_path = tmp_path / "drifted.json"
    write_json(drifted_path, drifted)
    with pytest.raises(BackupTargetError, match="requirements_drift"):
        load_protocol(drifted_path, repository_root=ROOT)
