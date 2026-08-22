from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_set_member_intake import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    BackupSetIntakeError,
    BackupSetIntakeNotReady,
    activate_set,
    audit_readiness,
    build_slot_contract,
    build_slot_kit,
    import_member,
    load_protocol,
    simulate_matrix,
    synthetic_member_attestation,
    validate_member,
    verify_activation,
    verify_registry,
    verify_slot_kit,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/formal_backup_set_member_intake_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_set_intake.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _contract(
    protocol: dict[str, object], count: int, slot: int
) -> dict[str, object]:
    return build_slot_contract(
        protocol,
        member_count=count,
        slot=slot,
        challenge_id=stable_hash({"count": count, "slot": slot}),
        issued_epoch=10_000,
    )


def _prepared(
    protocol: dict[str, object], count: int = 4
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    contracts = [_contract(protocol, count, slot) for slot in range(count)]
    attestations = [
        synthetic_member_attestation(
            contract,
            identity_seed=f"member-{slot}",
            observation_epoch=10_000,
        )
        for slot, contract in enumerate(contracts)
    ]
    events: list[dict[str, object]] = []
    for contract, attestation in zip(contracts, attestations, strict=True):
        events = import_member(
            ROOT,
            events,
            contract,
            attestation,
            observation_epoch=10_000,
            require_real=False,
        )
    return contracts, attestations, events


def _rehash(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result["attestation_sha256"] = stable_hash(
        {key: child for key, child in result.items() if key != "attestation_sha256"}
    )
    return result


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
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
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize(
    ("count", "thresholds"),
    [
        (2, [(533_012_676_608, 56_321), (530_865_192_960, 56_221)]),
        (
            3,
            [
                (426_130_625_878, 44_623),
                (353_921_488_213, 37_834),
                (318_890_661_205, 34_490),
            ],
        ),
        (
            4,
            [
                (285_112_532_992, 30_413),
                (282_965_049_344, 30_313),
                (282_965_049_344, 30_313),
                (282_965_049_344, 30_313),
            ],
        ),
    ],
)
def test_slot_contract_reuses_topology_thresholds(
    protocol: dict[str, object],
    count: int,
    thresholds: list[tuple[int, int]],
) -> None:
    contracts = [_contract(protocol, count, slot) for slot in range(count)]
    assert [
        (
            row["slot_requirements"]["minimum_available_bytes"],
            row["slot_requirements"]["minimum_available_inodes"],
        )
        for row in contracts
    ] == thresholds
    assert sorted(
        shard for row in contracts for shard in row["allowed_shards"]
    ) == list(range(20))


def test_kit_is_deterministic_and_bound(protocol: dict[str, object], tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    args = {
        "member_count": 4,
        "slot": 2,
        "challenge_id": stable_hash({"kit": 2}),
        "issued_epoch": 10_000,
    }
    build_slot_kit(ROOT, protocol, output=first, **args)
    build_slot_kit(ROOT, protocol, output=second, **args)
    assert first.read_bytes() == second.read_bytes()
    verified = verify_slot_kit(first, protocol, repository_root=ROOT)
    assert verified["slot"] == 2
    assert verified["member_count"] == 4


@pytest.mark.parametrize("environment_index", [0, 1])
def test_kit_verifier_runs_without_repository_or_site_packages(
    protocol: dict[str, object], tmp_path: Path, environment_index: int
) -> None:
    kit = tmp_path / f"kit-{environment_index}.zip"
    build_slot_kit(
        ROOT,
        protocol,
        member_count=2,
        slot=environment_index,
        challenge_id=stable_hash({"environment": environment_index}),
        issued_epoch=10_000,
        output=kit,
    )
    isolated = tmp_path / f"isolated-{environment_index}"
    isolated.mkdir()
    with zipfile.ZipFile(kit) as archive:
        archive.extract("verify.py", isolated)
        archive.extract("slot_contract.json", isolated)
    home = tmp_path / f"home-{environment_index}"
    temporary = tmp_path / f"tmp-{environment_index}"
    home.mkdir()
    temporary.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(isolated / "verify.py"),
            "verify-contract",
            "--contract",
            str(isolated / "slot_contract.json"),
        ],
        cwd=home,
        env={
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "PATH": os.environ.get("PATH", ""),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["status"] == "slot_kit_verified"


def test_incremental_import_and_activation(protocol: dict[str, object]) -> None:
    contracts, attestations, events = _prepared(protocol)
    assert [event["state_after"] for event in events] == [
        "qualified",
        "reserved",
    ] * 4
    activated, receipt = activate_set(
        ROOT,
        protocol,
        events,
        contracts,
        attestations,
        member_count=4,
        observation_epoch=10_000,
        require_real=False,
    )
    assert len(activated) == 12
    assert receipt["synthetic_only"] is True
    verify_activation(
        ROOT,
        protocol,
        receipt,
        contracts,
        attestations,
        observation_epoch=10_000,
        require_real=False,
    )


def test_partial_members_cannot_activate(protocol: dict[str, object]) -> None:
    contracts, attestations, events = _prepared(protocol)
    with pytest.raises(BackupSetIntakeNotReady, match="required_slots_missing"):
        activate_set(
            ROOT,
            protocol,
            events[:-2],
            contracts[:-1],
            attestations[:-1],
            member_count=4,
            observation_epoch=10_000,
            require_real=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "device_identity",
        "filesystem_identity",
        "quota_pool_identity",
        "failure_domain_identity",
        "management_domain_identity",
    ],
)
def test_member_identity_and_capacity_cannot_be_double_counted(
    protocol: dict[str, object], field: str
) -> None:
    contracts, attestations, events = _prepared(protocol)
    attestations[1]["observations"][field] = attestations[0]["observations"][field]
    identity_fields = (
        "device_identity",
        "failure_domain_identity",
        "filesystem_identity",
        "management_domain_identity",
        "quota_pool_identity",
        "storage_service_identity",
    )
    attestations[1]["target_identity"] = stable_hash(
        {
            key: attestations[1]["observations"][key]
            for key in identity_fields
        }
    )
    attestations[1] = _rehash(attestations[1])
    with pytest.raises(BackupSetIntakeError, match=f"duplicate_member_{field}"):
        activate_set(
            ROOT,
            protocol,
            events,
            contracts,
            attestations,
            member_count=4,
            observation_epoch=10_000,
            require_real=False,
        )


def test_slot_mismatch_and_challenge_replay_fail(protocol: dict[str, object]) -> None:
    contracts, attestations, events = _prepared(protocol)
    with pytest.raises(BackupSetIntakeError, match="attestation_binding_invalid"):
        validate_member(
            ROOT,
            contracts[1],
            attestations[0],
            observation_epoch=10_000,
            require_real=False,
        )
    with pytest.raises(BackupSetIntakeError, match="slot_already_consumed"):
        import_member(
            ROOT,
            events,
            contracts[0],
            attestations[0],
            observation_epoch=10_000,
            require_real=False,
        )


def test_expired_revoked_and_capacity_drift_fail(
    protocol: dict[str, object],
) -> None:
    contract = _contract(protocol, 2, 0)
    attestation = synthetic_member_attestation(
        contract, identity_seed="member", observation_epoch=10_000
    )
    with pytest.raises(BackupSetIntakeError, match="observations_invalid"):
        validate_member(
            ROOT,
            contract,
            attestation,
            observation_epoch=100_000,
            require_real=False,
        )
    revoked = copy.deepcopy(attestation)
    revoked["revoked"] = True
    revoked = _rehash(revoked)
    with pytest.raises(BackupSetIntakeError, match="attestation_binding_invalid"):
        validate_member(
            ROOT,
            contract,
            revoked,
            observation_epoch=10_000,
            require_real=False,
        )
    reduced = copy.deepcopy(attestation)
    reduced["observations"]["available_bytes"] = 0
    reduced["checks"]["available_bytes"] = False
    reduced = _rehash(reduced)
    with pytest.raises(BackupSetIntakeError, match="attestation_not_qualified"):
        validate_member(
            ROOT,
            contract,
            reduced,
            observation_epoch=10_000,
            require_real=False,
        )


def test_synthetic_member_never_activates_real_set(
    protocol: dict[str, object],
) -> None:
    contracts, attestations, events = _prepared(protocol)
    with pytest.raises(BackupSetIntakeError, match="attestation_binding_invalid"):
        activate_set(
            ROOT,
            protocol,
            events,
            contracts,
            attestations,
            member_count=4,
            observation_epoch=10_000,
            require_real=True,
        )


def test_registry_tamper_and_invalid_transition_fail(
    protocol: dict[str, object],
) -> None:
    _contracts, _attestations, events = _prepared(protocol, 2)
    tampered = copy.deepcopy(events)
    tampered[1]["previous_event_sha256"] = "0" * 64
    with pytest.raises(BackupSetIntakeError, match="registry_chain_invalid"):
        verify_registry(tampered)
    reordered = list(reversed(events))
    with pytest.raises(BackupSetIntakeError, match="registry_chain_invalid"):
        verify_registry(reordered)


def test_simulation_matrix_and_readiness_are_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_matrix(ROOT, protocol)
    second = simulate_matrix(ROOT, protocol)
    assert first == second
    assert first["scenario_count"] == first["passed_count"] == 12
    assert first["recovery"] == {
        "duplicate_request_count": 0,
        "ledger_conserved": True,
        "query_count": 1000,
        "shard_count": 20,
    }
    readiness = audit_readiness(protocol)
    assert readiness["exit_code"] == EXIT_NOT_READY
    assert [row["missing_real_slot_count"] for row in readiness["plans"]] == [
        2,
        3,
        4,
    ]


def test_cli_exit_codes_and_machine_json() -> None:
    simulation = _run_cli("simulate-matrix")
    assert simulation.returncode == EXIT_READY
    assert simulation.stderr == b""
    assert json.loads(simulation.stdout)["passed_count"] == 12
    readiness = _run_cli("audit-readiness")
    assert readiness.returncode == EXIT_NOT_READY
    assert readiness.stderr == b""
    assert json.loads(readiness.stdout)["qualified_real_member_count"] == 0
    usage = _run_cli("build-slot-kit")
    assert usage.returncode == EXIT_USAGE
    assert json.loads(usage.stdout)["status"] == "usage_error"
    missing = _run_cli(
        "verify-member",
        "--kit",
        "missing.zip",
        "--attestation",
        "missing.json",
        "--observation-epoch",
        "10000",
    )
    assert missing.returncode == EXIT_VIOLATION
    assert missing.stderr == b""
