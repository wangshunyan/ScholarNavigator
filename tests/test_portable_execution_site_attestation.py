from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.portable_execution_site_attestation import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_VIOLATION,
    PortableSiteError,
    audit_readiness,
    build_kit,
    build_site_contract,
    build_topology_contract,
    canonical_json,
    import_attestation,
    load_protocol,
    read_kit,
    simulate_sites,
    stable_hash,
    synthetic_attestation,
    validate_attestation,
    verify_import_receipt,
    verify_kit,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/portable_execution_site_attestation_v1_protocol.json"
)
CLI = ROOT / "scripts/check_portable_execution_site.py"
RUNTIME = ROOT / "scripts/portable_execution_site_runtime.py"
CHALLENGE = hashlib.sha256(b"portable-site-test-challenge").hexdigest()
ISSUED = 1_700_000_000


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture()
def contract(protocol: dict[str, object]) -> dict[str, object]:
    return build_site_contract(
        protocol, challenge_id=CHALLENGE, issued_epoch=ISSUED
    )


def _run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_contract_reuses_exact_multivolume_requirements(
    protocol: dict[str, object],
) -> None:
    topology = build_topology_contract(protocol)
    assert len(topology["shard_assignments"]) == 20
    assert [row["shard_index"] for row in topology["shard_assignments"]] == list(
        range(20)
    )
    assert topology["requirements"]["primary-00"]["required_bytes"] == (
        363_193_171_968
    )
    assert topology["requirements"]["primary-01"]["required_bytes"] == (
        361_045_688_320
    )
    assert topology["requirements"]["backup-00"]["required_bytes"] == (
        1_064_883_453_952
    )
    assert topology["requirements"]["backup-01"]["required_bytes"] == (
        1_064_883_453_952
    )
    assert protocol["bindings"]["multivolume_storage"]["path"] == (
        "benchmark/formal_multivolume_storage_v1_protocol.json"
    )


def test_kit_is_byte_deterministic_and_has_no_project_dependency(
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
    report = verify_kit(first, protocol, repository_root=ROOT)
    assert report["exit_code"] == EXIT_READY
    manifest, files = read_kit(first)
    assert manifest["challenge_id"] == CHALLENGE
    runtime = files["probe.py"].decode("utf-8")
    assert "scholar_agent" not in runtime
    assert ".env" not in runtime
    assert files["probe.py"] == files["verify.py"] == RUNTIME.read_bytes()


def test_two_no_repository_environments_verify_same_attestation(
    tmp_path: Path,
    protocol: dict[str, object],
    contract: dict[str, object],
) -> None:
    kit = tmp_path / "site.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    attestation = synthetic_attestation(ROOT, contract)
    outputs = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        with zipfile.ZipFile(kit) as archive:
            archive.extractall(root)
        write_json(root / "attestation.json", attestation)
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root / "home"),
            "TMPDIR": str(root / "tmp"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(root / "verify.py"),
                "verify",
                "--contract",
                str(root / "site_contract.json"),
                "--attestation",
                str(root / "attestation.json"),
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == EXIT_READY
        assert result.stderr == b""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("qualified", "execution_site_qualified"),
        ("primary_insufficient", "not_ready_no_qualified_external_site"),
        ("backup_insufficient", "not_ready_no_qualified_external_site"),
        ("same_fault_domain", "not_ready_no_qualified_external_site"),
        ("quota_unknown", "not_ready_no_qualified_external_site"),
    ],
)
def test_site_profiles_fail_closed(
    contract: dict[str, object], scenario: str, expected: str
) -> None:
    value = synthetic_attestation(ROOT, contract, scenario=scenario)
    assert value["status"] == expected
    validate_attestation(
        ROOT, contract, value, require_qualified=scenario == "qualified"
    )


def test_directory_alias_and_mount_replacement_do_not_qualify(
    contract: dict[str, object],
) -> None:
    value = synthetic_attestation(ROOT, contract)
    aliased = copy.deepcopy(value)
    aliased["volumes"][1]["filesystem_identity"] = aliased["volumes"][0][
        "filesystem_identity"
    ]
    payload = dict(aliased)
    payload.pop("attestation_sha256")
    aliased["attestation_sha256"] = stable_hash(payload)
    with pytest.raises(
        PortableSiteError, match="attestation_status_inconsistent"
    ):
        validate_attestation(ROOT, contract, aliased, require_qualified=True)
    replaced = copy.deepcopy(value)
    replaced["volumes"][0]["mount_identity"] = stable_hash(
        {"replacement": True}
    )
    payload = dict(replaced)
    payload.pop("attestation_sha256")
    replaced["attestation_sha256"] = stable_hash(payload)
    assert replaced["attestation_sha256"] != value["attestation_sha256"]


def test_import_receipt_is_one_time_fresh_and_launch_bound(
    tmp_path: Path,
    protocol: dict[str, object],
    contract: dict[str, object],
) -> None:
    kit = tmp_path / "site.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    attestation = synthetic_attestation(ROOT, contract)
    attestation_path = tmp_path / "attestation.json"
    ledger = tmp_path / "ledger.json"
    write_json(attestation_path, attestation)
    receipt = import_attestation(
        ROOT,
        protocol,
        kit_path=kit,
        attestation_path=attestation_path,
        ledger_path=ledger,
        current_epoch=ISSUED + 10,
        allow_synthetic=True,
    )
    assert receipt["launch_control_sha256"] == protocol["bindings"][
        "launch_control"
    ]["sha256"]
    assert receipt["multivolume_topology_sha256"] == contract["topology"][
        "topology_sha256"
    ]
    verify_import_receipt(receipt, attestation, protocol)
    moved = copy.deepcopy(attestation)
    moved["volumes"][0]["mount_identity"] = stable_hash({"moved": True})
    payload = dict(moved)
    payload.pop("attestation_sha256")
    moved["attestation_sha256"] = stable_hash(payload)
    with pytest.raises(PortableSiteError, match="import_receipt_binding_invalid"):
        verify_import_receipt(receipt, moved, protocol)
    with pytest.raises(PortableSiteError, match="challenge_replay"):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation_path,
            ledger_path=ledger,
            current_epoch=ISSUED + 11,
            allow_synthetic=True,
        )
    with pytest.raises(PortableSiteError, match="attestation_stale"):
        import_attestation(
            ROOT,
            protocol,
            kit_path=kit,
            attestation_path=attestation_path,
            ledger_path=tmp_path / "fresh-ledger.json",
            current_epoch=ISSUED + 86_401,
            allow_synthetic=True,
        )


def test_cross_commit_plan_topology_and_attestation_tamper_fail(
    contract: dict[str, object],
) -> None:
    original = synthetic_attestation(ROOT, contract)
    for key, replacement in (
        ("source_commit", "0" * 40),
        ("plan_sha256", "1" * 64),
        ("topology_sha256", "2" * 64),
    ):
        changed = copy.deepcopy(original)
        changed[key] = replacement
        payload = dict(changed)
        payload.pop("attestation_sha256")
        changed["attestation_sha256"] = stable_hash(payload)
        with pytest.raises(PortableSiteError, match="attestation_binding_invalid"):
            validate_attestation(ROOT, contract, changed, require_qualified=True)
    changed = copy.deepcopy(original)
    changed["volumes"][0]["available_bytes"] = 0
    with pytest.raises(PortableSiteError, match="attestation_digest_invalid"):
        validate_attestation(ROOT, contract, changed, require_qualified=True)


def test_resealed_incomplete_capabilities_and_stale_observation_fail_closed(
    contract: dict[str, object],
) -> None:
    original = synthetic_attestation(ROOT, contract)
    missing = copy.deepcopy(original)
    missing["volumes"][0]["capabilities"].pop("directory_fsync")
    payload = dict(missing)
    payload.pop("attestation_sha256")
    missing["attestation_sha256"] = stable_hash(payload)
    with pytest.raises(
        PortableSiteError, match="attestation_capabilities_invalid"
    ):
        validate_attestation(ROOT, contract, missing, require_qualified=True)
    stale = copy.deepcopy(original)
    stale["observation_epoch"] = ISSUED - 1
    payload = dict(stale)
    payload.pop("attestation_sha256")
    stale["attestation_sha256"] = stable_hash(payload)
    with pytest.raises(PortableSiteError, match="attestation_binding_invalid"):
        validate_attestation(ROOT, contract, stale, require_qualified=True)


def test_kit_tamper_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit = tmp_path / "site.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    root = tmp_path / "unpacked"
    with zipfile.ZipFile(kit) as archive:
        archive.extractall(root)
    (root / "probe.py").write_bytes(b"tampered")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(root.iterdir()):
            archive.write(path, path.name)
    with pytest.raises(PortableSiteError):
        verify_kit(tampered, protocol, repository_root=ROOT)


def test_runtime_probe_does_not_serialize_paths_or_environment(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit = tmp_path / "site.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    root = tmp_path / "isolated"
    with zipfile.ZipFile(kit) as archive:
        archive.extractall(root)
    evidence = {
        "evidence_type": "portable_site_operator_observation_v1",
        "challenge_id": CHALLENGE,
        "volumes": {},
    }
    args = []
    for slot in ("primary-00", "primary-01", "backup-00", "backup-01"):
        path = root / f"secret-volume-{slot}"
        path.mkdir()
        args.extend(["--volume", f"{slot}={path}"])
        evidence["volumes"][slot] = {
            "filesystem_quota_bytes": 10**15,
            "failure_domain_identity": stable_hash({"domain": slot}),
        }
    write_json(root / "site-evidence.json", evidence)
    output = root / "attestation.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(root / "probe.py"),
            "probe",
            "--contract",
            str(root / "site_contract.json"),
            *args,
            "--site-evidence",
            str(root / "site-evidence.json"),
            "--observation-epoch",
            str(ISSUED + 1),
            "--output",
            str(output),
        ],
        cwd=root,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root / "secret-home"),
            "SENSITIVE_SENTINEL": "DO_NOT_SERIALIZE_THIS",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == EXIT_NOT_READY
    assert result.stderr == b""
    serialized = output.read_text(encoding="utf-8")
    assert str(root) not in serialized
    assert "secret-volume" not in serialized
    assert "DO_NOT_SERIALIZE_THIS" not in serialized
    assert "secret-home" not in serialized


def test_simulation_and_readiness_are_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_sites(ROOT, protocol)
    second = simulate_sites(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["passed_count"] == first["scenario_count"] == 10
    readiness = audit_readiness(ROOT, protocol)
    assert readiness["exit_code"] == EXIT_NOT_READY
    assert readiness["current_site_qualified"] is False
    assert readiness["formal_blockers"] == [
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    ]


def test_cli_statuses_and_output_are_stable(tmp_path: Path) -> None:
    first = _run_cli("simulate-site")
    second = _run_cli("simulate-site")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    readiness = _run_cli("audit-readiness")
    assert readiness.returncode == EXIT_NOT_READY
    assert readiness.stderr == b""
    invalid = _run_cli("verify-kit", "--kit", str(tmp_path / "missing.zip"))
    assert invalid.returncode == EXIT_VIOLATION
    assert invalid.stderr == b""
    assert b"Traceback" not in invalid.stdout


def test_cli_rejects_real_import_of_synthetic_attestation(
    tmp_path: Path,
    protocol: dict[str, object],
    contract: dict[str, object],
) -> None:
    kit = tmp_path / "site.zip"
    build_kit(
        ROOT,
        protocol,
        challenge_id=CHALLENGE,
        issued_epoch=ISSUED,
        output=kit,
    )
    attestation = tmp_path / "attestation.json"
    write_json(attestation, synthetic_attestation(ROOT, contract))
    result = _run_cli(
        "import-attestation",
        "--kit",
        str(kit),
        "--attestation",
        str(attestation),
        "--ledger",
        str(tmp_path / "ledger.json"),
        "--current-epoch",
        str(ISSUED + 1),
    )
    assert result.returncode == EXIT_VIOLATION
    assert json.loads(result.stdout)["reason_code"] == (
        "synthetic_attestation_forbidden"
    )


def test_missing_or_drifted_protocol_binding_fails_closed(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    changed = copy.deepcopy(protocol)
    changed["bindings"]["execution_plan"]["sha256"] = "0" * 64
    payload = dict(changed)
    payload.pop("protocol_sha256")
    changed["protocol_sha256"] = stable_hash(payload)
    path = tmp_path / "protocol.json"
    write_json(path, changed)
    with pytest.raises(PortableSiteError, match="binding_hash_mismatch"):
        load_protocol(path, repository_root=ROOT)
