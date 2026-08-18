from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_agent.evaluation import formal_run_disaster_recovery as recovery
from scholar_agent.evaluation.formal_run_disaster_recovery import (
    BackupManifest,
    DisasterRecoveryError,
    _AdapterCounter,
    _execute_shard,
    _initialize_fake_run,
    _load_manifest,
    _safe_relative,
    _validate_parent_chain,
    _write_authority,
    audit_readiness,
    canonical_json,
    create_backup,
    load_protocol,
    restore_backup,
    simulate_disaster,
    verify_backup,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_run_disaster_recovery_v1_protocol.json"
SCRIPT = ROOT / "scripts/check_formal_run_recovery.py"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture(scope="module")
def completed_simulations(
    protocol: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    return simulate_disaster(ROOT, protocol), simulate_disaster(ROOT, protocol)


@pytest.fixture
def partial_backup(
    tmp_path: Path, protocol: dict[str, object]
) -> tuple[Path, Path, dict[str, object]]:
    run_root = tmp_path / "run"
    backup_root = tmp_path / "offsite"
    (
        prepared,
        authorization,
        machine,
        plan,
        statuses,
        generations,
        run_identity,
    ) = _initialize_fake_run(run_root, ROOT, protocol)
    counter = _AdapterCounter()
    attempt = machine.selected_attempts[0]
    generations["0"] = _execute_shard(
        run_root, plan["sharding"]["shards"][0], attempt, counter
    )
    statuses[attempt] = "completed"
    machine.complete_shard(0)
    _write_authority(
        run_root,
        prepared=prepared,
        authorization=authorization,
        machine=machine,
        run_identity=run_identity,
        plan_sha256=plan["plan_sha256"],
        attempt_statuses=statuses,
        generation_by_shard=generations,
        adapter_call_count=counter.total,
        aggregate=None,
    )
    report = create_backup(
        run_root,
        backup_root,
        repository_root=ROOT,
        protocol=protocol,
    )
    return run_root, backup_root, report


def test_protocol_binds_full1000_authority(protocol: dict[str, object]) -> None:
    assert protocol["source_commit"] == "905b4d24aa50705eb6b95aa26a3e538795eb29d8"
    assert protocol["population"] == {
        "query_count": 1000,
        "query_order_sha256": "1d310756a0a5115ea33aec23939a4e9867302c85750448810252f174e5e74563",
        "shard_count": 20,
    }
    assert protocol["activation"]["real_run_started"] is False
    assert protocol["activation"]["real_backup_created"] is False


def test_full_disaster_simulation_is_equivalent_and_byte_deterministic(
    completed_simulations: tuple[dict[str, object], dict[str, object]],
) -> None:
    first, second = completed_simulations
    assert canonical_json(first) == canonical_json(second)
    assert first["query_count"] == 1000
    assert first["partial_backup_query_cursor"] == 500
    assert first["restored_query_cursor"] == 500
    assert first["final_request_count"] == 1000
    assert first["duplicate_request_count"] == 0
    assert all(first["equivalence"].values())
    assert first["replacement_shard"] == 15
    assert first["parent_chain_length"] == 2


def test_fault_matrix_covers_every_required_failure(
    completed_simulations: tuple[dict[str, object], dict[str, object]],
) -> None:
    scenarios = completed_simulations[0]["scenarios"]
    assert {item["scenario"] for item in scenarios} == {
        "audit_chain_truncation",
        "backup_interruption",
        "concurrent_restorer",
        "duplicate_charge_after_resume",
        "hash_tamper",
        "missing_member",
        "mixed_generation",
        "non_empty_target",
        "old_backup_rollback",
        "parent_chain_break_or_cycle",
        "raw_response_missing",
    }
    assert all(item["blocked"] is True for item in scenarios)


def test_incremental_backup_restore_is_repeatable_without_authority_mirrors(
    tmp_path: Path,
    partial_backup: tuple[Path, Path, dict[str, object]],
    protocol: dict[str, object],
) -> None:
    _run_root, backup_root, report = partial_backup
    verified = verify_backup(
        backup_root, repository_root=ROOT, protocol=protocol
    )
    assert verified["query_cursor"] == 50
    assert verified["completed_shard_count"] == 1
    assert verified["request_count"] == 50
    assert report["new_object_count"] > 0
    first = tmp_path / "restored-a"
    second = tmp_path / "restored-b"
    restore_backup(
        backup_root, first, repository_root=ROOT, protocol=protocol
    )
    restore_backup(
        backup_root, second, repository_root=ROOT, protocol=protocol
    )
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert not any("results.jsonl" in name for name in first_files)
    joined = b"\n".join(first_files.values())
    assert str(tmp_path).encode() not in joined
    assert b".env" not in joined


def test_interrupted_backup_preserves_previous_latest(
    partial_backup: tuple[Path, Path, dict[str, object]],
    protocol: dict[str, object],
) -> None:
    run_root, backup_root, first = partial_backup
    with pytest.raises(DisasterRecoveryError, match="interruption"):
        create_backup(
            run_root,
            backup_root,
            repository_root=ROOT,
            protocol=protocol,
            fault="after_manifest",
        )
    assert recovery._latest(backup_root) == first["backup_id"]
    assert verify_backup(
        backup_root, repository_root=ROOT, protocol=protocol
    )["backup_id"] == first["backup_id"]


def test_restore_failure_leaves_no_partial_authority(
    tmp_path: Path,
    partial_backup: tuple[Path, Path, dict[str, object]],
    protocol: dict[str, object],
) -> None:
    _run_root, backup_root, report = partial_backup
    manifest = _load_manifest(backup_root, report["backup_id"])
    recovery._object_path(backup_root, manifest.files[0].sha256).unlink()
    target = tmp_path / "failed-restore"
    with pytest.raises(DisasterRecoveryError):
        restore_backup(
            backup_root, target, repository_root=ROOT, protocol=protocol
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".failed-restore.restore-*.pending"))


def test_parent_cycle_and_rollback_are_rejected(
    partial_backup: tuple[Path, Path, dict[str, object]],
    protocol: dict[str, object],
) -> None:
    _run_root, backup_root, report = partial_backup
    manifest = _load_manifest(backup_root, report["backup_id"])
    cyclic = manifest.model_copy(
        update={"parent_backup_id": manifest.backup_id}
    )
    original = recovery._load_manifest
    try:
        recovery._load_manifest = lambda _root, _identity: cyclic
        with pytest.raises(DisasterRecoveryError, match="cycle"):
            _validate_parent_chain(backup_root, cyclic)
    finally:
        recovery._load_manifest = original
    with pytest.raises(DisasterRecoveryError, match="rollback"):
        verify_backup(
            backup_root,
            repository_root=ROOT,
            protocol=protocol,
            backup_id="0" * 64,
        )


def test_manifest_inventory_and_backup_paths_are_fail_closed(
    partial_backup: tuple[Path, Path, dict[str, object]],
) -> None:
    _run_root, backup_root, report = partial_backup
    payload = _load_manifest(
        backup_root, report["backup_id"]
    ).model_dump(mode="json")
    payload["files"].append(dict(payload["files"][0]))
    payload["manifest_sha256"] = recovery.stable_hash(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValidationError):
        BackupManifest.model_validate(payload)
    for unsafe in (".env", "../escape", "/absolute", "cache/.pending-x"):
        with pytest.raises(DisasterRecoveryError):
            _safe_relative(unsafe)


def test_duplicate_committed_query_is_rejected() -> None:
    counter = _AdapterCounter()
    counter.call("opaque-query")
    with pytest.raises(DisasterRecoveryError, match="repeated"):
        counter.call("opaque-query")


def test_restore_commit_mismatch_is_read_only(
    tmp_path: Path,
    partial_backup: tuple[Path, Path, dict[str, object]],
    protocol: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_root, backup_root, _report = partial_backup
    monkeypatch.setattr(recovery, "_git_head", lambda _root: "0" * 40)
    report = restore_backup(
        backup_root,
        tmp_path / "read-only",
        repository_root=ROOT,
        protocol=protocol,
    )
    assert report["execution_allowed"] is False
    assert report["resume_requires_new_authorization"] is True
    assert report["status"] == "restored_read_only_commit_mismatch"


def test_real_readiness_remains_blocked_and_creates_no_backup(
    protocol: dict[str, object],
) -> None:
    report = audit_readiness(ROOT, protocol)
    assert report["exit_code"] == 3
    assert report["status"] == "external_run_not_started"
    assert report["real_backup_created"] is False
    assert report["full1000_completed"] is False
    assert report["formal_validation_complete"] is False


def test_readiness_and_freshness_register_only_engineering_claim() -> None:
    readiness = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text(
            encoding="utf-8"
        )
    )
    claim = next(
        item
        for item in readiness["claims"]
        if item["claim_id"] == "architecture_formal_run_disaster_recovery_ready"
    )
    assert claim["scope"] == "engineering_capability"
    assert "Full1000 remains incomplete" in claim["boundary"]
    assert {
        item["blocker_id"] for item in readiness["blockers"]
    } == {
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    }
    freshness = json.loads(
        (ROOT / "benchmark/validation_evidence_freshness_v1_spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert freshness["claim_component_bindings"][claim["claim_id"]] == [
        "formal_run_disaster_recovery",
        "full1000_execution",
        "full1000_launch_control",
        "provider_ingest_provenance",
    ]


def test_cli_readiness_and_usage_have_stable_machine_output() -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "audit-readiness",
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    assert first.returncode == second.returncode == 3
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert json.loads(first.stdout)["status"] == "external_run_not_started"
    usage = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert usage.returncode == 4
    assert usage.stderr == b""


def test_cli_missing_backup_is_not_eligible_and_does_not_traceback(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify-backup",
            "--backup-root",
            str(tmp_path / "missing"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 3
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["reason_code"] == "backup_latest_missing"


def test_cli_rejects_rehashed_protocol_policy_drift_without_traceback(
    tmp_path: Path,
) -> None:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    value["activation"]["real_run_started"] = True
    value["protocol_sha256"] = recovery._protocol_digest(value)
    drift = tmp_path / "drift.json"
    drift.write_bytes(canonical_json(value))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--protocol",
            str(drift),
            "audit-readiness",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["reason_code"] == "protocol_policy_drift"


def test_no_test_artifact_escapes_temporary_directories(
    tmp_path: Path,
    partial_backup: tuple[Path, Path, dict[str, object]],
) -> None:
    run_root, backup_root, _report = partial_backup
    assert run_root.is_relative_to(tmp_path)
    assert backup_root.is_relative_to(tmp_path)
    assert not any(path.suffix in {".zip", ".whl"} for path in ROOT.glob("*"))
