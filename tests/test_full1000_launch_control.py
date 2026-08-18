from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_agent.evaluation import full1000_launch_control as launch_control
from scholar_agent.evaluation.full1000_launch_control import (
    LaunchControlError,
    LaunchOperationMachine,
    OperationAuditLog,
    _contract_digest,
    _fixture_protocol,
    build_authorization,
    build_preparation,
    load_protocol,
    simulate_operations,
    validate_authorization,
    validate_authorization_context,
    validate_launch_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/full1000_launch_control_v1_protocol.json"
SCRIPT = ROOT / "scripts/check_full1000_launch_control.py"


@pytest.fixture
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH)


def _prepared(
    tmp_path: Path, protocol: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    fixture = _fixture_protocol(ROOT, protocol)
    prepared = build_preparation(
        ROOT,
        fixture,
        authoritative_root=tmp_path / "run",
        check_freshness=False,
    )
    authorization = build_authorization(prepared, fixture)
    return prepared, authorization


def test_frozen_protocol_binds_plan_population_and_controls(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == "3222b1282e87bf450dc359e17cdae8f6ac361b67"
    assert protocol["population"] == {
        "query_count": 1000,
        "query_order_sha256": "1d310756a0a5115ea33aec23939a4e9867302c85750448810252f174e5e74563",
        "query_stable_identity_sha256": "31e2daab09e10db78a91d0bf07fd292a9dc2c56ca077bf6c7d92974e81ea7cfb",
        "shard_assignment_sha256": "8b06a6498c18d44a20d9355c0923baaf9be522a2e37213a31a21b3ca86fb2ef6",
        "shard_count": 20,
    }
    required = protocol["execution_contract"]["required_observability"]
    assert all(required.values())
    assert protocol["activation"]["real_launch_allowed_by_this_protocol"] is False


def test_simulation_is_complete_and_byte_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_operations(ROOT, protocol)
    second = simulate_operations(ROOT, protocol)
    assert first == second
    assert first["status"] == "launch_controls_ready"
    assert first["query_count"] == 1000
    assert first["shard_count"] == 20
    assert first["scenario_count"] == 10
    assert all(item["blocked"] for item in first["scenarios"])


def test_authorization_tamper_and_cross_commit_binding_fail(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    fixture = _fixture_protocol(ROOT, protocol)
    tampered = copy.deepcopy(authorization)
    tampered["observed_head"] = "0" * 40
    with pytest.raises(LaunchControlError, match="authorization"):
        validate_authorization(prepared, tampered, fixture)


def test_authorization_pair_cannot_be_reused_after_head_changes(
    tmp_path: Path,
    protocol: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    fixture = _fixture_protocol(ROOT, protocol)
    monkeypatch.setattr(launch_control, "_git_head", lambda _root: "0" * 40)
    with pytest.raises(LaunchControlError, match="commit_drift"):
        validate_authorization_context(ROOT, prepared, authorization, fixture)


def test_old_artifact_and_nonempty_output_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "results.jsonl").write_text("{}\n", encoding="utf-8")
    fixture = _fixture_protocol(ROOT, protocol)
    with pytest.raises(LaunchControlError, match="not_empty"):
        build_preparation(
            ROOT,
            fixture,
            authoritative_root=run_root,
            check_freshness=False,
        )


def test_config_and_plan_drift_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    drift = _fixture_protocol(ROOT, protocol)
    drift["execution_contract"]["configuration_sha256"] = "0" * 64
    drift["protocol_sha256"] = _contract_digest(drift)
    with pytest.raises(LaunchControlError, match="configuration"):
        build_preparation(
            ROOT,
            drift,
            authoritative_root=tmp_path / "run",
            check_freshness=False,
        )


def test_duplicate_start_revoke_and_resume_fail_closed(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    machine = LaunchOperationMachine(prepared, authorization)
    machine.authorize()
    machine.start()
    with pytest.raises(LaunchControlError, match="start"):
        machine.start()
    with pytest.raises(LaunchControlError, match="resume"):
        machine.resume()
    machine.revoke()
    with pytest.raises(LaunchControlError):
        machine.complete_shard(0)


def test_attempt_supersession_and_aggregate_rules(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    machine = LaunchOperationMachine(prepared, authorization)
    machine.authorize()
    machine.start()
    machine.fail_shard(2)
    with pytest.raises(LaunchControlError, match="failed_attempt"):
        machine.complete_shard(2)
    machine.supersede(2)
    with pytest.raises(LaunchControlError, match="stale_attempt"):
        machine.complete_shard(2, "shard-02-attempt-0")
    machine.complete_shard(2)
    with pytest.raises(LaunchControlError, match="aggregate"):
        machine.aggregate()
    for shard in range(20):
        if shard != 2:
            machine.complete_shard(shard)
    machine.aggregate()
    validate_launch_evidence(
        prepared,
        authorization,
        machine.audit_log().model_dump(mode="json"),
        _fixture_protocol(ROOT, protocol),
    )


def test_cancel_blocks_followup_consumption(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    machine = LaunchOperationMachine(prepared, authorization)
    machine.authorize()
    machine.start()
    machine.cancel()
    with pytest.raises(LaunchControlError, match="not_authorized"):
        machine.complete_shard(0)
    with pytest.raises(LaunchControlError, match="resume"):
        machine.resume()


def test_audit_chain_break_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    machine = LaunchOperationMachine(prepared, authorization)
    machine.authorize()
    machine.start()
    payload = machine.audit_log().model_dump(mode="json")
    payload["entries"][1]["previous_entry_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        OperationAuditLog.model_validate(payload)


def test_direct_runner_without_seal_is_not_authoritative(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    prepared, authorization = _prepared(tmp_path, protocol)
    with pytest.raises(LaunchControlError, match="audit"):
        validate_launch_evidence(
            prepared,
            authorization,
            {"schema_version": "1", "protocol": "full1000_launch_control_v1"},
            _fixture_protocol(ROOT, protocol),
        )


def test_cli_simulation_double_run_is_identical() -> None:
    command = [sys.executable, str(SCRIPT), "simulate-operations"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout


def test_cli_malformed_state_has_no_traceback(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "prepared.json").write_text("{", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "authorize-dry-run",
            "--state-dir",
            str(state),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout


def test_protocol_self_hash_rejects_policy_edit(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    changed = copy.deepcopy(protocol)
    changed["authorization"]["allowed_shards"] = [0]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(LaunchControlError, match="digest"):
        load_protocol(path)
