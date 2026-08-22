from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_external_blocker_action_pack import (
    CHAIN_READY,
    EXIT_MISSING,
    EXIT_READY,
    EXIT_VIOLATION,
    ExternalBlockerActionError,
    audit_chains,
    audit_readiness,
    build_pack,
    canonical_json,
    load_protocol,
    read_object,
    stable_hash,
    verify_pack,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_external_blocker_action_pack_v1_protocol.json"
PACK_PATH = ROOT / "benchmark/formal_external_blocker_action_pack_v1_pack.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _runner(_root: Path, gate: dict[str, object]) -> dict[str, object]:
    return {"exit_code": gate["expected_exit_code"],
            "status": gate["expected_status"]}


def _audit(protocol: dict[str, object]) -> dict[str, object]:
    return audit_chains(ROOT, protocol, runner=_runner, verify_freshness=False)


def test_three_existing_chains_close_without_engineering_gap(
    protocol: dict[str, object],
) -> None:
    report = _audit(protocol)
    assert report["exit_code"] == EXIT_READY
    assert report["engineering_gap_count"] == 0
    assert report["development_freeze_state"] == "external_action_required"
    assert [row["chain_id"] for row in report["chains"]] == [
        "backup_members", "human_annotation", "official_scorer"
    ]
    assert {row["state"] for row in report["chains"]} == {CHAIN_READY}


def test_gate_drift_and_missing_script_fail_closed(protocol: dict[str, object]) -> None:
    def wrong_status(_root: Path, gate: dict[str, object]) -> dict[str, object]:
        return {"exit_code": gate["expected_exit_code"], "status": "unexpected"}

    report = audit_chains(ROOT, protocol, runner=wrong_status, verify_freshness=False)
    assert report["exit_code"] == EXIT_VIOLATION
    assert report["engineering_gap_count"] == 8
    assert {row["reason_code"] for row in report["violations"]} == {
        "blocked_gate_contract_drift"
    }


def test_pack_is_deterministic_and_preserves_three_blockers(
    protocol: dict[str, object],
) -> None:
    first = build_pack(protocol, _audit(protocol))
    second = build_pack(protocol, _audit(protocol))
    assert canonical_json(first) == canonical_json(second)
    assert first == read_object(PACK_PATH)
    verified = verify_pack(first, protocol)
    assert verified["engineering_gap_count"] == 0
    assert verified["formal_blockers"] == [
        "full1000_incomplete", "human_precision_missing",
        "official_scorer_schema_missing",
    ]
    readiness = audit_readiness(first, protocol)
    assert readiness["exit_code"] == EXIT_MISSING
    assert readiness["status"] == "external_inputs_still_missing"
    assert readiness["formal_validation_complete"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "formal_validation_complete"),
        ("formal_validation_complete", True),
        ("formal_blockers", ["full1000_incomplete"]),
    ],
)
def test_resealed_pack_semantic_tampering_is_rejected(
    protocol: dict[str, object], field: str, value: object,
) -> None:
    pack = copy.deepcopy(read_object(PACK_PATH))
    pack[field] = value
    payload = dict(pack)
    payload.pop("pack_sha256")
    pack["pack_sha256"] = stable_hash(payload)
    with pytest.raises(ExternalBlockerActionError):
        verify_pack(pack, protocol)


def test_resealed_action_or_freeze_bypass_is_rejected(
    protocol: dict[str, object],
) -> None:
    for mutate in ("action", "freeze"):
        pack = copy.deepcopy(read_object(PACK_PATH))
        if mutate == "action":
            pack["chains"][0]["action_plan"][0]["command"] = "python scripts/other.py"
        else:
            pack["development_freeze"]["new_governance_task_allowed"] = True
        payload = dict(pack)
        payload.pop("pack_sha256")
        pack["pack_sha256"] = stable_hash(payload)
        with pytest.raises(ExternalBlockerActionError):
            verify_pack(pack, protocol)


def test_protocol_rejects_sensitive_or_absolute_content(tmp_path: Path) -> None:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    value["chains"][0]["missing_inputs"].append("/Users/operator/private-input")
    payload = dict(value)
    payload.pop("protocol_sha256")
    value["protocol_sha256"] = stable_hash(payload)
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json(value))
    with pytest.raises(ExternalBlockerActionError, match="absolute_path"):
        load_protocol(path, repository_root=ROOT)


def test_cli_usage_and_current_readiness_are_machine_stable() -> None:
    usage = subprocess.run(
        [sys.executable, "scripts/check_external_blocker_actions.py", "build-pack"],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert usage.returncode == 4
    assert usage.stderr == b""
    assert json.loads(usage.stdout)["status"] == "usage_error"

    first = subprocess.run(
        [sys.executable, "scripts/check_external_blocker_actions.py", "audit-readiness"],
        cwd=ROOT, capture_output=True, check=False,
    )
    second = subprocess.run(
        [sys.executable, "scripts/check_external_blocker_actions.py", "audit-readiness"],
        cwd=ROOT, capture_output=True, check=False,
    )
    if first.returncode == 2:
        report = json.loads(first.stdout)
        if any(
            row.get("reason_code") == "protocol_or_evidence_not_fresh"
            for row in report.get("violations", [])
            if isinstance(row, dict)
        ):
            pytest.skip("frozen validation freshness is not available in this checkout")
    assert first.returncode == second.returncode == EXIT_MISSING
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["development_freeze_state"] == "external_action_required"
    assert report["engineering_gap_count"] == 0
