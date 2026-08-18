from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_validation_dress_rehearsal import (
    EXIT_BLOCKED,
    FAILURE_SCENARIOS,
    REAL_BLOCKERS,
    STAGE_ORDER,
    DressRehearsalError,
    RehearsalMachine,
    audit_readiness,
    build_handoff_checklist,
    canonical_json,
    load_protocol,
    read_json,
    simulate_failures,
    stable_hash,
    verify_rehearsal_report,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_validation_dress_rehearsal_v1_protocol.json"
REPORT_PATH = (
    ROOT / "benchmark/formal_validation_dress_rehearsal_v1_evidence/rehearsal.json"
)
CLI = ROOT / "scripts/check_formal_validation_dress_rehearsal.py"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return read_json(REPORT_PATH)


def _rehash_report(value: dict[str, object]) -> None:
    value["report_sha256"] = "0" * 64
    value["report_sha256"] = stable_hash(value)


def test_tracked_rehearsal_closes_all_stages_without_real_state(
    protocol: dict[str, object], report: dict[str, object]
) -> None:
    verified = verify_rehearsal_report(report, protocol)
    assert verified["status"] == "rehearsal_completed"
    assert report["query_count"] == 1000
    assert report["shard_count"] == 20
    assert report["human_item_count"] == 471
    assert report["scorer_query_count"] == 1000
    assert [row["name"] for row in report["stages"]] == list(STAGE_ORDER)
    assert report["cleanup"] == {
        "labels_persisted": False,
        "receipt_persisted": False,
        "run_artifacts_persisted": False,
        "temporary_namespace_cleaned": True,
    }
    assert report["formal_validation_complete"] is False
    assert report["real_state_mutation_count"] == 0


def test_stage_machine_rejects_missing_reordered_and_duplicate_receipt(
    protocol: dict[str, object],
) -> None:
    machine = RehearsalMachine(protocol)
    with pytest.raises(DressRehearsalError, match="stage_order"):
        machine.advance("launch_authorized", {})
    machine.advance("preregistration_sealed", {})
    with pytest.raises(DressRehearsalError, match="stage_order"):
        machine.finish()
    receipt = {
        "receipt_sha256": "a" * 64,
        "synthetic_test_only": True,
        "formal_validation_complete": False,
    }
    machine.register_receipt(receipt)
    with pytest.raises(DressRehearsalError, match="duplicate_receipt"):
        machine.register_receipt(receipt)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["stages"].pop(),
        lambda value: value["stages"].reverse(),
        lambda value: value.__setitem__("synthetic_rehearsal_only", False),
        lambda value: value.__setitem__("real_state_mutation_count", 1),
        lambda value: value["test_receipt"].__setitem__(
            "formal_validation_complete", True
        ),
    ],
)
def test_rehearsal_report_fails_closed_on_stage_or_isolation_drift(
    protocol: dict[str, object],
    report: dict[str, object],
    mutator,
) -> None:
    changed = copy.deepcopy(report)
    mutator(changed)
    _rehash_report(changed)
    with pytest.raises(DressRehearsalError):
        verify_rehearsal_report(changed, protocol)


def test_failure_matrix_rejects_every_preregistered_attack(
    protocol: dict[str, object], report: dict[str, object]
) -> None:
    first = simulate_failures(ROOT, protocol, report)
    second = simulate_failures(ROOT, protocol, report)
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == len(FAILURE_SCENARIOS)
    assert [row["scenario"] for row in first["scenarios"]] == list(
        FAILURE_SCENARIOS
    )
    assert all(row["rejected"] is True for row in first["scenarios"])
    assert first["real_state_mutation_count"] == 0


def test_protocol_and_handoff_are_hash_bound_and_deterministic(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    first = build_handoff_checklist(protocol)
    second = build_handoff_checklist(protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["formal_validation_complete"] is False
    assert first["real_blockers"] == list(REAL_BLOCKERS)
    assert len(first["steps"]) == 8

    drifted = copy.deepcopy(protocol)
    drifted["population"]["query_count"] = 999
    path = tmp_path / "drifted.json"
    path.write_bytes(canonical_json(drifted))
    with pytest.raises(DressRehearsalError):
        load_protocol(path, repository_root=ROOT)


def test_real_readiness_remains_blocked(
    protocol: dict[str, object], report: dict[str, object]
) -> None:
    readiness = audit_readiness(ROOT, protocol, report)
    assert readiness["exit_code"] == EXIT_BLOCKED
    assert readiness["status"] == "real_external_evidence_still_blocked"
    assert readiness["real_blockers"] == list(REAL_BLOCKERS)
    assert readiness["real_blocker_count"] == 3
    assert readiness["formal_validation_complete"] is False


def test_cli_verify_failure_matrix_and_readiness_are_stable() -> None:
    commands = (
        (["verify"], 0, "rehearsal_completed"),
        (["simulate-failure"], 0, "rehearsal_completed"),
        (["audit-readiness"], 3, "real_external_evidence_still_blocked"),
    )
    for arguments, expected_exit, expected_status in commands:
        runs = [
            subprocess.run(
                [sys.executable, str(CLI), *arguments],
                cwd=ROOT,
                capture_output=True,
                check=False,
                timeout=60,
            )
            for _ in range(2)
        ]
        assert runs[0].returncode == runs[1].returncode == expected_exit
        assert runs[0].stderr == runs[1].stderr == b""
        assert runs[0].stdout == runs[1].stdout
        assert json.loads(runs[0].stdout)["status"] == expected_status


def test_cli_malformed_input_has_no_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "protocol.json"
    malformed.write_text('{"protocol": "formal_validation_dress_rehearsal_v1"}')
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--protocol",
            str(malformed),
            "verify",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    payload = json.loads(result.stdout)
    assert payload["status"] == "integration_or_isolation_violation"
    assert "traceback" not in result.stdout.decode("utf-8").lower()
