from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
import importlib.util

import pytest

from scholar_agent.evaluation.evidence_revocation import (
    EXIT_BLOCKED,
    EXIT_READY,
    ActiveIncident,
    RevocationError,
    append_event,
    assert_no_active_incident,
    audit_current,
    load_current,
    new_empty_ledger,
    propagation_report,
    simulate_incidents,
    stable_hash,
    verify_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/check_evidence_revocation.py"


def _inputs():
    return load_current(ROOT)


def _revoked(evidence_id: str = "ranking_decision_manifest"):
    protocol, _ledger, freshness, readiness = _inputs()
    ledger = append_event(
        new_empty_ledger(protocol),
        protocol,
        evidence_id=evidence_id,
        after_state="revoked",
        reason_code="implementation_defect",
        trigger_evidence_sha256=stable_hash({"trigger": evidence_id}),
        operator_identity="operator_test_auditor",
        impact_scope=["claims", "gates", "publication"],
    )
    return protocol, ledger, freshness, readiness


def test_real_ledger_is_empty_and_controls_are_ready() -> None:
    first = audit_current(ROOT)
    second = audit_current(ROOT)
    assert first == second
    assert first["exit_code"] == EXIT_READY
    assert first["real_ledger_empty"] is True
    assert first["active_incident_count"] == 0


@pytest.mark.parametrize(
    "reason",
    [
        "content_tampering",
        "duplicate_or_wrong_publication",
        "erroneous_extrapolation",
        "implementation_defect",
        "input_identity_error",
        "protocol_error",
        "sensitive_information_leakage",
        "stale_dependency",
        "statistical_error",
    ],
)
def test_all_structured_reason_codes_are_accepted(reason: str) -> None:
    protocol, _ledger, freshness, _readiness = _inputs()
    ledger = append_event(
        new_empty_ledger(protocol),
        protocol,
        evidence_id="ranking_decision_manifest",
        after_state="under_investigation",
        reason_code=reason,
        trigger_evidence_sha256=stable_hash({"reason": reason}),
        operator_identity="operator_test_auditor",
        impact_scope=["claims"],
    )
    assert verify_ledger(
        ledger, protocol, freshness_contract=freshness
    )["active_incident_count"] == 1


def test_revocation_propagates_and_preserves_unrelated_evidence() -> None:
    protocol, ledger, freshness, readiness = _revoked(
        "human_annotation_delivery_protocol"
    )
    report = propagation_report(ledger, protocol, freshness, readiness)
    assert report["exit_code"] == EXIT_BLOCKED
    assert "human_annotation_delivery_protocol" in report["impacted_evidence_ids"]
    assert "source_reliability_protocol" in report["unaffected_evidence_ids"]
    assert set(report["invalidated_publication_targets"]) == {
        "clearance_receipt",
        "release_candidate",
        "standalone_auditor_bundle",
        "validation_readiness_bundle",
    }


def test_active_incident_blocks_every_publication_target(tmp_path: Path) -> None:
    protocol, ledger, _freshness, _readiness = _revoked("evidence_registry_gate")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_bytes(
        (
            json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode()
    )
    for target in (
        "clearance_receipt",
        "release_candidate",
        "standalone_auditor_bundle",
        "validation_readiness_bundle",
    ):
        with pytest.raises(ActiveIncident):
            assert_no_active_incident(
                ROOT, target=target, ledger_path=ledger_path
            )


def test_supersession_and_restoration_require_fresh_gated_replacement() -> None:
    protocol, ledger, freshness, _readiness = _revoked()
    replacement = next(
        row
        for row in freshness["bindings"]["evidence"]
        if row["declared_state"] == "fresh"
        and row["evidence_id"] != "ranking_decision_manifest"
    )
    gate = next(
        row["gate_id"]
        for row in freshness["bindings"]["gates"]
        if row["declared_state"] == "fresh"
    )
    superseded = append_event(
        ledger,
        protocol,
        evidence_id="ranking_decision_manifest",
        after_state="superseded",
        reason_code="implementation_defect",
        trigger_evidence_sha256=stable_hash({"replacement": 1}),
        operator_identity="operator_test_auditor",
        impact_scope=["claims", "gates"],
        replacement_evidence_id=replacement["evidence_id"],
        replacement_evidence_sha256=replacement["artifact_sha256"],
        replacement_gate_ids=[gate],
        freshness_contract=freshness,
    )
    restored = append_event(
        superseded,
        protocol,
        evidence_id="ranking_decision_manifest",
        after_state="restored",
        reason_code="implementation_defect",
        trigger_evidence_sha256=stable_hash({"replacement": 2}),
        operator_identity="operator_test_auditor",
        impact_scope=["claims", "gates"],
        replacement_evidence_id=replacement["evidence_id"],
        replacement_evidence_sha256=replacement["artifact_sha256"],
        replacement_gate_ids=[gate],
        freshness_contract=freshness,
    )
    assert verify_ledger(
        restored, protocol, freshness_contract=freshness
    )["active_incident_count"] == 0
    other = next(
        row
        for row in freshness["bindings"]["evidence"]
        if row["declared_state"] == "fresh"
        and row["evidence_id"]
        not in {"ranking_decision_manifest", replacement["evidence_id"]}
    )
    with pytest.raises(
        RevocationError, match="restoration_replacement_identity_drift"
    ):
        append_event(
            superseded,
            protocol,
            evidence_id="ranking_decision_manifest",
            after_state="restored",
            reason_code="implementation_defect",
            trigger_evidence_sha256=stable_hash({"replacement": "changed"}),
            operator_identity="operator_test_auditor",
            impact_scope=["claims", "gates"],
            replacement_evidence_id=other["evidence_id"],
            replacement_evidence_sha256=other["artifact_sha256"],
            replacement_gate_ids=[gate],
            freshness_contract=freshness,
        )
    with pytest.raises(RevocationError, match="replacement_evidence_not_fully_gated"):
        append_event(
            ledger,
            protocol,
            evidence_id="ranking_decision_manifest",
            after_state="superseded",
            reason_code="implementation_defect",
            trigger_evidence_sha256=stable_hash({"replacement": "forged"}),
            operator_identity="operator_test_auditor",
            impact_scope=["claims"],
            replacement_evidence_id=replacement["evidence_id"],
            replacement_evidence_sha256="f" * 64,
            replacement_gate_ids=[gate],
            freshness_contract=freshness,
        )


@pytest.mark.parametrize("attack", ["delete", "reorder", "forge", "duplicate"])
def test_append_only_ledger_attacks_fail_closed(attack: str) -> None:
    protocol, ledger, freshness, _readiness = _revoked()
    second = append_event(
        ledger,
        protocol,
        evidence_id="source_reliability_protocol",
        after_state="revoked",
        reason_code="protocol_error",
        trigger_evidence_sha256=stable_hash({"second": True}),
        operator_identity="operator_test_auditor",
        impact_scope=["claims"],
    )
    tampered = copy.deepcopy(second)
    if attack == "delete":
        tampered["events"] = tampered["events"][1:]
    elif attack == "reorder":
        tampered["events"].reverse()
    elif attack == "forge":
        tampered["events"][0]["reason_code"] = "statistical_error"
    else:
        tampered["events"].append(copy.deepcopy(tampered["events"][0]))
    content = dict(tampered)
    content["ledger_sha256"] = "0" * 64
    tampered["ledger_sha256"] = stable_hash(content)
    with pytest.raises(RevocationError):
        verify_ledger(tampered, protocol, freshness_contract=freshness)


def test_synthetic_matrix_closes_expected_scenarios() -> None:
    protocol, _ledger, freshness, readiness = _inputs()
    first = simulate_incidents(protocol, freshness, readiness)
    second = simulate_incidents(protocol, freshness, readiness)
    assert first == second
    assert first["scenario_count"] == 5
    assert first["status"] == "revocation_controls_ready"


def test_cli_reports_active_incident_without_traceback(tmp_path: Path) -> None:
    _protocol, ledger, _freshness, _readiness = _revoked()
    ledger_path = tmp_path / "incident.json"
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--ledger",
            str(ledger_path),
            "audit-current",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == EXIT_BLOCKED
    assert completed.stderr == ""
    assert payload["status"] == "active_incident_blocks_release"


def test_cli_malformed_ledger_is_stable_violation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text('{"events":[],"events":[]}', encoding="utf-8")
    command = [
        sys.executable,
        str(CLI),
        "--ledger",
        str(path),
        "audit-current",
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    second = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert first.returncode == 2
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""


def _script_module(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("check_release_candidate.py", ["audit-readiness"]),
        ("check_formal_validation_clearance.py", ["audit-current"]),
        ("check_validation_readiness.py", ["verify"]),
    ],
)
def test_publication_clis_consume_active_revocation_state(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    script: str,
    arguments: list[str],
) -> None:
    module = _script_module(script)

    def blocked(*_args, **_kwargs):
        raise ActiveIncident("active_incident_blocks_test")

    monkeypatch.setattr(module, "assert_no_active_incident", blocked)
    assert module.main(arguments) == EXIT_BLOCKED
    captured = capfd.readouterr()
    assert captured.err == ""
    assert "active_incident" in captured.out
