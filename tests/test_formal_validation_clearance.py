from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.check_formal_validation_clearance as cli
from scholar_agent.evaluation.formal_validation_clearance import (
    BLOCKERS,
    EXIT_BLOCKED,
    EXIT_VALID,
    EXIT_VIOLATION,
    ClearanceBlocked,
    ClearanceError,
    build_current_evidence,
    canonical_json,
    conformance_evidence,
    evaluate,
    issue_receipt,
    load_protocol,
    stable_hash,
    verify_receipt,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_validation_clearance_v1_protocol.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH)


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(value)
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = stable_hash(value)
    return value


def test_current_evidence_preserves_all_three_blockers(protocol: dict[str, object]) -> None:
    evidence = build_current_evidence(protocol, repository_root=ROOT)
    report = evaluate(evidence)
    assert report["status"] == "partially_satisfied"
    assert report["exit_code"] == EXIT_BLOCKED
    assert report["formal_validation_complete"] is False
    assert report["blockers"]["full1000"]["state"] == "partially_satisfied"
    assert report["blockers"]["human_precision"]["state"] == "partially_satisfied"
    assert report["blockers"]["official_scorer"]["state"] == "blocked"
    assert report["global_prerequisites"]["failed"] == []


@pytest.mark.parametrize("satisfied", [(), ("full1000",), ("human_precision",), ("official_scorer",), ("full1000", "human_precision")])
def test_partial_transition_matrix(satisfied: tuple[str, ...]) -> None:
    report = evaluate(conformance_evidence(satisfied=satisfied))
    assert report["exit_code"] == EXIT_BLOCKED
    assert report["status"] == "partially_satisfied"
    for blocker in BLOCKERS:
        expected_blocker = "eligible_for_clearance" if blocker in satisfied else "partially_satisfied"
        assert report["blockers"][blocker]["state"] == expected_blocker


def test_all_synthetic_conformance_transitions_to_test_only_cleared(
    protocol: dict[str, object],
) -> None:
    evidence = conformance_evidence()
    report = evaluate(evidence)
    assert report["status"] == "eligible_for_clearance"
    receipt = issue_receipt(evidence, protocol)
    assert receipt["status"] == "cleared"
    assert receipt["synthetic_test_only"] is True
    assert receipt["formal_validation_complete"] is False
    verified = verify_receipt(receipt, evidence, protocol)
    assert verified["status"] == "cleared"
    assert verified["formal_validation_complete"] is False


@pytest.mark.parametrize(
    ("path", "value", "failed_predicate"),
    [
        (("full1000", "legacy_input"), True, "not_legacy_partial_or_synthetic"),
        (("full1000", "synthetic_input"), True, "not_legacy_partial_or_synthetic"),
        (("human_precision", "synthetic_only"), True, "real_human_only"),
        (("human_precision", "label_origin"), "llm", "real_human_only"),
        (("official_scorer", "synthetic_only"), True, "not_synthetic"),
    ],
)
def test_spoofed_or_legacy_evidence_cannot_clear(
    path: tuple[str, str], value: object, failed_predicate: str
) -> None:
    evidence = conformance_evidence()
    evidence["blockers"][path[0]][path[1]] = value
    evidence = _rehash(evidence)
    report = evaluate(evidence)
    assert report["exit_code"] == EXIT_BLOCKED
    assert failed_predicate in report["blockers"][path[0]]["failed"]


@pytest.mark.parametrize(
    ("field", "value", "failed"),
    [
        ("stale_count", 1, "fresh"),
        ("current_rules_default", False, "current_rules_default"),
        ("deterministic_tiebreak_v2_default", True, "default_tiebreak_unchanged"),
        ("source_commit_compatible", False, "source_commit_compatible"),
    ],
)
def test_global_prerequisite_drift_blocks_clearance(field: str, value: object, failed: str) -> None:
    evidence = conformance_evidence()
    evidence["global_prerequisites"][field] = value
    evidence = _rehash(evidence)
    report = evaluate(evidence)
    assert report["exit_code"] == EXIT_BLOCKED
    assert failed in report["global_prerequisites"]["failed"]


def test_hash_tamper_is_invalid() -> None:
    evidence = conformance_evidence()
    evidence["blockers"]["full1000"]["committed_query_count"] = 999
    report = evaluate(evidence)
    assert report["status"] == "invalid"
    assert report["exit_code"] == EXIT_VIOLATION
    assert report["error_code"] == "evidence_hash_mismatch"


def test_cross_commit_receipt_is_rejected(protocol: dict[str, object]) -> None:
    evidence = conformance_evidence()
    evidence["source_commit"] = "2" * 40
    evidence = _rehash(evidence)
    with pytest.raises(ClearanceError, match="commit_mismatch"):
        issue_receipt(evidence, protocol)


def test_receipt_tamper_is_invalid(protocol: dict[str, object]) -> None:
    evidence = conformance_evidence()
    receipt = issue_receipt(evidence, protocol)
    receipt["verification_commands"] = []
    report = verify_receipt(receipt, evidence, protocol)
    assert report["status"] == "invalid"
    assert report["exit_code"] == EXIT_VIOLATION


def test_duplicate_receipt_write_is_rejected(tmp_path: Path, protocol: dict[str, object]) -> None:
    receipt = issue_receipt(conformance_evidence(), protocol)
    path = tmp_path / "receipt.json"
    write_json(path, receipt, exclusive=True)
    with pytest.raises(ClearanceError, match="already_exists"):
        write_json(path, receipt, exclusive=True)


def test_report_and_receipt_are_byte_deterministic(protocol: dict[str, object]) -> None:
    first = conformance_evidence()
    second = conformance_evidence()
    assert canonical_json(evaluate(first)) == canonical_json(evaluate(second))
    assert canonical_json(issue_receipt(first, protocol)) == canonical_json(issue_receipt(second, protocol))


def test_cli_current_audit_and_blocked_issue(
    tmp_path: Path, protocol: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["audit-current"]) == EXIT_BLOCKED
    current = build_current_evidence(protocol, repository_root=ROOT)
    evidence_path = tmp_path / "current.json"
    write_json(evidence_path, current)
    assert cli.main(["issue-receipt", "--evidence", str(evidence_path), "--receipt", str(tmp_path / "receipt.json")]) == EXIT_BLOCKED
    output = capsys.readouterr().out
    assert "formal_validation_complete" in output


def test_cli_synthetic_receipt_roundtrip_and_duplicate_rejection(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "synthetic.json"
    receipt_path = tmp_path / "receipt.json"
    write_json(evidence_path, conformance_evidence())
    args = ["issue-receipt", "--evidence", str(evidence_path), "--receipt", str(receipt_path)]
    assert cli.main(args) == EXIT_VALID
    assert cli.main(args) == EXIT_VIOLATION
    assert cli.main(["verify-receipt", "--evidence", str(evidence_path), "--receipt", str(receipt_path)]) == EXIT_VALID
