from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_validation_preregistration import (
    EXIT_BLOCKED,
    EXIT_SEALED,
    PreregistrationError,
    audit_readiness,
    build_seal,
    canonical_json,
    evaluate_amendment,
    evaluate_timeline,
    load_protocol,
    synthetic_amendment_matrix,
    verify_seal,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_validation_preregistration_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_validation_preregistration.py"


def test_current_protocol_seal_is_deterministic_and_closed() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    first = build_seal(protocol, repository_root=ROOT)
    second = build_seal(protocol, repository_root=ROOT)
    assert canonical_json(first) == canonical_json(second)
    assert first["state"] == "sealed"
    assert first["external_evidence"] == {
        "full1000": "not_available",
        "human_precision": "not_available",
        "official_scorer": "not_provided",
    }
    report = verify_seal(first, protocol, repository_root=ROOT)
    assert report["status"] == "preregistration_sealed"
    assert report["execution"]["quality_metric_count"] == 0


def test_seal_or_registered_dependency_tamper_is_rejected() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    seal = build_seal(protocol, repository_root=ROOT)
    tampered = copy.deepcopy(seal)
    tampered["allowed_outputs"].append("unregistered.json")
    with pytest.raises(PreregistrationError, match="seal_content_or_dependency_drift"):
        verify_seal(tampered, protocol, repository_root=ROOT)

    drifted = copy.deepcopy(protocol)
    drifted["dependencies"][0]["sha256"] = "0" * 64
    with pytest.raises(PreregistrationError, match="protocol_digest_mismatch"):
        build_seal(drifted, repository_root=ROOT)


def test_chronology_requires_seal_before_intake_and_execution_before_scoring() -> None:
    good = [
        {"event": "preregistration_sealed"},
        {"event": "execution_started"},
        {"event": "evidence_intake"},
        {"event": "unblind_or_score"},
    ]
    assert evaluate_timeline(good)["chronology_valid"] is True
    with pytest.raises(PreregistrationError, match="chronology_violation"):
        evaluate_timeline(
            [
                {"event": "evidence_intake"},
                {"event": "preregistration_sealed"},
                {"event": "execution_started"},
                {"event": "unblind_or_score"},
            ]
        )
    with pytest.raises(PreregistrationError, match="timeline_event_missing"):
        evaluate_timeline(good[:-1])


@pytest.mark.parametrize(
    "pointer",
    [
        "/human_annotation/coverage_threshold",
        "/population/exclusion_rules",
        "/statistics/resampling_unit",
        "/statistics/new_metric",
        "/analysis/strategy",
        "/declaration_boundaries/full1000",
    ],
)
def test_post_evidence_semantic_changes_invalidate_formal_claim(pointer: str) -> None:
    report = evaluate_amendment(
        changed_pointers=[pointer],
        evidence_intake_present=True,
        semantic_digest_before="a" * 64,
        semantic_digest_after="b" * 64,
    )
    assert report["state"] == "invalid_post_evidence_change"
    assert report["valid"] is False


def test_pre_evidence_amendment_and_proven_nonsemantic_erratum() -> None:
    before = evaluate_amendment(
        changed_pointers=["/statistics/report_decimal_places"],
        evidence_intake_present=False,
        semantic_digest_before="a" * 64,
        semantic_digest_after="b" * 64,
    )
    assert before["state"] == "amended_before_evidence"
    assert before["valid"] is True

    erratum = evaluate_amendment(
        changed_pointers=["/documentation/typo"],
        evidence_intake_present=True,
        semantic_digest_before="a" * 64,
        semantic_digest_after="a" * 64,
        declared_nonsemantic=True,
    )
    assert erratum["state"] == "sealed"
    with pytest.raises(PreregistrationError, match="nonsemantic_erratum_not_proven"):
        evaluate_amendment(
            changed_pointers=["/statistics/method"],
            evidence_intake_present=True,
            semantic_digest_before="a" * 64,
            semantic_digest_after="a" * 64,
            declared_nonsemantic=True,
        )


def test_synthetic_matrix_and_real_readiness_keep_three_blockers() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    seal = build_seal(protocol, repository_root=ROOT)
    matrix = synthetic_amendment_matrix()
    assert matrix["scenario_count"] == 7
    assert matrix["timeline_scenario_count"] == 2
    assert matrix["scenarios"][0]["state"] == "sealed"
    assert len(canonical_json(matrix)) == len(canonical_json(synthetic_amendment_matrix()))
    report = audit_readiness(protocol, seal, repository_root=ROOT)
    assert report["exit_code"] == EXIT_BLOCKED
    assert report["blocker_count"] == 3
    assert report["formal_validation_complete"] is False


def test_cli_build_verify_simulate_and_audit_are_stable(tmp_path: Path) -> None:
    seal = tmp_path / "seal.json"
    build = subprocess.run(
        [sys.executable, str(CLI), "build", "--output", str(seal)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert build.returncode == EXIT_SEALED
    assert build.stderr == b""
    assert build.stdout == seal.read_bytes()

    verify = subprocess.run(
        [sys.executable, str(CLI), "verify", "--seal", str(seal)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert verify.returncode == EXIT_SEALED
    assert json.loads(verify.stdout)["status"] == "preregistration_sealed"

    first = subprocess.run(
        [sys.executable, str(CLI), "simulate-amendment"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    second = subprocess.run(
        [sys.executable, str(CLI), "simulate-amendment"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert first.returncode == second.returncode == EXIT_SEALED
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""

    audit = subprocess.run(
        [sys.executable, str(CLI), "audit-readiness", "--seal", str(seal)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert audit.returncode == EXIT_BLOCKED
    assert json.loads(audit.stdout)["status"] == "sealed_with_external_evidence_blockers"


def test_cli_malformed_input_has_no_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "protocol.json"
    malformed.write_text('{"schema_version": "1"}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "--protocol", str(malformed), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout
