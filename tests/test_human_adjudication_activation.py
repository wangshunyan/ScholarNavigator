from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.human_adjudication_activation import (
    EXECUTION_ZERO,
    HumanAdjudicationActivationError,
    _digest_without,
    _prepare_synthetic,
    audit_readiness,
    build_synthetic_result,
    canonical_json,
    load_protocol,
    read_object,
    simulate_matrix,
    unlock_statistics,
    verify_event_chain,
    verify_package,
    verify_result,
    write_object,
)
from scholar_agent.evaluation.human_annotation_submission_intake import (
    _append_event as _append_submission_event,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/human_adjudication_activation_v1_protocol.json"
SCRIPT = ROOT / "scripts/check_human_adjudication_activation.py"


@pytest.fixture()
def protocol() -> dict:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture()
def synthetic(tmp_path: Path, protocol: dict) -> dict:
    return _prepare_synthetic(tmp_path, ROOT, protocol)


def _rehash_result(value: dict) -> None:
    import hashlib

    value["decisions_sha256"] = hashlib.sha256(
        canonical_json(value["decisions"])
    ).hexdigest()
    value["result_sha256"] = _digest_without(value, "result_sha256")


def test_protocol_and_blind_package_are_bound(
    protocol: dict, synthetic: dict
) -> None:
    package = verify_package(synthetic["package"], protocol)
    assert package["item_count"] > 1
    assert set(package["rows"][0]) == {
        "abstract",
        "annotation_a",
        "annotation_b",
        "disagreement_alias",
        "query",
        "title",
        "year",
    }
    serialized = canonical_json(package).lower()
    for token in (
        b'"arm"',
        b'"rank"',
        b'"source"',
        b'"score"',
        b'"gold"',
        b'"qrels"',
        b'"item_id"',
    ):
        assert token not in serialized


def test_valid_result_advances_append_only_state(
    protocol: dict, synthetic: dict
) -> None:
    report = verify_result(
        synthetic["package"],
        synthetic["ack"],
        synthetic["result"],
        synthetic["activation_ledger"],
        synthetic["submission_ledger"],
        protocol,
    )
    assert report["state"] == "validated"
    ledger = read_object(synthetic["activation_ledger"])
    assert verify_event_chain(ledger) == "validated"
    assert [row["state"] for row in ledger["events"]] == [
        "queue_ready",
        "issued",
        "acknowledged",
        "adjudication_submitted",
        "validated",
    ]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "adjudication_coverage_invalid"),
        ("extra", "adjudication_coverage_invalid"),
        ("illegal", "adjudication_decision_invalid"),
        ("wrong_actor", "adjudication_result_invalid"),
        ("empty_rationale", "adjudication_decision_invalid"),
    ],
)
def test_result_rejects_invalid_population_and_actor(
    tmp_path: Path,
    protocol: dict,
    synthetic: dict,
    mutation: str,
    reason: str,
) -> None:
    value = read_object(synthetic["result"])
    if mutation == "missing":
        value["decisions"] = value["decisions"][:-1]
    elif mutation == "extra":
        value["decisions"].append(
            {
                "disagreement_alias": "disagreement-" + "f" * 24,
                "final_label": value["decisions"][0]["final_label"],
                "rationale": "extra",
            }
        )
    elif mutation == "illegal":
        value["decisions"][0]["final_label"] = "illegal"
    elif mutation == "wrong_actor":
        value["adjudicator_principal_id"] = "prn_ffffffffffffffff"
    else:
        value["decisions"][0]["rationale"] = ""
    _rehash_result(value)
    path = tmp_path / f"{mutation}.json"
    write_object(path, value)
    with pytest.raises(HumanAdjudicationActivationError, match=reason):
        verify_result(
            synthetic["package"],
            synthetic["ack"],
            path,
            synthetic["activation_ledger"],
            synthetic["submission_ledger"],
            protocol,
        )


def test_post_lock_tamper_and_duplicate_submission_fail(
    protocol: dict, synthetic: dict
) -> None:
    value = read_object(synthetic["result"])
    value["decisions"][0]["rationale"] = "changed without relock"
    write_object(synthetic["result"], value)
    with pytest.raises(
        HumanAdjudicationActivationError, match="adjudication_result_invalid"
    ):
        verify_result(
            synthetic["package"],
            synthetic["ack"],
            synthetic["result"],
            synthetic["activation_ledger"],
            synthetic["submission_ledger"],
            protocol,
        )
    build_synthetic_result(synthetic["package"], synthetic["result"])
    verify_result(
        synthetic["package"],
        synthetic["ack"],
        synthetic["result"],
        synthetic["activation_ledger"],
        synthetic["submission_ledger"],
        protocol,
    )
    with pytest.raises(
        HumanAdjudicationActivationError, match="duplicate_or_invalid"
    ):
        verify_result(
            synthetic["package"],
            synthetic["ack"],
            synthetic["result"],
            synthetic["activation_ledger"],
            synthetic["submission_ledger"],
            protocol,
        )


def test_no_disagreement_package_is_empty(tmp_path: Path, protocol: dict) -> None:
    fixture = _prepare_synthetic(
        tmp_path, ROOT, protocol, no_disagreement=True
    )
    package = read_object(fixture["package"])
    assert package["item_count"] == 0
    assert package["rows"] == []
    verify_result(
        fixture["package"],
        fixture["ack"],
        fixture["result"],
        fixture["activation_ledger"],
        fixture["submission_ledger"],
        protocol,
    )


def test_complete_chain_unlocks_only_existing_non_official_statistics(
    protocol: dict, synthetic: dict
) -> None:
    verify_result(
        synthetic["package"],
        synthetic["ack"],
        synthetic["result"],
        synthetic["activation_ledger"],
        synthetic["submission_ledger"],
        protocol,
    )
    report = unlock_statistics(
        ROOT,
        protocol,
        package_path=synthetic["package"],
        acknowledgement_path=synthetic["ack"],
        result_path=synthetic["result"],
        ledger_path=synthetic["activation_ledger"],
        submission_ledger_path=synthetic["submission_ledger"],
        operator_mapping_path=synthetic["mapping"],
        submissions=synthetic["submissions"],
    )
    assert report["state"] == "statistics_eligible"
    assert report["statistics_scope"] == "human_internal_non_official"
    assert report["official_result"] is False
    assert report["absolute_precision_at_20"] is None
    assert report["absolute_precision_reason"] == (
        "unsupported_from_change_only_package"
    )


def test_statistics_rejects_result_relocked_after_validation(
    protocol: dict, synthetic: dict
) -> None:
    verify_result(
        synthetic["package"],
        synthetic["ack"],
        synthetic["result"],
        synthetic["activation_ledger"],
        synthetic["submission_ledger"],
        protocol,
    )
    result = read_object(synthetic["result"])
    result["decisions"][0]["rationale"] = "synthetic replacement after validation"
    _rehash_result(result)
    write_object(synthetic["result"], result)
    with pytest.raises(
        HumanAdjudicationActivationError,
        match="validated_result_ledger_binding_mismatch",
    ):
        unlock_statistics(
            ROOT,
            protocol,
            package_path=synthetic["package"],
            acknowledgement_path=synthetic["ack"],
            result_path=synthetic["result"],
            ledger_path=synthetic["activation_ledger"],
            submission_ledger_path=synthetic["submission_ledger"],
            operator_mapping_path=synthetic["mapping"],
            submissions=synthetic["submissions"],
        )


def test_upstream_revocation_invalidates_result_and_statistics(
    protocol: dict, synthetic: dict
) -> None:
    ledger = read_object(synthetic["submission_ledger"])
    _append_submission_event(
        ledger,
        state="revoked",
        role=None,
        binding=None,
        queue_sha256=ledger["queue"]["queue_sha256"],
    )
    write_object(synthetic["submission_ledger"], ledger)
    with pytest.raises(
        HumanAdjudicationActivationError, match="upstream_submission_revoked"
    ):
        verify_result(
            synthetic["package"],
            synthetic["ack"],
            synthetic["result"],
            synthetic["activation_ledger"],
            synthetic["submission_ledger"],
            protocol,
        )
    with pytest.raises(
        HumanAdjudicationActivationError, match="upstream_submission_revoked"
    ):
        unlock_statistics(
            ROOT,
            protocol,
            package_path=synthetic["package"],
            acknowledgement_path=synthetic["ack"],
            result_path=synthetic["result"],
            ledger_path=synthetic["activation_ledger"],
            submission_ledger_path=synthetic["submission_ledger"],
            operator_mapping_path=synthetic["mapping"],
            submissions=synthetic["submissions"],
        )


def test_matrix_is_complete_deterministic_and_non_persistent(protocol: dict) -> None:
    first = simulate_matrix(ROOT, protocol)
    second = simulate_matrix(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["passed_scenario_count"] == 13
    assert first["statistics"] is None
    assert first["execution"] == EXECUTION_ZERO
    assert first["synthetic_artifacts_persisted"] is False


def test_real_readiness_remains_blocked(protocol: dict) -> None:
    report = audit_readiness(protocol)
    assert report["exit_code"] == 3
    assert report["formal_validation_complete"] is False
    assert report["statistics"] is None
    assert set(report["formal_blockers"]) == {
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    }
    assert "real_adjudication_submission_missing" in report["blocked_reasons"]


def test_cli_audit_is_standard_library_only_and_deterministic() -> None:
    commands = [
        [sys.executable, "-I", "-S", str(SCRIPT), "audit-readiness"],
        [sys.executable, str(SCRIPT), "audit-readiness"],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd="/",
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 3
        assert completed.stderr == b""
        value = json.loads(completed.stdout)
        assert value["status"] == "not_ready_missing_real_labels_or_adjudication"
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_cli_audit_works_in_two_repository_free_environments(
    tmp_path: Path,
) -> None:
    outputs = []
    for ordinal in (1, 2):
        root = tmp_path / f"offline-{ordinal}"
        root.mkdir()
        script = root / "check.py"
        protocol = root / "protocol.json"
        shutil.copyfile(SCRIPT, script)
        shutil.copyfile(PROTOCOL_PATH, protocol)
        fake_home = root / "home"
        fake_tmp = root / "tmp"
        fake_home.mkdir()
        fake_tmp.mkdir()
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(script),
                "--protocol",
                str(protocol),
                "audit-readiness",
            ],
            cwd=root,
            env={
                "HOME": str(fake_home),
                "TMPDIR": str(fake_tmp),
                "PATH": "/usr/bin:/bin",
            },
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 3
        assert completed.stderr == b""
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_cli_usage_and_malformed_protocol_never_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"protocol":"wrong"}', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--protocol",
            str(bad),
            "audit-readiness",
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert b"Traceback" not in completed.stdout
