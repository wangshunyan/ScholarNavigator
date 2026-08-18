from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.human_annotation_assignment_activation import (
    HumanAnnotationAssignmentError,
    HumanAnnotationAssignmentNotReady,
    PROTOCOL,
    ROLES,
    build_acknowledgement,
    canonical_json,
    issue_assignments,
    label_intake_allowed,
    load_protocol,
    read_object,
    sha256_bytes,
    simulate_matrix,
    verify_acknowledgements,
    verify_bundle,
    verify_event_chain,
    write_object,
)
from scholar_agent.evaluation.human_annotator_qualification_intake import (
    build_contract,
    build_synthetic_submission,
    load_protocol as load_qualification_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/human_annotation_assignment_activation_v1_protocol.json"
)
QUALIFICATION_PROTOCOL_PATH = (
    ROOT / "benchmark/human_annotator_qualification_intake_v1_protocol.json"
)
SCRIPT = ROOT / "scripts/check_human_annotation_assignment.py"
RUNTIME = ROOT / "scripts/human_annotation_assignment_runtime.py"


def _protocol() -> dict:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _qualification_protocol() -> dict:
    return load_qualification_protocol(
        QUALIFICATION_PROTOCOL_PATH, repository_root=ROOT
    )


def _qualifications(tmp_path: Path) -> list[dict]:
    protocol = _qualification_protocol()
    rows = []
    for ordinal, role in enumerate(ROLES, 1):
        contract = build_contract(
            protocol,
            challenge=sha256_bytes(f"qualification:{role}".encode()),
            role=role,
        )
        path = tmp_path / f"qualification-{role}.json"
        build_synthetic_submission(
            contract,
            path,
            principal_id=f"prn_{ordinal:016x}",
            principal_commitment=sha256_bytes(f"principal:{role}".encode()),
        )
        rows.append(json.loads(path.read_text()))
    return rows


def _challenges() -> dict[str, str]:
    return {
        role: sha256_bytes(f"assignment:{role}".encode()) for role in ROLES
    }


def _issue(
    tmp_path: Path,
) -> tuple[dict, dict[str, Path], Path, list[dict]]:
    protocol = _protocol()
    qualifications = _qualifications(tmp_path)
    output = tmp_path / "bundles"
    ledger = tmp_path / "ledger.json"
    result = issue_assignments(
        ROOT,
        protocol,
        qualifications,
        challenges=_challenges(),
        output_root=output,
        ledger_path=ledger,
        allow_synthetic=True,
    )
    bundles = {role: output / f"{role}.zip" for role in ROLES}
    return result, bundles, ledger, qualifications


def _receipts(
    tmp_path: Path, bundles: dict[str, Path], protocol: dict
) -> list[Path]:
    paths = []
    for role in ROLES:
        path = tmp_path / f"receipt-{role}.json"
        build_acknowledgement(
            bundles[role],
            protocol,
            repository_root=ROOT,
            output=path,
        )
        paths.append(path)
    return paths


def test_protocol_is_frozen_and_all_bindings_are_current() -> None:
    protocol = _protocol()
    assert protocol["protocol"] == PROTOCOL
    assert protocol["source_commit"] == (
        "80cd4bf6f5263231a34a3ad535759f6c6910e835"
    )
    assert protocol["issuance"]["item_count_per_annotator"] == 471
    assert protocol["issuance"]["operator_mapping"] == "excluded"
    assert protocol["formal_validation_complete"] is False


def test_issue_binds_three_roles_and_preserves_blinding(tmp_path: Path) -> None:
    result, bundles, ledger, _ = _issue(tmp_path)
    assert result["state"] == "issued"
    assert result["real_label_count"] == 0
    assert not label_intake_allowed(ledger)
    aliases = {}
    for role in ROLES:
        manifest = verify_bundle(
            bundles[role],
            _protocol(),
            repository_root=ROOT,
            expected_role=role,
        )
        assert manifest["principal_id"].startswith("prn_")
        with zipfile.ZipFile(bundles[role]) as archive:
            names = set(archive.namelist())
            assert not any(name.startswith("operator/") for name in names)
            assert ".env" not in names
            if role == "adjudicator":
                assert "payload/items.json" not in names
                assert "disagreement_view_contract.json" in names
                assert manifest["item_count"] == 0
            else:
                items = json.loads(archive.read("payload/items.json"))
                assert len(items) == 471
                aliases[role] = {row["alias"] for row in items}
                assert all(
                    set(row) == {"abstract", "alias", "query", "title", "year"}
                    for row in items
                )
    assert aliases["annotator_a"].isdisjoint(aliases["annotator_b"])


def test_all_acknowledgements_lock_label_intake(tmp_path: Path) -> None:
    _, bundles, ledger, _ = _issue(tmp_path)
    protocol = _protocol()
    receipts = _receipts(tmp_path, bundles, protocol)
    result = verify_acknowledgements(
        bundles,
        receipts,
        ledger,
        protocol,
        repository_root=ROOT,
    )
    assert result["state"] == "locked_for_submission"
    assert result["label_intake_allowed"] is True
    assert result["real_label_count"] == 0
    assert label_intake_allowed(ledger)
    states = verify_event_chain(read_object(ledger))
    assert states == {role: "locked_for_submission" for role in ROLES}


def test_partial_acknowledgement_cannot_enable_label_intake(
    tmp_path: Path,
) -> None:
    _, bundles, ledger, _ = _issue(tmp_path)
    protocol = _protocol()
    receipts = _receipts(tmp_path, bundles, protocol)
    before = ledger.read_bytes()
    with pytest.raises(
        HumanAnnotationAssignmentNotReady, match="acknowledgement_incomplete"
    ):
        verify_acknowledgements(
            bundles,
            receipts[:2],
            ledger,
            protocol,
            repository_root=ROOT,
        )
    assert ledger.read_bytes() == before
    assert not label_intake_allowed(ledger)


def test_wrong_principal_coordinator_and_duplicate_claim_fail_closed(
    tmp_path: Path,
) -> None:
    _, bundles, ledger, _ = _issue(tmp_path)
    protocol = _protocol()
    receipts = _receipts(tmp_path, bundles, protocol)
    wrong = read_object(receipts[0])
    wrong["principal_id"] = "prn_ffffffffffffffff"
    wrong["receipt_sha256"] = "0" * 64
    wrong["receipt_sha256"] = sha256_bytes(
        canonical_json({**wrong, "receipt_sha256": "0" * 64})
    )
    write_object(receipts[0], wrong)
    with pytest.raises(
        HumanAnnotationAssignmentError, match="receipt_binding_invalid"
    ):
        verify_acknowledgements(
            bundles,
            receipts,
            ledger,
            protocol,
            repository_root=ROOT,
        )
    build_acknowledgement(
        bundles["annotator_a"],
        protocol,
        repository_root=ROOT,
        output=receipts[0],
    )
    verify_acknowledgements(
        bundles,
        receipts,
        ledger,
        protocol,
        repository_root=ROOT,
    )
    with pytest.raises(HumanAnnotationAssignmentError):
        verify_acknowledgements(
            bundles,
            receipts,
            ledger,
            protocol,
            repository_root=ROOT,
        )


def test_event_chain_tamper_and_semantic_drift_are_rejected(
    tmp_path: Path,
) -> None:
    _, bundles, ledger, _ = _issue(tmp_path)
    value = read_object(ledger)
    value["events"][2]["principal_id"] = "prn_ffffffffffffffff"
    write_object(ledger, value)
    with pytest.raises(HumanAnnotationAssignmentError):
        verify_event_chain(read_object(ledger))

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    _, clean_bundles, clean_ledger, _ = _issue(clean_root)
    protocol = _protocol()
    receipts = _receipts(clean_root, clean_bundles, protocol)
    receipt = read_object(receipts[0])
    receipt["assignment_protocol_sha256"] = "f" * 64
    receipt["receipt_sha256"] = sha256_bytes(
        canonical_json({**receipt, "receipt_sha256": "0" * 64})
    )
    write_object(receipts[0], receipt)
    with pytest.raises(HumanAnnotationAssignmentError):
        verify_acknowledgements(
            clean_bundles,
            receipts,
            clean_ledger,
            protocol,
            repository_root=ROOT,
        )


def test_synthetic_matrix_covers_all_attacks() -> None:
    result = simulate_matrix(ROOT, _protocol(), _qualification_protocol())
    assert result["scenario_count"] == 12
    assert result["passed_count"] == 12
    assert result["real_label_count"] == 0
    assert result["real_package_distributed"] is False
    assert {row["scenario"] for row in result["scenarios"]} == {
        "valid_three_role_issue",
        "annotator_package_swap",
        "shared_principal",
        "adjudicator_early_unblinding",
        "duplicate_claim",
        "package_tamper",
        "revoked_qualification",
        "expired_qualification",
        "commit_drift",
        "partial_acknowledgement",
        "post_issue_protocol_change",
        "coordinator_claim",
    }
    assert all(row["expected"] == row["observed"] for row in result["scenarios"])


def test_bundle_and_ledger_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, first_bundles, first_ledger, _ = _issue(first)
    _, second_bundles, second_ledger, _ = _issue(second)
    assert first_ledger.read_bytes() == second_ledger.read_bytes()
    for role in ROLES:
        assert first_bundles[role].read_bytes() == second_bundles[role].read_bytes()


def test_standard_library_verifier_works_in_two_repo_free_environments(
    tmp_path: Path,
) -> None:
    _, bundles, _, _ = _issue(tmp_path)
    for ordinal, role in enumerate(("annotator_a", "adjudicator"), 1):
        isolated = tmp_path / f"isolated-{ordinal}"
        isolated.mkdir()
        verifier = isolated / "verify.py"
        verifier.write_bytes(RUNTIME.read_bytes())
        bundle = isolated / "assignment.zip"
        bundle.write_bytes(bundles[role].read_bytes())
        home = isolated / "home"
        temp = isolated / "temp"
        home.mkdir()
        temp.mkdir()
        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "TMPDIR": str(temp),
        }
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(verifier), str(bundle)],
            cwd=isolated,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == ""
        assert json.loads(completed.stdout)["status"] == (
            "assignment_bundle_verified"
        )


def test_cli_audit_readiness_lists_real_role_and_acknowledgement_gaps() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "audit-readiness"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 3
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result["formal_validation_complete"] is False
    assert result["human_precision_verified"] is False
    assert result["label_intake_allowed"] is False
    assert result["blocked_reasons"] == [
        "annotator_a_real_qualification_missing",
        "annotator_b_real_qualification_missing",
        "adjudicator_real_qualification_missing",
        "annotator_a_assignment_acknowledgement_missing",
        "annotator_b_assignment_acknowledgement_missing",
        "adjudicator_assignment_acknowledgement_missing",
    ]


def test_cli_prepare_is_deterministic_and_contains_no_identity(
    tmp_path: Path,
) -> None:
    outputs = []
    for ordinal in (1, 2):
        path = tmp_path / f"prepared-{ordinal}.json"
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "prepare", "--output", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == ""
        outputs.append(path.read_bytes())
        result = json.loads(completed.stdout)
        assert result["state"] == "prepared"
    assert outputs[0] == outputs[1]
    assert b"principal" not in outputs[0]
    prepared = json.loads(outputs[0])
    assert "labels" not in prepared
    assert prepared["label_intake_allowed"] is False


def test_cli_usage_and_malformed_inputs_never_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "protocol.json"
    malformed.write_text('{"protocol":')
    for arguments, expected in (
        ([], 4),
        (["--protocol", str(malformed), "audit-readiness"], 2),
    ):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == expected
        assert completed.stderr == ""
        assert "Traceback" not in completed.stdout


def test_real_issue_rejects_synthetic_qualification(tmp_path: Path) -> None:
    with pytest.raises(
        HumanAnnotationAssignmentError, match="real_qualification_required"
    ):
        issue_assignments(
            ROOT,
            _protocol(),
            _qualifications(tmp_path),
            challenges=_challenges(),
            output_root=tmp_path / "bundles",
            ledger_path=tmp_path / "ledger.json",
            allow_synthetic=False,
        )
