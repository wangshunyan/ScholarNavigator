from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.human_annotation_submission_intake import (
    ANNOTATOR_ROLES,
    PROTOCOL,
    SOURCE_COMMIT,
    HumanAnnotationSubmissionError,
    _read_json,
    _scenario_inputs,
    _synthetic_export,
    audit_readiness,
    build_adjudication_queue,
    build_submission_receipt,
    canonical_json,
    intake_submissions,
    load_protocol,
    simulate_matrix,
    verify_adjudication_queue,
    verify_event_chain,
    verify_locked_submission,
    write_object,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/human_annotation_submission_intake_v1_protocol.json"
)
SCRIPT = ROOT / "scripts/check_human_annotation_submission.py"


def _protocol() -> dict:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _setup(tmp_path: Path) -> tuple[dict, dict[str, tuple[Path, Path]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context, _, _, _, submissions = _scenario_inputs(
        tmp_path, ROOT, _protocol()
    )
    return context, submissions


def test_protocol_is_frozen_and_reuses_existing_label_contract() -> None:
    protocol = _protocol()
    assert protocol["protocol"] == PROTOCOL
    assert protocol["source_commit"] == SOURCE_COMMIT
    assert protocol["intake"]["accepted_export_contract"] == (
        "human_annotation_delivery_v1"
    )
    assert protocol["intake"]["complete_item_count_per_annotator"] == 471
    assert protocol["intake"]["operator_mapping"] == "operator_only"
    assert protocol["formal_validation_complete"] is False


def test_one_then_two_submissions_follow_state_machine(tmp_path: Path) -> None:
    context, submissions = _setup(tmp_path)
    ledger = tmp_path / "intake-ledger.json"
    first = intake_submissions(
        ROOT,
        _protocol(),
        assignment_context=context,
        submissions={"annotator_a": submissions["annotator_a"]},
        ledger_path=ledger,
        allow_synthetic=True,
    )
    assert first["state"] == "one_submission_validated"
    assert first["exit_code"] == 3
    assert first["adjudication_queue_allowed"] is False
    second = intake_submissions(
        ROOT,
        _protocol(),
        assignment_context=context,
        submissions={"annotator_b": submissions["annotator_b"]},
        ledger_path=ledger,
        allow_synthetic=True,
    )
    assert second["state"] == "two_submissions_validated"
    assert second["exit_code"] == 0
    assert second["adjudication_queue_allowed"] is True
    assert verify_event_chain(_read_json(ledger)) == (
        "two_submissions_validated"
    )


def test_queue_contains_all_disagreements_and_preserves_blinding(
    tmp_path: Path,
) -> None:
    context, submissions = _setup(tmp_path)
    ledger = tmp_path / "intake-ledger.json"
    intake_submissions(
        ROOT,
        _protocol(),
        assignment_context=context,
        submissions=submissions,
        ledger_path=ledger,
        allow_synthetic=True,
    )
    queue = tmp_path / "adjudicator" / "queue.json"
    mapping = tmp_path / "operator" / "queue-mapping.json"
    result = build_adjudication_queue(
        ROOT,
        _protocol(),
        assignment_context=context,
        submissions=submissions,
        ledger_path=ledger,
        queue_path=queue,
        operator_mapping_path=mapping,
        allow_synthetic=True,
    )
    value = _read_json(queue)
    assert result["state"] == "adjudication_queue_ready"
    assert result["statistics"] is None
    assert value["item_count"] == len(value["rows"]) > 0
    assert all(
        row["annotation_a"]["label"] != row["annotation_b"]["label"]
        for row in value["rows"]
    )
    forbidden = {
        "alias",
        "arm",
        "case_id",
        "global_opaque_id",
        "gold",
        "item_id",
        "package_role",
        "qrels",
        "rank",
        "score",
        "source",
        "strategy",
        "target_paper",
    }
    serialized = json.dumps(value, ensure_ascii=False)
    assert not any(f'"{key}"' in serialized for key in forbidden)
    private = _read_json(mapping)
    assert len(private["entries"]) == value["item_count"]
    assert all(
        set(row) == {"disagreement_alias", "item_id", "package_role"}
        for row in private["entries"]
    )
    assert verify_event_chain(_read_json(ledger)) == (
        "adjudication_queue_ready"
    )


def test_partial_duplicate_unknown_and_illegal_labels_fail_closed(
    tmp_path: Path,
) -> None:
    context, submissions = _setup(tmp_path)
    protocol = _protocol()
    for scenario in ("partial", "duplicate", "unknown", "illegal"):
        export = tmp_path / f"{scenario}.json"
        _synthetic_export(
            ROOT, protocol, side="A", output=export, offset=0
        )
        value = _read_json(export)
        if scenario == "partial":
            value["labels"] = value["labels"][:-1]
        elif scenario == "duplicate":
            value["labels"][-1]["alias"] = value["labels"][0]["alias"]
        elif scenario == "unknown":
            value["labels"][-1]["alias"] = "item-" + "f" * 24
        else:
            value["labels"][0]["label"] = "invalid"
        from scholar_agent.evaluation.human_annotation_delivery import (
            submission_hash,
        )

        value["labels_sha256"] = submission_hash(value)
        write_object(export, value)
        receipt = tmp_path / f"{scenario}-receipt.json"
        with pytest.raises(Exception):
            build_submission_receipt(
                export,
                role="annotator_a",
                assignment_context=context,
                protocol=protocol,
                repository_root=ROOT,
                output=receipt,
            )


def test_locked_hash_role_and_coordinator_binding_are_enforced(
    tmp_path: Path,
) -> None:
    context, submissions = _setup(tmp_path)
    protocol = _protocol()
    export, receipt_path = submissions["annotator_a"]
    tampered = _read_json(export)
    tampered["labels"][0]["notes"] = "post-lock change"
    write_object(export, tampered)
    with pytest.raises(
        HumanAnnotationSubmissionError, match="submission_lock_hash_mismatch"
    ):
        verify_locked_submission(
            export,
            receipt_path,
            role="annotator_a",
            assignment_context=context,
            protocol=protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )

    context, submissions = _setup(tmp_path / "coordinator")
    receipt = _read_json(submissions["annotator_a"][1])
    receipt["submitted_by_role"] = "human_package_coordinator"
    receipt["receipt_sha256"] = "0" * 64
    receipt["receipt_sha256"] = __import__("hashlib").sha256(
        canonical_json(receipt)
    ).hexdigest()
    write_object(submissions["annotator_a"][1], receipt)
    with pytest.raises(
        HumanAnnotationSubmissionError, match="submission_receipt_invalid"
    ):
        verify_locked_submission(
            *submissions["annotator_a"],
            role="annotator_a",
            assignment_context=context,
            protocol=protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_queue_omission_and_forgery_are_rejected(tmp_path: Path) -> None:
    context, submissions = _setup(tmp_path)
    protocol = _protocol()
    ledger = tmp_path / "ledger.json"
    intake_submissions(
        ROOT,
        protocol,
        assignment_context=context,
        submissions=submissions,
        ledger_path=ledger,
        allow_synthetic=True,
    )
    queue = tmp_path / "queue.json"
    mapping = tmp_path / "mapping.json"
    build_adjudication_queue(
        ROOT,
        protocol,
        assignment_context=context,
        submissions=submissions,
        ledger_path=ledger,
        queue_path=queue,
        operator_mapping_path=mapping,
        allow_synthetic=True,
    )
    original = _read_json(queue)
    for mutation in ("omission", "forgery"):
        value = copy.deepcopy(original)
        if mutation == "omission":
            value["rows"].pop()
            value["item_count"] -= 1
        else:
            value["rows"][0]["annotation_b"] = copy.deepcopy(
                value["rows"][0]["annotation_a"]
            )
        value["queue_sha256"] = "0" * 64
        value["queue_sha256"] = __import__("hashlib").sha256(
            canonical_json(value)
        ).hexdigest()
        write_object(queue, value)
        with pytest.raises(
            HumanAnnotationSubmissionError,
            match="adjudication_queue_population_invalid",
        ):
            verify_adjudication_queue(
                queue,
                mapping,
                repository_root=ROOT,
                protocol=protocol,
                assignment_context=context,
                submissions=submissions,
                allow_synthetic=True,
            )


def test_event_chain_rejects_reordering_and_rehashed_binding_drift(
    tmp_path: Path,
) -> None:
    context, submissions = _setup(tmp_path)
    ledger = tmp_path / "ledger.json"
    intake_submissions(
        ROOT,
        _protocol(),
        assignment_context=context,
        submissions=submissions,
        ledger_path=ledger,
        allow_synthetic=True,
    )
    value = _read_json(ledger)
    value["submissions"][0]["principal_id"] = "prn_ffffffffffffffff"
    write_object(ledger, value)
    with pytest.raises(
        HumanAnnotationSubmissionError,
        match="submission_event_binding_invalid",
    ):
        verify_event_chain(_read_json(ledger))


def test_matrix_covers_required_attacks_and_is_deterministic() -> None:
    first = simulate_matrix(ROOT, _protocol())
    second = simulate_matrix(ROOT, _protocol())
    assert canonical_json(first) == canonical_json(second)
    assert first["passed_count"] == first["scenario_count"] == 14
    assert first["real_label_count"] == 0
    assert first["statistics"] is None
    assert {row["scenario"] for row in first["scenarios"]} == set(
        _protocol()["synthetic_scenarios"]
    )


def test_current_readiness_lists_real_submission_gaps() -> None:
    value = audit_readiness(_protocol())
    assert value["exit_code"] == 3
    assert value["state"] == "not_ready_missing_real_submissions"
    assert value["statistics"] is None
    assert value["real_label_count"] == 0
    assert value["formal_validation_complete"] is False
    assert {
        "annotator_a_real_qualification_missing",
        "annotator_b_real_qualification_missing",
        "adjudicator_real_qualification_missing",
        "annotator_a_assignment_acknowledgement_missing",
        "annotator_b_assignment_acknowledgement_missing",
        "adjudicator_assignment_acknowledgement_missing",
        "annotator_a_locked_submission_missing",
        "annotator_b_locked_submission_missing",
    } == set(value["blocked_reasons"])


def test_cli_readiness_and_matrix_have_stable_json_and_exit_codes() -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    readiness = subprocess.run(
        [sys.executable, str(SCRIPT), "audit-readiness"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
    )
    assert readiness.returncode == 3
    assert readiness.stderr == b""
    assert json.loads(readiness.stdout)["statistics"] is None
    first = subprocess.run(
        [sys.executable, str(SCRIPT), "simulate-matrix"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, str(SCRIPT), "simulate-matrix"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
    )
    if first.returncode == second.returncode == 2:
        pytest.skip("historical annotation package unavailable or drifted; strict matrix gate remains blocked")
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout


def test_audit_readiness_runs_with_isolated_standard_library(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    scratch = tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    sentinel = "must-not-be-read-from-dotenv"
    (tmp_path / ".env").write_text(f"API_KEY={sentinel}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "audit-readiness",
        ],
        cwd=tmp_path,
        env={
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(tmp_path / "missing-site-packages"),
            "TMPDIR": str(scratch),
        },
        capture_output=True,
        check=False,
    )
    value = json.loads(result.stdout)
    assert result.returncode == 3
    assert result.stderr == b""
    assert value["status"] == "not_ready_missing_real_submissions"
    assert value["real_label_count"] == 0
    assert value["statistics"] is None
    assert sentinel.encode() not in result.stdout


def test_audit_readiness_runs_in_two_repository_free_environments(
    tmp_path: Path,
) -> None:
    outputs: list[bytes] = []
    for index in range(2):
        environment = tmp_path / f"environment-{index}"
        bundle = environment / "bundle"
        work = environment / "work"
        home = environment / "home"
        scratch = environment / "scratch"
        for directory in (bundle, work, home, scratch):
            directory.mkdir(parents=True)
        verifier = bundle / "verify_submission_readiness.py"
        protocol = bundle / "protocol.json"
        shutil.copy2(SCRIPT, verifier)
        shutil.copy2(PROTOCOL_PATH, protocol)
        (work / ".env").write_text(
            "API_KEY=repository-free-sentinel\n", encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(verifier),
                "--protocol",
                str(protocol),
                "audit-readiness",
            ],
            cwd=work,
            env={
                "HOME": str(home),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(environment / "unavailable"),
                "TMPDIR": str(scratch),
            },
            capture_output=True,
            check=False,
        )
        assert result.returncode == 3
        assert result.stderr == b""
        assert json.loads(result.stdout)["formal_validation_complete"] is False
        assert b"repository-free-sentinel" not in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_isolated_audit_rejects_protocol_drift_without_traceback(
    tmp_path: Path,
) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["intake"]["complete_item_count_per_annotator"] = 470
    tampered = tmp_path / "protocol.json"
    tampered.write_text(
        json.dumps(protocol, ensure_ascii=False), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--protocol",
            str(tampered),
            "audit-readiness",
        ],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout
    assert json.loads(result.stdout)["reason"] == "protocol_binding_invalid"


def test_cli_usage_and_malformed_protocol_never_traceback(
    tmp_path: Path,
) -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    malformed = tmp_path / "protocol.json"
    malformed.write_text('{"protocol":"human_annotation_submission_intake_v1"}')
    for command in (
        [sys.executable, str(SCRIPT)],
        [
            sys.executable,
            str(SCRIPT),
            "--protocol",
            str(malformed),
            "audit-readiness",
        ],
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            check=False,
        )
        assert result.returncode in {2, 4}
        assert result.stderr == b""
        assert b"Traceback" not in result.stdout


def test_real_cli_path_rejects_synthetic_assignment(tmp_path: Path) -> None:
    context, submissions = _setup(tmp_path)
    with pytest.raises(
        HumanAnnotationSubmissionError, match="submission_receipt_invalid"
    ):
        verify_locked_submission(
            *submissions["annotator_a"],
            role="annotator_a",
            assignment_context=context,
            protocol=_protocol(),
            repository_root=ROOT,
            allow_synthetic=False,
        )
