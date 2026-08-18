from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.human_annotator_qualification_intake import (
    ROLES,
    HumanAnnotatorQualificationError,
    audit_readiness,
    build_kit,
    build_synthetic_submission,
    canonical_json,
    import_dry_run,
    load_protocol,
    sha256_bytes,
    simulate_matrix,
    verify_kit,
    verify_submission,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/human_annotator_qualification_intake_v1_protocol.json"
CLI = ROOT / "scripts/check_human_annotator_qualification.py"


@pytest.fixture
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _kits_and_submissions(
    tmp_path: Path,
    protocol: dict[str, object],
    *,
    challenge_prefix: str = "challenge",
    principal_offset: int = 0,
) -> tuple[dict[str, Path], dict[str, Path]]:
    kits: dict[str, Path] = {}
    submissions: dict[str, Path] = {}
    for ordinal, role in enumerate(ROLES, 1):
        kit = tmp_path / f"{role}.zip"
        build_kit(
            ROOT,
            protocol,
            challenge=sha256_bytes(f"{challenge_prefix}:{role}".encode()),
            role=role,
            output=kit,
        )
        contract = verify_kit(kit, protocol, repository_root=ROOT)
        submission = tmp_path / f"{role}.json"
        build_synthetic_submission(
            contract,
            submission,
            principal_id=f"prn_{ordinal + principal_offset:016x}",
            principal_commitment=sha256_bytes(
                f"principal:{principal_offset}:{role}".encode()
            ),
        )
        kits[role] = kit
        submissions[role] = submission
    return kits, submissions


def _inputs(
    kits: dict[str, Path], submissions: dict[str, Path]
) -> list[tuple[Path, Path]]:
    return [(kits[role], submissions[role]) for role in ROLES]


def test_real_readiness_is_blocked_without_labels(
    protocol: dict[str, object],
) -> None:
    report = audit_readiness(protocol)
    assert report["exit_code"] == 3
    assert report["blocked_reasons"] == [
        "annotator_a_real_qualification_missing",
        "annotator_b_real_qualification_missing",
        "adjudicator_real_qualification_missing",
    ]
    assert report["real_label_count"] == 0
    assert report["real_package_distributed"] is False
    assert report["human_precision_verified"] is False
    assert report["formal_validation_complete"] is False


def test_kit_is_deterministic_and_runs_in_two_no_repository_environments(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    challenge = sha256_bytes(b"no-repository-qualification")
    build_kit(
        ROOT,
        protocol,
        challenge=challenge,
        role="annotator_a",
        output=first,
    )
    build_kit(
        ROOT,
        protocol,
        challenge=challenge,
        role="annotator_a",
        output=second,
    )
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        structured = {
            name: json.loads(archive.read(name))
            for name in (
                "calibration_items.json",
                "contract.json",
                "submission_template.json",
            )
        }
    structured_keys = set()

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                structured_keys.add(key.lower())
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(structured)
    assert structured_keys.isdisjoint(
        {"case_id", "qrels", "target_paper", "private_mapping", "global_opaque_id"}
    )

    for ordinal in (1, 2):
        isolated = tmp_path / f"isolated-{ordinal}"
        isolated.mkdir()
        with zipfile.ZipFile(first) as archive:
            archive.extract("verify.py", isolated)
        env = {
            "HOME": str(isolated / "home"),
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": str(ordinal),
            "TMPDIR": str(isolated),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(isolated / "verify.py"),
                "--kit",
                str(first),
            ],
            cwd=isolated,
            env=env,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        assert json.loads(completed.stdout)["status"] == "qualification_kit_verified"


def test_valid_three_role_import_creates_append_only_proposal_only(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kits, submissions = _kits_and_submissions(tmp_path, protocol)
    proposal = tmp_path / "proposal.json"
    ledger = tmp_path / "ledger-a.json"
    first = import_dry_run(
        _inputs(kits, submissions),
        ledger,
        protocol,
        repository_root=ROOT,
        proposal_path=proposal,
        allow_synthetic=True,
    )
    second = import_dry_run(
        _inputs(kits, submissions),
        tmp_path / "ledger-b.json",
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    assert canonical_json(first) == canonical_json(second)
    value = json.loads(proposal.read_text())
    assert value["status"] == "ready_for_real_assignment"
    assert value["sequence"] == 1
    assert value["previous_sha256"] == "0" * 64
    assert value["real_package_distributed"] is False
    assert value["human_precision_verified"] is False
    assert first["real_package_distributed"] is False

    next_kits, next_submissions = _kits_and_submissions(
        tmp_path / "next",
        protocol,
        challenge_prefix="next-challenge",
        principal_offset=10,
    )
    next_proposal = tmp_path / "next-proposal.json"
    import_dry_run(
        _inputs(next_kits, next_submissions),
        ledger,
        protocol,
        repository_root=ROOT,
        proposal_path=next_proposal,
        allow_synthetic=True,
    )
    chained = json.loads(next_proposal.read_text())
    assert chained["sequence"] == 2
    assert chained["previous_sha256"] == value["proposal_sha256"]


def test_synthetic_matrix_closes_role_and_intake_attacks(
    protocol: dict[str, object],
) -> None:
    first = simulate_matrix(ROOT, protocol)
    second = simulate_matrix(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == first["passed_count"] == 12
    assert first["scenarios"][0]["observed"] == "passed"
    assert all(row["observed"] == "rejected" for row in first["scenarios"][1:])
    assert first["real_label_count"] == 0


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("coordinator_proxy", "coordinator_proxy_forbidden"),
        ("calibration_incomplete", "calibration_incomplete"),
        ("illegal_label", "calibration_label_invalid"),
        ("commit_drift", "qualification_binding_invalid"),
        ("revoked_reuse", "qualification_not_active"),
        ("expired_qualification", "qualification_binding_invalid"),
    ],
)
def test_submission_fail_closed_scenarios(
    tmp_path: Path,
    protocol: dict[str, object],
    scenario: str,
    reason: str,
) -> None:
    kit = tmp_path / "kit.zip"
    build_kit(
        ROOT,
        protocol,
        challenge=sha256_bytes(b"scenario"),
        role="annotator_a",
        output=kit,
    )
    contract = verify_kit(kit, protocol, repository_root=ROOT)
    submission = tmp_path / "submission.json"
    build_synthetic_submission(
        contract,
        submission,
        principal_id="prn_0000000000000001",
        principal_commitment=sha256_bytes(b"principal"),
        scenario=scenario,
    )
    with pytest.raises(HumanAnnotatorQualificationError, match=reason):
        verify_submission(
            kit,
            submission,
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_same_principal_and_alias_rebinding_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kits, submissions = _kits_and_submissions(tmp_path, protocol)
    a = json.loads(submissions["annotator_a"].read_text())
    b = json.loads(submissions["annotator_b"].read_text())
    b["principal_id"] = a["principal_id"]
    b["qualification_sha256"] = "0" * 64
    from scholar_agent.evaluation.human_annotator_qualification_intake import (
        _digest_without,
    )

    b["qualification_sha256"] = _digest_without(b, "qualification_sha256")
    submissions["annotator_b"].write_bytes(canonical_json(b))
    with pytest.raises(HumanAnnotatorQualificationError, match="principal_role_conflict"):
        import_dry_run(
            _inputs(kits, submissions),
            tmp_path / "same-ledger.json",
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )

    kits, submissions = _kits_and_submissions(tmp_path / "alias", protocol)
    a = json.loads(submissions["annotator_a"].read_text())
    b = json.loads(submissions["annotator_b"].read_text())
    b["principal_commitment"] = a["principal_commitment"]
    b["qualification_sha256"] = "0" * 64
    b["qualification_sha256"] = _digest_without(b, "qualification_sha256")
    submissions["annotator_b"].write_bytes(canonical_json(b))
    with pytest.raises(
        HumanAnnotatorQualificationError, match="principal_alias_rebinding"
    ):
        import_dry_run(
            _inputs(kits, submissions),
            tmp_path / "alias-ledger.json",
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_challenge_replay_and_qualification_tamper_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kits, submissions = _kits_and_submissions(tmp_path, protocol)
    ledger = tmp_path / "ledger.json"
    import_dry_run(
        _inputs(kits, submissions),
        ledger,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    with pytest.raises(HumanAnnotatorQualificationError, match="challenge_replay"):
        import_dry_run(
            _inputs(kits, submissions),
            ledger,
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )

    value = json.loads(submissions["annotator_a"].read_text())
    value["rubric_acknowledged"] = False
    submissions["annotator_a"].write_bytes(canonical_json(value))
    with pytest.raises(
        HumanAnnotatorQualificationError, match="qualification_binding_invalid"
    ):
        verify_submission(
            kits["annotator_a"],
            submissions["annotator_a"],
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_real_mode_rejects_synthetic_principals(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kits, submissions = _kits_and_submissions(tmp_path, protocol)
    with pytest.raises(
        HumanAnnotatorQualificationError, match="synthetic_principal_not_real"
    ):
        verify_submission(
            kits["annotator_a"],
            submissions["annotator_a"],
            protocol,
            repository_root=ROOT,
        )


def test_cli_has_stable_exit_codes_and_no_traceback(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    matrix = subprocess.run(
        [sys.executable, str(CLI), "simulate-matrix"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )
    assert matrix.returncode == 0
    assert matrix.stderr == b""
    assert json.loads(matrix.stdout)["scenario_count"] == 12

    readiness = subprocess.run(
        [sys.executable, str(CLI), "audit-readiness"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )
    assert readiness.returncode == 3
    assert readiness.stderr == b""
    assert json.loads(readiness.stdout)["real_label_count"] == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    usage = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--protocol",
            str(malformed),
            "audit-readiness",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )
    assert usage.returncode == 2
    assert usage.stderr == b""
    assert b"Traceback" not in usage.stdout
