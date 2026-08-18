from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.evidence_transparency_log import (
    REAL_BLOCKERS,
    TransparencyError,
    append_record,
    build_checkpoint,
    build_current,
    build_log,
    consistency_proof,
    finalize_record,
    inclusion_proof,
    load_protocol,
    parse_json_bytes,
    simulate_matrix,
    synthetic_record,
    verify_checkpoint,
    verify_consistency_proof,
    verify_inclusion_proof,
    verify_log,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/evidence_transparency_log_v1_protocol.json"
SCRIPT = ROOT / "scripts/check_evidence_transparency.py"


def _protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL)


def _base_log() -> dict[str, object]:
    return build_log(
        [
            synthetic_record(
                sequence=0,
                previous="0" * 64,
                release_identity="candidate:0",
                digest_seed="a",
                code_commit="1" * 40,
            )
        ]
    )


def _append(
    log: dict[str, object],
    *,
    identity: str,
    digest_seed: str,
    commit: str,
    supersedes: str | None = None,
) -> dict[str, object]:
    record = synthetic_record(
        sequence=len(log["records"]),
        previous=log["records"][-1]["content_sha256"],
        release_identity=identity,
        digest_seed=digest_seed,
        code_commit=commit,
        supersedes=supersedes,
    )
    return append_record(log, record)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    env = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_current_candidate_is_deterministic_and_not_public() -> None:
    first = build_current(ROOT, _protocol())
    second = build_current(ROOT, _protocol())
    assert first == second
    log, checkpoint = first
    report = verify_log(log)
    verify_checkpoint(checkpoint, log)
    assert report["record_count"] == 1
    assert report["public_release_count"] == 0
    assert report["latest_release_status"] == "candidate_only"
    assert report["formal_validation_complete"] is False
    assert checkpoint["status"] == "candidate_checkpoint_no_public_release"
    assert checkpoint["blockers"] == list(REAL_BLOCKERS)


def test_merkle_inclusion_and_prefix_consistency() -> None:
    one = _base_log()
    two = _append(
        one, identity="candidate:1", digest_seed="b", commit="2" * 40
    )
    three = _append(
        two, identity="candidate:2", digest_seed="c", commit="3" * 40
    )
    for sequence in range(3):
        verify_inclusion_proof(inclusion_proof(three, sequence))
    proof = consistency_proof(one, three)
    verify_consistency_proof(proof)
    assert proof["old_size"] == 1
    assert proof["new_size"] == 3


def test_deletion_reorder_rewrite_and_fork_are_rejected() -> None:
    one = _base_log()
    two = _append(
        one, identity="candidate:1", digest_seed="b", commit="2" * 40
    )
    with pytest.raises(TransparencyError, match="immutable_prefix"):
        consistency_proof(two, one)

    reordered = copy.deepcopy(two)
    reordered["records"].reverse()
    with pytest.raises(TransparencyError):
        build_log(reordered["records"])

    rewritten = copy.deepcopy(two)
    rewritten["records"][0]["claims"]["claim_count"] = 4
    with pytest.raises(TransparencyError):
        build_log(rewritten["records"])

    branch = _append(
        one, identity="candidate:fork", digest_seed="d", commit="4" * 40
    )
    with pytest.raises(TransparencyError, match="immutable_prefix"):
        consistency_proof(two, branch)


def test_same_identity_rollback_and_unsourced_supersession_are_rejected() -> None:
    one = _base_log()
    duplicate = synthetic_record(
        sequence=1,
        previous=one["records"][-1]["content_sha256"],
        release_identity="candidate:0",
        digest_seed="b",
        code_commit="2" * 40,
    )
    with pytest.raises(TransparencyError, match="record_identity"):
        append_record(one, duplicate)

    two = _append(
        one, identity="candidate:1", digest_seed="b", commit="2" * 40
    )
    rollback = synthetic_record(
        sequence=2,
        previous=two["records"][-1]["content_sha256"],
        release_identity="candidate:rollback",
        digest_seed="a",
        code_commit="1" * 40,
    )
    with pytest.raises(TransparencyError, match="rollback"):
        append_record(two, rollback)

    unsourced = synthetic_record(
        sequence=1,
        previous=one["records"][-1]["content_sha256"],
        release_identity="candidate:replacement",
        digest_seed="e",
        code_commit="5" * 40,
        supersedes="candidate:0",
    )
    unsourced["supersession"]["revocation_event_sha256"] = None
    unsourced = finalize_record(unsourced)
    with pytest.raises(TransparencyError, match="supersession"):
        append_record(one, unsourced)


def test_blocker_freshness_incident_and_receipt_guards() -> None:
    one = _base_log()
    base = synthetic_record(
        sequence=1,
        previous=one["records"][-1]["content_sha256"],
        release_identity="candidate:bad",
        digest_seed="b",
        code_commit="2" * 40,
    )
    for mutate, reason in (
        (lambda row: row.__setitem__("blockers", []), "blocker"),
        (
            lambda row: row["freshness"].update(
                {"status": "stale", "stale_count": 1}
            ),
            "stale",
        ),
        (
            lambda row: row["revocation"].update(
                {"active_incident_count": 1}
            ),
            "revoked",
        ),
    ):
        candidate = copy.deepcopy(base)
        mutate(candidate)
        candidate = finalize_record(candidate)
        with pytest.raises(TransparencyError, match=reason):
            append_record(one, candidate)

    formal = synthetic_record(
        sequence=1,
        previous=one["records"][-1]["content_sha256"],
        release_identity="formal:bad",
        digest_seed="c",
        code_commit="3" * 40,
    )
    formal["release_status"] = "formal_release"
    formal["formal_validation_complete"] = True
    formal["blockers"] = []
    for role in ("readiness", "release_candidate", "standalone"):
        formal["artifacts"][role]["formal_validation_complete"] = True
    formal["artifacts"]["clearance_receipt"] = {
        "evidence_sha256": "d" * 64,
        "formal_validation_complete": True,
        "sha256": "e" * 64,
        "source_commit": "4" * 40,
        "status": "cleared",
    }
    formal = finalize_record(formal)
    with pytest.raises(TransparencyError, match="receipt"):
        append_record(one, formal)


def test_legal_revocation_supersession_keeps_history() -> None:
    one = _base_log()
    two = _append(
        one,
        identity="candidate:replacement",
        digest_seed="b",
        commit="2" * 40,
        supersedes="candidate:0",
    )
    report = verify_log(two)
    assert report["record_count"] == 2
    assert two["records"][0]["release_identity"] == "candidate:0"
    assert (
        two["records"][1]["supersession"]["supersedes_release_identity"]
        == "candidate:0"
    )


def test_json_input_is_fail_closed() -> None:
    with pytest.raises(TransparencyError, match="duplicate"):
        parse_json_bytes(b'{"value":1,"value":2}')
    with pytest.raises(TransparencyError, match="nonfinite"):
        parse_json_bytes(b'{"value":NaN}')
    with pytest.raises(TransparencyError, match="invalid"):
        parse_json_bytes(b"\xff")


def test_synthetic_matrix_covers_release_and_attack_paths() -> None:
    report = simulate_matrix()
    assert report["scenario_count"] == 12
    assert report["accepted_scenario_count"] == 4
    assert report["rejected_scenario_count"] == 8
    assert report["synthetic_only"] is True
    assert report["formal_validation_complete"] is False


def test_cli_build_verify_proofs_and_real_readiness(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    checkpoint = tmp_path / "checkpoint.json"
    built = _run(
        "build-log",
        "--output",
        str(log),
        "--checkpoint-output",
        str(checkpoint),
    )
    assert built.returncode == 0
    assert built.stderr == ""

    first = _run(
        "verify-log", "--log", str(log), "--checkpoint", str(checkpoint)
    )
    second = _run(
        "verify-log", "--log", str(log), "--checkpoint", str(checkpoint)
    )
    assert first.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""

    inclusion = _run(
        "prove-inclusion", "--log", str(log), "--sequence", "0"
    )
    assert inclusion.returncode == 0
    consistency = _run(
        "prove-consistency",
        "--old-log",
        str(log),
        "--new-log",
        str(log),
    )
    assert consistency.returncode == 0

    audit = _run(
        "audit-readiness", "--log", str(log), "--checkpoint", str(checkpoint)
    )
    assert audit.returncode == 3
    assert json.loads(audit.stdout)["status"] == "no_public_release_checkpoint"
    assert audit.stderr == ""


def test_cli_invalid_input_never_returns_exit_one(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"records":"not-a-list"}', encoding="utf-8")
    result = _run(
        "verify-log",
        "--log",
        str(malformed),
        "--checkpoint",
        str(malformed),
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert "Traceback" not in result.stdout
    assert json.loads(result.stdout)["reason_code"]

    usage = _run("prove-inclusion")
    assert usage.returncode == 4
    assert usage.stderr == ""
    assert json.loads(usage.stdout)["status"] == "usage_error"


def test_checkpoint_detects_log_or_checkpoint_tamper() -> None:
    log = _base_log()
    checkpoint = build_checkpoint(log)
    tampered = copy.deepcopy(checkpoint)
    tampered["blockers"] = []
    with pytest.raises(TransparencyError):
        verify_checkpoint(tampered, log)
    changed = _append(
        log, identity="candidate:1", digest_seed="b", commit="2" * 40
    )
    with pytest.raises(TransparencyError):
        verify_checkpoint(checkpoint, changed)
