from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_validation_roles import (
    ACTIONS,
    ENTRYPOINT_ACTIONS,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_VIOLATION,
    PROTOCOL,
    SOURCE_COMMIT,
    ZERO_SHA256,
    RoleControlError,
    RoleControlNotReady,
    build_synthetic_ceremony,
    canonical_json,
    load_protocol,
    make_authorization,
    opaque_identity,
    simulate_matrix,
    stable_hash,
    synthetic_assignments,
    verify_assignments,
    verify_ceremony,
    verify_authorization_chain,
    verify_entry_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_formal_validation_roles.py"
PROTOCOL_PATH = (
    ROOT / "benchmark/formal_validation_separation_of_duties_v1_protocol.json"
)
ASSIGNMENTS_PATH = (
    ROOT / "benchmark/formal_validation_separation_of_duties_v1_assignments.json"
)


def _run(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )


def _rechain_authorizations(rows: list[dict[str, object]]) -> None:
    previous = ZERO_SHA256
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
        row["previous_sha256"] = previous
        payload = dict(row)
        payload["content_sha256"] = ZERO_SHA256
        row["content_sha256"] = stable_hash(payload)
        previous = str(row["content_sha256"])


def test_protocol_and_current_missing_assignment_are_fail_closed() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["protocol"] == PROTOCOL
    result = _run("audit-readiness")
    assert result.returncode == EXIT_NOT_READY
    assert result.stderr == b""
    report = json.loads(result.stdout)
    assert report["real_role_assignment_count"] == 0
    assert report["formal_blockers"] == [
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    ]


def test_synthetic_ceremony_and_report_are_deterministic() -> None:
    first = simulate_matrix()
    second = simulate_matrix()
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == 11
    assert all(
        row["status"] == ("passed" if row["scenario"].startswith("legal_") else "rejected")
        for row in first["scenarios"]
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("annotator_a", "annotator_b"),
        ("annotator_a", "adjudicator"),
        ("annotator_b", "adjudicator"),
        ("execution_operator", "launch_authorizer"),
        ("execution_operator", "clearance_approver"),
        ("release_signer", "revocation_administrator"),
    ],
)
def test_forbidden_role_combinations(first: str, second: str) -> None:
    assignments = synthetic_assignments()
    assignments["assignments"][second] = assignments["assignments"][first]
    with pytest.raises(RoleControlError):
        verify_assignments(assignments)


def test_alias_rebinding_cannot_hide_indirect_identity_reuse() -> None:
    assignments = synthetic_assignments()
    identity = assignments["assignments"]["annotator_a"]
    assignments["aliases"]["alias_indirect"] = {
        "opaque_id": identity,
        "principal_binding_sha256": "a" * 64,
    }
    with pytest.raises(RoleControlError, match="alias_principal_rebinding"):
        verify_assignments(assignments)


def test_cross_commit_tampering_and_duplicate_approval_are_rejected() -> None:
    assignments, authorizations, events = build_synthetic_ceremony()
    tampered = copy.deepcopy(authorizations)
    tampered[0]["code_commit"] = "b" * 40
    _rechain_authorizations(tampered)
    with pytest.raises(RoleControlError):
        verify_ceremony(
            assignments=assignments,
            authorizations=tampered,
            events=events,
        )
    duplicated = copy.deepcopy(events)
    duplicated.append(copy.deepcopy(events[-1]))
    with pytest.raises(RoleControlError):
        verify_ceremony(
            assignments=assignments,
            authorizations=authorizations,
            events=duplicated,
        )


def test_revoked_authorization_and_formal_synthetic_use_are_rejected() -> None:
    assignments = synthetic_assignments()
    actor = assignments["assignments"]["launch_authorizer"]
    alias = next(
        key
        for key, value in assignments["aliases"].items()
        if value["opaque_id"] == actor
    )
    artifact = stable_hash({"artifact": "launch"})
    revoked = make_authorization(
        sequence=0,
        previous_sha256=ZERO_SHA256,
        action="authorize_launch",
        actor_opaque_id=actor,
        alias=alias,
        artifact_type="full1000_launch_control",
        artifact_sha256=artifact,
        state="revoked",
    )
    with pytest.raises(RoleControlError):
        verify_entry_authorization(
            entrypoint="full1000_launch_control",
            action="authorize_launch",
            artifact_sha256=artifact,
            authorization=revoked,
            assignments=assignments,
            formal_mode=False,
        )
    active = make_authorization(
        sequence=0,
        previous_sha256=ZERO_SHA256,
        action="authorize_launch",
        actor_opaque_id=actor,
        alias=alias,
        artifact_type="full1000_launch_control",
        artifact_sha256=artifact,
    )
    with pytest.raises(RoleControlNotReady):
        verify_entry_authorization(
            entrypoint="full1000_launch_control",
            action="authorize_launch",
            artifact_sha256=artifact,
            authorization=active,
            assignments=assignments,
            formal_mode=True,
        )


def test_revocation_and_rotation_require_an_active_prior_authorization() -> None:
    assignments = synthetic_assignments()
    actor = assignments["assignments"]["release_signer"]
    alias = next(
        key
        for key, value in assignments["aliases"].items()
        if value["opaque_id"] == actor
    )
    artifact = stable_hash({"artifact": "release"})
    active = make_authorization(
        sequence=0,
        previous_sha256=ZERO_SHA256,
        action="sign_release",
        actor_opaque_id=actor,
        alias=alias,
        artifact_type="release_authenticity_signing",
        artifact_sha256=artifact,
    )
    revoked = make_authorization(
        sequence=1,
        previous_sha256=active["content_sha256"],
        action="sign_release",
        actor_opaque_id=actor,
        alias=alias,
        artifact_type="release_authenticity_signing",
        artifact_sha256=artifact,
        state="revoked",
    )
    report = verify_authorization_chain([active, revoked], assignments)
    assert report["active_action_count"] == 0
    revoked_without_prior = make_authorization(
        sequence=0,
        previous_sha256=ZERO_SHA256,
        action="sign_release",
        actor_opaque_id=actor,
        alias=alias,
        artifact_type="release_authenticity_signing",
        artifact_sha256=artifact,
        state="revoked",
    )
    with pytest.raises(RoleControlError, match="authorization_state_transition_invalid"):
        verify_authorization_chain([revoked_without_prior], assignments)


def test_real_assignment_requires_no_stored_identity_mapping() -> None:
    assignments = synthetic_assignments()
    assignments["test_only"] = False
    assignments["status"] = "active_real_assignments"
    verify_assignments(assignments, require_real=True)
    serialized = canonical_json(assignments).decode("utf-8")
    for forbidden in ("email", "username", "hostname", "person_name"):
        assert forbidden not in serialized


def test_every_bound_formal_entrypoint_requires_exact_action_and_artifact() -> None:
    assignments = synthetic_assignments()
    for entrypoint, actions in ENTRYPOINT_ACTIONS.items():
        for action in actions:
            role = ACTIONS[action]
            actor = assignments["assignments"][role]
            alias = next(
                key
                for key, value in assignments["aliases"].items()
                if value["opaque_id"] == actor
            )
            artifact = stable_hash({"entrypoint": entrypoint, "action": action})
            authorization = make_authorization(
                sequence=0,
                previous_sha256=ZERO_SHA256,
                action=action,
                actor_opaque_id=actor,
                alias=alias,
                artifact_type=entrypoint,
                artifact_sha256=artifact,
            )
            verify_entry_authorization(
                entrypoint=entrypoint,
                action=action,
                artifact_sha256=artifact,
                authorization=authorization,
                assignments=assignments,
                formal_mode=False,
            )
            with pytest.raises(RoleControlError):
                verify_entry_authorization(
                    entrypoint=entrypoint,
                    action=action,
                    artifact_sha256="f" * 64,
                    authorization=authorization,
                    assignments=assignments,
                    formal_mode=False,
                )


def test_legal_personnel_rotation_uses_new_principal_binding() -> None:
    assignments = synthetic_assignments()
    role = "human_package_coordinator"
    previous = assignments["assignments"][role]
    replacement = opaque_identity("replacement")
    assignments["assignments"][role] = replacement
    alias = next(
        key
        for key, value in assignments["aliases"].items()
        if value["opaque_id"] == previous
    )
    assignments["aliases"][alias] = {
        "opaque_id": replacement,
        "principal_binding_sha256": stable_hash({"principal": "replacement"}),
    }
    verify_assignments(assignments)


def test_protocol_policy_drift_is_rejected_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["minimum_independence"]["distinct_principals"] = 1
    payload = copy.deepcopy(protocol)
    payload["protocol_sha256"] = ZERO_SHA256
    protocol["protocol_sha256"] = stable_hash(payload)
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json(protocol))
    with pytest.raises(RoleControlError, match="protocol_schema_invalid"):
        load_protocol(path)


def test_cli_policy_simulation_and_usage_contract() -> None:
    first = _run("simulate-ceremony")
    second = _run("simulate-ceremony")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert _run("verify-policy").returncode == EXIT_READY
    usage = _run("verify-authorization")
    assert usage.returncode == 4
    assert usage.stderr == b""


def test_cli_malformed_assignment_has_stable_exit_two(tmp_path: Path) -> None:
    malformed = tmp_path / "assignments.json"
    malformed.write_text('{"assignments":{},"assignments":{}}', encoding="utf-8")
    result = _run("audit-readiness", "--assignments", str(malformed))
    assert result.returncode == EXIT_VIOLATION
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout


def test_tracked_assignment_contains_no_real_principal_data() -> None:
    value = json.loads(ASSIGNMENTS_PATH.read_text(encoding="utf-8"))
    assert value["assignments"] == {}
    assert value["aliases"] == {}
    assert value["real_identity_mapping"] == "not_available"
    assert value["formal_validation_complete"] is False
    assert SOURCE_COMMIT in ASSIGNMENTS_PATH.read_text(encoding="utf-8")
