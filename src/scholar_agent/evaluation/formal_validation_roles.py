"""Deterministic separation-of-duties controls for formal validation.

Only opaque principal identities and role bindings are accepted.  Real-world
identity mapping is an external governance input and is never stored here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL = "formal_validation_separation_of_duties_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "0f3b987543b0aec7db9ced25e1047220dd441862"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ZERO_SHA256 = "0" * 64
HEX = frozenset("0123456789abcdef")
OPAQUE_ID = re.compile(r"^opq_[0-9a-f]{16}$")
REAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
ROLES = (
    "plan_sealer",
    "launch_authorizer",
    "execution_operator",
    "human_package_coordinator",
    "annotator_a",
    "annotator_b",
    "adjudicator",
    "scorer_recipient",
    "evidence_auditor",
    "clearance_approver",
    "release_signer",
    "revocation_administrator",
)
ROLE_DOMAINS = {
    "plan_sealer": "governance",
    "launch_authorizer": "governance",
    "execution_operator": "operations",
    "human_package_coordinator": "operations",
    "annotator_a": "independent_evaluation",
    "annotator_b": "independent_evaluation",
    "adjudicator": "independent_evaluation",
    "scorer_recipient": "external_evidence_intake",
    "evidence_auditor": "assurance",
    "clearance_approver": "clearance_governance",
    "release_signer": "publication",
    "revocation_administrator": "incident_response",
}
ACTIONS = {
    "seal_plan": "plan_sealer",
    "authorize_launch": "launch_authorizer",
    "execute_run": "execution_operator",
    "coordinate_human_package": "human_package_coordinator",
    "annotate_a": "annotator_a",
    "annotate_b": "annotator_b",
    "adjudicate": "adjudicator",
    "receive_scorer": "scorer_recipient",
    "audit_evidence": "evidence_auditor",
    "approve_clearance": "clearance_approver",
    "sign_release": "release_signer",
    "approve_revocation": "revocation_administrator",
}
ENTRYPOINT_ACTIONS = {
    "formal_validation_preregistration": ("seal_plan",),
    "full1000_launch_control": ("authorize_launch", "execute_run"),
    "human_precision_adjudication": (
        "coordinate_human_package",
        "annotate_a",
        "annotate_b",
        "adjudicate",
    ),
    "external_scorer_handoff": ("receive_scorer",),
    "formal_evidence_quarantine": ("audit_evidence",),
    "formal_validation_clearance": ("approve_clearance",),
    "evidence_revocation": ("approve_revocation",),
    "release_authenticity_signing": ("sign_release",),
}
BINDINGS = {
    "clearance": "formal_validation_clearance_v1",
    "external_scorer": "external_scorer_handoff_v1",
    "human_adjudication": "human_precision_adjudication_v1",
    "launch": "full1000_launch_control_v1",
    "preregistration": "formal_validation_preregistration_v1",
    "quarantine": "formal_evidence_quarantine_v1",
    "release_signing": "release_authenticity_signing_v1",
    "revocation": "evidence_revocation_response_v1",
    "standalone_audit": "standalone_auditor_bundle_v1",
    "validation_readiness": "validation_readiness_bundle_v1",
}
FORBIDDEN_PAIRS = (
    ("annotator_a", "annotator_b"),
    ("annotator_a", "adjudicator"),
    ("annotator_b", "adjudicator"),
    ("execution_operator", "launch_authorizer"),
    ("execution_operator", "clearance_approver"),
    ("release_signer", "revocation_administrator"),
)
AUTHORIZATION_STATES = ("active", "rotated", "revoked")
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
AUTHORIZATION_POLICY = {
    "alias_rebinding": "forbidden",
    "append_only_hash_chain": True,
    "cross_commit_reuse": "forbidden",
    "posthoc_signature": "forbidden",
    "revoked_authorization_use": "forbidden",
    "wildcard_identity": "forbidden",
}
IDENTITY_POLICY = {
    "allowed": "stable_opaque_id_and_role_binding_only",
    "forbidden": [
        "credential",
        "email",
        "hostname",
        "operating_system_account",
        "person_name",
        "username",
    ],
    "real_identity_mapping": "not_available",
}
MINIMUM_INDEPENDENCE = {
    "clearance_approval_domains": 2,
    "distinct_principals": 8,
    "formal_clearance_requires_independent_auditor_and_approver": True,
}
PROTOCOL_KEYS = {
    "actions",
    "authorization",
    "bindings",
    "entrypoints",
    "execution",
    "formal_validation_complete",
    "forbidden_combinations",
    "identity",
    "minimum_independence",
    "protocol",
    "protocol_sha256",
    "real_blockers",
    "role_domains",
    "roles",
    "schema_version",
    "source_commit",
}
ASSIGNMENT_KEYS = {
    "aliases",
    "assignments",
    "formal_validation_complete",
    "protocol",
    "real_identity_mapping",
    "schema_version",
    "source_commit",
    "status",
    "test_only",
}
AUTHORIZATION_KEYS = {
    "action",
    "actor_opaque_id",
    "alias",
    "artifact_sha256",
    "artifact_type",
    "authorization_id",
    "code_commit",
    "content_sha256",
    "previous_sha256",
    "protocol",
    "role",
    "sequence",
    "state",
    "test_only",
}
EVENT_KEYS = {
    "action",
    "actor_opaque_id",
    "artifact_sha256",
    "authorization_id",
    "content_sha256",
    "event_id",
    "previous_sha256",
    "sequence",
    "test_only",
}


class RoleControlError(RuntimeError):
    """Role assignment, authorization, or ceremony integrity is invalid."""


class RoleControlNotReady(RoleControlError):
    """No complete real-world role assignment has been provisioned."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise RoleControlError("duplicate_json_key")
        value[key] = child
    return value


def _depth(value: Any, level: int = 0) -> None:
    if level > 48:
        raise RoleControlError("json_nesting_limit")
    if isinstance(value, Mapping):
        if len(value) > 4096:
            raise RoleControlError("json_member_limit")
        for child in value.values():
            _depth(child, level + 1)
    elif isinstance(value, list):
        if len(value) > 10000:
            raise RoleControlError("json_member_limit")
        for child in value:
            _depth(child, level + 1)


def parse_json_value(value: bytes) -> Any:
    if len(value) > 8 * 1024 * 1024:
        raise RoleControlError("json_size_limit")
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                RoleControlError("nonfinite_json_number")
            ),
        )
    except RoleControlError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise RoleControlError("json_input_invalid") from exc
    _depth(parsed)
    return parsed


def parse_json_bytes(value: bytes) -> dict[str, Any]:
    parsed = parse_json_value(value)
    if not isinstance(parsed, dict):
        raise RoleControlError("json_root_not_object")
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes())
    except OSError as exc:
        raise RoleControlError("json_input_unavailable") from exc


def read_json_sequence(path: Path) -> list[Any]:
    try:
        value = parse_json_value(path.read_bytes())
    except OSError as exc:
        raise RoleControlError("json_input_unavailable") from exc
    if not isinstance(value, list):
        raise RoleControlError("json_root_not_array")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise RoleControlError("json_output_unavailable") from exc


def _require_keys(value: Any, keys: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RoleControlError(reason)
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _protocol_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    return payload


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require_keys(value, PROTOCOL_KEYS, "protocol_schema_invalid")
    expected_pairs = [list(pair) for pair in FORBIDDEN_PAIRS]
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["roles"] != list(ROLES)
        or value["role_domains"] != ROLE_DOMAINS
        or value["actions"] != ACTIONS
        or value["authorization"] != AUTHORIZATION_POLICY
        or value["bindings"] != BINDINGS
        or value["entrypoints"] != {
            key: list(actions) for key, actions in ENTRYPOINT_ACTIONS.items()
        }
        or value["forbidden_combinations"] != expected_pairs
        or value["real_blockers"] != list(REAL_BLOCKERS)
        or value["execution"] != EXECUTION
        or value["identity"] != IDENTITY_POLICY
        or value["minimum_independence"] != MINIMUM_INDEPENDENCE
        or value["formal_validation_complete"] is not False
        or not _is_digest(value["protocol_sha256"])
        or stable_hash(_protocol_payload(value)) != value["protocol_sha256"]
    ):
        raise RoleControlError("protocol_schema_invalid")
    return dict(value)


def opaque_identity(seed: str) -> str:
    return "opq_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def synthetic_assignments() -> dict[str, Any]:
    assignments = {
        role: opaque_identity(f"synthetic:{role}") for role in ROLES
    }
    aliases = {
        f"alias_{index:02d}": {
            "opaque_id": assignments[role],
            "principal_binding_sha256": stable_hash(
                {"synthetic_principal": role}
            ),
        }
        for index, role in enumerate(ROLES)
    }
    return {
        "aliases": aliases,
        "assignments": assignments,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "real_identity_mapping": "not_available",
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "status": "synthetic_assignments_only",
        "test_only": True,
    }


def verify_assignments(value: Mapping[str, Any], *, require_real: bool = False) -> None:
    _require_keys(value, ASSIGNMENT_KEYS, "assignment_schema_invalid")
    assignments = value.get("assignments")
    aliases = value.get("aliases")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["real_identity_mapping"] != "not_available"
        or not isinstance(value["test_only"], bool)
        or not isinstance(assignments, Mapping)
        or not isinstance(aliases, Mapping)
        or set(assignments) != set(ROLES)
    ):
        raise RoleControlError("assignment_schema_invalid")
    for role, identity in assignments.items():
        if role not in ROLES or not isinstance(identity, str) or not OPAQUE_ID.fullmatch(identity):
            raise RoleControlError("opaque_identity_invalid")
    alias_principals: dict[str, str] = {}
    identity_principals: dict[str, str] = {}
    for alias, row in aliases.items():
        if (
            not isinstance(alias, str)
            or not alias.startswith("alias_")
            or not isinstance(row, Mapping)
            or set(row) != {"opaque_id", "principal_binding_sha256"}
            or row["opaque_id"] not in assignments.values()
            or not _is_digest(row["principal_binding_sha256"])
        ):
            raise RoleControlError("alias_binding_invalid")
        if alias in alias_principals:
            raise RoleControlError("alias_duplicate")
        previous = identity_principals.get(row["opaque_id"])
        if previous is not None and previous != row["principal_binding_sha256"]:
            raise RoleControlError("alias_principal_rebinding")
        alias_principals[alias] = row["principal_binding_sha256"]
        identity_principals[row["opaque_id"]] = row["principal_binding_sha256"]
    if set(identity_principals) != set(assignments.values()):
        raise RoleControlError("alias_coverage_invalid")
    for first, second in FORBIDDEN_PAIRS:
        if assignments[first] == assignments[second]:
            raise RoleControlError(f"forbidden_role_combination:{first}:{second}")
    if len({assignments[role] for role in ROLES}) < MINIMUM_INDEPENDENCE[
        "distinct_principals"
    ]:
        raise RoleControlError("minimum_distinct_principals_not_met")
    if require_real and value["test_only"]:
        raise RoleControlNotReady("missing_real_role_assignments")


def _authorization_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["content_sha256"] = ZERO_SHA256
    return payload


def make_authorization(
    *,
    sequence: int,
    previous_sha256: str,
    action: str,
    actor_opaque_id: str,
    alias: str,
    artifact_type: str,
    artifact_sha256: str,
    code_commit: str = SOURCE_COMMIT,
    state: str = "active",
    test_only: bool = True,
) -> dict[str, Any]:
    role = ACTIONS.get(action)
    value = {
        "action": action,
        "actor_opaque_id": actor_opaque_id,
        "alias": alias,
        "artifact_sha256": artifact_sha256,
        "artifact_type": artifact_type,
        "authorization_id": stable_hash(
            {
                "action": action,
                "actor": actor_opaque_id,
                "artifact": artifact_sha256,
                "sequence": sequence,
            }
        )[:24],
        "code_commit": code_commit,
        "content_sha256": ZERO_SHA256,
        "previous_sha256": previous_sha256,
        "protocol": PROTOCOL,
        "role": role,
        "sequence": sequence,
        "state": state,
        "test_only": test_only,
    }
    value["content_sha256"] = stable_hash(_authorization_payload(value))
    return value


def verify_authorization_chain(
    authorizations: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Any],
) -> dict[str, Any]:
    verify_assignments(assignments)
    aliases = assignments["aliases"]
    previous = ZERO_SHA256
    seen_ids: set[str] = set()
    active_actions: set[str] = set()
    action_states: dict[str, str] = {}
    for sequence, row in enumerate(authorizations):
        _require_keys(row, AUTHORIZATION_KEYS, "authorization_schema_invalid")
        if (
            row["sequence"] != sequence
            or row["previous_sha256"] != previous
            or row["authorization_id"] in seen_ids
            or row["action"] not in ACTIONS
            or row["role"] != ACTIONS[row["action"]]
            or row["actor_opaque_id"]
            != assignments["assignments"][row["role"]]
            or row["alias"] not in aliases
            or aliases[row["alias"]]["opaque_id"] != row["actor_opaque_id"]
            or not _is_digest(row["artifact_sha256"])
            or not _is_commit(row["code_commit"])
            or row["code_commit"] != SOURCE_COMMIT
            or row["protocol"] != PROTOCOL
            or row["state"] not in AUTHORIZATION_STATES
            or row["test_only"] != assignments["test_only"]
            or not _is_digest(row["content_sha256"])
            or stable_hash(_authorization_payload(row)) != row["content_sha256"]
        ):
            raise RoleControlError("authorization_chain_invalid")
        if row["state"] == "active":
            if action_states.get(row["action"]) == "active":
                raise RoleControlError("duplicate_active_authorization")
            action_states[row["action"]] = "active"
            active_actions.add(row["action"])
        else:
            if action_states.get(row["action"]) != "active":
                raise RoleControlError("authorization_state_transition_invalid")
            action_states[row["action"]] = row["state"]
            active_actions.discard(row["action"])
        seen_ids.add(row["authorization_id"])
        previous = row["content_sha256"]
    return {
        "active_action_count": len(active_actions),
        "authorization_count": len(authorizations),
        "chain_head_sha256": previous,
    }


def _event_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["content_sha256"] = ZERO_SHA256
    return payload


def make_event(
    *,
    sequence: int,
    previous_sha256: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "action": authorization["action"],
        "actor_opaque_id": authorization["actor_opaque_id"],
        "artifact_sha256": authorization["artifact_sha256"],
        "authorization_id": authorization["authorization_id"],
        "content_sha256": ZERO_SHA256,
        "event_id": stable_hash(
            {
                "action": authorization["action"],
                "authorization": authorization["authorization_id"],
                "sequence": sequence,
            }
        )[:24],
        "previous_sha256": previous_sha256,
        "sequence": sequence,
        "test_only": authorization["test_only"],
    }
    value["content_sha256"] = stable_hash(_event_payload(value))
    return value


def verify_ceremony(
    *,
    assignments: Mapping[str, Any],
    authorizations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authorization_report = verify_authorization_chain(
        authorizations, assignments
    )
    by_id = {row["authorization_id"]: row for row in authorizations}
    previous = ZERO_SHA256
    seen_events: set[str] = set()
    seen_actions: list[str] = []
    for sequence, event in enumerate(events):
        _require_keys(event, EVENT_KEYS, "ceremony_event_schema_invalid")
        authorization = by_id.get(event["authorization_id"])
        if (
            event["sequence"] != sequence
            or event["previous_sha256"] != previous
            or event["event_id"] in seen_events
            or authorization is None
            or authorization["state"] != "active"
            or event["action"] != authorization["action"]
            or event["actor_opaque_id"] != authorization["actor_opaque_id"]
            or event["artifact_sha256"] != authorization["artifact_sha256"]
            or event["test_only"] != assignments["test_only"]
            or not _is_digest(event["content_sha256"])
            or stable_hash(_event_payload(event)) != event["content_sha256"]
        ):
            raise RoleControlError("ceremony_event_chain_invalid")
        if event["action"] in seen_actions:
            raise RoleControlError("duplicate_approval_or_action")
        seen_actions.append(event["action"])
        seen_events.add(event["event_id"])
        previous = event["content_sha256"]

    expected_order = list(ACTIONS)
    positions = [expected_order.index(action) for action in seen_actions]
    if positions != sorted(positions):
        raise RoleControlError("ceremony_action_order_invalid")
    if set(seen_actions) != set(ACTIONS):
        raise RoleControlError("ceremony_action_missing")
    assignment_map = assignments["assignments"]
    upstream_roles = (
        "plan_sealer",
        "launch_authorizer",
        "execution_operator",
        "human_package_coordinator",
        "adjudicator",
        "scorer_recipient",
        "evidence_auditor",
    )
    upstream_generators = {assignment_map[role] for role in upstream_roles}
    clearance = assignment_map["clearance_approver"]
    if upstream_generators == {clearance}:
        raise RoleControlError("self_generated_self_clearance")
    approval_principals = {
        assignment_map["evidence_auditor"],
        assignment_map["clearance_approver"],
    }
    approval_domains = {
        ROLE_DOMAINS["evidence_auditor"],
        ROLE_DOMAINS["clearance_approver"],
    }
    if (
        len(approval_principals) < MINIMUM_INDEPENDENCE[
            "clearance_approval_domains"
        ]
        or len(approval_domains)
        < MINIMUM_INDEPENDENCE["clearance_approval_domains"]
    ):
        raise RoleControlError("minimum_clearance_independence_not_met")
    if assignment_map["release_signer"] == assignment_map["revocation_administrator"]:
        raise RoleControlError("signer_revocation_conflict")
    return {
        **authorization_report,
        "approval_domain_count": len(approval_domains),
        "approval_principal_count": len(approval_principals),
        "ceremony_chain_head_sha256": previous,
        "event_count": len(events),
        "status": "separation_controls_ready",
    }


def build_synthetic_ceremony() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = synthetic_assignments()
    alias_for_identity = {
        row["opaque_id"]: alias for alias, row in assignments["aliases"].items()
    }
    authorizations: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for sequence, action in enumerate(ACTIONS):
        role = ACTIONS[action]
        actor = assignments["assignments"][role]
        row = make_authorization(
            sequence=sequence,
            previous_sha256=previous,
            action=action,
            actor_opaque_id=actor,
            alias=alias_for_identity[actor],
            artifact_type=next(
                entry
                for entry, actions in ENTRYPOINT_ACTIONS.items()
                if action in actions
            ),
            artifact_sha256=stable_hash(
                {"action": action, "synthetic": True}
            ),
        )
        authorizations.append(row)
        previous = row["content_sha256"]
    events: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for sequence, authorization in enumerate(authorizations):
        event = make_event(
            sequence=sequence,
            previous_sha256=previous,
            authorization=authorization,
        )
        events.append(event)
        previous = event["content_sha256"]
    return assignments, authorizations, events


def verify_entry_authorization(
    *,
    entrypoint: str,
    action: str,
    artifact_sha256: str,
    authorization: Mapping[str, Any],
    assignments: Mapping[str, Any],
    formal_mode: bool,
) -> None:
    if entrypoint not in ENTRYPOINT_ACTIONS or action not in ENTRYPOINT_ACTIONS[entrypoint]:
        raise RoleControlError("entrypoint_action_not_allowed")
    verify_assignments(assignments, require_real=formal_mode)
    verify_authorization_chain([authorization], assignments)
    if (
        authorization["action"] != action
        or authorization["artifact_type"] != entrypoint
        or authorization["artifact_sha256"] != artifact_sha256
    ):
        raise RoleControlError("entrypoint_authorization_binding_invalid")
    if authorization["state"] != "active":
        raise RoleControlError("authorization_not_active")
    if formal_mode and authorization["test_only"]:
        raise RoleControlNotReady("missing_real_role_assignments")


def simulate_matrix() -> dict[str, Any]:
    base_assignments, base_authorizations, base_events = (
        build_synthetic_ceremony()
    )
    verify_ceremony(
        assignments=base_assignments,
        authorizations=base_authorizations,
        events=base_events,
    )
    scenarios = [
        {"scenario": "legal_full_ceremony", "status": "passed"},
    ]

    def rejected(name: str, mutate) -> None:
        assignments = copy.deepcopy(base_assignments)
        authorizations = copy.deepcopy(base_authorizations)
        events = copy.deepcopy(base_events)
        mutate(assignments, authorizations, events)
        try:
            verify_ceremony(
                assignments=assignments,
                authorizations=authorizations,
                events=events,
            )
        except RoleControlError:
            scenarios.append({"scenario": name, "status": "rejected"})
        else:
            raise RoleControlError(f"attack_not_rejected:{name}")

    rejected(
        "same_person_double_annotation",
        lambda a, _u, _e: a["assignments"].__setitem__(
            "annotator_b", a["assignments"]["annotator_a"]
        ),
    )
    rejected(
        "annotator_is_adjudicator",
        lambda a, _u, _e: a["assignments"].__setitem__(
            "adjudicator", a["assignments"]["annotator_a"]
        ),
    )
    rejected(
        "executor_self_authorizes",
        lambda a, _u, _e: a["assignments"].__setitem__(
            "launch_authorizer", a["assignments"]["execution_operator"]
        ),
    )
    rejected(
        "self_generated_self_clearance",
        lambda a, _u, _e: [
            a["assignments"].__setitem__(
                role, a["assignments"]["clearance_approver"]
            )
            for role in (
                "plan_sealer",
                "launch_authorizer",
                "execution_operator",
                "human_package_coordinator",
                "adjudicator",
                "scorer_recipient",
                "evidence_auditor",
            )
        ],
    )
    rejected(
        "signer_self_approves_revocation",
        lambda a, _u, _e: a["assignments"].__setitem__(
            "revocation_administrator", a["assignments"]["release_signer"]
        ),
    )
    rejected(
        "alias_rebinding",
        lambda a, _u, _e: a["aliases"].__setitem__(
            "alias_rebound",
            {
                "opaque_id": next(iter(a["aliases"].values()))["opaque_id"],
                "principal_binding_sha256": "1" * 64,
            },
        ),
    )
    rejected(
        "authorization_tampering",
        lambda _a, u, _e: u[0].__setitem__("artifact_sha256", "1" * 64),
    )
    rejected(
        "duplicate_approval",
        lambda _a, _u, e: e.append(copy.deepcopy(e[-1])),
    )
    rejected(
        "cross_commit_reuse",
        lambda _a, u, _e: u[0].__setitem__("code_commit", "1" * 40),
    )

    rotated = copy.deepcopy(base_assignments)
    old = rotated["assignments"]["human_package_coordinator"]
    new = opaque_identity("synthetic:replacement-coordinator")
    rotated["assignments"]["human_package_coordinator"] = new
    alias = next(
        alias
        for alias, row in rotated["aliases"].items()
        if row["opaque_id"] == old
    )
    rotated["aliases"][alias] = {
        "opaque_id": new,
        "principal_binding_sha256": stable_hash(
            {"synthetic_principal": "replacement-coordinator"}
        ),
    }
    verify_assignments(rotated)
    scenarios.append({"scenario": "legal_personnel_rotation", "status": "passed"})
    return {
        "execution": dict(EXECUTION),
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "schema_version": SCHEMA_VERSION,
        "status": "separation_controls_ready",
    }


def audit_current(protocol: Mapping[str, Any], assignment: Mapping[str, Any]) -> dict[str, Any]:
    if protocol["protocol"] != PROTOCOL:
        raise RoleControlError("protocol_schema_invalid")
    _require_keys(assignment, ASSIGNMENT_KEYS, "assignment_schema_invalid")
    if (
        assignment["protocol"] != PROTOCOL
        or assignment["schema_version"] != SCHEMA_VERSION
        or assignment["source_commit"] != SOURCE_COMMIT
        or assignment["status"] != "not_ready_missing_real_role_assignments"
        or assignment["assignments"] != {}
        or assignment["aliases"] != {}
        or assignment["test_only"] is not False
        or assignment["real_identity_mapping"] != "not_available"
        or assignment["formal_validation_complete"] is not False
    ):
        raise RoleControlError("assignment_schema_invalid")
    return {
        "controls_ready": True,
        "execution": dict(EXECUTION),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(REAL_BLOCKERS),
        "formal_validation_complete": False,
        "minimum_clearance_approval_domains": 2,
        "protocol": PROTOCOL,
        "real_identity_mapping": "not_available",
        "real_role_assignment_count": 0,
        "role_count": len(ROLES),
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_real_role_assignments",
    }
