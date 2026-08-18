"""Evidence revocation, propagation, and incident-response controls.

The authoritative ledger is append-only and contains no evidence payloads.
It records only stable evidence identities, structured reason codes, and a
hash chain.  Historical evidence files are never rewritten by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "evidence_revocation_response_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
STATES = ("active", "under_investigation", "revoked", "superseded", "restored")
BLOCKING_STATES = frozenset({"under_investigation", "revoked", "superseded"})
REASON_CODES = (
    "content_tampering",
    "duplicate_or_wrong_publication",
    "erroneous_extrapolation",
    "implementation_defect",
    "input_identity_error",
    "protocol_error",
    "sensitive_information_leakage",
    "stale_dependency",
    "statistical_error",
)
TRANSITIONS = {
    "active": frozenset({"under_investigation", "revoked"}),
    "under_investigation": frozenset({"revoked"}),
    "revoked": frozenset({"superseded"}),
    "superseded": frozenset({"restored"}),
    "restored": frozenset(),
}
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
PUBLICATION_TARGETS = (
    "clearance_receipt",
    "release_candidate",
    "standalone_auditor_bundle",
    "validation_readiness_bundle",
)
HEX = frozenset("0123456789abcdef")
DEFAULT_PROTOCOL = Path("benchmark/evidence_revocation_response_v1_protocol.json")
DEFAULT_LEDGER = Path("benchmark/evidence_revocation_response_v1_ledger.json")


class RevocationError(RuntimeError):
    """The revocation contract, ledger, or propagation result is invalid."""


class ActiveIncident(RevocationError):
    """A valid active incident blocks publication or formal clearance."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise RevocationError("duplicate_json_key")
        value[key] = child
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                RevocationError("nonfinite_json_number")
            ),
        )
    except RevocationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RevocationError("json_input_unavailable") from exc
    if not isinstance(value, dict):
        raise RevocationError("json_root_not_object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _require_keys(value: Mapping[str, Any], expected: set[str], reason: str) -> None:
    if set(value) != expected:
        raise RevocationError(reason)


def _safe_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", "..", ".env"} for part in value.parts)
        or value.parts[0] == "third_party"
    ):
        raise RevocationError("unsafe_contract_path")
    path = (root / Path(*value.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RevocationError("contract_path_escape") from exc
    return path


def _protocol_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["protocol_sha256"] = "0" * 64
    return payload


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require_keys(
        value,
        {
            "execution",
            "formal_validation_complete",
            "paths",
            "propagation",
            "protocol",
            "protocol_sha256",
            "reason_codes",
            "schema_version",
            "source_commit",
            "states",
            "transitions",
        },
        "protocol_schema_invalid",
    )
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["states"] != list(STATES)
        or value["reason_codes"] != list(REASON_CODES)
        or value["execution"] != EXECUTION
        or value["formal_validation_complete"] is not False
        or not _is_commit(value["source_commit"])
    ):
        raise RevocationError("protocol_semantics_invalid")
    expected_transitions = {
        state: sorted(targets) for state, targets in TRANSITIONS.items()
    }
    if value["transitions"] != expected_transitions:
        raise RevocationError("protocol_transition_drift")
    paths = value["paths"]
    if not isinstance(paths, dict) or set(paths) != {
        "freshness_contract",
        "ledger",
        "readiness_contract",
    }:
        raise RevocationError("protocol_paths_invalid")
    if any(not isinstance(item, str) for item in paths.values()):
        raise RevocationError("protocol_paths_invalid")
    propagation = value["propagation"]
    if not isinstance(propagation, dict) or propagation != {
        "active_states": sorted(BLOCKING_STATES),
        "publication_targets": list(PUBLICATION_TARGETS),
        "unrelated_evidence_preserved": True,
    }:
        raise RevocationError("protocol_propagation_drift")
    if (
        not _is_digest(value["protocol_sha256"])
        or stable_hash(_protocol_payload(value)) != value["protocol_sha256"]
    ):
        raise RevocationError("protocol_hash_mismatch")
    return value


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(event)
    value["event_sha256"] = "0" * 64
    return value


def _ledger_payload(ledger: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(ledger)
    value["ledger_sha256"] = "0" * 64
    return value


EVENT_KEYS = {
    "after_state",
    "before_state",
    "event_id",
    "event_index",
    "event_sha256",
    "evidence_id",
    "impact_scope",
    "operator_identity",
    "previous_event_sha256",
    "reason_code",
    "replacement_evidence_id",
    "replacement_evidence_sha256",
    "replacement_gate_ids",
    "trigger_evidence_sha256",
}


def verify_ledger(
    ledger: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    freshness_contract: Mapping[str, Any] | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    _require_keys(
        ledger,
        {
            "events",
            "formal_validation_complete",
            "ledger_sha256",
            "protocol",
            "protocol_sha256",
            "schema_version",
            "source_commit",
        },
        "ledger_schema_invalid",
    )
    if (
        ledger["protocol"] != PROTOCOL
        or ledger["schema_version"] != SCHEMA_VERSION
        or ledger["protocol_sha256"] != protocol["protocol_sha256"]
        or ledger["source_commit"] != protocol["source_commit"]
        or ledger["formal_validation_complete"] is not False
        or not isinstance(ledger["events"], list)
        or not _is_digest(ledger["ledger_sha256"])
        or stable_hash(_ledger_payload(ledger)) != ledger["ledger_sha256"]
    ):
        raise RevocationError("ledger_identity_or_hash_invalid")
    states: dict[str, str] = {}
    replacements: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    previous = "0" * 64
    seen_event_ids: set[str] = set()
    for index, raw in enumerate(ledger["events"]):
        if not isinstance(raw, dict):
            raise RevocationError("ledger_event_not_object")
        _require_keys(raw, EVENT_KEYS, "ledger_event_schema_invalid")
        evidence_id = raw["evidence_id"]
        if (
            raw["event_index"] != index
            or not isinstance(evidence_id, str)
            or not evidence_id
            or not isinstance(raw["event_id"], str)
            or not raw["event_id"]
            or raw["event_id"] in seen_event_ids
            or raw["previous_event_sha256"] != previous
            or raw["before_state"] != states.get(evidence_id, "active")
            or raw["after_state"] not in TRANSITIONS.get(raw["before_state"], ())
            or raw["reason_code"] not in REASON_CODES
            or not _is_digest(raw["trigger_evidence_sha256"])
            or not isinstance(raw["operator_identity"], str)
            or not raw["operator_identity"].startswith("operator_")
            or not isinstance(raw["impact_scope"], list)
            or raw["impact_scope"] != sorted(set(raw["impact_scope"]))
            or any(not isinstance(item, str) or not item for item in raw["impact_scope"])
            or not isinstance(raw["replacement_gate_ids"], list)
            or raw["replacement_gate_ids"] != sorted(set(raw["replacement_gate_ids"]))
            or not _is_digest(raw["event_sha256"])
            or stable_hash(_event_payload(raw)) != raw["event_sha256"]
        ):
            raise RevocationError("ledger_event_integrity_invalid")
        replacement_id = raw["replacement_evidence_id"]
        replacement_sha = raw["replacement_evidence_sha256"]
        if raw["after_state"] in {"superseded", "restored"}:
            if (
                not isinstance(replacement_id, str)
                or not replacement_id
                or not _is_digest(replacement_sha)
                or not raw["replacement_gate_ids"]
            ):
                raise RevocationError("replacement_evidence_required")
            if freshness_contract is not None:
                _verify_replacement(
                    replacement_id,
                    replacement_sha,
                    raw["replacement_gate_ids"],
                    freshness_contract,
                    repository_root,
                )
            replacement = (
                str(replacement_id),
                str(replacement_sha),
                tuple(raw["replacement_gate_ids"]),
            )
            if (
                raw["after_state"] == "restored"
                and replacements.get(evidence_id) != replacement
            ):
                raise RevocationError("restoration_replacement_identity_drift")
            replacements[evidence_id] = replacement
        elif (
            replacement_id is not None
            or replacement_sha is not None
            or raw["replacement_gate_ids"]
        ):
            raise RevocationError("unexpected_replacement_evidence")
        seen_event_ids.add(raw["event_id"])
        states[evidence_id] = raw["after_state"]
        previous = raw["event_sha256"]
    active = sorted(
        evidence_id for evidence_id, state in states.items() if state in BLOCKING_STATES
    )
    return {
        "active_incident_count": len(active),
        "active_incident_evidence_ids": active,
        "event_count": len(ledger["events"]),
        "evidence_states": [
            {"evidence_id": evidence_id, "state": state}
            for evidence_id, state in sorted(states.items())
        ],
        "formal_validation_complete": False,
        "ledger_sha256": ledger["ledger_sha256"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "active_incident_blocks_release" if active else "revocation_controls_ready",
        "exit_code": EXIT_BLOCKED if active else EXIT_READY,
    }


def _verify_replacement(
    evidence_id: str,
    evidence_sha256: str,
    gate_ids: Sequence[str],
    contract: Mapping[str, Any],
    repository_root: Path | None,
) -> None:
    rows = {
        str(row.get("evidence_id")): row
        for row in (contract.get("bindings") or {}).get("evidence", [])
        if isinstance(row, dict)
    }
    gates = {
        str(row.get("gate_id")): row
        for row in (contract.get("bindings") or {}).get("gates", [])
        if isinstance(row, dict)
    }
    row = rows.get(evidence_id)
    if (
        row is None
        or row.get("declared_state") != "fresh"
        or row.get("artifact_sha256") != evidence_sha256
        or any(gate not in gates or gates[gate].get("declared_state") != "fresh" for gate in gate_ids)
    ):
        raise RevocationError("replacement_evidence_not_fully_gated")
    if repository_root is not None:
        path = _safe_path(repository_root, str(row["artifact_path"]))
        if not path.is_file() or sha256_file(path) != evidence_sha256:
            raise RevocationError("replacement_evidence_artifact_drift")


def new_empty_ledger(protocol: Mapping[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "events": [],
        "formal_validation_complete": False,
        "ledger_sha256": "0" * 64,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
    }
    ledger["ledger_sha256"] = stable_hash(_ledger_payload(ledger))
    return ledger


def append_event(
    ledger: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    evidence_id: str,
    after_state: str,
    reason_code: str,
    trigger_evidence_sha256: str,
    operator_identity: str,
    impact_scope: Sequence[str],
    replacement_evidence_id: str | None = None,
    replacement_evidence_sha256: str | None = None,
    replacement_gate_ids: Sequence[str] = (),
    freshness_contract: Mapping[str, Any] | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    verified = verify_ledger(
        ledger,
        protocol,
        freshness_contract=freshness_contract,
        repository_root=repository_root,
    )
    state_map = {
        row["evidence_id"]: row["state"] for row in verified["evidence_states"]
    }
    before_state = state_map.get(evidence_id, "active")
    if after_state not in TRANSITIONS.get(before_state, ()):
        raise RevocationError("invalid_state_transition")
    events = [dict(row) for row in ledger["events"]]
    event: dict[str, Any] = {
        "after_state": after_state,
        "before_state": before_state,
        "event_id": "",
        "event_index": len(events),
        "event_sha256": "0" * 64,
        "evidence_id": evidence_id,
        "impact_scope": sorted(set(impact_scope)),
        "operator_identity": operator_identity,
        "previous_event_sha256": events[-1]["event_sha256"] if events else "0" * 64,
        "reason_code": reason_code,
        "replacement_evidence_id": replacement_evidence_id,
        "replacement_evidence_sha256": replacement_evidence_sha256,
        "replacement_gate_ids": sorted(set(replacement_gate_ids)),
        "trigger_evidence_sha256": trigger_evidence_sha256,
    }
    event["event_id"] = "event_" + stable_hash(
        {key: value for key, value in event.items() if key not in {"event_id", "event_sha256"}}
    )[:24]
    event["event_sha256"] = stable_hash(_event_payload(event))
    result = dict(ledger)
    result["events"] = [*events, event]
    result["ledger_sha256"] = "0" * 64
    result["ledger_sha256"] = stable_hash(_ledger_payload(result))
    verify_ledger(
        result,
        protocol,
        freshness_contract=freshness_contract,
        repository_root=repository_root,
    )
    return result


def load_current(
    repository_root: Path,
    *,
    protocol_path: Path | None = None,
    ledger_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repository_root.resolve()
    protocol_file = protocol_path or root / DEFAULT_PROTOCOL
    if not protocol_file.is_absolute():
        protocol_file = root / protocol_file
    protocol = load_protocol(protocol_file)
    paths = protocol["paths"]
    freshness = read_json(_safe_path(root, str(paths["freshness_contract"])))
    readiness = read_json(_safe_path(root, str(paths["readiness_contract"])))
    ledger_file = ledger_path or _safe_path(root, str(paths["ledger"]))
    if not ledger_file.is_absolute():
        ledger_file = root / ledger_file
    ledger = read_json(ledger_file)
    return protocol, ledger, freshness, readiness


def propagation_report(
    ledger: Mapping[str, Any],
    protocol: Mapping[str, Any],
    freshness: Mapping[str, Any],
    readiness: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    verification = verify_ledger(
        ledger,
        protocol,
        freshness_contract=freshness,
        repository_root=repository_root,
    )
    active_ids = set(verification["active_incident_evidence_ids"])
    evidence_rows = {
        str(row["evidence_id"]): row
        for row in (freshness.get("bindings") or {}).get("evidence", [])
        if isinstance(row, dict) and row.get("evidence_id")
    }
    readiness_ids = {
        str(row["evidence_id"])
        for row in readiness.get("evidence", [])
        if isinstance(row, dict) and row.get("evidence_id")
    }
    unknown = sorted(active_ids - (set(evidence_rows) | readiness_ids))
    if unknown:
        raise RevocationError("revoked_evidence_not_registered")
    reverse: dict[str, set[str]] = defaultdict(set)
    for evidence_id, row in evidence_rows.items():
        for dependency in row.get("depends_on_evidence") or []:
            reverse[str(dependency)].add(evidence_id)
    impacted_evidence = set(active_ids)
    queue = deque(sorted(active_ids))
    while queue:
        current = queue.popleft()
        for child in sorted(reverse.get(current, ())):
            if child not in impacted_evidence:
                impacted_evidence.add(child)
                queue.append(child)
    components = {
        component
        for evidence_id in impacted_evidence
        for component in evidence_rows.get(evidence_id, {}).get("components", [])
    }
    impacted_claims: set[str] = set()
    for row in (freshness.get("bindings") or {}).get("claims", []):
        if set(row.get("evidence_ids") or []) & impacted_evidence or set(
            row.get("components") or []
        ) & components:
            impacted_claims.add(str(row["claim_id"]))
    impacted_gates: set[str] = set()
    for row in (freshness.get("bindings") or {}).get("gates", []):
        if set(row.get("evidence_ids") or []) & impacted_evidence or set(
            row.get("components") or []
        ) & components:
            impacted_gates.add(str(row["gate_id"]))
    for evidence_id in impacted_evidence:
        impacted_gates.update(
            str(item)
            for item in evidence_rows.get(evidence_id, {}).get("rerun_gate_ids", [])
        )
    all_evidence_ids = set(evidence_rows) | readiness_ids
    unaffected = sorted(all_evidence_ids - impacted_evidence)
    blocking = bool(active_ids)
    return {
        **verification,
        "impacted_claim_ids": sorted(impacted_claims),
        "impacted_component_ids": sorted(components),
        "impacted_evidence_ids": sorted(impacted_evidence),
        "invalidated_publication_targets": list(PUBLICATION_TARGETS) if blocking else [],
        "minimum_rerun_gate_ids": sorted(impacted_gates),
        "prohibited_publication_actions": (
            [
                "issue_clearance_receipt",
                "publish_release_candidate",
                "publish_standalone_auditor_bundle",
                "publish_validation_readiness_bundle",
            ]
            if blocking
            else []
        ),
        "unaffected_evidence_ids": unaffected,
    }


def audit_current(
    repository_root: Path,
    *,
    protocol_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    protocol, ledger, freshness, readiness = load_current(
        repository_root,
        protocol_path=protocol_path,
        ledger_path=ledger_path,
    )
    report = propagation_report(
        ledger,
        protocol,
        freshness,
        readiness,
        repository_root=repository_root.resolve(),
    )
    report["execution"] = EXECUTION
    report["formal_validation_complete"] = False
    report["real_ledger_empty"] = len(ledger["events"]) == 0
    return report


def assert_no_active_incident(
    repository_root: Path,
    *,
    target: str,
    protocol_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    if target not in PUBLICATION_TARGETS:
        raise RevocationError("unknown_publication_target")
    report = audit_current(
        repository_root,
        protocol_path=protocol_path,
        ledger_path=ledger_path,
    )
    if report["active_incident_count"]:
        raise ActiveIncident(f"active_incident_blocks:{target}")
    return report


def _first_fresh_replacement(
    freshness: Mapping[str, Any], excluded: str
) -> tuple[str, str, list[str]]:
    gates = [
        str(row["gate_id"])
        for row in (freshness.get("bindings") or {}).get("gates", [])
        if row.get("declared_state") == "fresh"
    ]
    for row in (freshness.get("bindings") or {}).get("evidence", []):
        if row.get("evidence_id") != excluded and row.get("declared_state") == "fresh":
            return str(row["evidence_id"]), str(row["artifact_sha256"]), [gates[0]]
    raise RevocationError("synthetic_replacement_unavailable")


def simulate_incidents(
    protocol: Mapping[str, Any],
    freshness: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    def revoke(evidence_id: str, reason: str) -> dict[str, Any]:
        ledger = new_empty_ledger(protocol)
        return append_event(
            ledger,
            protocol,
            evidence_id=evidence_id,
            after_state="revoked",
            reason_code=reason,
            trigger_evidence_sha256=stable_hash({"synthetic": evidence_id}),
            operator_identity="operator_synthetic_auditor",
            impact_scope=["claims", "gates", "publication"],
        )

    ranking = revoke("ranking_decision_manifest", "implementation_defect")
    ranking_report = propagation_report(ranking, protocol, freshness, readiness)
    human = revoke("human_annotation_delivery_protocol", "protocol_error")
    human_report = propagation_report(human, protocol, freshness, readiness)
    policy = revoke("evidence_registry_gate", "content_tampering")
    policy_report = propagation_report(policy, protocol, freshness, readiness)

    replacement_id, replacement_sha, gate_ids = _first_fresh_replacement(
        freshness, "ranking_decision_manifest"
    )
    recovered = append_event(
        ranking,
        protocol,
        evidence_id="ranking_decision_manifest",
        after_state="superseded",
        reason_code="implementation_defect",
        trigger_evidence_sha256=stable_hash({"synthetic": "replacement"}),
        operator_identity="operator_synthetic_auditor",
        impact_scope=["claims", "gates", "publication"],
        replacement_evidence_id=replacement_id,
        replacement_evidence_sha256=replacement_sha,
        replacement_gate_ids=gate_ids,
        freshness_contract=freshness,
    )
    recovered = append_event(
        recovered,
        protocol,
        evidence_id="ranking_decision_manifest",
        after_state="restored",
        reason_code="implementation_defect",
        trigger_evidence_sha256=stable_hash({"synthetic": "restoration"}),
        operator_identity="operator_synthetic_auditor",
        impact_scope=["claims", "gates", "publication"],
        replacement_evidence_id=replacement_id,
        replacement_evidence_sha256=replacement_sha,
        replacement_gate_ids=gate_ids,
        freshness_contract=freshness,
    )
    recovery_report = propagation_report(recovered, protocol, freshness, readiness)

    rejected: list[str] = []
    for name, mutate in (
        ("deleted_event", lambda rows: rows[1:]),
        ("reordered_events", lambda rows: list(reversed(rows))),
        (
            "forged_event",
            lambda rows: [
                {**rows[0], "reason_code": "statistical_error"},
                *rows[1:],
            ],
        ),
    ):
        tampered = dict(recovered)
        tampered["events"] = mutate([dict(row) for row in recovered["events"]])
        tampered["ledger_sha256"] = "0" * 64
        tampered["ledger_sha256"] = stable_hash(_ledger_payload(tampered))
        try:
            verify_ledger(tampered, protocol, freshness_contract=freshness)
        except RevocationError:
            rejected.append(name)
    if set(rejected) != {"deleted_event", "reordered_events", "forged_event"}:
        raise RevocationError("synthetic_ledger_attack_not_rejected")
    if "source_reliability_protocol" in set(human_report["impacted_evidence_ids"]):
        raise RevocationError("unrelated_evidence_propagation")
    if recovery_report["active_incident_count"] != 0:
        raise RevocationError("synthetic_restoration_failed")
    return {
        "execution": EXECUTION,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "scenarios": [
            {
                "scenario": "ranking_evidence_revoked",
                "active_incident_count": ranking_report["active_incident_count"],
                "impacted_claim_count": len(ranking_report["impacted_claim_ids"]),
                "impacted_evidence_count": len(ranking_report["impacted_evidence_ids"]),
                "minimum_rerun_gate_ids": ranking_report["minimum_rerun_gate_ids"],
            },
            {
                "scenario": "human_delivery_revoked",
                "active_incident_count": human_report["active_incident_count"],
                "source_reliability_preserved": "source_reliability_protocol"
                not in set(human_report["impacted_evidence_ids"]),
            },
            {
                "scenario": "default_policy_evidence_revoked",
                "active_incident_count": policy_report["active_incident_count"],
                "invalidated_publication_targets": policy_report[
                    "invalidated_publication_targets"
                ],
            },
            {
                "scenario": "fully_gated_supersession_and_restoration",
                "active_incident_count": recovery_report["active_incident_count"],
                "replacement_evidence_id": replacement_id,
            },
            {
                "scenario": "ledger_attacks",
                "rejected": sorted(rejected),
            },
        ],
        "scenario_count": 5,
        "status": "revocation_controls_ready",
        "exit_code": EXIT_READY,
    }
