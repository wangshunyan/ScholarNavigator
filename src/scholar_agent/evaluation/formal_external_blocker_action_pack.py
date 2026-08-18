"""Deterministic handoff for the three remaining formal-validation blockers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL = "formal_external_blocker_action_pack_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "1742e566be0cd5d073023e9813f086d05912fe2b"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_MISSING = 3
EXIT_USAGE = 4
CHAIN_READY = "engineering_ready_external_input_missing"
OVERALL_STATE = "external_action_required"
FORMAL_BLOCKERS = [
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
]
EXECUTION_ZERO = {
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
SENSITIVE = re.compile(
    r"(?:^|[^a-z])(?:\.env|api[_-]?key|authorization|bearer|password|secret|"
    r"query[_ -]?text|paper[_ -]?abstract|gold|qrels)(?:$|[^a-z])",
    re.IGNORECASE,
)


class ExternalBlockerActionError(RuntimeError):
    pass


class ExternalInputsMissing(ExternalBlockerActionError):
    pass


GateRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _unique(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ExternalBlockerActionError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ExternalBlockerActionError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ExternalBlockerActionError("nonfinite_json_number")
            ),
        )
    except ExternalBlockerActionError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ExternalBlockerActionError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ExternalBlockerActionError("json_root_not_object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise ExternalBlockerActionError("relative_path_invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ExternalBlockerActionError("relative_path_invalid")
    return value


def _validate_no_sensitive_content(value: Any) -> None:
    encoded = canonical_json(value).decode("utf-8")
    if SENSITIVE.search(encoded):
        raise ExternalBlockerActionError("sensitive_or_private_content_forbidden")
    for token in encoded.replace('"', " ").split():
        if token.startswith(("/Users/", "/home/", "C:\\Users\\")):
            raise ExternalBlockerActionError("absolute_path_forbidden")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "chains", "development_freeze", "execution", "formal_blockers",
        "formal_validation_complete", "protocol", "protocol_sha256",
        "schema_version", "source_commit",
    }
    if set(value) != required:
        raise ExternalBlockerActionError("protocol_schema_invalid")
    payload = dict(value)
    claimed = payload.pop("protocol_sha256")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_blockers"] != FORMAL_BLOCKERS
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or claimed != stable_hash(payload)
    ):
        raise ExternalBlockerActionError("protocol_identity_invalid")
    freeze = value["development_freeze"]
    if freeze != {
        "allowed_unfreeze_triggers": [
            "machine_verified_concrete_engineering_defect",
            "real_external_input_received",
        ],
        "forbidden_without_trigger": "new_governance_or_preparation_task",
        "state_when_all_chains_ready": OVERALL_STATE,
    }:
        raise ExternalBlockerActionError("development_freeze_policy_invalid")
    chains = value["chains"]
    if not isinstance(chains, list) or [row.get("chain_id") for row in chains] != [
        "backup_members", "human_annotation", "official_scorer"
    ]:
        raise ExternalBlockerActionError("chain_inventory_invalid")
    gate_ids: set[str] = set()
    for chain in chains:
        if not isinstance(chain, dict) or set(chain) != {
            "action_plan", "chain_id", "gates", "input_contract",
            "missing_inputs", "prohibited_actions",
        }:
            raise ExternalBlockerActionError("chain_schema_invalid")
        if not chain["missing_inputs"] or not chain["action_plan"] or not chain["gates"]:
            raise ExternalBlockerActionError("chain_required_list_empty")
        if not isinstance(chain["input_contract"], dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in chain["input_contract"].items()
        ):
            raise ExternalBlockerActionError("input_contract_invalid")
        if not isinstance(chain["prohibited_actions"], list) or not all(
            isinstance(item, str) for item in chain["prohibited_actions"]
        ):
            raise ExternalBlockerActionError("prohibited_action_invalid")
        for action in chain["action_plan"]:
            if not isinstance(action, dict) or set(action) != {
                "action_id", "command", "expected_exit_code", "failure_rollback",
                "success_criterion",
            }:
                raise ExternalBlockerActionError("action_schema_invalid")
            if (not all(isinstance(action[key], str) for key in
                        ("action_id", "command", "failure_rollback", "success_criterion"))
                    or action["expected_exit_code"] != EXIT_READY
                    or not action["command"].startswith("python scripts/")):
                raise ExternalBlockerActionError("action_contract_invalid")
        for gate in chain["gates"]:
            if not isinstance(gate, dict) or set(gate) != {
                "expected_exit_code", "expected_status", "gate_id", "script"
            }:
                raise ExternalBlockerActionError("gate_schema_invalid")
            gate_id = gate["gate_id"]
            if not isinstance(gate_id, str) or gate_id in gate_ids:
                raise ExternalBlockerActionError("gate_identity_invalid")
            gate_ids.add(gate_id)
            _safe_relative(gate["script"])
            if gate["expected_exit_code"] != EXIT_MISSING:
                raise ExternalBlockerActionError("blocked_gate_exit_invalid")
    _validate_no_sensitive_content(value)
    return value


def _subprocess_gate(repository_root: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    script = repository_root / _safe_relative(gate["script"])
    if not script.is_file():
        return {"exit_code": EXIT_VIOLATION, "status": "gate_script_missing"}
    completed = subprocess.run(
        [sys.executable, str(script), "audit-readiness"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        timeout=120,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "src",
        },
    )
    if completed.stderr:
        return {"exit_code": completed.returncode, "status": "gate_stderr_nonempty"}
    try:
        value = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, ValueError, json.JSONDecodeError, ExternalBlockerActionError):
        return {"exit_code": completed.returncode, "status": "gate_output_invalid"}
    if not isinstance(value, dict):
        return {"exit_code": completed.returncode, "status": "gate_output_invalid"}
    return {"exit_code": completed.returncode, "status": value.get("status")}


def _freshness_gate(repository_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "scripts/check_validation_freshness.py", "verify-current"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        timeout=180,
        env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "PATH": os.environ.get("PATH", ""), "PYTHONPATH": "src"},
    )
    if completed.returncode != 0 or completed.stderr:
        return {"exit_code": completed.returncode, "status": "freshness_not_closed"}
    try:
        value = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, ValueError, json.JSONDecodeError, ExternalBlockerActionError):
        return {"exit_code": EXIT_VIOLATION, "status": "freshness_output_invalid"}
    return {"exit_code": completed.returncode, "status": value.get("status")}


def audit_chains(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    runner: GateRunner | None = None,
    verify_freshness: bool = True,
) -> dict[str, Any]:
    run = runner or _subprocess_gate
    chain_rows: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for chain in protocol["chains"]:
        gate_rows: list[dict[str, Any]] = []
        for gate in chain["gates"]:
            observed = dict(run(repository_root, gate))
            passed = (
                observed.get("exit_code") == gate["expected_exit_code"]
                and observed.get("status") == gate["expected_status"]
            )
            gate_rows.append({"gate_id": gate["gate_id"], "passed": passed,
                              "observed_exit_code": observed.get("exit_code"),
                              "observed_status": observed.get("status")})
            if not passed:
                violations.append({"chain_id": chain["chain_id"],
                                   "gate_id": gate["gate_id"],
                                   "reason_code": "blocked_gate_contract_drift"})
        chain_rows.append({
            "chain_id": chain["chain_id"],
            "engineering_gap_count": sum(not row["passed"] for row in gate_rows),
            "gate_count": len(gate_rows),
            "gates": gate_rows,
            "state": CHAIN_READY if all(row["passed"] for row in gate_rows)
            else "engineering_gap_detected",
        })
    freshness = (_freshness_gate(repository_root) if verify_freshness else
                 {"exit_code": 0, "status": "verification_deferred_until_baseline_closed"})
    if verify_freshness and freshness != {"exit_code": 0,
                                          "status": "fresh_with_declared_blockers"}:
        violations.append({"chain_id": "all", "gate_id": "validation_freshness",
                           "reason_code": "protocol_or_evidence_not_fresh"})
    ready = not violations and all(row["state"] == CHAIN_READY for row in chain_rows)
    return {
        "chain_count": len(chain_rows),
        "chains": chain_rows,
        "development_freeze_state": OVERALL_STATE if ready else "invalid",
        "engineering_gap_count": sum(row["engineering_gap_count"] for row in chain_rows),
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY if ready else EXIT_VIOLATION,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "freshness": freshness,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "external_action_pack_ready" if ready else "chain_or_pack_violation",
        "violations": violations,
    }


def build_pack(protocol: Mapping[str, Any], chain_audit: Mapping[str, Any]) -> dict[str, Any]:
    if chain_audit.get("exit_code") != EXIT_READY:
        raise ExternalBlockerActionError("chain_audit_not_ready")
    pack: dict[str, Any] = {
        "chains": [],
        "development_freeze": {
            "allowed_unfreeze_triggers": list(
                protocol["development_freeze"]["allowed_unfreeze_triggers"]
            ),
            "decision": OVERALL_STATE,
            "new_governance_task_allowed": False,
        },
        "execution": dict(EXECUTION_ZERO),
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "pack_sha256": "0" * 64,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "status": OVERALL_STATE,
    }
    audit_by_id = {row["chain_id"]: row for row in chain_audit["chains"]}
    for chain in protocol["chains"]:
        audit = audit_by_id[chain["chain_id"]]
        pack["chains"].append({
            "action_plan": chain["action_plan"],
            "chain_id": chain["chain_id"],
            "current_missing_inputs": chain["missing_inputs"],
            "engineering_gap_count": audit["engineering_gap_count"],
            "input_contract": chain["input_contract"],
            "prohibited_actions": chain["prohibited_actions"],
            "state": audit["state"],
        })
    payload = dict(pack)
    payload.pop("pack_sha256")
    pack["pack_sha256"] = stable_hash(payload)
    _validate_no_sensitive_content(pack)
    return pack


def verify_pack(pack: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if set(pack) != {
        "chains", "development_freeze", "execution", "formal_blockers",
        "formal_validation_complete", "pack_sha256", "protocol", "protocol_sha256",
        "schema_version", "source_commit", "status",
    }:
        raise ExternalBlockerActionError("pack_schema_invalid")
    payload = dict(pack)
    claimed = payload.pop("pack_sha256")
    if (
        pack["protocol"] != PROTOCOL
        or pack["schema_version"] != SCHEMA_VERSION
        or pack["source_commit"] != SOURCE_COMMIT
        or pack["protocol_sha256"] != protocol["protocol_sha256"]
        or pack["formal_blockers"] != FORMAL_BLOCKERS
        or pack["formal_validation_complete"] is not False
        or pack["execution"] != EXECUTION_ZERO
        or pack["status"] != OVERALL_STATE
        or claimed != stable_hash(payload)
    ):
        raise ExternalBlockerActionError("pack_identity_invalid")
    expected = [row["chain_id"] for row in protocol["chains"]]
    if not isinstance(pack["chains"], list) or [row.get("chain_id") for row in pack["chains"]] != expected:
        raise ExternalBlockerActionError("pack_chain_inventory_invalid")
    for row, chain in zip(pack["chains"], protocol["chains"], strict=True):
        if set(row) != {
            "action_plan", "chain_id", "current_missing_inputs",
            "engineering_gap_count", "input_contract", "prohibited_actions", "state",
        }:
            raise ExternalBlockerActionError("pack_chain_schema_invalid")
        if (row["state"] != CHAIN_READY or row["engineering_gap_count"] != 0
                or row["action_plan"] != chain["action_plan"]
                or row["current_missing_inputs"] != chain["missing_inputs"]
                or row["input_contract"] != chain["input_contract"]
                or row["prohibited_actions"] != chain["prohibited_actions"]):
            raise ExternalBlockerActionError("pack_engineering_state_invalid")
    if pack["development_freeze"] != {
        "allowed_unfreeze_triggers": protocol["development_freeze"]["allowed_unfreeze_triggers"],
        "decision": OVERALL_STATE,
        "new_governance_task_allowed": False,
    }:
        raise ExternalBlockerActionError("pack_freeze_decision_invalid")
    _validate_no_sensitive_content(pack)
    return {
        "chain_count": len(pack["chains"]),
        "development_freeze_state": OVERALL_STATE,
        "engineering_gap_count": 0,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "pack_sha256": pack["pack_sha256"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "external_action_pack_ready",
    }


def audit_readiness(pack: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_pack(pack, protocol)
    return {
        **verified,
        "exit_code": EXIT_MISSING,
        "missing_external_input_classes": [
            "qualified_real_backup_members",
            "qualified_real_annotators_and_locked_labels",
            "verified_official_scorer_package_schema_and_namespace",
        ],
        "status": "external_inputs_still_missing",
    }
