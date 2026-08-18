"""Fail-closed state machine for formal validation blocker clearance.

This module does not run retrieval, annotation, or an external scorer.  It
only evaluates versioned, hash-bound evidence and can issue an unsigned,
self-hashed receipt after every external blocker is independently satisfied.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "formal_validation_clearance_v1"
RECEIPT_PROTOCOL = "clearance_receipt_v1"
SCHEMA_VERSION = "1"
EXIT_VALID = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
STATES = {"blocked", "partially_satisfied", "eligible_for_clearance", "cleared", "invalid"}
BLOCKERS = ("full1000", "human_precision", "official_scorer")
HEX_DIGEST = frozenset("0123456789abcdef")
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class ClearanceError(RuntimeError):
    """Evidence or state transition violates the clearance contract."""


class ClearanceBlocked(ClearanceError):
    """Evidence is valid but one or more external blockers remain."""


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


def write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    try:
        with path.open(mode) as handle:
            handle.write(canonical_json(value))
    except FileExistsError as exc:
        raise ClearanceError("receipt_already_exists") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClearanceError("json_evidence_unavailable") from exc
    if not isinstance(value, dict):
        raise ClearanceError("json_root_not_object")
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX_DIGEST


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX_DIGEST


def _require_keys(value: Mapping[str, Any], keys: set[str], location: str) -> None:
    if set(value) != keys:
        raise ClearanceError(f"schema_keys_invalid:{location}")


def _repo_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ClearanceError("unsafe_evidence_path")
    if value.parts[0] == "third_party" or value.name == ".env":
        raise ClearanceError("prohibited_evidence_path")
    path = (root / Path(*value.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ClearanceError("evidence_path_escape") from exc
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("protocol") != PROTOCOL or value.get("schema_version") != SCHEMA_VERSION:
        raise ClearanceError("protocol_version_invalid")
    if value.get("states") != sorted(STATES):
        raise ClearanceError("state_inventory_drift")
    if value.get("blockers") != list(BLOCKERS):
        raise ClearanceError("blocker_inventory_drift")
    if value.get("execution") != EXECUTION or value.get("formal_validation_complete") is not False:
        raise ClearanceError("offline_or_formal_boundary_drift")
    if not _is_commit(value.get("source_commit")):
        raise ClearanceError("source_commit_invalid")
    return value


def _artifact(path: str, root: Path) -> dict[str, Any]:
    file_path = _repo_path(root, path)
    if not file_path.is_file():
        raise ClearanceError(f"current_evidence_missing:{path}")
    return {"path": path, "sha256": sha256_file(file_path), "size": file_path.stat().st_size}


def _git_source_compatible(root: Path, source_commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        timeout=10,
    )
    return completed.returncode == 0


def build_current_evidence(protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """Normalize current tracked evidence without treating partial artifacts as completion."""

    root = repository_root.resolve()
    paths = protocol["current_evidence_paths"]
    # Freshness is a live prerequisite rather than a self-referential artifact
    # binding: its report includes this gate's evidence.  Binding that report's
    # hash back into this evidence would create an impossible digest cycle.
    artifacts = {
        name: _artifact(str(path), root)
        for name, path in sorted(paths.items())
        if name != "freshness_current"
    }
    plan = _read_json(_repo_path(root, paths["full1000_plan"]))
    dry_run = _read_json(_repo_path(root, paths["full1000_dry_run"]))
    human = _read_json(_repo_path(root, paths["human_readiness"]))
    scorer = _read_json(_repo_path(root, paths["scorer_readiness"]))
    scorer_matrix = _read_json(_repo_path(root, paths["scorer_synthetic_matrix"]))
    freshness = _read_json(_repo_path(root, paths["freshness_current"]))
    registry = _read_json(_repo_path(root, paths["evidence_registry_result"]))

    experimental = (plan.get("execution_contract") or {}).get("experimental_features") or {}
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "synthetic_test_only": False,
        "source_commit": protocol["source_commit"],
        "global_prerequisites": {
            "freshness_status": freshness.get("status"),
            "stale_count": int((freshness.get("state_counts") or {}).get("stale", -1)),
            "dependency_violation_count": int(freshness.get("violation_count", -1)),
            "source_commit_compatible": _git_source_compatible(root, str(protocol["source_commit"])),
            "current_rules_default": (registry.get("inventory") or {}).get("default_enabled_strategy_ids") == ["current_rules"],
            "deterministic_tiebreak_v2_default": bool(experimental.get("deterministic_tiebreak_v2", True)),
        },
        "blockers": {
            "full1000": {
                "plan_bound": plan.get("contract") == "full1000_execution_plan_v1",
                "expected_query_count": int((plan.get("population") or {}).get("count", -1)),
                "committed_query_count": 0,
                "unique_query_count": 0,
                "run_manifest_verified": False,
                "generation_chain_verified": False,
                "resource_ledger_verified": False,
                "aggregate_verified": False,
                "aggregate_query_count": 0,
                "legacy_input": False,
                "synthetic_input": bool(dry_run.get("fixture_only", False)),
                "partial_input": True,
            },
            "human_precision": {
                "package_bound": int(human.get("item_count", -1)) == 471,
                "expected_item_count": 471,
                "annotator_a_covered_count": 0,
                "annotator_b_covered_count": 0,
                "independent_annotators": False,
                "anonymous_annotators": False,
                "gate_state": "awaiting_labels",
                "adjudication_validated": False,
                "unresolved_disagreement_count": 0,
                "synthetic_only": False,
                "label_origin": "not_available",
            },
            "official_scorer": {
                "package_name": str((scorer.get("official_package") or {}).get("scorer_name", "unknown")),
                "package_version": str((scorer.get("official_package") or {}).get("scorer_version", "unknown")),
                "package_sha256": str((scorer.get("official_package") or {}).get("package_sha256", "not_provided")),
                "input_schema": str((scorer.get("official_package") or {}).get("input_schema", "not_provided")),
                "output_schema": str((scorer.get("official_package") or {}).get("output_schema", "not_provided")),
                "metric_namespace": str((scorer.get("official_package") or {}).get("metric_namespace", "not_provided")),
                "package_verified": False,
                "complete_input_query_count": 0,
                "sandbox_verified": scorer_matrix.get("status") == "handoff_chain_verified",
                "output_verified": False,
                "synthetic_only": True,
            },
        },
        "artifacts": artifacts,
        "execution": EXECUTION,
    }
    evidence["evidence_sha256"] = stable_hash(evidence)
    return evidence


def _validate_evidence_shape(evidence: Mapping[str, Any]) -> None:
    _require_keys(
        evidence,
        {"artifacts", "blockers", "evidence_sha256", "execution", "global_prerequisites", "protocol", "schema_version", "source_commit", "synthetic_test_only"},
        "$",
    )
    if evidence.get("protocol") != PROTOCOL or evidence.get("schema_version") != SCHEMA_VERSION:
        raise ClearanceError("evidence_version_invalid")
    if evidence.get("execution") != EXECUTION or not _is_commit(evidence.get("source_commit")):
        raise ClearanceError("evidence_execution_or_commit_invalid")
    content = dict(evidence)
    claimed = content.pop("evidence_sha256", None)
    if not _is_digest(claimed) or stable_hash(content) != claimed:
        raise ClearanceError("evidence_hash_mismatch")
    _require_keys(
        evidence["global_prerequisites"],
        {"current_rules_default", "dependency_violation_count", "deterministic_tiebreak_v2_default", "freshness_status", "source_commit_compatible", "stale_count"},
        "$.global_prerequisites",
    )
    if set(evidence["blockers"]) != set(BLOCKERS):
        raise ClearanceError("blocker_evidence_inventory_invalid")
    artifact_paths: set[str] = set()
    for name, artifact in evidence["artifacts"].items():
        _require_keys(artifact, {"path", "sha256", "size"}, f"$.artifacts/{name}")
        if artifact["path"] in artifact_paths or not _is_digest(artifact["sha256"]) or int(artifact["size"]) < 0:
            raise ClearanceError("artifact_binding_invalid")
        artifact_paths.add(artifact["path"])
    _require_keys(
        evidence["blockers"]["full1000"],
        {"aggregate_query_count", "aggregate_verified", "committed_query_count", "expected_query_count", "generation_chain_verified", "legacy_input", "partial_input", "plan_bound", "resource_ledger_verified", "run_manifest_verified", "synthetic_input", "unique_query_count"},
        "$.blockers/full1000",
    )
    _require_keys(
        evidence["blockers"]["human_precision"],
        {"adjudication_validated", "anonymous_annotators", "annotator_a_covered_count", "annotator_b_covered_count", "expected_item_count", "gate_state", "independent_annotators", "label_origin", "package_bound", "synthetic_only", "unresolved_disagreement_count"},
        "$.blockers/human_precision",
    )
    _require_keys(
        evidence["blockers"]["official_scorer"],
        {"complete_input_query_count", "input_schema", "metric_namespace", "output_schema", "output_verified", "package_name", "package_sha256", "package_verified", "package_version", "sandbox_verified", "synthetic_only"},
        "$.blockers/official_scorer",
    )


def _predicates(evidence: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    full = evidence["blockers"]["full1000"]
    human = evidence["blockers"]["human_precision"]
    scorer = evidence["blockers"]["official_scorer"]
    expected = int(full["expected_query_count"])
    human_expected = int(human["expected_item_count"])
    return {
        "full1000": {
            "plan_bound": full["plan_bound"] is True,
            "complete_1000": expected == 1000 and int(full["committed_query_count"]) == expected,
            "unique_identity_1000": int(full["unique_query_count"]) == expected == 1000,
            "run_manifest_verified": full["run_manifest_verified"] is True,
            "generation_chain_verified": full["generation_chain_verified"] is True,
            "resource_ledger_verified": full["resource_ledger_verified"] is True,
            "aggregate_verified": full["aggregate_verified"] is True and int(full["aggregate_query_count"]) == 1000,
            "not_legacy_partial_or_synthetic": not any(full[key] for key in ("legacy_input", "partial_input", "synthetic_input")),
        },
        "human_precision": {
            "package_bound": human["package_bound"] is True and human_expected == 471,
            "complete_a": int(human["annotator_a_covered_count"]) == human_expected,
            "complete_b": int(human["annotator_b_covered_count"]) == human_expected,
            "independent_anonymous": human["independent_annotators"] is True and human["anonymous_annotators"] is True,
            "adjudication_validated": human["gate_state"] == "validated" and human["adjudication_validated"] is True,
            "no_unresolved_disagreement": int(human["unresolved_disagreement_count"]) == 0,
            "real_human_only": human["synthetic_only"] is False and human["label_origin"] == "human",
        },
        "official_scorer": {
            "official_identity": scorer["package_name"] != "unknown" and scorer["package_version"] != "unknown",
            "package_hash": _is_digest(scorer["package_sha256"]),
            "schemas_provided": scorer["input_schema"] != "not_provided" and scorer["output_schema"] != "not_provided",
            "metric_namespace_provided": scorer["metric_namespace"] not in {"unknown", "not_provided", "synthetic_handoff"},
            "package_verified": scorer["package_verified"] is True,
            "complete_input": int(scorer["complete_input_query_count"]) == 1000,
            "sandbox_and_output_verified": scorer["sandbox_verified"] is True and scorer["output_verified"] is True,
            "not_synthetic": scorer["synthetic_only"] is False,
        },
    }


def evaluate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    try:
        _validate_evidence_shape(evidence)
        predicates = _predicates(evidence)
    except (ClearanceError, TypeError, ValueError, KeyError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "invalid",
            "exit_code": EXIT_VIOLATION,
            "error_code": str(exc),
            "formal_validation_complete": False,
            "execution": EXECUTION,
        }
    blocker_rows: dict[str, Any] = {}
    eligible_count = 0
    any_progress = False
    for blocker in BLOCKERS:
        values = predicates[blocker]
        passed = sorted(key for key, value in values.items() if value)
        failed = sorted(key for key, value in values.items() if not value)
        if not failed:
            state = "eligible_for_clearance"
            eligible_count += 1
        elif passed:
            state = "partially_satisfied"
            any_progress = True
        else:
            state = "blocked"
        blocker_rows[blocker] = {"state": state, "passed": passed, "failed": failed}
    global_values = evidence["global_prerequisites"]
    globals_passed = {
        "fresh": global_values["freshness_status"] == "fresh_with_declared_blockers" and int(global_values["stale_count"]) == 0,
        "dependency_closure": int(global_values["dependency_violation_count"]) == 0,
        "source_commit_compatible": global_values["source_commit_compatible"] is True,
        "current_rules_default": global_values["current_rules_default"] is True,
        "default_tiebreak_unchanged": global_values["deterministic_tiebreak_v2_default"] is False,
    }
    eligible = eligible_count == len(BLOCKERS) and all(globals_passed.values())
    status = "eligible_for_clearance" if eligible else ("partially_satisfied" if any_progress or eligible_count else "blocked")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": EXIT_VALID if eligible else EXIT_BLOCKED,
        "blockers": blocker_rows,
        "global_prerequisites": {
            "passed": sorted(key for key, value in globals_passed.items() if value),
            "failed": sorted(key for key, value in globals_passed.items() if not value),
        },
        "evidence_sha256": evidence["evidence_sha256"],
        "evidence_artifact_count": len(evidence["artifacts"]),
        "artifact_bindings_sha256": stable_hash(evidence["artifacts"]),
        "source_commit": evidence["source_commit"],
        "synthetic_test_only": evidence["synthetic_test_only"],
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
    report["state_summary_sha256"] = stable_hash(report)
    return report


def issue_receipt(evidence: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    report = evaluate(evidence)
    if report["status"] == "invalid":
        raise ClearanceError(str(report.get("error_code") or "invalid_evidence"))
    if report["status"] != "eligible_for_clearance":
        raise ClearanceBlocked("external_blockers_not_eligible")
    if evidence["source_commit"] != protocol["source_commit"]:
        raise ClearanceError("evidence_protocol_commit_mismatch")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol": RECEIPT_PROTOCOL,
        "clearance_protocol": PROTOCOL,
        "source_commit": protocol["source_commit"],
        "status": "cleared",
        "synthetic_test_only": bool(evidence["synthetic_test_only"]),
        "formal_validation_complete": not bool(evidence["synthetic_test_only"]),
        "evidence_sha256": evidence["evidence_sha256"],
        "state_summary_sha256": report["state_summary_sha256"],
        "blocker_evidence_sha256": {
            blocker: stable_hash(evidence["blockers"][blocker]) for blocker in BLOCKERS
        },
        "verification_commands": list(protocol["verification_commands"]),
        "signature": "none_self_hashed_receipt_no_private_key",
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def verify_receipt(
    receipt: Mapping[str, Any], evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        _require_keys(
            receipt,
            {"blocker_evidence_sha256", "clearance_protocol", "evidence_sha256", "formal_validation_complete", "protocol", "receipt_sha256", "schema_version", "signature", "source_commit", "state_summary_sha256", "status", "synthetic_test_only", "verification_commands"},
            "$receipt",
        )
        content = dict(receipt)
        claimed = content.pop("receipt_sha256")
        if not _is_digest(claimed) or stable_hash(content) != claimed:
            raise ClearanceError("receipt_hash_mismatch")
        expected = issue_receipt(evidence, protocol)
        if receipt != expected:
            raise ClearanceError("receipt_evidence_or_version_mismatch")
        if receipt["status"] not in STATES or receipt["status"] != "cleared":
            raise ClearanceError("receipt_state_invalid")
    except (ClearanceError, ClearanceBlocked, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "invalid",
            "exit_code": EXIT_VIOLATION,
            "error_code": str(exc),
            "formal_validation_complete": False,
            "execution": EXECUTION,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "cleared",
        "exit_code": EXIT_VALID,
        "receipt_sha256": receipt["receipt_sha256"],
        "synthetic_test_only": receipt["synthetic_test_only"],
        "formal_validation_complete": receipt["formal_validation_complete"],
        "execution": EXECUTION,
    }


def conformance_evidence(*, satisfied: tuple[str, ...] = BLOCKERS) -> dict[str, Any]:
    """Build deterministic test-only evidence; never persist it as real evidence."""

    digest = "a" * 64
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "synthetic_test_only": True,
        "source_commit": "e1e2545cab9d6f2ecf0e95be157d4c1e71376ec8",
        "global_prerequisites": {
            "freshness_status": "fresh_with_declared_blockers",
            "stale_count": 0,
            "dependency_violation_count": 0,
            "source_commit_compatible": True,
            "current_rules_default": True,
            "deterministic_tiebreak_v2_default": False,
        },
        "blockers": {
            "full1000": {
                "plan_bound": True,
                "expected_query_count": 1000,
                "committed_query_count": 1000,
                "unique_query_count": 1000,
                "run_manifest_verified": True,
                "generation_chain_verified": True,
                "resource_ledger_verified": True,
                "aggregate_verified": True,
                "aggregate_query_count": 1000,
                "legacy_input": False,
                "synthetic_input": False,
                "partial_input": False,
            },
            "human_precision": {
                "package_bound": True,
                "expected_item_count": 471,
                "annotator_a_covered_count": 471,
                "annotator_b_covered_count": 471,
                "independent_annotators": True,
                "anonymous_annotators": True,
                "gate_state": "validated",
                "adjudication_validated": True,
                "unresolved_disagreement_count": 0,
                "synthetic_only": False,
                "label_origin": "human",
            },
            "official_scorer": {
                "package_name": "official-scorer-fixture",
                "package_version": "1",
                "package_sha256": digest,
                "input_schema": "provided",
                "output_schema": "provided",
                "metric_namespace": "official.fixture",
                "package_verified": True,
                "complete_input_query_count": 1000,
                "sandbox_verified": True,
                "output_verified": True,
                "synthetic_only": False,
            },
        },
        "artifacts": {"fixture": {"path": "synthetic/fixture.json", "sha256": digest, "size": 1}},
        "execution": EXECUTION,
    }
    if "full1000" not in satisfied:
        evidence["blockers"]["full1000"]["committed_query_count"] = 0
    if "human_precision" not in satisfied:
        evidence["blockers"]["human_precision"]["annotator_b_covered_count"] = 0
    if "official_scorer" not in satisfied:
        evidence["blockers"]["official_scorer"]["package_name"] = "unknown"
    evidence["evidence_sha256"] = stable_hash(evidence)
    return evidence
