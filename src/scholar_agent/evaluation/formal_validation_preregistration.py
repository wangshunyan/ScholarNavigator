"""Immutable preregistration gate for future formal validation evidence.

The gate freezes analysis inputs and decision rules before any real Full1000,
human-label, or official-scorer evidence exists.  It does not execute
retrieval, read labels, or compute a quality metric.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "formal_validation_preregistration_v1"
SEAL_CONTRACT = "formal_validation_preregistration_seal_v1"
SCHEMA_VERSION = "1"
EXIT_SEALED = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
AMENDMENT_STATES = (
    "sealed",
    "amended_before_evidence",
    "invalid_post_evidence_change",
)
EXTERNAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
_HEX = frozenset("0123456789abcdef")
_FROZEN_POLICY_KEYS = (
    "analysis",
    "amendments",
    "declaration_boundaries",
    "execution",
    "formal_validation_complete",
    "human_annotation",
    "official_scorer",
    "population",
    "protocol",
    "schema_version",
    "source_commit",
    "statistics",
    "stopping_rules",
)


class PreregistrationError(RuntimeError):
    """A seal, protocol, chronology, or amendment invariant was violated."""


class PreregistrationBlocked(PreregistrationError):
    """The plan is sealed but real external evidence is still unavailable."""


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


def _strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise PreregistrationError("duplicate_json_key")
        value[key] = child
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                PreregistrationError("nonfinite_json_number")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PreregistrationError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise PreregistrationError("json_root_not_object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(canonical_json(value))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise PreregistrationError("json_output_unavailable") from exc


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _HEX


def _safe_repo_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise PreregistrationError("unsafe_registered_path")
    if value.parts[0] == "third_party" or value.name == ".env":
        raise PreregistrationError("prohibited_registered_path")
    path = (root / Path(*value.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PreregistrationError("registered_path_escape") from exc
    return path


def _string_list(value: Any, *, reason: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise PreregistrationError(reason)
    rows = [str(item) for item in value]
    if rows != sorted(set(rows)) or (not allow_empty and not rows):
        raise PreregistrationError(reason)
    return rows


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        *_FROZEN_POLICY_KEYS,
        "allowed_outputs",
        "analysis_scripts",
        "dependencies",
        "protocol_sha256",
    }
    if set(value) != required:
        raise PreregistrationError("protocol_schema_invalid")
    if (
        value.get("protocol") != PROTOCOL
        or value.get("schema_version") != SCHEMA_VERSION
        or not _is_commit(value.get("source_commit"))
        or value.get("formal_validation_complete") is not False
        or value.get("execution") != EXECUTION
    ):
        raise PreregistrationError("protocol_identity_or_boundary_invalid")
    if value.get("amendments", {}).get("states") != list(AMENDMENT_STATES):
        raise PreregistrationError("amendment_state_inventory_drift")
    population = value.get("population")
    if not isinstance(population, Mapping) or population.get("query_count") != 1000:
        raise PreregistrationError("population_contract_invalid")
    for key in ("input_sha256", "order_sha256", "stable_identity_sha256"):
        if not _is_digest(population.get(key)):
            raise PreregistrationError("population_digest_invalid")
    analysis = value.get("analysis")
    if not isinstance(analysis, Mapping):
        raise PreregistrationError("analysis_contract_invalid")
    if (
        analysis.get("strategy") != "current_rules"
        or analysis.get("full_rerun_required") is not True
        or analysis.get("deterministic_tiebreak_v2_enabled") is not False
        or analysis.get("sources") != ["arxiv", "openalex", "pubmed", "semantic_scholar"]
    ):
        raise PreregistrationError("analysis_policy_drift")
    human = value.get("human_annotation")
    if not isinstance(human, Mapping) or human.get("expected_item_count") != 471:
        raise PreregistrationError("human_contract_invalid")
    if human.get("independent_annotator_count") != 2:
        raise PreregistrationError("human_independence_drift")
    if human.get("labels") != [
        "insufficient_information",
        "not_relevant",
        "partially_relevant",
        "relevant",
    ]:
        raise PreregistrationError("human_label_inventory_drift")
    scorer = value.get("official_scorer")
    if not isinstance(scorer, Mapping) or any(
        scorer.get(key) not in {"unknown", "not_provided"}
        for key in ("name", "version", "input_schema", "output_schema", "metric_namespace", "direction")
    ):
        raise PreregistrationError("official_scorer_slot_must_remain_unknown")
    statistics = value.get("statistics")
    if not isinstance(statistics, Mapping):
        raise PreregistrationError("statistics_contract_invalid")
    if (
        statistics.get("analysis_population") != "change_only_paired_items"
        or statistics.get("resampling_unit") != "frozen_query_connected_component"
        or statistics.get("bootstrap_iterations") != 20000
        or statistics.get("confidence_level") != 0.95
        or statistics.get("multiple_comparison_correction") != "holm_bonferroni_fixed_family"
    ):
        raise PreregistrationError("statistics_policy_drift")
    for key in ("stopping_rules", "declaration_boundaries"):
        if not isinstance(value.get(key), Mapping):
            raise PreregistrationError(f"{key}_invalid")
    dependencies = value.get("dependencies")
    scripts = value.get("analysis_scripts")
    if not isinstance(dependencies, list) or not dependencies:
        raise PreregistrationError("dependencies_invalid")
    if not isinstance(scripts, list) or not scripts:
        raise PreregistrationError("analysis_scripts_invalid")
    for inventory, reason in (
        (dependencies, "dependency_entry_invalid"),
        (scripts, "analysis_script_entry_invalid"),
    ):
        seen: set[str] = set()
        for row in inventory:
            if not isinstance(row, Mapping) or set(row) != {"path", "role", "sha256"}:
                raise PreregistrationError(reason)
            path = row.get("path")
            if not isinstance(path, str) or path in seen or not _is_digest(row.get("sha256")):
                raise PreregistrationError(reason)
            seen.add(path)
            if not isinstance(row.get("role"), str) or not row["role"]:
                raise PreregistrationError(reason)
    _string_list(value.get("allowed_outputs"), reason="allowed_outputs_invalid")
    claimed = value.get("protocol_sha256")
    if not _is_digest(claimed):
        raise PreregistrationError("protocol_digest_invalid")
    payload = dict(value)
    payload["protocol_sha256"] = "0" * 64
    if stable_hash(payload) != claimed:
        raise PreregistrationError("protocol_digest_mismatch")
    return dict(value)


def load_protocol(path: Path) -> dict[str, Any]:
    return validate_protocol(read_json(path))


def verify_registered_files(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for group in ("dependencies", "analysis_scripts"):
        for row in protocol[group]:
            path = _safe_repo_path(repository_root, str(row["path"]))
            if not path.is_file():
                raise PreregistrationError(f"registered_file_missing:{row['role']}")
            actual = sha256_file(path)
            if actual != row["sha256"]:
                raise PreregistrationError(f"registered_file_hash_drift:{row['role']}")
            verified.append(
                {
                    "path": row["path"],
                    "role": row["role"],
                    "sha256": actual,
                    "size": path.stat().st_size,
                }
            )
    return sorted(verified, key=lambda row: (str(row["role"]), str(row["path"])))


def build_seal(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    protocol = validate_protocol(protocol)
    verified = verify_registered_files(protocol, repository_root=repository_root)
    seal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SEAL_CONTRACT,
        "protocol": PROTOCOL,
        "source_commit": protocol["source_commit"],
        "protocol_sha256": protocol["protocol_sha256"],
        "state": "sealed",
        "formal_validation_complete": False,
        "registered_files": verified,
        "registered_files_sha256": stable_hash(verified),
        "allowed_outputs": list(protocol["allowed_outputs"]),
        "semantic_policy_sha256": stable_hash(
            {key: protocol[key] for key in _FROZEN_POLICY_KEYS}
        ),
        "external_evidence": {
            "full1000": "not_available",
            "human_precision": "not_available",
            "official_scorer": "not_provided",
        },
        "blockers": list(EXTERNAL_BLOCKERS),
        "seal_sha256": "0" * 64,
    }
    seal["seal_sha256"] = stable_hash(seal)
    return seal


def verify_seal(
    seal: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    expected = build_seal(protocol, repository_root=repository_root)
    if dict(seal) != expected:
        raise PreregistrationError("seal_content_or_dependency_drift")
    if not _git_ancestor(repository_root, str(seal["source_commit"]), "HEAD"):
        raise PreregistrationError("seal_source_commit_not_ancestor")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "preregistration_sealed",
        "exit_code": EXIT_SEALED,
        "seal_sha256": seal["seal_sha256"],
        "registered_file_count": len(seal["registered_files"]),
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def _git_ancestor(root: Path, older: str, newer: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return completed.returncode == 0


def evaluate_timeline(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("preregistration_sealed", "execution_started", "evidence_intake", "unblind_or_score")
    positions: dict[str, int] = {}
    for index, row in enumerate(events):
        event = row.get("event")
        if event not in required or event in positions:
            raise PreregistrationError("timeline_event_invalid")
        positions[str(event)] = index
    missing = [item for item in required if item not in positions]
    if missing:
        raise PreregistrationError("timeline_event_missing")
    if not (
        positions["preregistration_sealed"] < positions["evidence_intake"]
        and positions["execution_started"] < positions["unblind_or_score"]
        and positions["evidence_intake"] <= positions["unblind_or_score"]
    ):
        raise PreregistrationError("chronology_violation")
    return {"chronology_valid": True, "event_count": len(events)}


def evaluate_amendment(
    *,
    changed_pointers: Sequence[str],
    evidence_intake_present: bool,
    semantic_digest_before: str,
    semantic_digest_after: str,
    declared_nonsemantic: bool = False,
) -> dict[str, Any]:
    if (
        not changed_pointers
        or list(changed_pointers) != sorted(set(changed_pointers))
        or any(not value.startswith("/") for value in changed_pointers)
    ):
        raise PreregistrationError("amendment_pointer_set_invalid")
    semantic_changed = semantic_digest_before != semantic_digest_after
    nonsemantic_paths = all(value.startswith("/documentation/") for value in changed_pointers)
    if declared_nonsemantic and (semantic_changed or not nonsemantic_paths):
        raise PreregistrationError("nonsemantic_erratum_not_proven")
    if evidence_intake_present and semantic_changed:
        state = "invalid_post_evidence_change"
        valid = False
    elif evidence_intake_present:
        state = "sealed"
        valid = True
    else:
        state = "amended_before_evidence"
        valid = True
    return {
        "state": state,
        "valid": valid,
        "evidence_intake_present": evidence_intake_present,
        "semantic_changed": semantic_changed,
        "changed_pointers": list(changed_pointers),
    }


def synthetic_amendment_matrix() -> dict[str, Any]:
    before = stable_hash({"threshold": 1, "metric": "fixed"})
    changed = stable_hash({"threshold": 2, "metric": "fixed"})
    scenarios = [
        ("before_evidence_amendment", False, ["/statistics/report_precision"], before, changed, False, "amended_before_evidence"),
        ("post_evidence_threshold_change", True, ["/human_annotation/coverage_threshold"], before, changed, False, "invalid_post_evidence_change"),
        ("posthoc_exclusion", True, ["/population/exclusion_rules"], before, changed, False, "invalid_post_evidence_change"),
        ("statistics_method_replacement", True, ["/statistics/resampling_unit"], before, changed, False, "invalid_post_evidence_change"),
        ("invented_official_metric", True, ["/official_scorer/metric_namespace"], before, changed, False, "invalid_post_evidence_change"),
        ("nonsemantic_erratum", True, ["/documentation/typo"], before, before, True, "sealed"),
    ]
    rows: list[dict[str, Any]] = [
        {
            "scenario": "legal_seal",
            "state": "sealed",
            "valid": True,
            "evidence_intake_present": False,
            "semantic_changed": False,
            "changed_pointers": [],
        }
    ]
    for name, intake, pointers, first, second, nonsemantic, expected in scenarios:
        result = evaluate_amendment(
            changed_pointers=pointers,
            evidence_intake_present=intake,
            semantic_digest_before=first,
            semantic_digest_after=second,
            declared_nonsemantic=nonsemantic,
        )
        if result["state"] != expected:
            raise PreregistrationError("synthetic_amendment_expectation_failed")
        rows.append({"scenario": name, **result})
    valid_timeline = evaluate_timeline(
        [
            {"event": "preregistration_sealed"},
            {"event": "execution_started"},
            {"event": "evidence_intake"},
            {"event": "unblind_or_score"},
        ]
    )
    invalid_timeline_error = ""
    try:
        evaluate_timeline(
            [
                {"event": "evidence_intake"},
                {"event": "preregistration_sealed"},
                {"event": "execution_started"},
                {"event": "unblind_or_score"},
            ]
        )
    except PreregistrationError as exc:
        invalid_timeline_error = str(exc)
    if invalid_timeline_error != "chronology_violation":
        raise PreregistrationError("synthetic_timeline_expectation_failed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "preregistration_sealed",
        "exit_code": EXIT_SEALED,
        "scenario_count": len(rows),
        "scenarios": rows,
        "timeline_scenario_count": 2,
        "timeline_scenarios": [
            {"scenario": "valid_preregistration_timeline", **valid_timeline},
            {
                "scenario": "evidence_before_seal",
                "chronology_valid": False,
                "error_code": invalid_timeline_error,
                "event_count": 4,
            },
        ],
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def audit_readiness(
    protocol: Mapping[str, Any],
    seal: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    verified = verify_seal(seal, protocol, repository_root=repository_root)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "sealed_with_external_evidence_blockers",
        "exit_code": EXIT_BLOCKED,
        "seal_sha256": verified["seal_sha256"],
        "blockers": list(EXTERNAL_BLOCKERS),
        "blocker_count": len(EXTERNAL_BLOCKERS),
        "preregistration_state": "sealed",
        "official_scorer_schema": "not_provided",
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def assert_current_preregistration(
    repository_root: Path,
    *,
    protocol_path: str = "benchmark/formal_validation_preregistration_v1_protocol.json",
    seal_path: str = "benchmark/formal_validation_preregistration_v1_seal.json",
) -> None:
    protocol = load_protocol(_safe_repo_path(repository_root, protocol_path))
    seal = read_json(_safe_repo_path(repository_root, seal_path))
    verify_seal(seal, protocol, repository_root=repository_root)
