"""Deterministic change-risk classification and validation-scope planning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.validation_evidence_freshness import (
    _semantic_digest as freshness_semantic_digest,
)


PROTOCOL = "change_risk_validation_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "9654075256e719dd448bf56f53fe6f64c10f542a"
EXIT_SATISFIED = 0
EXIT_VIOLATION = 2
EXIT_INCOMPLETE = 3
EXIT_USAGE = 4
ZERO_SHA256 = "0" * 64
HEX = frozenset("0123456789abcdef")
RISK_ORDER = {"low": 0, "targeted": 1, "frontend": 2, "high": 3}
SEMANTIC_SUFFIXES = {
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
IGNORED_EXISTING_PATHS = ("third_party/paper-qa",)
CONTROL_OUTPUT_PATHS = (
    "benchmark/change_risk_validation_v1_evidence/attestation.json",
    "benchmark/change_risk_validation_v1_evidence/audit.json",
    "benchmark/change_risk_validation_v1_evidence/plan.json",
)
ALWAYS_VALIDATIONS = (
    "deterministic_plan_double_run",
    "sensitive_scan",
    "git_diff_check",
    "head_upstream_check",
)
HIGH_TRIGGERS = (
    "high_risk_component",
    "targeted_check_failed",
    "validation_evidence_missing",
    "unregistered_semantic_change",
    "tested_commit_differs_from_final_head",
)
FRONTEND_PREFIXES = (
    "frontend/",
    "src/scholar_agent/app/api",
    "src/scholar_agent/app/schemas",
    "src/scholar_agent/models/",
)
LOW_PREFIXES = ("docs/",)
HIGH_PATHS = (
    "AGENTS.md",
    "pytest.ini",
    "requirements-dev.txt",
    "requirements.txt",
)
HIGH_PREFIXES = (
    "benchmark/",
    "src/scholar_agent/agents/",
    "src/scholar_agent/connectors/",
    "src/scholar_agent/services/",
)
TARGETED_COMPONENTS = {
    "completion_bias",
    "constraint_decision",
    "external_scorer",
    "human_annotation",
    "human_precision",
    "ranking_decision",
    "source_fusion",
    "source_reliability",
}
FRONTEND_COMPONENTS = {"frontend_reproducible_build"}
COMPONENT_TESTS = {
    "completion_bias": ["tests/test_completion_bias_audit.py"],
    "constraint_decision": ["tests/test_constraint_decision_audit.py"],
    "external_scorer": ["tests/test_external_scorer_handoff.py"],
    "frontend_reproducible_build": [
        "tests/test_frontend_reproducible_build.py"
    ],
    "human_annotation": ["tests/test_human_annotation_delivery.py"],
    "human_precision": ["tests/test_human_precision_adjudication.py"],
    "ranking_decision": ["tests/test_ranking_decision_audit.py"],
    "source_fusion": ["tests/test_source_fusion_ablation.py"],
    "source_reliability": ["tests/test_source_reliability_diagnostics.py"],
}
PLAN_KEYS = {
    "change_set_sha256",
    "changes",
    "component_coverage",
    "conditional_validations",
    "execution",
    "explicit_skips",
    "full_pytest_required",
    "overall_risk",
    "plan_sha256",
    "protocol",
    "required_validations",
    "risk_counts",
    "schema_version",
    "target",
}


class ValidationScopeError(RuntimeError):
    """Risk mapping, plan integrity, or execution evidence is invalid."""


class ValidationScopeIncomplete(ValidationScopeError):
    """Required validation evidence has not been supplied."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationScopeError("input_file_unavailable") from exc
    return digest.hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise ValidationScopeError("duplicate_json_key")
        value[key] = child
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise ValidationScopeError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValidationScopeError("nonfinite_json_number")
            ),
        )
    except ValidationScopeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationScopeError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ValidationScopeError("json_root_not_object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value))
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationScopeError("json_output_invalid") from exc


def protocol_template() -> dict[str, Any]:
    return {
        "always_validations": list(ALWAYS_VALIDATIONS),
        "component_risk": {
            "default_registered_component": "high",
            "frontend": sorted(FRONTEND_COMPONENTS),
            "targeted": sorted(TARGETED_COMPONENTS),
        },
        "component_tests": COMPONENT_TESTS,
        "control_output_paths": list(CONTROL_OUTPUT_PATHS),
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "formal_validation_complete": False,
        "frontend_prefixes": list(FRONTEND_PREFIXES),
        "full_pytest_triggers": list(HIGH_TRIGGERS),
        "high_paths": list(HIGH_PATHS),
        "high_prefixes": list(HIGH_PREFIXES),
        "ignored_existing_paths": list(IGNORED_EXISTING_PATHS),
        "low_prefixes": list(LOW_PREFIXES),
        "protocol": PROTOCOL,
        "protocol_sha256": ZERO_SHA256,
        "risk_levels": ["low", "targeted", "high", "frontend"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "unregistered_semantic_files": "fail_closed",
    }


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    expected = protocol_template()
    if set(value) != set(expected):
        raise ValidationScopeError("protocol_schema_invalid")
    digest = value.get("protocol_sha256")
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    expected_payload = dict(expected)
    expected_payload["protocol_sha256"] = ZERO_SHA256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or set(digest) - HEX
        or stable_hash(payload) != digest
        or payload != expected_payload
    ):
        raise ValidationScopeError("protocol_schema_invalid")
    return value


def _git(
    repository_root: Path,
    *arguments: str,
    allowed_codes: tuple[int, ...] = (0,),
) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        timeout=60,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if completed.returncode not in allowed_codes:
        raise ValidationScopeError("git_query_failed")
    return completed.stdout


def git_head(repository_root: Path) -> str:
    return _git(repository_root, "rev-parse", "HEAD").decode().strip()


def _normalize_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationScopeError("change_path_invalid")
    return path.as_posix()


def commit_changes(
    repository_root: Path, from_commit: str, to_commit: str
) -> list[dict[str, Any]]:
    raw = _git(
        repository_root,
        "diff",
        "--name-status",
        "-M",
        from_commit,
        to_commit,
    ).decode("utf-8")
    changes = _parse_name_status(raw)
    for row in changes:
        row["semantic_equivalent"] = _semantic_equivalent(
            repository_root,
            path=row["path"],
            from_ref=from_commit,
            to_ref=to_commit,
            status=row["status"],
        )
    return changes


def worktree_changes(repository_root: Path) -> list[dict[str, Any]]:
    tracked = _git(
        repository_root, "diff", "--name-status", "-M", "HEAD"
    ).decode("utf-8")
    changes = _parse_name_status(tracked)
    for row in changes:
        row["semantic_equivalent"] = _semantic_equivalent(
            repository_root,
            path=row["path"],
            from_ref="HEAD",
            to_ref=None,
            status=row["status"],
        )
    untracked = _git(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for raw_path in untracked.split(b"\0"):
        if raw_path:
            changes.append(
                {
                    "old_path": None,
                    "path": _normalize_path(raw_path.decode("utf-8")),
                    "semantic_equivalent": False,
                    "status": "A",
                }
            )
    status = _git(repository_root, "status", "--porcelain=v1").decode("utf-8")
    if "third_party/paper-qa" in status and not any(
        row["path"] == "third_party/paper-qa" for row in changes
    ):
        changes.append(
            {
                "old_path": None,
                "path": "third_party/paper-qa",
                "status": "m",
            }
        )
    return sorted(
        (
            row
            for row in changes
            if row["path"] not in CONTROL_OUTPUT_PATHS
        ),
        key=lambda row: (row["path"], row["status"]),
    )


def _parse_name_status(raw: str) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) == 3:
            changes.append(
                {
                    "old_path": _normalize_path(parts[1]),
                    "path": _normalize_path(parts[2]),
                    "status": "R",
                }
            )
        elif len(parts) == 2:
            changes.append(
                {
                    "old_path": None,
                    "path": _normalize_path(parts[1]),
                    "status": status[0],
                }
            )
        else:
            raise ValidationScopeError("git_diff_record_invalid")
    return sorted(changes, key=lambda row: (row["path"], row["status"]))


def _git_file(
    repository_root: Path, ref: str, path: str
) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        timeout=30,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    return completed.stdout if completed.returncode == 0 else None


def _semantic_equivalent(
    repository_root: Path,
    *,
    path: str,
    from_ref: str,
    to_ref: str | None,
    status: str,
) -> bool:
    if status != "M" or ".env" in PurePosixPath(path).parts:
        return False
    old = _git_file(repository_root, from_ref, path)
    candidate = repository_root / path
    new = (
        _git_file(repository_root, to_ref, path)
        if to_ref is not None
        else candidate.read_bytes() if candidate.is_file() else None
    )
    return (
        old is not None
        and new is not None
        and freshness_semantic_digest(path, old)
        == freshness_semantic_digest(path, new)
    )


def worktree_digest(repository_root: Path, changes: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for change in changes:
        path = str(change["path"])
        if path in IGNORED_EXISTING_PATHS:
            continue
        candidate = repository_root / path
        rows.append(
            {
                "old_path": change.get("old_path"),
                "path": path,
                "sha256": sha256_file(candidate) if candidate.is_file() else None,
                "status": change["status"],
            }
        )
    return stable_hash(rows)


def _component_index(freshness: Mapping[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    components = freshness.get("components")
    if not isinstance(components, list):
        raise ValidationScopeError("freshness_components_missing")
    for component in components:
        if (
            not isinstance(component, Mapping)
            or not isinstance(component.get("component_id"), str)
            or not isinstance(component.get("files"), list)
        ):
            raise ValidationScopeError("freshness_component_invalid")
        for path in component["files"]:
            if not isinstance(path, str):
                raise ValidationScopeError("freshness_component_invalid")
            index.setdefault(path, []).append(component["component_id"])
    return {key: sorted(set(value)) for key, value in index.items()}


def _risk_for_components(components: Sequence[str]) -> str:
    if any(value in FRONTEND_COMPONENTS for value in components):
        return "frontend"
    if components and all(value in TARGETED_COMPONENTS for value in components):
        return "targeted"
    return "high"


def classify_change(
    change: Mapping[str, Any],
    component_index: Mapping[str, list[str]],
) -> dict[str, Any]:
    path = str(change["path"])
    old_path = change.get("old_path")
    if any(
        ".env" in PurePosixPath(candidate).parts
        for candidate in (path, str(old_path) if old_path else "")
        if candidate
    ):
        raise ValidationScopeError("forbidden_env_path")
    if path in IGNORED_EXISTING_PATHS:
        return {
            **change,
            "components": [],
            "reason": "preserved_existing_submodule_state",
            "risk": "low",
            "validation_relevant": False,
        }
    paths = [path] + ([str(old_path)] if old_path else [])
    components = sorted(
        {
            component
            for candidate in paths
            for component in component_index.get(candidate, [])
        }
    )
    if change.get("semantic_equivalent") is True:
        risk, reason = "low", "freshness_semantic_digest_unchanged"
    elif any(candidate.startswith(FRONTEND_PREFIXES) for candidate in paths):
        risk, reason = "frontend", "frontend_or_api_contract_path"
    elif any(candidate.startswith(LOW_PREFIXES) for candidate in paths):
        risk, reason = "low", "documentation_path"
    elif path.startswith("tests/") and path.endswith(".py"):
        risk, reason = "targeted", "test_module_change"
    elif path in HIGH_PATHS or any(
        candidate.startswith(HIGH_PREFIXES) for candidate in paths
    ):
        risk, reason = "high", "global_or_runtime_path"
    elif components:
        risk, reason = _risk_for_components(components), "freshness_component_mapping"
    elif PurePosixPath(path).suffix.lower() in SEMANTIC_SUFFIXES:
        raise ValidationScopeError(f"unregistered_semantic_change:{path}")
    else:
        raise ValidationScopeError(f"unregistered_change:{path}")
    if change["status"] == "R" and not components:
        raise ValidationScopeError(f"unregistered_rename:{path}")
    return {
        **change,
        "components": components,
        "reason": reason,
        "risk": risk,
        "validation_relevant": True,
    }


def _validation(
    validation_id: str,
    command: str,
    reason: str,
    *,
    category: str,
    expected_exit_code: int = 0,
) -> dict[str, Any]:
    return {
        "category": category,
        "command": command,
        "expected_exit_code": expected_exit_code,
        "reason": reason,
        "validation_id": validation_id,
    }


def _gate_command(gate: Mapping[str, Any]) -> str | None:
    arguments = gate.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        return None
    if any("{temporary_output}" in str(value) for value in arguments):
        return None
    return "PYTHONPATH=src python " + shlex.join([str(value) for value in arguments])


def build_plan(
    *,
    changes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    freshness: Mapping[str, Any],
    readiness: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    if protocol.get("protocol") != PROTOCOL:
        raise ValidationScopeError("protocol_mismatch")
    component_index = _component_index(freshness)
    classified = sorted(
        (classify_change(change, component_index) for change in changes),
        key=lambda row: (
            str(row["path"]),
            str(row["status"]),
            str(row.get("old_path") or ""),
        ),
    )
    relevant = [row for row in classified if row["validation_relevant"]]
    risks = [str(row["risk"]) for row in relevant]
    overall = max(risks, key=lambda value: RISK_ORDER[value]) if risks else "low"
    changed_components = sorted(
        {component for row in relevant for component in row["components"]}
    )
    tests = {
        row["path"]
        for row in relevant
        if row["path"].startswith("tests/") and row["path"].endswith(".py")
    }
    for component in changed_components:
        tests.update(COMPONENT_TESTS.get(component, []))
    required: list[dict[str, Any]] = []
    for test in sorted(tests):
        required.append(
            _validation(
                f"pytest:{test}",
                f"PYTHONPATH=src pytest -q {shlex.quote(test)}",
                "changed_or_component_bound_test",
                category="targeted_pytest",
            )
        )
    gate_rows = freshness.get("gates")
    if gate_rows is None and isinstance(freshness.get("bindings"), Mapping):
        gate_rows = freshness["bindings"].get("gates")
    gate_components = {
        row["gate_id"]: set(row.get("components") or [])
        for row in gate_rows or []
        if isinstance(row, Mapping) and isinstance(row.get("gate_id"), str)
    }
    for gate in readiness.get("read_only_gates", []):
        gate_id = str(gate.get("gate_id"))
        if set(changed_components) & gate_components.get(gate_id, set()):
            command = _gate_command(gate)
            if command:
                expected_exit_code = gate.get("expected_exit_code")
                if (
                    not isinstance(expected_exit_code, int)
                    or isinstance(expected_exit_code, bool)
                ):
                    raise ValidationScopeError(
                        "readiness_gate_exit_code_missing"
                    )
                required.append(
                    _validation(
                        f"gate:{gate_id}",
                        command,
                        "freshness_component_bound_gate",
                        category="read_only_gate",
                        expected_exit_code=expected_exit_code,
                    )
                )
    required.append(
        _validation(
            "gate:validation_freshness",
            "PYTHONPATH=src python scripts/check_validation_freshness.py verify-current",
            "global_dependency_closure",
            category="read_only_gate",
        )
    )
    required.extend(
        [
            _validation(
                "deterministic_plan_double_run",
                "cmp PLAN_RUN_1 PLAN_RUN_2",
                "plan_output_must_be_byte_deterministic",
                category="determinism",
            ),
            _validation(
                "sensitive_scan",
                "git diff --no-ext-diff | sensitive-pattern-scan",
                "prevent_secret_or_machine_path_leakage",
                category="security",
            ),
            _validation(
                "git_diff_check",
                "git diff --check",
                "patch_whitespace_integrity",
                category="repository",
            ),
            _validation(
                "head_upstream_check",
                "git rev-list --left-right --count HEAD...@{upstream}",
                "repository_provenance",
                category="repository",
            ),
        ]
    )
    full_pytest = "high" in risks
    if full_pytest:
        required.append(
            _validation(
                "pytest:full",
                "PYTHONPATH=src pytest -q",
                "high_risk_component",
                category="full_pytest",
            )
        )
    frontend = "frontend" in risks
    if frontend:
        required.extend(
            [
                _validation(
                    "frontend:lint",
                    "cd frontend && npm run lint",
                    "frontend_or_api_contract_change",
                    category="frontend",
                ),
                _validation(
                    "frontend:build",
                    "cd frontend && npm run build",
                    "frontend_or_api_contract_change",
                    category="frontend",
                ),
            ]
        )
    required = sorted(
        {row["validation_id"]: row for row in required}.values(),
        key=lambda row: row["validation_id"],
    )
    conditional = []
    if not full_pytest:
        conditional.append(
            {
                "condition": list(HIGH_TRIGGERS),
                "validation_id": "pytest:full",
            }
        )
    skips = []
    if not frontend:
        skips.extend(
            [
                {
                    "reason": "no_frontend_api_schema_or_build_change",
                    "validation_id": "frontend:lint",
                },
                {
                    "reason": "no_frontend_api_schema_or_build_change",
                    "validation_id": "frontend:build",
                },
            ]
        )
    plan = {
        "change_set_sha256": stable_hash(classified),
        "changes": classified,
        "component_coverage": {
            "changed_component_count": len(changed_components),
            "freshness_component_count": len(freshness["components"]),
            "mapped_change_count": len(relevant),
            "unregistered_semantic_count": 0,
        },
        "conditional_validations": conditional,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "explicit_skips": skips,
        "full_pytest_required": full_pytest,
        "overall_risk": overall,
        "plan_sha256": ZERO_SHA256,
        "protocol": PROTOCOL,
        "required_validations": required,
        "risk_counts": {
            risk: sum(row["risk"] == risk for row in relevant)
            for risk in ("low", "targeted", "high", "frontend")
        },
        "schema_version": SCHEMA_VERSION,
        "target": dict(target),
    }
    payload = dict(plan)
    payload["plan_sha256"] = ZERO_SHA256
    plan["plan_sha256"] = stable_hash(payload)
    return plan


def verify_plan(value: Mapping[str, Any]) -> None:
    if set(value) != PLAN_KEYS or value.get("protocol") != PROTOCOL:
        raise ValidationScopeError("plan_schema_invalid")
    digest = value.get("plan_sha256")
    payload = dict(value)
    payload["plan_sha256"] = ZERO_SHA256
    if (
        not isinstance(digest, str)
        or stable_hash(payload) != digest
        or not isinstance(value.get("required_validations"), list)
    ):
        raise ValidationScopeError("plan_integrity_invalid")


def verify_execution(
    plan: Mapping[str, Any], attestation: Mapping[str, Any]
) -> dict[str, Any]:
    verify_plan(plan)
    required_attestation_keys = {
        "attestation_sha256",
        "change_set_sha256",
        "executed_head",
        "executions",
        "final_head",
        "formal_validation_complete",
        "plan_sha256",
        "protocol",
        "schema_version",
        "status",
        "tested_worktree_sha256",
    }
    if set(attestation) != required_attestation_keys:
        raise ValidationScopeError("attestation_schema_invalid")
    payload = dict(attestation)
    digest = payload["attestation_sha256"]
    payload["attestation_sha256"] = ZERO_SHA256
    if (
        attestation["protocol"] != PROTOCOL
        or attestation["schema_version"] != SCHEMA_VERSION
        or attestation["plan_sha256"] != plan["plan_sha256"]
        or attestation["change_set_sha256"] != plan["change_set_sha256"]
        or attestation["formal_validation_complete"] is not False
        or stable_hash(payload) != digest
    ):
        raise ValidationScopeError("attestation_integrity_invalid")
    executions = attestation["executions"]
    if not isinstance(executions, list):
        raise ValidationScopeError("attestation_execution_invalid")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in executions:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"command", "exit_code", "output_sha256", "validation_id"}
            or row["validation_id"] in by_id
        ):
            raise ValidationScopeError("attestation_execution_invalid")
        by_id[str(row["validation_id"])] = row
    required = {
        str(row["validation_id"]): row for row in plan["required_validations"]
    }
    if set(by_id) != set(required):
        raise ValidationScopeIncomplete("validation_execution_incomplete")
    for validation_id, expected in required.items():
        actual = by_id[validation_id]
        if (
            actual["command"] != expected["command"]
            or actual["exit_code"] != expected["expected_exit_code"]
            or not isinstance(actual["output_sha256"], str)
            or len(actual["output_sha256"]) != 64
        ):
            raise ValidationScopeError("validation_execution_failed")
    target = plan["target"]
    if target["mode"] == "commit":
        if attestation["executed_head"] != target["target_commit"]:
            raise ValidationScopeError("tested_commit_mismatch")
    else:
        if attestation["tested_worktree_sha256"] != target["worktree_sha256"]:
            raise ValidationScopeError("tested_worktree_mismatch")
        if (
            attestation["executed_head"] != attestation["final_head"]
            and "pytest:full" not in by_id
        ):
            raise ValidationScopeIncomplete("final_head_drift_requires_full_pytest")
    return {
        "attestation_sha256": digest,
        "executed_validation_count": len(executions),
        "exit_code": EXIT_SATISFIED,
        "formal_validation_complete": False,
        "plan_sha256": plan["plan_sha256"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "validation_scope_satisfied",
    }


def build_attestation(
    plan: Mapping[str, Any],
    *,
    executions: Sequence[Mapping[str, Any]],
    executed_head: str,
    final_head: str,
    tested_worktree_sha256: str,
) -> dict[str, Any]:
    value = {
        "attestation_sha256": ZERO_SHA256,
        "change_set_sha256": plan["change_set_sha256"],
        "executed_head": executed_head,
        "executions": [dict(row) for row in executions],
        "final_head": final_head,
        "formal_validation_complete": False,
        "plan_sha256": plan["plan_sha256"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "validation_scope_satisfied",
        "tested_worktree_sha256": tested_worktree_sha256,
    }
    payload = dict(value)
    payload["attestation_sha256"] = ZERO_SHA256
    value["attestation_sha256"] = stable_hash(payload)
    return value


def audit_current(
    protocol: Mapping[str, Any],
    freshness: Mapping[str, Any],
) -> dict[str, Any]:
    if protocol.get("protocol") != PROTOCOL:
        raise ValidationScopeError("protocol_mismatch")
    components = freshness.get("components")
    if not isinstance(components, list) or not components:
        raise ValidationScopeError("freshness_components_missing")
    ids = [row.get("component_id") for row in components if isinstance(row, Mapping)]
    if len(ids) != len(components) or len(set(ids)) != len(ids):
        raise ValidationScopeError("freshness_component_inventory_invalid")
    return {
        "component_mapping_coverage": 1.0,
        "exit_code": EXIT_SATISFIED,
        "formal_validation_complete": False,
        "freshness_component_count": len(components),
        "ignored_existing_paths": list(IGNORED_EXISTING_PATHS),
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "validation_scope_satisfied",
    }
