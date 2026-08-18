"""Deterministic evidence freshness and change-impact gate.

The gate never recomputes benchmark quality.  It binds the existing readiness
inventory to precise semantic components and propagates file changes through
evidence, read-only gates, and published claims.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal


SCHEMA_VERSION = "1"
PROTOCOL_VERSION = "validation_evidence_freshness_v1"
EXIT_FRESH = 0
EXIT_STALE = 2
EXIT_MISSING = 3
EXIT_USAGE = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_STATES = {"fresh", "stale", "blocked", "not_applicable"}
SEMANTIC_ROOTS = (
    "src/",
    "scripts/",
    "frontend/src/",
    "benchmark/",
    "docs/",
    "README.md",
)


class FreshnessError(RuntimeError):
    """The dependency contract or observed state is invalid."""


class FreshnessBaselineMissing(FreshnessError):
    """A required baseline input is unavailable."""


class ChangeRecord(dict[str, Any]):
    """Normalized changed-path record used by impact analysis."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise FreshnessError("unsafe_dependency_path")
    if posix.parts[0] == "third_party" or posix.name == ".env":
        raise FreshnessError("prohibited_dependency_path")
    resolved = (root / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FreshnessError("dependency_path_escape") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreshnessBaselineMissing("required_json_unavailable") from exc
    if not isinstance(value, dict):
        raise FreshnessError("json_root_not_object")
    return value


def load_contract(path: str | Path, *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = repository_root.resolve()
    value = _read_json(Path(path))
    if value.get("schema_version") != SCHEMA_VERSION or value.get("protocol") != PROTOCOL_VERSION:
        raise FreshnessError("contract_version_invalid")
    if value.get("execution") != {
        "gold_or_qrels_loaded": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise FreshnessError("offline_execution_contract_drift")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise FreshnessError("components_missing")
    component_ids = [str(item.get("component_id") or "") for item in components]
    if any(not item for item in component_ids) or len(component_ids) != len(set(component_ids)):
        raise FreshnessError("component_identity_invalid")
    registered_paths: set[str] = set()
    for component in components:
        files = component.get("files")
        if not isinstance(files, list) or not files or files != sorted(set(files)):
            raise FreshnessError("component_files_invalid")
        for relative in files:
            _safe_path(root, str(relative))
            registered_paths.add(str(relative))
        if not _is_sha256(component.get("basis_digest")):
            raise FreshnessError("component_basis_digest_invalid")
        if not _is_commit(component.get("implementation_commit")):
            raise FreshnessError("component_implementation_commit_invalid")
    bindings = value.get("bindings") or {}
    for name in ("evidence", "gates", "claims"):
        rows = bindings.get(name)
        if not isinstance(rows, list):
            raise FreshnessError(f"{name}_bindings_missing")
        identity_key = {"evidence": "evidence_id", "gates": "gate_id", "claims": "claim_id"}[name]
        ids = [str(row.get(identity_key) or "") for row in rows]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise FreshnessError(f"{name}_binding_identity_invalid")
        for row in rows:
            dependencies = row.get("components") or []
            if not dependencies or any(item not in component_ids for item in dependencies):
                raise FreshnessError(f"{name}_component_reference_invalid")
            if not _is_sha256(row.get("evidence_basis_digest") if name == "evidence" else row.get("basis_digest")):
                raise FreshnessError(f"{name}_basis_digest_invalid")
    _validate_dependency_graph(bindings["evidence"])
    _validate_inventory(value, root)
    return value


def component_digest(component: Mapping[str, Any], root: Path) -> str:
    entries: list[dict[str, Any]] = []
    for relative in component["files"]:
        path = _safe_path(root, str(relative))
        if not path.is_file():
            raise FreshnessBaselineMissing(f"dependency_file_missing:{relative}")
        entries.append({"path": str(relative), "sha256": sha256_file(path), "size": path.stat().st_size})
    return stable_hash(entries)


def evidence_basis_digest(binding: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]]) -> str:
    return stable_hash(
        {
            "artifact_sha256": binding["artifact_sha256"],
            "baseline_commit": binding["baseline_commit"],
            "components": [
                {"component_id": item, "basis_digest": components[item]["basis_digest"]}
                for item in sorted(binding["components"])
            ],
            "depends_on_evidence": sorted(binding.get("depends_on_evidence") or []),
        }
    )


def entity_basis_digest(binding: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]]) -> str:
    payload: dict[str, Any] = {
        "components": [
            {"component_id": item, "basis_digest": components[item]["basis_digest"]}
            for item in sorted(binding["components"])
        ],
        "evidence_ids": sorted(binding.get("evidence_ids") or []),
    }
    if binding.get("source_document"):
        payload["source_document"] = binding["source_document"]
        payload["source_sha256"] = binding["source_sha256"]
    if binding.get("declared_status"):
        payload["declared_status"] = binding["declared_status"]
    return stable_hash(payload)


def verify_current(contract: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = repository_root.resolve()
    violations: list[dict[str, Any]] = []
    components = {str(item["component_id"]): item for item in contract["components"]}
    observed_components: dict[str, str | None] = {}
    for component_id, component in sorted(components.items()):
        try:
            observed = component_digest(component, root)
        except FreshnessBaselineMissing:
            observed = None
        observed_components[component_id] = observed
        if observed != component["basis_digest"]:
            violations.append(_violation("component_basis_drift", component_id, component["basis_digest"], observed))
        if not _is_ancestor(root, str(component["implementation_commit"]), str(contract["baseline"]["head"])):
            violations.append(_violation("implementation_commit_after_baseline", component_id, "ancestor", component["implementation_commit"]))
    evidence_rows = {str(item["evidence_id"]): item for item in contract["bindings"]["evidence"]}
    evidence_states: list[dict[str, Any]] = []
    for evidence_id in _topological_evidence(evidence_rows):
        binding = evidence_rows[evidence_id]
        path = _safe_path(root, str(binding["artifact_path"]))
        actual = sha256_file(path) if path.is_file() else None
        stale_dependencies = [item for item in binding["components"] if observed_components[item] != components[item]["basis_digest"]]
        upstream_stale = [
            item for item in binding.get("depends_on_evidence") or []
            if next(row for row in evidence_states if row["evidence_id"] == item)["state"] == "stale"
        ]
        basis_actual = evidence_basis_digest(binding, components)
        state = str(binding["declared_state"])
        reasons: list[str] = []
        if actual != binding["artifact_sha256"]:
            state = "stale"; reasons.append("artifact_hash_drift")
        if stale_dependencies:
            state = "stale"; reasons.append("semantic_dependency_drift")
        if upstream_stale:
            state = "stale"; reasons.append("upstream_evidence_stale")
        if basis_actual != binding["evidence_basis_digest"]:
            state = "stale"; reasons.append("basis_contract_drift")
        if not _is_ancestor(root, str(binding["artifact_commit"]), str(binding["baseline_commit"])):
            state = "stale"; reasons.append("evidence_baseline_predates_artifact")
        evidence_states.append({
            "evidence_id": evidence_id,
            "state": state,
            "declared_state": binding["declared_state"],
            "stale_components": sorted(stale_dependencies),
            "stale_upstream_evidence": sorted(upstream_stale),
            "reasons": sorted(reasons),
        })
    evidence_state_map = {row["evidence_id"]: row for row in evidence_states}
    gate_states = _current_entity_states(contract["bindings"]["gates"], components, observed_components, evidence_state_map, "gate_id")
    claim_states = _current_entity_states(contract["bindings"]["claims"], components, observed_components, evidence_state_map, "claim_id", root=root)
    stale = [row for row in evidence_states + gate_states + claim_states if row["state"] == "stale"]
    if stale:
        violations.append(_violation("stale_entities_present", "$", 0, len(stale)))
    report = _report_base(contract)
    report.update({
        "status": "fresh_with_declared_blockers" if not violations else "stale_or_dependency_violation",
        "exit_code": EXIT_FRESH if not violations else EXIT_STALE,
        "component_count": len(components),
        "evidence": sorted(evidence_states, key=lambda row: row["evidence_id"]),
        "gates": sorted(gate_states, key=lambda row: row["gate_id"]),
        "claims": sorted(claim_states, key=lambda row: row["claim_id"]),
        "state_counts": _state_counts(evidence_states + gate_states + claim_states),
        "minimum_rerun_gate_ids": sorted({gate for row in evidence_states if row["state"] == "stale" for gate in evidence_rows[row["evidence_id"]].get("rerun_gate_ids") or []}),
        "violations": violations,
        "violation_count": len(violations),
    })
    return report


def impact_analysis(contract: Mapping[str, Any], changes: Sequence[Mapping[str, Any]], *, repository_root: Path = REPOSITORY_ROOT, from_ref: str | None = None, to_ref: str | None = None, mode: str = "synthetic") -> dict[str, Any]:
    root = repository_root.resolve()
    components = {str(item["component_id"]): item for item in contract["components"]}
    path_components: dict[str, set[str]] = defaultdict(set)
    for component_id, component in components.items():
        for relative in component["files"]:
            path_components[str(relative)].add(component_id)
    registered_claim_sources = {
        str(item.get("source_document"))
        for item in contract["bindings"]["claims"]
        if item.get("source_document")
    }
    changed_components: set[str] = set()
    violations: list[dict[str, Any]] = []
    normalized_changes: list[dict[str, Any]] = []
    for raw in changes:
        change = _normalize_change(raw)
        paths = [item for item in (change.get("old_path"), change.get("path")) if item]
        affected = sorted({component for path in paths for component in path_components.get(path, set())})
        exempt_reason = None
        if change["status"] == "M" and change.get("semantic_equivalent") is True:
            exempt_reason = "registered_file_semantic_digest_unchanged"
        elif all(_matches_exemption(path, contract.get("exemptions") or []) for path in paths):
            exempt_reason = "explicit_non_semantic_path_exemption"
        elif (
            not affected
            and not all(path in registered_claim_sources for path in paths)
            and any(_is_semantic_path(path) for path in paths)
        ):
            violations.append(_violation("unregistered_semantic_dependency", change.get("path") or change.get("old_path"), "registered_or_exempt", change["status"]))
        if affected and not exempt_reason:
            changed_components.update(affected)
        normalized_changes.append({**change, "affected_components": affected, "exempt_reason": exempt_reason})
    evidence_rows = {str(item["evidence_id"]): item for item in contract["bindings"]["evidence"]}
    evidence_states: list[dict[str, Any]] = []
    for evidence_id in _topological_evidence(evidence_rows):
        binding = evidence_rows[evidence_id]
        direct = sorted(set(binding["components"]) & changed_components)
        upstream = sorted(item for item in binding.get("depends_on_evidence") or [] if next(row for row in evidence_states if row["evidence_id"] == item)["state"] == "stale")
        state = "stale" if direct or upstream else str(binding["declared_state"])
        evidence_states.append({"evidence_id": evidence_id, "state": state, "declared_state": binding["declared_state"], "stale_components": direct, "stale_upstream_evidence": upstream})
    evidence_map = {row["evidence_id"]: row for row in evidence_states}
    gate_states = _impact_entity_states(contract["bindings"]["gates"], changed_components, evidence_map, "gate_id")
    claim_states = _impact_entity_states(contract["bindings"]["claims"], changed_components, evidence_map, "claim_id", changed_paths={path for row in normalized_changes for path in (row.get("path"), row.get("old_path")) if path})
    minimum_reruns = sorted({gate for row in evidence_states if row["state"] == "stale" for gate in evidence_rows[row["evidence_id"]].get("rerun_gate_ids") or []} | {row["gate_id"] for row in gate_states if row["state"] == "stale"})
    stale_count = sum(row["state"] == "stale" for row in evidence_states + gate_states + claim_states)
    report = _report_base(contract)
    report.update({
        "status": "fresh_with_declared_blockers" if not stale_count and not violations else "stale_or_dependency_violation",
        "exit_code": EXIT_FRESH if not stale_count and not violations else EXIT_STALE,
        "impact_mode": mode,
        "from_ref": from_ref,
        "to_ref": to_ref,
        "changes": normalized_changes,
        "changed_component_ids": sorted(changed_components),
        "evidence": sorted(evidence_states, key=lambda row: row["evidence_id"]),
        "gates": sorted(gate_states, key=lambda row: row["gate_id"]),
        "claims": sorted(claim_states, key=lambda row: row["claim_id"]),
        "state_counts": _state_counts(evidence_states + gate_states + claim_states),
        "minimum_rerun_gate_ids": minimum_reruns,
        "violations": violations,
        "violation_count": len(violations),
    })
    return report


def git_impact(contract: Mapping[str, Any], *, repository_root: Path, from_ref: str, to_ref: str) -> dict[str, Any]:
    changes = _git_changes(repository_root, from_ref, to_ref)
    return impact_analysis(contract, changes, repository_root=repository_root, from_ref=from_ref, to_ref=to_ref, mode="git_range")


def worktree_impact(contract: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    changes = _git_changes(repository_root, "HEAD", None)
    return impact_analysis(contract, changes, repository_root=repository_root, from_ref="HEAD", to_ref="WORKTREE", mode="worktree")


def synthetic_impact_matrix(
    contract: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    """Run the four preregistered, content-free impact scenarios."""

    scenarios = [
        (
            "ranking_implementation_change",
            [{"status": "M", "path": "src/scholar_agent/agents/reranker.py", "semantic_equivalent": False}],
        ),
        (
            "human_annotation_interface_change",
            [{"status": "M", "path": "benchmark/human_annotation_delivery_v1_release/annotator-A/app.js", "semantic_equivalent": False}],
        ),
        (
            "default_policy_change",
            [{"status": "M", "path": "src/scholar_agent/retrieval/query_adapter.py", "semantic_equivalent": False}],
        ),
        (
            "non_semantic_python_comment",
            [{"status": "M", "path": "src/scholar_agent/agents/reranker.py", "semantic_equivalent": True}],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for scenario_id, changes in scenarios:
        report = impact_analysis(
            contract,
            changes,
            repository_root=repository_root,
            mode="synthetic_preregistered",
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "changed_component_ids": report["changed_component_ids"],
                "stale_claim_ids": sorted(
                    row["claim_id"] for row in report["claims"] if row["state"] == "stale"
                ),
                "stale_evidence_ids": sorted(
                    row["evidence_id"] for row in report["evidence"] if row["state"] == "stale"
                ),
                "stale_gate_ids": sorted(
                    row["gate_id"] for row in report["gates"] if row["state"] == "stale"
                ),
                "minimum_rerun_gate_ids": report["minimum_rerun_gate_ids"],
                "violation_count": report["violation_count"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "status": "synthetic_impact_verified",
        "scenario_count": len(rows),
        "scenarios": rows,
        "formal_validation_complete": False,
        "execution": contract["execution"],
    }


def audit_release(contract: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    report = verify_current(contract, repository_root=repository_root)
    readiness = _read_json(repository_root / "benchmark/validation_readiness_bundle_v1_release/readiness.json")
    if readiness.get("formal_validation_complete") is not False or int(readiness.get("blocker_count", -1)) != 3:
        report["violations"].append(_violation("readiness_blocker_boundary_drift", "readiness", {"formal": False, "blockers": 3}, {"formal": readiness.get("formal_validation_complete"), "blockers": readiness.get("blocker_count")}))
        report["violation_count"] = len(report["violations"])
        report["status"] = "stale_or_dependency_violation"
        report["exit_code"] = EXIT_STALE
    report["release_audit"] = {"blocker_count": readiness.get("blocker_count"), "formal_validation_complete": readiness.get("formal_validation_complete"), "freshness_evidence_registered": any(item.get("evidence_id") == "validation_freshness_current" for item in _read_json(repository_root / "benchmark/validation_readiness_bundle_v1_contract.json").get("evidence", []))}
    return report


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _current_entity_states(bindings: Sequence[Mapping[str, Any]], components: Mapping[str, Mapping[str, Any]], observed: Mapping[str, str | None], evidence: Mapping[str, Mapping[str, Any]], identity_key: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        direct = sorted(item for item in binding["components"] if observed[item] != components[item]["basis_digest"])
        stale_evidence = sorted(item for item in binding.get("evidence_ids") or [] if evidence[item]["state"] == "stale")
        reasons = []
        if direct: reasons.append("semantic_dependency_drift")
        if stale_evidence: reasons.append("referenced_evidence_stale")
        if root is not None and binding.get("source_document"):
            path = _safe_path(root, str(binding["source_document"])); actual = sha256_file(path) if path.is_file() else None
            if actual != binding.get("source_sha256"): reasons.append("claim_source_drift")
        expected_basis = entity_basis_digest(binding, components)
        if expected_basis != binding["basis_digest"]: reasons.append("entity_basis_contract_drift")
        state = "stale" if reasons else _normalized_declared_state(binding)
        rows.append({identity_key: binding[identity_key], "state": state, "declared_state": binding.get("declared_state") or binding.get("declared_status"), "stale_components": direct, "stale_evidence": stale_evidence, "reasons": sorted(reasons)})
    return rows


def _impact_entity_states(bindings: Sequence[Mapping[str, Any]], changed_components: set[str], evidence: Mapping[str, Mapping[str, Any]], identity_key: str, *, changed_paths: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for binding in bindings:
        direct = sorted(set(binding["components"]) & changed_components)
        stale_evidence = sorted(item for item in binding.get("evidence_ids") or [] if evidence[item]["state"] == "stale")
        document_changed = bool(changed_paths and binding.get("source_document") in changed_paths)
        state = "stale" if direct or stale_evidence or document_changed else _normalized_declared_state(binding)
        rows.append({identity_key: binding[identity_key], "state": state, "declared_state": binding.get("declared_state") or binding.get("declared_status"), "stale_components": direct, "stale_evidence": stale_evidence, "source_document_changed": document_changed})
    return rows


def _validate_inventory(contract: Mapping[str, Any], root: Path) -> None:
    readiness = _read_json(root / str(contract["readiness_scope"]["contract_path"]))
    exclusions = contract["readiness_scope"].get("self_exclusions") or {}
    expected = {
        "evidence": sorted(str(item["evidence_id"]) for item in contract["bindings"]["evidence"]),
        "gates": sorted(str(item["gate_id"]) for item in contract["bindings"]["gates"]),
        "claims": sorted(str(item["claim_id"]) for item in contract["bindings"]["claims"]),
    }
    actual = {
        "evidence": sorted(str(item["evidence_id"]) for item in readiness["evidence"] if item["evidence_id"] not in set(exclusions.get("evidence") or [])),
        "gates": sorted(str(item["gate_id"]) for item in readiness["read_only_gates"] if item["gate_id"] not in set(exclusions.get("gates") or [])),
        "claims": sorted(str(item["claim_id"]) for item in readiness["claims"] if item["claim_id"] not in set(exclusions.get("claims") or [])),
    }
    if actual != expected:
        raise FreshnessError("readiness_inventory_not_closed")


def _validate_dependency_graph(rows: Sequence[Mapping[str, Any]]) -> None:
    graph = {str(row["evidence_id"]): set(str(item) for item in row.get("depends_on_evidence") or []) for row in rows}
    for node, deps in graph.items():
        if any(dep not in graph for dep in deps):
            raise FreshnessError("unknown_evidence_dependency")
        if node in deps:
            raise FreshnessError("evidence_dependency_cycle")
    _topological_evidence({str(row["evidence_id"]): row for row in rows})


def _topological_evidence(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    indegree = {node: 0 for node in rows}
    consumers: dict[str, set[str]] = defaultdict(set)
    for node, row in rows.items():
        for dependency in row.get("depends_on_evidence") or []:
            if dependency not in rows:
                raise FreshnessError("unknown_evidence_dependency")
            indegree[node] += 1; consumers[str(dependency)].add(node)
    queue = deque(sorted(node for node, count in indegree.items() if count == 0)); ordered = []
    while queue:
        node = queue.popleft(); ordered.append(node)
        for consumer in sorted(consumers[node]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0: queue.append(consumer)
    if len(ordered) != len(rows):
        raise FreshnessError("evidence_dependency_cycle")
    return ordered


def _git_changes(root: Path, from_ref: str, to_ref: str | None) -> list[dict[str, Any]]:
    command = ["git", "diff", "--name-status", "-M", from_ref]
    if to_ref is not None: command.append(to_ref)
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True, env=_git_env(), timeout=30)
    if completed.returncode != 0: raise FreshnessBaselineMissing("git_diff_unavailable")
    changes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.split("\t"); raw_status = parts[0]; status = raw_status[0]
        if status in {"R", "C"}:
            old_path, path = parts[1], parts[2]
        else:
            old_path, path = None, parts[1]
        if _is_ignored_worktree_path(path) and (old_path is None or _is_ignored_worktree_path(old_path)):
            continue
        semantic_equivalent = False
        if status == "M":
            old = _git_file(root, from_ref, path)
            new = _git_file(root, to_ref, path) if to_ref else (_safe_path(root, path).read_bytes() if _safe_path(root, path).is_file() else None)
            semantic_equivalent = old is not None and new is not None and _semantic_digest(path, old) == _semantic_digest(path, new)
        changes.append({"status": status, "path": path, "old_path": old_path, "semantic_equivalent": semantic_equivalent})
    if to_ref is None:
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root, check=False, capture_output=True, text=True, env=_git_env(), timeout=30)
        if untracked.returncode != 0: raise FreshnessBaselineMissing("git_untracked_scan_unavailable")
        changes.extend(
            {"status": "A", "path": path, "old_path": None, "semantic_equivalent": False}
            for path in untracked.stdout.splitlines()
            if path and not _is_ignored_worktree_path(path)
        )
    return sorted(changes, key=lambda row: (str(row.get("path")), str(row.get("old_path"))))


def _is_ignored_worktree_path(path: str) -> bool:
    """Exclude protected, non-semantic local state from worktree impact scans."""

    parts = PurePosixPath(path).parts
    return bool(parts) and (parts[0] == "third_party" or parts[-1] == ".env")


def _git_file(root: Path, ref: str | None, path: str) -> bytes | None:
    if ref is None:
        candidate = _safe_path(root, path); return candidate.read_bytes() if candidate.is_file() else None
    completed = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=root, check=False, capture_output=True, env=_git_env(), timeout=20)
    return completed.stdout if completed.returncode == 0 else None


def _semantic_digest(path: str, content: bytes) -> str:
    if path.endswith(".py"):
        try:
            tree = ast.parse(content.decode("utf-8")); _strip_docstrings(tree)
            return stable_hash({"python_ast": ast.dump(tree, annotate_fields=True, include_attributes=False)})
        except (UnicodeDecodeError, SyntaxError): pass
    if path.endswith(".json"):
        try: return stable_hash({"json": json.loads(content.decode("utf-8"))})
        except (UnicodeDecodeError, json.JSONDecodeError): pass
    return hashlib.sha256(content).hexdigest()


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)


def _normalize_change(raw: Mapping[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status") or "")[:1]
    if status not in {"A", "M", "D", "R", "C"}: raise FreshnessError("change_status_invalid")
    path = str(raw.get("path") or ""); old_path = str(raw.get("old_path") or "") or None
    if not path or (status in {"R", "C"} and not old_path): raise FreshnessError("change_path_invalid")
    return ChangeRecord(status=status, path=path, old_path=old_path, semantic_equivalent=bool(raw.get("semantic_equivalent", False)))


def _matches_exemption(path: str, exemptions: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(item.get("kind")) == "test_only" and path.startswith(str(item.get("path_prefix"))) for item in exemptions)


def _is_semantic_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in SEMANTIC_ROOTS)


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root, check=False, capture_output=True, env=_git_env(), timeout=20)
    return completed.returncode == 0


def _git_env() -> dict[str, str]:
    return {"LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _violation(code: str, path: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {"code": code, "path": path, "expected": expected, "actual": actual}


def _state_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {state: sum(row.get("state") == state for row in rows) for state in sorted(ALLOWED_STATES)}


def _normalized_declared_state(binding: Mapping[str, Any]) -> str:
    value = str(binding.get("declared_state") or binding.get("declared_status") or "fresh")
    if value == "blocked":
        return "blocked"
    if value == "not_applicable":
        return "not_applicable"
    return "fresh"


def _report_base(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "baseline_head": contract["baseline"]["head"],
        "formal_validation_complete": False,
        "execution": contract["execution"],
    }
