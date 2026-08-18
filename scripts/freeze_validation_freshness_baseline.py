#!/usr/bin/env python3
"""One-shot deterministic builder for validation_evidence_freshness_v1.

This command is intentionally separate from the verification CLI.  Updating a
baseline requires reviewing the explicit spec and regenerating the contract;
normal freshness checks never rewrite it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.validation_evidence_freshness import (  # noqa: E402
    component_digest,
    entity_basis_digest,
    evidence_basis_digest,
    sha256_file,
    stable_hash,
    write_json,
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={"LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")},
    )
    if completed.returncode != 0:
        raise RuntimeError("git_baseline_query_failed")
    return completed.stdout.strip()


def merge_addenda(
    spec: dict[str, Any], addenda: dict[str, Any] | None
) -> dict[str, Any]:
    """Extend the sealed base spec without rewriting its historical bytes."""

    if addenda is None:
        return copy.deepcopy(spec)
    if set(addenda) != {
        "blocked_evidence_ids",
        "claim_component_bindings",
        "components",
        "evidence_component_bindings",
        "gate_component_bindings",
        "protocol",
        "schema_version",
    } or addenda.get("protocol") != "validation_evidence_freshness_v1_addenda":
        raise RuntimeError("freshness_addenda_schema_invalid")
    merged = copy.deepcopy(spec)
    component_ids = {row["component_id"] for row in merged["components"]}
    for row in addenda["components"]:
        if row["component_id"] in component_ids:
            raise RuntimeError("freshness_addenda_component_conflict")
        merged["components"].append(copy.deepcopy(row))
        component_ids.add(row["component_id"])
    for key in (
        "claim_component_bindings",
        "evidence_component_bindings",
        "gate_component_bindings",
    ):
        overlap = set(merged[key]) & set(addenda[key])
        if overlap:
            raise RuntimeError("freshness_addenda_binding_conflict")
        merged[key].update(copy.deepcopy(addenda[key]))
    merged["blocked_evidence_ids"] = sorted(
        set(merged["blocked_evidence_ids"])
        | set(addenda["blocked_evidence_ids"])
    )
    return merged


def build(spec: dict[str, Any]) -> dict[str, Any]:
    readiness_path = ROOT / spec["readiness_contract_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    current_head = _git("rev-parse", "HEAD")
    sealed_base_head = str(spec["baseline_head"])
    try:
        _git("merge-base", "--is-ancestor", sealed_base_head, current_head)
    except RuntimeError as exc:
        raise RuntimeError("baseline_head_not_ancestor_of_current_head") from exc
    # The sealed spec remains historical input, while each reviewed extension
    # closes the composite dependency graph at the current source commit.
    base_head = current_head
    components: list[dict[str, Any]] = []
    for source in spec["components"]:
        row = dict(source)
        row["files"] = sorted(set(row["files"]))
        # A reviewed baseline may introduce a new tracked artifact in the same
        # commit that will contain this contract.  Before that commit exists,
        # git log has no path history; the explicitly verified baseline HEAD is
        # the only admissible provenance boundary (never a later commit).
        row["implementation_commit"] = (
            _git("log", "-1", "--format=%H", "--", *row["files"]) or current_head
        )
        row["basis_digest"] = component_digest(row, ROOT)
        components.append(row)
    components.sort(key=lambda row: row["component_id"])
    component_map = {row["component_id"]: row for row in components}
    exclusions = spec["self_exclusions"]
    evidence_sources = {
        row["evidence_id"]: row
        for row in readiness["evidence"]
        if row["evidence_id"] not in set(exclusions["evidence"])
    }
    if set(evidence_sources) != set(spec["evidence_component_bindings"]):
        raise RuntimeError("evidence_spec_inventory_mismatch")
    evidence: list[dict[str, Any]] = []
    for evidence_id, components_for_evidence in sorted(spec["evidence_component_bindings"].items()):
        source = evidence_sources[evidence_id]
        artifact_path = str(source["path"])
        row = {
            "evidence_id": evidence_id,
            "artifact_path": artifact_path,
            "artifact_sha256": str(source["sha256"]),
            "artifact_commit": (
                _git("log", "-1", "--format=%H", "--", artifact_path) or base_head
            ),
            "baseline_commit": base_head,
            "components": sorted(components_for_evidence),
            "declared_state": "blocked" if evidence_id in set(spec["blocked_evidence_ids"]) else "fresh",
            "depends_on_evidence": sorted(source.get("dependencies") or []),
            "rerun_gate_ids": [],
        }
        row["evidence_basis_digest"] = evidence_basis_digest(row, component_map)
        evidence.append(row)
    evidence_map = {row["evidence_id"]: row for row in evidence}
    gate_sources = {
        row["gate_id"]: row
        for row in readiness["read_only_gates"]
        if row["gate_id"] not in set(exclusions["gates"])
    }
    if set(gate_sources) != set(spec["gate_component_bindings"]):
        raise RuntimeError("gate_spec_inventory_mismatch")
    gates = []
    for gate_id, component_ids in sorted(spec["gate_component_bindings"].items()):
        row = {
            "gate_id": gate_id,
            "components": sorted(component_ids),
            "declared_state": "fresh",
            "evidence_ids": [],
        }
        row["basis_digest"] = entity_basis_digest(row, component_map)
        gates.append(row)
    claim_sources = {
        row["claim_id"]: row
        for row in readiness["claims"]
        if row["claim_id"] not in set(exclusions["claims"])
    }
    if set(claim_sources) != set(spec["claim_component_bindings"]):
        raise RuntimeError("claim_spec_inventory_mismatch")
    documents = {row["document_id"]: row for row in readiness["claim_sources"]}
    claims = []
    for claim_id, component_ids in sorted(spec["claim_component_bindings"].items()):
        source = claim_sources[claim_id]
        document = documents[source["document_id"]]
        row = {
            "claim_id": claim_id,
            "components": sorted(set(component_ids) | {"readiness_publication"}),
            "declared_status": source["status"],
            "evidence_ids": sorted(source["evidence_ids"]),
            "source_document": document["path"],
            "source_sha256": sha256_file(ROOT / document["path"]),
        }
        row["basis_digest"] = entity_basis_digest(row, component_map)
        claims.append(row)
    inventory = {
        "claims": sorted(claim_sources),
        "evidence": sorted(evidence_sources),
        "gates": sorted(gate_sources),
    }
    return {
        "schema_version": "1",
        "protocol": "validation_evidence_freshness_v1",
        "baseline": {
            "head": base_head,
            "inventory_digest": stable_hash(inventory),
            "inventory_counts": {key: len(value) for key, value in inventory.items()},
            "capture_semantics": "tracked_content_reviewed_at_baseline_v1",
        },
        "readiness_scope": {
            "contract_path": spec["readiness_contract_path"],
            "self_exclusions": exclusions,
        },
        "components": components,
        "bindings": {"claims": claims, "evidence": evidence, "gates": gates},
        "exemptions": spec["exemptions"],
        "unregistered_dependency_policy": "fail_closed_for_semantic_roots",
        "semantic_comparison": {
            "python": "ast_without_comments_or_docstrings_v1",
            "json": "parsed_canonical_json_v1",
            "other": "byte_sha256_v1",
            "rename": "always_semantic_for_registered_dependency",
        },
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "formal_validation_complete": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="benchmark/validation_evidence_freshness_v1_spec.json")
    parser.add_argument(
        "--addenda",
        default="benchmark/validation_evidence_freshness_v1_addenda.json",
    )
    parser.add_argument("--output", default="benchmark/validation_evidence_freshness_v1_contract.json")
    args = parser.parse_args()
    spec = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    addenda_path = ROOT / args.addenda
    addenda = (
        json.loads(addenda_path.read_text(encoding="utf-8"))
        if addenda_path.is_file()
        else None
    )
    write_json(ROOT / args.output, build(merge_addenda(spec, addenda)))
    print(json.dumps({"status": "baseline_frozen", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
