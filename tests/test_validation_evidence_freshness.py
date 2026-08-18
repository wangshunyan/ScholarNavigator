from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.validation_evidence_freshness import (
    FreshnessError,
    _is_ignored_worktree_path,
    impact_analysis,
    load_contract,
    stable_hash,
    synthetic_impact_matrix,
    verify_current,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmark/validation_evidence_freshness_v1_contract.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    return load_contract(CONTRACT_PATH, repository_root=ROOT)


def _stale_ids(report: dict[str, object], section: str, identity: str) -> set[str]:
    return {row[identity] for row in report[section] if row["state"] == "stale"}


def test_current_inventory_is_closed_and_fresh(contract: dict[str, object]) -> None:
    report = verify_current(contract, repository_root=ROOT)
    assert report["status"] == "fresh_with_declared_blockers"
    assert report["exit_code"] == 0
    assert report["component_count"] == 61
    assert report["state_counts"] == {
        "blocked": 41,
        "fresh": 236,
        "not_applicable": 1,
        "stale": 0,
    }
    assert report["minimum_rerun_gate_ids"] == []


def test_ranking_change_propagates_only_to_ranking_top20_and_downstream(
    contract: dict[str, object],
) -> None:
    report = impact_analysis(
        contract,
        [{"status": "M", "path": "src/scholar_agent/agents/reranker.py"}],
        repository_root=ROOT,
    )
    assert report["changed_component_ids"] == ["ranking_runtime"]
    assert _stale_ids(report, "gates", "gate_id") == {
        "ranking_decisions",
        "source_fusion",
        "source_reliability",
        "tiebreak_qualification",
        "top20_delivery_fidelity",
    }
    assert "human_annotation_delivery" not in _stale_ids(report, "gates", "gate_id")
    assert "completion_bias" not in _stale_ids(report, "gates", "gate_id")


def test_annotation_interface_does_not_invalidate_source_funnel(
    contract: dict[str, object],
) -> None:
    report = impact_analysis(
        contract,
        [
            {
                "status": "M",
                "path": "benchmark/human_annotation_delivery_v1_release/annotator-A/app.js",
            }
        ],
        repository_root=ROOT,
    )
    assert _stale_ids(report, "gates", "gate_id") == {"human_annotation_delivery"}
    assert _stale_ids(report, "evidence", "evidence_id") == {
        "formal_evidence_quarantine_readiness",
        "formal_validation_clearance_current",
        "human_annotation_delivery_dry_run",
        "human_annotation_delivery_protocol",
        "human_annotation_delivery_readiness",
            "human_annotation_assignment_protocol",
            "human_annotation_assignment_readiness",
            "human_annotation_assignment_simulation",
            "human_annotation_submission_protocol",
            "human_annotation_submission_readiness",
            "human_annotation_submission_simulation",
            "human_annotator_qualification_protocol",
            "human_annotator_qualification_readiness",
            "human_annotator_qualification_simulation",
            "human_adjudication_activation_protocol",
            "human_adjudication_activation_readiness",
            "human_adjudication_activation_simulation",
        }
    assert "source_reliability" not in _stale_ids(report, "gates", "gate_id")


def test_default_policy_change_invalidates_registry_claims(
    contract: dict[str, object],
) -> None:
    report = impact_analysis(
        contract,
        [{"status": "M", "path": "src/scholar_agent/retrieval/query_adapter.py"}],
        repository_root=ROOT,
    )
    assert report["changed_component_ids"] == ["default_policy"]
    assert _stale_ids(report, "evidence", "evidence_id") == {"evidence_registry_gate"}
    assert _stale_ids(report, "gates", "gate_id") == {"evidence_registry"}
    assert _stale_ids(report, "claims", "claim_id") == {
        "architecture_default_policy_isolation",
        "readme_operational_entrypoint",
    }


def test_comment_only_change_is_explicitly_semantic_noop(
    contract: dict[str, object],
) -> None:
    report = impact_analysis(
        contract,
        [
            {
                "status": "M",
                "path": "src/scholar_agent/agents/reranker.py",
                "semantic_equivalent": True,
            }
        ],
        repository_root=ROOT,
    )
    assert report["changed_component_ids"] == []
    assert report["state_counts"]["stale"] == 0
    assert report["changes"][0]["exempt_reason"] == "registered_file_semantic_digest_unchanged"


def test_rename_cannot_evade_registered_dependency(contract: dict[str, object]) -> None:
    report = impact_analysis(
        contract,
        [
            {
                "status": "R",
                "old_path": "src/scholar_agent/agents/reranker.py",
                "path": "src/scholar_agent/agents/reranker_moved.py",
            }
        ],
        repository_root=ROOT,
    )
    assert report["changed_component_ids"] == ["ranking_runtime"]
    assert report["state_counts"]["stale"] > 0


def test_claim_document_change_invalidates_all_bound_statements(
    contract: dict[str, object],
) -> None:
    report = impact_analysis(
        contract,
        [{"status": "M", "path": "docs/architecture.md"}],
        repository_root=ROOT,
    )
    stale = _stale_ids(report, "claims", "claim_id")
    assert {
        "architecture_default_policy_isolation",
        "architecture_external_scorer_handoff_ready",
        "architecture_human_annotation_delivery_ready",
        "architecture_offline_integrity_gates",
    } <= stale
    assert report["violation_count"] == 0


def test_unregistered_semantic_file_fails_closed(contract: dict[str, object]) -> None:
    report = impact_analysis(
        contract,
        [{"status": "A", "path": "src/scholar_agent/evaluation/unregistered_gate.py"}],
        repository_root=ROOT,
    )
    assert report["exit_code"] == 2
    assert report["violations"][0]["code"] == "unregistered_semantic_dependency"


def test_dependency_cycle_and_cross_version_are_rejected(tmp_path: Path) -> None:
    value = json.loads(CONTRACT_PATH.read_text())
    first, second = value["bindings"]["evidence"][:2]
    first["depends_on_evidence"] = [second["evidence_id"]]
    second["depends_on_evidence"] = [first["evidence_id"]]
    cycle = tmp_path / "cycle.json"
    cycle.write_text(json.dumps(value))
    with pytest.raises(FreshnessError, match="cycle"):
        load_contract(cycle, repository_root=ROOT)

    value = json.loads(CONTRACT_PATH.read_text())
    value["protocol"] = "validation_evidence_freshness_v2"
    version = tmp_path / "version.json"
    version.write_text(json.dumps(value))
    with pytest.raises(FreshnessError, match="version"):
        load_contract(version, repository_root=ROOT)


def test_deleted_dependency_and_future_implementation_are_stale(
    contract: dict[str, object],
) -> None:
    broken = copy.deepcopy(contract)
    component = broken["components"][0]
    component["files"].append("src/scholar_agent/evaluation/definitely_missing.py")
    component["files"].sort()
    component["basis_digest"] = stable_hash("missing-baseline")
    component["implementation_commit"] = "f" * 40
    report = verify_current(broken, repository_root=ROOT)
    codes = {row["code"] for row in report["violations"]}
    assert "component_basis_drift" in codes
    assert "implementation_commit_after_baseline" in codes


def test_synthetic_matrix_and_report_are_byte_deterministic(
    contract: dict[str, object],
) -> None:
    first = synthetic_impact_matrix(contract, repository_root=ROOT)
    second = synthetic_impact_matrix(contract, repository_root=ROOT)
    assert first == second
    assert first["scenario_count"] == 4
    rows = {row["scenario_id"]: row for row in first["scenarios"]}
    assert rows["non_semantic_python_comment"]["stale_evidence_ids"] == []
    assert rows["human_annotation_interface_change"]["stale_gate_ids"] == [
        "human_annotation_delivery"
    ]


def test_protected_local_state_is_excluded_from_worktree_impact() -> None:
    assert _is_ignored_worktree_path("third_party/paper-qa")
    assert _is_ignored_worktree_path("nested/.env")
    assert not _is_ignored_worktree_path("src/scholar_agent/settings.py")
