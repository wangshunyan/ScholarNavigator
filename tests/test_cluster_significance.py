from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.cluster_significance import (
    analyze_cluster_queries,
    check_cluster_significance_regression,
    cluster_metric_statistics,
    prepare_cluster_query_rows,
    run_cluster_significance_audit,
    write_cluster_significance_audit,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark/lexical_normalization_cluster_significance_manifest.json"


def _stats(values: list[float]) -> dict:
    return cluster_metric_statistics(
        values,
        [0.25] * len(values),
        [0.25 + value for value in values],
        bootstrap_seed=17,
        permutation_seed=23,
        bootstrap_iterations=500,
        permutation_iterations=500,
        tie_tolerance=1e-12,
    )


def _query(
    scope: str,
    case_id: str,
    component_id: str,
    difference: float,
    *,
    included_decontaminated: bool = True,
) -> dict:
    return {
        "scope": scope,
        "dataset": "autoscholar_record160" if scope == "record160" else "scifact",
        "case_id": case_id,
        "component_id": component_id,
        "evaluable_gold_count": 1,
        "included_full": True,
        "included_decontaminated": included_decontaminated,
        "evaluable_pair": True,
        "exclusion_reasons": [],
        "metrics": {
            metric: {
                "baseline": 0.2,
                "experiment": 0.2 + difference,
                "difference": difference,
            }
            for metric in ("candidate_recall", "recall_at_20", "f1_at_20")
        },
    }


def _manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["bootstrap"]["iterations"] = 500
    manifest["permutation_test"]["iterations"] = 500
    manifest["query_level_comparator"]["bootstrap_iterations"] = 500
    manifest["query_level_comparator"]["permutation_iterations"] = 500
    return manifest


def test_single_cluster_and_all_ties_have_deterministic_null_result() -> None:
    result = _stats([0.0])

    assert result["mean_paired_difference"] == 0.0
    assert result["bootstrap_ci_95"]["low"] == 0.0
    assert result["bootstrap_ci_95"]["high"] == 0.0
    assert result["cluster_sign_flip"]["p_value_two_sided"] == 1.0
    assert result["outcomes"] == {"improved": 0, "tied": 1, "regressed": 0}


def test_multiple_clusters_positive_and_negative_effects_cancel() -> None:
    result = _stats([0.2, -0.2])

    assert result["mean_paired_difference"] == pytest.approx(0.0)
    assert result["outcomes"] == {"improved": 1, "tied": 0, "regressed": 1}
    assert result["cluster_sign_flip"]["p_value_two_sided"] == 1.0


def test_unequal_cluster_sizes_use_component_equal_primary_estimand() -> None:
    rows = [
        _query("record160", "large-1", "large", 0.3),
        _query("record160", "large-2", "large", 0.3),
        _query("record160", "large-3", "large", 0.3),
        _query("record160", "small", "small", -0.3),
        _query("existing65", "science", "external:science", 0.0),
    ]
    clusters, result = analyze_cluster_queries(
        rows,
        [{"component_id": "large", "query_count": 3}, {"component_id": "small", "query_count": 1}],
        _manifest(),
    )

    view = result["views"]["record160_full"]
    assert view["component_equal_metrics"]["recall_at_20"][
        "mean_paired_difference"
    ] == pytest.approx(0.0)
    assert view["historical_query_equal_comparator"]["recall_at_20"][
        "mean_paired_difference"
    ] == pytest.approx(0.15)
    assert sum(row["query_count"] for row in clusters if row["view"] == "record160_full") == 4


def test_cross_dataset_combination_does_not_reweight_dataset_labels() -> None:
    rows = [
        _query("existing65", "science", "external:science", 0.1),
        {
            **_query("existing65", "auto", "auto-component", -0.1),
            "dataset": "auto_dev",
        },
        _query("record160", "record", "record-component", 0.0),
    ]
    _, result = analyze_cluster_queries(
        rows,
        [{"component_id": "record-component", "query_count": 1}],
        _manifest(),
    )

    assert result["views"]["existing65_full"]["component_equal_metrics"][
        "f1_at_20"
    ]["mean_paired_difference"] == pytest.approx(0.0)


def test_prepare_rows_reuses_auto_component_and_makes_scifact_singleton() -> None:
    assignments = [{"query_id": "auto", "component_id": "component:frozen"}]
    diagnostics = [
        {
            "scope": "existing65",
            "case_id": "auto",
            "component_id": "component:frozen",
            "candidate_recall": 1.0,
            "baseline": {"recall_at_20": 0.0, "f1_at_20": 0.0},
            "experiment": {"recall_at_20": 1.0, "f1_at_20": 0.1},
            "included_full": True,
            "included_decontaminated": False,
            "exclusion_reason": "cross_stratum_contaminated_component",
        },
        {
            "scope": "record160",
            "case_id": "science",
            "component_id": None,
            "candidate_recall": 0.0,
            "baseline": {"recall_at_20": 0.0, "f1_at_20": 0.0},
            "experiment": {"recall_at_20": 0.0, "f1_at_20": 0.0},
            "included_full": True,
            "included_decontaminated": True,
            "exclusion_reason": None,
        },
    ]
    rows = prepare_cluster_query_rows(
        assignments,
        diagnostics,
        [{"case_id": "auto", "case_order": 0, "dataset": "auto_dev", "evaluable_gold_count": 1}],
        [{"case_id": "science", "case_order": 0, "dataset": "scifact", "evaluable_gold_count": 1}],
    )

    assert rows[0]["component_id"] == "component:frozen"
    assert rows[0]["external_singleton_component"] is False
    assert rows[1]["component_id"].startswith("external:scifact:")
    assert rows[1]["external_singleton_component"] is True


def test_decontaminated_exclusions_name_frozen_component_contamination() -> None:
    rows = [
        _query(
            "record160",
            "contaminated",
            "component:a",
            0.1,
            included_decontaminated=False,
        ),
        _query("record160", "clean", "component:b", 0.0),
        _query("existing65", "science", "external:science", 0.0),
    ]
    _, result = analyze_cluster_queries(
        rows, [{"component_id": "component:b", "query_count": 1}], _manifest()
    )

    view = result["views"]["record160_decontaminated"]
    assert view["excluded_reason_counts"] == {
        "cross_stratum_contaminated_component": 1
    }
    assert view["included_query_count"] == 1


def test_written_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    rows = [_query("record160", "r", "component:r", 0.1), _query("existing65", "s", "external:s", 0.0)]
    clusters, statistics = analyze_cluster_queries(
        rows, [{"component_id": "component:r", "query_count": 1}], _manifest()
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_cluster_significance_audit(first, rows, clusters, statistics, MANIFEST)
    write_cluster_significance_audit(second, rows, clusters, statistics, MANIFEST)

    for name in ("paired_queries.jsonl", "paired_components.jsonl", "statistics.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.cluster_significance_regression
def test_frozen_cluster_significance_gate(tmp_path: Path) -> None:
    report = check_cluster_significance_regression(MANIFEST, tmp_path / "gate")
    assert report["passed"] is True
    assert report["drift_count"] == 0

    _, _, statistics = run_cluster_significance_audit(MANIFEST)
    assert statistics["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "input_mode": "frozen_replay_and_frozen_component_assignments",
        "components_recomputed": False,
    }
