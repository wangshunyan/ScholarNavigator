from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.paired_significance import (
    analyze_paired_rows,
    paired_metric_statistics,
    prepare_paired_rows,
    run_paired_significance_audit,
    write_paired_significance_audit,
)


ROOT = Path(__file__).resolve().parents[1]
POWER = {
    "alpha": 0.05,
    "target_power": 0.8,
    "minimum_detectable_absolute_lift": 0.01,
    "future_validation_query_count": 1000,
    "method": "normal_approximation_paired_mean_using_observed_difference_sd",
}


def _statistics(baseline: list[float], experiment: list[float]) -> dict:
    return paired_metric_statistics(
        baseline,
        experiment,
        bootstrap_seed=123,
        bootstrap_iterations=500,
        permutation_seed=456,
        permutation_iterations=500,
        exact_nonzero_pair_limit=20,
        tie_tolerance=1e-12,
        power_config=POWER,
    )


def _case(
    case_id: str,
    *,
    evaluable: int = 1,
    candidate_gold: int = 1,
    baseline_recall: float | None = 0.0,
    experiment_recall: float | None = 0.0,
    parity: bool = True,
) -> dict:
    baseline = (
        {"recall_at_20": baseline_recall, "f1_at_20": baseline_recall / 2}
        if baseline_recall is not None
        else {}
    )
    experiment = (
        {
            "recall_at_20": experiment_recall,
            "f1_at_20": experiment_recall / 2,
        }
        if experiment_recall is not None
        else {}
    )
    return {
        "dataset": "scifact",
        "case_order": int(case_id),
        "case_id": case_id,
        "evaluable_gold_count": evaluable,
        "candidate_gold_count": candidate_gold,
        "candidate_identity_parity": parity,
        "baseline": baseline,
        "experiment": experiment,
    }


def _context(case_id: str, query: str) -> tuple[tuple[str, str], dict]:
    return (
        ("scifact", case_id),
        {
            "normalized_query": query.casefold(),
            "query_sha256": f"query-{case_id}",
            "terminal_signature_sha256": f"terminal-{case_id}",
            "source_terminal_counts": {"arxiv:success": 1},
        },
    )


def test_all_ties_have_zero_interval_and_unit_p_value() -> None:
    result = _statistics([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])

    assert result["mean_paired_difference"] == 0.0
    assert result["bootstrap_ci_95"] == {
        "low": 0.0,
        "high": 0.0,
        "iterations": 500,
        "method": "paired_query_resampling_percentile",
    }
    assert result["paired_permutation"]["p_value_two_sided"] == 1.0
    assert result["outcomes"] == {"improved": 0, "tied": 3, "regressed": 0}
    assert result["power_planning"]["reason"] == "zero_observed_variance"


def test_one_sided_improvements_use_two_sided_exact_permutation() -> None:
    result = _statistics([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])

    assert result["outcomes"] == {"improved": 3, "tied": 0, "regressed": 0}
    assert result["paired_permutation"]["method"] == "exact_sign_flip"
    assert result["paired_permutation"]["p_value_two_sided"] == 0.25


def test_positive_and_negative_differences_can_cancel() -> None:
    result = _statistics([0.0, 0.1], [0.1, 0.0])

    assert result["mean_paired_difference"] == 0.0
    assert result["outcomes"] == {"improved": 1, "tied": 0, "regressed": 1}
    assert result["paired_permutation"]["p_value_two_sided"] == 1.0


def test_single_pair_is_reported_with_small_sample_power_limit() -> None:
    result = _statistics([0.0], [0.2])

    assert result["bootstrap_ci_95"]["low"] == pytest.approx(0.2)
    assert result["bootstrap_ci_95"]["high"] == pytest.approx(0.2)
    assert result["paired_permutation"]["p_value_two_sided"] == 1.0
    assert result["power_planning"]["reason"] == "fewer_than_two_pairs"


def test_missing_pairs_unavailable_gold_and_candidate_drift_are_excluded() -> None:
    rows = [
        _case("1", evaluable=0),
        _case("2", baseline_recall=None),
        _case("3", parity=False),
    ]
    contexts = dict(_context(str(index), f"query {index}") for index in range(1, 4))

    paired = prepare_paired_rows(rows, contexts)

    assert paired[0]["all_evaluable_exclusion_reasons"] == [
        "identity_unavailable_gold"
    ]
    assert "missing_pair:recall_at_20" in paired[1]["all_evaluable_exclusion_reasons"]
    assert paired[2]["included_all_evaluable"] is True
    assert paired[2]["included_strict_comparable"] is False
    assert "candidate_identity_drift" in paired[2]["strict_exclusion_reasons"]


def test_missing_shared_source_terminal_is_not_strictly_comparable() -> None:
    context = dict([_context("1", "query")])
    context[("scifact", "1")]["terminal_signature_sha256"] = None

    paired = prepare_paired_rows([_case("1")], context)

    assert paired[0]["included_all_evaluable"] is True
    assert paired[0]["included_strict_comparable"] is False
    assert paired[0]["strict_exclusion_reasons"] == [
        "missing_shared_terminal_signature"
    ]


def test_duplicate_normalized_queries_are_all_excluded() -> None:
    rows = [_case("1"), _case("2")]
    contexts = dict([_context("1", "Same Query"), _context("2", "same query")])

    paired = prepare_paired_rows(rows, contexts)

    assert not any(row["included_all_evaluable"] for row in paired)
    assert all(
        "duplicate_normalized_query" in row["all_evaluable_exclusion_reasons"]
        for row in paired
    )


def test_analysis_and_written_artifacts_are_deterministic(tmp_path: Path) -> None:
    manifest = json.loads(
        (ROOT / "benchmark/lexical_normalization_significance_manifest.json").read_text()
    )
    manifest["bootstrap"]["iterations"] = 500
    manifest["permutation_test"]["iterations"] = 500
    rows = [_case("1", experiment_recall=0.2), _case("2")]
    contexts = dict(_context(str(index), f"query {index}") for index in (1, 2))
    paired = prepare_paired_rows(rows, contexts)

    first = analyze_paired_rows(paired, manifest)
    second = analyze_paired_rows(paired, manifest)
    assert first == second

    manifest_path = tmp_path / "manifest-source.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_one = tmp_path / "one"
    run_two = tmp_path / "two"
    write_paired_significance_audit(run_one, paired, first, manifest_path)
    write_paired_significance_audit(run_two, paired, second, manifest_path)
    for name in ("paired_queries.jsonl", "statistics.json", "manifest.json"):
        assert (run_one / name).read_bytes() == (run_two / name).read_bytes()


def test_frozen_lexical_replay_pairing_is_complete() -> None:
    rows, result = run_paired_significance_audit(
        ROOT / "benchmark/lexical_normalization_significance_manifest.json"
    )
    tracked = json.loads(
        (ROOT / "benchmark/lexical_normalization_significance_result.json").read_text()
    )

    assert len(rows) == 65
    assert result["pairing"]["all_evaluable_query_count"] == 56
    assert result["pairing"]["strict_comparable_query_count"] == 56
    assert result["pairing"]["duplicate_query_count"] == 0
    assert result["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "input_mode": "frozen_replay_artifacts",
    }
    pooled = result["scopes"]["all_evaluable"]["datasets"][
        "combined_query_equal"
    ]["metrics"]
    assert tracked["pooled_query_equal"]["recall_at_20"][
        "mean_difference"
    ] == pooled["recall_at_20"]["mean_paired_difference"]
    assert tracked["pooled_query_equal"]["f1_at_20"][
        "paired_permutation_p_two_sided"
    ] == pooled["f1_at_20"]["paired_permutation"]["p_value_two_sided"]
