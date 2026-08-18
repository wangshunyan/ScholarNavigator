from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.gold_metric_semantics import (
    GoldMetricSemanticsError,
    build_full_gold_denominator_audit,
    check_gold_metric_semantics_regression,
    compare_frozen_metric_profiles,
)


def test_full_denominator_audit_closes_duplicate_relations() -> None:
    queries = [
        EvalQuery(
            query_id="q1",
            query="one",
            gold_papers=[
                EvalGoldPaper(doi="10.1/shared"),
                EvalGoldPaper(doi="https://doi.org/10.1/shared"),
                EvalGoldPaper(arxiv_id="2401.00001"),
            ],
        ),
        EvalQuery(
            query_id="q2",
            query="two",
            gold_papers=[EvalGoldPaper(pubmed_id="PMID:123")],
        ),
    ]

    rows, duplicates, summary = build_full_gold_denominator_audit(queries)

    assert summary == {
        "query_count": 2,
        "legacy_evaluable_gold_count": 4,
        "deduplicated_evaluable_gold_count": 3,
        "duplicate_relation_count": 1,
        "impacted_query_count": 1,
        "count_closed": True,
    }
    assert rows[0]["denominator_delta"] == -1
    assert duplicates[0]["rule"] == "shared_stable_identifier"
    assert duplicates[0]["denominator_delta"] == -1


def test_full_denominator_audit_is_deterministic_for_fixed_input() -> None:
    query = EvalQuery(
        query_id="q",
        query="query",
        gold_papers=[
            EvalGoldPaper(arxiv_id="2401.00001v2"),
            EvalGoldPaper(arxiv_id="arXiv:2401.00001"),
        ],
    )

    assert build_full_gold_denominator_audit([query]) == (
        build_full_gold_denominator_audit([query])
    )


def test_profile_comparison_reports_only_metric_changes() -> None:
    before = _profile(denominator=2, candidate=0.5, recall=0.5, f1=0.1)
    after = _profile(denominator=1, candidate=1.0, recall=1.0, f1=0.2)

    rows, summary = compare_frozen_metric_profiles(before, after)

    assert rows[0]["duplicate_relation_count"] == 1
    assert rows[0]["delta"] == {
        "candidate_recall": 0.5,
        "recall_at_20": 0.5,
        "f1_at_20": 0.1,
    }
    assert summary["fixture"]["impacted_query_count"] == 1


@pytest.mark.metric_semantics_regression
def test_real_metric_semantics_regression_gate(tmp_path: Path) -> None:
    report = check_gold_metric_semantics_regression(
        Path("benchmark/gold_metric_semantics_manifest.json"),
        tmp_path / "gate",
    )

    assert report["passed"] is True
    assert report["query_count"] == 1000
    assert report["duplicate_relation_count"] == 5
    assert report["frozen_replay_case_count"] == 65
    assert report["drift_count"] == 0
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "retrieval_invoked": False,
        "snapshot_mode": "read_only",
    }


@pytest.mark.parametrize(
    "field",
    ("candidate_identities", "returned_identities", "source_terminals"),
)
def test_profile_comparison_rejects_non_metric_drift(field: str) -> None:
    before = _profile(denominator=2, candidate=0.5, recall=0.5, f1=0.1)
    after = deepcopy(before)
    after["datasets"]["fixture"]["cases"]["q1"][field] = ["drift"]

    with pytest.raises(GoldMetricSemanticsError, match="non-metric"):
        compare_frozen_metric_profiles(before, after)


def _profile(
    *, denominator: int, candidate: float, recall: float, f1: float
) -> dict:
    metrics = {
        "evaluable_gold_count": denominator,
        "candidate_recall": candidate,
        "recall_at_20": recall,
        "f1_at_20": f1,
    }
    return {
        "datasets": {
            "fixture": {
                "summary_metrics": {
                    "case_count": 1,
                    "evaluable_case_count": 1,
                    **{key: metrics[key] for key in ("candidate_recall", "recall_at_20", "f1_at_20")},
                },
                "cases": {
                    "q1": {
                        "status": "success",
                        "candidate_identities": ["paper:a"],
                        "returned_identities": ["paper:a"],
                        "source_terminals": [{"source": "fixture", "status": "success"}],
                        "required_retrieval_keys": ["key"],
                        "metrics": metrics,
                    }
                },
            }
        }
    }
