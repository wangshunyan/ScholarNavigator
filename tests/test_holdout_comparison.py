from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_judgement_holdout import holdout_options
from scholar_agent.evaluation.holdout_comparison import (
    BOOTSTRAP_METRICS,
    HOLDOUT_CASE_IDS,
    HOLDOUT_LIMIT,
    HOLDOUT_OFFSET,
    _assert_zero_replay_cost,
    _candidate_snapshot_hash,
    _collection_round_summary,
    _validate_candidate_identity,
    paired_bootstrap,
    query_features,
    query_slice_analysis,
)


def _result_row(index: int, *, candidate_title: str = "Candidate") -> dict:
    return {
        "case_id": f"AutoScholarQuery_test_{index}",
        "query": "compare graph method on QM9 dataset",
        "result": {
            "query_analysis": {
                "constraints": {"must_have_terms": ["graph"]},
            },
            "search_plan": {
                "query_planning": {
                    "facets": [
                        {"facet_type": "topic", "terms": ["graph"]},
                        {"facet_type": "method", "terms": ["message passing"]},
                        {"facet_type": "dataset", "terms": ["QM9"]},
                    ]
                }
            },
        },
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": "initial_deduplicated",
                    "candidates": [
                        {
                            "identifiers": {"arxiv_id": f"2401.{index:05d}"},
                            "title": candidate_title,
                            "year": 2024,
                        }
                    ],
                }
            ]
        },
    }


def _bootstrap_rows() -> list[dict]:
    rows = []
    for index, case_id in enumerate(HOLDOUT_CASE_IDS):
        current = {metric: 0.0 for metric in BOOTSTRAP_METRICS}
        calibrated = {
            metric: (0.1 if index % 3 == 0 else 0.0)
            for metric in BOOTSTRAP_METRICS
        }
        rows.append(
            {
                "case_id": case_id,
                "query_features": {
                    "topic_structure": "single_topic",
                    "method_presence": "no_method",
                    "dataset_presence": "no_dataset",
                    "must_have_presence": "has_must_have",
                    "paper_type_presence": "no_paper_type",
                    "query_length_bin": "short_1_10",
                },
                "current_rules": {"candidate_recall": 0.2, **current},
                "calibrated_rules_v1": {"candidate_recall": 0.2, **calibrated},
            }
        )
    return rows


def test_holdout_range_and_protocol_are_fixed() -> None:
    current = holdout_options(
        policy="current_rules",
        snapshot_dir=Path("/tmp/frozen"),
        output_dir=Path("/tmp/output"),
        resume=False,
    )
    calibrated = holdout_options(
        policy="calibrated_rules_v1",
        snapshot_dir=Path("/tmp/frozen"),
        output_dir=Path("/tmp/output"),
        resume=False,
    )

    assert HOLDOUT_OFFSET == 20
    assert HOLDOUT_LIMIT == 30
    assert HOLDOUT_CASE_IDS[0] == "AutoScholarQuery_test_20"
    assert HOLDOUT_CASE_IDS[-1] == "AutoScholarQuery_test_49"
    assert current.snapshot_dir == calibrated.snapshot_dir
    assert current.offset == calibrated.offset == 20
    assert current.limit == calibrated.limit == 30
    assert current.sources == calibrated.sources == ["arxiv"]
    assert current.retrieval_mode == calibrated.retrieval_mode == "replay"
    assert current.judgement_config_path is None
    assert calibrated.judgement_config_path is None


def test_policy_runs_share_identical_retrieval_candidates() -> None:
    rows = {
        case_id: _result_row(index)
        for index, case_id in zip(range(20, 50), HOLDOUT_CASE_IDS, strict=True)
    }

    _validate_candidate_identity(rows, dict(rows))
    assert _candidate_snapshot_hash(rows) == _candidate_snapshot_hash(dict(rows))

    changed = dict(rows)
    changed[HOLDOUT_CASE_IDS[-1]] = _result_row(49, candidate_title="Changed")
    with pytest.raises(ValueError, match="retrieval candidates differ"):
        _validate_candidate_identity(rows, changed)


def test_paired_bootstrap_is_deterministic() -> None:
    rows = _bootstrap_rows()

    first = paired_bootstrap(rows, seed=1234, iterations=500)
    second = paired_bootstrap(rows, seed=1234, iterations=500)

    assert first == second
    assert first["case_count"] == 30
    assert first["metrics"]["f1_at_20"]["mean_difference"] == pytest.approx(
        1 / 30
    )
    assert first["metrics"]["f1_at_20"]["ci_95_low"] >= 0.0


def test_query_slices_are_defined_without_gold() -> None:
    features = query_features(_result_row(20))
    rows = _bootstrap_rows()
    rows[0]["query_features"] = features

    slices = query_slice_analysis(rows)

    assert features["topic_structure"] == "compound_query"
    assert features["method_presence"] == "has_method"
    assert features["dataset_presence"] == "has_dataset"
    assert features["must_have_presence"] == "has_must_have"
    assert "gold" not in features
    assert slices["topic_structure"]["compound_query"]["case_count"] == 1


def test_replay_cost_must_be_zero_network() -> None:
    zero = {
        "snapshot_costs": {
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0,
        }
    }

    _assert_zero_replay_cost(zero)

    nonzero = {
        "snapshot_costs": {
            **zero["snapshot_costs"],
            "replay_execution_request_count": 1,
        }
    }
    with pytest.raises(ValueError, match="executed network work"):
        _assert_zero_replay_cost(nonzero)


def test_snapshot_collection_summary_does_not_copy_keys() -> None:
    summary = _collection_round_summary(
        {
            "round_index": 2,
            "collected_entry_count": 5,
            "request_count": 5,
            "failed_entry_count": 0,
            "elapsed_seconds": 1.25,
            "completed_keys": ["secretly-large-key-list"],
            "coverage": {"retrieval_keys": ["also-large"]},
        }
    )

    assert summary["round_index"] == 2
    assert summary["request_count"] == 5
    assert "completed_keys" not in summary
    assert "coverage" not in summary
