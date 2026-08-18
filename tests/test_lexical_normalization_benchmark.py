from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.search_schemas import JudgementFeatureVector
from scholar_agent.evaluation.lexical_normalization_benchmark import (
    METRIC_VERSION,
    _aggregate_dataset,
    _identity_instance_key,
    assert_candidate_identity_parity,
    classify_candidate_transition,
    write_lexical_normalization_benchmark,
)


@pytest.mark.parametrize(
    ("is_gold", "before", "after", "before_rank", "after_rank", "expected"),
    [
        (True, False, True, 21, 20, "recovered_gold"),
        (True, True, False, 20, 21, "lost_gold"),
        (True, True, True, 10, 8, "gold_rank_changed"),
        (False, False, True, 21, 20, "benchmark_non_gold_admitted"),
        (False, True, False, 20, 21, "benchmark_non_gold_removed"),
        (False, True, True, 8, 10, "benchmark_non_gold_rank_changed"),
        (False, True, True, 8, 8, "unchanged"),
    ],
)
def test_candidate_transition_classification_is_exhaustive(
    is_gold: bool,
    before: bool,
    after: bool,
    before_rank: int,
    after_rank: int,
    expected: str,
) -> None:
    assert classify_candidate_transition(
        is_gold=is_gold,
        baseline_returned=before,
        experiment_returned=after,
        baseline_rank=before_rank,
        experiment_rank=after_rank,
    ) == expected


def test_candidate_parity_requires_same_identity_and_pre_sort_order() -> None:
    first = Paper(title="First", doi="10.1/first")
    second = Paper(title="Second", doi="10.1/second")
    assert_candidate_identity_parity([first, second], [first, second])
    with pytest.raises(ValueError, match="identity/order"):
        assert_candidate_identity_parity([first, second], [second, first])
    with pytest.raises(ValueError, match="identity/order"):
        assert_candidate_identity_parity([first], [first, second])


def test_audit_identity_key_keeps_conflicting_same_doi_records_separate() -> None:
    first = Paper(
        title="First",
        doi="10.1/shared",
        arxiv_id="2401.00001",
    )
    second = Paper(
        title="Second",
        doi="10.1/shared",
        arxiv_id="2401.00002",
    )

    assert _identity_instance_key(first) != _identity_instance_key(second)


def test_legacy_feature_vector_defaults_new_diagnostics_to_empty() -> None:
    feature = JudgementFeatureVector.model_validate(
        {
            "config_version": "legacy",
            "config_hash": "0" * 64,
            "metadata_completeness": 1.0,
            "final_score": 0.5,
            "highly_relevant_threshold": 0.72,
            "partially_relevant_threshold": 0.45,
            "weakly_relevant_threshold": 0.25,
            "category_reason": "legacy_snapshot",
        }
    )
    assert feature.lexical_normalization_matches == []


def test_manifest_freezes_default_off_policy_before_metrics() -> None:
    manifest = json.loads(
        Path("benchmark/lexical_normalization_v1_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["policy"]["default"] == "off"
    assert manifest["policy"]["experimental_value"] == (
        "lexical_normalization_v1"
    )
    assert manifest["frozen_invariants"]["network_request_count"] == 0
    assert manifest["frozen_invariants"]["snapshot_write_count"] == 0


def test_audit_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"dataset": "x", "case_id": "1", "comparison": "tied"}]
    candidates = [
        {
            "dataset": "x",
            "candidate_id": "doi:10.1/example",
            "transition": "unchanged",
        }
    ]
    aggregate = {"schema_version": "1", "datasets": {"x": {"count": 1}}}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_lexical_normalization_benchmark(
        first, cases, candidates, aggregate
    )
    write_lexical_normalization_benchmark(
        second, cases, candidates, aggregate
    )
    for name in (
        "case_comparison.jsonl",
        "candidate_diagnostics.jsonl",
        "aggregate.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_dataset_metrics_exclude_identity_unavailable_cases() -> None:
    gold = EvalGoldPaper(arxiv_id="1234.5678")
    hit = Paper(title="Matched", identifiers={"arxiv_id": "1234.5678"})
    case_rows = [
        {
            "comparison": "tied",
            "candidate_identity_parity": True,
            "evaluable_gold_count": 1,
            "candidate_gold_count": 1,
            "recovered_unique_gold_ids": [],
            "lost_unique_gold_ids": [],
        },
        {
            "comparison": "tied",
            "candidate_identity_parity": True,
            "evaluable_gold_count": 0,
            "candidate_gold_count": 0,
            "recovered_unique_gold_ids": [],
            "lost_unique_gold_ids": [],
        },
    ]
    states = [
        {
            "gold": [gold],
            "evaluable_gold_count": 1,
            "baseline_returned": [hit],
            "experiment_returned": [hit],
        },
        {
            "gold": [],
            "evaluable_gold_count": 0,
            "baseline_returned": [],
            "experiment_returned": [],
        },
    ]
    aggregate = _aggregate_dataset("fixture", case_rows, [], states)
    assert aggregate["evaluable_case_count"] == 1
    assert aggregate["identity_unavailable_case_count"] == 1
    assert aggregate["candidate_recall"] == 1.0
    assert aggregate["baseline"]["recall_at_20"] == 1.0
    assert METRIC_VERSION == "deduplicated_gold_identity_v2"
