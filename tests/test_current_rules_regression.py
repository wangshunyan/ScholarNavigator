from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.current_rules_regression import (
    BASELINE_APPROVAL_TOKEN,
    RegressionGateError,
    authorize_baseline_proposal,
    canonicalize_config,
    compare_profiles,
    validate_current_rules_config,
    write_gate_artifacts,
)


def _profile() -> dict:
    return {
        "datasets": {
            "sample": {
                "summary_metrics": {
                    "candidate_recall": 0.5,
                    "recall_at_20": 0.25,
                    "f1_at_20": 0.1,
                },
                "snapshot_integrity": {
                    "required_retrieval_keys": ["key-a", "key-b"],
                    "missing_key_count": 0,
                },
                "cases": {
                    "q1": {
                        "candidate_identities": [
                            json.dumps({"identifiers": ["doi:one"]}),
                            json.dumps({"identifiers": ["doi:two"]}),
                        ],
                        "source_terminals": [
                            {
                                "snapshot_key": "key-a",
                                "snapshot_terminal_status": "success",
                            }
                        ],
                        "metrics": {
                            "candidate_recall": 0.5,
                            "recall_at_20": 0.25,
                            "f1_at_20": 0.1,
                            "matched_gold_ids": ["doi:one"],
                        },
                    }
                },
            }
        }
    }


def test_metric_drift_has_minimal_path() -> None:
    expected = _profile()
    actual = json.loads(json.dumps(expected))
    actual["datasets"]["sample"]["cases"]["q1"]["metrics"]["f1_at_20"] = 0.09

    diffs = compare_profiles(expected, actual)

    assert diffs == [
        {
            "path": "$.datasets.sample.cases.q1.metrics.f1_at_20",
            "kind": "value_changed",
            "expected": 0.1,
            "actual": 0.09,
        }
    ]


def test_candidate_addition_and_removal_are_reported_as_set_diff() -> None:
    expected = _profile()
    actual = json.loads(json.dumps(expected))
    actual["datasets"]["sample"]["cases"]["q1"]["candidate_identities"] = [
        json.dumps({"identifiers": ["doi:two"]}),
        json.dumps({"identifiers": ["doi:three"]}),
    ]

    diffs = compare_profiles(expected, actual)

    candidate = next(item for item in diffs if item["kind"] == "set_changed")
    assert candidate["path"].endswith("candidate_identities")
    assert candidate["removed"] == [json.dumps({"identifiers": ["doi:one"]})]
    assert candidate["added"] == [json.dumps({"identifiers": ["doi:three"]})]


def test_terminal_change_and_missing_key_are_located() -> None:
    expected = _profile()
    actual = json.loads(json.dumps(expected))
    dataset = actual["datasets"]["sample"]
    dataset["cases"]["q1"]["source_terminals"][0][
        "snapshot_terminal_status"
    ] = "failed"
    dataset["snapshot_integrity"]["required_retrieval_keys"] = ["key-a"]
    dataset["snapshot_integrity"]["missing_key_count"] = 1

    diffs = compare_profiles(expected, actual)
    paths = {item["path"] for item in diffs}

    assert (
        "$.datasets.sample.cases.q1.source_terminals[0].snapshot_terminal_status"
        in paths
    )
    assert "$.datasets.sample.snapshot_integrity.required_retrieval_keys" in paths
    assert "$.datasets.sample.snapshot_integrity.missing_key_count" in paths


def test_default_experiment_switch_drift_is_rejected() -> None:
    config = {
        "sources": ["openalex", "arxiv", "semantic_scholar", "pubmed"],
        "retrieval_mode": "replay",
        "query_planning_policy": "current_rules",
        "query_adapter_policy": "adaptive",
        "query_evolution_policy": "off",
        "ranking_policy": "current_rules",
        "judgement_policy": "current_rules",
        "result_policy": "highly_and_partial",
        "run_profile": "balanced",
        "top_k": 20,
        "enable_prf": True,
        "llm": {"requested": False},
    }

    violations = validate_current_rules_config(config)

    assert "enable_prf:must_be_false" in violations


def test_nondeterministic_config_fields_and_temporary_roots_are_ignored(
    tmp_path: Path,
) -> None:
    first = {
        "started_at": "one",
        "code": {"commit": "old"},
        "runtime_code_hash": "old",
        "resume_signature": "old",
        "dataset_source_path": str(tmp_path / "one" / "dataset.jsonl"),
        "snapshot": {"directory": str(tmp_path / "one" / "snapshot")},
        "top_k": 20,
        "llm": {"requested": False, "runtime_available": True, "model": "a"},
    }
    second = {
        **first,
        "started_at": "two",
        "code": {"commit": "new"},
        "runtime_code_hash": "new",
        "resume_signature": "new",
        "dataset_source_path": str(tmp_path / "two" / "dataset.jsonl"),
        "snapshot": {"directory": str(tmp_path / "two" / "snapshot")},
        "llm": {"requested": False, "runtime_available": False, "model": "b"},
    }

    assert canonicalize_config(first) == canonicalize_config(second)
    assert "started_at" not in canonicalize_config(first)


def test_baseline_proposal_requires_explicit_token_and_reason() -> None:
    with pytest.raises(RegressionGateError, match="explicit baseline"):
        authorize_baseline_proposal(approval_token="wrong", reason="valid long reason")
    with pytest.raises(RegressionGateError, match="at least 12"):
        authorize_baseline_proposal(
            approval_token=BASELINE_APPROVAL_TOKEN,
            reason="short",
        )

    authorize_baseline_proposal(
        approval_token=BASELINE_APPROVAL_TOKEN,
        reason="reviewed intentional metric migration",
    )


def test_gate_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    observed = _profile()
    report = {"passed": True, "drifts": [], "execution": {"network": 0}}
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_gate_artifacts(first, observed=observed, report=report)
    write_gate_artifacts(second, observed=observed, report=report)

    for name in ("observed_profile.json", "regression_report.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
