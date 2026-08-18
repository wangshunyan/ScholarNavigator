from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_disjunctive_holdout import (
    BOOTSTRAP_METRICS,
    HOLDOUT_CASE_IDS,
    HOLDOUT_LIMIT,
    HOLDOUT_OFFSET,
    _assert_zero_replay_cost,
    build_blocked_collection_status,
    build_disjunctive_holdout_analysis,
    paired_bootstrap,
)
from scholar_agent.agents.query_planning import plan_disjunctive_facets


def test_holdout_analysis_accepts_frozen_candidate_and_attributes_or(
    tmp_path: Path,
) -> None:
    current = _run(tmp_path, "current", "current_rules")
    disjunctive = _run(
        tmp_path,
        "disjunctive",
        "disjunctive_facets",
        unique_gold=6,
        candidate_recall=0.25,
        requests=50,
        weak_count=3,
    )

    result = build_disjunctive_holdout_analysis(
        current_run=current,
        disjunctive_run=disjunctive,
        output_dir=tmp_path / "analysis",
        bootstrap_iterations=500,
    )

    assert result["acceptance"]["accepted"] is True
    assert result["acceptance"]["unique_gold_gain"] == 2
    assert result["high_recall_profile_candidate"] is True
    assert result["product_default"] == "current_rules"
    assert result["product_default_changed"] is False
    contribution = result["groups"]["disjunctive_facets"][
        "or_query_contribution"
    ]
    assert contribution["logical_query_count"] == 40
    assert contribution["exclusive_candidate_count"] == 120
    assert contribution["post_run_unique_gold_hit_count"] == 2
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    status = json.loads(
        (tmp_path / "analysis" / "experiment_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["evaluation_status"] == "completed"
    assert status["metrics_available"] is True
    assert status["acceptance"] == "passed"
    assert status["gold_metrics_read"] is True
    assert (tmp_path / "analysis" / "per_query_diagnostics.jsonl").is_file()
    assert (tmp_path / "analysis" / "summary.md").is_file()


def test_holdout_analysis_rejects_without_two_new_gold(tmp_path: Path) -> None:
    current = _run(tmp_path, "current", "current_rules")
    disjunctive = _run(
        tmp_path,
        "disjunctive",
        "disjunctive_facets",
        unique_gold=5,
        candidate_recall=0.225,
        requests=50,
    )

    result = build_disjunctive_holdout_analysis(
        current_run=current,
        disjunctive_run=disjunctive,
        output_dir=tmp_path / "analysis",
        bootstrap_iterations=500,
    )

    assert result["acceptance"]["checks"][
        "at_least_two_new_unique_gold"
    ] is False
    assert result["high_recall_profile_candidate"] is False


def test_holdout_fixed_slice_and_shared_protocol_are_enforced(
    tmp_path: Path,
) -> None:
    current = _run(tmp_path, "current", "current_rules")
    disjunctive = _run(tmp_path, "disjunctive", "disjunctive_facets")
    config_path = disjunctive / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["offset"] = 171
    _json(config_path, config)

    with pytest.raises(ValueError, match="protocol mismatch"):
        build_disjunctive_holdout_analysis(
            current_run=current,
            disjunctive_run=disjunctive,
            output_dir=tmp_path / "analysis",
        )

    assert HOLDOUT_OFFSET == 170
    assert HOLDOUT_LIMIT == 40
    assert HOLDOUT_CASE_IDS[0] == "AutoScholarQuery_test_170"
    assert HOLDOUT_CASE_IDS[-1] == "AutoScholarQuery_test_209"


def test_holdout_bootstrap_is_deterministic() -> None:
    rows = []
    for index, case_id in enumerate(HOLDOUT_CASE_IDS):
        delta = 0.1 if index % 4 == 0 else 0.0
        rows.append(
            {
                "case_id": case_id,
                "current_rules": {metric: 0.0 for metric in BOOTSTRAP_METRICS},
                "disjunctive_facets": {
                    metric: delta for metric in BOOTSTRAP_METRICS
                },
                "deltas": {metric: delta for metric in BOOTSTRAP_METRICS},
            }
        )

    first = paired_bootstrap(rows, seed=1234, iterations=500)
    second = paired_bootstrap(rows, seed=1234, iterations=500)

    assert first == second
    assert first["case_count"] == 40
    assert first["metrics"]["candidate_recall"][
        "mean_difference"
    ] == pytest.approx(0.025)
    assert first["metrics"]["f1_at_20"]["ci_95_low"] >= 0.0


def test_holdout_replay_must_be_zero_network() -> None:
    metrics = {
        "snapshot_costs": {
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0,
        }
    }
    _assert_zero_replay_cost(metrics)
    metrics["snapshot_costs"]["replay_execution_retry_count"] = 1

    with pytest.raises(ValueError, match="executed network work"):
        _assert_zero_replay_cost(metrics)


def test_holdout_gold_is_absent_from_production_planner_interface() -> None:
    parameters = inspect.signature(plan_disjunctive_facets).parameters
    source = inspect.getsource(plan_disjunctive_facets).casefold()

    assert "gold" not in parameters
    assert "case_id" not in parameters
    assert "gold" not in source
    assert "case_id" not in source


def test_blocked_collection_report_does_not_fabricate_metrics(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    collection = tmp_path / "collection.json"
    _json(plan, {"entries": [{"key": "one"}, {"key": "two"}]})
    _json(
        collection,
        {
            "group": "baseline",
            "request_count": 4,
            "failed_entry_count": 2,
            "covered_success": 0,
            "covered_failed": 2,
            "missing_entries": 99,
            "blocked_sources": ["arxiv"],
            "source_failure_counts": {"arxiv": 2},
            "stop_reason": "snapshot_collection_source_cooldown",
            "elapsed_seconds": 35.0,
            "completed_keys": ["omitted-from-report"],
        },
    )

    status = build_blocked_collection_status(
        plan_path=plan,
        collection_result_path=collection,
        output_dir=tmp_path / "analysis",
    )

    assert status["evaluation_status"] == "blocked"
    assert status["metrics_available"] is False
    assert status["acceptance"] == "not_evaluated"
    assert status["high_recall_profile_candidate"] is False
    assert status["gold_metrics_read"] is False
    assert "completed_keys" not in status
    assert (tmp_path / "analysis" / "experiment_status.json").is_file()


def _run(
    root: Path,
    name: str,
    policy: str,
    *,
    unique_gold: int = 4,
    candidate_recall: float = 0.2,
    requests: int = 40,
    weak_count: int = 2,
) -> Path:
    path = root / name
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_ids": list(HOLDOUT_CASE_IDS),
        "offset": HOLDOUT_OFFSET,
        "limit": HOLDOUT_LIMIT,
        "sources": ["arxiv"],
        "query_adapter_policy": "adaptive",
        "query_planning_policy": policy,
        "query_planner_version": "1.5.0",
        "judgement_policy": "current_rules",
        "judgement_config_hash": "hash",
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": 2026,
        "max_workers": 1,
        "budgets": {"max_search_rounds": 2},
        "diagnostics": True,
        "llm": {"query_understanding": False, "judgement": False},
        "enable_query_evolution": False,
        "query_evolution_policy": "off",
        "enable_refchain": False,
        "retrieval_mode": "replay",
        "runtime_code_hash": "runtime",
    }
    metric = 0.1
    per_case = []
    results = []
    for index, case_id in enumerate(HOLDOUT_CASE_IDS):
        improved = policy == "disjunctive_facets" and index < 2
        case_metric = metric + (0.02 if improved else 0.0)
        per_case.append(
            {
                "case_id": case_id,
                "metrics": {
                    "f1_at_k": {"20": case_metric},
                    "recall_at_k": {"20": case_metric},
                    "ndcg_at_k": {"20": case_metric},
                    "mrr": case_metric,
                },
            }
        )
        categories = [
            *({"category": "weakly_relevant"} for _ in range(weak_count)),
            *({"category": "partially_relevant"} for _ in range(10 - weak_count)),
        ]
        any_subquery = (
            [
                {
                    "combination_mode": "any",
                    "adapted_queries": ["(all:graph OR all:retrieval)"],
                    "raw_candidate_count": 10,
                    "unique_candidate_count": 8,
                    "exclusive_candidate_count": 3,
                    "post_run_unique_gold_hit_count": int(index < 2),
                    "recorded_request_count": 1,
                    "recorded_latency_seconds": 0.5,
                }
            ]
            if policy == "disjunctive_facets"
            else []
        )
        results.append(
            {
                "case_id": case_id,
                "query": f"fixture query {index}",
                "status": "succeeded",
                "stage_diagnostics": {
                    "stage_metrics": {
                        "candidate_recall": {
                            "initial_retrieval": candidate_recall
                        }
                    },
                    "judgement": {
                        "gold_judged_highly_relevant": int(index < unique_gold),
                        "gold_judged_partially_relevant": 0,
                    },
                    "gold_diagnostics": [
                        {
                            "drop_reason": (
                                "returned" if index < unique_gold else "not_retrieved"
                            )
                        }
                    ],
                    "initial_query_planning": {
                        "subqueries": any_subquery,
                    },
                    "snapshots": [
                        {"stage": "initial_judged", "candidates": categories}
                    ],
                },
            }
        )
    metrics = {
        "case_count": HOLDOUT_LIMIT,
        "end_to_end_metrics": {
            "f1_at_k": {"5": metric, "10": metric, "20": metric},
            "precision_at_k": {"20": metric},
            "recall_at_k": {"20": metric},
            "ndcg_at_k": {"20": metric},
            "mrr": metric,
        },
        "per_case": per_case,
        "snapshot_costs": {
            "recorded_search_request_count": requests,
            "recorded_retry_count": 0,
            "recorded_error_count": 0,
            "recorded_latency_seconds": 20.0,
            "retrieval_snapshot_hits": requests,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    planning = {
        "subquery_count": 80,
        "average_subquery_count": 2.0,
        "adapted_query_count": 80,
        "average_adapted_query_count": 2.0,
        "unique_candidate_count": 400,
        "duplicate_candidate_ratio": 0.2,
        "unique_gold_count": unique_gold,
        "effective_request_count": requests,
        "recorded_latency_seconds": 20.0,
        "source_error_rate": 0.0,
    }
    stage_metrics = {
        "case_count": HOLDOUT_LIMIT,
        "gold_count": HOLDOUT_LIMIT,
        "initial_retrieval_recall": candidate_recall,
        "initial_query_planning": planning,
        "judgement": {
            "gold_judged_highly_relevant": unique_gold,
            "gold_judged_partially_relevant": 0,
        },
    }
    _json(path / "config.json", config)
    _json(path / "metrics.json", metrics)
    _json(path / "stage_metrics.json", stage_metrics)
    (path / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in results
        ),
        encoding="utf-8",
    )
    return path


def _json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
