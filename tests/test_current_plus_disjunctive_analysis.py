from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_current_plus_disjunctive import (
    _assert_zero_replay_cost,
    build_current_plus_analysis,
)


def test_analysis_accepts_additive_strategy_and_reports_retention(
    tmp_path: Path,
) -> None:
    development_current = _run(tmp_path, "dev-current", "current_rules", 210)
    development_candidate = _run(
        tmp_path,
        "dev-candidate",
        "current_plus_disjunctive",
        210,
        found={0, 1, 2, 3, 4},
        requests=60,
    )
    validation_current = _run(tmp_path, "val-current", "current_rules", 230)
    validation_candidate = _run(
        tmp_path,
        "val-candidate",
        "current_plus_disjunctive",
        230,
        found={0, 1, 2, 3, 4},
        requests=60,
    )

    result = build_current_plus_analysis(
        development_current=development_current,
        development_candidate=development_candidate,
        validation_current=validation_current,
        validation_candidate=validation_candidate,
        output_dir=tmp_path / "analysis",
    )

    retention = result["gold_retention"]["validation"]
    assert retention == {
        "baseline_retrieved_gold_count": 4,
        "candidate_retrieved_gold_count": 5,
        "retained_baseline_gold_count": 4,
        "lost_baseline_gold_count": 0,
        "net_new_gold_count": 1,
        "all_baseline_gold_retained": True,
    }
    assert result["validation_acceptance"]["accepted"] is True
    assert result["high_recall_profile_candidate"] is True
    candidate = result["splits"]["validation"]["current_plus_disjunctive"]
    assert candidate["gold_judgement_retained_count"] == 5
    assert candidate["final_returned_gold_count"] == 5
    assert candidate["or_query_contribution"]["logical_query_count"] == 20
    assert candidate["or_query_contribution"]["exclusive_candidate_count"] == 40
    assert candidate["or_execution"]["executed_query_count"] == 20
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    assert (tmp_path / "analysis" / "development_query_diagnostics.jsonl").is_file()
    assert (tmp_path / "analysis" / "validation_query_diagnostics.jsonl").is_file()
    assert (tmp_path / "analysis" / "summary.md").is_file()


def test_analysis_rejects_loss_of_baseline_gold(tmp_path: Path) -> None:
    development_current = _run(tmp_path, "dev-current", "current_rules", 210)
    development_candidate = _run(
        tmp_path,
        "dev-candidate",
        "current_plus_disjunctive",
        210,
        found={0, 1, 2, 4, 5},
    )
    validation_current = _run(tmp_path, "val-current", "current_rules", 230)
    validation_candidate = _run(
        tmp_path,
        "val-candidate",
        "current_plus_disjunctive",
        230,
        found={0, 1, 2, 4, 5},
    )

    result = build_current_plus_analysis(
        development_current=development_current,
        development_candidate=development_candidate,
        validation_current=validation_current,
        validation_candidate=validation_candidate,
        output_dir=tmp_path / "analysis",
    )

    retention = result["gold_retention"]["validation"]
    assert retention["lost_baseline_gold_count"] == 1
    assert retention["net_new_gold_count"] == 2
    assert result["validation_acceptance"]["checks"][
        "all_baseline_gold_retained"
    ] is False
    assert result["validation_acceptance"]["accepted"] is False


def test_analysis_enforces_fixed_slice_and_shared_protocol(tmp_path: Path) -> None:
    development_current = _run(tmp_path, "dev-current", "current_rules", 210)
    development_candidate = _run(
        tmp_path, "dev-candidate", "current_plus_disjunctive", 210
    )
    validation_current = _run(tmp_path, "val-current", "current_rules", 230)
    validation_candidate = _run(
        tmp_path, "val-candidate", "current_plus_disjunctive", 230
    )
    config_path = validation_candidate / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sources"] = ["openalex"]
    _json(config_path, config)

    with pytest.raises(ValueError, match="incompatible current-plus runs"):
        build_current_plus_analysis(
            development_current=development_current,
            development_candidate=development_candidate,
            validation_current=validation_current,
            validation_candidate=validation_candidate,
            output_dir=tmp_path / "analysis",
        )


def test_analysis_requires_zero_network_replay() -> None:
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


def _run(
    root: Path,
    name: str,
    policy: str,
    offset: int,
    *,
    found: set[int] | None = None,
    requests: int = 40,
) -> Path:
    path = root / name
    path.mkdir()
    found = {0, 1, 2, 3} if found is None else found
    case_ids = [f"AutoScholarQuery_test_{offset + index}" for index in range(20)]
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_ids": case_ids,
        "offset": offset,
        "limit": 20,
        "sources": ["arxiv"],
        "query_adapter_policy": "adaptive",
        "query_planning_policy": policy,
        "query_planner_version": "1.8.1",
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
    results = []
    for index, case_id in enumerate(case_ids):
        candidate = policy == "current_plus_disjunctive"
        gold_found = index in found
        subqueries = []
        if candidate:
            subqueries.append(
                {
                    "purpose": "current_plus_disjunctive_any",
                    "combination_mode": "any",
                    "status": "executed",
                    "adapted_queries": ["(all:graph OR all:retrieval)"],
                    "raw_candidate_count": 4,
                    "unique_candidate_count": 3,
                    "exclusive_candidate_count": 2,
                    "post_run_unique_gold_hit_count": int(index == 4),
                    "recorded_request_count": 1,
                    "skip_reasons": [],
                }
            )
        categories = [
            {"category": "weakly_relevant"},
            *({"category": "partially_relevant"} for _ in range(9)),
        ]
        results.append(
            {
                "case_id": case_id,
                "query": f"fixture query {index}",
                "status": "succeeded",
                "stage_diagnostics": {
                    "gold_diagnostics": [
                        {
                            "gold_id": f"gold-{index}",
                            "gold_title": f"Gold {index}",
                            "found": gold_found,
                            "drop_reason": "returned" if gold_found else "not_retrieved",
                        }
                    ],
                    "initial_query_planning": {
                        "planning": {"skipped_facets": []},
                        "subqueries": subqueries,
                    },
                    "snapshots": [
                        {"stage": "initial_judged", "candidates": categories}
                    ],
                },
                "result": {
                    "budget_status": {
                        "candidate_limit_applied": False,
                        "candidate_truncations": [],
                    }
                },
            }
        )
    metric = 0.1
    metrics = {
        "case_count": 20,
        "end_to_end_metrics": {
            "f1_at_k": {"5": metric, "10": metric, "20": metric},
            "precision_at_k": {"20": metric},
            "recall_at_k": {"20": metric},
            "ndcg_at_k": {"20": metric},
            "mrr": metric,
        },
        "snapshot_costs": {
            "recorded_search_request_count": requests,
            "recorded_retry_count": 0,
            "recorded_error_count": 0,
            "recorded_latency_seconds": 10.0,
            "retrieval_snapshot_hits": requests,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    stage_metrics = {
        "case_count": 20,
        "gold_count": 20,
        "initial_retrieval_recall": len(found) / 20,
        "initial_query_planning": {
            "subquery_count": 60 + int(policy == "current_plus_disjunctive") * 20,
            "average_subquery_count": 3.0 + int(policy == "current_plus_disjunctive"),
            "adapted_query_count": 60 + int(policy == "current_plus_disjunctive") * 20,
            "average_adapted_query_count": 3.0 + int(policy == "current_plus_disjunctive"),
            "unique_candidate_count": 200,
            "duplicate_candidate_ratio": 0.2,
            "unique_gold_count": len(found),
            "effective_request_count": requests,
            "recorded_latency_seconds": 10.0,
            "source_error_rate": 0.0,
        },
        "judgement": {
            "gold_judged_highly_relevant": len(found),
            "gold_judged_partially_relevant": 0,
        },
    }
    _json(path / "config.json", config)
    _json(path / "metrics.json", metrics)
    _json(path / "stage_metrics.json", stage_metrics)
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    return path


def _json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
