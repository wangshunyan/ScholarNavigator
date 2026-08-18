from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_disjunctive_facets import (
    build_disjunctive_facets_analysis,
)


def test_analysis_accepts_new_gold_and_keeps_product_default(
    tmp_path: Path,
) -> None:
    paths = _comparison_runs(tmp_path)

    result = build_disjunctive_facets_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    acceptance = result["validation_acceptance"]
    assert acceptance["accepted"] is True
    assert acceptance["unique_gold_gain"] == 1
    assert result["recommended_policy"] == "disjunctive_facets"
    assert result["product_default"] == "current_rules"
    assert result["product_default_changed"] is False
    assert result["development_rule_status"] == "frozen_before_validation"
    contribution = result["splits"]["validation"]["disjunctive_facets"][
        "or_query_contribution"
    ]
    assert contribution == {
        "adapted_query_count": 1,
        "exclusive_candidate_count": 4,
        "logical_query_count": 1,
        "post_run_unique_gold_hit_count": 1,
        "raw_candidate_count": 10,
        "recorded_latency_milliseconds": 1200,
        "recorded_request_count": 10,
        "unique_candidate_count": 8,
    }
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    assert (tmp_path / "analysis" / "summary.md").is_file()


def test_analysis_rejects_excess_noise_or_replay_network_activity(
    tmp_path: Path,
) -> None:
    paths = _comparison_runs(
        tmp_path,
        disjunctive_noise_count=8,
        disjunctive_replay_requests=1,
    )

    result = build_disjunctive_facets_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    checks = result["validation_acceptance"]["checks"]
    assert checks["weak_irrelevant_ratio_within_0_10"] is False
    assert checks["frozen_replay_zero_network"] is False
    assert result["recommended_policy"] == "current_rules"


def test_analysis_reports_query_matching_and_ranking_deltas(tmp_path: Path) -> None:
    paths = _comparison_runs(tmp_path)

    result = build_disjunctive_facets_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    assert result["validation_failure_deltas"] == {
        "over_broad": 1,
        "query_not_matched": -1,
        "ranking_cutoff": 1,
    }


def test_analysis_enforces_frozen_fixed_split_and_runtime(tmp_path: Path) -> None:
    paths = _comparison_runs(tmp_path)
    config_path = paths["validation_disjunctive"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["retrieval_mode"] = "live"
    _json(config_path, config)

    with pytest.raises(ValueError, match="frozen replay"):
        build_disjunctive_facets_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def _comparison_runs(
    root: Path,
    *,
    disjunctive_noise_count: int = 2,
    disjunctive_replay_requests: int = 0,
) -> dict[str, Path]:
    return {
        "development_current": _run(
            root, "development-current", "current_rules", offset=130
        ),
        "development_disjunctive": _run(
            root,
            "development-disjunctive",
            "disjunctive_facets",
            offset=130,
            unique_gold=3,
            requests=30,
            query_not_matched=1,
            ranking_cutoff=1,
            over_broad=2,
        ),
        "validation_current": _run(
            root, "validation-current", "current_rules", offset=150
        ),
        "validation_disjunctive": _run(
            root,
            "validation-disjunctive",
            "disjunctive_facets",
            offset=150,
            unique_gold=3,
            requests=30,
            noise_count=disjunctive_noise_count,
            replay_requests=disjunctive_replay_requests,
            query_not_matched=1,
            ranking_cutoff=1,
            over_broad=2,
        ),
    }


def _run(
    root: Path,
    name: str,
    policy: str,
    *,
    offset: int,
    unique_gold: int = 2,
    requests: int = 20,
    noise_count: int = 2,
    replay_requests: int = 0,
    query_not_matched: int = 2,
    ranking_cutoff: int = 0,
    over_broad: int = 1,
) -> Path:
    path = root / name
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_ids": [f"case-{offset + index}" for index in range(20)],
        "offset": offset,
        "limit": 20,
        "sources": ["arxiv"],
        "query_adapter_policy": "adaptive",
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
        "query_planning_policy": policy,
        "judgement_policy": "current_rules",
        "retrieval_mode": "replay",
    }
    metric = 0.3
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
            "replay_execution_request_count": replay_requests,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    planning = {
        "subquery_count": 40,
        "average_subquery_count": 2.0,
        "adapted_query_count": 40,
        "average_adapted_query_count": 2.0,
        "unique_candidate_count": 100,
        "duplicate_candidate_ratio": 0.2,
        "unique_gold_count": unique_gold,
        "effective_request_count": requests,
        "recorded_latency_seconds": 10.0,
        "source_error_rate": 0.0,
        "facet_contribution": {},
        "ineffective_reasons": {"over_broad": over_broad},
    }
    stage_metrics = {
        "case_count": 20,
        "initial_retrieval_recall": 0.3,
        "initial_query_planning": planning,
    }
    purpose = (
        "disjunctive_facet_any"
        if policy == "disjunctive_facets"
        else "normalized_keywords"
    )
    categories = [
        *({"category": "weakly_relevant"} for _ in range(noise_count)),
        *({"category": "partially_relevant"} for _ in range(10 - noise_count)),
    ]
    gold_diagnostics = [
        *({"drop_reason": "not_retrieved"} for _ in range(query_not_matched)),
        *({"drop_reason": "outside_final_top_k"} for _ in range(ranking_cutoff)),
    ]
    result = {
        "case_id": f"case-{offset}",
        "query": "fixture query",
        "status": "succeeded",
        "stage_diagnostics": {
            "initial_query_planning": {
                "planner_version": "1.5.0",
                **planning,
                "subqueries": [
                    {
                        "combination_mode": (
                            "any" if policy == "disjunctive_facets" else "all"
                        ),
                        "purpose": purpose,
                        "adapted_queries": ["(all:graph OR all:retrieval)"],
                        "raw_candidate_count": 10,
                        "unique_candidate_count": 8,
                        "exclusive_candidate_count": 4,
                        "post_run_unique_gold_hit_count": 1,
                        "recorded_request_count": 10,
                        "recorded_latency_seconds": 1.2,
                    }
                ],
            },
            "snapshots": [{"stage": "initial_judged", "candidates": categories}],
            "gold_diagnostics": gold_diagnostics,
        },
    }
    _json(path / "config.json", config)
    _json(path / "metrics.json", metrics)
    _json(path / "stage_metrics.json", stage_metrics)
    (path / "results.jsonl").write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
