from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_facet_union import build_facet_union_analysis


def test_analysis_accepts_new_gold_and_attributes_facet_type(
    tmp_path: Path,
) -> None:
    development_current = _run(tmp_path, "dev-current", "current_rules", 250)
    development_candidate = _run(
        tmp_path,
        "dev-candidate",
        "facet_union",
        250,
        found={0, 1, 2, 3, 4},
        requests=60,
    )
    validation_current = _run(tmp_path, "val-current", "current_rules", 270)
    validation_candidate = _run(
        tmp_path,
        "val-candidate",
        "facet_union",
        270,
        found={0, 1, 2, 3, 4},
        requests=60,
    )

    result = build_facet_union_analysis(
        development_current=development_current,
        development_candidate=development_candidate,
        validation_current=validation_current,
        validation_candidate=validation_candidate,
        output_dir=tmp_path / "analysis",
    )

    retention = result["gold_retention"]["validation"]
    assert retention["retained_baseline_gold_count"] == 4
    assert retention["lost_baseline_gold_count"] == 0
    assert retention["net_new_gold_count"] == 1
    assert result["validation_acceptance"]["accepted"] is True
    assert result["rule_query_planning_frozen"] is False
    diagnostics = result["splits"]["validation"]["facet_union"][
        "facet_union_diagnostics"
    ]
    assert diagnostics["logical_query_count"] == 20
    assert diagnostics["exclusive_candidate_count"] == 40
    assert diagnostics["post_run_unique_gold_hit_count"] == 1
    assert diagnostics["by_facet_type"]["method"]["logical_query_count"] == 20
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    assert (tmp_path / "analysis" / "summary.md").is_file()


def test_analysis_rejects_without_new_gold_and_freezes_rules(
    tmp_path: Path,
) -> None:
    paths = {
        "development_current": _run(
            tmp_path, "dev-current", "current_rules", 250
        ),
        "development_candidate": _run(
            tmp_path, "dev-candidate", "facet_union", 250
        ),
        "validation_current": _run(
            tmp_path, "val-current", "current_rules", 270
        ),
        "validation_candidate": _run(
            tmp_path, "val-candidate", "facet_union", 270
        ),
    }

    result = build_facet_union_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    assert result["validation_acceptance"]["checks"][
        "at_least_one_net_new_unique_gold"
    ] is False
    assert result["validation_acceptance"]["accepted"] is False
    assert result["rule_query_planning_frozen"] is True
    assert result["next_planning_direction"] == (
        "llm_semantic_or_other_semantic_retrieval"
    )


def test_analysis_enforces_exact_fixed_slice(tmp_path: Path) -> None:
    paths = {
        "development_current": _run(
            tmp_path, "dev-current", "current_rules", 250
        ),
        "development_candidate": _run(
            tmp_path, "dev-candidate", "facet_union", 250
        ),
        "validation_current": _run(
            tmp_path, "val-current", "current_rules", 270
        ),
        "validation_candidate": _run(
            tmp_path, "val-candidate", "facet_union", 270
        ),
    }
    for group in ("validation_current", "validation_candidate"):
        config_path = paths[group] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["case_ids"][-1] = "AutoScholarQuery_test_999"
        _json(config_path, config)

    with pytest.raises(ValueError, match="case ids mismatch"):
        build_facet_union_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def test_analysis_rejects_nonzero_replay_network_cost(tmp_path: Path) -> None:
    paths = {
        "development_current": _run(
            tmp_path, "dev-current", "current_rules", 250
        ),
        "development_candidate": _run(
            tmp_path, "dev-candidate", "facet_union", 250
        ),
        "validation_current": _run(
            tmp_path, "val-current", "current_rules", 270
        ),
        "validation_candidate": _run(
            tmp_path, "val-candidate", "facet_union", 270
        ),
    }
    metrics_path = paths["validation_candidate"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["snapshot_costs"]["replay_execution_request_count"] = 1
    _json(metrics_path, metrics)

    with pytest.raises(ValueError, match="executed network work"):
        build_facet_union_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


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
        candidate = policy == "facet_union"
        gold_found = index in found
        subqueries = []
        if candidate:
            subqueries.append(
                {
                    "purpose": "facet_union_method",
                    "combination_mode": "all",
                    "status": "executed",
                    "adapted_queries": ["contrastive learning"],
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
            "subquery_count": 60 + int(policy == "facet_union") * 20,
            "average_subquery_count": 3.0 + int(policy == "facet_union"),
            "adapted_query_count": 60 + int(policy == "facet_union") * 20,
            "average_adapted_query_count": 3.0 + int(policy == "facet_union"),
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
