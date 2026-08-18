from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_prf import build_prf_analysis


def test_prf_analysis_reports_contribution_budget_and_determinism(
    tmp_path: Path,
) -> None:
    paths: dict[tuple[str, str], Path] = {}
    for split, config in {
        "scifact": ("beir_scifact", "test", 0, 50),
        "development": ("auto_scholar_query", "development", 0, 10),
        "validation": ("auto_scholar_query", "validation", 10, 5),
    }.items():
        for policy in ("current_rules", "prf_v1"):
            paths[(split, policy)] = _run(
                tmp_path,
                split,
                policy,
                config=config,
                hit=policy == "prf_v1",
            )
    kwargs = {
        "scifact_current": paths[("scifact", "current_rules")],
        "scifact_prf": paths[("scifact", "prf_v1")],
        "development_current": paths[("development", "current_rules")],
        "development_prf": paths[("development", "prf_v1")],
        "validation_current": paths[("validation", "current_rules")],
        "validation_prf": paths[("validation", "prf_v1")],
    }

    comparison = build_prf_analysis(**kwargs, output_dir=tmp_path / "first")
    build_prf_analysis(**kwargs, output_dir=tmp_path / "second")

    assert comparison["parameters"]["query_budget_growth"] == 0
    assert comparison["splits"]["scifact"]["budget_parity"] == {
        "case_count": 1,
        "same_planned_subquery_count": 1,
        "original_query_first": 1,
    }
    assert comparison["splits"]["development"]["prf_contribution"] == {
        "applied_case_count": 1,
        "fallback_case_count": 0,
        "unique_candidate_count": 2,
        "independent_candidate_count": 1,
        "gold_hit_count": 1,
        "independent_gold_count": 1,
        "recorded_request_count": 1,
        "recorded_latency_seconds": 0.25,
    }
    assert comparison["decision"]["recommend_continue"] is True
    for split in ("scifact", "development", "validation"):
        for policy in ("current_rules", "prf_v1"):
            run = comparison["splits"][split]["runs"][policy]
            assert "run_dir" not in run
            assert "facet_contribution" not in run
    for name in ("comparison.json", "per_query.jsonl", "summary.md"):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def _run(
    root: Path,
    split: str,
    policy: str,
    *,
    config: tuple[str, str, int, int],
    hit: bool,
) -> Path:
    dataset, dataset_split, offset, limit = config
    run = root / f"{split}-{policy}"
    run.mkdir()
    common = {
        "dataset": dataset,
        "dataset_sha256": "fixture",
        "dataset_split": dataset_split,
        "case_ids": [f"{split}-1"],
        "offset": offset,
        "limit": limit,
        "sources": ["openalex", "arxiv", "semantic_scholar", "pubmed"],
        "query_adapter_policy": "adaptive",
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": None,
        "max_workers": 1,
        "budgets": {"max_candidate_papers": 200},
        "diagnostics": True,
        "llm": {"requested": False},
        "enable_query_evolution": False,
        "query_evolution_policy": "off",
        "enable_refchain": False,
        "query_planning_policy": policy,
    }
    metric_value = float(hit)
    metrics = {
        "aggregate": {
            "recall_at_k": {"20": metric_value},
            "precision_at_k": {"20": metric_value / 20},
            "f1_at_k": {"20": 0.1 if hit else 0.0},
            "ndcg_at_k": {"20": metric_value},
            "mrr": metric_value,
        },
        "per_case": [
            {
                "case_id": f"{split}-1",
                "metrics": {
                    "recall_at_k": {"20": metric_value},
                    "f1_at_k": {"20": 0.1 if hit else 0.0},
                },
            }
        ],
        "snapshot_costs": {
            "recorded_search_request_count": 2,
            "recorded_retry_count": 0,
            "recorded_error_count": 0,
            "recorded_latency_seconds": 0.5,
            "retrieval_snapshot_hits": 2,
            "retrieval_snapshot_writes": 0,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    stage = {
        "case_count": 1,
        "initial_retrieval_recall": metric_value,
        "initial_query_planning": {
            "effective_request_count": 2,
            "recorded_latency_seconds": 0.5,
            "unique_candidate_count": 2,
            "unique_gold_count": int(hit),
        },
        "source_contribution": {
            "sources": {
                "arxiv": {
                    "returned_candidate_count": 2,
                    "unique_candidate_count": 2,
                    "gold_hit_count": int(hit),
                    "unique_gold_hit_count": int(hit),
                    "success_count": 2,
                    "error_count": 0,
                }
            }
        },
    }
    purpose = "prf_v1" if policy == "prf_v1" else "normalized_keywords"
    selected = [
        {"query": "graph retrieval", "purpose": "original_query"},
        {"query": "graph retrieval neural", "purpose": purpose},
    ]
    result = {
        "case_id": f"{split}-1",
        "query": "graph retrieval",
        "status": "succeeded",
        "stage_diagnostics": {
            "stage_metrics": {
                "candidate_recall": {"initial_retrieval": metric_value}
            },
            "initial_query_planning": {
                "source_error_count": 0,
                "unique_gold_count": int(hit),
                "planning": {
                    "selected_subqueries": selected,
                    "prf_skip_reason": None,
                },
                "subqueries": [
                    {
                        "purpose": purpose,
                        "unique_candidate_count": 2,
                        "exclusive_candidate_count": int(hit),
                        "post_run_gold_hit_count": int(hit),
                        "post_run_unique_gold_hit_count": int(hit),
                        "recorded_request_count": 1,
                        "recorded_latency_seconds": 0.25,
                    }
                ],
            },
        },
    }
    _write(run / "config.json", common)
    _write(run / "metrics.json", metrics)
    _write(run / "stage_metrics.json", stage)
    (run / "results.jsonl").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
