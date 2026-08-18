from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_concept_projection import build_concept_projection_analysis


def test_analysis_reports_pairs_projection_and_source_deltas(tmp_path: Path) -> None:
    runs: dict[tuple[str, str], Path] = {}
    split_config = {
        "scifact": ("beir_scifact", "test", 0, 50),
        "development": ("auto_scholar_query", "development", 0, 10),
        "validation": ("auto_scholar_query", "validation", 10, 5),
    }
    for split, values in split_config.items():
        for policy in ("current_rules", "concept_projection"):
            runs[(split, policy)] = _run(
                tmp_path,
                split,
                policy,
                config=values,
                hit=policy == "concept_projection",
            )

    output = tmp_path / "analysis"
    comparison = build_concept_projection_analysis(
        scifact_current=runs[("scifact", "current_rules")],
        scifact_projection=runs[("scifact", "concept_projection")],
        development_current=runs[("development", "current_rules")],
        development_projection=runs[("development", "concept_projection")],
        validation_current=runs[("validation", "current_rules")],
        validation_projection=runs[("validation", "concept_projection")],
        output_dir=output,
    )

    assert comparison["rule_status"] == "frozen_before_metrics"
    assert comparison["splits"]["scifact"]["full_lower_bound"]["wins"][
        "candidate_recall"
    ] == {"improved": 1}
    assert comparison["splits"]["development"]["runs"][
        "concept_projection"
    ]["projection_outcomes"] == {"applied": 1}
    assert comparison["splits"]["validation"]["source_deltas"]["arxiv"][
        "gold_hit_count_delta"
    ] == 1
    assert comparison["splits"]["scifact"]["query_budget_parity"] == {
        "case_count": 1,
        "equal_selected_subquery_count": 1,
        "original_query_first_in_both": 1,
    }
    assert comparison["cross_dataset"]["all_splits_non_regression"] is True
    assert (output / "comparison.json").exists()
    assert len((output / "per_query.jsonl").read_text().splitlines()) == 3


def test_analysis_outputs_are_deterministic(tmp_path: Path) -> None:
    paths = {}
    for split, config in {
        "scifact": ("beir_scifact", "test", 0, 50),
        "development": ("auto_scholar_query", "development", 0, 10),
        "validation": ("auto_scholar_query", "validation", 10, 5),
    }.items():
        for policy in ("current_rules", "concept_projection"):
            paths[(split, policy)] = _run(
                tmp_path,
                split,
                policy,
                config=config,
                hit=False,
            )
    kwargs = {
        "scifact_current": paths[("scifact", "current_rules")],
        "scifact_projection": paths[("scifact", "concept_projection")],
        "development_current": paths[("development", "current_rules")],
        "development_projection": paths[("development", "concept_projection")],
        "validation_current": paths[("validation", "current_rules")],
        "validation_projection": paths[("validation", "concept_projection")],
    }

    build_concept_projection_analysis(**kwargs, output_dir=tmp_path / "first")
    build_concept_projection_analysis(**kwargs, output_dir=tmp_path / "second")

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
    common_config = {
        "dataset": dataset,
        "dataset_sha256": "fixture",
        "dataset_split": dataset_split,
        "case_ids": [f"{split}-1"],
        "offset": offset,
        "limit": limit,
        "sources": ["arxiv", "openalex", "semantic_scholar", "pubmed"],
        "query_adapter_policy": "adaptive",
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": 2026,
        "max_workers": 1,
        "budgets": {"max_candidate_papers": 200},
        "diagnostics": True,
        "llm": {"query_understanding": False, "judgement": False},
        "enable_query_evolution": False,
        "query_evolution_policy": "off",
        "enable_refchain": False,
        "query_planning_policy": policy,
    }
    metrics = {
        "aggregate": {
            "recall_at_k": {"20": float(hit)},
            "precision_at_k": {"20": float(hit) / 20},
            "f1_at_k": {"20": 0.1 if hit else 0.0},
            "ndcg_at_k": {"20": float(hit)},
            "mrr": float(hit),
        },
        "per_case": [
            {
                "case_id": f"{split}-1",
                "metrics": {
                    "recall_at_k": {"20": float(hit)},
                    "f1_at_k": {"20": 0.1 if hit else 0.0},
                },
            }
        ],
        "snapshot_costs": {
            "recorded_retry_count": 0,
            "recorded_latency_seconds": 1.0,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    stage = {
        "case_count": 1,
        "initial_retrieval_recall": float(hit),
        "initial_query_planning": {
            "effective_request_count": 4,
            "recorded_request_count": 4,
            "source_error_count": 0,
            "recorded_latency_seconds": 1.0,
            "unique_candidate_count": 1 + int(hit),
            "unique_gold_count": int(hit),
        },
        "source_contribution": {
            "sources": {
                "arxiv": {
                    "returned_candidate_count": 1 + int(hit),
                    "unique_candidate_count": 1 + int(hit),
                    "gold_hit_count": int(hit),
                    "unique_gold_hit_count": int(hit),
                    "success_count": 1,
                    "error_count": 0,
                }
            }
        },
    }
    result = {
        "case_id": f"{split}-1",
        "query": "graph retrieval",
        "status": "succeeded",
        "stage_diagnostics": {
            "stage_metrics": {
                "candidate_recall": {"initial_retrieval": float(hit)}
            },
            "initial_query_planning": {
                "source_error_count": 0,
                "unique_gold_count": int(hit),
                "planning": {
                    "concept_projection_replaced_query": (
                        "graph retrieval papers"
                        if policy == "concept_projection"
                        else None
                    ),
                    "concept_projection_skip_reason": None,
                    "selected_subquery_count": 1,
                    "selected_subqueries": [
                        {
                            "query": "graph retrieval",
                            "purpose": "original_query",
                        }
                    ],
                },
                "subqueries": [
                    {
                        "purpose": (
                            "concept_projection"
                            if policy == "concept_projection"
                            else "normalized_keywords"
                        ),
                        "exclusive_candidate_count": int(hit),
                        "post_run_unique_gold_hit_count": int(hit),
                    }
                ],
            },
        },
    }
    _write(run / "config.json", common_config)
    _write(run / "metrics.json", metrics)
    _write(run / "stage_metrics.json", stage)
    (run / "results.jsonl").write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
