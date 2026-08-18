from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_llm_query_planning import (  # noqa: E402
    build_llm_query_planning_analysis,
    build_unavailable_llm_baseline_analysis,
)


def test_analysis_keeps_llm_optional_when_validation_has_no_gain(tmp_path: Path) -> None:
    dev_current = _run(tmp_path, "dev-current", "current_rules", 0)
    dev_llm = _run(tmp_path, "dev-llm", "llm_semantic", 0, llm_calls=1)
    val_current = _run(tmp_path, "val-current", "current_rules", 10)
    val_llm = _run(
        tmp_path,
        "val-llm",
        "llm_semantic",
        10,
        requests=12,
        llm_calls=1,
    )
    output = tmp_path / "analysis"

    result = build_llm_query_planning_analysis(
        development_current=dev_current,
        development_llm=dev_llm,
        validation_current=val_current,
        validation_llm=val_llm,
        output_dir=output,
    )

    assert result["validation_acceptance"]["accepted"] is False
    assert result["product_default"] == "current_rules"
    assert result["llm_semantic_default_enabled"] is False
    assert (output / "comparison.json").is_file()
    assert (output / "summary.md").is_file()
    assert len(_jsonl(output / "development_query_diagnostics.jsonl")) == 2


def test_analysis_acceptance_requires_new_gold_or_ranking_lift(tmp_path: Path) -> None:
    result = build_llm_query_planning_analysis(
        development_current=_run(tmp_path, "dc", "current_rules", 0),
        development_llm=_run(tmp_path, "dl", "llm_semantic", 0),
        validation_current=_run(tmp_path, "vc", "current_rules", 10),
        validation_llm=_run(
            tmp_path,
            "vl",
            "llm_semantic",
            10,
            metric=0.3,
            added_gold=1,
        ),
        output_dir=tmp_path / "analysis",
    )

    assert result["validation_acceptance"]["accepted"] is True


def test_unavailable_llm_analysis_does_not_fabricate_llm_metrics(
    tmp_path: Path,
) -> None:
    result = build_unavailable_llm_baseline_analysis(
        development_current=_run(tmp_path, "dc", "current_rules", 0),
        validation_current=_run(tmp_path, "vc", "current_rules", 10),
        reason="provider_disabled",
        output_dir=tmp_path / "analysis",
    )

    assert result["experiment_status"] == "llm_not_run"
    assert result["splits"]["development"]["llm_semantic"] is None
    assert result["validation_acceptance"]["evaluated"] is False


def _run(
    root: Path,
    name: str,
    policy: str,
    offset: int,
    *,
    metric: float = 0.2,
    requests: int = 10,
    llm_calls: int = 0,
    added_gold: int = 0,
) -> Path:
    path = root / name
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_ids": [f"case-{offset}"],
        "offset": offset,
        "limit": 1,
        "sources": ["arxiv"],
        "query_adapter_policy": "adaptive",
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": 2026,
        "max_workers": 1,
        "budgets": {"max_search_rounds": 2},
        "diagnostics": True,
        "enable_query_evolution": False,
        "query_evolution_policy": "off",
        "enable_refchain": False,
        "query_planning_policy": policy,
    }
    metrics = {
        "case_count": 1,
        "end_to_end_metrics": {
            "f1_at_k": {"5": metric, "10": metric, "20": metric},
            "precision_at_k": {"20": metric},
            "recall_at_k": {"20": metric},
            "ndcg_at_k": {"20": metric},
            "mrr": metric,
        },
        "snapshot_costs": {
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0,
        },
        "llm_planning_costs": {
            "live_call_count": llm_calls,
            "total_tokens": 20 * llm_calls,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0,
        },
    }
    planning_summary = {
        "case_count": 1,
        "subquery_count": 2,
        "adapted_query_count": 2,
        "unique_candidate_count": 20,
        "unique_gold_count": 2 + added_gold,
        "duplicate_candidate_ratio": 0.2,
        "effective_request_count": requests,
        "recorded_latency_seconds": 2.0,
    }
    result = {
        "case_id": f"case-{offset}",
        "query": "fixture query",
        "status": "succeeded",
        "stage_diagnostics": {
            "initial_query_planning": {
                **planning_summary,
                "planning": {
                    "policy": policy,
                    "output_valid": policy == "llm_semantic",
                    "fallback_used": False,
                    "llm_call_attempted": bool(llm_calls),
                },
                "subqueries": (
                    [
                        {
                            "purpose": "llm_semantic:expansion",
                            "exclusive_candidate_count": 3,
                            "post_run_unique_gold_hit_count": added_gold,
                            "recorded_request_count": 2,
                        }
                    ]
                    if policy == "llm_semantic"
                    else []
                ),
            }
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "stage_metrics.json").write_text(
        json.dumps(
            {
                "case_count": 1,
                "initial_retrieval_recall": metric,
                "initial_query_planning": planning_summary,
            }
        ),
        encoding="utf-8",
    )
    (path / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]
