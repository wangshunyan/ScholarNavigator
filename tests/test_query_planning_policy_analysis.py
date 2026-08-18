from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_planning_policies import (  # noqa: E402
    build_query_planning_analysis,
)


def test_analysis_writes_split_diagnostics_and_rejects_cost_without_gold_gain(
    tmp_path: Path,
) -> None:
    dev_current = _run(tmp_path, "dev-current", "current_rules", offset=0)
    dev_facet = _run(
        tmp_path,
        "dev-facet",
        "facet_balanced",
        offset=0,
        candidate_recall=0.6,
        requests=12,
        unique_gold=2,
    )
    validation_current = _run(tmp_path, "val-current", "current_rules", offset=10)
    validation_facet = _run(
        tmp_path,
        "val-facet",
        "facet_balanced",
        offset=10,
        candidate_recall=0.5,
        requests=12,
        unique_gold=2,
    )
    output_dir = tmp_path / "analysis"

    result = build_query_planning_analysis(
        development_current=dev_current,
        development_facet=dev_facet,
        validation_current=validation_current,
        validation_facet=validation_facet,
        output_dir=output_dir,
    )

    acceptance = result["validation_acceptance"]
    assert acceptance["accepted"] is False
    assert acceptance["checks"]["not_higher_cost_without_new_gold"] is False
    assert result["product_default"] == "current_rules"
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "summary.md").is_file()
    dev_rows = _jsonl(output_dir / "development_query_diagnostics.jsonl")
    validation_rows = _jsonl(output_dir / "validation_query_diagnostics.jsonl")
    assert [row["policy"] for row in dev_rows] == [
        "current_rules",
        "facet_balanced",
    ]
    assert {row["split"] for row in validation_rows} == {"validation"}
    assert result["splits"]["development"]["facet_balanced"][
        "replay_execution_request_count"
    ] == 0


def test_analysis_accepts_non_regression_with_new_gold_and_bounded_cost(
    tmp_path: Path,
) -> None:
    dev_current = _run(tmp_path, "dev-current", "current_rules", offset=0)
    dev_facet = _run(tmp_path, "dev-facet", "facet_balanced", offset=0)
    validation_current = _run(tmp_path, "val-current", "current_rules", offset=10)
    validation_facet = _run(
        tmp_path,
        "val-facet",
        "facet_balanced",
        offset=10,
        candidate_recall=0.6,
        metric=0.3,
        requests=12,
        unique_gold=3,
        duplicate_ratio=0.22,
    )

    result = build_query_planning_analysis(
        development_current=dev_current,
        development_facet=dev_facet,
        validation_current=validation_current,
        validation_facet=validation_facet,
        output_dir=tmp_path / "analysis",
    )

    assert result["validation_acceptance"]["accepted"] is True
    assert result["product_default"] == "facet_balanced"


def test_analysis_rejects_incomparable_split(tmp_path: Path) -> None:
    dev_current = _run(tmp_path, "dev-current", "current_rules", offset=0)
    dev_facet = _run(
        tmp_path,
        "dev-facet",
        "facet_balanced",
        offset=1,
    )
    val_current = _run(tmp_path, "val-current", "current_rules", offset=10)
    val_facet = _run(tmp_path, "val-facet", "facet_balanced", offset=10)

    with pytest.raises(ValueError, match="incompatible query planning runs"):
        build_query_planning_analysis(
            development_current=dev_current,
            development_facet=dev_facet,
            validation_current=val_current,
            validation_facet=val_facet,
            output_dir=tmp_path / "analysis",
        )


def _run(
    root: Path,
    name: str,
    policy: str,
    *,
    offset: int,
    candidate_recall: float = 0.5,
    metric: float = 0.2,
    requests: int = 10,
    unique_gold: int = 2,
    duplicate_ratio: float = 0.2,
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
        "llm": {"query_understanding": False, "judgement": False},
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
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    planning = {
        "case_count": 1,
        "subquery_count": 2,
        "average_subquery_count": 2.0,
        "adapted_query_count": 2,
        "average_adapted_query_count": 2.0,
        "unique_candidate_count": 20,
        "duplicate_candidate_ratio": duplicate_ratio,
        "unique_gold_count": unique_gold,
        "effective_request_count": requests,
        "recorded_latency_seconds": 3.0,
        "source_error_rate": 0.0,
        "facet_contribution": {},
        "ineffective_reasons": {},
    }
    stage_metrics = {
        "case_count": 1,
        "initial_retrieval_recall": candidate_recall,
        "initial_query_planning": planning,
    }
    result = {
        "case_id": f"case-{offset}",
        "query": "fixture query",
        "status": "succeeded",
        "stage_diagnostics": {
            "initial_query_planning": {
                "planner_version": "1.0.0",
                **planning,
                "subqueries": [],
            }
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


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
