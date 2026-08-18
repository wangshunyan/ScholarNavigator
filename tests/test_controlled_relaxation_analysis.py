from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_controlled_relaxation import (
    build_controlled_relaxation_analysis,
)


def test_analysis_accepts_frozen_non_regression_with_new_gold(
    tmp_path: Path,
) -> None:
    paths = _comparison_runs(tmp_path)

    result = build_controlled_relaxation_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    acceptance = result["validation_acceptance"]
    assert acceptance["accepted"] is True
    assert acceptance["unique_gold_gain"] == 1
    assert result["product_default"] == "controlled_relaxation"
    assert result["development_rule_status"] == "frozen_before_validation"
    contribution = result["splits"]["validation"]["controlled_relaxation"][
        "supplemental_query_contribution"
    ]
    assert contribution["controlled_core_topic"] == {
        "exclusive_candidate_count": 4,
        "post_run_unique_gold_hit_count": 1,
    }
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    assert (tmp_path / "analysis" / "summary.md").is_file()


def test_analysis_rejects_replay_that_records_network_activity(
    tmp_path: Path,
) -> None:
    paths = _comparison_runs(tmp_path, controlled_replay_requests=1)

    result = build_controlled_relaxation_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    checks = result["validation_acceptance"]["checks"]
    assert checks["frozen_replay_zero_network"] is False
    assert result["product_default"] == "current_rules"


def test_analysis_enforces_untouched_fixed_validation_split(
    tmp_path: Path,
) -> None:
    paths = _comparison_runs(tmp_path)
    wrong = _run(
        tmp_path,
        "validation-controlled-wrong",
        "controlled_relaxation",
        offset=71,
        unique_gold=3,
        requests=15,
    )
    paths["validation_controlled"] = wrong

    with pytest.raises(ValueError, match="incompatible|unexpected fixed split"):
        build_controlled_relaxation_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def test_analysis_rejects_llm_or_later_stage_changes(tmp_path: Path) -> None:
    paths = _comparison_runs(tmp_path)
    config_path = paths["validation_controlled"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["llm"]["query_understanding"] = True
    _json(config_path, config)

    with pytest.raises(ValueError, match="incompatible|LLM off"):
        build_controlled_relaxation_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def _comparison_runs(
    root: Path,
    *,
    controlled_replay_requests: int = 0,
) -> dict[str, Path]:
    return {
        "development_current": _run(
            root, "development-current", "current_rules", offset=50
        ),
        "development_controlled": _run(
            root,
            "development-controlled",
            "controlled_relaxation",
            offset=50,
            unique_gold=3,
            requests=15,
        ),
        "validation_current": _run(
            root, "validation-current", "current_rules", offset=70
        ),
        "validation_controlled": _run(
            root,
            "validation-controlled",
            "controlled_relaxation",
            offset=70,
            unique_gold=3,
            requests=15,
            replay_requests=controlled_replay_requests,
        ),
    }


def _run(
    root: Path,
    name: str,
    policy: str,
    *,
    offset: int,
    unique_gold: int = 2,
    requests: int = 10,
    replay_requests: int = 0,
) -> Path:
    path = root / name
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_ids": [f"case-{offset}"],
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
    }
    metric = 0.3
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
            "replay_execution_request_count": replay_requests,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    planning = {
        "case_count": 1,
        "subquery_count": 2,
        "average_subquery_count": 2.0,
        "adapted_query_count": 3,
        "average_adapted_query_count": 3.0,
        "unique_candidate_count": 20,
        "duplicate_candidate_ratio": 0.2,
        "unique_gold_count": unique_gold,
        "effective_request_count": requests,
        "recorded_latency_seconds": 3.0,
        "source_error_rate": 0.0,
        "facet_contribution": {},
        "ineffective_reasons": {"over_restrictive": 1},
    }
    stage_metrics = {
        "case_count": 1,
        "initial_retrieval_recall": 0.5,
        "initial_query_planning": planning,
    }
    purpose = (
        "controlled_core_topic"
        if policy == "controlled_relaxation"
        else "normalized_keywords"
    )
    result = {
        "case_id": f"case-{offset}",
        "query": "fixture query",
        "status": "succeeded",
        "stage_diagnostics": {
            "initial_query_planning": {
                "planner_version": "1.4.0",
                **planning,
                "subqueries": [
                    {
                        "purpose": purpose,
                        "exclusive_candidate_count": 4,
                        "post_run_unique_gold_hit_count": 1,
                    }
                ],
            },
            "snapshots": [
                {
                    "stage": "initial_judged",
                    "candidates": [
                        {"category": "irrelevant"},
                        *({"category": "partially_relevant"} for _ in range(9)),
                    ],
                }
            ],
            "gold_diagnostics": [{"drop_reason": "not_retrieved"}],
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
