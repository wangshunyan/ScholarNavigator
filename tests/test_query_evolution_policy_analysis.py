from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_evolution_policies import analyze


def _write_run(
    root: Path,
    *,
    policy: str,
    f1: float,
    recall: float,
    api_calls: int,
    categories: dict[str, int],
    new_gold: int,
    triggered: bool,
) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"query_evolution_policy": policy}),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "end_to_end_metrics": {
                    "f1_at_k": {"20": f1},
                    "recall_at_k": {"20": recall},
                    "precision_at_k": {"20": f1},
                    "mrr": f1,
                    "ndcg_at_k": {"20": f1},
                }
            }
        ),
        encoding="utf-8",
    )
    module = {
        "policy": policy,
        "triggered": triggered,
        "selected_seed_count": int(triggered),
        "generated_query_count": int(triggered),
        "evolved_raw_candidate_count": sum(categories.values()),
        "evolved_unique_candidate_count": sum(categories.values()),
        "evolved_new_unique_candidate_count": sum(categories.values()),
        "evolved_new_unique_gold_count": new_gold,
        "new_candidate_categories": {"counts": categories},
        "quality_gate": {
            "raw_candidate_count": sum(categories.values()),
            "filtered_candidate_count": categories.get("irrelevant", 0),
        },
        "skipped_reasons": [] if triggered else ["coverage_sufficient"],
        "queries": [],
    }
    row = {
        "case_id": "case-1",
        "query": "graph retrieval",
        "status": "succeeded",
        "latency_seconds": 1.0,
        "cost_report": {"search_api_call_count": api_calls},
        "snapshot_cost_report": {
            "recorded_search_request_count": api_calls,
            "recorded_latency_seconds": float(api_calls),
        },
        "stage_diagnostics": {"query_evolution": module},
    }
    (root / "results.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    return root


def test_analysis_writes_diagnostics_and_enforces_acceptance(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path / "baseline",
        policy="off",
        f1=0.1,
        recall=0.2,
        api_calls=2,
        categories={},
        new_gold=0,
        triggered=False,
    )
    seed = _write_run(
        tmp_path / "seed",
        policy="seed_expansion",
        f1=0.1,
        recall=0.2,
        api_calls=5,
        categories={"irrelevant": 3, "partially_relevant": 1},
        new_gold=0,
        triggered=True,
    )
    gap = _write_run(
        tmp_path / "gap",
        policy="coverage_gap",
        f1=0.1,
        recall=0.2,
        api_calls=3,
        categories={"partially_relevant": 2},
        new_gold=1,
        triggered=True,
    )

    payload = analyze(
        baseline=baseline,
        seed_expansion=seed,
        coverage_gap=gap,
        output_dir=tmp_path / "analysis",
        label="验证集",
    )

    assert payload["acceptance"]["passed"] is True
    assert payload["policies"]["coverage_gap"]["api_calls_per_new_gold"] == 3.0
    assert payload["policies"]["seed_expansion"]["api_calls_per_new_gold"] is None
    assert (tmp_path / "analysis" / "comparison.json").is_file()
    assert (tmp_path / "analysis" / "query_diagnostics.jsonl").is_file()
    summary = (tmp_path / "analysis" / "summary.md").read_text(encoding="utf-8")
    assert "验证集" in summary
    assert "演化查询无效原因" in summary
