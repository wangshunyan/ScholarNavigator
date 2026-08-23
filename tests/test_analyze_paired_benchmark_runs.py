from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_paired_benchmark_runs import analyze_paired_runs


def _write_run(
    root: Path,
    name: str,
    *,
    candidate: bool = False,
    drift: bool = False,
    runtime_code_hash: str = "runtime-hash",
) -> Path:
    run = root / name
    run.mkdir()
    case_ids = ["case-0", "case-1"]
    config = {
        "dataset": "fixture",
        "dataset_split": "test",
        "dataset_sha256": "dataset-hash",
        "case_ids": case_ids,
        "offset": 0,
        "limit": 2,
        "query_adapter_policy": "adaptive",
        "run_profile": "high_recall",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": None,
        "max_workers": 1,
        "budgets": {"max_candidate_papers": 300},
        "diagnostics": True,
        "llm": {"requested": False},
        "prompts": [],
        "runtime_code_hash": runtime_code_hash,
        "sources": ["candidate"] if candidate else ["baseline"],
        "ranking_policy": "candidate" if candidate else "current_rules",
    }
    if drift:
        config["run_profile"] = "fast"
    rows = []
    for index, case_id in enumerate(case_ids):
        base = 0.1 * (index + 1)
        gain = 0.05 if candidate else 0.0
        rows.append(
            {
                "case_id": case_id,
                "metrics": {
                    "f1_at_k": {"5": base + gain, "10": base + gain, "20": base + gain},
                    "recall_at_k": {"20": base + gain},
                },
            }
        )
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"per_case": rows}), encoding="utf-8"
    )
    return run


def test_analyze_paired_runs_reports_deterministic_ci(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)

    report = analyze_paired_runs(baseline, candidate, seed=7, iterations=200)

    assert report["shared_inputs_match"] is True
    assert report["query_count"] == 2
    assert report["strict_positive_improvement"]["f1_at_20"] is True
    assert report["comparisons"]["f1_at_20"]["seed"] == 7
    assert report["comparisons"]["f1_at_20"]["iterations"] == 200
    assert report["reported_config_differences"]["sources"]["candidate"] == [
        "candidate"
    ]


def test_analyze_paired_runs_rejects_shared_config_drift(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True, drift=True)

    with pytest.raises(ValueError, match="shared_config_drift:run_profile"):
        analyze_paired_runs(baseline, candidate)


def test_analyze_paired_runs_reports_runtime_code_change(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", runtime_code_hash="old-code")
    candidate = _write_run(
        tmp_path,
        "candidate",
        candidate=True,
        runtime_code_hash="new-code",
    )

    report = analyze_paired_runs(baseline, candidate, seed=7, iterations=200)

    assert report["shared_inputs_match"] is True
    assert report["reported_config_differences"]["runtime_code_hash"] == {
        "baseline": "old-code",
        "candidate": "new-code",
    }


def test_analyze_paired_runs_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    metrics["per_case"][1]["case_id"] = "case-0"
    (candidate / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="per_case_metric_identity_invalid"):
        analyze_paired_runs(baseline, candidate)
