from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eval_and_summary_scripts_run_with_sample_fixtures(tmp_path: Path) -> None:
    output_root = tmp_path / "eval_runs"
    eval_command = [
        sys.executable,
        "scripts/eval_search_service.py",
        "--fixtures-dir",
        "datasets/eval_fixtures/sample",
        "--output-root",
        str(output_root),
        "--run-id",
        "script-test",
        "--max-workers",
        "1",
    ]
    eval_result = subprocess.run(
        eval_command,
        check=True,
        text=True,
        capture_output=True,
    )
    result_path = output_root / "script-test" / "result.json"

    assert str(result_path) in eval_result.stdout
    assert result_path.exists()

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(data["aggregate_metrics"]) == {
        "baseline",
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    }
    assert data["aggregate_metrics"]["query_evolution_only"]["raw_count"] > data[
        "aggregate_metrics"
    ]["baseline"]["raw_count"]
    assert set(data["aggregate_reports"]) == set(data["aggregate_metrics"])
    assert "f1_at_k" in data["aggregate_reports"]["baseline"]["end_to_end_metrics"]

    summary_command = [
        sys.executable,
        "scripts/summarize_eval_results.py",
        str(result_path),
    ]
    summary_result = subprocess.run(
        summary_command,
        check=True,
        text=True,
        capture_output=True,
    )
    summary_path = result_path.with_name("summary.md")
    summary = summary_path.read_text(encoding="utf-8")

    assert str(summary_path) in summary_result.stdout
    assert "| 分组 | F1@5 | F1@10 | F1@20 |" in summary
    assert "| baseline |" in summary
    assert "| query_evolution_only |" in summary
    assert "| refchain_only |" in summary
    assert "| query_evolution_plus_refchain |" in summary
    assert "sample fixture 仅验证评测流程，不代表真实 benchmark 性能" in summary
