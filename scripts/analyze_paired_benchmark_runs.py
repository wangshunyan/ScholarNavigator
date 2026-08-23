#!/usr/bin/env python3
"""Produce a deterministic paired analysis for two completed benchmark runs.

This utility consumes only ``config.json`` and evaluator-produced
``metrics.json`` from each run.  It never loads the query gold, qrels, result
documents, or any online connector, so it cannot leak evaluator data into
retrieval.  Source/policy differences are reported explicitly while shared
execution inputs are required to match.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


_SHARED_FIELDS = (
    "dataset",
    "dataset_split",
    "dataset_sha256",
    "case_ids",
    "offset",
    "limit",
    "query_adapter_policy",
    "run_profile",
    "result_policy",
    "top_k",
    "current_year",
    "max_workers",
    "budgets",
    "diagnostics",
    "llm",
    "prompts",
)
_REPORTED_DIFFERENCES = (
    "runtime_code_hash",
    "sources",
    "ranking_policy",
    "judgement_policy",
    "query_planning_policy",
    "query_evolution_policy",
    "local_bm25",
    "local_hybrid",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--iterations", type=int, default=5000)
    return parser


def analyze_paired_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    seed: int = 20260818,
    iterations: int = 5000,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed_must_be_non_negative")
    if iterations <= 0:
        raise ValueError("iterations_must_be_positive")
    baseline = _load_run(baseline_dir)
    candidate = _load_run(candidate_dir)
    baseline_config = baseline["config"]
    candidate_config = candidate["config"]
    mismatched = [
        field
        for field in _SHARED_FIELDS
        if baseline_config.get(field) != candidate_config.get(field)
    ]
    if mismatched:
        raise ValueError("shared_config_drift:" + ",".join(mismatched))

    baseline_cases = _case_metrics(baseline["metrics"])
    candidate_cases = _case_metrics(candidate["metrics"])
    if list(baseline_cases) != list(candidate_cases):
        raise ValueError("query_order_or_identity_drift")
    differences = {
        metric: [
            candidate_cases[case_id][metric] - baseline_cases[case_id][metric]
            for case_id in baseline_cases
        ]
        for metric in ("f1_at_5", "f1_at_10", "f1_at_20", "recall_at_20")
    }
    comparisons = {
        metric: _bootstrap_delta(values, seed=seed, iterations=iterations)
        for metric, values in differences.items()
    }
    return {
        "schema_version": "paired-benchmark-analysis-v1",
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "query_count": len(baseline_cases),
        "baseline_runtime_code_hash": baseline_config.get("runtime_code_hash"),
        "candidate_runtime_code_hash": candidate_config.get("runtime_code_hash"),
        "shared_inputs_match": True,
        "reported_config_differences": {
            field: {
                "baseline": _redact_config_value(
                    baseline_config.get(field)
                ),
                "candidate": _redact_config_value(
                    candidate_config.get(field)
                ),
            }
            for field in _REPORTED_DIFFERENCES
            if baseline_config.get(field) != candidate_config.get(field)
        },
        "comparisons": comparisons,
        "strict_positive_improvement": {
            metric: value["mean_delta"] > 0 and value["ci95"]["low"] > 0
            for metric, value in comparisons.items()
        },
        "internal_metric_scope": "not_official_competition_scorer",
    }


def _load_run(path: Path) -> dict[str, Any]:
    run_dir = path.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    run_id = str(run_dir.name)
    if not isinstance(config, dict) or not isinstance(metrics, dict):
        raise ValueError("benchmark_run_json_must_be_objects")
    return {"run_id": run_id, "config": config, "metrics": metrics}


def _case_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = metrics.get("per_case")
    if not isinstance(rows, list) or not rows:
        raise ValueError("per_case_metrics_missing")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("per_case_metric_row_invalid")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in result:
            raise ValueError("per_case_metric_identity_invalid")
        values = row.get("metrics")
        if not isinstance(values, dict):
            raise ValueError("per_case_metric_values_missing")
        f1 = values.get("f1_at_k") or {}
        recall = values.get("recall_at_k") or {}
        result[case_id] = {
            "f1_at_5": float(f1.get("5", 0.0)),
            "f1_at_10": float(f1.get("10", 0.0)),
            "f1_at_20": float(f1.get("20", 0.0)),
            "recall_at_20": float(recall.get("20", 0.0)),
        }
    return result


def _bootstrap_delta(
    differences: list[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    if not differences:
        raise ValueError("paired_difference_empty")
    rng = random.Random(seed)
    count = len(differences)
    samples = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    ]
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return {
        "mean_delta": sum(differences) / count,
        "ci95": {"low": samples[low_index], "high": samples[high_index]},
        "seed": seed,
        "iterations": iterations,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"benchmark_run_input_unavailable:{path.name}") from exc


def _redact_config_value(value: Any, *, key: str = "") -> Any:
    """Keep comparison context while excluding machine-specific paths."""

    lowered = key.casefold()
    if lowered.endswith("path") or lowered.endswith("dir"):
        return "<redacted-path>" if value is not None else None
    if isinstance(value, dict):
        return {
            str(item_key): _redact_config_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_config_value(item, key=key) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = analyze_paired_runs(
            args.baseline,
            args.candidate,
            seed=args.seed,
            iterations=args.iterations,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
