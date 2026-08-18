#!/usr/bin/env python3
"""Offline gate for the fixed 200-query contest qualification runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.resource_accounting import (  # noqa: E402
    ResourceLedgerV1,
    validate_resource_ledger,
)


EXPECTED_BASELINE = "contest_qual200_bm25_v1"
EXPECTED_CANDIDATES = {
    "contest_qual200_dense_v1",
    "contest_qual200_reranker_v1",
}
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_ITERATIONS = 5000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def check_qualification(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline_run = _load_run(baseline, EXPECTED_BASELINE)
    candidate_run = _load_run(candidate, candidate.name)
    if candidate.name not in EXPECTED_CANDIDATES:
        raise ValueError("candidate_run_id_invalid")
    if baseline_run["config_hashes"] != candidate_run["config_hashes"]:
        raise ValueError("qualification_shared_config_drift")
    baseline_cases = _case_metrics(baseline_run["metrics"])
    candidate_cases = _case_metrics(candidate_run["metrics"])
    if list(baseline_cases) != list(candidate_cases):
        raise ValueError("qualification_query_order_or_identity_drift")
    comparisons = {
        metric: _bootstrap_delta(
            [candidate_cases[key][metric] - baseline_cases[key][metric] for key in baseline_cases]
        )
        for metric in ("f1_at_20", "recall_at_20")
    }
    improvements = {
        metric: value["mean_delta"] > 0 and value["ci95"]["low"] > 0
        for metric, value in comparisons.items()
    }
    resource = {
        "baseline": baseline_run["resource_report"],
        "candidate": candidate_run["resource_report"],
    }
    resource_passed = all(item.get("status") == "passed" for item in resource.values())
    return {
        "schema_version": "contest-qualification-gate-v1",
        "baseline_run_id": baseline.name,
        "candidate_run_id": candidate.name,
        "query_count": len(baseline_cases),
        "metrics": comparisons,
        "strict_positive_improvement": improvements,
        "resource_ledger_passed": resource_passed,
        "eligible_for_full_1000": any(improvements.values()) and resource_passed,
        "internal_metric_scope": "not_official_competition_scorer",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = check_qualification(args.baseline, args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"qualification_gate_failed:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["eligible_for_full_1000"] else 2


def _load_run(path: Path, expected_run_id: str) -> dict[str, Any]:
    if path.name != expected_run_id:
        raise ValueError("run_id_does_not_match_requested_qualification")
    required = ("config.json", "metrics.json", "resource_ledger.json")
    if any(not (path / name).is_file() for name in required):
        raise ValueError("qualification_required_artifact_missing")
    completed = list((path / ".run_commits" / "generations").glob("generation-*/RUN_COMPLETED"))
    if not completed:
        raise ValueError("qualification_run_not_completed")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    case_count = int(metrics.get("case_statistics", {}).get("total_case_count", 0))
    if case_count != 200:
        raise ValueError("qualification_requires_exactly_200_queries")
    if int(metrics.get("case_statistics", {}).get("failed_case_count", 1)) != 0:
        raise ValueError("qualification_contains_failed_queries")
    ledger = ResourceLedgerV1.model_validate_json(
        (path / "resource_ledger.json").read_text(encoding="utf-8")
    )
    resource_report = validate_resource_ledger(ledger)
    config_hashes = {
        "dataset": config.get("dataset"),
        "dataset_split": config.get("dataset_split"),
        "offset": config.get("offset"),
        "limit": config.get("limit"),
        "top_k": config.get("top_k"),
        "query_adapter_policy": config.get("query_adapter_policy"),
        "judgement_policy": config.get("judgement_policy"),
        "data_hashes": config.get("data_hashes") or config.get("input_hashes"),
    }
    return {"config": config, "config_hashes": config_hashes, "metrics": metrics, "resource_report": resource_report}


def _case_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = metrics.get("per_case") or []
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        values = row.get("metrics") or {}
        result[case_id] = {
            "f1_at_20": float((values.get("f1_at_k") or {}).get("20", 0.0)),
            "recall_at_20": float((values.get("recall_at_k") or {}).get("20", 0.0)),
        }
    if len(result) != 200:
        raise ValueError("qualification_per_case_metrics_missing_or_duplicate")
    return result


def _bootstrap_delta(differences: list[float]) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(differences)
    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        samples.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    samples.sort()
    return {
        "mean_delta": sum(differences) / count,
        "ci95": {"low": samples[int(0.025 * (len(samples) - 1))], "high": samples[int(0.975 * (len(samples) - 1))]},
        "seed": BOOTSTRAP_SEED,
        "iterations": BOOTSTRAP_ITERATIONS,
    }


if __name__ == "__main__":
    raise SystemExit(main())
