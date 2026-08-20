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
from scripts.audit_contest_llm_run import audit_run  # noqa: E402


EXPECTED_BASELINE = "contest_qual200_bm25_v1"
SOFT_JUDGEMENT_BASELINE = "contest_qual200_reranker_v4_gpu1"
SOFT_JUDGEMENT_CANDIDATE = "contest_qual200_dense_reranker_soft_v1"
LLM_QUALIFICATION_BASELINE = "contest_qual200_reranker_v4_gpu1"
LLM_QUALIFICATION_CANDIDATE = "contest_qual200_dense_reranker_llm_v15"
EXPECTED_CANDIDATES = {
    "contest_qual200_dense_v1",
    "contest_qual200_reranker_v1",
    "contest_qual200_reranker_v2",
    "contest_qual200_reranker_v2_gpu1",
    "contest_qual200_reranker_v3_gpu1",
    "contest_qual200_reranker_v4_gpu1",
    "contest_qual200_dense_reranker_soft_v1",
    LLM_QUALIFICATION_CANDIDATE,
}
RERANKER_PROMPT_VERSION = "qwen3-reranker-v1"
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_ITERATIONS = 5000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def check_qualification(baseline: Path, candidate: Path) -> dict[str, Any]:
    expected_baseline = _expected_baseline_for(candidate.name)
    baseline_run = _load_run(baseline, expected_baseline)
    candidate_run = _load_run(candidate, candidate.name)
    if candidate.name not in EXPECTED_CANDIDATES:
        raise ValueError("candidate_run_id_invalid")
    if candidate.name == SOFT_JUDGEMENT_CANDIDATE:
        _validate_soft_judgement_pair(baseline_run["config"], candidate_run["config"])
    elif candidate.name == LLM_QUALIFICATION_CANDIDATE:
        _validate_llm_pair(baseline_run["config"], candidate_run["config"])
    elif baseline_run["config_hashes"] != candidate_run["config_hashes"]:
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
    reranker_audit = candidate_run.get("reranker_audit")
    reranker_passed = reranker_audit is None or reranker_audit.get("status") == "passed"
    llm_audit = candidate_run.get("llm_audit")
    llm_passed = llm_audit is None or llm_audit.get("status") == "passed"
    return {
        "schema_version": "contest-qualification-gate-v1",
        "baseline_run_id": baseline.name,
        "candidate_run_id": candidate.name,
        "query_count": len(baseline_cases),
        "metrics": comparisons,
        "strict_positive_improvement": improvements,
        "resource_ledger_passed": resource_passed,
        "reranker_audit": reranker_audit,
        "reranker_audit_passed": reranker_passed,
        "llm_audit": llm_audit,
        "llm_audit_passed": llm_passed,
        "eligible_for_full_1000": (
            any(improvements.values())
            and resource_passed
            and reranker_passed
            and llm_passed
        ),
        "internal_metric_scope": "not_official_competition_scorer",
    }


def _expected_baseline_for(candidate_run_id: str) -> str:
    if candidate_run_id in {SOFT_JUDGEMENT_CANDIDATE, LLM_QUALIFICATION_CANDIDATE}:
        return SOFT_JUDGEMENT_BASELINE
    return EXPECTED_BASELINE


def _validate_soft_judgement_pair(
    baseline_config: dict[str, Any], candidate_config: dict[str, Any]
) -> None:
    """Accept only the reviewed soft-threshold delta for this paired experiment."""

    comparable_keys = (
        "dataset",
        "dataset_split",
        "dataset_sha256",
        "case_count",
        "case_ids",
        "offset",
        "limit",
        "selection_order",
        "result_policy",
        "sources",
        "local_bm25",
        "local_hybrid",
        "run_profile",
        "top_k",
        "enable_query_evolution",
        "query_evolution_policy",
        "query_planning_policy",
        "ranking_policy",
        "query_planner_version",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "current_year",
        "max_workers",
        "budgets",
        "diagnostics",
        "enable_resource_ledger",
        "query_adapter_policy",
        "retrieval_mode",
        "llm_mode",
        "data_hashes",
    )
    if any(baseline_config.get(key) != candidate_config.get(key) for key in comparable_keys):
        raise ValueError("soft_judgement_shared_config_drift")

    baseline_judgement = dict(baseline_config.get("judgement_config") or {})
    candidate_judgement = dict(candidate_config.get("judgement_config") or {})
    changes = {
        key: (baseline_judgement.get(key), candidate_judgement.get(key))
        for key in sorted(set(baseline_judgement) | set(candidate_judgement))
        if baseline_judgement.get(key) != candidate_judgement.get(key)
    }
    if changes != {
        "config_version": ("current-rules-v1", "soft-current-rules-v1"),
        "partially_relevant_threshold": (0.45, 0.35),
    }:
        raise ValueError("soft_judgement_delta_not_allowlisted")


def _validate_llm_pair(
    baseline_config: dict[str, Any], candidate_config: dict[str, Any]
) -> None:
    """Allow only the reviewed planning and per-query budget change for LLM qualification."""

    comparable_keys = (
        "dataset", "dataset_split", "dataset_sha256", "case_count", "case_ids",
        "offset", "limit", "selection_order", "result_policy", "sources",
        "local_bm25", "local_hybrid", "run_profile", "top_k",
        "enable_query_evolution", "query_evolution_policy", "ranking_policy",
        "query_planner_version", "enable_refchain", "enable_semantic_seed_expansion",
        "current_year", "max_workers", "diagnostics", "enable_resource_ledger",
        "query_adapter_policy", "retrieval_mode", "data_hashes",
        "judgement_policy", "judgement_config",
    )
    if any(baseline_config.get(key) != candidate_config.get(key) for key in comparable_keys):
        raise ValueError("llm_qualification_shared_config_drift")
    if baseline_config.get("query_planning_policy") != "current_rules":
        raise ValueError("llm_qualification_baseline_policy_invalid")
    if candidate_config.get("query_planning_policy") != "llm_semantic":
        raise ValueError("llm_qualification_candidate_policy_invalid")

    baseline_budgets = dict(baseline_config.get("budgets") or {})
    candidate_budgets = dict(candidate_config.get("budgets") or {})
    candidate_calls = candidate_budgets.pop("max_llm_calls", None)
    candidate_rounds = candidate_budgets.pop("max_search_rounds", None)
    baseline_budgets.pop("max_llm_calls", None)
    baseline_budgets.pop("max_search_rounds", None)
    if baseline_budgets != candidate_budgets:
        raise ValueError("llm_qualification_budget_drift")
    if candidate_calls != 1 or candidate_rounds != 3:
        raise ValueError("llm_qualification_budget_contract_invalid")


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
    result = {
        "config": config,
        "config_hashes": config_hashes,
        "metrics": metrics,
        "resource_report": resource_report,
    }
    if "reranker" in expected_run_id:
        result["reranker_audit"] = _audit_reranker_run(path)
    if expected_run_id == LLM_QUALIFICATION_CANDIDATE:
        result["llm_audit"] = audit_run(path, expected_rows=200)
    return result


def _audit_reranker_run(path: Path, *, expected_rows: int = 200) -> dict[str, Any]:
    """Require evidence that the neural reranker actually inferred."""

    results_path = path / "results.jsonl"
    fallback_count = 0
    batch_count = 0
    success_count = 0
    candidate_count = 0
    prompt_versions: set[str] = set()
    devices: set[str] = set()
    max_lengths: set[int] = set()
    batch_sizes: set[int] = set()
    candidate_limits: set[int] = set()
    fingerprints: set[str] = set()
    latency_seconds: list[float] = []
    peak_vram_bytes = 0
    row_count = 0

    def visit(value: Any) -> None:
        nonlocal fallback_count, batch_count, success_count, candidate_count, peak_vram_bytes
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "local_model_fallback_count":
                    fallback_count += int(item or 0)
                elif key == "local_model_batch_count":
                    batch_count += int(item or 0)
                elif key == "local_model_inference_success_count":
                    success_count += int(item or 0)
                elif key == "local_model_candidate_count":
                    candidate_count += int(item or 0)
                elif key == "local_model_latency_seconds" and item:
                    latency_seconds.append(float(item))
                elif key == "local_model_peak_vram_bytes":
                    peak_vram_bytes = max(peak_vram_bytes, int(item or 0))
                elif key == "local_model_batch_size" and item:
                    batch_sizes.add(int(item))
                elif key == "local_model_candidate_limit" and item:
                    candidate_limits.add(int(item))
                elif key == "local_model_prompt_version" and item:
                    prompt_versions.add(str(item))
                elif key == "local_model_device" and item:
                    devices.add(str(item))
                elif key == "local_model_max_length" and item:
                    max_lengths.add(int(item))
                elif key == "local_model_fingerprint" and item:
                    fingerprints.add(str(item))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row_count += 1
                visit(json.loads(line))
    reasons: list[str] = []
    if row_count != expected_rows:
        reasons.append("reranker_result_row_count_invalid")
    if fallback_count != 0:
        reasons.append("reranker_fallback_detected")
    if batch_count <= 0 or success_count <= 0:
        reasons.append("reranker_inference_missing")
    if candidate_count <= 0:
        reasons.append("reranker_candidate_audit_missing")
    if prompt_versions != {RERANKER_PROMPT_VERSION}:
        reasons.append("reranker_prompt_version_invalid")
    if not devices or not fingerprints or not max_lengths:
        reasons.append("reranker_runtime_metadata_missing")
    if not all(device.startswith("cuda") for device in devices):
        reasons.append("reranker_gpu_inference_missing")
    if batch_sizes != {8} or candidate_limits != {120}:
        reasons.append("reranker_fixed_runtime_limits_missing")
    if not latency_seconds:
        reasons.append("reranker_latency_samples_missing")
    if peak_vram_bytes <= 0:
        reasons.append("reranker_peak_vram_missing")
    latency_seconds.sort()
    p50 = latency_seconds[(len(latency_seconds) - 1) * 50 // 100] if latency_seconds else 0.0
    p95 = latency_seconds[(len(latency_seconds) - 1) * 95 // 100] if latency_seconds else 0.0
    throughput = candidate_count / sum(latency_seconds) if sum(latency_seconds) > 0 else 0.0
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "result_row_count": row_count,
        "fallback_count": fallback_count,
        "batch_count": batch_count,
        "inference_success_count": success_count,
        "candidate_count": candidate_count,
        "prompt_versions": sorted(prompt_versions),
        "devices": sorted(devices),
        "max_lengths": sorted(max_lengths),
        "fingerprint_count": len(fingerprints),
        "batch_sizes": sorted(batch_sizes),
        "candidate_limits": sorted(candidate_limits),
        "latency_sample_count": len(latency_seconds),
        "latency_p50_seconds": p50,
        "latency_p95_seconds": p95,
        "throughput_candidates_per_second": throughput,
        "peak_vram_bytes": peak_vram_bytes,
    }


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
