#!/usr/bin/env python3
"""Audit the controlled 1000-query LLM planning ablation without secrets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-rows",
        type=int,
        choices=(200, 1000),
        default=1000,
        help="Expected completed query count: 200 for qualification or 1000 for full.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"object_required:{path.name}")
    return value


def _planning(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("stage_diagnostics") or {}
    initial = diagnostics.get("initial_query_planning") or {}
    value = initial.get("planning") or {}
    return value if isinstance(value, dict) else {}


def _subqueries(row: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = row.get("stage_diagnostics") or {}
    initial = diagnostics.get("initial_query_planning") or {}
    values = initial.get("subqueries") or []
    return [value for value in values if isinstance(value, dict)]


def audit_run(path: Path, *, expected_rows: int = 1000) -> dict[str, Any]:
    if expected_rows not in (200, 1000):
        raise ValueError("expected_rows_must_be_200_or_1000")
    root = path.expanduser().resolve()
    required = ("config.json", "metrics.json", "resource_ledger.json", "results.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    completed = list((root / ".run_commits" / "generations").glob("generation-*/RUN_COMPLETED"))
    config = _read_json(root / "config.json") if not missing else {}
    rows = [
        json.loads(line)
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if not missing else []
    reasons: list[str] = []
    if missing:
        reasons.extend(f"missing_artifact:{name}" for name in missing)
    if not completed:
        reasons.append("run_not_completed")
    if config.get("query_planning_policy") != "llm_semantic":
        reasons.append("llm_policy_missing")
    if int(config.get("limit") or 0) != expected_rows or int(config.get("top_k") or 0) != 20:
        contract = "full1000" if expected_rows == 1000 else "qualification200"
        reasons.append(f"{contract}_or_topk_contract_drift")
    budgets = config.get("budgets") or {}
    if int(budgets.get("max_llm_calls") or -1) != 1:
        reasons.append("per_query_llm_call_limit_drift")
    if int(budgets.get("max_search_rounds") or -1) != 3:
        reasons.append("supplemental_query_limit_drift")
    if len(rows) != expected_rows:
        reasons.append("result_row_count_invalid")

    attempted = fallbacks = schema_rejections = retained = 0
    execution_metadata_rows = transport_metadata_rows = http_attempts_total = http_429_total = 0
    schema_contract_rows = temperature_zero_rows = supplemental_budget_rows = 0
    retry_wait_seconds_total = 0.0
    cache_hit_count = 0
    failure_classes: Counter[str] = Counter()
    prompt_versions: set[str] = set()
    models: set[str] = set()
    call_counts: list[int] = []
    latencies: list[float] = []
    for row in rows:
        planning = _planning(row)
        subqueries = _subqueries(row)
        if planning.get("policy") != "llm_semantic":
            reasons.append("llm_planning_record_missing")
        attempted += int(bool(planning.get("llm_call_attempted")))
        fallbacks += int(bool(planning.get("fallback_used")))
        schema_rejections += int(planning.get("fallback_reason") == "invalid_schema")
        retained += int(bool(planning.get("original_query_retained")))
        if planning.get("prompt_version"):
            prompt_versions.add(str(planning["prompt_version"]))
        if planning.get("model"):
            models.add(str(planning["model"]))
        execution_fields = (
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_total_tokens",
            "recorded_llm_latency_seconds",
        )
        if all(field in planning for field in execution_fields):
            execution_metadata_rows += 1
            latencies.append(float(planning["recorded_llm_latency_seconds"]))
        if planning.get("llm_schema_version"):
            schema_contract_rows += 1
        try:
            temperature_zero_rows += int(float(planning.get("llm_temperature")) == 0.0)
        except (TypeError, ValueError):
            pass
        supplemental_budget_rows += int(
            planning.get("llm_max_supplemental_queries") == 2
        )
        transport_fields = (
            "llm_http_attempts",
            "llm_http_429_count",
            "llm_retry_after_seconds",
            "llm_retry_wait_seconds",
            "llm_provider_failure_class",
            "llm_provider_cache_hit",
        )
        if all(field in planning for field in transport_fields):
            transport_metadata_rows += 1
            http_attempts_total += max(0, int(planning.get("llm_http_attempts") or 0))
            http_429_total += max(0, int(planning.get("llm_http_429_count") or 0))
            retry_wait_seconds_total += max(
                0.0,
                float(planning.get("llm_retry_wait_seconds") or 0.0),
            )
            cache_hit_count += int(bool(planning.get("llm_provider_cache_hit")))
            failure_class = planning.get("llm_provider_failure_class")
            if failure_class:
                failure_classes[str(failure_class)] += 1
        supplemental = [
            item for item in subqueries
            if str(item.get("purpose") or "").startswith("llm_semantic:")
        ]
        if len(supplemental) > 2:
            reasons.append("supplemental_query_count_exceeded")
        cost = row.get("cost_report") or {}
        calls = int(cost.get("llm_call_count") or 0)
        call_counts.append(calls)
        if calls > 1:
            reasons.append("per_query_llm_call_count_exceeded")
    if attempted != len(rows):
        reasons.append("llm_call_not_attempted_for_every_query")
    if fallbacks:
        reasons.append("llm_fallback_detected")
    if retained != len(rows):
        reasons.append("original_query_not_retained")
    if not prompt_versions or not models:
        reasons.append("llm_runtime_metadata_missing")
    if execution_metadata_rows != len(rows):
        reasons.append("llm_execution_metadata_missing")
    if transport_metadata_rows != len(rows):
        reasons.append("llm_transport_metadata_missing")
    if schema_contract_rows != len(rows):
        reasons.append("llm_schema_contract_missing")
    if temperature_zero_rows != len(rows):
        reasons.append("llm_temperature_contract_drift")
    if supplemental_budget_rows != len(rows):
        reasons.append("llm_supplemental_budget_metadata_missing")
    if any(row.get("status") != "succeeded" for row in rows):
        reasons.append("failed_query_present")
    latencies.sort()
    p50 = latencies[(len(latencies) - 1) * 50 // 100] if latencies else 0.0
    p95 = latencies[(len(latencies) - 1) * 95 // 100] if latencies else 0.0
    return {
        "schema_version": "contest-llm-ablation-audit-v1",
        "run_id": root.name,
        "status": "passed" if not reasons else "failed",
        "reasons": sorted(set(reasons)),
        "expected_row_count": expected_rows,
        "result_row_count": len(rows),
        "llm_call_attempted_count": attempted,
        "fallback_count": fallbacks,
        "schema_rejection_count": schema_rejections,
        "original_query_retained_count": retained,
        "per_query_call_maximum": max(call_counts, default=0),
        "prompt_versions": sorted(prompt_versions),
        "models": sorted(models),
        "latency_p50_seconds": p50,
        "latency_p95_seconds": p95,
        "execution_metadata_available_row_count": execution_metadata_rows,
        "schema_contract_available_row_count": schema_contract_rows,
        "temperature_zero_row_count": temperature_zero_rows,
        "supplemental_budget_available_row_count": supplemental_budget_rows,
        "transport_telemetry": {
            "available_row_count": transport_metadata_rows,
            "http_attempt_total": http_attempts_total,
            "http_429_total": http_429_total,
            "retry_wait_seconds_total": retry_wait_seconds_total,
            "provider_cache_hit_count": cache_hit_count,
            "failure_classes": dict(sorted(failure_classes.items())),
        },
        "claimable_live_llm_effect": not reasons and fallbacks == 0,
        "internal_metric_scope": "not_official_competition_scorer",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_run(args.run, expected_rows=args.expected_rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"llm_audit_failed:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
