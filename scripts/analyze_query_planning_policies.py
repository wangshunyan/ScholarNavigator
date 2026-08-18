#!/usr/bin/env python3
"""比较 current_rules 与 facet_balanced 初始查询规划策略。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICIES = ("current_rules", "facet_balanced")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/benchmark_runs/initial_query_planning_analysis"
)
COMPARABLE_FIELDS = (
    "dataset",
    "dataset_sha256",
    "case_ids",
    "offset",
    "limit",
    "sources",
    "query_adapter_policy",
    "run_profile",
    "result_policy",
    "top_k",
    "current_year",
    "max_workers",
    "budgets",
    "diagnostics",
    "llm",
    "enable_query_evolution",
    "query_evolution_policy",
    "enable_refchain",
)


def build_query_planning_analysis(
    *,
    development_current: Path,
    development_facet: Path,
    validation_current: Path,
    validation_facet: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "current_rules": _load_run(development_current),
            "facet_balanced": _load_run(development_facet),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "facet_balanced": _load_run(validation_facet),
        },
    }
    for split_runs in runs.values():
        _validate_pair(split_runs)

    summaries = {
        split: {
            policy: _summarize_run(run, split=split, policy=policy)
            for policy, run in split_runs.items()
        }
        for split, split_runs in runs.items()
    }
    acceptance = _validation_acceptance(
        summaries["validation"]["current_rules"],
        summaries["validation"]["facet_balanced"],
    )
    comparison = {
        "policies": list(POLICIES),
        "splits": summaries,
        "validation_acceptance": acceptance,
        "product_default": (
            "facet_balanced" if acceptance["accepted"] else "current_rules"
        ),
        "limitations": [
            "固定小样本仅用于受控策略诊断，不代表完整 Benchmark 成绩。",
            "gold 仅用于运行后评估，未参与查询规划。",
            "录制请求成本来自冻结快照元数据；replay 执行网络请求必须为零。",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    for split in ("development", "validation"):
        rows = [
            row
            for policy in POLICIES
            for row in _diagnostic_rows(
                runs[split][policy],
                split=split,
                policy=policy,
            )
        ]
        _write_jsonl(output_dir / f"{split}_query_diagnostics.jsonl", rows)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _load_run(path: Path) -> dict[str, Any]:
    run_dir = path.expanduser().resolve()
    return {
        "run_dir": str(run_dir),
        "config": _read_json(run_dir / "config.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
        "stage_metrics": _read_json(run_dir / "stage_metrics.json"),
        "results": _read_jsonl(run_dir / "results.jsonl"),
    }


def _validate_pair(runs: dict[str, dict[str, Any]]) -> None:
    if set(runs) != set(POLICIES):
        raise ValueError("each split requires current_rules and facet_balanced")
    current = runs["current_rules"]
    facet = runs["facet_balanced"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if current["config"].get(field) != facet["config"].get(field)
    ]
    if mismatched:
        raise ValueError("incompatible query planning runs: " + ",".join(mismatched))
    observed = {
        policy: run["config"].get("query_planning_policy", "current_rules")
        for policy, run in runs.items()
    }
    if any(policy != value for policy, value in observed.items()):
        raise ValueError(f"query planning policy mismatch: {observed}")


def _summarize_run(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> dict[str, Any]:
    metrics = run["metrics"]
    end_to_end = metrics.get("end_to_end_metrics") or metrics.get("aggregate") or {}
    stage = run["stage_metrics"]
    planning = stage.get("initial_query_planning") or {}
    snapshot_costs = metrics.get("snapshot_costs") or {}
    case_count = int(stage.get("case_count") or metrics.get("case_count") or 0)
    effective_requests = int(planning.get("effective_request_count") or 0)
    recorded_latency = float(planning.get("recorded_latency_seconds") or 0.0)
    unique_candidates = int(planning.get("unique_candidate_count") or 0)
    unique_gold = int(planning.get("unique_gold_count") or 0)
    average_requests = effective_requests / case_count if case_count else 0.0
    return {
        "split": split,
        "policy": policy,
        "run_dir": run["run_dir"],
        "case_count": case_count,
        "candidate_recall": stage.get("initial_retrieval_recall"),
        "f1_at_5": _at_k(end_to_end, "f1_at_k", 5),
        "f1_at_10": _at_k(end_to_end, "f1_at_k", 10),
        "f1_at_20": _at_k(end_to_end, "f1_at_k", 20),
        "precision_at_20": _at_k(end_to_end, "precision_at_k", 20),
        "recall_at_20": _at_k(end_to_end, "recall_at_k", 20),
        "mrr": end_to_end.get("mrr"),
        "ndcg_at_20": _at_k(end_to_end, "ndcg_at_k", 20),
        "subquery_count": int(planning.get("subquery_count") or 0),
        "average_subquery_count": float(
            planning.get("average_subquery_count") or 0.0
        ),
        "adapted_query_count": int(planning.get("adapted_query_count") or 0),
        "average_adapted_query_count": float(
            planning.get("average_adapted_query_count") or 0.0
        ),
        "unique_candidate_count": unique_candidates,
        "duplicate_candidate_ratio": planning.get("duplicate_candidate_ratio"),
        "unique_gold_count": unique_gold,
        "average_recorded_api_calls": average_requests,
        "average_recorded_latency_seconds": (
            recorded_latency / case_count if case_count else 0.0
        ),
        "api_calls_per_unique_candidate": _safe_ratio(
            effective_requests,
            unique_candidates,
        ),
        "api_calls_per_unique_gold": _safe_ratio(effective_requests, unique_gold),
        "source_error_rate": float(planning.get("source_error_rate") or 0.0),
        "replay_execution_request_count": int(
            snapshot_costs.get("replay_execution_request_count") or 0
        ),
        "replay_execution_retry_count": int(
            snapshot_costs.get("replay_execution_retry_count") or 0
        ),
        "replay_execution_network_wait_seconds": float(
            snapshot_costs.get("replay_execution_network_wait_seconds") or 0.0
        ),
        "facet_contribution": planning.get("facet_contribution") or {},
        "ineffective_reasons": planning.get("ineffective_reasons") or {},
    }


def _validation_acceptance(
    current: dict[str, Any],
    facet: dict[str, Any],
) -> dict[str, Any]:
    non_regression_fields = (
        "candidate_recall",
        "recall_at_20",
        "f1_at_5",
        "f1_at_10",
        "f1_at_20",
    )
    checks = {
        f"{field}_non_regression": _non_lower(facet.get(field), current.get(field))
        for field in non_regression_fields
    }
    current_api = float(current.get("average_recorded_api_calls") or 0.0)
    facet_api = float(facet.get("average_recorded_api_calls") or 0.0)
    checks["api_calls_within_1_5x"] = (
        facet_api == 0.0 if current_api == 0.0 else facet_api <= current_api * 1.5
    )
    current_duplicate = current.get("duplicate_candidate_ratio")
    facet_duplicate = facet.get("duplicate_candidate_ratio")
    checks["duplicate_ratio_not_notably_worse"] = (
        True
        if current_duplicate is None or facet_duplicate is None
        else float(facet_duplicate) <= float(current_duplicate) + 0.05
    )
    checks["frozen_replay_zero_network"] = all(
        float(facet.get(field) or 0.0) == 0.0
        and float(current.get(field) or 0.0) == 0.0
        for field in (
            "replay_execution_request_count",
            "replay_execution_retry_count",
            "replay_execution_network_wait_seconds",
        )
    )
    no_new_gold_with_higher_cost = (
        int(facet.get("unique_gold_count") or 0)
        <= int(current.get("unique_gold_count") or 0)
        and facet_api > current_api
    )
    checks["not_higher_cost_without_new_gold"] = not no_new_gold_with_higher_cost
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "api_call_ratio": _safe_ratio(facet_api, current_api),
        "duplicate_ratio_tolerance": 0.05,
        "decision": (
            "facet_balanced 可作为产品默认策略"
            if all(checks.values())
            else "保留 current_rules 为产品默认策略"
        ),
    }


def _diagnostic_rows(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in run["results"]:
        diagnostics = result.get("stage_diagnostics") or {}
        planning = diagnostics.get("initial_query_planning") or {}
        rows.append(
            {
                "split": split,
                "policy": policy,
                "case_id": result.get("case_id"),
                "query": result.get("query"),
                "status": result.get("status"),
                "planner_version": planning.get("planner_version"),
                "query_analysis": planning.get("query_analysis"),
                "planning": planning.get("planning"),
                "subqueries": planning.get("subqueries") or [],
                "subquery_count": planning.get("subquery_count"),
                "adapted_query_count": planning.get("adapted_query_count"),
                "unique_candidate_count": planning.get("unique_candidate_count"),
                "duplicate_candidate_ratio": planning.get(
                    "duplicate_candidate_ratio"
                ),
                "unique_gold_count": planning.get("unique_gold_count"),
                "recorded_request_count": planning.get("recorded_request_count"),
                "recorded_latency_seconds": planning.get(
                    "recorded_latency_seconds"
                ),
                "facet_contribution": planning.get("facet_contribution") or {},
                "ineffective_reasons": planning.get("ineffective_reasons") or {},
            }
        )
    return rows


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 初始查询规划策略对比",
        "",
        "> 固定小样本受控诊断；gold 仅用于运行后评估，不代表完整 Benchmark 成绩。",
        "",
        "| 数据集 | 策略 | 候选 Recall | F1@5 | F1@10 | F1@20 | R@20 | "
        "子查询/例 | 适配查询/例 | 唯一候选 | 唯一 gold | 重复率 | 录制 API/例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "validation"):
        for policy in POLICIES:
            row = comparison["splits"][split][policy]
            lines.append(
                f"| {split} | {policy} | {_fmt(row['candidate_recall'])} | "
                f"{_fmt(row['f1_at_5'])} | {_fmt(row['f1_at_10'])} | "
                f"{_fmt(row['f1_at_20'])} | {_fmt(row['recall_at_20'])} | "
                f"{_fmt(row['average_subquery_count'])} | "
                f"{_fmt(row['average_adapted_query_count'])} | "
                f"{row['unique_candidate_count']} | {row['unique_gold_count']} | "
                f"{_fmt(row['duplicate_candidate_ratio'])} | "
                f"{_fmt(row['average_recorded_api_calls'])} |"
            )
    acceptance = comparison["validation_acceptance"]
    lines.extend(
        [
            "",
            "## 验证集验收",
            "",
            f"- 结论：{acceptance['decision']}。",
            f"- 全部验收项通过：{'是' if acceptance['accepted'] else '否'}。",
            f"- facet/current API 比：{_fmt(acceptance['api_call_ratio'])}。",
            "- replay 执行网络请求、重试和网络等待必须均为 0。",
            "",
            "## 限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in comparison["limitations"])
    lines.append("")
    return "\n".join(lines)


def _at_k(metrics: dict[str, Any], field: str, k: int) -> Any:
    values = metrics.get(field) or {}
    return values.get(str(k), values.get(k))


def _non_lower(value: Any, baseline: Any) -> bool:
    if value is None or baseline is None:
        return False
    return float(value) + 1e-12 >= float(baseline)


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-current", type=Path, required=True)
    parser.add_argument("--development-facet", type=Path, required=True)
    parser.add_argument("--validation-current", type=Path, required=True)
    parser.add_argument("--validation-facet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = build_query_planning_analysis(
        development_current=args.development_current,
        development_facet=args.development_facet,
        validation_current=args.validation_current,
        validation_facet=args.validation_facet,
        output_dir=args.output_dir,
    )
    print(json.dumps(comparison["validation_acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
