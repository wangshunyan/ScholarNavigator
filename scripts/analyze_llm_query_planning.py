#!/usr/bin/env python3
"""比较 current_rules 与冻结 llm_semantic 初始查询规划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICIES = ("current_rules", "llm_semantic")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/llm_query_planning_analysis")
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
    "enable_query_evolution",
    "query_evolution_policy",
    "enable_refchain",
)


def build_llm_query_planning_analysis(
    *,
    development_current: Path,
    development_llm: Path,
    validation_current: Path,
    validation_llm: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "current_rules": _load_run(development_current),
            "llm_semantic": _load_run(development_llm),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "llm_semantic": _load_run(validation_llm),
        },
    }
    for split_runs in runs.values():
        _validate_pair(split_runs)
    summaries = {
        split: {
            policy: _summarize(run, split=split, policy=policy)
            for policy, run in split_runs.items()
        }
        for split, split_runs in runs.items()
    }
    acceptance = _acceptance(
        summaries["validation"]["current_rules"],
        summaries["validation"]["llm_semantic"],
    )
    comparison = {
        "policies": list(POLICIES),
        "splits": summaries,
        "validation_acceptance": acceptance,
        "product_default": "current_rules",
        "llm_semantic_default_enabled": False,
        "limitations": [
            "固定小样本只用于受控策略诊断，不代表完整 Benchmark 成绩。",
            "gold 只用于运行后评估，未进入 Prompt、快照键或查询规划。",
            "冻结 replay 的 LLM、检索、引用、重试与网络等待成本必须为零。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    for split in ("development", "validation"):
        rows = [
            row
            for policy in POLICIES
            for row in _diagnostic_rows(runs[split][policy], split, policy)
        ]
        _write_jsonl(output_dir / f"{split}_query_diagnostics.jsonl", rows)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def build_unavailable_llm_baseline_analysis(
    *,
    development_current: Path,
    validation_current: Path,
    reason: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """无可用 LLM 配置时只汇总真实 current_rules 基线。"""

    runs = {
        "development": _load_run(development_current),
        "validation": _load_run(validation_current),
    }
    for run in runs.values():
        if run["config"].get("query_planning_policy") != "current_rules":
            raise ValueError("baseline policy must be current_rules")
    summaries = {
        split: {
            "current_rules": _summarize(
                run,
                split=split,
                policy="current_rules",
            ),
            "llm_semantic": None,
        }
        for split, run in runs.items()
    }
    comparison = {
        "policies": list(POLICIES),
        "experiment_status": "llm_not_run",
        "llm_unavailable_reason": reason,
        "splits": summaries,
        "validation_acceptance": {
            "accepted": False,
            "evaluated": False,
            "checks": {},
            "decision": "未运行 LLM 对比，保留 current_rules 为默认策略",
        },
        "product_default": "current_rules",
        "llm_semantic_default_enabled": False,
        "limitations": [
            "当前进程没有可用 LLM 配置，未生成或伪造 llm_semantic 指标。",
            "current_rules 使用固定开发集和独立验证集的冻结检索快照回放。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    for split, run in runs.items():
        _write_jsonl(
            output_dir / f"{split}_query_diagnostics.jsonl",
            _diagnostic_rows(run, split, "current_rules"),
        )
    (output_dir / "summary.md").write_text(
        _unavailable_summary(comparison),
        encoding="utf-8",
    )
    return comparison


def _load_run(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve()
    return {
        "run_dir": str(root),
        "config": _read_json(root / "config.json"),
        "metrics": _read_json(root / "metrics.json"),
        "stage_metrics": _read_json(root / "stage_metrics.json"),
        "results": _read_jsonl(root / "results.jsonl"),
    }


def _validate_pair(runs: dict[str, dict[str, Any]]) -> None:
    if set(runs) != set(POLICIES):
        raise ValueError("each split requires current_rules and llm_semantic")
    current, llm = runs["current_rules"], runs["llm_semantic"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if current["config"].get(field) != llm["config"].get(field)
    ]
    if mismatched:
        raise ValueError("incompatible LLM planning runs: " + ",".join(mismatched))
    if current["config"].get("query_planning_policy") != "current_rules":
        raise ValueError("current_rules policy mismatch")
    if llm["config"].get("query_planning_policy") != "llm_semantic":
        raise ValueError("llm_semantic policy mismatch")


def _summarize(run: dict[str, Any], *, split: str, policy: str) -> dict[str, Any]:
    metrics = run["metrics"]
    ranked = metrics.get("end_to_end_metrics") or {}
    stage = run["stage_metrics"]
    planning = stage.get("initial_query_planning") or {}
    snapshot = metrics.get("snapshot_costs") or {}
    llm_cost = metrics.get("llm_planning_costs") or {}
    case_count = int(stage.get("case_count") or metrics.get("case_count") or 0)
    requests = int(planning.get("effective_request_count") or 0)
    unique_candidates = int(planning.get("unique_candidate_count") or 0)
    unique_gold = int(planning.get("unique_gold_count") or 0)
    contribution = _supplemental_contribution(run["results"])
    planning_rows = [
        ((row.get("stage_diagnostics") or {}).get("initial_query_planning") or {}).get(
            "planning"
        )
        or {}
        for row in run["results"]
    ]
    fallback_count = sum(bool(item.get("fallback_used")) for item in planning_rows)
    invalid_count = sum(
        item.get("policy") == "llm_semantic" and not item.get("output_valid")
        for item in planning_rows
    )
    return {
        "split": split,
        "policy": policy,
        "run_dir": run["run_dir"],
        "case_count": case_count,
        "candidate_recall": stage.get("initial_retrieval_recall"),
        "f1_at_5": _at_k(ranked, "f1_at_k", 5),
        "f1_at_10": _at_k(ranked, "f1_at_k", 10),
        "f1_at_20": _at_k(ranked, "f1_at_k", 20),
        "precision_at_20": _at_k(ranked, "precision_at_k", 20),
        "recall_at_20": _at_k(ranked, "recall_at_k", 20),
        "mrr": ranked.get("mrr"),
        "ndcg_at_20": _at_k(ranked, "ndcg_at_k", 20),
        "logical_subquery_count": int(planning.get("subquery_count") or 0),
        "adapted_query_count": int(planning.get("adapted_query_count") or 0),
        "unique_candidate_count": unique_candidates,
        "duplicate_candidate_ratio": planning.get("duplicate_candidate_ratio"),
        "unique_gold_count": unique_gold,
        "average_retrieval_api_calls": _ratio(requests, case_count),
        "average_recorded_retrieval_latency_seconds": _ratio(
            float(planning.get("recorded_latency_seconds") or 0.0), case_count
        ),
        "llm_call_count": int(llm_cost.get("live_call_count") or 0),
        "llm_prompt_tokens": int(llm_cost.get("prompt_tokens") or 0),
        "llm_completion_tokens": int(llm_cost.get("completion_tokens") or 0),
        "llm_total_tokens": int(llm_cost.get("total_tokens") or 0),
        "recorded_llm_latency_seconds": float(
            llm_cost.get("recorded_latency_seconds") or 0.0
        ),
        "fallback_rate": _ratio(fallback_count, case_count),
        "invalid_output_rate": _ratio(invalid_count, case_count),
        "replay_execution_request_count": int(
            snapshot.get("replay_execution_request_count") or 0
        )
        + int(llm_cost.get("replay_execution_request_count") or 0),
        "replay_execution_retry_count": int(
            snapshot.get("replay_execution_retry_count") or 0
        )
        + int(llm_cost.get("replay_execution_retry_count") or 0),
        "replay_execution_network_wait_seconds": float(
            snapshot.get("replay_execution_network_wait_seconds") or 0.0
        )
        + float(llm_cost.get("replay_execution_network_wait_seconds") or 0.0),
        "supplemental_query_contribution": contribution,
        "api_calls_per_added_candidate": _ratio(
            contribution["retrieval_api_calls"], contribution["added_candidate_count"]
        ),
        "api_calls_per_added_gold": _ratio(
            contribution["retrieval_api_calls"], contribution["added_gold_count"]
        ),
        "tokens_per_added_gold": _ratio(
            int(llm_cost.get("total_tokens") or 0), contribution["added_gold_count"]
        ),
    }


def _supplemental_contribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = gold = requests = 0
    query_count = 0
    for row in results:
        planning = (row.get("stage_diagnostics") or {}).get(
            "initial_query_planning"
        ) or {}
        for subquery in planning.get("subqueries") or []:
            if not str(subquery.get("purpose") or "").startswith("llm_semantic:"):
                continue
            query_count += 1
            candidates += int(subquery.get("exclusive_candidate_count") or 0)
            gold += int(subquery.get("post_run_unique_gold_hit_count") or 0)
            requests += int(subquery.get("recorded_request_count") or 0)
    return {
        "query_count": query_count,
        "added_candidate_count": candidates,
        "added_gold_count": gold,
        "judgement_retained_count": None,
        "weak_or_irrelevant_count": None,
        "retrieval_api_calls": requests,
    }


def _acceptance(current: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_recall_non_regression": _non_lower(
            llm.get("candidate_recall"), current.get("candidate_recall")
        ),
        "recall_at_20_non_regression": _non_lower(
            llm.get("recall_at_20"), current.get("recall_at_20")
        ),
        "f1_at_20_non_regression": _non_lower(
            llm.get("f1_at_20"), current.get("f1_at_20")
        ),
    }
    current_api = float(current.get("average_retrieval_api_calls") or 0.0)
    llm_api = float(llm.get("average_retrieval_api_calls") or 0.0)
    checks["retrieval_api_calls_within_1_5x"] = (
        llm_api == 0 if current_api == 0 else llm_api <= current_api * 1.5
    )
    contribution = llm["supplemental_query_contribution"]
    ranking_lift = (
        _greater(llm.get("mrr"), current.get("mrr"))
        or _greater(llm.get("ndcg_at_20"), current.get("ndcg_at_20"))
    )
    checks["new_gold_or_ranking_lift"] = bool(
        int(contribution.get("added_gold_count") or 0) >= 1 or ranking_lift
    )
    checks["at_most_one_llm_call_per_case"] = int(
        llm.get("llm_call_count") or 0
    ) <= int(llm.get("case_count") or 0)
    checks["frozen_replay_zero_network"] = all(
        float(llm.get(field) or 0.0) == 0.0
        and float(current.get(field) or 0.0) == 0.0
        for field in (
            "replay_execution_request_count",
            "replay_execution_retry_count",
            "replay_execution_network_wait_seconds",
        )
    )
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "checks": checks,
        "retrieval_api_call_ratio": _ratio(llm_api, current_api),
        "decision": (
            "llm_semantic 通过小样本验收，但仍保持可选且默认关闭"
            if accepted
            else "缺少验证集收益证据，保留 current_rules 为默认策略"
        ),
    }


def _diagnostic_rows(
    run: dict[str, Any], split: str, policy: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in run["results"]:
        planning = (result.get("stage_diagnostics") or {}).get(
            "initial_query_planning"
        ) or {}
        metadata = planning.get("planning") or {}
        rows.append(
            {
                "split": split,
                "policy": policy,
                "case_id": result.get("case_id"),
                "query": result.get("query"),
                "status": result.get("status"),
                "provider": metadata.get("provider"),
                "model": metadata.get("model"),
                "prompt_name": metadata.get("prompt_name"),
                "prompt_version": metadata.get("prompt_version"),
                "prompt_hash": metadata.get("prompt_hash"),
                "snapshot_key": metadata.get("snapshot_key"),
                "snapshot_status": metadata.get("snapshot_status"),
                "llm_call_attempted": metadata.get("llm_call_attempted", False),
                "replayed": metadata.get("replayed", False),
                "fallback_used": metadata.get("fallback_used", False),
                "fallback_reason": metadata.get("fallback_reason"),
                "generated_query_count": metadata.get("generated_query_count", 0),
                "accepted_query_count": metadata.get("accepted_query_count", 0),
                "rejected_query_count": metadata.get("rejected_query_count", 0),
                "rejection_reasons": metadata.get("rejection_reasons") or {},
                "original_query_retained": metadata.get(
                    "original_query_retained", True
                ),
                "accepted_queries": metadata.get("accepted_queries") or [],
                "terminology_expansions": metadata.get(
                    "terminology_expansions"
                )
                or [],
                "facet_coverage": {
                    key: metadata.get(key)
                    for key in (
                        "topic_coverage",
                        "method_coverage",
                        "dataset_coverage",
                        "task_coverage",
                        "paper_type_coverage",
                    )
                },
                "llm_prompt_tokens": metadata.get("llm_prompt_tokens", 0),
                "llm_completion_tokens": metadata.get(
                    "llm_completion_tokens", 0
                ),
                "llm_total_tokens": metadata.get("llm_total_tokens", 0),
                "recorded_llm_latency_seconds": metadata.get(
                    "recorded_llm_latency_seconds", 0.0
                ),
                "subqueries": planning.get("subqueries") or [],
            }
        )
    return rows


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# LLM 语义查询规划受控对比",
        "",
        "> 固定小样本；gold 仅用于运行后评估，LLM 与检索均以冻结回放结果为准。",
        "",
        "| 子集 | 策略 | 候选 Recall | F1@5 | F1@10 | F1@20 | R@20 | MRR | nDCG@20 | 唯一候选 | 唯一 gold | 检索 API/例 | LLM 调用 | Token | fallback 率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "validation"):
        for policy in POLICIES:
            row = comparison["splits"][split][policy]
            lines.append(
                f"| {split} | {policy} | {_fmt(row['candidate_recall'])} | "
                f"{_fmt(row['f1_at_5'])} | {_fmt(row['f1_at_10'])} | "
                f"{_fmt(row['f1_at_20'])} | {_fmt(row['recall_at_20'])} | "
                f"{_fmt(row['mrr'])} | {_fmt(row['ndcg_at_20'])} | "
                f"{row['unique_candidate_count']} | {row['unique_gold_count']} | "
                f"{_fmt(row['average_retrieval_api_calls'])} | "
                f"{row['llm_call_count']} | {row['llm_total_tokens']} | "
                f"{_fmt(row['fallback_rate'])} |"
            )
    acceptance = comparison["validation_acceptance"]
    lines.extend(
        [
            "",
            "## 验收结论",
            "",
            f"- {acceptance['decision']}。",
            f"- 全部验收项通过：{'是' if acceptance['accepted'] else '否'}。",
            "- 产品默认策略保持 `current_rules`；`llm_semantic` 默认关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def _unavailable_summary(comparison: dict[str, Any]) -> str:
    lines = [
        "# LLM 语义查询规划受控对比",
        "",
        f"> LLM 对比未运行：`{comparison['llm_unavailable_reason']}`。未生成或伪造 LLM 指标。",
        "",
        "| 子集 | 策略 | 候选 Recall | F1@5 | F1@10 | F1@20 | R@20 | MRR | nDCG@20 | 检索 API/例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "validation"):
        row = comparison["splits"][split]["current_rules"]
        lines.append(
            f"| {split} | current_rules | {_fmt(row['candidate_recall'])} | "
            f"{_fmt(row['f1_at_5'])} | {_fmt(row['f1_at_10'])} | "
            f"{_fmt(row['f1_at_20'])} | {_fmt(row['recall_at_20'])} | "
            f"{_fmt(row['mrr'])} | {_fmt(row['ndcg_at_20'])} | "
            f"{_fmt(row['average_retrieval_api_calls'])} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 验证集 LLM 验收未执行。",
            "- 产品默认策略保持 `current_rules`；`llm_semantic` 默认关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def _at_k(metrics: dict[str, Any], field: str, k: int) -> Any:
    values = metrics.get(field) or {}
    return values.get(str(k), values.get(k))


def _non_lower(value: Any, baseline: Any) -> bool:
    return value is not None and baseline is not None and float(value) + 1e-12 >= float(baseline)


def _greater(value: Any, baseline: Any) -> bool:
    return value is not None and baseline is not None and float(value) > float(baseline) + 1e-12


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析冻结 LLM 语义查询规划对比。")
    parser.add_argument("--development-current", type=Path, required=True)
    parser.add_argument("--development-llm", type=Path, required=True)
    parser.add_argument("--validation-current", type=Path, required=True)
    parser.add_argument("--validation-llm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_llm_query_planning_analysis(
        development_current=args.development_current,
        development_llm=args.development_llm,
        validation_current=args.validation_current,
        validation_llm=args.validation_llm,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
