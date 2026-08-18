#!/usr/bin/env python3
"""在固定 holdout40 上比较 current_rules 与 disjunctive_facets。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_disjunctive_facets import (  # noqa: E402
    _or_query_contribution,
    _weak_irrelevant_ratio,
)
from scripts.analyze_query_planning_policies import (  # noqa: E402
    COMPARABLE_FIELDS,
    _at_k,
    _load_run,
    _non_lower,
    _safe_ratio,
    _summarize_run,
    _write_json,
    _write_jsonl,
)


HOLDOUT_OFFSET = 170
HOLDOUT_LIMIT = 40
HOLDOUT_CASE_IDS = tuple(
    f"AutoScholarQuery_test_{index}"
    for index in range(HOLDOUT_OFFSET, HOLDOUT_OFFSET + HOLDOUT_LIMIT)
)
BOOTSTRAP_SEED = 20260720
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_METRICS = (
    "candidate_recall",
    "f1_at_20",
    "recall_at_20",
    "mrr",
    "ndcg_at_20",
)
RANKING_REGRESSION_TOLERANCE = 0.01
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/disjunctive_holdout40")


def build_disjunctive_holdout_analysis(
    *,
    current_run: Path,
    disjunctive_run: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """校验冻结协议，生成汇总、逐查询差值和确定性 bootstrap。"""

    runs = {
        "current_rules": _load_run(current_run),
        "disjunctive_facets": _load_run(disjunctive_run),
    }
    _validate_protocol(runs)
    per_query = _paired_query_rows(runs)
    summaries = {
        policy: _run_summary(run, policy=policy)
        for policy, run in runs.items()
    }
    bootstrap = paired_bootstrap(
        per_query,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    acceptance = _acceptance(
        summaries["current_rules"],
        summaries["disjunctive_facets"],
    )
    comparison = {
        "protocol": {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "case_ids": list(HOLDOUT_CASE_IDS),
            "sources": ["arxiv"],
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "query_evolution_policy": "off",
            "refchain": False,
            "llm": False,
            "run_profile": "balanced",
            "top_k": 20,
            "result_policy": "highly_and_partial",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
            "holdout_tuning_forbidden": True,
            "gold_usage": "post_search_evaluation_only",
        },
        "groups": summaries,
        "deltas": _summary_deltas(
            summaries["current_rules"],
            summaries["disjunctive_facets"],
        ),
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
        "high_recall_profile_candidate": acceptance["accepted"],
        "product_default": "current_rules",
        "product_default_changed": False,
        "production_rules_frozen": True,
        "limitations": [
            "固定 40 条保留集只用于冻结复核，不代表完整 Benchmark。",
            "95% 区间是固定随机种子的成对 bootstrap 描述，不作夸张显著性声明。",
            "gold 只在 SearchService 完成后用于指标与 OR 贡献归因。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(
        output_dir / "experiment_status.json",
        {
            "evaluation_status": "completed",
            "metrics_available": True,
            "final_replay_executed": True,
            "gold_metrics_read": True,
            "acceptance": (
                "passed" if acceptance["accepted"] else "failed"
            ),
            "high_recall_profile_candidate": acceptance["accepted"],
            "product_default": "current_rules",
            "product_default_changed": False,
            "production_rules_frozen": True,
            "decision": acceptance["decision"],
        },
    )
    _write_jsonl(output_dir / "per_query_diagnostics.jsonl", per_query)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def build_blocked_collection_status(
    *,
    plan_path: Path,
    collection_result_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """在公共来源安全停止时，仅汇总覆盖状态，不生成替代指标。"""

    plan = _read_json(plan_path)
    collection = _read_json(collection_result_path)
    entries = plan.get("entries") or []
    coverage = collection.get("coverage") or {}
    status = {
        "evaluation_status": "blocked",
        "blocker": collection.get("stop_reason")
        or coverage.get("stop_reason")
        or "snapshot_collection_incomplete",
        "attempted_policy": (
            "current_rules"
            if collection.get("group") == "baseline"
            else collection.get("group") or "current_rules"
        ),
        "required_key_count": len(entries),
        "covered_success_count": int(collection.get("covered_success") or 0),
        "covered_failed_count": int(collection.get("covered_failed") or 0),
        "missing_key_count": int(collection.get("missing_entries") or 0),
        "recorded_request_count": int(collection.get("request_count") or 0),
        "failed_entry_count": int(collection.get("failed_entry_count") or 0),
        "blocked_sources": list(collection.get("blocked_sources") or []),
        "source_failure_counts": dict(
            collection.get("source_failure_counts") or {}
        ),
        "recorded_elapsed_seconds": float(
            collection.get("elapsed_seconds") or 0.0
        ),
        "disjunctive_collection_started": False,
        "final_replay_executed": False,
        "metrics_available": False,
        "acceptance": "not_evaluated",
        "high_recall_profile_candidate": False,
        "product_default": "current_rules",
        "product_default_changed": False,
        "gold_metrics_read": False,
        "note": "公共来源连续失败后按冻结协议安全停止，未生成替代结果。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment_status.json", status)
    (output_dir / "summary.md").write_text(
        _blocked_summary_markdown(status),
        encoding="utf-8",
    )
    return status


def paired_bootstrap(
    per_query: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if not per_query:
        raise ValueError("paired bootstrap requires at least one case")
    if iterations < 100:
        raise ValueError("paired bootstrap requires at least 100 iterations")
    randomizer = random.Random(seed)
    count = len(per_query)
    metrics: dict[str, Any] = {}
    for metric in BOOTSTRAP_METRICS:
        differences = [float(row["deltas"][metric]) for row in per_query]
        sampled = sorted(
            sum(
                differences[randomizer.randrange(count)]
                for _ in range(count)
            )
            / count
            for _ in range(iterations)
        )
        metrics[metric] = {
            "current_mean": _average(
                row["current_rules"][metric] for row in per_query
            ),
            "disjunctive_mean": _average(
                row["disjunctive_facets"][metric] for row in per_query
            ),
            "mean_difference": _average(differences),
            "ci_95_low": _percentile(sampled, 0.025),
            "ci_95_high": _percentile(sampled, 0.975),
            "bootstrap_positive_share": (
                sum(value > 0 for value in sampled) / len(sampled)
            ),
        }
    return {
        "seed": seed,
        "iterations": iterations,
        "case_count": count,
        "interval": "percentile_95",
        "metrics": metrics,
        "small_sample_warning": "holdout40_diagnostic_only",
    }


def _validate_protocol(runs: dict[str, dict[str, Any]]) -> None:
    if set(runs) != {"current_rules", "disjunctive_facets"}:
        raise ValueError("holdout requires current_rules and disjunctive_facets")
    current_config = runs["current_rules"]["config"]
    for policy, run in runs.items():
        config = run["config"]
        expected = {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "sources": ["arxiv"],
            "query_planning_policy": policy,
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "query_evolution_policy": "off",
            "enable_query_evolution": False,
            "enable_refchain": False,
            "run_profile": "balanced",
            "top_k": 20,
            "result_policy": "highly_and_partial",
            "retrieval_mode": "replay",
            "max_workers": 1,
        }
        mismatched = [
            key for key, value in expected.items() if config.get(key) != value
        ]
        if mismatched:
            raise ValueError("holdout protocol mismatch:" + ",".join(mismatched))
        if tuple(config.get("case_ids") or ()) != HOLDOUT_CASE_IDS:
            raise ValueError("holdout case ids are not fixed offset=170 limit=40")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError("holdout requires LLM off")
        _assert_zero_replay_cost(run["metrics"])
    comparable = (*COMPARABLE_FIELDS, "query_planner_version", "runtime_code_hash")
    differences = [
        field
        for field in comparable
        if field != "query_planning_policy"
        and current_config.get(field)
        != runs["disjunctive_facets"]["config"].get(field)
    ]
    if differences:
        raise ValueError("holdout runs are incompatible:" + ",".join(differences))


def _assert_zero_replay_cost(metrics: dict[str, Any]) -> None:
    costs = metrics.get("snapshot_costs") or {}
    for field in (
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
    ):
        if float(costs.get(field) or 0.0) != 0.0:
            raise ValueError(f"holdout replay executed network work:{field}")


def _run_summary(run: dict[str, Any], *, policy: str) -> dict[str, Any]:
    summary = _summarize_run(run, split="holdout40", policy=policy)
    stage = run["stage_metrics"]
    judgement = stage.get("judgement") or {}
    snapshot_costs = run["metrics"].get("snapshot_costs") or {}
    weak_ratio, categories = _weak_irrelevant_ratio(run)
    summary.update(
        {
            "gold_count": int(stage.get("gold_count") or 0),
            "gold_judgement_retained_count": (
                int(judgement.get("gold_judged_highly_relevant") or 0)
                + int(judgement.get("gold_judged_partially_relevant") or 0)
            ),
            "final_returned_gold_count": _returned_gold_count(run),
            "or_query_contribution": _or_query_contribution(run),
            "weak_irrelevant_ratio": weak_ratio,
            "judgement_category_counts": categories,
            "recorded_live_cost": {
                "search_request_count": float(
                    snapshot_costs.get("recorded_search_request_count") or 0.0
                ),
                "retry_count": float(
                    snapshot_costs.get("recorded_retry_count") or 0.0
                ),
                "error_count": float(
                    snapshot_costs.get("recorded_error_count") or 0.0
                ),
                "latency_seconds": float(
                    snapshot_costs.get("recorded_latency_seconds") or 0.0
                ),
            },
            "replay_execution_cost": {
                "snapshot_hits": float(
                    snapshot_costs.get("retrieval_snapshot_hits") or 0.0
                ),
                "http_requests": float(
                    snapshot_costs.get("replay_execution_request_count") or 0.0
                ),
                "retries": float(
                    snapshot_costs.get("replay_execution_retry_count") or 0.0
                ),
                "network_wait_seconds": float(
                    snapshot_costs.get("replay_execution_network_wait_seconds")
                    or 0.0
                ),
            },
        }
    )
    return summary


def _returned_gold_count(run: dict[str, Any]) -> int:
    return sum(
        gold.get("drop_reason") == "returned"
        for result in run["results"]
        for gold in (result.get("stage_diagnostics") or {}).get(
            "gold_diagnostics", []
        )
    )


def _paired_query_rows(
    runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = {
        policy: {str(row.get("case_id")): row for row in run["results"]}
        for policy, run in runs.items()
    }
    metric_rows = {
        policy: {
            str(row.get("case_id")): row
            for row in run["metrics"].get("per_case") or []
        }
        for policy, run in runs.items()
    }
    if any(tuple(rows) != HOLDOUT_CASE_IDS for rows in results.values()):
        raise ValueError("holdout result order changed")
    output: list[dict[str, Any]] = []
    for case_id in HOLDOUT_CASE_IDS:
        current_row = results["current_rules"][case_id]
        disjunctive_row = results["disjunctive_facets"][case_id]
        current = _case_summary(
            current_row,
            metric_rows["current_rules"][case_id],
        )
        disjunctive = _case_summary(
            disjunctive_row,
            metric_rows["disjunctive_facets"][case_id],
        )
        output.append(
            {
                "case_id": case_id,
                "query": current_row.get("query"),
                "current_rules": current,
                "disjunctive_facets": disjunctive,
                "deltas": {
                    metric: disjunctive[metric] - current[metric]
                    for metric in BOOTSTRAP_METRICS
                },
                "or_query_contribution": _case_or_contribution(disjunctive_row),
            }
        )
    return output


def _case_summary(row: dict[str, Any], metric_row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("stage_diagnostics") or {}
    stage_metrics = diagnostics.get("stage_metrics") or {}
    candidate_recall = (stage_metrics.get("candidate_recall") or {}).get(
        "initial_retrieval"
    )
    metrics = metric_row.get("metrics") or {}
    judgement = diagnostics.get("judgement") or {}
    gold = diagnostics.get("gold_diagnostics") or []
    return {
        "candidate_recall": float(candidate_recall or 0.0),
        "f1_at_20": float(_at_k(metrics, "f1_at_k", 20) or 0.0),
        "recall_at_20": float(_at_k(metrics, "recall_at_k", 20) or 0.0),
        "mrr": float(metrics.get("mrr") or 0.0),
        "ndcg_at_20": float(_at_k(metrics, "ndcg_at_k", 20) or 0.0),
        "gold_count": len(gold),
        "gold_judgement_retained_count": (
            int(judgement.get("gold_judged_highly_relevant") or 0)
            + int(judgement.get("gold_judged_partially_relevant") or 0)
        ),
        "final_returned_gold_count": sum(
            item.get("drop_reason") == "returned" for item in gold
        ),
    }


def _case_or_contribution(row: dict[str, Any]) -> dict[str, int]:
    planning = (row.get("stage_diagnostics") or {}).get(
        "initial_query_planning"
    ) or {}
    values: Counter[str] = Counter()
    for subquery in planning.get("subqueries") or []:
        if subquery.get("combination_mode") != "any":
            continue
        values["logical_query_count"] += 1
        values["exclusive_candidate_count"] += int(
            subquery.get("exclusive_candidate_count") or 0
        )
        values["post_run_unique_gold_hit_count"] += int(
            subquery.get("post_run_unique_gold_hit_count") or 0
        )
    return dict(sorted(values.items()))


def _summary_deltas(
    current: dict[str, Any],
    disjunctive: dict[str, Any],
) -> dict[str, float]:
    fields = (
        "candidate_recall",
        "f1_at_5",
        "f1_at_10",
        "f1_at_20",
        "precision_at_20",
        "recall_at_20",
        "mrr",
        "ndcg_at_20",
        "unique_gold_count",
        "unique_candidate_count",
        "duplicate_candidate_ratio",
        "weak_irrelevant_ratio",
        "average_recorded_api_calls",
        "average_recorded_latency_seconds",
        "source_error_rate",
        "gold_judgement_retained_count",
        "final_returned_gold_count",
    )
    return {
        field: float(disjunctive.get(field) or 0.0)
        - float(current.get(field) or 0.0)
        for field in fields
    }


def _acceptance(current: dict[str, Any], disjunctive: dict[str, Any]) -> dict[str, Any]:
    current_api = float(current.get("average_recorded_api_calls") or 0.0)
    disjunctive_api = float(disjunctive.get("average_recorded_api_calls") or 0.0)
    checks = {
        "at_least_two_new_unique_gold": (
            int(disjunctive.get("unique_gold_count") or 0)
            >= int(current.get("unique_gold_count") or 0) + 2
        ),
        "candidate_recall_non_regression": _non_lower(
            disjunctive.get("candidate_recall"), current.get("candidate_recall")
        ),
        "recall_at_20_non_regression": _non_lower(
            disjunctive.get("recall_at_20"), current.get("recall_at_20")
        ),
        "f1_at_20_non_regression": _non_lower(
            disjunctive.get("f1_at_20"), current.get("f1_at_20")
        ),
        "mrr_no_clear_regression": _not_below_tolerance(
            disjunctive.get("mrr"),
            current.get("mrr"),
            RANKING_REGRESSION_TOLERANCE,
        ),
        "ndcg_at_20_no_clear_regression": _not_below_tolerance(
            disjunctive.get("ndcg_at_20"),
            current.get("ndcg_at_20"),
            RANKING_REGRESSION_TOLERANCE,
        ),
        "api_calls_within_1_5x": (
            disjunctive_api == 0.0
            if current_api == 0.0
            else disjunctive_api <= current_api * 1.5
        ),
        "weak_irrelevant_ratio_within_0_10": _not_above_tolerance(
            disjunctive.get("weak_irrelevant_ratio"),
            current.get("weak_irrelevant_ratio"),
            0.10,
        ),
        "frozen_replay_zero_network": all(
            float(row["replay_execution_cost"][field] or 0.0) == 0.0
            for row in (current, disjunctive)
            for field in ("http_requests", "retries", "network_wait_seconds")
        ),
        "production_rules_frozen": True,
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "checks": checks,
        "unique_gold_gain": (
            int(disjunctive.get("unique_gold_count") or 0)
            - int(current.get("unique_gold_count") or 0)
        ),
        "api_call_ratio": _safe_ratio(disjunctive_api, current_api),
        "ranking_regression_tolerance": RANKING_REGRESSION_TOLERANCE,
        "decision": (
            "disjunctive_facets 可作为 high_recall profile 候选"
            if accepted
            else "disjunctive_facets 继续仅作为实验策略"
        ),
    }


def _not_below_tolerance(value: Any, baseline: Any, tolerance: float) -> bool:
    if value is None or baseline is None:
        return False
    return float(value) + tolerance + 1e-12 >= float(baseline)


def _not_above_tolerance(value: Any, baseline: Any, tolerance: float) -> bool:
    if value is None or baseline is None:
        return False
    return float(value) <= float(baseline) + tolerance + 1e-12


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 析取式分面 holdout40 复核",
        "",
        "> 固定 AutoScholarQuery offset=170、limit=40；不在本保留集调参。",
        "",
        "| 策略 | 候选 Recall | unique gold | F1@5 | F1@10 | F1@20 | P@20 | R@20 | MRR | nDCG@20 | API/例 | 弱相关+无关率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in ("current_rules", "disjunctive_facets"):
        row = comparison["groups"][policy]
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    _fmt(row.get("candidate_recall")),
                    str(row.get("unique_gold_count") or 0),
                    _fmt(row.get("f1_at_5")),
                    _fmt(row.get("f1_at_10")),
                    _fmt(row.get("f1_at_20")),
                    _fmt(row.get("precision_at_20")),
                    _fmt(row.get("recall_at_20")),
                    _fmt(row.get("mrr")),
                    _fmt(row.get("ndcg_at_20")),
                    _fmt(row.get("average_recorded_api_calls")),
                    _fmt(row.get("weak_irrelevant_ratio")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 成对 bootstrap",
            "",
            "| 指标 | 平均差 | 95% 区间 | 正差占比 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric, values in comparison["paired_bootstrap"]["metrics"].items():
        lines.append(
            f"| {metric} | {_fmt(values['mean_difference'])} | "
            f"[{_fmt(values['ci_95_low'])}, {_fmt(values['ci_95_high'])}] | "
            f"{_fmt(values['bootstrap_positive_share'])} |"
        )
    acceptance = comparison["acceptance"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- {acceptance['decision']}。",
            f"- 新增 unique gold：{acceptance['unique_gold_gain']}。",
            f"- 全部门槛通过：{'是' if acceptance['accepted'] else '否'}。",
            "- 产品默认仍为 current_rules，未在保留集调参。",
            "",
        ]
    )
    return "\n".join(lines)


def _blocked_summary_markdown(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 析取式分面 holdout40 复核",
            "",
            "> 公共来源连续失败，实验按预设保护条件安全停止。",
            "",
            f"- 停止原因：{status['blocker']}。",
            f"- 已记录请求：{status['recorded_request_count']}。",
            f"- 成功/失败/缺失键：{status['covered_success_count']}/"
            f"{status['covered_failed_count']}/{status['missing_key_count']}。",
            "- 未启动候选组，未执行最终 replay，未读取 gold 指标。",
            "- 验收状态：未评估；产品默认保持 current_rules。",
            "",
        ]
    )


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object:{path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path)
    parser.add_argument("--disjunctive-run", type=Path)
    parser.add_argument("--blocked-plan", type=Path)
    parser.add_argument("--blocked-collection-result", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.blocked_plan or args.blocked_collection_result:
        if not args.blocked_plan or not args.blocked_collection_result:
            raise SystemExit(
                "--blocked-plan and --blocked-collection-result are both required"
            )
        status = build_blocked_collection_status(
            plan_path=args.blocked_plan,
            collection_result_path=args.blocked_collection_result,
            output_dir=args.output_dir,
        )
        print(json.dumps(status, ensure_ascii=False))
        return 0
    if args.current_run is None or args.disjunctive_run is None:
        raise SystemExit("--current-run and --disjunctive-run are both required")
    comparison = build_disjunctive_holdout_analysis(
        current_run=args.current_run,
        disjunctive_run=args.disjunctive_run,
        output_dir=args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(comparison["acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
