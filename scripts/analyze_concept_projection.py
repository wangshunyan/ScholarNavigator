#!/usr/bin/env python3
"""审计 current_rules 与固定预算 concept_projection 的配对 Replay。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_planning_policies import (  # noqa: E402
    COMPARABLE_FIELDS,
    _at_k,
    _load_run,
    _summarize_run,
    _write_json,
    _write_jsonl,
)


POLICIES = ("current_rules", "concept_projection")
SPLITS = ("scifact", "development", "validation")
EXPECTED_SPLITS = {
    "scifact": ("beir_scifact", "test", 0, 50),
    "development": ("auto_scholar_query", "development", 0, 10),
    "validation": ("auto_scholar_query", "validation", 10, 5),
}
METRICS = ("candidate_recall", "recall_at_20", "f1_at_20")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/concept_projection_analysis")


def build_concept_projection_analysis(
    *,
    scifact_current: Path,
    scifact_projection: Path,
    development_current: Path,
    development_projection: Path,
    validation_current: Path,
    validation_projection: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "scifact": {
            "current_rules": _load_run(scifact_current),
            "concept_projection": _load_run(scifact_projection),
        },
        "development": {
            "current_rules": _load_run(development_current),
            "concept_projection": _load_run(development_projection),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "concept_projection": _load_run(validation_projection),
        },
    }
    for split, split_runs in runs.items():
        _validate_pair(split, split_runs)

    summaries: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_runs = runs[split]
        split_pairs = _paired_rows(split, split_runs)
        paired_rows.extend(split_pairs)
        summaries[split] = {
            "runs": {
                policy: _run_summary(run, split=split, policy=policy)
                for policy, run in split_runs.items()
            },
            "full_lower_bound": _paired_summary(split_pairs),
            "both_complete_subset": _paired_summary(
                [row for row in split_pairs if row["both_complete"]]
            ),
            "query_budget_parity": _query_budget_parity(split_pairs),
            "source_deltas": _source_deltas(split_runs),
        }

    non_regression = {
        split: all(
            summaries[split]["runs"]["concept_projection"][metric]
            + 1e-12
            >= summaries[split]["runs"]["current_rules"][metric]
            for metric in METRICS
        )
        for split in SPLITS
    }
    comparison = {
        "policies": list(POLICIES),
        "rule_status": "frozen_before_metrics",
        "splits": summaries,
        "cross_dataset": {
            "non_regression_by_split": non_regression,
            "all_splits_non_regression": all(non_regression.values()),
            "strict_metric_gain_by_split": {
                split: any(
                    summaries[split]["runs"]["concept_projection"][metric]
                    > summaries[split]["runs"]["current_rules"][metric]
                    + 1e-12
                    for metric in METRICS
                )
                for split in SPLITS
            },
        },
        "limitations": [
            "全量结果把外部失败保留为下界；both_complete_subset 仅含双方来源调用均无错误的配对查询。",
            "gold 仅在 SearchService 返回候选后参与离线评估与来源贡献归因。",
            "概念投影固定替换最低优先级派生查询，不增加逻辑查询或来源请求预算。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    _write_jsonl(output_dir / "per_query.jsonl", paired_rows)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _validate_pair(split: str, runs: dict[str, dict[str, Any]]) -> None:
    if set(runs) != set(POLICIES):
        raise ValueError(f"{split} requires current_rules and concept_projection")
    baseline = runs["current_rules"]
    projected = runs["concept_projection"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if baseline["config"].get(field) != projected["config"].get(field)
    ]
    if mismatched:
        raise ValueError(f"incompatible {split} runs: {','.join(mismatched)}")
    dataset, dataset_split, offset, limit = EXPECTED_SPLITS[split]
    for policy, run in runs.items():
        config = run["config"]
        observed_limit = config.get("limit")
        if split == "scifact" and observed_limit is None:
            observed_limit = int((run["stage_metrics"] or {}).get("case_count") or 0)
        expected = (
            config.get("dataset"),
            config.get("dataset_split"),
            config.get("offset"),
            observed_limit,
        )
        if expected != (dataset, dataset_split, offset, limit):
            raise ValueError(f"unexpected fixed split {split}: {expected}")
        if config.get("query_planning_policy") != policy:
            raise ValueError(f"query planning policy mismatch: {policy}")
        if config.get("run_profile") != "balanced" or config.get("top_k") != 20:
            raise ValueError(f"{split} requires balanced top20")
        if (
            config.get("enable_query_evolution")
            or config.get("query_evolution_policy") != "off"
            or config.get("enable_refchain")
        ):
            raise ValueError(f"{split} requires later planning stages off")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError(f"{split} requires LLM features off")


def _run_summary(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> dict[str, Any]:
    summary = _summarize_run(run, split=split, policy=policy)
    planning = run["stage_metrics"].get("initial_query_planning") or {}
    metrics = run["metrics"]
    snapshot = metrics.get("snapshot_costs") or {}
    projection = Counter()
    for result in run["results"]:
        plan = (
            ((result.get("stage_diagnostics") or {}).get("initial_query_planning") or {})
            .get("planning")
            or {}
        )
        if policy != "concept_projection":
            continue
        reason = plan.get("concept_projection_skip_reason")
        if reason:
            projection[f"skipped:{reason}"] += 1
        elif plan.get("concept_projection_replaced_query"):
            projection["applied"] += 1
    summary.update(
        {
            "projection_outcomes": dict(sorted(projection.items())),
            "recorded_request_count": int(planning.get("recorded_request_count") or 0),
            "recorded_error_count": int(planning.get("source_error_count") or 0),
            "recorded_retry_count": int(snapshot.get("recorded_retry_count") or 0),
            "recorded_latency_seconds": float(
                snapshot.get("recorded_latency_seconds") or 0.0
            ),
            "replay_request_count": int(
                snapshot.get("replay_execution_request_count") or 0
            ),
            "replay_retry_count": int(
                snapshot.get("replay_execution_retry_count") or 0
            ),
            "replay_network_wait_seconds": float(
                snapshot.get("replay_execution_network_wait_seconds") or 0.0
            ),
        }
    )
    return summary


def _paired_rows(
    split: str,
    runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = {
        policy: {str(row.get("case_id")): row for row in run["results"]}
        for policy, run in runs.items()
    }
    metrics = {
        policy: {
            str(row.get("case_id")): row
            for row in (run["metrics"].get("per_case") or [])
        }
        for policy, run in runs.items()
    }
    order = [str(row.get("case_id")) for row in runs["current_rules"]["results"]]
    if set(order) != set(results["concept_projection"]):
        raise ValueError(f"paired case IDs differ for {split}")
    rows: list[dict[str, Any]] = []
    for case_id in order:
        baseline = _case_summary(
            results["current_rules"][case_id],
            metrics["current_rules"].get(case_id, {}),
        )
        projected = _case_summary(
            results["concept_projection"][case_id],
            metrics["concept_projection"].get(case_id, {}),
        )
        planning = (
            (
                results["concept_projection"][case_id].get("stage_diagnostics")
                or {}
            ).get("initial_query_planning")
            or {}
        )
        rows.append(
            {
                "split": split,
                "case_id": case_id,
                "query": results["current_rules"][case_id].get("query"),
                "current_rules": baseline,
                "concept_projection": projected,
                "both_complete": baseline["complete"] and projected["complete"],
                "deltas": {
                    metric: projected[metric] - baseline[metric]
                    for metric in METRICS
                },
                "projection": planning.get("planning") or {},
                "projection_query_contribution": next(
                    (
                        item
                        for item in planning.get("subqueries") or []
                        if item.get("purpose") == "concept_projection"
                    ),
                    None,
                ),
            }
        )
    return rows


def _case_summary(result: dict[str, Any], metric_row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("stage_diagnostics") or {}
    stage = diagnostics.get("stage_metrics") or {}
    metric = metric_row.get("metrics") or {}
    planning = diagnostics.get("initial_query_planning") or {}
    plan = planning.get("planning") or {}
    selected = plan.get("selected_subqueries") or []
    return {
        "candidate_recall": float(
            ((stage.get("candidate_recall") or {}).get("initial_retrieval")) or 0.0
        ),
        "recall_at_20": float(_at_k(metric, "recall_at_k", 20) or 0.0),
        "f1_at_20": float(_at_k(metric, "f1_at_k", 20) or 0.0),
        "unique_gold_count": int(planning.get("unique_gold_count") or 0),
        "source_error_count": int(planning.get("source_error_count") or 0),
        "selected_subquery_count": int(
            plan.get("selected_subquery_count") or len(selected)
        ),
        "original_query_first": bool(
            selected
            and selected[0].get("purpose") == "original_query"
            and selected[0].get("query") == result.get("query")
        ),
        "status": result.get("status"),
        "complete": (
            result.get("status") == "succeeded"
            and int(planning.get("source_error_count") or 0) == 0
        ),
    }


def _query_budget_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "equal_selected_subquery_count": sum(
            row["current_rules"]["selected_subquery_count"]
            == row["concept_projection"]["selected_subquery_count"]
            for row in rows
        ),
        "original_query_first_in_both": sum(
            row["current_rules"]["original_query_first"]
            and row["concept_projection"]["original_query_first"]
            for row in rows
        ),
    }


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins: dict[str, dict[str, int]] = {}
    for metric in METRICS:
        counts = Counter()
        for row in rows:
            delta = float(row["deltas"][metric])
            counts["improved" if delta > 1e-12 else "degraded" if delta < -1e-12 else "tied"] += 1
        wins[metric] = dict(counts)
    return {
        "case_count": len(rows),
        "wins": wins,
        "mean_deltas": {
            metric: (
                sum(float(row["deltas"][metric]) for row in rows) / len(rows)
                if rows
                else None
            )
            for metric in METRICS
        },
    }


def _source_deltas(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    source_rows = {
        policy: (
            (run["stage_metrics"].get("source_contribution") or {}).get("sources")
            or {}
        )
        for policy, run in runs.items()
    }
    output: dict[str, dict[str, int]] = {}
    for source in sorted(set(source_rows["current_rules"]) | set(source_rows["concept_projection"])):
        current = source_rows["current_rules"].get(source) or {}
        projected = source_rows["concept_projection"].get(source) or {}
        output[source] = {
            f"{field}_delta": int(projected.get(field) or 0) - int(current.get(field) or 0)
            for field in (
                "returned_candidate_count",
                "unique_candidate_count",
                "gold_hit_count",
                "unique_gold_hit_count",
                "success_count",
                "error_count",
            )
        }
    return output


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 概念投影查询配对评测",
        "",
        "> 规则在查看指标前冻结；gold 仅用于离线评估。",
        "",
        "| 数据集 | 策略 | 候选 Recall | Recall@20 | F1@20 | 唯一 gold | 请求 | 错误 | 录制延迟(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for policy in POLICIES:
            row = comparison["splits"][split]["runs"][policy]
            lines.append(
                f"| {split} | {policy} | {row['candidate_recall']:.4f} | "
                f"{row['recall_at_20']:.4f} | {row['f1_at_20']:.4f} | "
                f"{row['unique_gold_count']} | {row['recorded_request_count']} | "
                f"{row['recorded_error_count']} | {row['recorded_latency_seconds']:.2f} |"
            )
    lines.extend(["", "## 完整性", ""])
    for split in SPLITS:
        full = comparison["splits"][split]["full_lower_bound"]
        paired = comparison["splits"][split]["both_complete_subset"]
        lines.append(
            f"- {split}: 全量 {full['case_count']}，双方完整配对 {paired['case_count']}。"
        )
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}-current", type=Path, required=True)
        parser.add_argument(f"--{split}-projection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = build_concept_projection_analysis(
        scifact_current=args.scifact_current,
        scifact_projection=args.scifact_projection,
        development_current=args.development_current,
        development_projection=args.development_projection,
        validation_current=args.validation_current,
        validation_projection=args.validation_projection,
        output_dir=args.output_dir,
    )
    print(json.dumps(comparison["cross_dataset"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
