#!/usr/bin/env python3
"""Audit paired current_rules and deterministic prf_v1 Benchmark Replays."""

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


POLICIES = ("current_rules", "prf_v1")
SPLITS = ("scifact", "development", "validation")
EXPECTED_SPLITS = {
    "scifact": ("beir_scifact", "test", 0, 50),
    "development": ("auto_scholar_query", "development", 0, 10),
    "validation": ("auto_scholar_query", "validation", 10, 5),
}
METRICS = ("candidate_recall", "recall_at_20", "f1_at_20")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/prf_v1_analysis")


def build_prf_analysis(
    *,
    scifact_current: Path,
    scifact_prf: Path,
    development_current: Path,
    development_prf: Path,
    validation_current: Path,
    validation_prf: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = {
        "scifact": (scifact_current, scifact_prf),
        "development": (development_current, development_prf),
        "validation": (validation_current, validation_prf),
    }
    runs = {
        split: {
            "current_rules": _load_run(current),
            "prf_v1": _load_run(prf),
        }
        for split, (current, prf) in paths.items()
    }
    for split, split_runs in runs.items():
        _validate_pair(split, split_runs)

    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows = _paired_rows(split, runs[split])
        all_rows.extend(rows)
        summaries[split] = {
            "runs": {
                policy: _run_summary(run, split=split, policy=policy)
                for policy, run in runs[split].items()
            },
            "full_lower_bound": _paired_summary(rows),
            "both_success_subset": _paired_summary(
                [row for row in rows if row["both_success"]]
            ),
            "budget_parity": {
                "case_count": len(rows),
                "same_planned_subquery_count": sum(
                    row["current_rules"]["selected_subquery_count"]
                    == row["prf_v1"]["selected_subquery_count"]
                    for row in rows
                ),
                "original_query_first": sum(
                    row["current_rules"]["original_query_first"]
                    and row["prf_v1"]["original_query_first"]
                    for row in rows
                ),
            },
            "prf_contribution": _prf_contribution(rows),
            "source_deltas": _source_deltas(runs[split]),
        }

    non_regression = {
        split: all(
            float(summaries[split]["runs"]["prf_v1"].get(metric) or 0.0)
            + 1e-12
            >= float(
                summaries[split]["runs"]["current_rules"].get(metric) or 0.0
            )
            for metric in METRICS
        )
        for split in SPLITS
    }
    final_gain = {
        split: any(
            float(summaries[split]["runs"]["prf_v1"].get(metric) or 0.0)
            > float(
                summaries[split]["runs"]["current_rules"].get(metric) or 0.0
            )
            + 1e-12
            for metric in ("recall_at_20", "f1_at_20")
        )
        for split in SPLITS
    }
    decision = {
        "non_regression_by_split": non_regression,
        "final_gain_by_split": final_gain,
        "non_regression_all_splits": all(non_regression.values()),
        "final_gain_split_count": sum(final_gain.values()),
    }
    decision["recommend_continue"] = bool(
        decision["non_regression_all_splits"]
        and decision["final_gain_split_count"] >= 2
    )
    comparison = {
        "policies": list(POLICIES),
        "parameters": {
            "seed_count": 5,
            "minimum_seed_document_frequency": 2,
            "maximum_feedback_terms": 6,
            "rank_discount": "1/rank",
            "query_budget_growth": 0,
            "frozen_before_metrics": True,
        },
        "splits": summaries,
        "decision": decision,
        "limitations": [
            "全量结果保留来源失败为下界；both_success_subset 只含双方来源调用均无错误的查询。",
            "反馈只读取首轮候选标题/摘要，gold 仅在返回后进入离线 evaluator。",
            "独立 gold 为该 PRF 子查询命中且其他计划子查询未命中的统一身份 gold。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    _write_jsonl(output_dir / "per_query.jsonl", all_rows)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison), encoding="utf-8"
    )
    return comparison


def _validate_pair(split: str, runs: dict[str, dict[str, Any]]) -> None:
    baseline = runs["current_rules"]
    prf = runs["prf_v1"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if baseline["config"].get(field) != prf["config"].get(field)
    ]
    if mismatched:
        raise ValueError(f"incompatible {split} runs: {','.join(mismatched)}")
    expected_dataset, expected_split, offset, limit = EXPECTED_SPLITS[split]
    for policy, run in runs.items():
        config = run["config"]
        observed_limit = config.get("limit")
        if split == "scifact" and observed_limit is None:
            observed_limit = int(run["stage_metrics"].get("case_count") or 0)
        observed = (
            config.get("dataset"),
            config.get("dataset_split"),
            config.get("offset"),
            observed_limit,
        )
        if observed != (expected_dataset, expected_split, offset, limit):
            raise ValueError(f"unexpected fixed split {split}: {observed}")
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
        if llm.get("requested"):
            raise ValueError(f"{split} requires all LLM features off")


def _run_summary(
    run: dict[str, Any], *, split: str, policy: str
) -> dict[str, Any]:
    summary = _summarize_run(run, split=split, policy=policy)
    # Run directories identify the Replay replica rather than an evaluation
    # result. Omitting them keeps equivalent Replay audits byte-stable.
    summary.pop("run_dir", None)
    # Candidate-level facet ownership can vary when concurrent source results
    # arrive in a different order, while identities, gold diagnostics, and all
    # reported PRF/source contributions remain stable. It is outside this
    # audit's result contract and must not make replicas differ.
    summary.pop("facet_contribution", None)
    snapshot = run["metrics"].get("snapshot_costs") or {}
    outcomes = Counter()
    for row in run["results"]:
        planning = _planning(row)
        if policy != "prf_v1":
            continue
        reason = planning.get("prf_skip_reason")
        outcomes[f"fallback:{reason}" if reason else "applied"] += 1
    summary.update(
        {
            "prf_outcomes": dict(sorted(outcomes.items())),
            "recorded_request_count": int(
                snapshot.get("recorded_search_request_count") or 0
            ),
            "recorded_retry_count": int(snapshot.get("recorded_retry_count") or 0),
            "recorded_error_count": int(snapshot.get("recorded_error_count") or 0),
            "recorded_latency_seconds": float(
                snapshot.get("recorded_latency_seconds") or 0.0
            ),
            "snapshot_hits": int(snapshot.get("retrieval_snapshot_hits") or 0),
            "snapshot_writes": int(snapshot.get("retrieval_snapshot_writes") or 0),
        }
    )
    return summary


def _paired_rows(
    split: str, runs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    result_maps = {
        policy: {str(row["case_id"]): row for row in run["results"]}
        for policy, run in runs.items()
    }
    metric_maps = {
        policy: {
            str(row["case_id"]): row
            for row in run["metrics"].get("per_case") or []
        }
        for policy, run in runs.items()
    }
    order = [str(row["case_id"]) for row in runs["current_rules"]["results"]]
    if set(order) != set(result_maps["prf_v1"]):
        raise ValueError(f"paired case IDs differ for {split}")
    rows: list[dict[str, Any]] = []
    for case_id in order:
        baseline = _case_summary(
            result_maps["current_rules"][case_id],
            metric_maps["current_rules"].get(case_id, {}),
        )
        prf = _case_summary(
            result_maps["prf_v1"][case_id],
            metric_maps["prf_v1"].get(case_id, {}),
        )
        planning_diagnostic = (
            result_maps["prf_v1"][case_id].get("stage_diagnostics") or {}
        ).get("initial_query_planning") or {}
        prf_row = next(
            (
                item
                for item in planning_diagnostic.get("subqueries") or []
                if item.get("purpose") == "prf_v1"
            ),
            None,
        )
        rows.append(
            {
                "split": split,
                "case_id": case_id,
                "query": result_maps["current_rules"][case_id].get("query"),
                "current_rules": baseline,
                "prf_v1": prf,
                "both_success": baseline["complete"] and prf["complete"],
                "deltas": {
                    metric: prf[metric] - baseline[metric] for metric in METRICS
                },
                "prf": _planning(result_maps["prf_v1"][case_id]),
                "prf_query_contribution": prf_row,
            }
        )
    return rows


def _case_summary(row: dict[str, Any], metric_row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("stage_diagnostics") or {}
    stage = diagnostics.get("stage_metrics") or {}
    planning = diagnostics.get("initial_query_planning") or {}
    selected = (planning.get("planning") or {}).get("selected_subqueries") or []
    metric = metric_row.get("metrics") or {}
    return {
        "candidate_recall": float(
            ((stage.get("candidate_recall") or {}).get("initial_retrieval")) or 0.0
        ),
        "recall_at_20": float(_at_k(metric, "recall_at_k", 20) or 0.0),
        "f1_at_20": float(_at_k(metric, "f1_at_k", 20) or 0.0),
        "unique_gold_count": int(planning.get("unique_gold_count") or 0),
        "source_error_count": int(planning.get("source_error_count") or 0),
        "selected_subquery_count": len(selected),
        "original_query_first": bool(
            selected
            and selected[0].get("purpose") == "original_query"
            and selected[0].get("query") == row.get("query")
        ),
        "complete": bool(
            row.get("status") == "succeeded"
            and int(planning.get("source_error_count") or 0) == 0
        ),
    }


def _planning(row: dict[str, Any]) -> dict[str, Any]:
    return (
        ((row.get("stage_diagnostics") or {}).get("initial_query_planning") or {})
        .get("planning")
        or {}
    )


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins: dict[str, dict[str, int]] = {}
    for metric in METRICS:
        counts = Counter()
        for row in rows:
            delta = float(row["deltas"][metric])
            key = "improved" if delta > 1e-12 else "degraded" if delta < -1e-12 else "tied"
            counts[key] += 1
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


def _prf_contribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_rows = [
        row["prf_query_contribution"]
        for row in rows
        if row["prf_query_contribution"] is not None
    ]
    return {
        "applied_case_count": len(query_rows),
        "fallback_case_count": len(rows) - len(query_rows),
        "unique_candidate_count": sum(
            int(item.get("unique_candidate_count") or 0) for item in query_rows
        ),
        "independent_candidate_count": sum(
            int(item.get("exclusive_candidate_count") or 0) for item in query_rows
        ),
        "gold_hit_count": sum(
            int(item.get("post_run_gold_hit_count") or 0) for item in query_rows
        ),
        "independent_gold_count": sum(
            int(item.get("post_run_unique_gold_hit_count") or 0)
            for item in query_rows
        ),
        "recorded_request_count": sum(
            int(item.get("recorded_request_count") or 0) for item in query_rows
        ),
        "recorded_latency_seconds": sum(
            float(item.get("recorded_latency_seconds") or 0.0)
            for item in query_rows
        ),
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
    sources = sorted(set(source_rows["current_rules"]) | set(source_rows["prf_v1"]))
    for source in sources:
        baseline = source_rows["current_rules"].get(source) or {}
        prf = source_rows["prf_v1"].get(source) or {}
        output[source] = {
            f"{field}_delta": int(prf.get(field) or 0) - int(baseline.get(field) or 0)
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
        "# PRF v1 配对评测",
        "",
        "> 参数在查看指标前冻结；gold 仅用于返回后的离线评估。",
        "",
        "| 数据集 | 策略 | 候选 Recall | Recall@20 | F1@20 | 唯一 gold | 录制请求 | 错误 | 延迟(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for policy in POLICIES:
            row = comparison["splits"][split]["runs"][policy]
            lines.append(
                f"| {split} | {policy} | {float(row['candidate_recall'] or 0):.4f} | "
                f"{float(row['recall_at_20'] or 0):.4f} | "
                f"{float(row['f1_at_20'] or 0):.4f} | {row['unique_gold_count']} | "
                f"{row['recorded_request_count']} | {row['recorded_error_count']} | "
                f"{row['recorded_latency_seconds']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            (
                "- 建议继续后续验证。"
                if comparison["decision"]["recommend_continue"]
                else "- 未达到跨集合门槛，保持默认关闭。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}-current", type=Path, required=True)
        parser.add_argument(f"--{split}-prf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_prf_analysis(
        scifact_current=args.scifact_current,
        scifact_prf=args.scifact_prf,
        development_current=args.development_current,
        development_prf=args.development_prf,
        validation_current=args.validation_current,
        validation_prf=args.validation_prf,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
