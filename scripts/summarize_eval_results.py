#!/usr/bin/env python3
"""Generate a Markdown summary for offline evaluation results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.core.evaluation_schemas import (  # noqa: E402
    EvalMetricSet,
    EvalSuiteResult,
)


DEFAULT_K_VALUES = (5, 10, 20)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize an offline evaluation result.json file."
    )
    parser.add_argument("result_json", help="Path to result.json.")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path. Defaults to summary.md next to result.json.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_json)
    with result_path.open("r", encoding="utf-8") as handle:
        result = EvalSuiteResult.model_validate(json.load(handle))

    summary = build_markdown_summary(result)
    output_path = Path(args.output) if args.output else result_path.with_name("summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(output_path)
    return 0


def build_markdown_summary(result: EvalSuiteResult) -> str:
    lines = [
        "# 离线评测汇总",
        "",
        f"查询数：{len(result.query_results)}",
        "",
        "> sample fixture 仅验证评测流程，不代表真实 benchmark 性能。",
        "",
    ]
    if result.aggregate_reports:
        for label, attribute in (
            ("端到端指标", "end_to_end_metrics"),
            ("仅成功案例指标", "success_only_metrics"),
        ):
            lines.extend([f"## {label}", "", *_metric_table(result, attribute), ""])
    else:
        lines.extend(["## 聚合指标", "", *_legacy_metric_table(result), ""])

    lines.extend(["## 案例统计与效率", "", *_statistics_table(result), ""])
    lines.append("")
    lines.append("## 单查询结果")
    for query_result in result.query_results:
        lines.append("")
        lines.append(f"### {query_result.query_id}")
        lines.append("")
        lines.append(query_result.query)
        lines.append("")
        lines.append("| 分组 | 排名标识 | 警告 |")
        lines.append("| --- | --- | --- |")
        for group, group_result in query_result.group_results.items():
            ranked_ids = ", ".join(group_result.ranked_paper_ids[:5]) or "-"
            warnings = ", ".join(group_result.warnings) or "-"
            lines.append(f"| {group} | {ranked_ids} | {warnings} |")
    lines.append("")
    return "\n".join(lines)


def _metric_table(result: EvalSuiteResult, attribute: str) -> list[str]:
    rows = [
        "| 分组 | F1@5 | F1@10 | F1@20 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR | nDCG@5 | nDCG@10 | nDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, report in result.aggregate_reports.items():
        rows.append(_metric_row(group, getattr(report, attribute)))
    return rows


def _legacy_metric_table(result: EvalSuiteResult) -> list[str]:
    rows = [
        "| 分组 | F1@5 | F1@10 | F1@20 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR | nDCG@5 | nDCG@10 | nDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(
        _metric_row(group, metrics)
        for group, metrics in result.aggregate_metrics.items()
    )
    return rows


def _metric_row(group: str, metrics: EvalMetricSet) -> str:
    return (
        "| {group} | {f5:.3f} | {f10:.3f} | {f20:.3f} | "
        "{r5:.3f} | {r10:.3f} | {r20:.3f} | "
        "{p5:.3f} | {p10:.3f} | {p20:.3f} | {mrr:.3f} | "
        "{n5:.3f} | {n10:.3f} | {n20:.3f} |"
    ).format(
        group=group,
        f5=_metric_at(metrics.f1_at_k, 5),
        f10=_metric_at(metrics.f1_at_k, 10),
        f20=_metric_at(metrics.f1_at_k, 20),
        r5=_metric_at(metrics.recall_at_k, 5),
        r10=_metric_at(metrics.recall_at_k, 10),
        r20=_metric_at(metrics.recall_at_k, 20),
        p5=_metric_at(metrics.precision_at_k, 5),
        p10=_metric_at(metrics.precision_at_k, 10),
        p20=_metric_at(metrics.precision_at_k, 20),
        mrr=metrics.mrr,
        n5=_metric_at(metrics.ndcg_at_k, 5),
        n10=_metric_at(metrics.ndcg_at_k, 10),
        n20=_metric_at(metrics.ndcg_at_k, 20),
    )


def _statistics_table(result: EvalSuiteResult) -> list[str]:
    rows = [
        "| 分组 | 总案例 | 成功 | 失败 | 缺少结果 | 缺少 gold | 成功率 | 平均延迟（秒） | 平均 API | 平均检索 API | 平均引用 API | 平均重试 | 平均错误 | 平均缓存命中 | 平均限流等待（秒） | 平均 LLM 调用 | 平均 LLM Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, report in result.aggregate_reports.items():
        stats = report.case_statistics
        efficiency = report.efficiency
        rows.append(
            f"| {group} | {stats.total_case_count} | "
            f"{stats.evaluated_success_count} | {stats.failed_case_count} | "
            f"{stats.missing_result_count} | {stats.missing_gold_count} | "
            f"{stats.success_rate:.3f} | {efficiency.average_latency_seconds:.3f} | "
            f"{efficiency.avg_api_call_count:.3f} | "
            f"{efficiency.avg_search_api_call_count:.3f} | "
            f"{efficiency.avg_reference_api_call_count:.3f} | "
            f"{efficiency.avg_retry_count:.3f} | "
            f"{efficiency.avg_error_count:.3f} | "
            f"{efficiency.avg_cache_hit_count:.3f} | "
            f"{efficiency.avg_rate_limit_wait_seconds:.3f} | "
            f"{efficiency.avg_llm_call_count:.3f} | "
            f"{efficiency.avg_llm_total_tokens:.3f} |"
        )
    return rows


def _metric_at(values: dict[int, float], k: int) -> float:
    return float(values.get(k, values.get(str(k), 0.0)))  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
