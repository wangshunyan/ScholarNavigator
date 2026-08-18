#!/usr/bin/env python3
"""比较使用同一固定子集和预算的多个 Benchmark 运行。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def compare_runs(
    run_dirs: list[Path],
    *,
    require_ablation_matrix: bool = False,
    allow_source_difference: bool = False,
) -> str:
    if len(run_dirs) < 2:
        raise ValueError("at least two --run directories are required")
    runs = [_load_run(path) for path in run_dirs]
    _validate_comparable(
        runs,
        allow_source_difference=allow_source_difference,
    )
    if require_ablation_matrix:
        _validate_ablation_matrix(runs)
        return _ablation_markdown(
            _ablation_comparison_data(runs, split="diagnostic")
        )
    baseline = runs[0]
    lines = [
        (
            "# Benchmark 来源消融对比"
            if allow_source_difference
            else "# Benchmark 配置对比"
        ),
        "",
        (
            f"> 固定查询集合，共 {baseline['config'].get('limit', '-')} 条；"
            "不代表隐藏测试集或比赛成绩，不进行显著性声明。"
        ),
        "",
        (
            "| 配置 | 检索源 | 策略（policy） | 成功率 | F1@5 | F1@10 | F1@20 | P@20 | R@20 | "
            "初始候选 Recall | 最终返回 Recall@20 | Judgement FN 率 | "
            "平均 gold rank | 平均 API | 平均延迟（秒） | compact 执行率 | "
            "compact 平均新增候选 | compact gold 增量 | 来源错误率 | 瓶颈标签 |"
        ),
        (
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        ),
    ]
    for run in runs:
        lines.append(_run_row(run))
    lines.extend(
        [
            "",
            "## 相对首个配置",
            "",
            "| 配置 | ΔF1@20 | ΔRecall@20 | Δ平均 API | Δ平均延迟（秒） |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        lines.append(_delta_row(run, baseline))
    lines.append("")
    return "\n".join(lines)


def build_ablation_comparison(
    run_dirs: list[Path],
    *,
    split: str,
    incomplete_reason: str | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if len(run_dirs) != 4 and not incomplete_reason:
        raise ValueError("ablation comparison requires exactly four runs")
    runs = [_load_run(path) for path in run_dirs]
    if runs:
        _validate_comparable(runs)
        _validate_ablation_subset(runs)
    if len(runs) == 4:
        _validate_ablation_matrix(runs)
    data = _ablation_comparison_data(
        runs,
        split=split,
        incomplete_reason=incomplete_reason,
    )
    return data, _ablation_markdown(data), _module_diagnostic_rows(runs, split)


def _load_run(path: Path) -> dict[str, Any]:
    run_dir = path.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    stage_metrics = _read_json(run_dir / "stage_metrics.json")
    results_path = run_dir / "results.jsonl"
    result_rows = _read_jsonl(results_path) if results_path.is_file() else []
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "config": config,
        "metrics": metrics,
        "stage_metrics": stage_metrics,
        "result_rows": result_rows,
    }


def _validate_comparable(
    runs: list[dict[str, Any]],
    *,
    allow_source_difference: bool = False,
) -> None:
    baseline = runs[0]["config"]
    fields = (
        "dataset",
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
        "runtime_code_hash",
    )
    if not allow_source_difference:
        fields = (*fields, "sources")
    for run in runs[1:]:
        mismatched = [
            field
            for field in fields
            if run["config"].get(field) != baseline.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"incompatible benchmark runs: {run['name']}: "
                + ",".join(mismatched)
            )


def _validate_ablation_matrix(runs: list[dict[str, Any]]) -> None:
    expected = {
        (False, False): "baseline",
        (True, False): "query_evolution_only",
        (False, True): "refchain_only",
        (True, True): "query_evolution_plus_refchain",
    }
    observed: dict[tuple[bool, bool], str] = {}
    for run in runs:
        config = run["config"]
        key = (
            bool(config.get("enable_query_evolution")),
            bool(config.get("enable_refchain")),
        )
        if key in observed:
            raise ValueError(f"duplicate ablation configuration: {expected[key]}")
        observed[key] = run["name"]
    missing = [name for key, name in expected.items() if key not in observed]
    if missing:
        raise ValueError("missing ablation configurations: " + ",".join(missing))


def _validate_ablation_subset(runs: list[dict[str, Any]]) -> None:
    observed: set[tuple[bool, bool]] = set()
    for run in runs:
        config = run["config"]
        key = (
            bool(config.get("enable_query_evolution")),
            bool(config.get("enable_refchain")),
        )
        if key in observed:
            raise ValueError(f"duplicate ablation configuration: {_ablation_name(config)}")
        observed.add(key)


def _ablation_name(config: dict[str, Any]) -> str:
    key = (
        bool(config.get("enable_query_evolution")),
        bool(config.get("enable_refchain")),
    )
    return {
        (False, False): "baseline",
        (True, False): "query_evolution_only",
        (False, True): "refchain_only",
        (True, True): "query_evolution_plus_refchain",
    }[key]


def _ablation_comparison_data(
    runs: list[dict[str, Any]],
    *,
    split: str,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    order = {
        "baseline": 0,
        "query_evolution_only": 1,
        "refchain_only": 2,
        "query_evolution_plus_refchain": 3,
    }
    for run in sorted(
        runs,
        key=lambda item: order[_ablation_name(item["config"])],
    ):
        metrics = run["metrics"]
        stage = run["stage_metrics"]
        end_to_end = metrics["end_to_end_metrics"]
        efficiency = metrics.get("efficiency") or {}
        benchmark = metrics["benchmark_statistics"]
        qe = stage.get("query_evolution") or {}
        refchain = stage.get("refchain") or {}
        costs = stage.get("stage_costs") or {}
        row = {
            "configuration": _ablation_name(run["config"]),
            "run_name": run["name"],
            "enable_query_evolution": bool(
                run["config"].get("enable_query_evolution")
            ),
            "enable_refchain": bool(run["config"].get("enable_refchain")),
            "f1_at_5": _at_k(end_to_end, "f1_at_k", 5),
            "f1_at_10": _at_k(end_to_end, "f1_at_k", 10),
            "f1_at_20": _at_k(end_to_end, "f1_at_k", 20),
            "recall_at_5": _at_k(end_to_end, "recall_at_k", 5),
            "recall_at_10": _at_k(end_to_end, "recall_at_k", 10),
            "recall_at_20": _at_k(end_to_end, "recall_at_k", 20),
            "precision_at_20": _at_k(end_to_end, "precision_at_k", 20),
            "initial_candidate_recall": stage.get("initial_retrieval_recall"),
            "post_query_evolution_candidate_recall": stage.get(
                "post_evolution_recall"
            ),
            "post_refchain_candidate_recall": stage.get("post_refchain_recall"),
            "final_returned_recall_at_20": (
                stage.get("final_returned_recall") or {}
            ).get("20"),
            "query_evolution_new_unique_candidates": int(
                qe.get("evolved_new_unique_candidate_count") or 0
            ),
            "query_evolution_new_unique_gold": int(
                qe.get("evolved_new_unique_gold_count") or 0
            ),
            "refchain_new_unique_candidates": int(
                refchain.get("new_unique_reference_count") or 0
            ),
            "refchain_new_unique_gold": int(
                refchain.get("new_unique_reference_gold_count") or 0
            ),
            "average_search_api_calls": float(
                efficiency.get("avg_search_api_call_count")
                or benchmark.get("average_api_calls")
                or 0.0
            ),
            "average_reference_api_calls": float(
                efficiency.get("avg_reference_api_call_count") or 0.0
            ),
            "average_total_api_calls": float(
                benchmark.get("average_api_calls") or 0.0
            ),
            "average_latency_seconds": float(
                benchmark.get("average_latency_seconds") or 0.0
            ),
            "average_query_evolution_latency_seconds": float(
                costs.get("average_query_evolution_latency_seconds") or 0.0
            ),
            "average_refchain_latency_seconds": float(
                costs.get("average_refchain_latency_seconds") or 0.0
            ),
            "recorded_initial_search_api_calls": int(
                costs.get("recorded_initial_search_api_calls") or 0
            ),
            "recorded_query_evolution_api_calls": int(
                costs.get("recorded_query_evolution_api_calls") or 0
            ),
            "recorded_refchain_api_calls": int(
                costs.get("recorded_refchain_api_calls") or 0
            ),
            "average_recorded_query_evolution_latency_seconds": float(
                costs.get("average_recorded_query_evolution_latency_seconds")
                or 0.0
            ),
            "average_recorded_refchain_latency_seconds": float(
                costs.get("average_recorded_refchain_latency_seconds") or 0.0
            ),
            "source_error_rate": float(
                (stage.get("source_contribution") or {}).get("source_error_rate")
                or 0.0
            ),
            "query_evolution_conclusions": list(qe.get("conclusions") or []),
            "refchain_conclusions": list(refchain.get("conclusions") or []),
            "query_evolution_marginal_costs": costs.get("query_evolution") or {},
            "refchain_marginal_costs": costs.get("refchain") or {},
            "query_evolution_gold_filtered_by_judgement": int(
                qe.get("gold_filtered_by_judgement_count") or 0
            ),
            "query_evolution_gold_lost_by_top_k": int(
                qe.get("gold_lost_by_top_k_count") or 0
            ),
            "refchain_gold_filtered_by_judgement": int(
                refchain.get("gold_filtered_by_judgement_count") or 0
            ),
            "refchain_gold_lost_by_top_k": int(
                refchain.get("gold_lost_by_top_k_count") or 0
            ),
            "query_evolution_diagnostics": qe,
            "refchain_diagnostics": refchain,
        }
        rows.append(row)
    baseline = next(
        (row for row in rows if row["configuration"] == "baseline"),
        None,
    )
    for row in rows:
        row["delta_f1_at_20"] = (
            row["f1_at_20"] - baseline["f1_at_20"] if baseline else None
        )
        row["delta_recall_at_20"] = (
            row["recall_at_20"] - baseline["recall_at_20"]
            if baseline
            else None
        )
        row["delta_total_api_calls"] = (
            row["average_total_api_calls"] - baseline["average_total_api_calls"]
            if baseline
            else None
        )
        row["delta_latency_seconds"] = (
            row["average_latency_seconds"] - baseline["average_latency_seconds"]
            if baseline
            else None
        )
    completed = {row["configuration"] for row in rows}
    expected = [
        "baseline",
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    ]
    first_config = runs[0]["config"] if runs else {}
    return {
        "split": split,
        "status": "complete" if len(rows) == 4 else "incomplete",
        "stop_reason": incomplete_reason,
        "sample_warning": "small_sample_diagnostic_only",
        "case_count": first_config.get("case_count"),
        "case_ids": first_config.get("case_ids") or [],
        "expected_configurations": expected,
        "missing_configurations": [name for name in expected if name not in completed],
        "shared_config": {
            key: first_config.get(key)
            for key in (
                "dataset",
                "offset",
                "limit",
                "sources",
                "query_adapter_policy",
                "run_profile",
                "result_policy",
                "top_k",
                "budgets",
                "llm",
            )
        },
        "configurations": rows,
    }


def _ablation_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"# QE/RefChain 四组消融：{data['split']}",
        "",
        "> small_sample_diagnostic_only；固定子集不代表完整 Benchmark。",
        "",
        (
            "| 配置 | F1@5/10/20 | R@20 | 候选 Recall 初始→QE→Ref→返回@20 | QE +候选/+gold | "
            "RefChain +候选/+gold | Search API | Ref API | 总 API | 延迟 | "
            "QE 延迟 | Ref 延迟 | 记录 QE/Ref API | 记录 QE/Ref 延迟 | "
            "错误率 | QE 结论 | RefChain 结论 |"
        ),
        (
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |"
        ),
    ]
    for row in data["configurations"]:
        lines.append(
            f"| {row['configuration']} | {row['f1_at_5']:.3f}/"
            f"{row['f1_at_10']:.3f}/{row['f1_at_20']:.3f} | "
            f"{row['recall_at_20']:.3f} | "
            f"{_optional(row['initial_candidate_recall'])}→"
            f"{_optional(row['post_query_evolution_candidate_recall'])}→"
            f"{_optional(row['post_refchain_candidate_recall'])}→"
            f"{_optional(row['final_returned_recall_at_20'])} | "
            f"{row['query_evolution_new_unique_candidates']}/"
            f"{row['query_evolution_new_unique_gold']} | "
            f"{row['refchain_new_unique_candidates']}/"
            f"{row['refchain_new_unique_gold']} | "
            f"{row['average_search_api_calls']:.3f} | "
            f"{row['average_reference_api_calls']:.3f} | "
            f"{row['average_total_api_calls']:.3f} | "
            f"{row['average_latency_seconds']:.3f} | "
            f"{row['average_query_evolution_latency_seconds']:.3f} | "
            f"{row['average_refchain_latency_seconds']:.3f} | "
            f"{row['recorded_query_evolution_api_calls']}/"
            f"{row['recorded_refchain_api_calls']} | "
            f"{row['average_recorded_query_evolution_latency_seconds']:.3f}/"
            f"{row['average_recorded_refchain_latency_seconds']:.3f} | "
            f"{row['source_error_rate']:.3f} | "
            f"{', '.join(row['query_evolution_conclusions']) or '-'} | "
            f"{', '.join(row['refchain_conclusions']) or '-'} |"
        )
    if data["status"] != "complete":
        lines.extend(
            [
                "",
                "## 未完成说明",
                "",
                f"- 停止原因：{data.get('stop_reason') or '未记录'}",
                "- 未完成配置："
                + "、".join(data.get("missing_configurations") or []),
                "- 未完成组没有指标，不以其他运行替代。",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _module_diagnostic_rows(
    runs: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        configuration = _ablation_name(run["config"])
        for result in run["result_rows"]:
            diagnostics = result.get("stage_diagnostics") or {}
            rows.append(
                {
                    "split": split,
                    "configuration": configuration,
                    "run_name": run["name"],
                    "case_id": result.get("case_id"),
                    "status": result.get("status"),
                    "query_evolution": diagnostics.get("query_evolution") or {},
                    "refchain": diagnostics.get("refchain") or {},
                    "stage_costs": diagnostics.get("stage_costs") or {},
                    "sample_warning": "small_sample_diagnostic_only",
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            str(item["split"]),
            str(item["configuration"]),
            str(item["case_id"]),
        ),
    )


def _run_row(run: dict[str, Any]) -> str:
    metrics = run["metrics"]
    stage = run["stage_metrics"]
    end_to_end = metrics["end_to_end_metrics"]
    stats = metrics["case_statistics"]
    efficiency = metrics["benchmark_statistics"]
    judgement = stage.get("judgement", {})
    reranking = stage.get("reranking", {})
    sources = stage.get("source_contribution", {})
    adaptive = (stage.get("retrieval_diagnostics") or {}).get("adaptive") or {}
    strategy = stage.get("query_strategy_contribution") or {}
    labels = ", ".join(stage.get("bottleneck_labels") or []) or "-"
    return (
        f"| {run['name']} | {', '.join(run['config'].get('sources') or []) or '-'} | "
        f"{run['config'].get('query_adapter_policy', '-')} | "
        f"{float(stats['success_rate']):.3f} | "
        f"{_at_k(end_to_end, 'f1_at_k', 5):.3f} | "
        f"{_at_k(end_to_end, 'f1_at_k', 10):.3f} | "
        f"{_at_k(end_to_end, 'f1_at_k', 20):.3f} | "
        f"{_at_k(end_to_end, 'precision_at_k', 20):.3f} | "
        f"{_at_k(end_to_end, 'recall_at_k', 20):.3f} | "
        f"{_optional(stage.get('initial_retrieval_recall'))} | "
        f"{_optional((stage.get('final_returned_recall') or {}).get('20'))} | "
        f"{float(judgement.get('gold_false_negative_rate') or 0.0):.3f} | "
        f"{_optional(reranking.get('average_gold_rank'))} | "
        f"{float(efficiency.get('average_api_calls') or 0.0):.3f} | "
        f"{float(efficiency.get('average_latency_seconds') or 0.0):.3f} | "
        f"{float(adaptive.get('compact_execution_ratio') or 0.0):.3f} | "
        f"{float(adaptive.get('compact_average_added_unique_candidates') or 0.0):.3f} | "
        f"{int(strategy.get('compact_gold_increment') or 0)} | "
        f"{float(sources.get('source_error_rate') or 0.0):.3f} | {labels} |"
    )


def _delta_row(run: dict[str, Any], baseline: dict[str, Any]) -> str:
    return (
        f"| {run['name']} | {_delta(run, baseline, 'f1'):.3f} | "
        f"{_delta(run, baseline, 'recall'):.3f} | "
        f"{_delta(run, baseline, 'api'):.3f} | "
        f"{_delta(run, baseline, 'latency'):.3f} |"
    )


def _delta(run: dict[str, Any], baseline: dict[str, Any], field: str) -> float:
    def value(item: dict[str, Any]) -> float:
        metrics = item["metrics"]
        if field == "f1":
            return _at_k(metrics["end_to_end_metrics"], "f1_at_k", 20)
        if field == "recall":
            return _at_k(metrics["end_to_end_metrics"], "recall_at_k", 20)
        if field == "api":
            return float(
                metrics["benchmark_statistics"].get("average_api_calls") or 0.0
            )
        return float(
            metrics["benchmark_statistics"].get("average_latency_seconds") or 0.0
        )

    return value(run) - value(baseline)


def _at_k(metrics: dict[str, Any], name: str, k: int) -> float:
    values = metrics.get(name) or {}
    return float(values.get(str(k), values.get(k, 0.0)))


def _optional(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "-"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing benchmark output: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid benchmark JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing benchmark output: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid benchmark JSONL: {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid benchmark JSONL object: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _merge_module_diagnostics(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    existing = _read_jsonl(path) if path.is_file() else []
    indexed = {
        (item.get("split"), item.get("configuration"), item.get("case_id")): item
        for item in [*existing, *rows]
    }
    ordered = [indexed[key] for key in sorted(indexed, key=lambda item: tuple(str(v) for v in item))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in ordered
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较多个 Benchmark 运行。")
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--split", default="diagnostic")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--module-diagnostics-output", default=None)
    parser.add_argument("--incomplete-reason", default=None)
    parser.add_argument(
        "--allow-source-difference",
        action="store_true",
        help="allow a source-ablation comparison while all other settings match",
    )
    args = parser.parse_args(argv)
    try:
        run_paths = [Path(item) for item in args.run]
        if args.ablation:
            data, report, diagnostic_rows = build_ablation_comparison(
                run_paths,
                split=args.split,
                incomplete_reason=args.incomplete_reason,
            )
        else:
            data = None
            diagnostic_rows = []
            report = compare_runs(
                run_paths,
                allow_source_difference=args.allow_source_difference,
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    if args.json_output:
        if data is None:
            raise ValueError("--json-output requires --ablation")
        _write_json(Path(args.json_output), data)
    if args.module_diagnostics_output:
        if data is None:
            raise ValueError("--module-diagnostics-output requires --ablation")
        _merge_module_diagnostics(
            Path(args.module_diagnostics_output),
            diagnostic_rows,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
