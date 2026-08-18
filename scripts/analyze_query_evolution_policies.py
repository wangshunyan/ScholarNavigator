#!/usr/bin/env python3
"""比较 baseline、旧 seed 扩展和覆盖缺口策略的冻结快照结果。"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


POLICIES = ("baseline", "seed_expansion", "coverage_gap")
INVALID_CATEGORIES = {
    "weakly_relevant",
    "irrelevant",
    "insufficient_evidence",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象：{path}")
        rows.append(payload)
    return rows


def _at_k(metrics: dict[str, Any], name: str, k: int) -> float:
    values = metrics.get(name) or {}
    return float(values.get(str(k), values.get(k, 0.0)) or 0.0)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _summarize_run(run_dir: Path, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = _read_json(run_dir / "metrics.json")
    config = _read_json(run_dir / "config.json")
    rows = _read_jsonl(run_dir / "results.jsonl")
    end_to_end = metrics.get("end_to_end_metrics") or {}
    categories: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    ineffective_reasons: Counter[str] = Counter()
    raw = unique = new_unique = new_gold = 0
    triggered = seeds = generated = 0
    quality_raw = quality_filtered = quality_duplicates = 0
    diagnostics_rows: list[dict[str, Any]] = []
    api_calls = 0
    recorded_api_calls = 0
    recorded_latency = 0.0
    latency = 0.0
    for row in rows:
        module = (row.get("stage_diagnostics") or {}).get("query_evolution") or {}
        costs = row.get("cost_report") or {}
        snapshot = row.get("snapshot_cost_report") or {}
        category_counts = (module.get("new_candidate_categories") or {}).get("counts") or {}
        gate = module.get("quality_gate") or {}
        query_details = module.get("queries") or []
        categories.update(category_counts)
        skipped = module.get("skipped_reasons") or []
        if isinstance(skipped, dict):
            skips.update({str(key): int(value) for key, value in skipped.items()})
        else:
            skips.update(str(value) for value in skipped)
        triggered += bool(module.get("triggered"))
        seeds += int(module.get("selected_seed_count") or 0)
        generated += int(module.get("generated_query_count") or 0)
        raw += int(module.get("evolved_raw_candidate_count") or 0)
        unique += int(module.get("evolved_unique_candidate_count") or 0)
        new_unique += int(module.get("evolved_new_unique_candidate_count") or 0)
        new_gold += int(module.get("evolved_new_unique_gold_count") or 0)
        quality_raw += int(gate.get("raw_candidate_count") or 0)
        quality_filtered += int(gate.get("filtered_candidate_count") or 0)
        quality_duplicates += int(gate.get("duplicate_candidate_count") or 0)
        for detail in query_details:
            ineffective_reasons.update(detail.get("ineffective_reasons") or [])
        api_calls += int(costs.get("search_api_call_count") or 0)
        recorded_api_calls += int(snapshot.get("recorded_search_request_count") or 0)
        recorded_latency += float(snapshot.get("recorded_latency_seconds") or 0.0)
        latency += float(row.get("latency_seconds") or 0.0)
        diagnostics_rows.append(
            {
                "policy": policy,
                "case_id": row.get("case_id"),
                "original_query": module.get("original_query") or row.get("query"),
                "query_intent": module.get("query_intent"),
                "constraints": module.get("constraints") or {},
                "eligible_seed_count": int(
                    module.get("eligible_seed_count") or 0
                ),
                "eligible_seeds": module.get("eligible_seed_titles") or [],
                "selected_seed_count": int(
                    module.get("selected_seed_count") or 0
                ),
                "selected_seeds": module.get("selected_seed_titles") or [],
                "generated_evolved_queries": query_details,
                "status": row.get("status"),
                "query_evolution": module,
                "search_api_call_count": int(costs.get("search_api_call_count") or 0),
                "recorded_search_request_count": int(
                    snapshot.get("recorded_search_request_count") or 0
                ),
                "recorded_latency_seconds": float(
                    snapshot.get("recorded_latency_seconds") or 0.0
                ),
            }
        )
    invalid = sum(categories[category] for category in INVALID_CATEGORIES)
    category_total = sum(categories.values())
    case_count = len(rows)
    effective_api_calls = recorded_api_calls or api_calls
    effective_latency = recorded_latency or latency
    summary = {
        "policy": policy,
        "run_dir": str(run_dir.resolve()),
        "case_count": case_count,
        "config_policy": config.get("query_evolution_policy", "off"),
        "f1_at_20": _at_k(end_to_end, "f1_at_k", 20),
        "recall_at_20": _at_k(end_to_end, "recall_at_k", 20),
        "precision_at_20": _at_k(end_to_end, "precision_at_k", 20),
        "mrr": float(end_to_end.get("mrr") or 0.0),
        "ndcg_at_20": _at_k(end_to_end, "ndcg_at_k", 20),
        "triggered_case_count": triggered,
        "skipped_case_count": max(0, case_count - triggered),
        "trigger_rate": _ratio(triggered, case_count),
        "skipped_reasons": dict(sorted(skips.items())),
        "ineffective_query_reasons": dict(
            sorted(ineffective_reasons.items())
        ),
        "selected_seed_count": seeds,
        "generated_query_count": generated,
        "queries_per_triggered_case": _ratio(generated, triggered),
        "seeds_per_triggered_case": _ratio(seeds, triggered),
        "raw_candidate_count": raw,
        "unique_candidate_count": unique,
        "new_unique_candidate_count": new_unique,
        "duplicate_ratio": _ratio(max(0, raw - unique), raw),
        "new_candidate_categories": {
            "counts": dict(sorted(categories.items())),
            "ratios": {
                key: value / category_total
                for key, value in sorted(categories.items())
            },
        },
        "invalid_candidate_count": invalid,
        "invalid_candidate_share": _ratio(invalid, category_total),
        "quality_gate_raw_candidate_count": quality_raw,
        "quality_gate_filtered_candidate_count": quality_filtered,
        "quality_gate_duplicate_candidate_count": quality_duplicates,
        "quality_gate_filtered_share": _ratio(quality_filtered, quality_raw),
        "new_unique_gold_count": new_gold,
        "search_api_call_count": api_calls,
        "recorded_search_request_count": recorded_api_calls,
        "effective_search_request_count": effective_api_calls,
        "latency_seconds": latency,
        "recorded_latency_seconds": recorded_latency,
        "effective_latency_seconds": effective_latency,
        "api_calls_per_new_gold": _ratio(effective_api_calls, new_gold),
        "latency_per_new_gold": _ratio(effective_latency, new_gold),
    }
    return summary, diagnostics_rows


def _acceptance(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summaries["baseline"]
    seed = summaries["seed_expansion"]
    gap = summaries["coverage_gap"]
    quality_better = (
        gap["invalid_candidate_share"] is not None
        and seed["invalid_candidate_share"] is not None
        and gap["invalid_candidate_share"] < seed["invalid_candidate_share"]
    ) or gap["invalid_candidate_count"] < seed["invalid_candidate_count"]
    checks = {
        "f1_not_below_baseline": gap["f1_at_20"] >= baseline["f1_at_20"],
        "recall_not_below_baseline": gap["recall_at_20"] >= baseline["recall_at_20"],
        "api_not_above_seed": (
            gap["effective_search_request_count"]
            <= seed["effective_search_request_count"]
        ),
        "candidate_quality_better_than_seed": quality_better,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summary_markdown(
    label: str,
    summaries: dict[str, dict[str, Any]],
    acceptance: dict[str, Any],
) -> str:
    lines = [
        f"# Query Evolution 策略对比：{label}",
        "",
        "| 策略 | F1@20 | Recall@20 | Precision@20 | MRR | nDCG@20 | 触发案例 | 演化查询 | 新 gold | 有效 API | 记录延迟(秒) | 无效候选占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        row = summaries[policy]
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    _format(row["f1_at_20"]),
                    _format(row["recall_at_20"]),
                    _format(row["precision_at_20"]),
                    _format(row["mrr"]),
                    _format(row["ndcg_at_20"]),
                    _format(row["triggered_case_count"]),
                    _format(row["generated_query_count"]),
                    _format(row["new_unique_gold_count"]),
                    _format(row["effective_search_request_count"]),
                    _format(row["recorded_latency_seconds"]),
                    _format(row["invalid_candidate_share"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- 验收结论：{'通过' if acceptance['passed'] else '未通过'}。",
            *(
                f"- {name}：{'通过' if passed else '未通过'}。"
                for name, passed in acceptance["checks"].items()
            ),
            "",
            "## 演化查询无效原因",
            "",
            *(
                f"- {policy}：{_format_reason_counts(summaries[policy]['ineffective_query_reasons'])}"
                for policy in POLICIES
            ),
            "",
            "> 结果仅适用于指定冻结快照、固定子集和当前代码版本，不代表完整 Benchmark 性能。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_reason_counts(reasons: dict[str, int]) -> str:
    if not reasons:
        return "无演化查询或未记录无效原因。"
    return "、".join(
        f"{reason}={count}"
        for reason, count in sorted(
            reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ) + "。"


def analyze(
    *,
    baseline: Path,
    seed_expansion: Path,
    coverage_gap: Path,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    paths = {
        "baseline": baseline,
        "seed_expansion": seed_expansion,
        "coverage_gap": coverage_gap,
    }
    summaries: dict[str, dict[str, Any]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for policy, path in paths.items():
        summaries[policy], rows = _summarize_run(path, policy)
        diagnostic_rows.extend(rows)
    acceptance = _acceptance(summaries)
    payload = {
        "label": label,
        "policies": summaries,
        "acceptance": acceptance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "comparison.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_dir / "query_diagnostics.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in diagnostic_rows
        ),
    )
    _atomic_write(
        output_dir / "summary.md",
        _summary_markdown(label, summaries, acceptance),
    )
    return payload


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--seed-expansion", type=Path, required=True)
    parser.add_argument("--coverage-gap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="固定子集")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = analyze(
        baseline=args.baseline,
        seed_expansion=args.seed_expansion,
        coverage_gap=args.coverage_gap,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(json.dumps(payload["acceptance"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
