#!/usr/bin/env python3
"""比较 current_rules 与 disjunctive_facets 的冻结回放结果。"""

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

from scripts.analyze_query_planning_policies import (
    COMPARABLE_FIELDS,
    _diagnostic_rows,
    _load_run,
    _non_lower,
    _safe_ratio,
    _summarize_run,
    _write_json,
    _write_jsonl,
)


POLICIES = ("current_rules", "disjunctive_facets")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/disjunctive_facets_analysis")
NOISE_CATEGORIES = {"weakly_relevant", "irrelevant"}


def build_disjunctive_facets_analysis(
    *,
    development_current: Path,
    development_disjunctive: Path,
    validation_current: Path,
    validation_disjunctive: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "current_rules": _load_run(development_current),
            "disjunctive_facets": _load_run(development_disjunctive),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "disjunctive_facets": _load_run(validation_disjunctive),
        },
    }
    _validate_split(runs["development"], offset=130)
    _validate_split(runs["validation"], offset=150)
    summaries = {
        split: {
            policy: _disjunctive_summary(run, split=split, policy=policy)
            for policy, run in split_runs.items()
        }
        for split, split_runs in runs.items()
    }
    acceptance = _acceptance(
        summaries["validation"]["current_rules"],
        summaries["validation"]["disjunctive_facets"],
    )
    comparison = {
        "policies": list(POLICIES),
        "development_rule_status": "frozen_before_validation",
        "validation_run_policy": "single_frozen_evaluation",
        "splits": summaries,
        "validation_failure_deltas": _failure_deltas(
            summaries["validation"]["current_rules"],
            summaries["validation"]["disjunctive_facets"],
        ),
        "validation_acceptance": acceptance,
        "recommended_policy": (
            "disjunctive_facets" if acceptance["accepted"] else "current_rules"
        ),
        "product_default": "current_rules",
        "product_default_changed": False,
        "limitations": [
            "固定 20+20 小样本用于受控策略诊断，不代表完整 Benchmark 成绩。",
            "gold 仅在 SearchService 完成后用于匹配和 OR 查询贡献归因。",
            "验证集只按冻结规则运行一次；产品默认策略本轮不自动切换。",
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


def _validate_split(runs: dict[str, dict[str, Any]], *, offset: int) -> None:
    if set(runs) != set(POLICIES):
        raise ValueError("each split requires current_rules and disjunctive_facets")
    current = runs["current_rules"]
    disjunctive = runs["disjunctive_facets"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if current["config"].get(field) != disjunctive["config"].get(field)
    ]
    if mismatched:
        raise ValueError("incompatible disjunctive runs: " + ",".join(mismatched))
    for policy, run in runs.items():
        config = run["config"]
        if config.get("query_planning_policy") != policy:
            raise ValueError(f"query planning policy mismatch: {policy}")
        if config.get("offset") != offset or config.get("limit") != 20:
            raise ValueError(f"unexpected fixed split: offset={offset},limit=20")
        if len(config.get("case_ids") or []) != 20:
            raise ValueError("fixed split must contain exactly 20 cases")
        if config.get("sources") != ["arxiv"]:
            raise ValueError("disjunctive comparison requires arxiv only")
        if config.get("query_adapter_policy") != "adaptive":
            raise ValueError("disjunctive comparison requires adaptive adapter")
        if config.get("retrieval_mode") != "replay":
            raise ValueError("disjunctive comparison requires frozen replay")
        if (
            config.get("enable_query_evolution")
            or config.get("query_evolution_policy") != "off"
            or config.get("enable_refchain")
        ):
            raise ValueError("disjunctive comparison requires later stages off")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError("disjunctive comparison requires LLM off")
        if config.get("judgement_policy") != "current_rules":
            raise ValueError("disjunctive comparison requires current_rules judgement")
        if (
            config.get("run_profile") != "balanced"
            or config.get("top_k") != 20
            or config.get("max_workers") != 1
        ):
            raise ValueError("disjunctive comparison requires balanced top20 single worker")
        if config.get("result_policy") != "highly_and_partial":
            raise ValueError("disjunctive comparison result policy mismatch")


def _disjunctive_summary(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> dict[str, Any]:
    summary = _summarize_run(run, split=split, policy=policy)
    summary["or_query_contribution"] = _or_query_contribution(run)
    ratio, counts = _weak_irrelevant_ratio(run)
    summary["weak_irrelevant_ratio"] = ratio
    summary["judgement_category_counts"] = counts
    summary["retrieval_failure_counts"] = _retrieval_failure_counts(run)
    return summary


def _or_query_contribution(run: dict[str, Any]) -> dict[str, int | float]:
    values: Counter[str] = Counter()
    for result in run["results"]:
        planning = (result.get("stage_diagnostics") or {}).get(
            "initial_query_planning"
        ) or {}
        for row in planning.get("subqueries") or []:
            if row.get("combination_mode") != "any":
                continue
            values["logical_query_count"] += 1
            values["adapted_query_count"] += len(row.get("adapted_queries") or [])
            values["raw_candidate_count"] += int(row.get("raw_candidate_count") or 0)
            values["unique_candidate_count"] += int(
                row.get("unique_candidate_count") or 0
            )
            values["exclusive_candidate_count"] += int(
                row.get("exclusive_candidate_count") or 0
            )
            values["post_run_unique_gold_hit_count"] += int(
                row.get("post_run_unique_gold_hit_count") or 0
            )
            values["recorded_request_count"] += int(
                row.get("recorded_request_count") or 0
            )
            values["recorded_latency_milliseconds"] += round(
                float(row.get("recorded_latency_seconds") or 0.0) * 1000
            )
    return dict(sorted(values.items()))


def _weak_irrelevant_ratio(
    run: dict[str, Any],
) -> tuple[float | None, dict[str, int]]:
    counts: Counter[str] = Counter()
    for result in run["results"]:
        snapshots = (result.get("stage_diagnostics") or {}).get("snapshots") or []
        judged = next(
            (item for item in snapshots if item.get("stage") == "initial_judged"),
            None,
        )
        if judged is None:
            continue
        counts.update(
            str(candidate.get("category") or "unjudged")
            for candidate in judged.get("candidates") or []
        )
    total = sum(counts.values())
    noise = sum(counts[category] for category in NOISE_CATEGORIES)
    return (noise / total if total else None), dict(sorted(counts.items()))


def _retrieval_failure_counts(run: dict[str, Any]) -> dict[str, int]:
    counts = Counter({"query_not_matched": 0, "ranking_cutoff": 0, "over_broad": 0})
    for result in run["results"]:
        diagnostics = result.get("stage_diagnostics") or {}
        planning = diagnostics.get("initial_query_planning") or {}
        counts["over_broad"] += int(
            (planning.get("ineffective_reasons") or {}).get("over_broad") or 0
        )
        for gold in diagnostics.get("gold_diagnostics") or []:
            if gold.get("drop_reason") == "not_retrieved":
                counts["query_not_matched"] += 1
            elif gold.get("drop_reason") == "outside_final_top_k":
                counts["ranking_cutoff"] += 1
    return dict(sorted(counts.items()))


def _failure_deltas(
    current: dict[str, Any],
    disjunctive: dict[str, Any],
) -> dict[str, int]:
    current_counts = current["retrieval_failure_counts"]
    disjunctive_counts = disjunctive["retrieval_failure_counts"]
    return {
        key: int(disjunctive_counts.get(key, 0)) - int(current_counts.get(key, 0))
        for key in sorted(set(current_counts) | set(disjunctive_counts))
    }


def _acceptance(current: dict[str, Any], disjunctive: dict[str, Any]) -> dict[str, Any]:
    current_api = float(current.get("average_recorded_api_calls") or 0.0)
    disjunctive_api = float(disjunctive.get("average_recorded_api_calls") or 0.0)
    checks = {
        "at_least_one_new_unique_gold": (
            int(disjunctive.get("unique_gold_count") or 0)
            >= int(current.get("unique_gold_count") or 0) + 1
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
        "api_calls_within_1_5x": (
            disjunctive_api == 0.0
            if current_api == 0.0
            else disjunctive_api <= current_api * 1.5
        ),
        "weak_irrelevant_ratio_within_0_10": _within_tolerance(
            disjunctive.get("weak_irrelevant_ratio"),
            current.get("weak_irrelevant_ratio"),
            0.10,
        ),
        "frozen_replay_zero_network": all(
            float(row.get(field) or 0.0) == 0.0
            for row in (current, disjunctive)
            for field in (
                "replay_execution_request_count",
                "replay_execution_retry_count",
                "replay_execution_network_wait_seconds",
            )
        ),
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "checks": checks,
        "api_call_ratio": _safe_ratio(disjunctive_api, current_api),
        "unique_gold_gain": (
            int(disjunctive.get("unique_gold_count") or 0)
            - int(current.get("unique_gold_count") or 0)
        ),
        "decision": (
            "disjunctive_facets 通过验证集小样本验收，但仍保持实验策略"
            if accepted
            else "保留 current_rules 为默认策略"
        ),
    }


def _within_tolerance(value: Any, baseline: Any, tolerance: float) -> bool:
    if value is None or baseline is None:
        return True
    return float(value) <= float(baseline) + tolerance


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 析取式分面检索对比",
        "",
        "> 开发集冻结规则后，仅在独立验证集运行一次；gold 只用于事后评估。",
        "",
        "| 切片 | 策略 | 候选 Recall | F1@5 | F1@10 | F1@20 | P@20 | R@20 | MRR | nDCG@20 | 唯一 gold | API/例 | 弱相关+无关率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "validation"):
        for policy in POLICIES:
            row = comparison["splits"][split][policy]
            values = [
                split,
                policy,
                _fmt(row.get("candidate_recall")),
                _fmt(row.get("f1_at_5")),
                _fmt(row.get("f1_at_10")),
                _fmt(row.get("f1_at_20")),
                _fmt(row.get("precision_at_20")),
                _fmt(row.get("recall_at_20")),
                _fmt(row.get("mrr")),
                _fmt(row.get("ndcg_at_20")),
                str(row.get("unique_gold_count") or 0),
                _fmt(row.get("average_recorded_api_calls")),
                _fmt(row.get("weak_irrelevant_ratio")),
            ]
            lines.append("| " + " | ".join(values) + " |")
    acceptance = comparison["validation_acceptance"]
    lines.extend(
        [
            "",
            "## 验收",
            "",
            f"- 结论：{acceptance['decision']}。",
            f"- 新增 unique gold：{acceptance['unique_gold_gain']}。",
            f"- 全部验收项通过：{'是' if acceptance['accepted'] else '否'}。",
            "- 产品默认策略保持 current_rules。",
            "- replay 的 HTTP、重试与网络等待必须均为 0。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-current", type=Path, required=True)
    parser.add_argument("--development-disjunctive", type=Path, required=True)
    parser.add_argument("--validation-current", type=Path, required=True)
    parser.add_argument("--validation-disjunctive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = build_disjunctive_facets_analysis(
        development_current=args.development_current,
        development_disjunctive=args.development_disjunctive,
        validation_current=args.validation_current,
        validation_disjunctive=args.validation_disjunctive,
        output_dir=args.output_dir,
    )
    print(json.dumps(comparison["validation_acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
