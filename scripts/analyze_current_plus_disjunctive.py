#!/usr/bin/env python3
"""比较 current_rules 与加法式单 OR 策略的冻结回放结果。"""

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

from scripts.analyze_disjunctive_facets import (  # noqa: E402
    _or_query_contribution,
    _weak_irrelevant_ratio,
)
from scripts.analyze_query_planning_policies import (  # noqa: E402
    COMPARABLE_FIELDS,
    _diagnostic_rows,
    _load_run,
    _non_lower,
    _safe_ratio,
    _summarize_run,
    _write_json,
    _write_jsonl,
)


POLICIES = ("current_rules", "current_plus_disjunctive")
SPLITS = {"development": 210, "validation": 230}
DEFAULT_OUTPUT_DIR = Path(
    "outputs/benchmark_runs/current_plus_disjunctive_analysis"
)


def build_current_plus_analysis(
    *,
    development_current: Path,
    development_candidate: Path,
    validation_current: Path,
    validation_candidate: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "current_rules": _load_run(development_current),
            "current_plus_disjunctive": _load_run(development_candidate),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "current_plus_disjunctive": _load_run(validation_candidate),
        },
    }
    for split, offset in SPLITS.items():
        _validate_split(runs[split], offset=offset)

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    retention: dict[str, dict[str, Any]] = {}
    for split, split_runs in runs.items():
        summaries[split] = {
            policy: _current_plus_summary(run, split=split, policy=policy)
            for policy, run in split_runs.items()
        }
        retention[split] = _gold_retention(
            split_runs["current_rules"],
            split_runs["current_plus_disjunctive"],
        )
        summaries[split]["current_plus_disjunctive"][
            "baseline_gold_retention"
        ] = retention[split]

    acceptance = _acceptance(
        summaries["validation"]["current_rules"],
        summaries["validation"]["current_plus_disjunctive"],
        retention["validation"],
    )
    comparison = {
        "policies": list(POLICIES),
        "protocol": {
            "development_offset": SPLITS["development"],
            "validation_offset": SPLITS["validation"],
            "limit": 20,
            "sources": ["arxiv"],
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "query_evolution_policy": "off",
            "refchain": False,
            "llm": False,
            "run_profile": "balanced",
            "top_k": 20,
            "gold_usage": "post_search_evaluation_only",
        },
        "development_rule_status": "frozen_before_validation",
        "validation_run_policy": "single_frozen_evaluation",
        "splits": summaries,
        "gold_retention": retention,
        "validation_acceptance": acceptance,
        "high_recall_profile_candidate": acceptance["accepted"],
        "product_default": "current_rules",
        "product_default_changed": False,
        "production_rules_gold_free": True,
        "limitations": [
            "固定 20+20 小样本只用于受控策略诊断，不代表完整 Benchmark。",
            "gold 只在两组 SearchService 完成后用于集合比较与贡献归因。",
            "验证集只按开发集冻结规则运行一次，查看结果后不调参。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", comparison)
    for split in SPLITS:
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
        raise ValueError("each split requires current and additive policies")
    current = runs["current_rules"]
    candidate = runs["current_plus_disjunctive"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if current["config"].get(field) != candidate["config"].get(field)
    ]
    if mismatched:
        raise ValueError("incompatible current-plus runs: " + ",".join(mismatched))
    for policy, run in runs.items():
        config = run["config"]
        if config.get("query_planning_policy") != policy:
            raise ValueError(f"query planning policy mismatch:{policy}")
        if config.get("offset") != offset or config.get("limit") != 20:
            raise ValueError(f"unexpected fixed split:offset={offset},limit=20")
        if len(config.get("case_ids") or []) != 20:
            raise ValueError("fixed split must contain exactly 20 cases")
        if config.get("sources") != ["arxiv"]:
            raise ValueError("current-plus comparison requires arxiv only")
        if config.get("query_adapter_policy") != "adaptive":
            raise ValueError("current-plus comparison requires adaptive adapter")
        if config.get("retrieval_mode") != "replay":
            raise ValueError("current-plus comparison requires frozen replay")
        if (
            config.get("enable_query_evolution")
            or config.get("query_evolution_policy") != "off"
            or config.get("enable_refchain")
        ):
            raise ValueError("current-plus comparison requires later stages off")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError("current-plus comparison requires LLM off")
        if config.get("judgement_policy") != "current_rules":
            raise ValueError("current-plus comparison requires current judgement")
        if (
            config.get("run_profile") != "balanced"
            or config.get("top_k") != 20
            or config.get("max_workers") != 1
        ):
            raise ValueError("current-plus comparison requires balanced top20 single worker")
        if config.get("result_policy") != "highly_and_partial":
            raise ValueError("current-plus result policy mismatch")
        _assert_zero_replay_cost(run["metrics"])


def _current_plus_summary(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> dict[str, Any]:
    summary = _summarize_run(run, split=split, policy=policy)
    judgement = run["stage_metrics"].get("judgement") or {}
    ratio, counts = _weak_irrelevant_ratio(run)
    summary["gold_judgement_retained_count"] = (
        int(judgement.get("gold_judged_highly_relevant") or 0)
        + int(judgement.get("gold_judged_partially_relevant") or 0)
    )
    summary["final_returned_gold_count"] = _returned_gold_count(run)
    summary["weak_irrelevant_ratio"] = ratio
    summary["judgement_category_counts"] = counts
    summary["or_query_contribution"] = _or_query_contribution(run)
    summary["or_execution"] = _or_execution(run)
    return summary


def _returned_gold_count(run: dict[str, Any]) -> int:
    return sum(
        gold.get("drop_reason") == "returned"
        for result in run["results"]
        for gold in (result.get("stage_diagnostics") or {}).get(
            "gold_diagnostics", []
        )
    )


def _gold_retention(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    current_keys = _retrieved_gold_keys(current)
    candidate_keys = _retrieved_gold_keys(candidate)
    lost = current_keys - candidate_keys
    added = candidate_keys - current_keys
    return {
        "baseline_retrieved_gold_count": len(current_keys),
        "candidate_retrieved_gold_count": len(candidate_keys),
        "retained_baseline_gold_count": len(current_keys & candidate_keys),
        "lost_baseline_gold_count": len(lost),
        "net_new_gold_count": len(added),
        "all_baseline_gold_retained": not lost,
    }


def _retrieved_gold_keys(run: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for result in run["results"]:
        case_id = str(result.get("case_id") or "")
        diagnostics = result.get("stage_diagnostics") or {}
        for index, gold in enumerate(diagnostics.get("gold_diagnostics") or []):
            if not gold.get("found"):
                continue
            identity = gold.get("gold_id") or gold.get("gold_title") or index
            keys.add(f"{case_id}:{str(identity).casefold().strip()}")
    return keys


def _or_execution(run: dict[str, Any]) -> dict[str, Any]:
    values: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    planning_skips: Counter[str] = Counter()
    for result in run["results"]:
        planning = (result.get("stage_diagnostics") or {}).get(
            "initial_query_planning"
        ) or {}
        for reason in (planning.get("planning") or {}).get("skipped_facets") or []:
            if "current_plus_disjunctive" in str(reason):
                planning_skips[str(reason)] += 1
        for row in planning.get("subqueries") or []:
            if row.get("purpose") != "current_plus_disjunctive_any":
                continue
            values[f"{row.get('status') or 'unknown'}_query_count"] += 1
            values["recorded_request_count"] += int(
                row.get("recorded_request_count") or 0
            )
            for reason in row.get("skip_reasons") or []:
                skip_reasons[str(reason)] += 1
        budget = (result.get("result") or {}).get("budget_status") or {}
        values["candidate_limit_applied_case_count"] += int(
            bool(budget.get("candidate_limit_applied"))
        )
        values["candidate_truncation_count"] += len(
            budget.get("candidate_truncations") or []
        )
    return {
        **dict(sorted(values.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "planning_skip_reasons": dict(sorted(planning_skips.items())),
    }


def _assert_zero_replay_cost(metrics: dict[str, Any]) -> None:
    costs = metrics.get("snapshot_costs") or {}
    values = {
        "http": float(costs.get("replay_execution_request_count") or 0.0),
        "retry": float(costs.get("replay_execution_retry_count") or 0.0),
        "network_wait": float(
            costs.get("replay_execution_network_wait_seconds") or 0.0
        ),
    }
    if any(values.values()):
        raise ValueError(f"frozen replay executed network work:{values}")


def _acceptance(
    current: dict[str, Any],
    candidate: dict[str, Any],
    retention: dict[str, Any],
) -> dict[str, Any]:
    current_api = float(current.get("average_recorded_api_calls") or 0.0)
    candidate_api = float(candidate.get("average_recorded_api_calls") or 0.0)
    checks = {
        "at_least_one_net_new_unique_gold": (
            int(retention["net_new_gold_count"]) >= 1
        ),
        "all_baseline_gold_retained": bool(
            retention["all_baseline_gold_retained"]
        ),
        "recall_at_20_non_regression": _non_lower(
            candidate.get("recall_at_20"), current.get("recall_at_20")
        ),
        "f1_at_20_non_regression": _non_lower(
            candidate.get("f1_at_20"), current.get("f1_at_20")
        ),
        "mrr_non_regression": _non_lower(
            candidate.get("mrr"), current.get("mrr")
        ),
        "ndcg_at_20_non_regression": _non_lower(
            candidate.get("ndcg_at_20"), current.get("ndcg_at_20")
        ),
        "api_calls_within_1_5x": (
            candidate_api == 0.0
            if current_api == 0.0
            else candidate_api <= current_api * 1.5
        ),
        "weak_irrelevant_ratio_within_0_10": _within_tolerance(
            candidate.get("weak_irrelevant_ratio"),
            current.get("weak_irrelevant_ratio"),
            0.10,
        ),
        "frozen_replay_zero_network": all(
            float(row.get(field) or 0.0) == 0.0
            for row in (current, candidate)
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
        "api_call_ratio": _safe_ratio(candidate_api, current_api),
        "net_new_gold_count": int(retention["net_new_gold_count"]),
        "lost_baseline_gold_count": int(retention["lost_baseline_gold_count"]),
        "decision": (
            "current_plus_disjunctive 可作为 high_recall profile 候选"
            if accepted
            else "停止继续扩展 OR 查询策略，产品默认保持 current_rules"
        ),
    }


def _within_tolerance(value: Any, baseline: Any, tolerance: float) -> bool:
    if value is None or baseline is None:
        return True
    return float(value) <= float(baseline) + tolerance


def _summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 加法式单 OR 检索对比",
        "",
        "> 开发集冻结规则后只运行一次独立验证；gold 只用于事后评估。",
        "",
        "| 切片 | 策略 | 候选 Recall | unique gold | F1@5 | F1@10 | F1@20 | R@20 | MRR | nDCG@20 | API/例 | 弱相关+无关率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for policy in POLICIES:
            row = comparison["splits"][split][policy]
            values = [
                split,
                policy,
                _fmt(row.get("candidate_recall")),
                str(row.get("unique_gold_count") or 0),
                _fmt(row.get("f1_at_5")),
                _fmt(row.get("f1_at_10")),
                _fmt(row.get("f1_at_20")),
                _fmt(row.get("recall_at_20")),
                _fmt(row.get("mrr")),
                _fmt(row.get("ndcg_at_20")),
                _fmt(row.get("average_recorded_api_calls")),
                _fmt(row.get("weak_irrelevant_ratio")),
            ]
            lines.append("| " + " | ".join(values) + " |")
    retention = comparison["gold_retention"]["validation"]
    acceptance = comparison["validation_acceptance"]
    lines.extend(
        [
            "",
            "## 阶段 gold",
            "",
            *(
                f"- {split} / {policy}：Judgement 保留 "
                f"{comparison['splits'][split][policy]['gold_judgement_retained_count']}，"
                f"最终返回 {comparison['splits'][split][policy]['final_returned_gold_count']}。"
                for split in SPLITS
                for policy in POLICIES
            ),
            "",
            "## 验收",
            "",
            f"- 基线 gold 全部保留：{'是' if retention['all_baseline_gold_retained'] else '否'}。",
            f"- 净新增 gold：{retention['net_new_gold_count']}。",
            f"- 结论：{acceptance['decision']}。",
            f"- 全部门槛通过：{'是' if acceptance['accepted'] else '否'}。",
            "- 产品默认仍为 current_rules。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-current", type=Path, required=True)
    parser.add_argument("--development-candidate", type=Path, required=True)
    parser.add_argument("--validation-current", type=Path, required=True)
    parser.add_argument("--validation-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = build_current_plus_analysis(
        development_current=args.development_current,
        development_candidate=args.development_candidate,
        validation_current=args.validation_current,
        validation_candidate=args.validation_candidate,
        output_dir=args.output_dir,
    )
    print(json.dumps(comparison["validation_acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
