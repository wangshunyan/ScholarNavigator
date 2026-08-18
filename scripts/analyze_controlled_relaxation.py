#!/usr/bin/env python3
"""比较 current_rules 与 controlled_relaxation 的冻结回放结果。"""

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


POLICIES = ("current_rules", "controlled_relaxation")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/benchmark_runs/controlled_relaxation_analysis"
)
INVALID_CATEGORIES = {"irrelevant", "insufficient_evidence"}


def build_controlled_relaxation_analysis(
    *,
    development_current: Path,
    development_controlled: Path,
    validation_current: Path,
    validation_controlled: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "current_rules": _load_run(development_current),
            "controlled_relaxation": _load_run(development_controlled),
        },
        "validation": {
            "current_rules": _load_run(validation_current),
            "controlled_relaxation": _load_run(validation_controlled),
        },
    }
    _validate_split(runs["development"], offset=50)
    _validate_split(runs["validation"], offset=70)
    summaries = {
        split: {
            policy: _controlled_summary(run, split=split, policy=policy)
            for policy, run in split_runs.items()
        }
        for split, split_runs in runs.items()
    }
    acceptance = _acceptance(
        summaries["validation"]["current_rules"],
        summaries["validation"]["controlled_relaxation"],
    )
    comparison = {
        "policies": list(POLICIES),
        "development_rule_status": "frozen_before_validation",
        "validation_run_policy": "single_frozen_evaluation",
        "splits": summaries,
        "validation_acceptance": acceptance,
        "product_default": (
            "controlled_relaxation" if acceptance["accepted"] else "current_rules"
        ),
        "limitations": [
            "固定 20+20 小样本用于受控策略诊断，不代表完整 Benchmark 成绩。",
            "gold 仅在 SearchService 完成后用于匹配和贡献归因。",
            "holdout30 未参与本策略的参数选择。",
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
        raise ValueError("each split requires current_rules and controlled_relaxation")
    current = runs["current_rules"]
    controlled = runs["controlled_relaxation"]
    mismatched = [
        field
        for field in COMPARABLE_FIELDS
        if current["config"].get(field) != controlled["config"].get(field)
    ]
    if mismatched:
        raise ValueError("incompatible controlled relaxation runs: " + ",".join(mismatched))
    for policy, run in runs.items():
        config = run["config"]
        if config.get("query_planning_policy") != policy:
            raise ValueError(f"query planning policy mismatch: {policy}")
        if config.get("offset") != offset or config.get("limit") != 20:
            raise ValueError(f"unexpected fixed split: offset={offset},limit=20")
        if config.get("sources") != ["arxiv"]:
            raise ValueError("controlled relaxation comparison requires arxiv only")
        if config.get("query_adapter_policy") != "adaptive":
            raise ValueError("controlled relaxation comparison requires adaptive adapter")
        if (
            config.get("enable_query_evolution")
            or config.get("query_evolution_policy") != "off"
            or config.get("enable_refchain")
        ):
            raise ValueError("controlled relaxation comparison requires later stages off")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError("controlled relaxation comparison requires LLM off")
        if config.get("judgement_policy") != "current_rules":
            raise ValueError("controlled relaxation comparison requires current_rules judgement")
        if config.get("run_profile") != "balanced" or config.get("top_k") != 20:
            raise ValueError("controlled relaxation comparison requires balanced top20")
        if config.get("result_policy") != "highly_and_partial":
            raise ValueError("controlled relaxation comparison result policy mismatch")


def _controlled_summary(
    run: dict[str, Any],
    *,
    split: str,
    policy: str,
) -> dict[str, Any]:
    summary = _summarize_run(run, split=split, policy=policy)
    summary["supplemental_query_contribution"] = _purpose_contribution(run)
    summary["invalid_candidate_ratio"] = _invalid_candidate_ratio(run)
    summary["retrieval_failure_counts"] = _retrieval_failure_counts(run)
    return summary


def _purpose_contribution(run: dict[str, Any]) -> dict[str, dict[str, int]]:
    contributions: dict[str, Counter[str]] = {}
    for result in run["results"]:
        planning = (result.get("stage_diagnostics") or {}).get(
            "initial_query_planning"
        ) or {}
        for row in planning.get("subqueries") or []:
            purpose = str(row.get("purpose") or "unknown")
            if purpose == "original_query":
                continue
            bucket = contributions.setdefault(purpose, Counter())
            bucket["exclusive_candidate_count"] += int(
                row.get("exclusive_candidate_count") or 0
            )
            bucket["post_run_unique_gold_hit_count"] += int(
                row.get("post_run_unique_gold_hit_count") or 0
            )
    return {
        purpose: dict(sorted(values.items()))
        for purpose, values in sorted(contributions.items())
    }


def _invalid_candidate_ratio(run: dict[str, Any]) -> float | None:
    total = invalid = 0
    for result in run["results"]:
        snapshots = (result.get("stage_diagnostics") or {}).get("snapshots") or []
        judged = next(
            (item for item in snapshots if item.get("stage") == "initial_judged"),
            None,
        )
        if judged is None:
            continue
        candidates = judged.get("candidates") or []
        total += len(candidates)
        invalid += sum(
            str(item.get("category") or "") in INVALID_CATEGORIES
            for item in candidates
        )
    return invalid / total if total else None


def _retrieval_failure_counts(run: dict[str, Any]) -> dict[str, int]:
    counts = Counter(
        {
            "query_over_restrictive": 0,
            "query_not_matched": 0,
            "ranking_cutoff": 0,
        }
    )
    for result in run["results"]:
        diagnostics = result.get("stage_diagnostics") or {}
        planning = diagnostics.get("initial_query_planning") or {}
        counts["query_over_restrictive"] += int(
            (planning.get("ineffective_reasons") or {}).get("over_restrictive")
            or 0
        )
        for gold in diagnostics.get("gold_diagnostics") or []:
            reason = gold.get("drop_reason")
            if reason == "not_retrieved":
                counts["query_not_matched"] += 1
            elif reason == "outside_final_top_k":
                counts["ranking_cutoff"] += 1
    return dict(sorted(counts.items()))


def _acceptance(current: dict[str, Any], controlled: dict[str, Any]) -> dict[str, Any]:
    current_api = float(current.get("average_recorded_api_calls") or 0.0)
    controlled_api = float(controlled.get("average_recorded_api_calls") or 0.0)
    current_duplicate = current.get("duplicate_candidate_ratio")
    controlled_duplicate = controlled.get("duplicate_candidate_ratio")
    current_invalid = current.get("invalid_candidate_ratio")
    controlled_invalid = controlled.get("invalid_candidate_ratio")
    checks = {
        "at_least_one_new_unique_gold": (
            int(controlled.get("unique_gold_count") or 0)
            >= int(current.get("unique_gold_count") or 0) + 1
        ),
        "candidate_recall_non_regression": _non_lower(
            controlled.get("candidate_recall"), current.get("candidate_recall")
        ),
        "recall_at_20_non_regression": _non_lower(
            controlled.get("recall_at_20"), current.get("recall_at_20")
        ),
        "f1_at_20_non_regression": _non_lower(
            controlled.get("f1_at_20"), current.get("f1_at_20")
        ),
        "api_calls_within_1_5x": (
            controlled_api == 0.0
            if current_api == 0.0
            else controlled_api <= current_api * 1.5
        ),
        "duplicate_ratio_not_notably_worse": _within_tolerance(
            controlled_duplicate, current_duplicate, 0.05
        ),
        "invalid_candidate_ratio_not_notably_worse": _within_tolerance(
            controlled_invalid, current_invalid, 0.05
        ),
        "frozen_replay_zero_network": all(
            float(row.get(field) or 0.0) == 0.0
            for row in (current, controlled)
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
        "api_call_ratio": _safe_ratio(controlled_api, current_api),
        "unique_gold_gain": (
            int(controlled.get("unique_gold_count") or 0)
            - int(current.get("unique_gold_count") or 0)
        ),
        "ratio_tolerance": 0.05,
        "decision": (
            "controlled_relaxation 可作为默认候选"
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
        "# 受控查询放宽对比",
        "",
        "> 开发集冻结规则后，仅在独立验证集运行一次；gold 只用于事后评估。",
        "",
        "| 切片 | 策略 | 候选 Recall | F1@5 | F1@10 | F1@20 | P@20 | R@20 | MRR | nDCG@20 | 唯一 gold | API/例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
    parser.add_argument("--development-controlled", type=Path, required=True)
    parser.add_argument("--validation-current", type=Path, required=True)
    parser.add_argument("--validation-controlled", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = build_controlled_relaxation_analysis(
        development_current=args.development_current,
        development_controlled=args.development_controlled,
        validation_current=args.validation_current,
        validation_controlled=args.validation_controlled,
        output_dir=args.output_dir,
    )
    print(json.dumps(comparison["validation_acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
