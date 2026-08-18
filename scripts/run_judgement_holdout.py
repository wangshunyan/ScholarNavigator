#!/usr/bin/env python3
"""在固定 holdout30 Retrieval Snapshot 上比较两种规则 Judgement。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_benchmark import BenchmarkRunOptions, run_benchmark  # noqa: E402
from scholar_agent.core.search_schemas import (  # noqa: E402
    JudgementPolicy,
    SearchBudget,
)
from scholar_agent.evaluation.holdout_comparison import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    HOLDOUT_LIMIT,
    HOLDOUT_OFFSET,
    analyze_holdout_runs,
)


def holdout_options(
    *,
    policy: JudgementPolicy,
    snapshot_dir: Path,
    output_dir: Path,
    resume: bool,
) -> BenchmarkRunOptions:
    """返回不可由 CLI 改写案例范围和搜索算法的固定 replay 配置。"""

    return BenchmarkRunOptions(
        dataset="auto_scholar_query",
        dataset_split="holdout",
        offset=HOLDOUT_OFFSET,
        limit=HOLDOUT_LIMIT,
        output_root=output_dir,
        run_id="H-current" if policy == "current_rules" else "H-calibrated",
        run_profile="balanced",
        sources=["arxiv"],
        result_policy="highly_and_partial",
        top_k=20,
        enable_query_evolution=False,
        query_evolution_policy="off",
        query_planning_policy="current_rules",
        judgement_policy=policy,
        enable_refchain=False,
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        max_workers=1,
        budgets=SearchBudget(
            max_search_rounds=2,
            max_candidate_papers=150,
            max_llm_calls=20,
            max_total_tokens=50000,
            max_latency_seconds=120.0,
        ),
        diagnostics=True,
        query_adapter_policy="adaptive",
        retrieval_mode="replay",
        snapshot_dir=snapshot_dir,
        resume=resume,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="固定 offset=20/limit=30，离线比较两种 Judgement policy。"
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument(
        "--output",
        default="outputs/benchmark_runs/holdout30_baseline",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="只重建报告；要求两组固定 replay 结果已经存在。",
    )
    args = parser.parse_args(argv)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.analyze_only:
            for policy in ("current_rules", "calibrated_rules_v1"):
                run_dir = output_dir / (
                    "H-current" if policy == "current_rules" else "H-calibrated"
                )
                run_benchmark(
                    holdout_options(
                        policy=policy,
                        snapshot_dir=snapshot_dir,
                        output_dir=output_dir,
                        resume=args.resume or run_dir.exists(),
                    )
                )
        comparison, per_query, errors, coverage = analyze_holdout_runs(
            output_dir / "H-current",
            output_dir / "H-calibrated",
            snapshot_dir,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        _atomic_write_json(output_dir / "comparison.json", comparison)
        _atomic_write_text(
            output_dir / "comparison.md",
            _comparison_markdown(comparison),
        )
        _atomic_write_jsonl(output_dir / "per_query_diagnostics.jsonl", per_query)
        _atomic_write_json(output_dir / "error_summary.json", errors)
        _atomic_write_json(output_dir / "snapshot_coverage.json", coverage)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output_dir)
    return 0


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    current = comparison["current_rules"]
    calibrated = comparison["calibrated_rules_v1"]
    fields = (
        "candidate_recall",
        "f1_at_5",
        "f1_at_10",
        "f1_at_20",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr",
        "ndcg_at_20",
        "gold_judgement_false_negative_rate",
        "average_returned_paper_count",
    )
    lines = [
        "# holdout30 Judgement 对比",
        "",
        "> 固定 AutoScholarQuery offset=20、limit=30；只作保留集诊断，不重新校准参数或作夸张显著性声明。",
        "",
        "| 指标 | current_rules | calibrated_rules_v1 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        lines.append(
            f"| {field} | {current[field]:.6f} | {calibrated[field]:.6f} "
            f"| {comparison['deltas'][field]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 配对 bootstrap",
            "",
            "| 指标 | 平均差 | 95% 区间 | 正差占比 |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, values in comparison["paired_bootstrap"]["metrics"].items():
        lines.append(
            f"| {metric} | {values['mean_difference']:.6f} "
            f"| [{values['ci_95_low']:.6f}, {values['ci_95_high']:.6f}] "
            f"| {values['bootstrap_positive_share']:.4f} |"
        )
    conclusion = comparison["conclusion"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 稳定优势：`{str(conclusion['calibrated_has_stable_advantage']).lower()}`",
            f"- 产品默认：`{conclusion['product_default']}`",
            f"- Retrieval 召回瓶颈：`{str(conclusion['retrieval_recall_bottleneck']).lower()}`",
            "- 分组仅使用查询规则分面和长度，不使用 gold 定义。",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
