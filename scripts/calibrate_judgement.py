#!/usr/bin/env python3
"""在冻结 Retrieval Snapshot 候选上校准确定性 Judgement。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_benchmark import _runtime_code_hash  # noqa: E402
from scholar_agent.agents.judgement_config import (  # noqa: E402
    CURRENT_RULES_CONFIG,
    judgement_config_hash,
)
from scholar_agent.core.search_schemas import QUERY_PLANNER_VERSION  # noqa: E402
from scholar_agent.evaluation.datasets import (  # noqa: E402
    dataset_source_path,
    load_dataset,
)
from scholar_agent.evaluation.judgement_calibration import (  # noqa: E402
    CALIBRATION_GRID_VERSION,
    SELECTION_OBJECTIVE,
    CalibrationEvaluation,
    FrozenJudgementCase,
    config_distance,
    evaluate_frozen_cases,
    judgement_parameter_grid,
    parameter_grid_hash,
    select_best_evaluation,
    threshold_sensitivity,
    validation_acceptance,
)
from scholar_agent.evaluation.snapshots import (  # noqa: E402
    SnapshotAwareReferenceFetcher,
    SnapshotAwareRetriever,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.prompts import load_manifest, load_prompt  # noqa: E402
from scholar_agent.services.search_service import SearchService  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在冻结候选上执行小型、确定性的 Judgement 校准。"
    )
    parser.add_argument("--dataset", default="auto_scholar_query")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--validation-offset", type=int, default=10)
    parser.add_argument("--validation-limit", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=["replay"], default="replay")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--validation-snapshot-dir", default=None)
    parser.add_argument(
        "--output",
        default="outputs/benchmark_runs/judgement_calibration",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"calibration output already exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "calibration_manifest.json"
    if args.resume and manifest_path.is_file():
        stored_manifest = _read_json(manifest_path)
        if stored_manifest.get("validation_evaluated") and not args.development_only:
            return 0

    source_path = dataset_source_path(
        args.dataset,
        Path(args.dataset_path) if args.dataset_path else None,
    )
    queries = load_dataset(args.dataset, path=source_path)
    development_queries = queries[args.offset : args.offset + args.limit]
    if not development_queries:
        raise ValueError("development selection is empty")
    development_snapshot = Path(args.snapshot_dir).expanduser().resolve()
    development_cases = _collect_frozen_cases(
        development_queries,
        snapshot_dir=development_snapshot,
    )

    baseline = evaluate_frozen_cases(
        development_cases,
        CURRENT_RULES_CONFIG,
        policy="current_rules",
        include_diagnostics=True,
    )
    _atomic_write_jsonl(
        output / "baseline_candidate_diagnostics.jsonl",
        baseline.candidate_diagnostics,
    )
    _atomic_write_text(
        output / "baseline_summary.md",
        _evaluation_markdown("开发集 current_rules 审计", baseline),
    )

    grid = judgement_parameter_grid()
    if not 50 <= len(grid) <= 300:
        raise ValueError(f"calibration grid size out of bounds: {len(grid)}")
    grid_hash = parameter_grid_hash(grid)
    grid_path = output / "development_grid_results.jsonl"
    existing = _load_grid_results(grid_path) if args.resume else {}
    evaluations: list[CalibrationEvaluation] = []
    for config in grid:
        key = judgement_config_hash(config)
        evaluation = existing.get(key)
        if evaluation is None:
            evaluation = evaluate_frozen_cases(
                development_cases,
                config,
                policy="calibrated_rules_v1",
                include_diagnostics=False,
            )
            existing[key] = evaluation
            _write_grid_results(grid_path, existing)
        evaluations.append(evaluation)

    chosen_grid_evaluation = select_best_evaluation(evaluations)
    selected_config = chosen_grid_evaluation.config.model_copy(
        update={"config_version": "calibrated-rules-v1"}
    )
    selected = evaluate_frozen_cases(
        development_cases,
        selected_config,
        policy="calibrated_rules_v1",
        include_diagnostics=True,
    )
    selected_path = output / "selected_config.json"
    if selected_path.exists():
        frozen = _read_json(selected_path)
        if frozen != selected_config.model_dump(mode="json"):
            raise ValueError("selected_config_is_frozen")
    else:
        _atomic_write_json(selected_path, selected_config.model_dump(mode="json"))

    _atomic_write_jsonl(
        output / "development_candidate_diagnostics.jsonl",
        selected.candidate_diagnostics,
    )
    _atomic_write_text(
        output / "development_comparison.md",
        _comparison_markdown("开发集", baseline, selected),
    )
    _atomic_write_json(
        output / "threshold_sensitivity.json",
        threshold_sensitivity(evaluations),
    )
    manifest = {
        "schema_version": "1",
        "calibration_grid_version": CALIBRATION_GRID_VERSION,
        "dataset": args.dataset,
        "dataset_path": str(source_path),
        "development": {"offset": args.offset, "limit": args.limit},
        "validation": {
            "offset": args.validation_offset,
            "limit": args.validation_limit,
        },
        "retrieval_snapshot_hash": _file_hash(
            development_snapshot / "manifest.json"
        ),
        "baseline_config_hash": judgement_config_hash(CURRENT_RULES_CONFIG),
        "parameter_grid_hash": grid_hash,
        "parameter_combination_count": len(grid),
        "selection_objective": SELECTION_OBJECTIVE,
        "selected_config_hash": judgement_config_hash(selected_config),
        "selected_grid_config_hash": chosen_grid_evaluation.config_hash,
        "selected_parameter_distance": config_distance(selected_config),
        "code_hash": _runtime_code_hash(),
        "prompt_hashes": _prompt_hashes(),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "validation_evaluated": False,
        "small_sample_diagnostic_only": True,
    }
    if manifest_path.exists():
        frozen_manifest = _read_json(manifest_path)
        immutable = (
            "dataset",
            "development",
            "validation",
            "retrieval_snapshot_hash",
            "baseline_config_hash",
            "parameter_grid_hash",
            "selection_objective",
            "selected_config_hash",
        )
        if any(frozen_manifest.get(key) != manifest.get(key) for key in immutable):
            raise ValueError("calibration_manifest_is_frozen")
        manifest = frozen_manifest
    else:
        _atomic_write_json(manifest_path, manifest)

    if args.development_only:
        _write_comparison_json(output, baseline, selected, None, None, None)
        return 0

    validation_snapshot = Path(
        args.validation_snapshot_dir or args.snapshot_dir
    ).expanduser().resolve()
    validation_queries = queries[
        args.validation_offset : args.validation_offset + args.validation_limit
    ]
    if not validation_queries:
        raise ValueError("validation selection is empty")
    validation_cases = _collect_frozen_cases(
        validation_queries,
        snapshot_dir=validation_snapshot,
    )
    validation_baseline = evaluate_frozen_cases(
        validation_cases,
        CURRENT_RULES_CONFIG,
        policy="current_rules",
        include_diagnostics=True,
    )
    validation_selected = evaluate_frozen_cases(
        validation_cases,
        selected_config,
        policy="calibrated_rules_v1",
        include_diagnostics=True,
    )
    acceptance = validation_acceptance(validation_baseline, validation_selected)
    _atomic_write_jsonl(
        output / "validation_candidate_diagnostics.jsonl",
        validation_selected.candidate_diagnostics,
    )
    _atomic_write_text(
        output / "validation_comparison.md",
        _comparison_markdown(
            "独立验证集",
            validation_baseline,
            validation_selected,
            acceptance=acceptance,
        ),
    )
    manifest.update(
        {
            "validation_snapshot_hash": _file_hash(
                validation_snapshot / "manifest.json"
            ),
            "validation_evaluated": True,
            "validation_evaluated_at": datetime.now(timezone.utc).isoformat(),
            "validation_code_hash": _runtime_code_hash(),
            "acceptance": acceptance,
            "product_default": "current_rules",
        }
    )
    _atomic_write_json(manifest_path, manifest)
    _write_comparison_json(
        output,
        baseline,
        selected,
        validation_baseline,
        validation_selected,
        acceptance,
    )
    return 0


def _collect_frozen_cases(
    queries: list[Any],
    *,
    snapshot_dir: Path,
) -> list[FrozenJudgementCase]:
    store = SnapshotStore(snapshot_dir)
    coverage = store.inspect().get("groups", {}).get("baseline", {})
    if not coverage.get("replay_ready"):
        raise ValueError("snapshot_group_not_replay_ready:baseline")
    runtime = SnapshotRuntime(
        store,
        mode="replay",
        group_name="baseline",
        query_evolution_policy="off",
        query_planning_policy="current_rules",
        query_planner_version=QUERY_PLANNER_VERSION,
        judgement_policy="current_rules",
        judgement_config_hash=judgement_config_hash(CURRENT_RULES_CONFIG),
    )

    def forbidden_reference(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reference network forbidden during calibration replay")

    service = SearchService(
        retriever=SnapshotAwareRetriever(runtime),
        reference_fetcher=forbidden_reference,
        max_workers=1,
        judgement_policy="current_rules",
        judgement_config=CURRENT_RULES_CONFIG,
    )
    cases: list[FrozenJudgementCase] = []
    for query in queries:
        runtime.begin_case(query.query_id)
        output = service.run_search(
            query.query,
            top_k=20,
            run_profile="balanced",
            enable_query_evolution=False,
            query_evolution_policy="off",
            query_planning_policy="current_rules",
            judgement_policy="current_rules",
            enable_refchain=False,
            enable_synthesis=False,
            enable_llm_query_understanding=False,
            enable_llm_judgement=False,
            sources_override=["arxiv"],
            collect_diagnostics=False,
            query_adapter_policy="adaptive",
        )
        runtime.assert_case_complete()
        cost = runtime.finish_case()
        if (
            cost.replay_execution_request_count
            or cost.replay_execution_retry_count
            or cost.replay_execution_network_wait_seconds
        ):
            raise AssertionError("calibration replay executed network work")
        cases.append(
            FrozenJudgementCase(
                case_id=query.query_id,
                query=query.query,
                query_analysis=output.search_plan.query_analysis,
                papers=[item.paper for item in output.judgements],
                gold_papers=query.gold_papers,
                replay_cost=cost.model_dump(mode="json"),
            )
        )
    return cases


def _write_comparison_json(
    output: Path,
    development_baseline: CalibrationEvaluation,
    development_calibrated: CalibrationEvaluation,
    validation_baseline: CalibrationEvaluation | None,
    validation_calibrated: CalibrationEvaluation | None,
    acceptance: dict[str, Any] | None,
) -> None:
    payload = {
        "scope": "small_sample_diagnostic_only",
        "development": {
            "current_rules": development_baseline.metrics,
            "calibrated_rules_v1": development_calibrated.metrics,
        },
        "validation": (
            {
                "current_rules": validation_baseline.metrics,
                "calibrated_rules_v1": validation_calibrated.metrics,
            }
            if validation_baseline is not None and validation_calibrated is not None
            else None
        ),
        "acceptance": acceptance,
        "product_default": "current_rules",
        "limitations": [
            "AutoScholarQuery 非 gold 候选不等于真实负例。",
            "开发集 10 条、验证集 5 条，仅用于小样本诊断，不作显著性声明。",
        ],
    }
    _atomic_write_json(output / "comparison.json", payload)
    lines = [
        "# Judgement 校准汇总",
        "",
        "- 范围：`small_sample_diagnostic_only`",
        "- 产品默认：`current_rules`",
        "- 非 gold 候选不等于真实负例。",
        "",
        _comparison_markdown(
            "开发集",
            development_baseline,
            development_calibrated,
        ),
    ]
    if validation_baseline is not None and validation_calibrated is not None:
        lines.extend(
            [
                "",
                _comparison_markdown(
                    "独立验证集",
                    validation_baseline,
                    validation_calibrated,
                    acceptance=acceptance,
                ),
            ]
        )
    _atomic_write_text(output / "summary.md", "\n".join(lines).strip() + "\n")


def _evaluation_markdown(
    title: str,
    evaluation: CalibrationEvaluation,
) -> str:
    metrics = evaluation.metrics
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- 配置：`{evaluation.config_hash}`",
            f"- Candidate Recall：{metrics['candidate_recall']:.6f}",
            f"- F1@20：{metrics['f1_at_20']:.6f}",
            f"- Precision@20：{metrics['precision_at_20']:.6f}",
            f"- Recall@20：{metrics['recall_at_20']:.6f}",
            "- Benchmark 非 gold 候选不等于真实负例。",
        ]
    ) + "\n"


def _comparison_markdown(
    title: str,
    baseline: CalibrationEvaluation,
    calibrated: CalibrationEvaluation,
    *,
    acceptance: dict[str, Any] | None = None,
) -> str:
    names = (
        "candidate_recall",
        "f1_at_20",
        "precision_at_20",
        "recall_at_20",
        "mrr",
        "ndcg_at_20",
        "gold_judgement_false_negative_rate",
        "average_returned_paper_count",
    )
    lines = [
        f"## {title}",
        "",
        "| 指标 | current_rules | calibrated_rules_v1 |",
        "|---|---:|---:|",
    ]
    for name in names:
        lines.append(
            f"| {name} | {_format_metric(baseline.metrics.get(name))} "
            f"| {_format_metric(calibrated.metrics.get(name))} |"
        )
    if acceptance is not None:
        lines.extend(
            [
                "",
                f"- 验收：`{acceptance['status']}`",
                "- 样本限制：`small_sample_diagnostic_only`",
            ]
        )
    return "\n".join(lines)


def _load_grid_results(path: Path) -> dict[str, CalibrationEvaluation]:
    if not path.is_file():
        return {}
    results: dict[str, CalibrationEvaluation] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        evaluation = CalibrationEvaluation.model_validate(payload)
        results[evaluation.config_hash] = evaluation
    return results


def _write_grid_results(
    path: Path,
    results: dict[str, CalibrationEvaluation],
) -> None:
    _atomic_write_jsonl(
        path,
        [
            results[key].model_dump(mode="json", exclude={"candidate_diagnostics"})
            for key in sorted(results)
        ],
    )


def _prompt_hashes() -> dict[str, str]:
    return {
        name: load_prompt(name).content_hash
        for name, entry in load_manifest().items()
        if entry.runtime_enabled
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _format_metric(value: Any) -> str:
    return "null" if value is None else f"{float(value):.6f}"


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
