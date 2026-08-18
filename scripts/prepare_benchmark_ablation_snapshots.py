#!/usr/bin/env python3
"""串行执行动态快照的离线规划、受限采集与固定点检查。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.collect_benchmark_snapshot_plan import collect_plan  # noqa: E402
from scripts.run_benchmark import (  # noqa: E402
    BenchmarkRunOptions,
    run_benchmark,
)
from scholar_agent.core.search_schemas import SearchBudget  # noqa: E402
from scholar_agent.evaluation.snapshots import SnapshotStore  # noqa: E402
from scholar_agent.evaluation.snapshots.planning import (  # noqa: E402
    ABLATION_GROUPS,
    SnapshotCollectionLimits,
    mark_group_stop_reason,
    plan_group_root,
    write_coverage_artifacts,
)
from scholar_agent.evaluation.snapshots.schemas import SnapshotPlanRound  # noqa: E402


DYNAMIC_GROUPS = ABLATION_GROUPS[1:]
SUPPORTED_GROUPS = (
    *ABLATION_GROUPS,
    "query_evolution_coverage_gap",
    "query_evolution_coverage_gap_plus_refchain",
    "facet_balanced",
    "facet_balanced_query_evolution_only",
    "facet_balanced_query_evolution_coverage_gap",
    "facet_balanced_refchain_only",
    "facet_balanced_query_evolution_plus_refchain",
    "facet_balanced_query_evolution_coverage_gap_plus_refchain",
    "semantic_seed_expansion",
)


def iterate_group(
    *,
    group: str,
    snapshot_dir: Path | str,
    limits: SnapshotCollectionLimits,
    plan_round: Callable[[str, int], Path],
    collect: Callable[..., dict[str, Any]] = collect_plan,
    retry_failed_snapshots: bool = False,
    clock: Callable[[], float] = time.monotonic,
    cancel_check: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """对单组执行可恢复的 plan→collect→plan，所有限制按组累计。"""

    started = clock()
    total_requests = 0
    total_failures = 0
    rounds: list[dict[str, Any]] = []
    stop_reason: str | None = None
    replay_ready = False
    manifest = SnapshotStore(snapshot_dir).read_manifest()
    prior = manifest.groups.get(group)
    first_round = (prior.last_plan_round if prior else 0) + 1

    for offset in range(limits.max_plan_rounds):
        round_index = first_round + offset
        if cancel_check():
            stop_reason = "snapshot_collection_cancelled"
            break
        if clock() - started >= limits.max_collection_seconds:
            stop_reason = "snapshot_collection_time_limit"
            break
        plan_path = plan_round(group, round_index)
        plan = SnapshotPlanRound.model_validate_json(
            plan_path.read_text(encoding="utf-8")
        )
        missing_count = plan.missing_retrieval_count + plan.missing_reference_count
        round_result: dict[str, Any] = {
            "round_index": round_index,
            "plan_path": str(plan_path),
            "missing_key_count": missing_count,
            "network_request_count_during_plan": plan.network_request_count,
        }
        if plan.network_request_count != 0:
            raise RuntimeError("snapshot_plan_network_request_detected")
        if missing_count == 0:
            replay_ready = True
            rounds.append(round_result)
            break

        remaining_requests = limits.max_new_requests - total_requests
        remaining_failures = limits.max_new_failed_entries - total_failures
        remaining_seconds = limits.max_collection_seconds - (clock() - started)
        if remaining_requests <= 0:
            stop_reason = "snapshot_collection_request_limit"
            rounds.append(round_result)
            break
        if remaining_failures <= 0:
            stop_reason = "snapshot_collection_failure_limit"
            rounds.append(round_result)
            break
        if remaining_seconds <= 0:
            stop_reason = "snapshot_collection_time_limit"
            rounds.append(round_result)
            break

        collection = collect(
            plan_path,
            snapshot_dir,
            max_new_requests=remaining_requests,
            max_new_failed_entries=remaining_failures,
            max_collection_seconds=remaining_seconds,
            retry_failed_snapshots=retry_failed_snapshots,
            source_failure_limit=limits.source_failure_limit,
            cancel_check=cancel_check,
        )
        total_requests += int(collection.get("request_count") or 0)
        total_failures += int(collection.get("failed_entry_count") or 0)
        round_result["collection"] = collection
        rounds.append(round_result)
        if collection.get("stop_reason"):
            stop_reason = str(collection["stop_reason"])
            break
    else:
        stop_reason = "snapshot_plan_not_converged"

    if not replay_ready and stop_reason is None:
        stop_reason = "snapshot_plan_not_converged"
    if stop_reason:
        mark_group_stop_reason(
            snapshot_dir,
            group=group,
            stop_reason=stop_reason,
        )
    last_round = rounds[-1]["round_index"] if rounds else 0
    if last_round:
        write_coverage_artifacts(
            snapshot_dir,
            group=group,
            round_index=last_round,
        )
    return {
        "group": group,
        "replay_ready": replay_ready,
        "stop_reason": stop_reason,
        "plan_rounds": len(rounds),
        "new_request_count": total_requests,
        "new_failed_entry_count": total_failures,
        "elapsed_seconds": clock() - started,
        "rounds": rounds,
    }


def prepare_groups(
    *,
    snapshot_dir: Path | str,
    output_root: Path | str,
    run_id_prefix: str,
    groups: list[str],
    limits: SnapshotCollectionLimits,
    retry_failed_snapshots: bool = False,
    cancel_check: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """按固定消融配置串行准备各组，不并行访问公共来源。"""

    root = Path(snapshot_dir).expanduser().resolve()
    store = SnapshotStore(root)
    manifest = store.read_manifest()
    budgets = SearchBudget.model_validate(manifest.budgets)
    output = Path(output_root).expanduser().resolve()
    group_results: list[dict[str, Any]] = []
    started = time.monotonic()
    total_requests = 0
    total_failures = 0

    def planner(group: str, round_index: int) -> Path:
        query_evolution = group in {
            "query_evolution_only",
            "query_evolution_plus_refchain",
            "query_evolution_coverage_gap",
            "query_evolution_coverage_gap_plus_refchain",
            "facet_balanced_query_evolution_only",
            "facet_balanced_query_evolution_coverage_gap",
            "facet_balanced_query_evolution_plus_refchain",
            "facet_balanced_query_evolution_coverage_gap_plus_refchain",
        }
        refchain = group in {
            "refchain_only",
            "query_evolution_plus_refchain",
            "query_evolution_coverage_gap_plus_refchain",
            "facet_balanced_refchain_only",
            "facet_balanced_query_evolution_plus_refchain",
            "facet_balanced_query_evolution_coverage_gap_plus_refchain",
        }
        semantic_seed_expansion = group == "semantic_seed_expansion"
        query_evolution_policy = (
            "coverage_gap"
            if "coverage_gap" in group
            else "seed_expansion"
        )
        query_planning_policy = (
            "facet_balanced"
            if group.startswith("facet_balanced")
            else "current_rules"
        )
        run_id = _available_run_id(
            output,
            f"{run_id_prefix}_plan_{group}_r{round_index}",
        )
        run_benchmark(
            BenchmarkRunOptions(
                dataset=manifest.dataset,
                dataset_split=manifest.split,
                offset=manifest.offset,
                limit=manifest.limit,
                output_root=output,
                run_id=run_id,
                run_profile=manifest.run_profile,  # type: ignore[arg-type]
                sources=list(manifest.sources),
                result_policy="highly_and_partial",
                top_k=20,
                enable_query_evolution=query_evolution,
                query_evolution_policy=query_evolution_policy,
                query_planning_policy=query_planning_policy,
                enable_refchain=refchain,
                enable_semantic_seed_expansion=semantic_seed_expansion,
                enable_llm_query_understanding=False,
                enable_llm_judgement=False,
                max_workers=1,
                budgets=budgets,
                diagnostics=False,
                query_adapter_policy=manifest.adapter_policy,  # type: ignore[arg-type]
                retrieval_mode="plan",
                snapshot_dir=root,
                plan_round=round_index,
            )
        )
        return plan_group_root(root, group) / f"plan_round_{round_index}.json"

    for group in groups:
        if cancel_check():
            break
        remaining = SnapshotCollectionLimits(
            max_plan_rounds=limits.max_plan_rounds,
            max_new_requests=max(0, limits.max_new_requests - total_requests),
            max_new_failed_entries=max(
                0,
                limits.max_new_failed_entries - total_failures,
            ),
            max_collection_seconds=max(
                0.0,
                limits.max_collection_seconds - (time.monotonic() - started),
            ),
            source_failure_limit=limits.source_failure_limit,
        )
        group_results.append(
            iterate_group(
                group=group,
                snapshot_dir=root,
                limits=remaining,
                plan_round=planner,
                retry_failed_snapshots=retry_failed_snapshots,
                cancel_check=cancel_check,
            )
        )
        total_requests += int(group_results[-1].get("new_request_count") or 0)
        total_failures += int(
            group_results[-1].get("new_failed_entry_count") or 0
        )
        if group_results[-1].get("stop_reason") in {
            "snapshot_collection_cancelled",
            "snapshot_collection_request_limit",
            "snapshot_collection_failure_limit",
            "snapshot_collection_time_limit",
            "snapshot_collection_source_cooldown",
        }:
            break
    return {
        "snapshot_name": root.name,
        "groups": group_results,
        "new_request_count": total_requests,
        "new_failed_entry_count": total_failures,
        "coverage": store.inspect().get("groups") or {},
    }


def _available_run_id(output_root: Path, base: str) -> str:
    if not (output_root / base).exists():
        return base
    suffix = 2
    while (output_root / f"{base}_{suffix}").exists():
        suffix += 1
    return f"{base}_{suffix}"


def _parse_groups(value: str) -> list[str]:
    groups: list[str] = []
    for raw in value.split(","):
        group = raw.strip()
        if not group or group in groups:
            continue
        if group not in SUPPORTED_GROUPS:
            raise ValueError(f"unsupported snapshot group: {group}")
        groups.append(group)
    if not groups:
        raise ValueError("at least one snapshot group is required")
    return groups


def _parser() -> argparse.ArgumentParser:
    defaults = SnapshotCollectionLimits()
    parser = argparse.ArgumentParser(description="迭代补齐四组消融动态快照。")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-root", default="outputs/benchmark_runs")
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--groups", default=",".join(DYNAMIC_GROUPS))
    parser.add_argument("--max-plan-rounds", type=int, default=defaults.max_plan_rounds)
    parser.add_argument("--max-new-requests", type=int, default=defaults.max_new_requests)
    parser.add_argument(
        "--max-new-failed-entries",
        type=int,
        default=defaults.max_new_failed_entries,
    )
    parser.add_argument(
        "--max-collection-seconds",
        type=float,
        default=defaults.max_collection_seconds,
    )
    parser.add_argument(
        "--source-failure-limit",
        type=int,
        default=defaults.source_failure_limit,
    )
    parser.add_argument("--retry-failed-snapshots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        limits = SnapshotCollectionLimits(
            max_plan_rounds=args.max_plan_rounds,
            max_new_requests=args.max_new_requests,
            max_new_failed_entries=args.max_new_failed_entries,
            max_collection_seconds=args.max_collection_seconds,
            source_failure_limit=args.source_failure_limit,
        )
        result = prepare_groups(
            snapshot_dir=args.snapshot_dir,
            output_root=args.output_root,
            run_id_prefix=args.run_id_prefix,
            groups=_parse_groups(args.groups),
            limits=limits,
            retry_failed_snapshots=args.retry_failed_snapshots,
        )
    except KeyboardInterrupt:
        print("snapshot_collection_cancelled", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(item["replay_ready"] for item in result["groups"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
