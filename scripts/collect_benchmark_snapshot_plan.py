#!/usr/bin/env python3
"""串行消费离线快照计划，并在保守上限内补齐缺失条目。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.connectors import (  # noqa: E402
    fetch_openalex_references_detailed,
    recommend_semantic_scholar_papers_detailed,
    resolve_semantic_scholar_paper_ids_detailed,
    search_arxiv_detailed,
    search_openalex_detailed,
    search_pubmed_detailed,
    search_semantic_scholar_detailed,
)
from scholar_agent.connectors.schemas import ConnectorSearchResult  # noqa: E402
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers  # noqa: E402
from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.snapshots import SnapshotRuntime, SnapshotStore  # noqa: E402
from scholar_agent.evaluation.llm_planning_snapshots import (  # noqa: E402
    LLMPlanningSnapshotStore,
    collect_llm_plan_entry,
)
from scholar_agent.llm.provider import OpenAICompatibleLLMClient  # noqa: E402
from scholar_agent.evaluation.snapshots.planning import (  # noqa: E402
    SnapshotCollectionLimits,
    atomic_write_json,
    plan_round_root,
    write_coverage_artifacts,
)
from scholar_agent.evaluation.snapshots.schemas import (  # noqa: E402
    SnapshotPlanEntry,
    SnapshotPlanRound,
)
from scholar_agent.evaluation.snapshots.store import SnapshotMissingError  # noqa: E402


DEFAULT_LIMITS = SnapshotCollectionLimits()
DEFAULT_MAX_NEW_REQUESTS = DEFAULT_LIMITS.max_new_requests
DEFAULT_MAX_NEW_FAILED_ENTRIES = DEFAULT_LIMITS.max_new_failed_entries
DEFAULT_MAX_COLLECTION_SECONDS = DEFAULT_LIMITS.max_collection_seconds
DEFAULT_SOURCE_FAILURE_LIMIT = DEFAULT_LIMITS.source_failure_limit


def collect_plan(
    plan_path: Path | str,
    snapshot_dir: Path | str,
    *,
    llm_snapshot_dir: Path | str | None = None,
    llm_client: Any | None = None,
    max_new_requests: int = DEFAULT_MAX_NEW_REQUESTS,
    max_new_failed_entries: int = DEFAULT_MAX_NEW_FAILED_ENTRIES,
    max_collection_seconds: float = DEFAULT_MAX_COLLECTION_SECONDS,
    retry_failed_snapshots: bool = False,
    source_failure_limit: int = DEFAULT_SOURCE_FAILURE_LIMIT,
    openalex_max_retries: int | None = None,
    searchers: dict[str, Callable[[str, int], ConnectorSearchResult]] | None = None,
    reference_fetcher: Callable[[Paper, int], ConnectorSearchResult] = (
        fetch_openalex_references_detailed
    ),
    recommendation_fetcher: Callable[[list[str], int], ConnectorSearchResult] = (
        recommend_semantic_scholar_papers_detailed
    ),
    semantic_seed_resolver: Callable[[list[str]], ConnectorSearchResult] = (
        resolve_semantic_scholar_paper_ids_detailed
    ),
    clock: Callable[[], float] = time.monotonic,
    cancel_check: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    path = Path(plan_path).expanduser().resolve()
    plan = SnapshotPlanRound.model_validate_json(path.read_text(encoding="utf-8"))
    store = SnapshotStore(snapshot_dir)
    llm_store = LLMPlanningSnapshotStore(llm_snapshot_dir or snapshot_dir)
    query_evolution_policy = _plan_query_evolution_policy(plan, store)
    query_planning_policy, query_planner_version = _plan_query_planning(plan, store)
    runtime = SnapshotRuntime(
        store,
        mode="record-missing",
        group_name=plan.group,
        retry_failed_snapshots=retry_failed_snapshots,
        plan_round=plan.round_index,
        query_evolution_policy=query_evolution_policy,
        query_planning_policy=query_planning_policy,
        query_planner_version=query_planner_version,
    )
    if openalex_max_retries is not None and openalex_max_retries < 0:
        raise ValueError("openalex_max_retries_must_be_nonnegative")
    registry = searchers
    if registry is None:
        openalex_search = (
            partial(search_openalex_detailed, max_retries=openalex_max_retries)
            if openalex_max_retries is not None
            else search_openalex_detailed
        )
        registry = {
            "arxiv": search_arxiv_detailed,
            "openalex": openalex_search,
            "semantic_scholar": search_semantic_scholar_detailed,
            "pubmed": search_pubmed_detailed,
        }
    started = clock()
    request_count = 0
    failed_count = 0
    collected_count = 0
    skipped_present_count = 0
    source_failures: dict[str, int] = {}
    blocked_sources: set[str] = set()
    completed_keys = _prior_completed_keys(
        plan_round_root(snapshot_dir, plan.group, plan.round_index)
        / "collection_result.json"
    )
    stop_reason: str | None = None
    round_root = plan_round_root(snapshot_dir, plan.group, plan.round_index)
    result_path = round_root / "collection_result.json"
    entries = sorted(plan.entries, key=lambda entry: (entry.priority, entry.key))
    runtime.begin_case(f"collection:{plan.group}:{plan.round_index}")

    for entry in entries:
        if cancel_check():
            stop_reason = "snapshot_collection_cancelled"
            break
        existing_status = _existing_status(store, llm_store, entry)
        if existing_status == "success" or (
            existing_status == "failed" and not retry_failed_snapshots
        ):
            skipped_present_count += 1
            continue
        if entry.source in blocked_sources:
            continue
        if request_count >= max_new_requests:
            stop_reason = "snapshot_collection_request_limit"
            break
        if failed_count >= max_new_failed_entries:
            stop_reason = "snapshot_collection_failure_limit"
            break
        if clock() - started >= max_collection_seconds:
            stop_reason = "snapshot_collection_time_limit"
            break

        if entry.entry_type == "llm_planning":
            planning_client = llm_client or OpenAICompatibleLLMClient.from_env()
            execution = collect_llm_plan_entry(entry, llm_store, planning_client)
            request_count += int(execution.llm_call_attempted)
            error_message = None
        else:
            result = _collect_entry(
                runtime,
                entry,
                registry,
                reference_fetcher,
                recommendation_fetcher,
                semantic_seed_resolver,
            )
            diagnostics = result.recorded_diagnostics or result.diagnostics
            request_count += diagnostics.request_count
            error_message = result.error_message
        collected_count += 1
        if entry.key not in completed_keys:
            completed_keys.append(entry.key)
        if error_message:
            failed_count += 1
            source_failures[entry.source] = source_failures.get(entry.source, 0) + 1
            if source_failures[entry.source] >= source_failure_limit:
                blocked_sources.add(entry.source)
        atomic_write_json(
            result_path,
            _collection_result(
                plan,
                request_count=request_count,
                failed_count=failed_count,
                collected_count=collected_count,
                skipped_present_count=skipped_present_count,
                completed_keys=completed_keys,
                blocked_sources=blocked_sources,
                source_failures=source_failures,
                stop_reason=stop_reason,
                elapsed_seconds=clock() - started,
            ),
        )

    remaining_sources = {
        entry.source
        for entry in entries
        if _existing_status(store, llm_store, entry) is None
    }
    if stop_reason is None and remaining_sources.intersection(blocked_sources):
        stop_reason = "snapshot_collection_source_cooldown"
    observation = runtime.finish_group(
        completed=stop_reason is None,
        stop_reason=stop_reason,
    )
    result = _collection_result(
        plan,
        request_count=request_count,
        failed_count=failed_count,
        collected_count=collected_count,
        skipped_present_count=skipped_present_count,
        completed_keys=completed_keys,
        blocked_sources=blocked_sources,
        source_failures=source_failures,
        stop_reason=stop_reason,
        elapsed_seconds=clock() - started,
    )
    result["coverage"] = observation.model_dump(mode="json")
    result["covered_success"] = observation.success_key_count
    result["covered_failed"] = observation.failed_key_count
    result["missing_entries"] = observation.missing_key_count
    atomic_write_json(result_path, result)
    write_coverage_artifacts(
        snapshot_dir,
        group=plan.group,
        round_index=plan.round_index,
    )
    return result


def _collect_entry(
    runtime: SnapshotRuntime,
    entry: SnapshotPlanEntry,
    registry: dict[str, Callable[[str, int], ConnectorSearchResult]],
    reference_fetcher: Callable[[Paper, int], ConnectorSearchResult],
    recommendation_fetcher: Callable[[list[str], int], ConnectorSearchResult],
    semantic_seed_resolver: Callable[[list[str]], ConnectorSearchResult],
) -> ConnectorSearchResult:
    if entry.entry_type == "retrieval":
        search = registry.get(entry.source)
        if search is None or entry.adapted_query is None or entry.adapter_policy is None:
            raise ValueError(f"snapshot_plan_entry_invalid:{entry.key}")
        return runtime.search(
            entry.source,
            entry.adapted_query,
            entry.limit,
            entry.adapter_policy,  # type: ignore[arg-type]
            search,
            stage=entry.stage,
            origin_subquery=entry.origin_subquery,
            generated_by=entry.generated_by,
            query_evolution_policy=entry.query_evolution_policy,
            query_planning_policy=entry.query_planning_policy,
            query_planner_version=entry.query_planner_version,
        )
    if entry.entry_type != "reference" or entry.seed_identifier is None:
        raise ValueError(f"snapshot_plan_seed_missing:{entry.key}")
    if entry.generated_by == "semantic_seed_resolution":
        if not entry.seed_identifiers:
            raise ValueError(f"snapshot_plan_seed_missing:{entry.key}")
        return runtime.resolve_semantic_seed_ids(
            entry.seed_identifiers,
            semantic_seed_resolver,
        )
    if entry.generated_by == "semantic_seed_expansion":
        if not entry.seed_identifiers:
            raise ValueError(f"snapshot_plan_seed_missing:{entry.key}")
        return runtime.fetch_recommendations(
            entry.seed_identifiers,
            entry.limit,
            recommendation_fetcher,
        )
    return runtime.fetch_references(
        _seed_paper(entry.seed_identifier),
        entry.limit,
        reference_fetcher,
    )


def _seed_paper(identifier: str) -> Paper:
    prefix, _, value = identifier.partition(":")
    identifiers = (
        PaperIdentifiers(openalex_id=value)
        if prefix == "openalex"
        else PaperIdentifiers(doi=value) if prefix == "doi" else PaperIdentifiers()
    )
    return Paper(title=f"snapshot-seed:{prefix}", identifiers=identifiers)


def _plan_query_evolution_policy(
    plan: SnapshotPlanRound,
    store: SnapshotStore,
) -> str:
    policies = {
        entry.query_evolution_policy
        for entry in plan.entries
        if entry.query_evolution_policy is not None
    }
    if len(policies) > 1:
        raise ValueError("snapshot_plan_mixed_query_evolution_policy")
    if policies:
        return policies.pop()
    prior = store.read_manifest().groups.get(plan.group)
    if prior is not None and prior.query_evolution_policy is not None:
        return prior.query_evolution_policy
    return "off"


def _plan_query_planning(
    plan: SnapshotPlanRound,
    store: SnapshotStore,
) -> tuple[str, str]:
    policies = {
        entry.query_planning_policy
        for entry in plan.entries
        if entry.query_planning_policy is not None
    }
    versions = {
        entry.query_planner_version
        for entry in plan.entries
        if entry.query_planner_version is not None
    }
    if len(policies) > 1:
        raise ValueError("snapshot_plan_mixed_query_planning_policy")
    if len(versions) > 1:
        raise ValueError("snapshot_plan_mixed_query_planner_version")
    manifest = store.read_manifest()
    prior = manifest.groups.get(plan.group)
    policy = (
        policies.pop()
        if policies
        else (
            prior.query_planning_policy
            if prior is not None and prior.query_planning_policy is not None
            else "current_rules"
        )
    )
    version = (
        versions.pop()
        if versions
        else (
            prior.query_planner_version
            if prior is not None and prior.query_planner_version is not None
            else manifest.query_planner_version
        )
    )
    return policy, version


def _existing_status(
    store: SnapshotStore,
    llm_store: LLMPlanningSnapshotStore,
    entry: SnapshotPlanEntry,
) -> str | None:
    try:
        if entry.entry_type == "llm_planning":
            stored = llm_store.read(entry.key)
        elif entry.entry_type == "retrieval":
            stored = store.read_retrieval(entry.key)
        else:
            stored = store.read_reference(entry.key)
    except SnapshotMissingError:
        return None
    return stored.status


def _prior_completed_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values = payload.get("completed_keys") if isinstance(payload, dict) else None
    return [str(value) for value in values or []]


def _collection_result(
    plan: SnapshotPlanRound,
    *,
    request_count: int,
    failed_count: int,
    collected_count: int,
    skipped_present_count: int,
    completed_keys: list[str],
    blocked_sources: set[str],
    source_failures: dict[str, int],
    stop_reason: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "group": plan.group,
        "round_index": plan.round_index,
        "request_count": request_count,
        "failed_entry_count": failed_count,
        "collected_entry_count": collected_count,
        "skipped_present_count": skipped_present_count,
        "completed_keys": list(completed_keys),
        "blocked_sources": sorted(blocked_sources),
        "source_failure_counts": dict(sorted(source_failures.items())),
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_seconds,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按离线计划有界补齐 Benchmark 快照。")
    parser.add_argument("--collect-plan", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--llm-snapshot-dir", default=None)
    parser.add_argument("--max-new-requests", type=int, default=DEFAULT_MAX_NEW_REQUESTS)
    parser.add_argument(
        "--max-new-failed-entries",
        type=int,
        default=DEFAULT_MAX_NEW_FAILED_ENTRIES,
    )
    parser.add_argument(
        "--max-collection-seconds",
        type=float,
        default=DEFAULT_MAX_COLLECTION_SECONDS,
    )
    parser.add_argument("--retry-failed-snapshots", action="store_true")
    parser.add_argument(
        "--source-failure-limit",
        type=int,
        default=DEFAULT_SOURCE_FAILURE_LIMIT,
    )
    parser.add_argument(
        "--openalex-max-retries",
        type=int,
        default=None,
        help="仅为 OpenAlex 快照采集设置连接器重试上限。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env(REPO_ROOT)
    args = _parser().parse_args(argv)
    try:
        result = collect_plan(
            args.collect_plan,
            args.snapshot_dir,
            llm_snapshot_dir=args.llm_snapshot_dir,
            max_new_requests=args.max_new_requests,
            max_new_failed_entries=args.max_new_failed_entries,
            max_collection_seconds=args.max_collection_seconds,
            retry_failed_snapshots=args.retry_failed_snapshots,
            source_failure_limit=args.source_failure_limit,
            openalex_max_retries=args.openalex_max_retries,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("stop_reason") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
