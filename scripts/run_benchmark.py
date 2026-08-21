#!/usr/bin/env python3
"""运行已注册公开 Benchmark，并写入可恢复的统一评测产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluate_search_batch import evaluate_batch_results  # noqa: E402
from scholar_agent.core.api_schemas import CostReport  # noqa: E402
from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.connectors import (  # noqa: E402
    LocalBM25Config,
    LocalBM25FieldConfig,
    LocalHybridConfig,
    configure_local_bm25,
    configure_local_hybrid,
    fetch_openalex_references_detailed,
    local_bm25_metadata,
    local_hybrid_metadata,
    recommend_semantic_scholar_papers_detailed,
    resolve_semantic_scholar_paper_ids_detailed,
)
from scholar_agent.core.evaluation_schemas import EvalQuery  # noqa: E402
from scholar_agent.core.paper_quality import (  # noqa: E402
    QualityEvidenceCandidateBinding,
    VerifiedQualityEvidenceLedger,
    bind_verified_quality_evidence_to_candidates,
)
from scholar_agent.core.search_schemas import (  # noqa: E402
    DEFAULT_SEARCH_SOURCES,
    JudgementPolicy,
    JudgementRuleConfig,
    QUERY_PLANNER_VERSION,
    QueryEvolutionPolicy,
    QueryPlanningPolicy,
    RankingPolicy,
    SUPPORTED_SEARCH_SOURCES,
    SearchBudget,
)
from scholar_agent.agents.judgement_config import (  # noqa: E402
    judgement_config_hash,
    load_judgement_config,
    resolve_judgement_config,
)
from scholar_agent.evaluation.datasets import (  # noqa: E402
    dataset_source_path,
    inspect_dataset,
    load_dataset,
    supported_datasets,
)
from scholar_agent.evaluation.crash_consistency import (  # noqa: E402
    BenchmarkRunCommitStore,
    durable_atomic_write_text,
)
from scholar_agent.evaluation.experiment_pairing import (  # noqa: E402
    comparison_binding,
    load_comparison_plan,
    opaque_query_identity,
)
from scholar_agent.evaluation.sharded_execution import (  # noqa: E402
    load_shard_plan,
    select_queries_for_shard,
    shard_binding,
)
from scholar_agent.evaluation.resource_accounting import (  # noqa: E402
    QueryResourceLedger,
    ResourceLedgerObserver,
    authority_manifest_identity,
    build_run_ledger,
    opaque_resource_identity,
    validate_resource_ledger,
)
from scholar_agent.evaluation.selection import ResultPolicy  # noqa: E402
from scholar_agent.evaluation.snapshot_resume import (  # noqa: E402
    ResumeRequest,
    ResumeRuntimeConfig,
    SnapshotResumeError,
    execute_resume_manifest,
    load_resume_manifest,
    validate_manifest_required_plan,
    validate_runtime_config,
)
from scholar_agent.evaluation.snapshots import (  # noqa: E402
    SnapshotAwareReferenceFetcher,
    SnapshotAwareRecommendationFetcher,
    SnapshotAwareSemanticSeedResolver,
    SnapshotAwareRetriever,
    SnapshotManifest,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.llm_planning_snapshots import (  # noqa: E402
    LLMPlanningSnapshotRuntime,
    LLMPlanningSnapshotStore,
)
from scholar_agent.evaluation.llm_feedback_snapshots import (  # noqa: E402
    LLMFeedbackSnapshotRuntime,
    LLMFeedbackSnapshotStore,
)
from scholar_agent.evaluation.snapshots.schemas import CONNECTOR_VERSIONS  # noqa: E402
from scholar_agent.evaluation.snapshots.schemas import QUERY_ADAPTER_VERSION  # noqa: E402
from scholar_agent.evaluation.snapshots.schemas import SnapshotPlanRound  # noqa: E402
from scholar_agent.evaluation.snapshots.planning import (  # noqa: E402
    atomic_write_json as _write_plan_json,
    atomic_write_jsonl as _write_plan_jsonl,
    plan_group_root,
    plan_round_root,
    write_coverage_artifacts,
)
from scholar_agent.evaluation.snapshots.store import (  # noqa: E402
    connector_version,
    utc_now,
)
from scholar_agent.evaluation.stage_diagnostics import (  # noqa: E402
    aggregate_stage_diagnostics,
    analyze_search_stages,
)
from scholar_agent.llm.provider import get_llm_runtime_config  # noqa: E402
from scholar_agent.agents.reranker import (  # noqa: E402
    QUALITY_SOFT_RANKING_POLICY,
    quality_soft_ranking_catalog,
)
from scholar_agent.prompts import load_manifest, load_prompt  # noqa: E402
from scholar_agent.retrieval.query_adapter import QueryAdapterPolicy  # noqa: E402
from scholar_agent.services.api_mapper import (  # noqa: E402
    map_search_service_output_to_api_result,
)
from scholar_agent.services.search_service import SearchService  # noqa: E402


RunProfile = Literal["fast", "balanced", "high_recall", "evaluation"]
_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_SENSITIVE_ENV_NAMES = (
    "SCHOLAR_AGENT_LLM_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "NCBI_API_KEY",
    "PUBMED_API_KEY",
)
_LLM_PLANNING_POLICIES = frozenset(
    {"llm_semantic", "llm_constrained_rewrite"}
)


class BenchmarkRunOptions(BaseModel):
    dataset: str
    dataset_path: Path | None = None
    dataset_split: str = "development"
    limit: int | None = Field(default=None, ge=1)
    offset: int = Field(default=0, ge=0)
    output_root: Path = Path("outputs/benchmark_runs")
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    run_profile: RunProfile = "balanced"
    sources: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SEARCH_SOURCES)
    )
    local_bm25_config: LocalBM25Config | None = None
    local_hybrid_config: LocalHybridConfig | None = None
    result_policy: ResultPolicy = "highly_and_partial"
    top_k: int = Field(default=20, ge=1, le=100)
    enable_query_evolution: bool = False
    query_evolution_policy: QueryEvolutionPolicy = "coverage_gap"
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    ranking_policy: RankingPolicy = "current_rules"
    quality_evidence_ledger_path: Path | None = None
    quality_evidence_candidate_identifiers_path: Path | None = None
    quality_evidence_candidate_report_path: Path | None = None
    judgement_policy: JudgementPolicy = "current_rules"
    judgement_config_path: Path | None = None
    enable_refchain: bool = False
    enable_semantic_seed_expansion: bool = False
    enable_llm_query_understanding: bool = False
    enable_llm_judgement: bool = False
    current_year: int | None = Field(default=None, ge=1900, le=2200)
    max_workers: int = Field(default=4, ge=1, le=32)
    budgets: SearchBudget = Field(default_factory=SearchBudget)
    diagnostics: bool = False
    resume: bool = False
    query_adapter_policy: QueryAdapterPolicy = "adaptive"
    retrieval_mode: Literal[
        "live", "record", "replay", "record-missing", "plan"
    ] = "live"
    snapshot_dir: Path | None = None
    llm_mode: Literal["live", "record", "replay", "record-missing"] = "live"
    llm_snapshot_dir: Path | None = None
    plan_round: int = Field(default=1, ge=1)
    retry_failed_snapshots: bool = False
    overwrite_snapshots: bool = False
    comparison_plan_path: Path | None = None
    comparison_role: Literal["baseline", "candidate"] | None = None
    shard_plan_path: Path | None = None
    shard_index: int | None = Field(default=None, ge=0)
    shard_attempt_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
    )
    shard_supersedes_attempt_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
    )
    enable_resource_ledger: bool = False

    @field_validator("sources", mode="before")
    @classmethod
    def validate_sources(cls, value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("sources must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_source in value:
            source = str(raw_source).strip().lower()
            if not source or source in seen:
                continue
            if source not in SUPPORTED_SEARCH_SOURCES:
                raise ValueError(f"unsupported source: {source}")
            normalized.append(source)
            seen.add(source)
        if not normalized:
            raise ValueError("sources must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_comparison_binding(self) -> "BenchmarkRunOptions":
        if (self.comparison_plan_path is None) != (self.comparison_role is None):
            raise ValueError(
                "comparison plan and role must be provided together"
            )
        shard_fields = (
            self.shard_plan_path,
            self.shard_index,
            self.shard_attempt_id,
        )
        if any(value is not None for value in shard_fields) and not all(
            value is not None for value in shard_fields
        ):
            raise ValueError(
                "shard plan, index and attempt id must be provided together"
            )
        if (
            self.shard_supersedes_attempt_id is not None
            and self.shard_plan_path is None
        ):
            raise ValueError("shard supersedes requires a shard plan")
        if "local_bm25" in self.sources and self.local_bm25_config is None:
            raise ValueError("local_bm25_config_required")
        if "local_hybrid" in self.sources and self.local_hybrid_config is None:
            raise ValueError("local_hybrid_config_required")
        if "local_bm25" in self.sources and "local_hybrid" in self.sources:
            raise ValueError("local_bm25_and_local_hybrid_are_mutually_exclusive")
        return self


class BenchmarkRunResult(BaseModel):
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    result_rows: list[dict[str, Any]]
    stage_metrics: dict[str, Any] | None = None


def _ablation_group_name(options: BenchmarkRunOptions) -> str:
    evolution_enabled = (
        options.enable_query_evolution
        and options.query_evolution_policy != "off"
    )
    if options.enable_semantic_seed_expansion:
        base_group = "semantic_seed_expansion"
    elif (
        evolution_enabled
        and options.query_evolution_policy == "coverage_gap"
        and options.enable_refchain
    ):
        base_group = "query_evolution_coverage_gap_plus_refchain"
    elif evolution_enabled and options.query_evolution_policy == "coverage_gap":
        base_group = "query_evolution_coverage_gap"
    elif evolution_enabled and options.enable_refchain:
        base_group = "query_evolution_plus_refchain"
    elif evolution_enabled:
        base_group = "query_evolution_only"
    elif options.enable_refchain:
        base_group = "refchain_only"
    else:
        base_group = "baseline"
    if options.query_planning_policy == "current_rules":
        return base_group
    prefix = options.query_planning_policy
    return prefix if base_group == "baseline" else f"{prefix}_{base_group}"


def _snapshot_manifest(
    options: BenchmarkRunOptions,
    config: dict[str, Any],
) -> SnapshotManifest:
    if options.snapshot_dir is None:
        raise ValueError("snapshot directory is required")
    prompt_rows = config.get("prompts") or []

    def prompt(name: str) -> dict[str, str | int | None]:
        for row in prompt_rows:
            if isinstance(row, dict) and row.get("name") == name:
                return {
                    "name": str(row.get("name") or name),
                    "version": str(row.get("version") or ""),
                    "hash": str(row.get("hash") or ""),
                }
        return {"name": name, "version": None, "hash": None}

    now = utc_now()
    code = config.get("code") or {}
    connector_versions = dict(CONNECTOR_VERSIONS)
    if "local_bm25" in options.sources:
        connector_versions["local_bm25"] = connector_version("local_bm25")
    if "local_hybrid" in options.sources:
        connector_versions["local_hybrid"] = connector_version("local_hybrid")
    return SnapshotManifest(
        snapshot_name=options.snapshot_dir.name,
        dataset=options.dataset,
        split=options.dataset_split,
        offset=options.offset,
        limit=options.limit,
        sources=list(options.sources),
        adapter_policy=options.query_adapter_policy,
        query_adapter_version=QUERY_ADAPTER_VERSION,
        query_planner_version=QUERY_PLANNER_VERSION,
        run_profile=options.run_profile,
        budgets=options.budgets.model_dump(mode="json"),
        llm_enabled=bool((config.get("llm") or {}).get("llm_enabled")),
        query_understanding_prompt=prompt("query_understanding"),
        llm_query_planning_prompt=(
            prompt(
                "llm_constrained_rewrite"
                if options.query_planning_policy == "llm_constrained_rewrite"
                else "llm_query_planning"
            )
            if options.query_planning_policy in _LLM_PLANNING_POLICIES
            else {}
        ),
        judgement_prompt=prompt("relevance_judgement"),
        connector_versions=connector_versions,
        code_hash=str(config.get("runtime_code_hash") or ""),
        git_commit=code.get("commit"),
        dirty_worktree=bool(code.get("dirty")),
        created_at=now,
        updated_at=now,
    )


def _aggregate_snapshot_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        row.get("snapshot_cost_report")
        for row in rows
        if isinstance(row.get("snapshot_cost_report"), dict)
    ]
    numeric_fields = (
        "retrieval_snapshot_hits",
        "reference_snapshot_hits",
        "retrieval_snapshot_writes",
        "reference_snapshot_writes",
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
        "recorded_search_request_count",
        "recorded_reference_request_count",
        "recorded_retry_count",
        "recorded_error_count",
        "recorded_rate_limit_wait_seconds",
        "recorded_latency_seconds",
    )
    return {
        "case_count": len(reports),
        **{
            field: sum(float(report.get(field) or 0) for report in reports)
            for field in numeric_fields
        },
        "missing_retrieval_keys": sorted(
            {
                key
                for report in reports
                for key in report.get("missing_retrieval_keys") or []
            }
        ),
        "missing_reference_keys": sorted(
            {
                key
                for report in reports
                for key in report.get("missing_reference_keys") or []
            }
        ),
        "fatal_errors": sorted(
            {
                error
                for report in reports
                for error in report.get("fatal_errors") or []
            }
        ),
    }


def _aggregate_llm_planning_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        row.get("llm_planning_cost_report")
        for row in rows
        if isinstance(row.get("llm_planning_cost_report"), dict)
    ]
    numeric_fields = (
        "snapshot_hits",
        "snapshot_writes",
        "live_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "recorded_latency_seconds",
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
    )
    return {
        "case_count": len(reports),
        **{
            field: sum(float(report.get(field) or 0) for report in reports)
            for field in numeric_fields
        },
        "missing_keys": sorted(
            {
                key
                for report in reports
                for key in report.get("missing_keys") or []
            }
        ),
    }


def _aggregate_llm_feedback_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        row.get("llm_feedback_cost_report")
        for row in rows
        if isinstance(row.get("llm_feedback_cost_report"), dict)
    ]
    numeric_fields = (
        "snapshot_hits",
        "snapshot_writes",
        "live_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "recorded_latency_seconds",
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
    )
    return {
        "enabled": bool(reports),
        **{
            field: sum(float(report.get(field, 0) or 0) for report in reports)
            for field in numeric_fields
        },
        "missing_keys": sorted(
            {
                key
                for report in reports
                for key in report.get("missing_keys") or []
            }
        ),
    }


def _write_snapshot_plan_artifacts(
    options: BenchmarkRunOptions,
    runtime: SnapshotRuntime,
    llm_runtime: LLMPlanningSnapshotRuntime | None = None,
) -> None:
    if options.snapshot_dir is None:
        return
    llm_entries = llm_runtime.plan_entries() if llm_runtime is not None else []
    # LLM 规划键是检索查询键的上游依赖。缺失时本轮只计划 LLM，不能用
    # fallback 查询猜测后续 adapted retrieval key。
    if llm_entries:
        entries = llm_entries
    else:
        entries = []
        for entry in runtime.plan_entries():
            dependency_keys = list(entry.dependency_keys)
            if llm_runtime is not None:
                for key in llm_runtime.dependency_keys(entry.case_id):
                    if key not in dependency_keys:
                        dependency_keys.insert(0, key)
            entries.append(entry.model_copy(update={"dependency_keys": dependency_keys}))
    missing = [entry for entry in entries if not entry.already_present]
    group = _ablation_group_name(options)
    group_root = plan_group_root(options.snapshot_dir, group)
    round_root = plan_round_root(options.snapshot_dir, group, options.plan_round)
    plan = SnapshotPlanRound(
        snapshot_name=options.snapshot_dir.name,
        group=group,
        round_index=options.plan_round,
        entries=entries,
        missing_retrieval_count=sum(
            entry.entry_type == "retrieval" for entry in missing
        ),
        missing_reference_count=sum(
            entry.entry_type == "reference" for entry in missing
        ),
        missing_llm_planning_count=sum(
            entry.entry_type == "llm_planning" for entry in missing
        ),
        network_request_count=0,
        converged=not missing,
        stop_reason=None if not missing else "snapshot_missing",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_plan_json(
        group_root / f"plan_round_{options.plan_round}.json",
        plan.model_dump(mode="json"),
    )
    _write_plan_jsonl(
        round_root / "missing_retrieval_keys.jsonl",
        [
            entry.model_dump(mode="json")
            for entry in missing
            if entry.entry_type == "retrieval"
        ],
    )
    _write_plan_jsonl(
        round_root / "missing_llm_planning_keys.jsonl",
        [
            entry.model_dump(mode="json")
            for entry in missing
            if entry.entry_type == "llm_planning"
        ],
    )
    _write_plan_jsonl(
        round_root / "missing_reference_keys.jsonl",
        [
            entry.model_dump(mode="json")
            for entry in missing
            if entry.entry_type == "reference"
        ],
    )
    result = {
        "group": group,
        "round_index": options.plan_round,
        "planned_key_count": len(entries),
        "missing_key_count": len(missing),
        "network_request_count": 0,
        "stop_reason": plan.stop_reason,
    }
    _write_plan_json(round_root / "collection_result.json", result)
    write_coverage_artifacts(
        options.snapshot_dir,
        group=group,
        round_index=options.plan_round,
    )


def run_benchmark(
    options: BenchmarkRunOptions,
    *,
    service: Any | None = None,
) -> BenchmarkRunResult:
    _configure_local_sources_for_run(options)
    _validate_llm_planning_runtime(options, service=service)
    quality_evidence_ledger, quality_evidence_binding = _load_quality_evidence_ledger(
        options
    )
    source_path = dataset_source_path(options.dataset, options.dataset_path)
    judgement_config = _resolve_options_judgement_config(options)
    all_queries = load_dataset(options.dataset, path=source_path)
    population = _select_queries(all_queries, options.offset, options.limit)
    _validate_comparison_population(options, population)
    selected = _select_shard_population(options, population)
    dataset_report = inspect_dataset(options.dataset, path=source_path)
    run_dir = options.output_root.expanduser().resolve() / options.run_id
    config = _build_config(
        options,
        source_path,
        selected,
        quality_evidence_ledger=quality_evidence_ledger,
        quality_evidence_binding=quality_evidence_binding,
    )
    commit_store = BenchmarkRunCommitStore(run_dir)

    existing_rows: dict[str, dict[str, Any]] = {}
    if options.resume:
        config, existing_rows = _prepare_resume(run_dir, config, selected)
        if not commit_store.has_commits:
            # Legacy runs remain ineligible for retrospective crash guarantees,
            # but a live resume starts a new authoritative generation chain.
            stored_report = _read_json(run_dir / "dataset_report.json")
            state = commit_store.initialize(
                run_id=options.run_id,
                expected_query_ids=[item.query_id for item in selected],
                config=config,
                dataset_report=stored_report,
            )
            for query in selected:
                if query.query_id in existing_rows:
                    state = commit_store.commit_record(existing_rows[query.query_id])
            commit_store.materialize_compatibility_view(state)
    else:
        if run_dir.exists():
            raise ValueError(f"run directory already exists; use --resume: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        state = commit_store.initialize(
            run_id=options.run_id,
            expected_query_ids=[item.query_id for item in selected],
            config=config,
            dataset_report=dataset_report.model_dump(mode="json"),
        )
        commit_store.materialize_compatibility_view(state)

    state = commit_store.load_latest()

    snapshot_runtime: SnapshotRuntime | None = None
    llm_planning_runtime: LLMPlanningSnapshotRuntime | None = None
    llm_feedback_runtime: LLMFeedbackSnapshotRuntime | None = None
    if (
        options.query_planning_policy in _LLM_PLANNING_POLICIES
        and options.llm_mode != "live"
    ):
        if service is not None:
            raise ValueError("LLM snapshot modes require the real SearchService")
        if options.llm_snapshot_dir is None:
            raise ValueError("--llm-snapshot-dir is required outside LLM live mode")
        llm_planning_runtime = LLMPlanningSnapshotRuntime(
            LLMPlanningSnapshotStore(options.llm_snapshot_dir),
            mode=options.llm_mode,
            group_name=_ablation_group_name(options),
        )
    if (
        options.query_evolution_policy == "llm_feedback"
        and options.enable_query_evolution
        and options.llm_mode != "live"
    ):
        if service is not None:
            raise ValueError("LLM snapshot modes require the real SearchService")
        if options.llm_snapshot_dir is None:
            raise ValueError("--llm-snapshot-dir is required outside LLM live mode")
        llm_feedback_runtime = LLMFeedbackSnapshotRuntime(
            LLMFeedbackSnapshotStore(options.llm_snapshot_dir),
            mode=options.llm_mode,
        )
    if options.retrieval_mode != "live":
        if service is not None:
            raise ValueError("snapshot modes require the real SearchService")
        if options.snapshot_dir is None:
            raise ValueError("--snapshot-dir is required outside live mode")
        store = SnapshotStore(options.snapshot_dir)
        manifest = store.ensure_manifest(_snapshot_manifest(options, config))
        group_name = _ablation_group_name(options)
        if options.retrieval_mode == "replay":
            coverage = store.inspect().get("groups", {}).get(group_name, {})
            if not coverage.get("replay_ready"):
                raise ValueError(f"snapshot_group_not_replay_ready:{group_name}")
        snapshot_runtime = SnapshotRuntime(
            store,
            mode=options.retrieval_mode,
            group_name=group_name,
            retry_failed_snapshots=options.retry_failed_snapshots,
            overwrite_snapshots=options.overwrite_snapshots,
            plan_round=options.plan_round,
            query_evolution_policy=(
                options.query_evolution_policy
                if options.enable_query_evolution
                else "off"
            ),
            query_planning_policy=options.query_planning_policy,
            query_planner_version=QUERY_PLANNER_VERSION,
            judgement_policy=options.judgement_policy,
            judgement_config_hash=judgement_config_hash(judgement_config),
        )
        runner = SearchService(
            retriever=SnapshotAwareRetriever(snapshot_runtime),
            reference_fetcher=SnapshotAwareReferenceFetcher(
                snapshot_runtime,
                fetch_openalex_references_detailed,
            ),
            recommendation_fetcher=SnapshotAwareRecommendationFetcher(
                snapshot_runtime,
                recommend_semantic_scholar_papers_detailed,
            ),
            semantic_seed_resolver=SnapshotAwareSemanticSeedResolver(
                snapshot_runtime,
                resolve_semantic_scholar_paper_ids_detailed,
            ),
            max_workers=options.max_workers,
            llm_planning_runtime=llm_planning_runtime,
            llm_feedback_runtime=llm_feedback_runtime,
            judgement_policy=options.judgement_policy,
            judgement_config=judgement_config,
        )
    else:
        runner = service or SearchService(
            max_workers=options.max_workers,
            llm_planning_runtime=llm_planning_runtime,
            llm_feedback_runtime=llm_feedback_runtime,
            judgement_policy=options.judgement_policy,
            judgement_config=judgement_config,
        )
    selected_ids = [query.query_id for query in selected]
    resource_query_ids = [
        opaque_resource_identity("query", item) for item in selected_ids
    ]
    resource_run_identity = opaque_resource_identity("run", options.run_id)
    resource_manifest_identity = authority_manifest_identity(
        options.run_id,
        expected_query_identities=resource_query_ids,
        configuration={"resume_signature": config["resume_signature"]},
    )
    for query in selected:
        previous = existing_rows.get(query.query_id)
        if previous is not None and previous.get("status") == "succeeded":
            continue
        resource_context: dict[str, Any] | None = None
        if options.enable_resource_ledger:
            previous_ledger = (
                previous.get("resource_ledger")
                if isinstance(previous, dict)
                else None
            )
            prior_attempt = (
                str(previous_ledger.get("attempt_identity"))
                if isinstance(previous_ledger, dict)
                and previous_ledger.get("attempt_identity")
                else None
            )
            resource_context = {
                "run_identity": resource_run_identity,
                "query_identity": opaque_resource_identity("query", query.query_id),
                "attempt_identity": opaque_resource_identity(
                    "attempt",
                    f"{options.run_id}:{query.query_id}:{state.generation + 1}",
                ),
                "superseded_attempt_identity": prior_attempt,
                "checkpoint_generation": state.generation + 1,
                "manifest_identity": resource_manifest_identity,
            }
        existing_rows[query.query_id] = _run_case(
            runner,
            query,
            options,
            snapshot_runtime=snapshot_runtime,
            llm_planning_runtime=llm_planning_runtime,
            llm_feedback_runtime=llm_feedback_runtime,
            judgement_config=judgement_config,
            resource_ledger_context=resource_context,
            quality_evidence_ledger=quality_evidence_ledger,
        )
        state = commit_store.commit_record(existing_rows[query.query_id])
        commit_store.materialize_compatibility_view(state)

    ordered_rows = [existing_rows[case_id] for case_id in selected_ids]
    metrics = _evaluate_rows(ordered_rows, selected, options.result_policy)
    metrics["snapshot_costs"] = _aggregate_snapshot_costs(ordered_rows)
    metrics["llm_planning_costs"] = _aggregate_llm_planning_costs(ordered_rows)
    metrics["llm_feedback_costs"] = _aggregate_llm_feedback_costs(ordered_rows)
    stage_metrics: dict[str, Any] | None = None
    reports: dict[str, bytes] = {
        "metrics.json": _json_bytes(metrics),
    }
    if options.enable_resource_ledger:
        query_ledgers = [
            QueryResourceLedger.model_validate(row["resource_ledger"])
            for row in ordered_rows
        ]
        selected_attempts = {
            item.query_identity: item.attempt_identity for item in query_ledgers
        }
        superseded_attempts = sorted(
            {
                str(attempt)
                for row in ordered_rows
                for attempt in row.get("resource_ledger_superseded_attempts", [])
            }
        )
        run_ledger = build_run_ledger(
            query_ledgers,
            run_identity=resource_run_identity,
            manifest_identity=resource_manifest_identity,
            expected_query_identities=resource_query_ids,
            selected_attempts=selected_attempts,
            superseded_attempts=superseded_attempts,
        )
        ledger_report = validate_resource_ledger(
            run_ledger,
            authoritative_records=ordered_rows,
        )
        if ledger_report["status"] != "passed":
            raise ValueError("resource_ledger_integrity_violation")
        reports["resource_ledger.json"] = _json_bytes(
            run_ledger.model_dump(mode="json")
        )
    if options.diagnostics:
        case_diagnostics = [
            row["stage_diagnostics"]
            for row in ordered_rows
            if isinstance(row.get("stage_diagnostics"), dict)
        ]
        stage_metrics, error_analysis, gold_diagnostics = (
            aggregate_stage_diagnostics(case_diagnostics)
        )
        reports.update(
            {
                "stage_metrics.json": _json_bytes(stage_metrics),
                "error_analysis.json": _json_bytes(error_analysis),
                "gold_diagnostics.jsonl": _jsonl_bytes(gold_diagnostics),
            }
        )
    reports["summary.md"] = _summary_markdown(
        config, metrics, stage_metrics
    ).encode("utf-8")
    if snapshot_runtime is not None:
        snapshot_runtime.finish_group(
            completed=all(row.get("status") == "succeeded" for row in ordered_rows)
        )
        if options.retrieval_mode == "plan" and options.snapshot_dir is not None:
            _write_snapshot_plan_artifacts(
                options,
                snapshot_runtime,
                llm_planning_runtime,
            )
    state = commit_store.commit_completion(reports)
    commit_store.materialize_compatibility_view(state)
    return BenchmarkRunResult(
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        result_rows=ordered_rows,
        stage_metrics=stage_metrics,
    )


def _uses_runtime_llm(options: BenchmarkRunOptions) -> bool:
    return (
        options.enable_llm_query_understanding
        or options.enable_llm_judgement
        or options.query_planning_policy in _LLM_PLANNING_POLICIES
        or (
            options.enable_query_evolution
            and options.query_evolution_policy == "llm_feedback"
        )
    )


def _validate_llm_planning_runtime(
    options: BenchmarkRunOptions,
    *,
    service: Any | None,
) -> None:
    if not _uses_runtime_llm(options) or service is not None:
        return
    runtime = get_llm_runtime_config()
    if options.llm_mode in {"live", "record"} and not runtime.available:
        raise ValueError(
            "LLM query planning requires an available LLM provider in live/record mode"
        )
    if options.llm_mode == "live":
        return
    if options.llm_snapshot_dir is None:
        raise ValueError("--llm-snapshot-dir is required outside LLM live mode")
    store = (
        LLMFeedbackSnapshotStore(options.llm_snapshot_dir)
        if options.query_evolution_policy == "llm_feedback"
        else LLMPlanningSnapshotStore(options.llm_snapshot_dir)
    )
    identity = store.identity()
    has_config_identity = runtime.provider != "disabled" and bool(runtime.model)
    if options.llm_mode in {"replay", "record-missing"} and identity is None:
        if not has_config_identity:
            raise ValueError("llm_planning_snapshot_identity_unavailable")


def _resolve_options_judgement_config(
    options: BenchmarkRunOptions,
) -> JudgementRuleConfig:
    explicit = (
        load_judgement_config(options.judgement_config_path)
        if options.judgement_config_path is not None
        else None
    )
    return resolve_judgement_config(options.judgement_policy, explicit)


def _load_quality_evidence_ledger(
    options: BenchmarkRunOptions,
) -> tuple[
    VerifiedQualityEvidenceLedger | None, QualityEvidenceCandidateBinding | None
]:
    if options.quality_evidence_ledger_path is None:
        if (
            options.quality_evidence_candidate_identifiers_path is not None
            or options.quality_evidence_candidate_report_path is not None
        ):
            raise ValueError("quality_evidence_candidate_binding_requires_ledger")
        return None, None
    if options.ranking_policy != QUALITY_SOFT_RANKING_POLICY:
        raise ValueError("quality_evidence_ledger_requires_quality_soft_ranking")
    if (
        options.quality_evidence_candidate_identifiers_path is None
        or options.quality_evidence_candidate_report_path is None
    ):
        raise ValueError("quality_evidence_candidate_binding_required")
    return bind_verified_quality_evidence_to_candidates(
        options.quality_evidence_ledger_path,
        options.quality_evidence_candidate_identifiers_path,
        options.quality_evidence_candidate_report_path,
    )


def _select_queries(
    queries: list[EvalQuery],
    offset: int,
    limit: int | None,
) -> list[EvalQuery]:
    selected = queries[offset:] if limit is None else queries[offset : offset + limit]
    if not selected:
        raise ValueError("offset/limit selected no benchmark cases")
    return selected


def _configure_local_sources_for_run(options: BenchmarkRunOptions) -> None:
    build_index = options.retrieval_mode in {"live", "record", "record-missing"}
    if "local_hybrid" in options.sources:
        if options.local_hybrid_config is None:
            raise ValueError(
                "--local-hybrid configuration is required when local_hybrid is selected"
            )
        configure_local_hybrid(
            options.local_hybrid_config,
            build_bm25_index=build_index,
        )
        return
    if "local_bm25" not in options.sources:
        configure_local_bm25(None)
        configure_local_hybrid(None)
        return
    if options.local_bm25_config is None:
        raise ValueError(
            "--local-bm25-corpus is required when local_bm25 is selected"
        )
    configure_local_bm25(options.local_bm25_config, build_index=build_index)


def _build_config(
    options: BenchmarkRunOptions,
    source_path: Path,
    selected: list[EvalQuery],
    *,
    quality_evidence_ledger: VerifiedQualityEvidenceLedger | None = None,
    quality_evidence_binding: QualityEvidenceCandidateBinding | None = None,
) -> dict[str, Any]:
    llm_runtime = get_llm_runtime_config()
    judgement_config = _resolve_options_judgement_config(options)
    requested_llm = _uses_runtime_llm(options)
    local_bm25 = None
    if "local_bm25" in options.sources:
        metadata = local_bm25_metadata()
        fields = (
            options.local_bm25_config.fields
            if options.local_bm25_config
            else None
        )
        local_bm25 = {
            "connector_version": connector_version("local_bm25"),
            "corpus_path": (
                str(options.local_bm25_config.corpus_path.expanduser().resolve())
                if options.local_bm25_config
                else None
            ),
            "corpus_sha256": metadata.corpus_sha256,
            "corpus_size_bytes": metadata.corpus_size_bytes,
            "document_count": metadata.document_count,
            "index_fingerprint": metadata.fingerprint,
            "index_cache_path": metadata.cache_path,
            "index_cache_hit": metadata.cache_hit,
            "index_load_seconds": metadata.index_load_seconds,
            "fields": asdict(fields) if fields is not None else None,
            "parameters": (
                {
                    "k1": options.local_bm25_config.k1,
                    "b": options.local_bm25_config.b,
                    "epsilon": options.local_bm25_config.epsilon,
                }
                if options.local_bm25_config
                else None
            ),
            "document_text": "title+abstract",
            "query_input": "current_rules_generated_text_only",
            "evaluator_data_access": False,
        }
    local_hybrid = None
    if "local_hybrid" in options.sources:
        if options.local_hybrid_config is None:
            raise ValueError("local_hybrid_config_required")
        metadata = local_hybrid_metadata()
        hybrid = options.local_hybrid_config
        local_hybrid = {
            "connector_version": connector_version("local_hybrid"),
            "bm25_corpus_path": str(
                hybrid.bm25_config.corpus_path.expanduser().resolve()
            ),
            "bm25_corpus_sha256": local_bm25_metadata().corpus_sha256,
            "bm25_document_count": local_bm25_metadata().document_count,
            "semantic_corpus_path": str(
                hybrid.semantic_corpus_path.expanduser().resolve()
            ),
            "semantic_corpus_sha256": metadata.semantic_corpus_sha256,
            "semantic_corpus_size_bytes": metadata.semantic_corpus_size_bytes,
            "semantic_document_count": metadata.document_count,
            "semantic_abstract_document_count": metadata.abstract_document_count,
            "embedding_dimension": metadata.embedding_dimension,
            "model_path": str(hybrid.model_path.expanduser().resolve()),
            "model_fingerprint": metadata.model_fingerprint,
            "reranker_model_path": (
                str(hybrid.reranker_model_path.expanduser().resolve())
                if hybrid.reranker_model_path is not None
                else None
            ),
            "reranker_candidate_limit": hybrid.reranker_candidate_limit,
            "reranker_batch_size": hybrid.reranker_batch_size,
            "reranker_device": hybrid.reranker_device,
            "index_dir": metadata.index_dir,
            "index_fingerprint": metadata.index_fingerprint,
            "index_cache_hit": metadata.cache_hit,
            "index_load_seconds": metadata.index_load_seconds,
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "rrf_k": hybrid.rrf_k,
                "bm25_candidate_limit": hybrid.bm25_candidate_limit,
                "semantic_candidate_limit": hybrid.semantic_candidate_limit,
            },
            "query_input": "current_rules_generated_text_only",
            "evaluator_data_access": False,
        }
    semantic_config = {
        "dataset": options.dataset,
        "dataset_source_path": str(source_path),
        "dataset_split": options.dataset_split,
        "dataset_sha256": _file_sha256(source_path),
        "case_count": len(selected),
        "case_ids": [item.query_id for item in selected],
        "offset": options.offset,
        "limit": options.limit,
        "selection_order": "source_order",
        "result_policy": options.result_policy,
        "sources": list(options.sources),
        "local_bm25": local_bm25,
        "local_hybrid": local_hybrid,
        "run_profile": options.run_profile,
        "top_k": options.top_k,
        "enable_query_evolution": options.enable_query_evolution,
        "query_evolution_policy": (
            options.query_evolution_policy
            if options.enable_query_evolution
            else "off"
        ),
        "query_planning_policy": options.query_planning_policy,
        "ranking_policy": options.ranking_policy,
        "quality_evidence_ledger": (
            {
                "schema_version": quality_evidence_ledger.schema_version,
                "file_sha256": quality_evidence_ledger.file_sha256,
                "semantic_sha256": quality_evidence_ledger.semantic_sha256,
                "record_count": quality_evidence_ledger.record_count,
            }
            if quality_evidence_ledger is not None
            else None
        ),
        "quality_evidence_candidate_binding": (
            quality_evidence_binding.model_dump(mode="json")
            if quality_evidence_binding is not None
            else None
        ),
        "query_planner_version": QUERY_PLANNER_VERSION,
        "judgement_policy": options.judgement_policy,
        "judgement_config": judgement_config.model_dump(mode="json"),
        "judgement_config_hash": judgement_config_hash(judgement_config),
        "enable_refchain": options.enable_refchain,
        "enable_semantic_seed_expansion": (
            options.enable_semantic_seed_expansion
        ),
        "current_year": options.current_year,
        "max_workers": options.max_workers,
        "budgets": options.budgets.model_dump(mode="json"),
        "diagnostics": options.diagnostics,
        "enable_resource_ledger": options.enable_resource_ledger,
        "query_adapter_policy": options.query_adapter_policy,
        "retrieval_mode": options.retrieval_mode,
        "llm_mode": options.llm_mode,
        "llm_snapshot": (
            {
                "directory": str(options.llm_snapshot_dir.expanduser().resolve()),
                "name": options.llm_snapshot_dir.name,
            }
            if options.llm_snapshot_dir is not None
            else None
        ),
        "snapshot": (
            {
                "directory": str(options.snapshot_dir.expanduser().resolve()),
                "name": options.snapshot_dir.name,
                "group": _ablation_group_name(options),
                "retry_failed_snapshots": options.retry_failed_snapshots,
                "overwrite_snapshots": options.overwrite_snapshots,
                "plan_round": options.plan_round,
            }
            if options.snapshot_dir is not None
            else None
        ),
        "llm": {
            "llm_enabled": bool(requested_llm and llm_runtime.available),
            "requested": requested_llm,
            "query_understanding": options.enable_llm_query_understanding,
            "judgement": options.enable_llm_judgement,
            "semantic_query_planning": (
                options.query_planning_policy == "llm_semantic"
            ),
            "constrained_query_rewrite": (
                options.query_planning_policy == "llm_constrained_rewrite"
            ),
            "feedback_query_evolution": (
                options.enable_query_evolution
                and options.query_evolution_policy == "llm_feedback"
            ),
            "provider": llm_runtime.provider,
            "model": llm_runtime.model,
            "runtime_available": llm_runtime.available,
        },
        "prompts": _prompt_metadata(),
        "runtime_code_hash": _runtime_code_hash(),
        "code": _git_metadata(),
        "execution": _execution_metadata(),
    }
    if options.comparison_plan_path is not None and options.comparison_role is not None:
        semantic_config["comparison"] = comparison_binding(
            options.comparison_plan_path, options.comparison_role
        )
    if options.ranking_policy == QUALITY_SOFT_RANKING_POLICY:
        semantic_config["quality_soft_ranking"] = quality_soft_ranking_catalog()
    if (
        options.shard_plan_path is not None
        and options.shard_index is not None
        and options.shard_attempt_id is not None
    ):
        semantic_config["shard"] = shard_binding(
            options.shard_plan_path,
            options.shard_index,
            options.shard_attempt_id,
            options.shard_supersedes_attempt_id,
        )
    signature_payload = {
        key: value
        for key, value in semantic_config.items()
        if key not in {"code", "execution", "local_bm25"}
    }
    if local_bm25 is not None:
        # Cache paths and load observations vary across restarts and machines.
        # They remain in config.json for auditability but must not invalidate a
        # checkpoint whose corpus, fields, and retrieval semantics are unchanged.
        signature_payload["local_bm25"] = {
            key: value
            for key, value in local_bm25.items()
            if key
            not in {
                "corpus_path",
                "index_cache_path",
                "index_cache_hit",
                "index_load_seconds",
            }
        }
    if local_hybrid is not None:
        signature_payload["local_hybrid"] = {
            key: value
            for key, value in local_hybrid.items()
            if key
            not in {
                "bm25_corpus_path",
                "semantic_corpus_path",
                "model_path",
                "index_dir",
                "index_cache_hit",
                "index_load_seconds",
            }
        }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **semantic_config,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "resume_signature": signature,
    }


def _validate_comparison_population(
    options: BenchmarkRunOptions, selected: list[EvalQuery]
) -> None:
    if options.comparison_plan_path is None:
        return
    plan = load_comparison_plan(options.comparison_plan_path)
    observed = [opaque_query_identity(item.query_id) for item in selected]
    if observed != plan.queries.identities:
        raise ValueError("comparison plan query population or order mismatch")


def _select_shard_population(
    options: BenchmarkRunOptions, selected: list[EvalQuery]
) -> list[EvalQuery]:
    if options.shard_plan_path is None:
        return selected
    if options.shard_index is None:
        raise ValueError("shard index is required")
    plan = load_shard_plan(options.shard_plan_path)
    try:
        return select_queries_for_shard(selected, plan, options.shard_index)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def _prompt_metadata() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for name, entry in load_manifest().items():
        if not entry.runtime_enabled:
            continue
        prompt = load_prompt(name)
        prompts.append(
            {
                "name": prompt.name,
                "version": prompt.version,
                "hash": prompt.content_hash,
            }
        )
    return prompts


def _git_metadata() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--porcelain", "--untracked-files=no"])
    diff = _git_bytes(["diff", "--binary", "HEAD", "--", "."])
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "working_tree_diff_hash": hashlib.sha256(diff).hexdigest(),
    }


def _execution_metadata() -> dict[str, Any]:
    """Record launch facts without allowing credentials into run artifacts."""

    log_path = os.getenv("SCHOLARNAVIGATOR_RUN_LOG_PATH", "").strip()
    if log_path and not Path(log_path).as_posix().startswith("outputs/run_logs/"):
        log_path = ""
    return {
        "process_id": os.getpid(),
        "launch_command": [_sanitize_message(value) for value in sys.argv],
        "log_path": log_path or None,
    }


def _runtime_code_hash() -> str:
    paths = sorted((REPO_ROOT / "src" / "scholar_agent").rglob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "scripts" / "evaluate_search_batch.py",
            REPO_ROOT / "scripts" / "run_benchmark.py",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_output(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _git_bytes(arguments: list[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return b""
    return result.stdout


def _prepare_resume(
    run_dir: Path,
    current_config: dict[str, Any],
    selected: list[EvalQuery],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    commit_store = BenchmarkRunCommitStore(run_dir)
    if commit_store.has_commits:
        state = commit_store.load_latest()
        stored = state.config
        if stored.get("resume_signature") != current_config.get("resume_signature"):
            raise ValueError("resume config is incompatible with the existing run")
        allowed_ids = {item.query_id for item in selected}
        indexed = state.rows_by_id
        if set(indexed) - allowed_ids:
            raise ValueError("invalid committed resume results: case_id")
        commit_store.materialize_compatibility_view(state)
        return stored, indexed
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    if not config_path.is_file() or not results_path.is_file():
        raise ValueError("resume requires existing config.json and results.jsonl")
    stored = _read_json(config_path)
    if stored.get("resume_signature") != current_config.get("resume_signature"):
        raise ValueError("resume config is incompatible with the existing run")

    allowed_ids = {item.query_id for item in selected}
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in _read_jsonl(results_path):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id not in allowed_ids:
            raise ValueError(f"invalid resume results at line {line_number}: case_id")
        if case_id in indexed:
            raise ValueError(
                f"invalid resume results at line {line_number}: duplicate {case_id}"
            )
        indexed[case_id] = row
    return stored, indexed


def _run_case(
    service: Any,
    query: EvalQuery,
    options: BenchmarkRunOptions,
    snapshot_runtime: SnapshotRuntime | None = None,
    llm_planning_runtime: LLMPlanningSnapshotRuntime | None = None,
    llm_feedback_runtime: LLMFeedbackSnapshotRuntime | None = None,
    judgement_config: JudgementRuleConfig | None = None,
    resource_ledger_context: Mapping[str, Any] | None = None,
    quality_evidence_ledger: VerifiedQualityEvidenceLedger | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resource_observer = (
        ResourceLedgerObserver(options.budgets)
        if resource_ledger_context is not None
        else None
    )
    if snapshot_runtime is not None:
        snapshot_runtime.begin_case(query.query_id)
    if llm_planning_runtime is not None:
        llm_planning_runtime.begin_case(query.query_id)
    if llm_feedback_runtime is not None:
        llm_feedback_runtime.begin_case(query.query_id)
    try:
        judgement_kwargs: dict[str, Any] = {}
        if (
            options.judgement_policy != "current_rules"
            or options.judgement_config_path is not None
        ):
            judgement_kwargs = {
                "judgement_policy": options.judgement_policy,
                "judgement_config": judgement_config,
            }
        resource_kwargs = (
            {"resource_accounting_observer": resource_observer}
            if resource_observer is not None
            else {}
        )
        output = service.run_search(
            query.query,
            top_k=options.top_k,
            run_profile=options.run_profile,
            enable_query_evolution=options.enable_query_evolution,
            query_evolution_policy=options.query_evolution_policy,
            query_planning_policy=options.query_planning_policy,
            ranking_policy=options.ranking_policy,
            enable_refchain=options.enable_refchain,
            enable_semantic_seed_expansion=(
                options.enable_semantic_seed_expansion
            ),
            enable_synthesis=True,
            current_year=options.current_year,
            enable_llm_query_understanding=options.enable_llm_query_understanding,
            enable_llm_judgement=options.enable_llm_judgement,
            sources_override=list(options.sources),
            budget=options.budgets,
            collect_diagnostics=options.diagnostics,
            query_adapter_policy=options.query_adapter_policy,
            verified_quality_evidence=(
                quality_evidence_ledger.evidence
                if quality_evidence_ledger is not None
                else ()
            ),
            **judgement_kwargs,
            **resource_kwargs,
        )
        result = map_search_service_output_to_api_result(
            run_id=f"benchmark_{query.query_id}",
            output=output,
            status="succeeded",
            partial=False,
        ).model_dump(mode="json")
        if snapshot_runtime is not None and options.retrieval_mode != "plan":
            snapshot_runtime.assert_case_complete()
        cost_report = dict(result.get("cost_report") or {})
        row = {
            "case_id": query.query_id,
            "query": query.query,
            "status": "succeeded",
            "result": result,
            "error": None,
            "latency_seconds": time.perf_counter() - started,
            "cost_report": cost_report,
        }
        if options.diagnostics:
            row["stage_diagnostics"] = analyze_search_stages(
                query,
                output,
                result_policy=options.result_policy,
            )
        if snapshot_runtime is not None:
            row["snapshot_cost_report"] = snapshot_runtime.finish_case().model_dump(
                mode="json"
            )
        if llm_planning_runtime is not None:
            row["llm_planning_cost_report"] = (
                llm_planning_runtime.finish_case().model_dump(mode="json")
            )
        if llm_feedback_runtime is not None:
            row["llm_feedback_cost_report"] = (
                llm_feedback_runtime.finish_case().model_dump(mode="json")
            )
        return _attach_resource_ledger(
            row,
            resource_observer,
            resource_ledger_context,
            terminal_status="succeeded",
        )
    except Exception as exc:  # noqa: BLE001 - isolate benchmark cases
        row = {
            "case_id": query.query_id,
            "query": query.query,
            "status": "failed",
            "result": None,
            "error": _sanitize_message(str(exc)),
            "error_type": type(exc).__name__,
            "latency_seconds": time.perf_counter() - started,
            "cost_report": CostReport().model_dump(mode="json"),
        }
        if snapshot_runtime is not None:
            row["snapshot_cost_report"] = snapshot_runtime.finish_case().model_dump(
                mode="json"
            )
        if llm_planning_runtime is not None:
            row["llm_planning_cost_report"] = (
                llm_planning_runtime.finish_case().model_dump(mode="json")
            )
        if llm_feedback_runtime is not None:
            row["llm_feedback_cost_report"] = (
                llm_feedback_runtime.finish_case().model_dump(mode="json")
            )
        terminal_status = (
            "cancelled" if type(exc).__name__ == "SearchCancelled" else "failed"
        )
        return _attach_resource_ledger(
            row,
            resource_observer,
            resource_ledger_context,
            terminal_status=terminal_status,
        )


def _attach_resource_ledger(
    row: dict[str, Any],
    observer: ResourceLedgerObserver | None,
    context: Mapping[str, Any] | None,
    *,
    terminal_status: Literal["succeeded", "failed", "cancelled", "not_started"],
) -> dict[str, Any]:
    if observer is None or context is None:
        return row
    ledger = observer.build_query_ledger(
        run_identity=str(context["run_identity"]),
        query_identity=str(context["query_identity"]),
        attempt_identity=str(context["attempt_identity"]),
        checkpoint_generation=int(context["checkpoint_generation"]),
        manifest_identity=str(context["manifest_identity"]),
        terminal_status=terminal_status,
    )
    row["resource_ledger"] = ledger.model_dump(mode="json")
    superseded = context.get("superseded_attempt_identity")
    if superseded is not None:
        row["resource_ledger_superseded_attempts"] = [str(superseded)]
    return row


def _evaluate_rows(
    rows: list[dict[str, Any]],
    queries: list[EvalQuery],
    result_policy: ResultPolicy,
) -> dict[str, Any]:
    gold_rows = [
        {
            "case_id": query.query_id,
            "relevant_papers": [
                paper.model_dump(mode="json") for paper in query.gold_papers
            ],
        }
        for query in queries
    ]
    metrics = evaluate_batch_results(
        rows,
        gold_rows,
        k_values=[5, 10, 20],
        result_policy=result_policy,
    )
    statistics = metrics["case_statistics"]
    efficiency = metrics["efficiency"]
    case_count = max(1, int(efficiency.get("case_count") or 0))
    failures = [row for row in rows if row.get("status") != "succeeded"]
    metrics["benchmark_statistics"] = {
        "success_rate": statistics["success_rate"],
        "failed_case_rate": statistics["failed_case_rate"],
        "missing_result_rate": statistics["missing_result_rate"],
        "average_api_calls": efficiency["avg_api_call_count"],
        "average_llm_calls": efficiency["avg_llm_call_count"],
        "average_tokens": efficiency["avg_llm_total_tokens"],
        "average_latency_seconds": efficiency["average_latency_seconds"],
        "average_candidate_count": (
            efficiency["total_deduplicated_count"] / case_count
        ),
        "average_final_result_count": (
            efficiency["total_returned_result_count"] / case_count
        ),
        "failure_reason_distribution": dict(
            sorted(Counter(str(row.get("error_type") or "Unknown") for row in failures).items())
        ),
    }
    return metrics


def _write_result_artifacts(
    run_dir: Path,
    selected_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    ordered = [rows_by_id[item] for item in selected_ids if item in rows_by_id]
    _atomic_write_jsonl(run_dir / "results.jsonl", ordered)
    _write_failures(run_dir / "failures.jsonl", ordered)


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [
        {
            "case_id": row["case_id"],
            "query": row["query"],
            "status": row["status"],
            "error_type": row.get("error_type") or "Unknown",
            "error_message": row.get("error") or "",
        }
        for row in rows
        if row.get("status") != "succeeded"
    ]
    _atomic_write_jsonl(path, failures)


def _summary_markdown(
    config: dict[str, Any],
    metrics: dict[str, Any],
    stage_metrics: dict[str, Any] | None = None,
) -> str:
    stats = metrics["case_statistics"]
    efficiency = metrics["benchmark_statistics"]
    lines = [
        "# Benchmark 基线汇总",
        "",
        f"- 数据集：`{config['dataset']}`",
        f"- 案例数：{config['case_count']}",
        f"- 成功率：{stats['success_rate']:.3f}",
        f"- 失败率：{stats['failed_case_rate']:.3f}",
        f"- 平均 API 调用：{efficiency['average_api_calls']:.3f}",
        f"- 平均 LLM 调用：{efficiency['average_llm_calls']:.3f}",
        f"- 平均 Token：{efficiency['average_tokens']:.3f}",
        f"- 平均延迟：{efficiency['average_latency_seconds']:.3f} 秒",
        "",
        "| 口径 | F1@5 | F1@10 | F1@20 | MRR | nDCG@5 | nDCG@10 | nDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_metric_row("仅成功案例", metrics["success_only_metrics"]),
        _summary_metric_row("端到端", metrics["end_to_end_metrics"]),
        "",
    ]
    if stage_metrics is not None:
        judgement = stage_metrics.get("judgement", {})
        reranking = stage_metrics.get("reranking", {})
        lines.extend(
            [
                "## 阶段诊断",
                "",
                "| 初始候选 Recall | 最终返回 Recall@20 | Judgement FN 率 | 平均 gold rank | 瓶颈标签 |",
                "| ---: | ---: | ---: | ---: | --- |",
                (
                    f"| {_format_optional(stage_metrics.get('initial_retrieval_recall'))} "
                    "| "
                    f"{_format_optional((stage_metrics.get('final_returned_recall') or {}).get('20'))} "
                    f"| {float(judgement.get('gold_false_negative_rate') or 0.0):.3f} "
                    f"| {_format_optional(reranking.get('average_gold_rank'))} "
                    f"| {', '.join(stage_metrics.get('bottleneck_labels') or []) or '-'} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "> 小规模 smoke 只验证真实 Benchmark 运行链路，不代表最终比赛成绩或完整 Benchmark 性能。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "-"


def _summary_metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {_at_k(metrics, 'f1_at_k', 5):.3f} | "
        f"{_at_k(metrics, 'f1_at_k', 10):.3f} | "
        f"{_at_k(metrics, 'f1_at_k', 20):.3f} | "
        f"{float(metrics.get('mrr') or 0.0):.3f} | "
        f"{_at_k(metrics, 'ndcg_at_k', 5):.3f} | "
        f"{_at_k(metrics, 'ndcg_at_k', 10):.3f} | "
        f"{_at_k(metrics, 'ndcg_at_k', 20):.3f} |"
    )


def _at_k(metrics: dict[str, Any], name: str, k: int) -> float:
    values = metrics.get(name) or {}
    return float(values.get(str(k), values.get(k, 0.0)))


def _sanitize_message(message: str) -> str:
    sanitized = message
    for env_name in _SENSITIVE_ENV_NAMES:
        secret = os.getenv(env_name)
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = re.sub(
        r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)[^\s&,;]+",
        r"\1\2[REDACTED]",
        sanitized,
    )
    return sanitized[:1000]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSONL object at line {line_number}: {path}")
        rows.append((line_number, payload))
    return rows


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    durable_atomic_write_text(path, text)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _parse_sources(value: str) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    supported = set(SUPPORTED_SEARCH_SOURCES)
    for raw in value.split(","):
        source = raw.strip().lower()
        if not source or source in seen:
            continue
        if source not in supported:
            raise ValueError(f"unsupported source: {source}")
        seen.add(source)
        sources.append(source)
    if not sources:
        raise ValueError("--sources must contain at least one supported source")
    return sources


def _reranker_device_argument(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?)", normalized):
        return normalized
    raise argparse.ArgumentTypeError(
        "reranker device must be auto, cpu, cuda, or cuda:<index>"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行公开学术检索 Benchmark。")
    parser.add_argument("--dataset", required=True, choices=supported_datasets())
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-split", default="development")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/benchmark_runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--run-profile",
        choices=["fast", "balanced", "high_recall", "evaluation"],
        default="balanced",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SEARCH_SOURCES),
    )
    parser.add_argument("--local-bm25-corpus", default=None)
    parser.add_argument(
        "--local-bm25-cache-dir",
        default="outputs/benchmark_cache/local_bm25",
    )
    parser.add_argument("--local-bm25-document-id-field", default="_id")
    parser.add_argument("--local-bm25-title-field", default="title")
    parser.add_argument("--local-bm25-abstract-field", default="abstract")
    parser.add_argument(
        "--local-bm25-document-id-identity",
        choices=[
            "doi",
            "arxiv_id",
            "semantic_scholar_id",
            "s2orc_corpus_id",
            "openalex_id",
            "pubmed_id",
        ],
        default="s2orc_corpus_id",
    )
    parser.add_argument("--local-bm25-doi-field", default=None)
    parser.add_argument("--local-bm25-arxiv-id-field", default=None)
    parser.add_argument("--local-bm25-semantic-scholar-id-field", default=None)
    parser.add_argument("--local-bm25-s2orc-corpus-id-field", default=None)
    parser.add_argument("--local-bm25-openalex-id-field", default=None)
    parser.add_argument("--local-bm25-pubmed-id-field", default=None)
    parser.add_argument("--local-hybrid-semantic-corpus", default=None)
    parser.add_argument(
        "--local-hybrid-index-dir",
        default="outputs/benchmark_cache/local_hybrid",
    )
    parser.add_argument("--local-hybrid-model", default=None)
    parser.add_argument("--local-hybrid-reranker-model", default=None)
    parser.add_argument(
        "--local-hybrid-reranker-candidate-limit", type=int, default=120
    )
    parser.add_argument("--local-hybrid-reranker-batch-size", type=int, default=8)
    parser.add_argument(
        "--local-hybrid-reranker-device",
        type=_reranker_device_argument,
        default="auto",
    )
    parser.add_argument(
        "--local-hybrid-bm25-candidate-limit",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--local-hybrid-semantic-candidate-limit",
        type=int,
        default=60,
    )
    parser.add_argument("--local-hybrid-rrf-k", type=int, default=60)
    parser.add_argument(
        "--result-policy",
        choices=["highly_only", "highly_and_partial"],
        default="highly_and_partial",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-query-evolution", action="store_true")
    parser.add_argument(
        "--query-evolution-policy",
        choices=["off", "seed_expansion", "coverage_gap", "llm_feedback"],
        default="coverage_gap",
    )
    parser.add_argument(
        "--query-planning-policy",
        choices=[
            "current_rules",
            "prf_v1",
            "concept_projection",
            "controlled_relaxation",
            "disjunctive_facets",
            "current_plus_disjunctive",
            "facet_union",
            "facet_balanced",
            "llm_semantic",
            "llm_constrained_rewrite",
        ],
        default="current_rules",
    )
    parser.add_argument(
        "--judgement-policy",
        choices=["current_rules", "calibrated_rules_v1"],
        default="current_rules",
    )
    parser.add_argument(
        "--ranking-policy",
        choices=["current_rules", "rrf_fusion", "quality_soft_v1"],
        default="current_rules",
    )
    parser.add_argument(
        "--quality-evidence-ledger",
        default=None,
        help=(
            "strict paper-quality-evidence-ledger-v1 JSONL; only accepted "
            "with --ranking-policy quality_soft_v1"
        ),
    )
    parser.add_argument(
        "--quality-evidence-candidate-identifiers",
        default=None,
        help="canonical arxiv: IDs exported from a completed baseline run",
    )
    parser.add_argument(
        "--quality-evidence-candidate-report",
        default=None,
        help="matching quality-evidence-candidate-export-v1 report",
    )
    parser.add_argument("--judgement-config", default=None)
    parser.add_argument("--enable-refchain", action="store_true")
    parser.add_argument("--enable-semantic-seed-expansion", action="store_true")
    parser.add_argument("--enable-llm-query-understanding", action="store_true")
    parser.add_argument("--enable-llm-judgement", action="store_true")
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-search-rounds", type=int, default=None)
    parser.add_argument("--max-candidate-papers", type=int, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-latency-seconds", type=float, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--resource-ledger",
        action="store_true",
        help=(
            "write optional resource_ledger_v1 from existing connector and "
            "budget signals; does not change execution budgets"
        ),
    )
    parser.add_argument(
        "--query-adapter-policy",
        choices=["safe_original", "hybrid", "adaptive"],
        default="adaptive",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retrieval-mode",
        choices=["live", "record", "replay", "record-missing", "plan"],
        default="live",
    )
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument(
        "--llm-mode",
        choices=["live", "record", "replay", "record-missing"],
        default="live",
    )
    parser.add_argument("--llm-snapshot-dir", default=None)
    parser.add_argument("--plan-round", type=int, default=1)
    parser.add_argument("--retry-failed-snapshots", action="store_true")
    parser.add_argument("--overwrite-snapshots", action="store_true")
    parser.add_argument(
        "--comparison-plan",
        default=None,
        help="pre-bind this run to a comparison_plan_v1 before generation zero",
    )
    parser.add_argument(
        "--comparison-role",
        choices=["baseline", "candidate"],
        default=None,
    )
    parser.add_argument(
        "--shard-plan",
        default=None,
        help="pre-bind this run to a shard_plan_v1 before generation zero",
    )
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-attempt-id", default=None)
    parser.add_argument("--shard-supersedes-attempt-id", default=None)
    parser.add_argument(
        "--resume-manifest",
        type=Path,
        default=None,
        help=(
            "execute only the immutable missing-key schedule; bypasses dataset "
            "loading, SearchService, ranking and evaluation"
        ),
    )
    parser.add_argument(
        "--resume-manifest-dry-run",
        action="store_true",
        help="validate and recompute resume progress without network or writes",
    )
    return parser


def _resume_runtime_config(args: argparse.Namespace) -> ResumeRuntimeConfig:
    default_budget = SearchBudget()
    return ResumeRuntimeConfig(
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        offset=args.offset,
        limit=args.limit,
        run_profile=args.run_profile,
        sources=_parse_sources(args.sources),
        result_policy=args.result_policy,
        top_k=args.top_k,
        query_adapter_policy=args.query_adapter_policy,
        query_planning_policy=args.query_planning_policy,
        ranking_policy=args.ranking_policy,
        judgement_policy=args.judgement_policy,
        enable_query_evolution=args.enable_query_evolution,
        query_evolution_policy=args.query_evolution_policy,
        enable_refchain=args.enable_refchain,
        enable_semantic_seed_expansion=args.enable_semantic_seed_expansion,
        enable_llm_query_understanding=args.enable_llm_query_understanding,
        enable_llm_judgement=args.enable_llm_judgement,
        current_year=args.current_year,
        budgets={
            "max_search_rounds": (
                default_budget.max_search_rounds
                if args.max_search_rounds is None
                else args.max_search_rounds
            ),
            "max_candidate_papers": (
                default_budget.max_candidate_papers
                if args.max_candidate_papers is None
                else args.max_candidate_papers
            ),
            "max_llm_calls": (
                default_budget.max_llm_calls
                if args.max_llm_calls is None
                else args.max_llm_calls
            ),
            "max_total_tokens": (
                default_budget.max_total_tokens
                if args.max_total_tokens is None
                else args.max_total_tokens
            ),
            "max_latency_seconds": (
                default_budget.max_latency_seconds
                if args.max_latency_seconds is None
                else args.max_latency_seconds
            ),
        },
    )


def _live_resume_executor(request: ResumeRequest):
    from scholar_agent.connectors import (  # noqa: PLC0415
        search_arxiv_detailed,
        search_local_bm25_detailed,
        search_local_hybrid_detailed,
        search_openalex_detailed,
        search_pubmed_detailed,
        search_semantic_scholar_detailed,
    )

    registry = {
        "openalex": search_openalex_detailed,
        "arxiv": search_arxiv_detailed,
        "semantic_scholar": search_semantic_scholar_detailed,
        "pubmed": search_pubmed_detailed,
        "local_bm25": search_local_bm25_detailed,
        "local_hybrid": search_local_hybrid_detailed,
    }
    try:
        search = registry[request.source]
    except KeyError as exc:
        raise SnapshotResumeError(
            f"unsupported resume source:{request.source}"
        ) from exc
    return search(request.adapted_query, request.limit)


def _run_resume_manifest_cli(args: argparse.Namespace) -> int:
    if args.resume_manifest_dry_run and args.resume_manifest is None:
        raise SnapshotResumeError("--resume-manifest-dry-run requires --resume-manifest")
    if args.retrieval_mode != "record-missing":
        raise SnapshotResumeError("resume manifest requires --retrieval-mode record-missing")
    if args.snapshot_dir is None:
        raise SnapshotResumeError("resume manifest requires --snapshot-dir")
    if args.limit is None:
        raise SnapshotResumeError("resume manifest requires explicit --limit")
    manifest = load_resume_manifest(args.resume_manifest)
    validate_manifest_required_plan(manifest, repository_root=REPO_ROOT)
    runtime_config = _resume_runtime_config(args)
    validate_runtime_config(manifest, runtime_config)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    expected_snapshot_dir = (REPO_ROOT / manifest.snapshot_dir).resolve()
    if snapshot_dir != expected_snapshot_dir:
        raise SnapshotResumeError("resume snapshot directory drift")
    if not args.resume_manifest_dry_run:
        load_project_env(REPO_ROOT)
    report = execute_resume_manifest(
        manifest,
        SnapshotStore(snapshot_dir),
        executor=None if args.resume_manifest_dry_run else _live_resume_executor,
        dry_run=args.resume_manifest_dry_run,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume_manifest is not None or args.resume_manifest_dry_run:
        try:
            return _run_resume_manifest_cli(args)
        except (SnapshotResumeError, ValueError, OSError) as exc:
            print(_sanitize_message(str(exc)), file=sys.stderr)
            return 1
    load_project_env(REPO_ROOT)
    try:
        sources = _parse_sources(args.sources)
        default_budget = SearchBudget()
        budgets = SearchBudget(
            max_search_rounds=(
                default_budget.max_search_rounds
                if args.max_search_rounds is None
                else args.max_search_rounds
            ),
            max_candidate_papers=(
                default_budget.max_candidate_papers
                if args.max_candidate_papers is None
                else args.max_candidate_papers
            ),
            max_llm_calls=(
                default_budget.max_llm_calls
                if args.max_llm_calls is None
                else args.max_llm_calls
            ),
            max_total_tokens=(
                default_budget.max_total_tokens
                if args.max_total_tokens is None
                else args.max_total_tokens
            ),
            max_latency_seconds=(
                default_budget.max_latency_seconds
                if args.max_latency_seconds is None
                else args.max_latency_seconds
            ),
        )
        options = BenchmarkRunOptions(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            limit=args.limit,
            offset=args.offset,
            output_root=args.output_root,
            run_id=args.run_id,
            run_profile=args.run_profile,
            sources=sources,
            local_bm25_config=(
                LocalBM25Config(
                    corpus_path=Path(args.local_bm25_corpus),
                    cache_dir=Path(args.local_bm25_cache_dir),
                    fields=LocalBM25FieldConfig(
                        document_id=args.local_bm25_document_id_field,
                        title=args.local_bm25_title_field,
                        abstract=args.local_bm25_abstract_field,
                        document_id_identity=args.local_bm25_document_id_identity,
                        doi=args.local_bm25_doi_field,
                        arxiv_id=args.local_bm25_arxiv_id_field,
                        semantic_scholar_id=(
                            args.local_bm25_semantic_scholar_id_field
                        ),
                        s2orc_corpus_id=args.local_bm25_s2orc_corpus_id_field,
                        openalex_id=args.local_bm25_openalex_id_field,
                        pubmed_id=args.local_bm25_pubmed_id_field,
                    ),
                )
                if "local_bm25" in sources and args.local_bm25_corpus
                else None
            ),
            local_hybrid_config=(
                LocalHybridConfig(
                    bm25_config=LocalBM25Config(
                        corpus_path=Path(args.local_bm25_corpus),
                        cache_dir=Path(args.local_bm25_cache_dir),
                        fields=LocalBM25FieldConfig(
                            document_id=args.local_bm25_document_id_field,
                            title=args.local_bm25_title_field,
                            abstract=args.local_bm25_abstract_field,
                            document_id_identity=args.local_bm25_document_id_identity,
                            doi=args.local_bm25_doi_field,
                            arxiv_id=args.local_bm25_arxiv_id_field,
                            semantic_scholar_id=(
                                args.local_bm25_semantic_scholar_id_field
                            ),
                            s2orc_corpus_id=args.local_bm25_s2orc_corpus_id_field,
                            openalex_id=args.local_bm25_openalex_id_field,
                            pubmed_id=args.local_bm25_pubmed_id_field,
                        ),
                    ),
                    semantic_corpus_path=Path(args.local_hybrid_semantic_corpus),
                    semantic_index_dir=Path(args.local_hybrid_index_dir),
                    model_path=Path(args.local_hybrid_model),
                    reranker_model_path=(
                        Path(args.local_hybrid_reranker_model)
                        if args.local_hybrid_reranker_model
                        else None
                    ),
                    reranker_candidate_limit=args.local_hybrid_reranker_candidate_limit,
                    reranker_batch_size=args.local_hybrid_reranker_batch_size,
                    reranker_device=args.local_hybrid_reranker_device,
                    bm25_candidate_limit=args.local_hybrid_bm25_candidate_limit,
                    semantic_candidate_limit=(
                        args.local_hybrid_semantic_candidate_limit
                    ),
                    rrf_k=args.local_hybrid_rrf_k,
                )
                if (
                    "local_hybrid" in sources
                    and args.local_bm25_corpus
                    and args.local_hybrid_semantic_corpus
                    and args.local_hybrid_model
                )
                else None
            ),
            result_policy=args.result_policy,
            top_k=args.top_k,
            enable_query_evolution=args.enable_query_evolution,
            query_evolution_policy=args.query_evolution_policy,
            query_planning_policy=args.query_planning_policy,
            ranking_policy=args.ranking_policy,
            quality_evidence_ledger_path=(
                Path(args.quality_evidence_ledger)
                if args.quality_evidence_ledger
                else None
            ),
            quality_evidence_candidate_identifiers_path=(
                Path(args.quality_evidence_candidate_identifiers)
                if args.quality_evidence_candidate_identifiers
                else None
            ),
            quality_evidence_candidate_report_path=(
                Path(args.quality_evidence_candidate_report)
                if args.quality_evidence_candidate_report
                else None
            ),
            judgement_policy=args.judgement_policy,
            judgement_config_path=(
                Path(args.judgement_config) if args.judgement_config else None
            ),
            enable_refchain=args.enable_refchain,
            enable_semantic_seed_expansion=args.enable_semantic_seed_expansion,
            enable_llm_query_understanding=args.enable_llm_query_understanding,
            enable_llm_judgement=args.enable_llm_judgement,
            current_year=args.current_year,
            max_workers=args.max_workers,
            budgets=budgets,
            diagnostics=args.diagnostics,
            enable_resource_ledger=args.resource_ledger,
            query_adapter_policy=args.query_adapter_policy,
            resume=args.resume,
            retrieval_mode=args.retrieval_mode,
            snapshot_dir=(Path(args.snapshot_dir) if args.snapshot_dir else None),
            llm_mode=args.llm_mode,
            llm_snapshot_dir=(
                Path(args.llm_snapshot_dir) if args.llm_snapshot_dir else None
            ),
            plan_round=args.plan_round,
            retry_failed_snapshots=args.retry_failed_snapshots,
            overwrite_snapshots=args.overwrite_snapshots,
            comparison_plan_path=(
                Path(args.comparison_plan) if args.comparison_plan else None
            ),
            comparison_role=args.comparison_role,
            shard_plan_path=(Path(args.shard_plan) if args.shard_plan else None),
            shard_index=args.shard_index,
            shard_attempt_id=args.shard_attempt_id,
            shard_supersedes_attempt_id=args.shard_supersedes_attempt_id,
        )
        result = run_benchmark(options)
    except (ValueError, OSError) as exc:
        print(_sanitize_message(str(exc)), file=sys.stderr)
        return 1
    print(result.run_dir)
    if args.retrieval_mode == "replay" and any(
        row.get("status") != "succeeded" for row in result.result_rows
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
