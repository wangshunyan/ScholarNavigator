"""Benchmark 检索与引用响应快照 Schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    QUERY_PLANNER_VERSION,
    QueryPlanningPolicy,
)


SNAPSHOT_SCHEMA_VERSION = "1"
QUERY_ADAPTER_VERSION = "1"
CONNECTOR_VERSIONS = {
    "arxiv": "search-v1",
    "openalex": "search-v1",
    "semantic_scholar": "search-v2",
    "semantic_scholar_recommendations": "recommendations-v1",
    "semantic_scholar_paper_batch": "paper-batch-v1",
    "pubmed": "search-v1",
    "local_hybrid": "local-hybrid-v1",
    "openalex_references": "references-v1",
}
SnapshotEntryStatus = Literal["success", "failed"]
SnapshotEntryType = Literal["llm_planning", "retrieval", "reference"]
SnapshotGeneratedBy = Literal[
    "llm_query_planning",
    "initial_retrieval",
    "query_evolution",
    "refchain",
    "semantic_seed_expansion",
    "semantic_seed_resolution",
]


class RetrievalSnapshotEntry(BaseModel):
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    key: str = Field(min_length=64, max_length=64)
    source: str
    adapted_query: str
    normalized_query: str
    limit: int = Field(ge=0)
    adapter_policy: str
    query_adapter_version: str = QUERY_ADAPTER_VERSION
    connector_version: str
    status: SnapshotEntryStatus
    papers: list[Paper] = Field(default_factory=list)
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)


class ReferenceSnapshotEntry(BaseModel):
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    key: str = Field(min_length=64, max_length=64)
    source: str = "openalex"
    seed_identifier: str
    seed_identifiers: list[str] = Field(default_factory=list)
    limit: int = Field(ge=0)
    connector_version: str
    status: SnapshotEntryStatus
    papers: list[Paper] = Field(default_factory=list)
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)
    reference_batch_status: Literal[
        "success", "partial_success", "missing_id", "failed"
    ] | None = None
    missing_reference_ids: list[str] = Field(default_factory=list)
    reference_batch_count: int = Field(default=0, ge=0)
    supplemental_request_count: int = Field(default=0, ge=0)


class SnapshotGroupObservation(BaseModel):
    query_planning_policy: QueryPlanningPolicy | None = None
    query_planner_version: str | None = None
    judgement_policy: Literal["current_rules", "calibrated_rules_v1"] = (
        "current_rules"
    )
    judgement_config_hash: str | None = None
    query_evolution_policy: Literal[
        "off", "seed_expansion", "coverage_gap", "llm_feedback"
    ] | None = None
    retrieval_keys: list[str] = Field(default_factory=list)
    reference_keys: list[str] = Field(default_factory=list)
    missing_retrieval_keys: list[str] = Field(default_factory=list)
    missing_reference_keys: list[str] = Field(default_factory=list)
    collection_started: bool = False
    collection_completed: bool = False
    replay_ready: bool = False
    replay_verified: bool = False
    required_key_count: int = Field(default=0, ge=0)
    success_key_count: int = Field(default=0, ge=0)
    failed_key_count: int = Field(default=0, ge=0)
    missing_key_count: int = Field(default=0, ge=0)
    last_plan_round: int = Field(default=0, ge=0)
    plan_rounds: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    completed: bool = False
    updated_at: str


class SnapshotPlanEntry(BaseModel):
    key: str = Field(min_length=64, max_length=64)
    entry_type: SnapshotEntryType
    source: str
    adapted_query: str | None = None
    seed_identifier: str | None = None
    seed_identifiers: list[str] = Field(default_factory=list)
    limit: int = Field(ge=0)
    adapter_policy: str | None = None
    connector_version: str
    required_by_group: str
    case_id: str
    stage: str
    origin_subquery: str | None = None
    generated_by: SnapshotGeneratedBy
    query_evolution_policy: Literal[
        "off", "seed_expansion", "coverage_gap", "llm_feedback"
    ] | None = None
    query_planning_policy: QueryPlanningPolicy | None = None
    query_planner_version: str | None = None
    dependency_keys: list[str] = Field(default_factory=list)
    priority: int = Field(ge=1)
    already_present: bool = False
    existing_status: SnapshotEntryStatus | None = None
    llm_request: dict[str, object] | None = None


class SnapshotPlanRound(BaseModel):
    snapshot_name: str
    group: str
    round_index: int = Field(ge=1)
    entries: list[SnapshotPlanEntry] = Field(default_factory=list)
    missing_retrieval_count: int = Field(default=0, ge=0)
    missing_reference_count: int = Field(default=0, ge=0)
    missing_llm_planning_count: int = Field(default=0, ge=0)
    network_request_count: int = Field(default=0, ge=0)
    converged: bool = False
    stop_reason: str | None = None
    created_at: str


class SnapshotManifest(BaseModel):
    snapshot_name: str
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    dataset: str
    split: str
    offset: int = Field(ge=0)
    limit: int | None = Field(default=None, ge=1)
    sources: list[str]
    adapter_policy: str
    query_adapter_version: str = QUERY_ADAPTER_VERSION
    query_planner_version: str = QUERY_PLANNER_VERSION
    run_profile: str
    budgets: dict[str, object]
    llm_enabled: bool
    query_understanding_prompt: dict[str, str | int | None]
    llm_query_planning_prompt: dict[str, str | int | None] = Field(default_factory=dict)
    judgement_prompt: dict[str, str | int | None]
    connector_versions: dict[str, str]
    code_hash: str
    git_commit: str | None = None
    dirty_worktree: bool
    retrieval_entry_count: int = Field(default=0, ge=0)
    reference_entry_count: int = Field(default=0, ge=0)
    groups: dict[str, SnapshotGroupObservation] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class SnapshotCostReport(BaseModel):
    mode: str
    retrieval_snapshot_hits: int = Field(default=0, ge=0)
    reference_snapshot_hits: int = Field(default=0, ge=0)
    retrieval_snapshot_writes: int = Field(default=0, ge=0)
    reference_snapshot_writes: int = Field(default=0, ge=0)
    missing_retrieval_keys: list[str] = Field(default_factory=list)
    missing_reference_keys: list[str] = Field(default_factory=list)
    fatal_errors: list[str] = Field(default_factory=list)
    observed_retrieval_keys: list[str] = Field(default_factory=list)
    observed_reference_keys: list[str] = Field(default_factory=list)
    replay_execution_request_count: int = Field(default=0, ge=0)
    replay_execution_retry_count: int = Field(default=0, ge=0)
    replay_execution_network_wait_seconds: float = Field(default=0.0, ge=0.0)
    recorded_search_request_count: int = Field(default=0, ge=0)
    recorded_reference_request_count: int = Field(default=0, ge=0)
    recorded_retry_count: int = Field(default=0, ge=0)
    recorded_error_count: int = Field(default=0, ge=0)
    recorded_rate_limit_wait_seconds: float = Field(default=0.0, ge=0.0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
