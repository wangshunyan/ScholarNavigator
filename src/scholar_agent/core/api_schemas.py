"""Pydantic schemas for the ScholarNavigator FastAPI API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.search_schemas import (
    BudgetStatus,
    JudgementPolicy,
    PaperType,
    QueryEvolutionPolicy,
    QueryPlanningPolicy,
    QueryPlanningResult,
    RankingPolicy,
    SearchBudget,
    SourceName,
    TimeRange,
    normalize_paper_types,
    normalize_search_sources,
)


RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RunProfile = Literal["fast", "balanced", "high_recall", "evaluation"]
RelevanceCategory = Literal[
    "highly_relevant",
    "partially_relevant",
    "weakly_relevant",
    "irrelevant",
    "insufficient_evidence",
]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    time: datetime


class LLMRuntimeConfig(BaseModel):
    provider: str
    model: str | None = None
    available: bool
    base_url_host: str | None = None
    reason: str | None = None


class ConnectorRuntimeConfig(BaseModel):
    name: str
    available: bool
    requires_key: bool
    reason: str | None = None


class RuntimeLimits(BaseModel):
    max_top_k: int
    max_search_rounds: int
    max_candidate_papers: int
    max_latency_seconds: int
    real_search_max_workers: int = 2
    real_search_background_workers: int = 2
    real_search_run_ttl_seconds: int = 3600
    real_search_max_stored_runs: int = 200


class RuntimeFeatures(BaseModel):
    query_evolution: bool
    refchain: bool
    evaluation: bool
    sse: bool
    real_search: bool = False
    real_search_cancel: bool = False
    real_search_sse: bool = False
    retrieval_cache: bool = False
    batch_cli: bool = False
    llm_query_understanding: bool = False
    llm_judgement: bool = False


class RuntimeConfigResponse(BaseModel):
    mode: str = "real_search"
    llm: LLMRuntimeConfig
    connectors: list[ConnectorRuntimeConfig]
    limits: RuntimeLimits
    features: RuntimeFeatures


TimeRangeConstraint = TimeRange


class SearchConstraints(BaseModel):
    time_range: TimeRangeConstraint | None = None
    venues: list[str] = Field(default_factory=list)
    must_have_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    paper_types: list[PaperType] = Field(default_factory=list)

    @field_validator("paper_types", mode="before")
    @classmethod
    def normalize_requested_paper_types(cls, value: object) -> list[str]:
        return normalize_paper_types(value)


class SearchBudgets(SearchBudget):
    """Public request shape backed by the shared internal budget defaults."""


class SearchOptions(BaseModel):
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    ranking_policy: RankingPolicy = "current_rules"
    judgement_policy: JudgementPolicy = "current_rules"
    enable_query_evolution: bool = False
    query_evolution_policy: QueryEvolutionPolicy = "coverage_gap"
    enable_refchain: bool = True
    enable_semantic_seed_expansion: bool = False
    enable_llm_query_understanding: bool | None = None
    enable_llm_judgement: bool | None = None
    refchain_depth: int = 1
    return_markdown: bool = True
    return_json: bool = True
    stream_events: bool = True


class SearchRunCreateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    locale: str = "zh-CN"
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    source_preferences: list[SourceName] | None = None
    run_profile: RunProfile = "balanced"
    top_k: int = Field(default=20, ge=1, le=100)
    budgets: SearchBudgets = Field(default_factory=SearchBudgets)
    options: SearchOptions = Field(default_factory=SearchOptions)

    @field_validator("source_preferences", mode="before")
    @classmethod
    def validate_source_preferences(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        normalized = normalize_search_sources(value)
        if not normalized:
            raise ValueError("source_preferences must be a non-empty list")
        return normalized


class SearchRunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: datetime
    links: dict[str, str]


class RunProgress(BaseModel):
    completed_stages: list[str] = Field(default_factory=list)
    skipped_stages: list[str] = Field(default_factory=list)
    candidate_paper_count: int = 0
    judged_paper_count: int = 0


class CostReport(BaseModel):
    api_call_count: int = 0
    logical_search_call_count: int = 0
    search_api_call_count: int = 0
    reference_api_call_count: int = 0
    retry_count: int = 0
    error_count: int = 0
    llm_call_count: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    latency_seconds: float = 0.0
    cache_hit_count: int = 0
    rate_limit_wait_seconds: float = 0.0
    search_rounds: int = 0
    judged_paper_count: int = 0
    raw_candidate_count: int = 0
    deduplicated_candidate_count: int = 0


class SearchRunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    current_stage: str
    progress: RunProgress
    cost_report: CostReport
    created_at: datetime
    updated_at: datetime


class PaperIdentifiers(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    s2orc_corpus_id: str | None = None
    openalex_id: str | None = None
    pubmed_id: str | None = None


class PaperUrls(BaseModel):
    landing_page: str | None = None
    pdf: str | None = None


class Paper(BaseModel):
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    abstract: str
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    urls: PaperUrls = Field(default_factory=PaperUrls)
    sources: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    source: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesisEvidenceRow(BaseModel):
    row_id: str
    citation_key: str
    rank: int
    paper_title: str
    year: int | None = None
    venue: str | None = None
    sources: list[str] = Field(default_factory=list)
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    category: str
    final_score: float = Field(ge=0.0, le=1.0)
    evidence_source: str
    evidence_text: str
    supported_terms: list[str] = Field(default_factory=list)
    supported_claim: str


class SynthesisFinding(BaseModel):
    text: str
    citation_keys: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_row_ids: list[str] = Field(default_factory=list)


class CitationCoverage(BaseModel):
    ranked_paper_count: int = 0
    cited_paper_count: int = 0
    evidence_row_count: int = 0
    cited_evidence_row_count: int = 0
    missing_evidence_count: int = 0
    source_error_count: int = 0
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class SynthesisOutput(BaseModel):
    answer_summary: str
    status: str = "succeeded"
    key_findings: list[SynthesisFinding] = Field(default_factory=list)
    evidence_table: list[SynthesisEvidenceRow] = Field(default_factory=list)
    citation_coverage: CitationCoverage = Field(default_factory=CitationCoverage)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RRFListContribution(BaseModel):
    source: str
    subquery: str
    rank: int = Field(ge=1)
    reciprocal_score: float = Field(ge=0.0)


class RankedPaper(BaseModel):
    result_identity: str
    authority_digest: str
    rank: int
    paper: Paper
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: RelevanceCategory
    matched_constraints: list[str] = Field(default_factory=list)
    ranking_reason: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    rrf_score: float | None = Field(default=None, ge=0.0)
    rrf_contributions: list[RRFListContribution] = Field(default_factory=list)
    original_rank: int | None = Field(default=None, ge=1)
    rrf_top_20_change: str | None = None
    rrf_rank_change_reason: str | None = None


class QueryAnalysis(BaseModel):
    intent_type: str
    domain: str
    research_topics: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class SearchPlan(BaseModel):
    expanded_queries: list[str] = Field(default_factory=list)
    source_preferences: list[str] = Field(default_factory=list)
    max_rounds: int = 1
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    ranking_policy: RankingPolicy = "current_rules"
    query_planning: QueryPlanningResult = Field(
        default_factory=QueryPlanningResult
    )
    query_evolution_policy: QueryEvolutionPolicy = "off"
    enable_semantic_seed_expansion: bool = False


class MethodCluster(BaseModel):
    name: str
    paper_ranks: list[int] = Field(default_factory=list)
    summary: str


class TimelineItem(BaseModel):
    year: int
    paper_ranks: list[int] = Field(default_factory=list)
    summary: str


class CitationGraphNode(BaseModel):
    id: str
    label: str
    rank: int | None = None


class CitationGraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "cites"


class CitationGraph(BaseModel):
    nodes: list[CitationGraphNode] = Field(default_factory=list)
    edges: list[CitationGraphEdge] = Field(default_factory=list)


class RetrievalSourceStats(BaseModel):
    source: str
    returned_count: int = 0
    latency_seconds: float = 0.0
    cache_hit: bool = False
    logical_call_executed: bool = True
    adaptation_strategy: str | None = None
    triggered_by: list[str] = Field(default_factory=list)
    safe_original_candidate_count: int | None = None
    safe_original_core_term_coverage: float | None = None
    safe_original_constraint_coverage: float | None = None
    sufficiency_reasons: list[str] = Field(default_factory=list)
    compact_query_executed: bool | None = None
    compact_query_skipped_reason: str | None = None
    error_message: str | None = None
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)


class RetrievalDiagnostics(BaseModel):
    raw_count: int = 0
    deduplicated_count: int = 0
    source_stats: list[RetrievalSourceStats] = Field(default_factory=list)


class SearchRunResultResponse(BaseModel):
    run_id: str
    status: RunStatus
    partial: bool
    query_analysis: QueryAnalysis
    search_plan: SearchPlan
    highly_relevant_papers: list[RankedPaper]
    partially_relevant_papers: list[RankedPaper]
    method_clusters: list[MethodCluster]
    timeline: list[TimelineItem]
    citation_graph: CitationGraph = Field(default_factory=CitationGraph)
    warnings: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    synthesis: SynthesisOutput | None = None
    retrieval_diagnostics: RetrievalDiagnostics = Field(
        default_factory=RetrievalDiagnostics
    )
    budget_status: BudgetStatus = Field(default_factory=BudgetStatus)
    cost_report: CostReport
    judgement_policy: JudgementPolicy = "current_rules"
    judgement_config_hash: str = ""
