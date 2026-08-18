"""Schemas for offline evaluation fixtures and metric summaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalGoldPaper(BaseModel):
    """Gold relevance record for one expected paper."""

    title: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2200)
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    s2orc_corpus_id: str | int | None = None
    pubmed_id: str | None = None
    relevance_grade: float = Field(default=1.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalQuery(BaseModel):
    """One offline search evaluation case."""

    query_id: str
    query: str = Field(..., min_length=1)
    gold_papers: list[EvalGoldPaper] = Field(default_factory=list)
    top_k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    run_profile: str = "balanced"
    current_year: int | None = Field(default=None, ge=1900, le=2200)
    metadata: dict[str, Any] = Field(default_factory=dict)


EvalMetricVersion = Literal[
    "legacy_gold_records_v1",
    "deduplicated_gold_identity_v2",
]
LEGACY_GOLD_METRIC_VERSION: EvalMetricVersion = "legacy_gold_records_v1"
DEDUPLICATED_GOLD_METRIC_VERSION: EvalMetricVersion = (
    "deduplicated_gold_identity_v2"
)


class EvalMetricSet(BaseModel):
    """Metric summary for one evaluated run or aggregate suite."""

    # Missing fields in historical JSON intentionally parse as legacy v1.
    metric_version: EvalMetricVersion = LEGACY_GOLD_METRIC_VERSION
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    f1_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(default=0.0, ge=0.0)
    raw_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    ranked_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    per_source_returned_count: dict[str, int] = Field(default_factory=dict)
    source_call_count: int = Field(default=0, ge=0)
    source_error_count: int = Field(default=0, ge=0)
    source_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    warning_count: int = Field(default=0, ge=0)
    query_warning_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failed_case_count: int = Field(default=0, ge=0)
    failed_case_rate: float = Field(default=0.0, ge=0.0, le=1.0)


EvalGroupName = Literal[
    "baseline",
    "query_evolution_only",
    "refchain_only",
    "query_evolution_plus_refchain",
]


class EvalCaseEfficiency(BaseModel):
    """Efficiency counters observed for one evaluated search."""

    latency_seconds: float = Field(default=0.0, ge=0.0)
    api_call_count: int = Field(default=0, ge=0)
    search_api_call_count: int = Field(default=0, ge=0)
    reference_api_call_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    llm_call_count: int = Field(default=0, ge=0)
    llm_total_tokens: int = Field(default=0, ge=0)
    search_rounds: int = Field(default=0, ge=0)
    raw_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    returned_result_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    rate_limit_wait_seconds: float = Field(default=0.0, ge=0.0)
    source_call_count: int = Field(default=0, ge=0)
    source_error_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class EvalAggregateEfficiency(BaseModel):
    """Aggregate efficiency totals and per-case averages."""

    case_count: int = Field(default=0, ge=0)
    average_latency_seconds: float = Field(default=0.0, ge=0.0)
    avg_api_call_count: float = Field(default=0.0, ge=0.0)
    avg_search_api_call_count: float = Field(default=0.0, ge=0.0)
    avg_reference_api_call_count: float = Field(default=0.0, ge=0.0)
    avg_retry_count: float = Field(default=0.0, ge=0.0)
    avg_error_count: float = Field(default=0.0, ge=0.0)
    avg_cache_hit_count: float = Field(default=0.0, ge=0.0)
    avg_rate_limit_wait_seconds: float = Field(default=0.0, ge=0.0)
    avg_llm_call_count: float = Field(default=0.0, ge=0.0)
    avg_llm_total_tokens: float = Field(default=0.0, ge=0.0)
    total_llm_call_count: int = Field(default=0, ge=0)
    total_llm_total_tokens: int = Field(default=0, ge=0)
    average_search_rounds: float = Field(default=0.0, ge=0.0)
    total_raw_count: int = Field(default=0, ge=0)
    total_deduplicated_count: int = Field(default=0, ge=0)
    total_returned_result_count: int = Field(default=0, ge=0)
    total_cache_hit_count: int = Field(default=0, ge=0)
    total_source_call_count: int = Field(default=0, ge=0)
    total_source_error_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class EvalCaseStatistics(BaseModel):
    total_case_count: int = Field(default=0, ge=0)
    gold_case_count: int = Field(default=0, ge=0)
    evaluated_success_count: int = Field(default=0, ge=0)
    failed_case_count: int = Field(default=0, ge=0)
    missing_result_count: int = Field(default=0, ge=0)
    missing_gold_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failed_case_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_result_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class EvalAggregateReport(BaseModel):
    success_only_metrics: EvalMetricSet = Field(default_factory=EvalMetricSet)
    end_to_end_metrics: EvalMetricSet = Field(default_factory=EvalMetricSet)
    case_statistics: EvalCaseStatistics = Field(default_factory=EvalCaseStatistics)
    efficiency: EvalAggregateEfficiency = Field(default_factory=EvalAggregateEfficiency)


class EvalGroupResult(BaseModel):
    """Evaluation result for one query under one feature group."""

    query_id: str
    group: EvalGroupName
    metrics: EvalMetricSet
    ranked_paper_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_stats: list[dict[str, Any]] = Field(default_factory=list)
    raw_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    efficiency: EvalCaseEfficiency = Field(default_factory=EvalCaseEfficiency)
    failed: bool = False
    error_message: str | None = None


class EvalQueryResult(BaseModel):
    """All group results for one query."""

    query_id: str
    query: str
    group_results: dict[EvalGroupName, EvalGroupResult] = Field(default_factory=dict)


class EvalSuiteResult(BaseModel):
    """Complete offline evaluation result."""

    query_results: list[EvalQueryResult] = Field(default_factory=list)
    aggregate_metrics: dict[EvalGroupName, EvalMetricSet] = Field(default_factory=dict)
    aggregate_reports: dict[EvalGroupName, EvalAggregateReport] = Field(
        default_factory=dict
    )
