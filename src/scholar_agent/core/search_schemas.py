"""Internal schemas for search planning and pipeline orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics


SourceName = Literal[
    "openalex",
    "arxiv",
    "semantic_scholar",
    "pubmed",
    "local_bm25",
    "local_hybrid",
]
PaperType = Literal[
    "survey",
    "review",
    "method",
    "benchmark",
    "dataset",
    "application",
    "comparison",
]
ConstraintField = Literal[
    "time_range",
    "venues",
    "methods",
    "datasets",
    "domains",
    "must_include_terms",
    "exclude_terms",
    "paper_types",
]
RunProfile = Literal["fast", "balanced", "high_recall", "evaluation"]
QueryEvolutionPolicy = Literal["off", "seed_expansion", "coverage_gap"]
QueryPlanningPolicy = Literal[
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
]
CombinationMode = Literal["all", "any"]
JudgementPolicy = Literal["current_rules", "calibrated_rules_v1"]
RankingPolicy = Literal["current_rules", "rrf_fusion"]
QueryFacetType = Literal[
    "topic",
    "method",
    "dataset",
    "task",
    "paper_type",
    "venue",
    "temporal",
]
LLMSemanticFacetType = Literal[
    "topic",
    "method",
    "dataset",
    "task",
    "paper_type",
    "venue",
    "temporal",
    "synonym",
]
QueryFacetSource = Literal["explicit", "llm", "rules"]
QueryLanguage = Literal["zh", "en", "mixed", "unknown"]
QueryIntent = Literal[
    "survey",
    "recent_progress",
    "method_comparison",
    "benchmark_or_dataset",
    "application",
    "paper_finding",
    "general",
]
ResearchDomain = Literal[
    "computer_science",
    "machine_learning",
    "biomedical",
    "general_science",
]
EvidenceSource = Literal["title", "abstract", "venue", "metadata"]
JudgementCategory = Literal[
    "highly_relevant",
    "partially_relevant",
    "weakly_relevant",
    "irrelevant",
    "insufficient_evidence",
]

DEFAULT_SEARCH_SOURCES: tuple[str, ...] = (
    "openalex",
    "arxiv",
    "semantic_scholar",
    "pubmed",
)
SUPPORTED_SEARCH_SOURCES: tuple[str, ...] = (
    *DEFAULT_SEARCH_SOURCES,
    "local_bm25",
    "local_hybrid",
)
SUPPORTED_PAPER_TYPES: tuple[str, ...] = (
    "survey",
    "review",
    "method",
    "benchmark",
    "dataset",
    "application",
    "comparison",
)
QUERY_PLANNER_VERSION = "1.9.0"
LLM_QUERY_PLANNING_SCHEMA_VERSION = "1"


class SearchBudget(BaseModel):
    """Execution limits shared by API, CLI, evaluators, and SearchService."""

    max_search_rounds: int = Field(default=2, ge=1, le=3)
    max_candidate_papers: int = Field(default=200, ge=1, le=300)
    max_llm_calls: int = Field(default=20, ge=0, le=100)
    max_total_tokens: int = Field(default=50_000, ge=0, le=1_000_000)
    max_latency_seconds: float = Field(default=90.0, gt=0.0, le=120.0)


DEFAULT_SEARCH_BUDGET = SearchBudget()


class CandidateTruncation(BaseModel):
    stage: str
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    truncated_count: int = Field(ge=0)


class BudgetStatus(BaseModel):
    exhausted: bool = False
    stop_reasons: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    max_search_rounds: int = DEFAULT_SEARCH_BUDGET.max_search_rounds
    completed_search_rounds: int = 0
    max_candidate_papers: int = DEFAULT_SEARCH_BUDGET.max_candidate_papers
    candidate_limit_applied: bool = False
    candidate_truncations: list[CandidateTruncation] = Field(default_factory=list)
    max_llm_calls: int = DEFAULT_SEARCH_BUDGET.max_llm_calls
    used_llm_calls: int = 0
    max_total_tokens: int = DEFAULT_SEARCH_BUDGET.max_total_tokens
    used_total_tokens: int = 0
    token_usage_precise: bool = True
    max_latency_seconds: float = DEFAULT_SEARCH_BUDGET.max_latency_seconds
    elapsed_seconds: float = 0.0
PAPER_TYPE_ALIASES: dict[str, str] = {
    "survey": "survey",
    "review": "review",
    "literature_review": "review",
    "systematic_review": "review",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "benchmark": "benchmark",
    "benchmarking": "benchmark",
    "dataset": "dataset",
    "data_set": "dataset",
    "application": "application",
    "applied": "application",
    "comparison": "comparison",
    "comparative": "comparison",
}


class TimeRange(BaseModel):
    start_year: int | None = Field(default=None, ge=1800, le=2200)
    end_year: int | None = Field(default=None, ge=1800, le=2200)
    label: str | None = None

    @model_validator(mode="after")
    def validate_year_order(self) -> "TimeRange":
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.end_year < self.start_year
        ):
            raise ValueError("end_year must be greater than or equal to start_year")
        return self


class QueryConstraint(BaseModel):
    time_range: TimeRange | None = None
    venues: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    must_include_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    paper_types: list[PaperType] = Field(default_factory=list)
    explicit_fields: list[ConstraintField] = Field(
        default_factory=list,
        exclude=True,
    )

    @field_validator(
        "venues",
        "methods",
        "datasets",
        "domains",
        "must_include_terms",
        "exclude_terms",
        mode="before",
    )
    @classmethod
    def normalize_string_constraints(cls, value: object) -> list[str]:
        return normalize_constraint_values(value)

    @field_validator("paper_types", mode="before")
    @classmethod
    def normalize_paper_type_constraints(cls, value: object) -> list[str]:
        return normalize_paper_types(value)

    @field_validator("explicit_fields", mode="before")
    @classmethod
    def normalize_explicit_fields(cls, value: object) -> list[str]:
        return normalize_constraint_values(value)


class QueryAnalysis(BaseModel):
    original_query: str
    language: QueryLanguage = "unknown"
    intent: QueryIntent = "general"
    domain: ResearchDomain = "general_science"
    constraints: QueryConstraint = Field(default_factory=QueryConstraint)
    needs_expansion: bool = False
    reasoning: list[str] = Field(default_factory=list)


class SearchSubquery(BaseModel):
    query: str = Field(..., min_length=1)
    combination_mode: CombinationMode = "all"
    source_hints: list[SourceName] = Field(
        default_factory=lambda: list(DEFAULT_SEARCH_SOURCES)
    )
    priority: int = Field(default=1, ge=1, le=5)
    purpose: str = Field(..., min_length=1)
    facet_types: list[QueryFacetType] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("source_hints", mode="before")
    @classmethod
    def normalize_source_hints(cls, value: object) -> list[str]:
        return _normalize_sources(value)


class QueryFacet(BaseModel):
    facet_type: QueryFacetType
    terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: QueryFacetSource = "rules"
    required: bool = False
    warnings: list[str] = Field(default_factory=list)


class LLMSemanticFacet(BaseModel):
    """LLM 语义规划输出中的可审计术语映射。"""

    model_config = ConfigDict(extra="forbid")

    facet_type: LLMSemanticFacetType
    original_terms: list[str] = Field(default_factory=list, max_length=12)
    normalized_terms: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMSemanticQuery(BaseModel):
    """一条来源无关、待本地校验的语义补充查询。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    purpose: str = Field(min_length=1, max_length=120)
    covered_facets: list[LLMSemanticFacetType] = Field(
        default_factory=list,
        max_length=8,
    )
    retained_must_have_terms: list[str] = Field(default_factory=list, max_length=12)
    terminology_expansions: list[str] = Field(default_factory=list, max_length=12)


class LLMQueryPlanningOutput(BaseModel):
    """独立语义查询规划 Prompt 的严格 JSON 输出。"""

    model_config = ConfigDict(extra="forbid")

    intent_summary: str = Field(min_length=1, max_length=240)
    facets: list[LLMSemanticFacet] = Field(default_factory=list, max_length=16)
    supplemental_queries: list[LLMSemanticQuery] = Field(
        default_factory=list,
        max_length=2,
    )
    warnings: list[str] = Field(default_factory=list, max_length=12)


class LLMConstrainedRewriteOutput(BaseModel):
    """受约束检索改写 Prompt 的严格 JSON 输出。"""

    model_config = ConfigDict(extra="forbid")

    input_summary: str = Field(min_length=1, max_length=240)
    rewritten_query: str = Field(min_length=1, max_length=200)
    preserved_terms: list[str] = Field(default_factory=list, max_length=24)
    generic_synonyms_used: list[str] = Field(default_factory=list, max_length=12)
    warnings: list[str] = Field(default_factory=list, max_length=12)


class PRFSeedCandidate(BaseModel):
    rank: int = Field(ge=1)
    title: str


class PRFFeedbackTerm(BaseModel):
    term: str
    ngram_size: Literal[1, 2]
    document_frequency: int = Field(ge=2)
    term_frequency: int = Field(ge=2)
    rank_discounted_frequency: float = Field(ge=0.0)


class QueryPlanningResult(BaseModel):
    policy: QueryPlanningPolicy = "current_rules"
    planner_version: str = QUERY_PLANNER_VERSION
    facets: list[QueryFacet] = Field(default_factory=list)
    selected_subqueries: list[SearchSubquery] = Field(default_factory=list)
    skipped_facets: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    identified_facet_count: int = Field(default=0, ge=0)
    selected_facet_count: int = Field(default=0, ge=0)
    explicit_facet_count: int = Field(default=0, ge=0)
    selected_subquery_count: int = Field(default=0, ge=0)
    duplicate_subquery_count: int = Field(default=0, ge=0)
    skipped_by_budget_count: int = Field(default=0, ge=0)
    topic_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    method_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    dataset_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    task_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    paper_type_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_projection_input_concepts: list[str] = Field(default_factory=list)
    concept_projection_selected_concepts: list[str] = Field(default_factory=list)
    concept_projection_query: str | None = None
    concept_projection_replaced_query: str | None = None
    concept_projection_replaced_purpose: str | None = None
    concept_projection_skip_reason: str | None = None
    prf_seed_candidates: list[PRFSeedCandidate] = Field(default_factory=list)
    prf_feedback_terms: list[PRFFeedbackTerm] = Field(default_factory=list)
    prf_query: str | None = None
    prf_replaced_index: int | None = Field(default=None, ge=0)
    prf_replaced_query: str | None = None
    prf_replaced_purpose: str | None = None
    prf_skip_reason: str | None = None
    prf_fallback_used: bool = False
    prf_first_round_source_statuses: dict[str, str] = Field(default_factory=dict)
    constrained_rewrite_input_summary: dict[str, object] = Field(
        default_factory=dict
    )
    constrained_rewrite_query: str | None = None
    constrained_rewrite_replaced_index: int | None = Field(default=None, ge=0)
    constrained_rewrite_replaced_query: str | None = None
    constrained_rewrite_replaced_purpose: str | None = None
    constrained_rewrite_skip_reason: str | None = None
    constrained_rewrite_validation_rejections: list[str] = Field(
        default_factory=list
    )
    provider: str | None = None
    model: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    snapshot_key: str | None = None
    snapshot_status: str | None = None
    llm_call_attempted: bool = False
    replayed: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    output_valid: bool = False
    original_query_retained: bool = True
    generated_query_count: int = Field(default=0, ge=0)
    accepted_query_count: int = Field(default=0, ge=0)
    rejected_query_count: int = Field(default=0, ge=0)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    accepted_queries: list[str] = Field(default_factory=list)
    terminology_expansions: list[str] = Field(default_factory=list)
    llm_prompt_tokens: int = Field(default=0, ge=0)
    llm_completion_tokens: int = Field(default=0, ge=0)
    llm_total_tokens: int = Field(default=0, ge=0)
    recorded_llm_latency_seconds: float = Field(default=0.0, ge=0.0)


class SearchPlan(BaseModel):
    query_analysis: QueryAnalysis
    subqueries: list[SearchSubquery] = Field(default_factory=list)
    selected_sources: list[SourceName] = Field(
        default_factory=lambda: list(DEFAULT_SEARCH_SOURCES)
    )
    limit_per_source: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=20, ge=1, le=100)
    run_profile: RunProfile = "balanced"
    enable_refchain: bool = False
    enable_semantic_seed_expansion: bool = False
    enable_query_evolution: bool = False
    query_evolution_policy: QueryEvolutionPolicy = "coverage_gap"
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    ranking_policy: RankingPolicy = "current_rules"
    query_planning: QueryPlanningResult = Field(
        default_factory=QueryPlanningResult
    )
    warnings: list[str] = Field(default_factory=list)

    @field_validator("selected_sources", mode="before")
    @classmethod
    def normalize_selected_sources(cls, value: object) -> list[str]:
        return _normalize_sources(value)


class QueryUnderstandingOptions(BaseModel):
    top_k: int = Field(default=20, ge=1, le=100)
    run_profile: RunProfile = "balanced"
    enable_refchain: bool = False
    enable_semantic_seed_expansion: bool = False
    enable_query_evolution: bool = False
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    current_year: int | None = Field(default=None, ge=1900, le=2200)
    use_llm: bool | None = None
    explicit_constraints: QueryConstraint | None = None


class EvidenceItem(BaseModel):
    source: EvidenceSource
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


LexicalNormalizationPolicy = Literal["off", "lexical_normalization_v1"]
LexicalNormalizationFacet = Literal["topic", "must_have", "method", "domain"]


class LexicalNormalizationMatch(BaseModel):
    policy_version: Literal["lexical-normalization-v1"]
    facet: LexicalNormalizationFacet
    original_term: str
    normalized_form: str
    field: Literal["title", "abstract"]
    score_impact: float = Field(ge=0.0, le=1.0)


class JudgementRuleConfig(BaseModel):
    """版本化、来源无关的确定性相关性判断参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str = Field(min_length=1, max_length=80)
    lexical_normalization_policy: LexicalNormalizationPolicy = "off"
    title_topic_weight: float = Field(ge=0.0, le=1.0)
    abstract_topic_weight: float = Field(ge=0.0, le=1.0)
    topic_max_score: float = Field(ge=0.0, le=1.0)
    title_must_have_weight: float = Field(ge=0.0, le=1.0)
    abstract_must_have_weight: float = Field(ge=0.0, le=1.0)
    must_have_max_score: float = Field(ge=0.0, le=1.0)
    title_method_weight: float = Field(ge=0.0, le=1.0)
    abstract_method_weight: float = Field(ge=0.0, le=1.0)
    method_max_score: float = Field(ge=0.0, le=1.0)
    title_dataset_weight: float = Field(ge=0.0, le=1.0)
    abstract_dataset_weight: float = Field(ge=0.0, le=1.0)
    dataset_max_score: float = Field(ge=0.0, le=1.0)
    title_domain_weight: float = Field(ge=0.0, le=1.0)
    abstract_domain_weight: float = Field(ge=0.0, le=1.0)
    domain_max_score: float = Field(ge=0.0, le=1.0)
    paper_type_match_weight: float = Field(ge=0.0, le=1.0)
    paper_type_max_score: float = Field(ge=0.0, le=1.0)
    paper_type_mismatch_penalty: float = Field(ge=0.0, le=1.0)
    venue_match_weight: float = Field(ge=0.0, le=1.0)
    venue_mismatch_penalty: float = Field(ge=0.0, le=1.0)
    temporal_match_weight: float = Field(ge=0.0, le=1.0)
    temporal_early_penalty: float = Field(ge=0.0, le=1.0)
    temporal_near_penalty: float = Field(ge=0.0, le=1.0)
    temporal_late_penalty: float = Field(ge=0.0, le=1.0)
    multi_dimension_bonus: float = Field(ge=0.0, le=1.0)
    multi_dimension_bonus_cap: float = Field(ge=0.0, le=1.0)
    insufficient_coverage_penalty: float = Field(ge=0.0, le=1.0)
    broad_topic_score_cap: float = Field(ge=0.0, le=1.0)
    explicit_dataset_penalty: float = Field(ge=0.0, le=1.0)
    missing_abstract_penalty: float = Field(ge=0.0, le=1.0)
    missing_metadata_penalty: float = Field(ge=0.0, le=1.0)
    highly_relevant_threshold: float = Field(ge=0.0, le=1.0)
    partially_relevant_threshold: float = Field(ge=0.0, le=1.0)
    weakly_relevant_threshold: float = Field(ge=0.0, le=1.0)
    minimum_evidence_count: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "JudgementRuleConfig":
        if not (
            self.weakly_relevant_threshold
            <= self.partially_relevant_threshold
            <= self.highly_relevant_threshold
        ):
            raise ValueError(
                "judgement thresholds must satisfy weak <= partial <= high"
            )
        return self


class JudgementFeatureVector(BaseModel):
    """不含摘要正文的候选级规则特征与可加和分数组件。"""

    config_version: str
    config_hash: str
    matched_topic_terms: list[str] = Field(default_factory=list)
    matched_method_terms: list[str] = Field(default_factory=list)
    matched_dataset_terms: list[str] = Field(default_factory=list)
    matched_task_terms: list[str] = Field(default_factory=list)
    matched_must_have_terms: list[str] = Field(default_factory=list)
    matched_paper_types: list[str] = Field(default_factory=list)
    lexical_normalization_matches: list[LexicalNormalizationMatch] = Field(
        default_factory=list
    )
    title_matched_terms: list[str] = Field(default_factory=list)
    abstract_matched_terms: list[str] = Field(default_factory=list)
    title_match_score: float = 0.0
    abstract_match_score: float = 0.0
    venue_match: bool | None = None
    temporal_match: bool | None = None
    metadata_completeness: float = Field(ge=0.0, le=1.0)
    constraint_results: dict[str, bool | None] = Field(default_factory=dict)
    hard_constraint_failures: list[str] = Field(default_factory=list)
    score_components: dict[str, float] = Field(default_factory=dict)
    evidence_count: int = Field(default=0, ge=0)
    final_score: float = Field(ge=0.0, le=1.0)
    highly_relevant_threshold: float = Field(ge=0.0, le=1.0)
    partially_relevant_threshold: float = Field(ge=0.0, le=1.0)
    weakly_relevant_threshold: float = Field(ge=0.0, le=1.0)
    category_reason: str


class JudgementResult(BaseModel):
    paper: Paper
    score: float = Field(ge=0.0, le=1.0)
    category: JudgementCategory
    reasoning: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    feature_vector: JudgementFeatureVector | None = None


class RerankScoreBreakdown(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    authority_score: float = Field(ge=0.0, le=1.0)
    timeliness_score: float = Field(ge=0.0, le=1.0)
    metadata_score: float = Field(ge=0.0, le=1.0)
    category_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    relevance_weight: float = Field(ge=0.0, le=1.0)
    authority_weight: float = Field(ge=0.0, le=1.0)
    timeliness_weight: float = Field(ge=0.0, le=1.0)
    metadata_weight: float = Field(ge=0.0, le=1.0)


class RRFListContribution(BaseModel):
    source: str
    subquery: str
    rank: int = Field(ge=1)
    reciprocal_score: float = Field(ge=0.0)


class RankedPaper(BaseModel):
    rank: int = Field(ge=1)
    paper: Paper
    final_score: float = Field(ge=0.0, le=1.0)
    category: JudgementCategory
    score_breakdown: RerankScoreBreakdown
    ranking_reason: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rrf_score: float | None = Field(default=None, ge=0.0)
    rrf_contributions: list[RRFListContribution] = Field(default_factory=list)
    original_rank: int | None = Field(default=None, ge=1)
    rrf_top_20_change: str | None = None
    rrf_rank_change_reason: str | None = None


class QueryEvolutionOptions(BaseModel):
    policy: QueryEvolutionPolicy = "coverage_gap"
    max_evolved_queries: int | None = Field(default=None, ge=0, le=10)
    max_seed_papers: int | None = Field(default=None, ge=0, le=20)
    min_seed_score: float = Field(default=0.45, ge=0.0, le=1.0)


class QueryCoverageGap(BaseModel):
    topic_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    method_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    dataset_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    must_have_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    paper_type_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    venue_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_topics: list[str] = Field(default_factory=list)
    missing_methods: list[str] = Field(default_factory=list)
    missing_datasets: list[str] = Field(default_factory=list)
    missing_must_have_terms: list[str] = Field(default_factory=list)
    missing_paper_types: list[str] = Field(default_factory=list)
    missing_venues: list[str] = Field(default_factory=list)
    needs_evolution: bool = False
    reasons: list[str] = Field(default_factory=list)


class QueryEvolutionQualityGate(BaseModel):
    raw_candidate_count: int = Field(default=0, ge=0)
    unique_candidate_count: int = Field(default=0, ge=0)
    duplicate_candidate_count: int = Field(default=0, ge=0)
    duplicate_with_initial_count: int = Field(default=0, ge=0)
    accepted_candidate_count: int = Field(default=0, ge=0)
    filtered_candidate_count: int = Field(default=0, ge=0)
    core_dimension_match_count: int = Field(default=0, ge=0)
    filtered_reason_counts: dict[str, int] = Field(default_factory=dict)
    source_candidate_counts: dict[str, int] = Field(default_factory=dict)
    accepted_source_counts: dict[str, int] = Field(default_factory=dict)


class EvolvedSubquery(BaseModel):
    query: str = Field(..., min_length=1)
    source_hints: list[SourceName] = Field(
        default_factory=lambda: list(DEFAULT_SEARCH_SOURCES)
    )
    priority: int = Field(default=1, ge=1, le=5)
    purpose: str = Field(..., min_length=1)
    seed_paper_titles: list[str] = Field(default_factory=list)
    generated_by: Literal["rules", "llm"] = "rules"
    generation_policy: QueryEvolutionPolicy = "coverage_gap"
    gap_dimensions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("source_hints", mode="before")
    @classmethod
    def normalize_source_hints(cls, value: object) -> list[str]:
        return _normalize_sources(value)


class QueryEvolutionRecord(BaseModel):
    round_index: int = Field(default=1, ge=1)
    policy: QueryEvolutionPolicy = "coverage_gap"
    eligible_seed_count: int = Field(default=0, ge=0)
    eligible_seed_titles: list[str] = Field(default_factory=list)
    seed_count: int = Field(default=0, ge=0)
    seed_paper_titles: list[str] = Field(default_factory=list)
    coverage_gap: QueryCoverageGap | None = None
    generated_queries: list[EvolvedSubquery] = Field(default_factory=list)
    quality_gate: QueryEvolutionQualityGate = Field(
        default_factory=QueryEvolutionQualityGate
    )
    skipped_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = Field(default=0.0, ge=0.0)


class RefChainOptions(BaseModel):
    max_seed_papers: int = Field(default=3, ge=0, le=20)
    max_references_per_seed: int = Field(default=15, ge=0, le=100)
    max_total_references: int = Field(default=50, ge=0, le=500)
    min_seed_score: float = Field(default=0.45, ge=0.0, le=1.0)


class RefChainSeed(BaseModel):
    paper: Paper
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class RefChainSeedDiagnostic(BaseModel):
    seed_id: str | None = None
    seed_rank: int = Field(ge=1)
    seed_category: JudgementCategory
    seed_score: float = Field(ge=0.0, le=1.0)
    identifier_type: str | None = None
    snapshot_key: str | None = None
    request_count: int = Field(default=0, ge=0)
    recorded_request_count: int = Field(default=0, ge=0)
    recorded_retry_count: int = Field(default=0, ge=0)
    recorded_error_count: int = Field(default=0, ge=0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    references_returned: int = Field(default=0, ge=0)
    unique_references_returned: int = Field(default=0, ge=0)
    skip_reason: str | None = None


class ReferenceEdge(BaseModel):
    seed_paper_id: str
    reference_paper_id: str
    source: SourceName = "openalex"
    relation: Literal["references"] = "references"


class RefChainRecord(BaseModel):
    seeds: list[RefChainSeed] = Field(default_factory=list)
    seed_diagnostics: list[RefChainSeedDiagnostic] = Field(default_factory=list)
    reference_edges: list[ReferenceEdge] = Field(default_factory=list)
    raw_reference_count: int = Field(default=0, ge=0)
    returned_reference_count: int = Field(default=0, ge=0)
    skipped_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)


class RefChainOutput(BaseModel):
    references: list[Paper] = Field(default_factory=list)
    reference_edges: list[ReferenceEdge] = Field(default_factory=list)
    record: RefChainRecord
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)


class SemanticSeedExpansionSeed(BaseModel):
    semantic_scholar_id: str
    rank: int = Field(ge=1)
    title: str
    source: Literal["direct", "resolved"] = "direct"
    resolution_identifier: str | None = None


class SemanticSeedResolutionRecord(BaseModel):
    requested_identifier: str
    rank: int = Field(ge=1)
    status: Literal["resolved", "missing", "conflict", "source_failure"]
    semantic_scholar_id: str | None = None
    identity_rule: str | None = None
    shared_identifiers: list[str] = Field(default_factory=list)
    conflicting_identifiers: list[str] = Field(default_factory=list)


class SemanticSeedExpansionRecord(BaseModel):
    status: Literal[
        "disabled", "no_eligible_seed", "success", "source_failure"
    ] = "disabled"
    seeds: list[SemanticSeedExpansionSeed] = Field(default_factory=list)
    snapshot_key: str | None = None
    resolution_snapshot_key: str | None = None
    resolution_status: Literal[
        "not_needed", "not_attempted", "success", "partial_success", "source_failure"
    ] = "not_needed"
    resolution_candidate_count: int = Field(default=0, ge=0)
    resolution_request_identifier_count: int = Field(default=0, ge=0)
    resolved_candidate_count: int = Field(default=0, ge=0)
    resolution_conflict_count: int = Field(default=0, ge=0)
    resolution_missing_count: int = Field(default=0, ge=0)
    direct_seed_count: int = Field(default=0, ge=0)
    resolved_seed_count: int = Field(default=0, ge=0)
    resolutions: list[SemanticSeedResolutionRecord] = Field(default_factory=list)
    resolution_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    recorded_resolution_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    resolution_latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_resolution_latency_seconds: float = Field(default=0.0, ge=0.0)
    raw_recommendation_count: int = Field(default=0, ge=0)
    new_unique_candidate_count: int = Field(default=0, ge=0)
    duplicate_candidate_count: int = Field(default=0, ge=0)
    identity_merges: list[dict[str, object]] = Field(default_factory=list)
    skip_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)


class SemanticSeedExpansionOutput(BaseModel):
    recommendations: list[Paper] = Field(default_factory=list)
    record: SemanticSeedExpansionRecord
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)


def _normalize_sources(value: object) -> list[str]:
    return normalize_search_sources(value)


def normalize_search_sources(value: object) -> list[str]:
    """Normalize supported search sources with stable de-duplication."""

    if value is None:
        return list(DEFAULT_SEARCH_SOURCES)
    if isinstance(value, str):
        raw_sources = [value]
    else:
        raw_sources = list(value)  # type: ignore[arg-type]

    normalized: list[str] = []
    seen: set[str] = set()
    for source in raw_sources:
        key = (
            str(source)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if key == "semanticscholar":
            key = "semantic_scholar"
        if not key or key in seen:
            continue
        if key not in SUPPORTED_SEARCH_SOURCES:
            raise ValueError(f"unsupported search source: {key}")
        normalized.append(key)
        seen.add(key)
    return normalized


def normalize_constraint_values(value: object) -> list[str]:
    """Trim and case-insensitively de-duplicate constraint strings."""

    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else list(value)  # type: ignore[arg-type]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        item = str(raw_value).strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        normalized.append(item)
        seen.add(key)
    return normalized


def normalize_paper_types(value: object) -> list[str]:
    """Normalize supported paper types or reject unknown values."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in normalize_constraint_values(value):
        key = reformat_enum_value(raw_value)
        paper_type = PAPER_TYPE_ALIASES.get(key)
        if paper_type is None:
            raise ValueError(f"unsupported paper type: {raw_value}")
        if paper_type in seen:
            continue
        normalized.append(paper_type)
        seen.add(paper_type)
    return normalized


def reformat_enum_value(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())
