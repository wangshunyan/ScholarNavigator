export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export type RunProfile = "fast" | "balanced" | "high_recall" | "evaluation";
export type QueryEvolutionPolicy = "off" | "seed_expansion" | "coverage_gap";
export type QueryPlanningPolicy =
  | "current_rules"
  | "prf_v1"
  | "concept_projection"
  | "controlled_relaxation"
  | "disjunctive_facets"
  | "current_plus_disjunctive"
  | "facet_union"
  | "facet_balanced"
  | "llm_semantic"
  | "llm_constrained_rewrite";
export type JudgementPolicy = "current_rules" | "calibrated_rules_v1";
export type RankingPolicy = "current_rules" | "rrf_fusion";
export type QueryFacetType =
  | "topic"
  | "method"
  | "dataset"
  | "task"
  | "paper_type"
  | "venue"
  | "temporal";

export type RelevanceCategory =
  | "highly_relevant"
  | "partially_relevant"
  | "weakly_relevant"
  | "irrelevant"
  | "insufficient_evidence";

export interface HealthResponse {
  status: string;
  version: string;
  time: string;
}

export interface RuntimeConfigResponse {
  mode: string;
  llm: {
    provider: string;
    model?: string | null;
    available: boolean;
    base_url_host?: string | null;
    reason?: string | null;
  };
  connectors: Array<{
    name: string;
    available: boolean;
    requires_key: boolean;
    reason?: string | null;
  }>;
  limits: {
    max_top_k: number;
    max_search_rounds: number;
    max_candidate_papers: number;
    max_latency_seconds: number;
    real_search_max_workers?: number;
    real_search_background_workers?: number;
    real_search_run_ttl_seconds?: number;
    real_search_max_stored_runs?: number;
  };
  features: {
    query_evolution: boolean;
    refchain: boolean;
    evaluation: boolean;
    sse: boolean;
    real_search?: boolean;
    real_search_cancel?: boolean;
    real_search_sse?: boolean;
    retrieval_cache?: boolean;
    batch_cli?: boolean;
    llm_query_understanding?: boolean;
    llm_judgement?: boolean;
  };
}

export interface SearchRunCreateRequest {
  query: string;
  locale?: string;
  constraints?: {
    time_range?: {
      start_year?: number | null;
      end_year?: number | null;
    } | null;
    venues?: string[];
    must_have_terms?: string[];
    excluded_terms?: string[];
    datasets?: string[];
    paper_types?: string[];
  };
  source_preferences?: string[];
  run_profile?: RunProfile;
  top_k?: number;
  budgets?: {
    max_search_rounds?: number;
    max_candidate_papers?: number;
    max_llm_calls?: number;
    max_total_tokens?: number;
    max_latency_seconds?: number;
  };
  options?: {
    query_planning_policy?: QueryPlanningPolicy;
    ranking_policy?: RankingPolicy;
    judgement_policy?: JudgementPolicy;
    enable_query_evolution?: boolean;
    query_evolution_policy?: QueryEvolutionPolicy;
    enable_refchain?: boolean;
    enable_semantic_seed_expansion?: boolean;
    enable_llm_query_understanding?: boolean | null;
    enable_llm_judgement?: boolean | null;
    refchain_depth?: number;
    return_markdown?: boolean;
    return_json?: boolean;
    stream_events?: boolean;
  };
}

export interface InternalSearchPreviewRequest {
  query: string;
  top_k?: number;
  run_profile?: RunProfile;
  enable_refchain?: boolean;
  enable_semantic_seed_expansion?: boolean;
  enable_query_evolution?: boolean;
  query_evolution_policy?: QueryEvolutionPolicy;
  query_planning_policy?: QueryPlanningPolicy;
  ranking_policy?: RankingPolicy;
  judgement_policy?: JudgementPolicy;
  enable_llm_query_understanding?: boolean | null;
  enable_llm_judgement?: boolean | null;
  current_year?: number | null;
}

export interface SearchRunCreateResponse {
  run_id: string;
  status: RunStatus;
  created_at: string;
  links: {
    self: string;
    events: string;
    result: string;
  };
}

export interface CostReport {
  api_call_count: number;
  logical_search_call_count: number;
  search_api_call_count: number;
  reference_api_call_count: number;
  retry_count: number;
  error_count: number;
  llm_call_count: number;
  llm_prompt_tokens?: number;
  llm_completion_tokens?: number;
  llm_total_tokens?: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_total_tokens: number;
  latency_seconds: number;
  cache_hit_count: number;
  rate_limit_wait_seconds: number;
  search_rounds: number;
  judged_paper_count: number;
  raw_candidate_count: number;
  deduplicated_candidate_count: number;
}

export interface ConnectorDiagnostics {
  request_count: number;
  retry_count: number;
  error_count: number;
  cache_hit_count: number;
  rate_limit_wait_seconds: number;
  latency_seconds: number;
}

export interface SearchRunStatusResponse {
  run_id: string;
  status: RunStatus;
  current_stage: string;
  progress: {
    completed_stages: string[];
    skipped_stages: string[];
    candidate_paper_count: number;
    judged_paper_count: number;
  };
  cost_report: CostReport;
  created_at: string;
  updated_at: string;
}

export interface PaperIdentifiers {
  doi?: string | null;
  arxiv_id?: string | null;
  semantic_scholar_id?: string | null;
  s2orc_corpus_id?: string | null;
  openalex_id?: string | null;
  pubmed_id?: string | null;
}

export interface PaperUrls {
  landing_page?: string | null;
  pdf?: string | null;
}

export interface Paper {
  title: string;
  authors: string[];
  year: number | null;
  venue?: string | null;
  abstract: string;
  identifiers: PaperIdentifiers;
  urls: PaperUrls;
  sources: string[];
}

export interface EvidenceItem {
  source: string;
  text: string;
  confidence: number;
}

export interface SynthesisEvidenceRow {
  row_id: string;
  citation_key: string;
  rank: number;
  paper_title: string;
  year?: number | null;
  venue?: string | null;
  sources: string[];
  identifiers: PaperIdentifiers;
  category: string;
  final_score: number;
  evidence_source: string;
  evidence_text: string;
  supported_terms: string[];
  supported_claim: string;
}

export interface SynthesisFinding {
  text: string;
  citation_keys: string[];
  confidence: number;
  evidence_row_ids: string[];
}

export interface CitationCoverage {
  ranked_paper_count: number;
  cited_paper_count: number;
  evidence_row_count: number;
  cited_evidence_row_count: number;
  missing_evidence_count: number;
  source_error_count: number;
  coverage_ratio: number;
}

export interface SynthesisOutput {
  answer_summary: string;
  status: string;
  key_findings: SynthesisFinding[];
  evidence_table: SynthesisEvidenceRow[];
  citation_coverage: CitationCoverage;
  limitations: string[];
  warnings: string[];
}

export interface RankedPaper {
  result_identity: string;
  authority_digest: string;
  rank: number;
  paper: Paper;
  relevance_score: number;
  category: RelevanceCategory;
  matched_constraints: string[];
  ranking_reason: string;
  evidence: EvidenceItem[];
  rrf_score?: number | null;
  rrf_contributions: Array<{
    source: string;
    subquery: string;
    rank: number;
    reciprocal_score: number;
  }>;
  original_rank?: number | null;
  rrf_top_20_change?: string | null;
  rrf_rank_change_reason?: string | null;
}

export interface QueryAnalysis {
  intent_type: string;
  domain: string;
  research_topics: string[];
  constraints: Record<string, unknown>;
}

export interface SearchPlan {
  expanded_queries: string[];
  source_preferences: string[];
  max_rounds: number;
  query_planning_policy: QueryPlanningPolicy;
  ranking_policy: RankingPolicy;
  enable_semantic_seed_expansion: boolean;
  query_planning: {
    policy: QueryPlanningPolicy;
    planner_version: string;
    facets: Array<{
      facet_type: QueryFacetType;
      terms: string[];
      confidence: number;
      source: "explicit" | "llm" | "rules";
      required: boolean;
      warnings: string[];
    }>;
    selected_subqueries: Array<{
      query: string;
      combination_mode: "all" | "any";
      source_hints: string[];
      priority: number;
      purpose: string;
      facet_types: QueryFacetType[];
      provenance: string[];
    }>;
    skipped_facets: string[];
    warnings: string[];
    identified_facet_count: number;
    selected_facet_count: number;
    explicit_facet_count: number;
    selected_subquery_count: number;
    duplicate_subquery_count: number;
    skipped_by_budget_count: number;
    topic_coverage: number;
    method_coverage: number;
    dataset_coverage: number;
    task_coverage: number;
    paper_type_coverage: number;
    concept_projection_input_concepts: string[];
    concept_projection_selected_concepts: string[];
    concept_projection_query: string | null;
    concept_projection_replaced_query: string | null;
    concept_projection_replaced_purpose: string | null;
    concept_projection_skip_reason: string | null;
    prf_seed_candidates: Array<{ rank: number; title: string }>;
    prf_feedback_terms: Array<{
      term: string;
      ngram_size: 1 | 2;
      document_frequency: number;
      term_frequency: number;
      rank_discounted_frequency: number;
    }>;
    prf_query: string | null;
    prf_replaced_index: number | null;
    prf_replaced_query: string | null;
    prf_replaced_purpose: string | null;
    prf_skip_reason: string | null;
    prf_fallback_used: boolean;
    prf_first_round_source_statuses: Record<string, string>;
    constrained_rewrite_input_summary: Record<string, unknown>;
    constrained_rewrite_query: string | null;
    constrained_rewrite_replaced_index: number | null;
    constrained_rewrite_replaced_query: string | null;
    constrained_rewrite_replaced_purpose: string | null;
    constrained_rewrite_skip_reason: string | null;
    constrained_rewrite_validation_rejections: string[];
    provider: string | null;
    model: string | null;
    prompt_name: string | null;
    prompt_version: string | null;
    prompt_hash: string | null;
    snapshot_key: string | null;
    snapshot_status: string | null;
    llm_call_attempted: boolean;
    replayed: boolean;
    fallback_used: boolean;
    fallback_reason: string | null;
    output_valid: boolean;
    original_query_retained: boolean;
    generated_query_count: number;
    accepted_query_count: number;
    rejected_query_count: number;
    rejection_reasons: Record<string, number>;
    accepted_queries: string[];
    terminology_expansions: string[];
    llm_prompt_tokens: number;
    llm_completion_tokens: number;
    llm_total_tokens: number;
    recorded_llm_latency_seconds: number;
  };
  query_evolution_policy: QueryEvolutionPolicy;
}

export interface MethodCluster {
  name: string;
  paper_ranks: number[];
  summary: string;
}

export interface TimelineItem {
  year: number;
  paper_ranks: number[];
  summary: string;
}

export interface CitationGraph {
  nodes: Array<{
    id: string;
    label: string;
    rank?: number | null;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relation: string;
  }>;
}

export interface RetrievalSourceStats {
  source: string;
  returned_count: number;
  latency_seconds: number;
  cache_hit: boolean;
  logical_call_executed: boolean;
  adaptation_strategy?: string | null;
  triggered_by: string[];
  safe_original_candidate_count?: number | null;
  safe_original_core_term_coverage?: number | null;
  safe_original_constraint_coverage?: number | null;
  sufficiency_reasons: string[];
  compact_query_executed?: boolean | null;
  compact_query_skipped_reason?: string | null;
  error_message?: string | null;
  diagnostics: ConnectorDiagnostics;
}

export interface RetrievalDiagnostics {
  raw_count: number;
  deduplicated_count: number;
  source_stats: RetrievalSourceStats[];
}

export interface BudgetStatus {
  exhausted: boolean;
  stop_reasons: string[];
  diagnostics: string[];
  max_search_rounds: number;
  completed_search_rounds: number;
  max_candidate_papers: number;
  candidate_limit_applied: boolean;
  candidate_truncations: Array<{
    stage: string;
    before_count: number;
    after_count: number;
    truncated_count: number;
  }>;
  max_llm_calls: number;
  used_llm_calls: number;
  max_total_tokens: number;
  used_total_tokens: number;
  token_usage_precise: boolean;
  max_latency_seconds: number;
  elapsed_seconds: number;
}

export interface SearchRunResultResponse {
  run_id: string;
  status: RunStatus;
  partial: boolean;
  query_analysis: QueryAnalysis;
  search_plan: SearchPlan;
  highly_relevant_papers: RankedPaper[];
  partially_relevant_papers: RankedPaper[];
  method_clusters: MethodCluster[];
  timeline: TimelineItem[];
  citation_graph: CitationGraph;
  warnings: string[];
  missing_evidence: string[];
  synthesis?: SynthesisOutput | null;
  retrieval_diagnostics: RetrievalDiagnostics;
  budget_status: BudgetStatus;
  cost_report: CostReport;
  judgement_policy: JudgementPolicy;
  judgement_config_hash: string;
}

export interface StreamEvent {
  event: string;
  payload: Record<string, unknown>;
  receivedAt: string;
}
