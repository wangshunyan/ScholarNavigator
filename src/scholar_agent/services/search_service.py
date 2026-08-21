"""Internal search pipeline service.

This service wires the backend modules into a real retrieval pipeline.
It is used by the Real Search FastAPI lifecycle endpoints.
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from multiprocessing import get_context
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from scholar_agent.agents.judgement import JudgementAgent
from scholar_agent.agents.judgement_config import (
    judgement_config_hash,
    resolve_judgement_config,
)
from scholar_agent.agents.query_evolution import (
    evolve_queries,
    filter_evolved_candidates,
)
from scholar_agent.agents.llm_feedback_evolution import evolve_with_llm_feedback
from scholar_agent.agents.pseudo_relevance_feedback import build_prf_plan
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.refchain import ReferenceFetcher, expand_refchain
from scholar_agent.agents.semantic_seed_expansion import (
    RecommendationFetcher,
    SeedResolver,
    expand_semantic_seeds,
)
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.agents.rrf_fusion import (
    build_retrieval_ranked_lists,
    fuse_ranked_papers,
)
from scholar_agent.agents.retriever import (
    RetrievalOutput,
    RetrievalRunContext,
    SourceStats,
    retrieve_papers,
)
from scholar_agent.connectors.openalex import fetch_openalex_references_detailed
from scholar_agent.connectors.semantic_scholar import (
    recommend_semantic_scholar_papers_detailed,
    resolve_semantic_scholar_paper_ids_detailed,
)
from scholar_agent.core.dedup import (
    deduplicate_papers,
    deduplicate_papers_with_audit,
    deduplicate_papers_with_lineage,
)
from scholar_agent.core.result_lineage import (
    opaque_query_identity,
    restrict_result_lineage_document,
)
from scholar_agent.core.untrusted_metadata import (
    UntrustedMetadataObserver,
    protect_source_error,
    safe_diagnostic_message,
)
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.pipeline_diagnostics import (
    PipelineDiagnosticsCollector,
    StageCandidateSnapshot,
)
from scholar_agent.core.search_schemas import (
    BudgetStatus,
    EvolvedSubquery,
    JudgementResult,
    JudgementPolicy,
    JudgementRuleConfig,
    QueryAnalysis,
    QueryConstraint,
    QueryEvolutionOptions,
    QueryEvolutionPolicy,
    QueryEvolutionRecord,
    QueryPlanningPolicy,
    RankingPolicy,
    RankedPaper,
    RefChainOptions,
    RefChainOutput,
    RunProfile,
    SearchPlan,
    SearchBudget,
    SearchSubquery,
    SemanticSeedExpansionOutput,
    SemanticSeedExpansionRecord,
    SUPPORTED_SEARCH_SOURCES,
    normalize_search_sources,
)
from scholar_agent.core.synthesis_schemas import SynthesisOutput
from scholar_agent.llm.provider import (
    LLMProviderError,
    OpenAICompatibleLLMClient,
    is_llm_enabled,
)
from scholar_agent.retrieval.query_adapter import (
    DEFAULT_QUERY_ADAPTER_POLICY,
    QueryAdapterPolicy,
)
from scholar_agent.services.search_budget import BudgetedLLMClient, SearchBudgetRuntime

ENABLE_LLM_QUERY_UNDERSTANDING_ENV = "SCHOLAR_AGENT_ENABLE_LLM_QUERY_UNDERSTANDING"
ENABLE_LLM_JUDGEMENT_ENV = "SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT"
RETRIEVAL_CLEANUP_GRACE_SECONDS = 0.5
ISOLATED_RETRIEVER_MIN_RUNTIME_SECONDS = 1.2

EventCallback = Callable[[str, dict[str, Any]], None]
ShouldCancel = Callable[[], bool]


def _isolated_retriever_entry(
    connection: Any,
    retriever: Any,
    query: str,
    limit_per_source: int,
    sources: list[str],
) -> None:
    """Spawn-safe bridge for retrievers that do not share run state."""

    try:
        result = retriever(query, limit_per_source=limit_per_source, sources=sources)
        connection.send(("ok", result))
    except BaseException as exc:  # pragma: no cover - exercised in child process
        connection.send(("error", type(exc).__name__, str(exc)[:500]))
    finally:
        connection.close()


class SearchCancelled(RuntimeError):
    """协作式取消信号；区别于检索失败。"""

    def __init__(self, stage: str) -> None:
        super().__init__(f"search_cancelled:{stage}")
        self.stage = stage


class SearchDeadlineExceeded(RuntimeError):
    """Search budget deadline reached before a retrieval task completed."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"search_deadline_exceeded:{stage}")
        self.stage = stage


class _ExecutionSignals:
    def __init__(
        self,
        event_callback: EventCallback | None,
        should_cancel: ShouldCancel | None,
        resource_accounting_observer: Any | None = None,
    ) -> None:
        self._event_callback = event_callback
        self._should_cancel = should_cancel
        self._resource_accounting_observer = resource_accounting_observer
        self.warnings: list[str] = []
        self._budget_reasons: set[str] = set()
        self._warning_events: set[str] = set()
        self._lock = RLock()
        self._deadline: float | None = None

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def remaining_seconds(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.perf_counter())

    def check_deadline(self, stage: str) -> None:
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            self._observe_cancellation(f"deadline:{stage}")
            raise SearchDeadlineExceeded(stage)

    def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        observation = getattr(
            self._resource_accounting_observer,
            "observe_semantic_event",
            None,
        )
        if callable(observation):
            try:
                observation(event_name, dict(payload or {}))
            except Exception:  # noqa: BLE001 - observation must not alter execution
                pass
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_name, dict(payload or {}))
        except Exception as exc:  # noqa: BLE001 - callback must not break search
            warning = f"event_callback_failed:{event_name}:{type(exc).__name__}"
            with self._lock:
                if warning not in self.warnings:
                    self.warnings.append(warning)

    def check_cancelled(self, stage: str) -> None:
        if self._should_cancel is None:
            return
        try:
            cancelled = bool(self._should_cancel())
        except Exception as exc:  # noqa: BLE001 - cancellation probe is advisory
            warning = f"cancellation_check_failed:{stage}:{type(exc).__name__}"
            with self._lock:
                if warning not in self.warnings:
                    self.warnings.append(warning)
            return
        if cancelled:
            self._observe_cancellation(stage)
            raise SearchCancelled(stage)

    def _observe_cancellation(self, stage: str) -> None:
        callback = getattr(
            self._resource_accounting_observer,
            "observe_cancellation",
            None,
        )
        if callable(callback):
            try:
                callback(stage)
            except Exception:  # noqa: BLE001 - observation must not alter execution
                pass

    def emit_budget_stops(
        self,
        runtime: SearchBudgetRuntime,
        stage: str,
    ) -> None:
        for reason in runtime.stop_reasons:
            if reason in self._budget_reasons:
                continue
            self._budget_reasons.add(reason)
            self.emit(
                "budget_stop",
                {
                    "stage": stage,
                    "reason": reason,
                    "budget_status": runtime.status().model_dump(mode="json"),
                },
            )

    def emit_warnings(self, warnings: list[str], stage: str) -> None:
        for warning in warnings:
            item = safe_diagnostic_message(warning)
            if not item or item in self._warning_events:
                continue
            self._warning_events.add(item)
            self.emit("warning", {"stage": stage, "message": item})


class RetrieverFn(Protocol):
    def __call__(
        self,
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        ...


class _RetrievalTaskResult(BaseModel):
    index: int
    output: RetrievalOutput


class _AdaptiveBudgetTracker:
    """在并行子查询之间共享 adaptive 候选与延迟预算视图。"""

    def __init__(self, runtime: SearchBudgetRuntime) -> None:
        self._runtime = runtime
        self._lock = RLock()
        self._papers: list[Paper] = []

    def check(self, papers: list[Paper]) -> str | None:
        with self._lock:
            if papers:
                self._papers = deduplicate_papers([*self._papers, *papers])
            latency_reason = self._runtime.latency_stop_reason()
            if latency_reason is not None:
                return latency_reason
            return self._runtime.candidate_stop_reason(len(self._papers))


class SearchServiceOutput(BaseModel):
    search_plan: SearchPlan
    retrieval_outputs: list[RetrievalOutput] = Field(default_factory=list)
    query_evolution_records: list[QueryEvolutionRecord] = Field(default_factory=list)
    refchain_output: RefChainOutput | None = None
    semantic_seed_expansion_output: SemanticSeedExpansionOutput | None = None
    synthesis_output: SynthesisOutput | None = None
    raw_count: int = 0
    deduplicated_count: int = 0
    judgements: list[JudgementResult] = Field(default_factory=list)
    ranked_papers: list[RankedPaper] = Field(default_factory=list)
    all_ranked_papers: list[RankedPaper] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_stats: list[SourceStats] = Field(default_factory=list)
    latency_seconds: float = 0.0
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    llm_call_count: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    search_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    reference_diagnostics: ConnectorDiagnostics = Field(
        default_factory=ConnectorDiagnostics
    )
    budget_status: BudgetStatus = Field(default_factory=BudgetStatus)
    stage_snapshots: list[StageCandidateSnapshot] = Field(default_factory=list)
    judgement_policy: JudgementPolicy = "current_rules"
    judgement_config_hash: str = ""

    @model_validator(mode="after")
    def derive_connector_diagnostics(self) -> "SearchServiceOutput":
        search = merge_connector_diagnostics(
            stat.diagnostics
            for stat in self.source_stats
            if stat.source not in {"refchain", "semantic_seed_expansion"}
        )
        reference = merge_connector_diagnostics(
            stat.diagnostics
            for stat in self.source_stats
            if stat.source in {"refchain", "semantic_seed_expansion"}
        )
        if (
            self.search_diagnostics == ConnectorDiagnostics()
            and search != ConnectorDiagnostics()
        ):
            self.search_diagnostics = search
        if (
            self.reference_diagnostics == ConnectorDiagnostics()
            and reference != ConnectorDiagnostics()
        ):
            self.reference_diagnostics = reference
        return self


class SearchService:
    """Run the internal real search pipeline."""

    def __init__(
        self,
        retriever: RetrieverFn = retrieve_papers,
        reference_fetcher: ReferenceFetcher = fetch_openalex_references_detailed,
        recommendation_fetcher: RecommendationFetcher = (
            recommend_semantic_scholar_papers_detailed
        ),
        semantic_seed_resolver: SeedResolver = (
            resolve_semantic_scholar_paper_ids_detailed
        ),
        max_workers: int = 4,
        llm_client: Any | None = None,
        llm_planning_runtime: Any | None = None,
        judgement_policy: JudgementPolicy = "current_rules",
        judgement_config: JudgementRuleConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._retriever_emits_connector_events = bool(
            retriever is retrieve_papers
            or getattr(retriever, "emits_connector_events", False)
        )
        self._process_isolation_available = self._can_isolate_retriever(retriever)
        self._isolated_processes: set[Any] = set()
        self._isolated_processes_lock = RLock()
        self._reference_fetcher = reference_fetcher
        self._recommendation_fetcher = recommendation_fetcher
        self._semantic_seed_resolver = semantic_seed_resolver
        self._max_workers = max(1, max_workers)
        self._llm_client = llm_client
        self._resolved_llm_client: Any | None = None
        self._llm_planning_runtime = llm_planning_runtime
        self._judgement_policy = judgement_policy
        self._judgement_config = judgement_config

    @staticmethod
    def _can_isolate_retriever(retriever: Any) -> bool:
        """Only isolate stateless, pickleable test/synthetic retrievers.

        The production retriever carries the per-run cache/locks and therefore
        stays in-process; its connector clients enforce their own HTTP bounds.
        Stateless retrievers are isolated so an uncooperative fake cannot leak
        a worker thread past a query deadline.
        """

        if retriever is retrieve_papers:
            return False
        try:
            pickle.dumps(retriever)
        except (pickle.PickleError, AttributeError, TypeError):
            return False
        return True

    def _register_isolated_process(self, process: Any) -> None:
        with self._isolated_processes_lock:
            self._isolated_processes.add(process)

    def _unregister_isolated_process(self, process: Any) -> None:
        with self._isolated_processes_lock:
            self._isolated_processes.discard(process)

    def _terminate_isolated_processes(self) -> None:
        with self._isolated_processes_lock:
            processes = list(self._isolated_processes)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            self._unregister_isolated_process(process)

    def run_search(
        self,
        query: str,
        top_k: int = 20,
        run_profile: RunProfile = "balanced",
        enable_refchain: bool = False,
        enable_semantic_seed_expansion: bool = False,
        enable_query_evolution: bool = False,
        query_evolution_policy: QueryEvolutionPolicy = "coverage_gap",
        query_planning_policy: QueryPlanningPolicy = "current_rules",
        ranking_policy: RankingPolicy = "current_rules",
        enable_synthesis: bool = True,
        current_year: int | None = None,
        enable_llm_query_understanding: bool | None = None,
        enable_llm_judgement: bool | None = None,
        judgement_policy: JudgementPolicy | None = None,
        judgement_config: JudgementRuleConfig | None = None,
        sources_override: list[str] | None = None,
        explicit_constraints: QueryConstraint | None = None,
        budget: SearchBudget | None = None,
        event_callback: EventCallback | None = None,
        should_cancel: ShouldCancel | None = None,
        collect_diagnostics: bool = False,
        query_adapter_policy: QueryAdapterPolicy = DEFAULT_QUERY_ADAPTER_POLICY,
        result_lineage_callback: Callable[[dict[str, Any]], None] | None = None,
        resource_accounting_observer: Any | None = None,
        untrusted_metadata_observer: UntrustedMetadataObserver | None = None,
    ) -> SearchServiceOutput:
        signals = _ExecutionSignals(
            event_callback,
            should_cancel,
            resource_accounting_observer,
        )
        diagnostics = PipelineDiagnosticsCollector(collect_diagnostics)
        retrieval_run_context = RetrievalRunContext()
        elapsed_seconds_provider = getattr(
            self._retriever,
            "budget_elapsed_seconds",
            None,
        )
        runtime = SearchBudgetRuntime(
            budget,
            elapsed_seconds_provider=(
                elapsed_seconds_provider
                if callable(elapsed_seconds_provider)
                else None
            ),
            resource_accounting_observer=resource_accounting_observer,
        )
        signals.set_deadline(
            runtime.started_at
            + runtime.budget.max_latency_seconds
            + RETRIEVAL_CLEANUP_GRACE_SECONDS
        )
        adaptive_budget_tracker = _AdaptiveBudgetTracker(runtime)
        effective_judgement_policy = judgement_policy or self._judgement_policy
        inherited_judgement_config = (
            self._judgement_config
            if effective_judgement_policy == self._judgement_policy
            else None
        )
        effective_judgement_config = resolve_judgement_config(
            effective_judgement_policy,
            judgement_config or inherited_judgement_config,
        )
        start = runtime.started_at
        stage_latencies: dict[str, float] = {}
        use_llm_query_understanding = (
            _env_flag(ENABLE_LLM_QUERY_UNDERSTANDING_ENV, default=False)
            if enable_llm_query_understanding is None
            else enable_llm_query_understanding
        )
        use_llm_judgement = (
            _env_flag(ENABLE_LLM_JUDGEMENT_ENV, default=False)
            if enable_llm_judgement is None
            else enable_llm_judgement
        )
        if (
            enable_query_evolution
            and query_evolution_policy == "llm_feedback"
            and (
                query_planning_policy
                in {"llm_semantic", "llm_constrained_rewrite"}
                or use_llm_query_understanding
                or use_llm_judgement
            )
        ):
            raise ValueError(
                "llm_feedback requires rule-only initial planning and judgement"
            )
        resolved_llm_client = self._resolve_llm_client(
            use_llm_query_understanding
            or use_llm_judgement
            or query_planning_policy
            in {"llm_semantic", "llm_constrained_rewrite"}
            or (
                enable_query_evolution
                and query_evolution_policy == "llm_feedback"
            )
        )
        llm_client = (
            BudgetedLLMClient(resolved_llm_client, runtime)
            if resolved_llm_client is not None
            else None
        )
        signals.check_cancelled("query_understanding:before")
        signals.emit(
            "query_understanding_started",
            {"stage": "query_understanding"},
        )
        stage_start = time.perf_counter()
        search_plan = analyze_query(
            query,
            top_k=top_k,
            run_profile=run_profile,
            enable_refchain=enable_refchain,
            enable_semantic_seed_expansion=enable_semantic_seed_expansion,
            enable_query_evolution=enable_query_evolution,
            query_planning_policy=query_planning_policy,
            current_year=current_year,
            use_llm=use_llm_query_understanding,
            llm_client=llm_client,
            llm_planning_runtime=self._llm_planning_runtime,
            explicit_constraints=explicit_constraints,
        )
        search_plan = _apply_sources_override(search_plan, sources_override)
        effective_query_evolution_policy: QueryEvolutionPolicy = (
            query_evolution_policy if enable_query_evolution else "off"
        )
        search_plan = search_plan.model_copy(
            update={
                "enable_query_evolution": enable_query_evolution,
                "query_evolution_policy": effective_query_evolution_policy,
                "ranking_policy": ranking_policy,
            }
        )
        signals.check_cancelled("query_understanding:after")
        query_understanding_latency = time.perf_counter() - stage_start
        _add_stage_latency(
            stage_latencies,
            "query_understanding",
            query_understanding_latency,
        )
        signals.emit(
            "query_understanding_completed",
            {
                "stage": "query_understanding",
                "subquery_count": len(search_plan.subqueries),
                "latency_seconds": query_understanding_latency,
            },
        )

        warnings: list[str] = list(search_plan.warnings)
        retrieval_outputs: list[RetrievalOutput] = []
        query_evolution_records: list[QueryEvolutionRecord] = []
        refchain_output: RefChainOutput | None = None
        semantic_seed_expansion_output: SemanticSeedExpansionOutput | None = None
        raw_papers: list[Paper] = []
        source_stats: list[SourceStats] = []
        deduplicated: list[Paper] = []
        judgements: list[JudgementResult] = []
        all_ranked_papers: list[RankedPaper] = []
        ranked_papers: list[RankedPaper] = []

        if runtime.latency_stop_reason() is None:
            signals.check_cancelled("retrieval:before_initial_batch")
            signals.emit(
                "retrieval_started",
                {
                    "stage": "retrieval",
                    "round_index": 1,
                    "query_count": len(search_plan.subqueries),
                },
            )
            stage_start = time.perf_counter()
            initial_subqueries = search_plan.subqueries
            if initial_subqueries:
                if query_planning_policy == "prf_v1":
                    retrieval_outputs, search_plan = self._retrieve_prf_initial_queries(
                        search_plan,
                        signals=signals,
                        run_context=retrieval_run_context,
                        query_adapter_policy=query_adapter_policy,
                        adaptive_budget_check=adaptive_budget_tracker.check,
                        judgement_policy=effective_judgement_policy,
                        judgement_config=effective_judgement_config,
                        ranking_policy=ranking_policy,
                    )
                    initial_subqueries = search_plan.subqueries
                elif query_planning_policy in {
                    "current_plus_disjunctive",
                    "facet_union",
                }:
                    retrieval_outputs = (
                        self._retrieve_baseline_then_supplemental(
                            search_plan,
                            supplemental_purpose=(
                                "current_plus_disjunctive_any"
                                if query_planning_policy
                                == "current_plus_disjunctive"
                                else "facet_union_"
                            ),
                            warning_prefix=query_planning_policy,
                            signals=signals,
                            runtime=runtime,
                            run_context=retrieval_run_context,
                            query_adapter_policy=query_adapter_policy,
                            adaptive_budget_check=adaptive_budget_tracker.check,
                        )
                    )
                else:
                    retrieval_outputs = self._retrieve_subqueries(
                        search_plan,
                        subqueries=initial_subqueries,
                        signals=signals,
                        run_context=retrieval_run_context,
                        query_adapter_policy=query_adapter_policy,
                        adaptive_budget_check=adaptive_budget_tracker.check,
                    )
                runtime.record_search_round()
            signals.check_cancelled("retrieval:after_initial_batch")
            retrieval_latency = time.perf_counter() - stage_start
            _add_stage_latency(
                stage_latencies,
                "retrieval",
                retrieval_latency,
            )
            raw_papers, source_stats, retrieval_warnings = _collect_retrieval_outputs(
                retrieval_outputs
            )
            diagnostics.register_retrieval(
                "initial_retrieval",
                retrieval_outputs,
                origin_kind_by_query={
                    subquery.query: (
                        "initial_query"
                        if subquery.purpose == "original_query"
                        else "initial_generated_subquery"
                    )
                    for subquery in initial_subqueries
                },
            )
            warnings.extend(retrieval_warnings)
            signals.emit_warnings(retrieval_warnings, "retrieval")
            signals.emit(
                "retrieval_completed",
                {
                    "stage": "retrieval",
                    "round_index": 1,
                    "raw_candidate_count": len(raw_papers),
                    "latency_seconds": retrieval_latency,
                },
            )
            deduplicated_papers, identity_audit = deduplicate_papers_with_audit(raw_papers)
            deduplicated = _apply_candidate_budget(
                deduplicated_papers,
                runtime,
                stage="initial_retrieval",
                source_order=search_plan.selected_sources,
            )
            diagnostics.snapshot_papers(
                "initial_deduplicated", deduplicated, identity_audit=identity_audit
            )
            signals.emit(
                "deduplication_completed",
                {
                    "stage": "deduplication",
                    "round_index": 1,
                    "raw_candidate_count": len(raw_papers),
                    "deduplicated_candidate_count": len(deduplicated),
                },
            )
            signals.emit_budget_stops(runtime, "deduplication")
            signals.check_cancelled("judgement:before")
            signals.emit(
                "judgement_started",
                {
                    "stage": "judgement",
                    "candidate_paper_count": len(deduplicated),
                },
            )
            stage_start = time.perf_counter()
            judgement_use_llm = (
                use_llm_judgement and runtime.latency_stop_reason() is None
            )
            judgements, _ = self._judge_papers(
                search_plan,
                deduplicated,
                use_llm=judgement_use_llm,
                llm_client=llm_client,
                signals=signals,
                judgement_policy=effective_judgement_policy,
                judgement_config=effective_judgement_config,
                metadata_observer=untrusted_metadata_observer,
            )
            diagnostics.snapshot_judgements("initial_judged", judgements)
            signals.check_cancelled("judgement:after")
            judgement_latency = time.perf_counter() - stage_start
            _add_stage_latency(
                stage_latencies,
                "judgement",
                judgement_latency,
            )
            signals.emit(
                "judgement_completed",
                {
                    "stage": "judgement",
                    "judged_paper_count": len(judgements),
                    "latency_seconds": judgement_latency,
                },
            )
            runtime.latency_stop_reason()
            signals.emit_budget_stops(runtime, "judgement")
            signals.check_cancelled("reranking:before")
            signals.emit(
                "reranking_started",
                {
                    "stage": "reranking",
                    "judged_paper_count": len(judgements),
                },
            )
            stage_start = time.perf_counter()
            all_ranked_papers, ranked_papers = _rerank_all_and_top(
                search_plan.query_analysis,
                judgements,
                top_k,
                ranking_policy=ranking_policy,
                retrieval_outputs=retrieval_outputs,
            )
            diagnostics.snapshot_ranked("initial_reranked", all_ranked_papers)
            signals.check_cancelled("reranking:after")
            reranking_latency = time.perf_counter() - stage_start
            _add_stage_latency(
                stage_latencies,
                "reranking",
                reranking_latency,
            )
            signals.emit(
                "reranking_completed",
                {
                    "stage": "reranking",
                    "ranked_paper_count": len(ranked_papers),
                    "latency_seconds": reranking_latency,
                },
            )
        else:
            signals.emit_budget_stops(runtime, "query_understanding")
            for stage in (
                "initial_retrieval",
                "initial_deduplicated",
                "initial_judged",
                "initial_reranked",
            ):
                diagnostics.skip(stage, "budget_stopped_before_retrieval")

        if enable_semantic_seed_expansion:
            if enable_query_evolution or enable_refchain:
                raise ValueError(
                    "semantic_seed_expansion_requires_query_evolution_and_refchain_disabled"
                )
            signals.check_cancelled("semantic_seed_expansion:before")
            signals.emit(
                "semantic_seed_expansion_started",
                {"stage": "semantic_seed_expansion"},
            )
            stage_start = time.perf_counter()
            stop_reason = (
                runtime.stop("budget_stop:max_search_rounds")
                if runtime.completed_search_rounds
                >= runtime.budget.max_search_rounds
                else runtime.latency_stop_reason()
            )
            if stop_reason is not None:
                semantic_seed_expansion_output = SemanticSeedExpansionOutput(
                    record=SemanticSeedExpansionRecord(
                        status="source_failure",
                        skip_reason=stop_reason,
                        warnings=[stop_reason],
                    ),
                    warnings=[stop_reason],
                )
                for stage in (
                    "semantic_seed_expansion_retrieval",
                    "post_semantic_seed_expansion_deduplicated",
                    "post_semantic_seed_expansion_judged",
                    "post_semantic_seed_expansion_reranked",
                ):
                    diagnostics.skip(stage, stop_reason)
            else:
                semantic_seed_expansion_output = expand_semantic_seeds(
                    all_ranked_papers,
                    self._recommendation_fetcher,
                    resolve_seed_ids=self._semantic_seed_resolver,
                    max_seeds=3,
                    limit=100,
                )
                if semantic_seed_expansion_output.record.seeds:
                    runtime.record_search_round()
                recommendations = semantic_seed_expansion_output.recommendations
                diagnostics.register_semantic_seed_expansion(
                    "semantic_seed_expansion_retrieval",
                    recommendations,
                )
                warnings.extend(semantic_seed_expansion_output.warnings)
                signals.emit_warnings(
                    semantic_seed_expansion_output.warnings,
                    "semantic_seed_expansion",
                )
                source_stats.append(
                    SourceStats(
                        source="semantic_seed_expansion",
                        terminal_status=semantic_seed_expansion_output.record.status,
                        query="semantic_seed_expansion",
                        returned_count=len(recommendations),
                        latency_seconds=(
                            semantic_seed_expansion_output.latency_seconds
                        ),
                        error_message=(
                            semantic_seed_expansion_output.record.skip_reason
                            if semantic_seed_expansion_output.record.status
                            == "source_failure"
                            else None
                        ),
                        logical_call_executed=bool(
                            semantic_seed_expansion_output.record.seeds
                            or semantic_seed_expansion_output.record.resolution_request_identifier_count
                        ),
                        warnings=list(semantic_seed_expansion_output.warnings),
                        diagnostics=semantic_seed_expansion_output.diagnostics,
                        snapshot_key=(
                            semantic_seed_expansion_output.record.snapshot_key
                            or semantic_seed_expansion_output.record.resolution_snapshot_key
                        ),
                        recorded_diagnostics=(
                            semantic_seed_expansion_output.record.recorded_diagnostics
                        ),
                        recorded_latency_seconds=(
                            semantic_seed_expansion_output.record.recorded_latency_seconds
                        ),
                    )
                )
                if recommendations:
                    prior_unique_count = len(deduplicated)
                    prior_merge_count = len(identity_audit)
                    raw_papers.extend(recommendations)
                    deduplicated_papers, identity_audit = (
                        deduplicate_papers_with_audit(raw_papers)
                    )
                    new_unique_count = max(
                        0,
                        len(deduplicated_papers) - prior_unique_count,
                    )
                    expansion_merges = identity_audit[prior_merge_count:]
                    semantic_seed_expansion_output.record = (
                        semantic_seed_expansion_output.record.model_copy(
                            update={
                                "new_unique_candidate_count": new_unique_count,
                                "duplicate_candidate_count": max(
                                    0,
                                    len(recommendations) - new_unique_count,
                                ),
                                "identity_merges": expansion_merges,
                            }
                        )
                    )
                    deduplicated = _apply_candidate_budget(
                        deduplicated_papers,
                        runtime,
                        stage="semantic_seed_expansion",
                        source_order=search_plan.selected_sources,
                    )
                    diagnostics.snapshot_papers(
                        "post_semantic_seed_expansion_deduplicated",
                        deduplicated,
                        identity_audit=expansion_merges,
                    )
                    signals.check_cancelled(
                        "judgement:before_semantic_seed_expansion"
                    )
                    judgement_use_llm = (
                        use_llm_judgement
                        and runtime.latency_stop_reason() is None
                    )
                    judgements, _ = self._judge_papers(
                        search_plan,
                        deduplicated,
                        use_llm=judgement_use_llm,
                        llm_client=llm_client,
                        signals=signals,
                        judgement_policy=effective_judgement_policy,
                        judgement_config=effective_judgement_config,
                        metadata_observer=untrusted_metadata_observer,
                    )
                    diagnostics.snapshot_judgements(
                        "post_semantic_seed_expansion_judged",
                        judgements,
                    )
                    all_ranked_papers, ranked_papers = _rerank_all_and_top(
                        search_plan.query_analysis,
                        judgements,
                        top_k,
                        ranking_policy=ranking_policy,
                        retrieval_outputs=retrieval_outputs,
                    )
                    diagnostics.snapshot_ranked(
                        "post_semantic_seed_expansion_reranked",
                        all_ranked_papers,
                    )
                else:
                    reason = (
                        semantic_seed_expansion_output.record.skip_reason
                        or "no_recommendation_candidates"
                    )
                    for stage in (
                        "post_semantic_seed_expansion_deduplicated",
                        "post_semantic_seed_expansion_judged",
                        "post_semantic_seed_expansion_reranked",
                    ):
                        diagnostics.skip(stage, reason)
            signals.check_cancelled("semantic_seed_expansion:after")
            _add_stage_latency(
                stage_latencies,
                "semantic_seed_expansion",
                time.perf_counter() - stage_start,
            )
            signals.emit(
                "semantic_seed_expansion_completed",
                {
                    "stage": "semantic_seed_expansion",
                    "status": (
                        semantic_seed_expansion_output.record.status
                        if semantic_seed_expansion_output is not None
                        else "source_failure"
                    ),
                },
            )
        else:
            for stage in (
                "semantic_seed_expansion_retrieval",
                "post_semantic_seed_expansion_deduplicated",
                "post_semantic_seed_expansion_judged",
                "post_semantic_seed_expansion_reranked",
            ):
                diagnostics.skip(stage, "disabled")

        if enable_query_evolution:
            initial_deduplicated = list(deduplicated)
            signals.check_cancelled("query_evolution:before")
            evolution_stop_reason = _query_evolution_budget_stop_reason(
                runtime,
                len(deduplicated),
            )
            if evolution_stop_reason is not None:
                query_evolution_records.append(
                    QueryEvolutionRecord(
                        round_index=2,
                        policy=effective_query_evolution_policy,
                        skipped_reasons=[evolution_stop_reason],
                        warnings=[evolution_stop_reason],
                    )
                )
                signals.emit_budget_stops(runtime, "query_evolution")
                signals.emit(
                    "query_evolution_skipped",
                    {
                        "stage": "query_evolution",
                        "reason": evolution_stop_reason,
                    },
                )
                for stage in (
                    "query_evolution_retrieval",
                    "post_evolution_deduplicated",
                    "post_evolution_judged",
                    "post_evolution_reranked",
                ):
                    diagnostics.skip(stage, evolution_stop_reason)
            else:
                signals.emit(
                    "query_evolution_started",
                    {"stage": "query_evolution"},
                )
                used_queries = {subquery.query for subquery in search_plan.subqueries}
                stage_start = time.perf_counter()
                if effective_query_evolution_policy == "llm_feedback":
                    evolution_record = evolve_with_llm_feedback(
                        search_plan.query_analysis,
                        search_plan,
                        judgements,
                        ranked_papers,
                        used_queries,
                        llm_client=llm_client,
                    )
                else:
                    evolution_record = evolve_queries(
                        search_plan.query_analysis,
                        search_plan,
                        judgements,
                        ranked_papers,
                        used_queries,
                        options=QueryEvolutionOptions(
                            policy=effective_query_evolution_policy,
                        ),
                    )
                evolution_record.round_index = 2
                query_evolution_records.append(evolution_record)
                warnings.extend(evolution_record.warnings)

                evolved_queries = _filter_new_evolved_queries(
                    evolution_record.generated_queries,
                    used_queries,
                )
                if len(evolved_queries) < len(evolution_record.generated_queries):
                    warnings.append("duplicate_evolved_query_skipped")
                evolution_generation_latency = time.perf_counter() - stage_start
                evolution_record.latency_seconds = evolution_generation_latency
                _add_stage_latency(
                    stage_latencies,
                    "query_evolution",
                    evolution_generation_latency,
                )
                retrieval_stop_reason = _query_evolution_budget_stop_reason(
                    runtime,
                    len(deduplicated),
                    check_rounds=False,
                )
                if retrieval_stop_reason is not None:
                    evolution_record.skipped_reasons = _dedupe_warnings(
                        [*evolution_record.skipped_reasons, retrieval_stop_reason]
                    )
                    evolution_record.warnings = _dedupe_warnings(
                        [*evolution_record.warnings, retrieval_stop_reason]
                    )
                    signals.emit_budget_stops(runtime, "query_evolution")
                    for stage in (
                        "query_evolution_retrieval",
                        "post_evolution_deduplicated",
                        "post_evolution_judged",
                        "post_evolution_reranked",
                    ):
                        diagnostics.skip(stage, retrieval_stop_reason)
                elif evolved_queries:
                    signals.check_cancelled("retrieval:before_evolved_batch")
                    signals.emit(
                        "retrieval_started",
                        {
                            "stage": "retrieval",
                            "round_index": 2,
                            "query_count": len(evolved_queries),
                        },
                    )
                    stage_start = time.perf_counter()
                    evolved_outputs = self._retrieve_evolved_queries(
                        search_plan,
                        evolved_queries,
                        signals=signals,
                        run_context=retrieval_run_context,
                        query_adapter_policy=query_adapter_policy,
                        adaptive_budget_check=adaptive_budget_tracker.check,
                    )
                    runtime.record_search_round()
                    signals.check_cancelled("retrieval:after_evolved_batch")
                    evolved_retrieval_latency = time.perf_counter() - stage_start
                    _add_stage_latency(
                        stage_latencies,
                        "retrieval",
                        evolved_retrieval_latency,
                    )
                    if collect_diagnostics:
                        _add_stage_latency(
                            stage_latencies,
                            "query_evolution_retrieval",
                            evolved_retrieval_latency,
                        )
                    retrieval_outputs.extend(evolved_outputs)
                    (
                        evolved_papers,
                        evolved_source_stats,
                        evolved_warnings,
                    ) = _collect_retrieval_outputs(evolved_outputs)
                    if effective_query_evolution_policy in {
                        "coverage_gap",
                        "llm_feedback",
                    }:
                        evolved_papers, quality_gate = filter_evolved_candidates(
                            search_plan.query_analysis,
                            initial_deduplicated,
                            evolved_papers,
                        )
                        evolution_record.quality_gate = quality_gate
                    diagnostics.register_retrieval(
                        "query_evolution_retrieval",
                        evolved_outputs,
                        origin_kind_by_query={
                            item.query: "query_evolution" for item in evolved_queries
                        },
                    )
                    raw_papers.extend(evolved_papers)
                    source_stats.extend(evolved_source_stats)
                    warnings.extend(evolved_warnings)
                    signals.emit_warnings(evolved_warnings, "retrieval")
                    signals.emit(
                        "retrieval_completed",
                        {
                            "stage": "retrieval",
                            "round_index": 2,
                            "raw_candidate_count": len(evolved_papers),
                            "latency_seconds": evolved_retrieval_latency,
                        },
                    )

                    deduplicated_papers, identity_audit = deduplicate_papers_with_audit(raw_papers)
                    deduplicated = _apply_candidate_budget(
                        deduplicated_papers,
                        runtime,
                        stage="query_evolution",
                        source_order=search_plan.selected_sources,
                    )
                    diagnostics.snapshot_papers(
                        "post_evolution_deduplicated",
                        deduplicated,
                        identity_audit=identity_audit,
                    )
                    signals.emit(
                        "deduplication_completed",
                        {
                            "stage": "deduplication",
                            "round_index": 2,
                            "raw_candidate_count": len(raw_papers),
                            "deduplicated_candidate_count": len(deduplicated),
                        },
                    )
                    signals.emit_budget_stops(runtime, "deduplication")
                    signals.check_cancelled("judgement:before_evolved")
                    signals.emit(
                        "judgement_started",
                        {
                            "stage": "judgement",
                            "round_index": 2,
                            "candidate_paper_count": len(deduplicated),
                        },
                    )
                    stage_start = time.perf_counter()
                    judgement_use_llm = (
                        use_llm_judgement
                        and runtime.latency_stop_reason() is None
                    )
                    judgements, _ = self._judge_papers(
                        search_plan,
                        deduplicated,
                        use_llm=judgement_use_llm,
                        llm_client=llm_client,
                        signals=signals,
                        judgement_policy=effective_judgement_policy,
                        judgement_config=effective_judgement_config,
                        metadata_observer=untrusted_metadata_observer,
                    )
                    diagnostics.snapshot_judgements(
                        "post_evolution_judged",
                        judgements,
                    )
                    signals.check_cancelled("judgement:after_evolved")
                    evolved_judgement_latency = time.perf_counter() - stage_start
                    _add_stage_latency(
                        stage_latencies,
                        "judgement",
                        evolved_judgement_latency,
                    )
                    if collect_diagnostics:
                        _add_stage_latency(
                            stage_latencies,
                            "query_evolution_judgement",
                            evolved_judgement_latency,
                        )
                    signals.emit(
                        "judgement_completed",
                        {
                            "stage": "judgement",
                            "round_index": 2,
                            "judged_paper_count": len(judgements),
                            "latency_seconds": evolved_judgement_latency,
                        },
                    )
                    runtime.latency_stop_reason()
                    signals.emit_budget_stops(runtime, "judgement")
                    signals.check_cancelled("reranking:before_evolved")
                    signals.emit(
                        "reranking_started",
                        {
                            "stage": "reranking",
                            "round_index": 2,
                            "judged_paper_count": len(judgements),
                        },
                    )
                    stage_start = time.perf_counter()
                    all_ranked_papers, ranked_papers = _rerank_all_and_top(
                        search_plan.query_analysis,
                        judgements,
                        top_k,
                        ranking_policy=ranking_policy,
                        retrieval_outputs=retrieval_outputs,
                    )
                    diagnostics.snapshot_ranked(
                        "post_evolution_reranked",
                        all_ranked_papers,
                    )
                    signals.check_cancelled("reranking:after_evolved")
                    evolved_reranking_latency = time.perf_counter() - stage_start
                    _add_stage_latency(
                        stage_latencies,
                        "reranking",
                        evolved_reranking_latency,
                    )
                    if collect_diagnostics:
                        _add_stage_latency(
                            stage_latencies,
                            "query_evolution_reranking",
                            evolved_reranking_latency,
                        )
                    signals.emit(
                        "reranking_completed",
                        {
                            "stage": "reranking",
                            "round_index": 2,
                            "ranked_paper_count": len(ranked_papers),
                            "latency_seconds": evolved_reranking_latency,
                        },
                    )
                else:
                    for stage in (
                        "query_evolution_retrieval",
                        "post_evolution_deduplicated",
                        "post_evolution_judged",
                        "post_evolution_reranked",
                    ):
                        diagnostics.skip(stage, "no_evolved_queries")
                signals.check_cancelled("query_evolution:after")
                signals.emit(
                    "query_evolution_completed",
                    {
                        "stage": "query_evolution",
                        "generated_query_count": len(evolved_queries),
                        "used_query_count": len(evolved_queries)
                        if retrieval_stop_reason is None
                        else 0,
                    },
                )
        else:
            signals.emit(
                "query_evolution_skipped",
                {"stage": "query_evolution", "reason": "disabled"},
            )
            for stage in (
                "query_evolution_retrieval",
                "post_evolution_deduplicated",
                "post_evolution_judged",
                "post_evolution_reranked",
            ):
                diagnostics.skip(stage, "disabled")

        if enable_refchain:
            signals.check_cancelled("refchain:before")
            signals.emit("refchain_started", {"stage": "refchain"})
            stage_start = time.perf_counter()
            runtime.latency_stop_reason()
            signals.emit_budget_stops(runtime, "refchain")
            remaining_candidates = max(
                0,
                runtime.budget.max_candidate_papers - len(deduplicated),
            )
            refchain_output = expand_refchain(
                search_plan.query_analysis,
                ranked_papers,
                self._reference_fetcher,
                options=RefChainOptions(
                    max_total_references=min(50, remaining_candidates),
                ),
                budget_check=lambda: (
                    runtime.latency_stop_reason()
                    or runtime.candidate_stop_reason(len(deduplicated))
                ),
                cancel_check=lambda: signals.check_cancelled(
                    "refchain:between_seeds"
                ),
            )
            signals.check_cancelled("refchain:after")
            raw_papers.extend(refchain_output.references)
            diagnostics.register_refchain(
                "refchain_retrieval",
                refchain_output.references,
            )
            warnings.extend(refchain_output.warnings)
            signals.emit_warnings(refchain_output.warnings, "refchain")
            source_stats.append(
                SourceStats(
                    source="refchain",
                    query="refchain",
                    returned_count=len(refchain_output.references),
                    latency_seconds=refchain_output.latency_seconds,
                    error_message=(
                        f"refchain_connector_errors:"
                        f"{refchain_output.diagnostics.error_count}"
                        if refchain_output.diagnostics.error_count
                        else None
                    ),
                    diagnostics=refchain_output.diagnostics,
                )
            )
            _add_stage_latency(
                stage_latencies,
                "refchain",
                time.perf_counter() - stage_start,
            )
            signals.emit(
                "refchain_completed",
                {
                    "stage": "refchain",
                    "seed_count": len(refchain_output.record.seeds),
                    "returned_count": len(refchain_output.references),
                    "request_count": refchain_output.diagnostics.request_count,
                    "retry_count": refchain_output.diagnostics.retry_count,
                    "latency_seconds": refchain_output.latency_seconds,
                },
            )
            if refchain_output.references:
                deduplicated_papers, identity_audit = deduplicate_papers_with_audit(raw_papers)
                deduplicated = _apply_candidate_budget(
                    deduplicated_papers,
                    runtime,
                    stage="refchain",
                    source_order=search_plan.selected_sources,
                )
                diagnostics.snapshot_papers(
                    "post_refchain_deduplicated",
                    deduplicated,
                    identity_audit=identity_audit,
                )
                signals.emit(
                    "deduplication_completed",
                    {
                        "stage": "deduplication",
                        "round_index": "refchain",
                        "raw_candidate_count": len(raw_papers),
                        "deduplicated_candidate_count": len(deduplicated),
                    },
                )
                signals.emit_budget_stops(runtime, "deduplication")
                signals.check_cancelled("judgement:before_refchain")
                signals.emit(
                    "judgement_started",
                    {
                        "stage": "judgement",
                        "round_index": "refchain",
                        "candidate_paper_count": len(deduplicated),
                    },
                )
                stage_start = time.perf_counter()
                judgement_use_llm = (
                    use_llm_judgement and runtime.latency_stop_reason() is None
                )
                judgements, _ = self._judge_papers(
                    search_plan,
                    deduplicated,
                    use_llm=judgement_use_llm,
                    llm_client=llm_client,
                    signals=signals,
                    judgement_policy=effective_judgement_policy,
                    judgement_config=effective_judgement_config,
                    metadata_observer=untrusted_metadata_observer,
                )
                diagnostics.snapshot_judgements(
                    "post_refchain_judged",
                    judgements,
                )
                signals.check_cancelled("judgement:after_refchain")
                refchain_judgement_latency = time.perf_counter() - stage_start
                _add_stage_latency(
                    stage_latencies,
                    "judgement",
                    refchain_judgement_latency,
                )
                if collect_diagnostics:
                    _add_stage_latency(
                        stage_latencies,
                        "refchain_judgement",
                        refchain_judgement_latency,
                    )
                signals.emit(
                    "judgement_completed",
                    {
                        "stage": "judgement",
                        "round_index": "refchain",
                        "judged_paper_count": len(judgements),
                        "latency_seconds": refchain_judgement_latency,
                    },
                )
                runtime.latency_stop_reason()
                signals.emit_budget_stops(runtime, "judgement")
                signals.check_cancelled("reranking:before_refchain")
                signals.emit(
                    "reranking_started",
                    {
                        "stage": "reranking",
                        "round_index": "refchain",
                        "judged_paper_count": len(judgements),
                    },
                )
                stage_start = time.perf_counter()
                all_ranked_papers, ranked_papers = _rerank_all_and_top(
                    search_plan.query_analysis,
                    judgements,
                    top_k,
                    ranking_policy=ranking_policy,
                    retrieval_outputs=retrieval_outputs,
                )
                diagnostics.snapshot_ranked(
                    "post_refchain_reranked",
                    all_ranked_papers,
                )
                signals.check_cancelled("reranking:after_refchain")
                refchain_reranking_latency = time.perf_counter() - stage_start
                _add_stage_latency(
                    stage_latencies,
                    "reranking",
                    refchain_reranking_latency,
                )
                if collect_diagnostics:
                    _add_stage_latency(
                        stage_latencies,
                        "refchain_reranking",
                        refchain_reranking_latency,
                    )
                signals.emit(
                    "reranking_completed",
                    {
                        "stage": "reranking",
                        "round_index": "refchain",
                        "ranked_paper_count": len(ranked_papers),
                        "latency_seconds": refchain_reranking_latency,
                    },
                )
            else:
                for stage in (
                    "post_refchain_deduplicated",
                    "post_refchain_judged",
                    "post_refchain_reranked",
                ):
                    diagnostics.skip(stage, "no_refchain_candidates")
        else:
            signals.emit(
                "refchain_skipped",
                {"stage": "refchain", "reason": "disabled"},
            )
            for stage in (
                "refchain_retrieval",
                "post_refchain_deduplicated",
                "post_refchain_judged",
                "post_refchain_reranked",
            ):
                diagnostics.skip(stage, "disabled")

        judgement_warnings = _judgement_warnings(judgements)
        warnings.extend(judgement_warnings)
        signals.emit_warnings(judgement_warnings, "judgement")
        signals.emit_budget_stops(runtime, "finalization")
        signals.check_cancelled("finalization:before_output")
        diagnostics.snapshot_ranked("final_ranked", all_ranked_papers)
        refchain_raw_count = (
            refchain_output.record.raw_reference_count
            if refchain_output is not None
            else 0
        )
        semantic_seed_raw_count = (
            semantic_seed_expansion_output.record.raw_recommendation_count
            if semantic_seed_expansion_output is not None
            else 0
        )
        output = SearchServiceOutput(
            search_plan=search_plan,
            retrieval_outputs=retrieval_outputs,
            query_evolution_records=query_evolution_records,
            refchain_output=refchain_output,
            semantic_seed_expansion_output=semantic_seed_expansion_output,
            raw_count=sum(output.raw_count for output in retrieval_outputs)
            + refchain_raw_count
            + semantic_seed_raw_count,
            deduplicated_count=len(deduplicated),
            judgements=judgements,
            ranked_papers=ranked_papers,
            all_ranked_papers=all_ranked_papers,
            warnings=_dedupe_warnings([*warnings, *signals.warnings]),
            source_stats=source_stats,
            latency_seconds=0.0,
            stage_latencies=stage_latencies,
            llm_call_count=runtime.used_llm_calls,
            llm_prompt_tokens=runtime.prompt_tokens,
            llm_completion_tokens=runtime.completion_tokens,
            llm_total_tokens=runtime.total_tokens,
            search_diagnostics=merge_connector_diagnostics(
                stat.diagnostics
                for stat in source_stats
                if stat.source not in {"refchain", "semantic_seed_expansion"}
            ),
            reference_diagnostics=merge_connector_diagnostics(
                stat.diagnostics
                for stat in source_stats
                if stat.source in {"refchain", "semantic_seed_expansion"}
            ),
            stage_snapshots=diagnostics.snapshots,
            judgement_policy=effective_judgement_policy,
            judgement_config_hash=judgement_config_hash(
                effective_judgement_config
            ),
        )
        if enable_synthesis and runtime.latency_stop_reason() is None:
            signals.check_cancelled("synthesis:before")
            signals.emit(
                "synthesis_started",
                {
                    "stage": "synthesis",
                    "ranked_paper_count": len(ranked_papers),
                },
            )
            from scholar_agent.agents.synthesis import synthesize_answer

            stage_start = time.perf_counter()
            output.synthesis_output = synthesize_answer(output)
            signals.check_cancelled("synthesis:after")
            synthesis_latency = time.perf_counter() - stage_start
            _add_stage_latency(
                output.stage_latencies,
                "synthesis",
                synthesis_latency,
            )
            signals.emit(
                "synthesis_completed",
                {
                    "stage": "synthesis",
                    "status": output.synthesis_output.status,
                    "latency_seconds": synthesis_latency,
                },
            )
        output.latency_seconds = time.perf_counter() - start
        output.budget_status = runtime.status()
        source_observer = getattr(
            resource_accounting_observer,
            "observe_source_stats",
            None,
        )
        if callable(source_observer):
            try:
                source_observer(
                    [
                        {
                            "source": item.source,
                            "returned_count": item.returned_count,
                            "request_count": item.diagnostics.request_count,
                            "retry_count": item.diagnostics.retry_count,
                            "error_count": item.diagnostics.error_count,
                            "cache_hit": item.cache_hit,
                            "cache_hit_count": item.diagnostics.cache_hit_count,
                            "logical_call_executed": item.logical_call_executed,
                            "source_skipped_reason": item.source_skipped_reason,
                            "error_message": (
                                safe_diagnostic_message(item.error_message)
                                if item.error_message is not None
                                else None
                            ),
                        }
                        for item in source_stats
                    ]
                )
            except Exception:  # noqa: BLE001 - observation must not alter execution
                pass
        runtime.finalize_resource_accounting(len(deduplicated))
        output.warnings = _dedupe_warnings(
            [
                *warnings,
                *runtime.stop_reasons,
                *runtime.diagnostics,
                *signals.warnings,
            ]
        )
        signals.emit_budget_stops(runtime, "completed")
        signals.emit_warnings(output.warnings, "completed")
        if result_lineage_callback is not None:
            query_identity = opaque_query_identity(query)
            if untrusted_metadata_observer is not None:
                for item in source_stats:
                    if item.error_message is not None:
                        protect_source_error(
                            item.error_message,
                            source=item.source,
                            query_identity=query_identity,
                            observer=untrusted_metadata_observer,
                        )
            isolation_document = (
                untrusted_metadata_observer.document(query_identity).model_dump(
                    mode="json"
                )
                if untrusted_metadata_observer is not None
                else None
            )
            _, _, lineage = deduplicate_papers_with_lineage(
                raw_papers,
                query_identity=query_identity,
                source_terminals=_result_lineage_source_terminals(
                    source_stats, search_plan.selected_sources, raw_papers
                ),
                untrusted_metadata_isolation=isolation_document,
            )
            result_lineage_callback(
                restrict_result_lineage_document(
                    lineage, [item.paper for item in ranked_papers]
                )
            )
        return output

    def _retrieve_subqueries(
        self,
        search_plan: SearchPlan,
        *,
        subqueries: list[SearchSubquery] | None = None,
        signals: _ExecutionSignals,
        run_context: RetrievalRunContext,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
        limit_per_source: int | None = None,
    ) -> list[RetrievalOutput]:
        return self._retrieve_query_batch(
            search_plan.subqueries if subqueries is None else subqueries,
            selected_sources=search_plan.selected_sources,
            limit_per_source=(
                search_plan.limit_per_source
                if limit_per_source is None
                else limit_per_source
            ),
            failure_prefix="subquery_failed",
            failure_source="subquery",
            signals=signals,
            constraints=search_plan.query_analysis.constraints,
            run_context=run_context,
            query_adapter_policy=query_adapter_policy,
            adaptive_budget_check=adaptive_budget_check,
        )

    def _retrieve_prf_initial_queries(
        self,
        search_plan: SearchPlan,
        *,
        signals: _ExecutionSignals,
        run_context: RetrievalRunContext,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
        judgement_policy: JudgementPolicy,
        judgement_config: JudgementRuleConfig,
        ranking_policy: RankingPolicy,
    ) -> tuple[list[RetrievalOutput], SearchPlan]:
        """Run original-query feedback before the remaining fixed-size plan."""

        original_index = next(
            (
                index
                for index, item in enumerate(search_plan.subqueries)
                if item.purpose == "original_query"
            ),
            None,
        )
        if original_index is None:
            planning = search_plan.query_planning.model_copy(
                update={
                    "prf_skip_reason": "original_query_missing",
                    "prf_fallback_used": True,
                }
            )
            fallback_plan = search_plan.model_copy(
                update={"query_planning": planning}
            )
            return (
                self._retrieve_subqueries(
                    fallback_plan,
                    signals=signals,
                    run_context=run_context,
                    query_adapter_policy=query_adapter_policy,
                    adaptive_budget_check=adaptive_budget_check,
                ),
                fallback_plan,
            )

        original = search_plan.subqueries[original_index]
        first_outputs = self._retrieve_subqueries(
            search_plan,
            subqueries=[original],
            signals=signals,
            run_context=run_context,
            query_adapter_policy=query_adapter_policy,
            adaptive_budget_check=adaptive_budget_check,
        )
        first_papers, _, _ = _collect_retrieval_outputs(first_outputs)
        source_statuses = _first_round_source_statuses(
            first_outputs,
            search_plan.selected_sources,
        )
        first_round_succeeded = any(
            status in {"success", "partial_failure"}
            for status in source_statuses.values()
        )
        first_unique = deduplicate_papers(first_papers)
        preliminary_judgements, _ = self._judge_papers(
            search_plan,
            first_unique,
            use_llm=False,
            llm_client=None,
            signals=signals,
            judgement_policy=judgement_policy,
            judgement_config=judgement_config,
        )
        preliminary_ranked, _ = _rerank_all_and_top(
            search_plan.query_analysis,
            preliminary_judgements,
            max(5, search_plan.top_k),
            ranking_policy=ranking_policy,
            retrieval_outputs=first_outputs,
        )
        outcome = build_prf_plan(
            search_plan.query_analysis.original_query,
            search_plan.subqueries,
            preliminary_ranked,
            first_round_succeeded=first_round_succeeded,
        )
        skipped = list(search_plan.query_planning.skipped_facets)
        warnings = list(search_plan.query_planning.warnings)
        plan_warnings = list(search_plan.warnings)
        if outcome.skip_reason is not None:
            marker = f"prf_v1:{outcome.skip_reason}"
            skipped.append(marker)
            warnings.append(marker)
            plan_warnings.append(marker)
        planning = search_plan.query_planning.model_copy(
            update={
                "selected_subqueries": outcome.subqueries,
                "selected_subquery_count": len(outcome.subqueries),
                "skipped_facets": _dedupe_warnings(skipped),
                "warnings": _dedupe_warnings(warnings),
                "prf_seed_candidates": outcome.seeds,
                "prf_feedback_terms": outcome.feedback_terms,
                "prf_query": outcome.query,
                "prf_replaced_index": outcome.replaced_index,
                "prf_replaced_query": outcome.replaced_query,
                "prf_replaced_purpose": outcome.replaced_purpose,
                "prf_skip_reason": outcome.skip_reason,
                "prf_fallback_used": outcome.fallback_used,
                "prf_first_round_source_statuses": source_statuses,
            }
        )
        updated_plan = search_plan.model_copy(
            update={
                "subqueries": outcome.subqueries,
                "query_planning": planning,
                "warnings": _dedupe_warnings(plan_warnings),
            }
        )
        remaining = [
            item
            for index, item in enumerate(outcome.subqueries)
            if index != original_index
        ]
        second_outputs = self._retrieve_subqueries(
            updated_plan,
            subqueries=remaining,
            signals=signals,
            run_context=run_context,
            query_adapter_policy=query_adapter_policy,
            adaptive_budget_check=adaptive_budget_check,
        )
        return [*first_outputs, *second_outputs], updated_plan

    def _retrieve_baseline_then_supplemental(
        self,
        search_plan: SearchPlan,
        *,
        supplemental_purpose: str,
        warning_prefix: str,
        signals: _ExecutionSignals,
        runtime: SearchBudgetRuntime,
        run_context: RetrievalRunContext,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
    ) -> list[RetrievalOutput]:
        """先完整执行旧规则查询，再用剩余候选预算执行单条补充查询。"""

        def is_supplemental(item: SearchSubquery) -> bool:
            return (
                item.purpose == supplemental_purpose
                if not supplemental_purpose.endswith("_")
                else item.purpose.startswith(supplemental_purpose)
            )

        baseline = [
            item
            for item in search_plan.subqueries
            if not is_supplemental(item)
        ]
        supplemental = [
            item
            for item in search_plan.subqueries
            if is_supplemental(item)
        ]
        outputs = self._retrieve_subqueries(
            search_plan,
            subqueries=baseline,
            signals=signals,
            run_context=run_context,
            query_adapter_policy=query_adapter_policy,
            adaptive_budget_check=adaptive_budget_check,
        )
        if not supplemental:
            return outputs

        baseline_papers, _, _ = _collect_retrieval_outputs(outputs)
        baseline_unique_count = len(deduplicate_papers(baseline_papers))
        remaining_candidates = max(
            0,
            runtime.budget.max_candidate_papers - baseline_unique_count,
        )
        skip_reason = runtime.latency_stop_reason()
        if skip_reason is None and remaining_candidates == 0:
            skip_reason = runtime.candidate_stop_reason(baseline_unique_count)
        if skip_reason is not None:
            outputs.extend(
                _skipped_supplemental_outputs(
                    supplemental,
                    selected_sources=search_plan.selected_sources,
                    reason=skip_reason,
                    warning_prefix=warning_prefix,
                )
            )
            return outputs

        outputs.extend(
            self._retrieve_subqueries(
                search_plan,
                subqueries=supplemental[:1],
                signals=signals,
                run_context=run_context,
                query_adapter_policy=query_adapter_policy,
                adaptive_budget_check=adaptive_budget_check,
                limit_per_source=min(
                    search_plan.limit_per_source,
                    remaining_candidates,
                ),
            )
        )
        return outputs

    def _judge_papers(
        self,
        search_plan: SearchPlan,
        papers: list[Paper],
        *,
        use_llm: bool,
        llm_client: Any | None,
        signals: _ExecutionSignals,
        judgement_policy: JudgementPolicy,
        judgement_config: JudgementRuleConfig,
        metadata_observer: UntrustedMetadataObserver | None = None,
    ) -> tuple[list[JudgementResult], int]:
        agent = JudgementAgent(
            llm_client=llm_client,
            policy=judgement_policy,
            config=judgement_config,
            metadata_observer=metadata_observer,
        )
        judgements = agent.judge(
            search_plan.query_analysis,
            papers,
            use_llm=use_llm,
            before_llm_batch=lambda: signals.check_cancelled(
                "judgement:before_llm_batch"
            ),
        )
        return judgements, agent.llm_call_count

    def _retrieve_evolved_queries(
        self,
        search_plan: SearchPlan,
        evolved_queries: list[EvolvedSubquery],
        *,
        signals: _ExecutionSignals,
        run_context: RetrievalRunContext,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
    ) -> list[RetrievalOutput]:
        subqueries = [
            SearchSubquery(
                query=query.query,
                source_hints=_limit_sources_to_selected(
                    query.source_hints,
                    search_plan.selected_sources,
                ),
                priority=query.priority,
                purpose=query.purpose,
            )
            for query in evolved_queries
        ]
        return self._retrieve_query_batch(
            subqueries,
            selected_sources=search_plan.selected_sources,
            limit_per_source=search_plan.limit_per_source,
            failure_prefix="evolved_query_failed",
            failure_source="evolved_query",
            signals=signals,
            constraints=search_plan.query_analysis.constraints,
            run_context=run_context,
            query_adapter_policy=query_adapter_policy,
            adaptive_budget_check=adaptive_budget_check,
        )

    def _retrieve_query_batch(
        self,
        subqueries: list[SearchSubquery],
        *,
        selected_sources: list[str],
        limit_per_source: int,
        failure_prefix: str,
        failure_source: str,
        signals: _ExecutionSignals,
        constraints: QueryConstraint,
        run_context: RetrievalRunContext,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
    ) -> list[RetrievalOutput]:
        if not subqueries:
            return []

        signals.check_cancelled(f"{failure_source}:before_batch")
        worker_count = min(self._max_workers, len(subqueries))
        results: list[RetrievalOutput | None] = [None] * len(subqueries)
        executor = ThreadPoolExecutor(max_workers=worker_count)
        pending: dict[Future[_RetrievalTaskResult], int] = {}
        next_index = 0
        aborted = False
        deadline_grace_used = False

        def terminal_output(index: int, status: str) -> RetrievalOutput:
            message = f"{status}:{failure_source}:{index}"
            return RetrievalOutput(
                query=subqueries[index].query,
                requested_sources=list(subqueries[index].source_hints or selected_sources),
                raw_count=0,
                deduplicated_count=0,
                papers=[],
                source_stats=[
                    SourceStats(
                        source=failure_source,
                        terminal_status=status,
                        query=subqueries[index].query,
                        returned_count=0,
                        error_message=message,
                    )
                ],
                warnings=[message],
                latency_seconds=0.0,
            )

        def submit_available() -> None:
            nonlocal next_index
            while next_index < len(subqueries) and len(pending) < worker_count:
                signals.check_cancelled(
                    f"{failure_source}:before_subquery:{next_index}"
                )
                signals.check_deadline(
                    f"{failure_source}:before_subquery:{next_index}"
                )
                future = executor.submit(
                    self._retrieve_one_subquery,
                    next_index,
                    subqueries[next_index],
                    selected_sources,
                    limit_per_source,
                    failure_prefix,
                    failure_source,
                    signals,
                    constraints,
                    run_context,
                    len(subqueries) - next_index - 1,
                    query_adapter_policy,
                    adaptive_budget_check,
                )
                pending[future] = next_index
                next_index += 1

        try:
            submit_available()
            while pending:
                remaining = signals.remaining_seconds()
                if remaining is not None and remaining <= 0:
                    completed = {future for future in pending if future.done()}
                    if not completed and not deadline_grace_used:
                        deadline_grace_used = True
                        completed, _ = wait(
                            pending,
                            timeout=RETRIEVAL_CLEANUP_GRACE_SECONDS,
                            return_when=FIRST_COMPLETED,
                        )
                    if completed:
                        for future in completed:
                            pending.pop(future, None)
                            result = future.result()
                            results[result.index] = result.output
                        continue
                    aborted = True
                    for future, index in pending.items():
                        future.cancel()
                        results[index] = terminal_output(index, "timeout")
                    pending.clear()
                    while next_index < len(subqueries):
                        results[next_index] = terminal_output(next_index, "not_started")
                        next_index += 1
                    self._terminate_isolated_processes()
                    break
                completed, _ = wait(
                    pending,
                    timeout=(0.05 if remaining is None else min(0.05, remaining)),
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    signals.check_cancelled(f"{failure_source}:between_subqueries")
                    continue
                for future in completed:
                    pending.pop(future, None)
                    result = future.result()
                    results[result.index] = result.output
                signals.check_cancelled(f"{failure_source}:between_subqueries")
                submit_available()
            if next_index < len(subqueries):
                remaining = signals.remaining_seconds()
                if remaining is not None and remaining <= 0:
                    while next_index < len(subqueries):
                        results[next_index] = terminal_output(
                            next_index,
                            "not_started",
                        )
                        next_index += 1
        except SearchCancelled:
            aborted = True
            for future in pending:
                future.cancel()
            self._terminate_isolated_processes()
            raise
        except SearchDeadlineExceeded:
            aborted = True
            for future, index in pending.items():
                future.cancel()
                results[index] = terminal_output(index, "timeout")
            pending.clear()
            while next_index < len(subqueries):
                results[next_index] = terminal_output(next_index, "not_started")
                next_index += 1
            self._terminate_isolated_processes()
        finally:
            # Never let executor __exit__ wait on an uncooperative connector.
            executor.shutdown(wait=not aborted, cancel_futures=True)

        return [output for output in results if output is not None]

    def _retrieve_one_subquery(
        self,
        index: int,
        subquery: SearchSubquery,
        selected_sources: list[str],
        limit_per_source: int,
        failure_prefix: str,
        failure_source: str,
        signals: _ExecutionSignals,
        constraints: QueryConstraint,
        run_context: RetrievalRunContext,
        remaining_subquery_count: int,
        query_adapter_policy: QueryAdapterPolicy,
        adaptive_budget_check: Callable[[list[Paper]], str | None],
    ) -> _RetrievalTaskResult:
        sources = subquery.source_hints or selected_sources
        signals.check_cancelled(f"{failure_source}:subquery:{index}:before")
        if not self._retriever_emits_connector_events:
            for source in sources:
                signals.emit(
                    "connector_started",
                    {
                        "stage": "retrieval",
                        "query_index": index,
                        "query": subquery.query,
                        "combination_mode": subquery.combination_mode,
                        "connector": source,
                        "source": source,
                    },
                )
        start = time.perf_counter()
        try:
            if self._process_isolation_available:
                output = self._run_isolated_retriever(
                    subquery.query,
                    limit_per_source,
                    sources,
                    signals,
                    index,
                )
            elif self._retriever_emits_connector_events:
                output = self._retriever(
                    subquery.query,
                    limit_per_source=limit_per_source,
                    sources=sources,
                    constraints=constraints,
                    run_context=run_context,
                    remaining_subquery_count=remaining_subquery_count,
                    query_adapter_policy=query_adapter_policy,
                    query_purpose=subquery.purpose,
                    combination_mode=subquery.combination_mode,
                    adaptive_budget_check=adaptive_budget_check,
                    connector_event_callback=lambda name, payload: (
                        self._handle_connector_event(
                            signals,
                            name,
                            payload,
                            query_index=index,
                            query=subquery.query,
                            failure_source=failure_source,
                        )
                    ),
                )
            else:
                output = self._retriever(
                    subquery.query,
                    limit_per_source=limit_per_source,
                    sources=sources,
                )
            signals.check_cancelled(f"{failure_source}:subquery:{index}:after")
        except SearchCancelled:
            raise
        except SearchDeadlineExceeded:
            latency_seconds = time.perf_counter() - start
            message = f"timeout:{failure_source}:{index}"
            output = RetrievalOutput(
                query=subquery.query,
                requested_sources=list(sources),
                raw_count=0,
                deduplicated_count=0,
                papers=[],
                source_stats=[
                    SourceStats(
                        source=failure_source,
                        terminal_status="timeout",
                        query=subquery.query,
                        returned_count=0,
                        latency_seconds=latency_seconds,
                        error_message=message,
                    )
                ],
                warnings=[message],
                latency_seconds=latency_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one subquery failure
            message = f"{failure_prefix}:{index}:{exc}"
            latency_seconds = time.perf_counter() - start
            output = RetrievalOutput(
                query=subquery.query,
                requested_sources=list(sources),
                raw_count=0,
                deduplicated_count=0,
                papers=[],
                source_stats=[
                    SourceStats(
                        source=failure_source,
                        terminal_status="source_failure",
                        query=subquery.query,
                        combination_mode=subquery.combination_mode,
                        returned_count=0,
                        latency_seconds=latency_seconds,
                        error_message=message,
                    )
                ],
                warnings=[message],
                latency_seconds=latency_seconds,
            )
        if not self._retriever_emits_connector_events:
            for stats in output.source_stats:
                signals.emit(
                    "connector_completed",
                    {
                        "stage": "retrieval",
                        "query_index": index,
                        "query": subquery.query,
                        "connector": stats.source,
                        "source": stats.source,
                        "returned_count": stats.returned_count,
                        "latency_seconds": stats.latency_seconds,
                        "request_count": stats.diagnostics.request_count,
                        "retry_count": stats.diagnostics.retry_count,
                        "error_count": stats.diagnostics.error_count,
                        "cache_hit": stats.cache_hit,
                        "cache_hit_count": stats.diagnostics.cache_hit_count,
                        "rate_limit_wait_seconds": (
                            stats.diagnostics.rate_limit_wait_seconds
                        ),
                        "retry_after_seconds": stats.diagnostics.retry_after_seconds,
                        "adapted_query": stats.adapted_query,
                        "adaptation_strategy": stats.adaptation_strategy,
                        "combination_mode": stats.combination_mode,
                        "run_dedupe_hit": stats.run_dedupe_hit,
                        "source_skipped_reason": (
                            safe_diagnostic_message(stats.source_skipped_reason)
                            if stats.source_skipped_reason is not None
                            else None
                        ),
                        "remaining_subquery_count": stats.remaining_subquery_count,
                        "error_message": (
                            safe_diagnostic_message(stats.error_message)
                            if stats.error_message is not None
                            else None
                        ),
                    },
                )
        return _RetrievalTaskResult(index=index, output=output)

    def _run_isolated_retriever(
        self,
        query: str,
        limit_per_source: int,
        sources: list[str],
        signals: _ExecutionSignals,
        index: int,
    ) -> RetrievalOutput:
        """Run a stateless retriever in a killable spawn child.

        A pipe is drained before joining the child, avoiding the feeder-pipe
        deadlock that occurs when a successful result contains many papers.
        """

        context = get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_isolated_retriever_entry,
            args=(send, self._retriever, query, limit_per_source, sources),
            daemon=True,
        )
        process.start()
        self._register_isolated_process(process)
        send.close()
        started = time.perf_counter()
        try:
            def receive_result() -> RetrievalOutput:
                message = receive.recv()
                if message[0] == "ok":
                    process.join(timeout=1.0)
                    return message[1]
                raise RuntimeError(
                    f"isolated_retriever_failed:{message[1]}:{message[2]}"
                )

            while True:
                signals.check_cancelled(f"retrieval:isolated:{index}")
                if receive.poll(0):
                    return receive_result()
                deadline_remaining = signals.remaining_seconds()
                if deadline_remaining is None:
                    remaining = None
                else:
                    minimum_remaining = (
                        ISOLATED_RETRIEVER_MIN_RUNTIME_SECONDS
                        - (time.perf_counter() - started)
                    )
                    remaining = max(deadline_remaining, minimum_remaining)
                if remaining is not None and remaining <= 0:
                    raise SearchDeadlineExceeded(f"retrieval:isolated:{index}")
                poll_for = 0.05 if remaining is None else min(0.05, remaining)
                if receive.poll(poll_for):
                    return receive_result()
                if not process.is_alive():
                    raise RuntimeError("isolated_retriever_exited_without_result")
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            receive.close()
            self._unregister_isolated_process(process)

    @staticmethod
    def _handle_connector_event(
        signals: _ExecutionSignals,
        event_name: str,
        payload: dict[str, object],
        *,
        query_index: int,
        query: str,
        failure_source: str,
    ) -> None:
        source = str(payload.get("source") or payload.get("connector") or "unknown")
        if event_name == "connector_started":
            signals.check_cancelled(
                f"{failure_source}:connector:{query_index}:{source}:before"
            )
        signals.emit(
            event_name,
            {
                "stage": "retrieval",
                "query_index": query_index,
                "query": query,
                **payload,
            },
        )
        if event_name == "connector_completed":
            signals.check_cancelled(
                f"{failure_source}:connector:{query_index}:{source}:after"
            )

    def _resolve_llm_client(self, enabled: bool) -> Any | None:
        if self._llm_client is not None:
            return self._llm_client
        if self._resolved_llm_client is not None:
            return self._resolved_llm_client
        if not enabled or not is_llm_enabled():
            return None
        try:
            self._resolved_llm_client = OpenAICompatibleLLMClient.from_env()
            return self._resolved_llm_client
        except LLMProviderError:
            return None


def run_search(
    query: str,
    top_k: int = 20,
    run_profile: RunProfile = "balanced",
    enable_refchain: bool = False,
    enable_semantic_seed_expansion: bool = False,
    enable_query_evolution: bool = False,
    query_evolution_policy: QueryEvolutionPolicy = "coverage_gap",
    query_planning_policy: QueryPlanningPolicy = "current_rules",
    ranking_policy: RankingPolicy = "current_rules",
    enable_synthesis: bool = True,
    current_year: int | None = None,
    enable_llm_query_understanding: bool | None = None,
    enable_llm_judgement: bool | None = None,
    sources_override: list[str] | None = None,
    explicit_constraints: QueryConstraint | None = None,
    budget: SearchBudget | None = None,
    event_callback: EventCallback | None = None,
    should_cancel: ShouldCancel | None = None,
    collect_diagnostics: bool = False,
    result_lineage_callback: Callable[[dict[str, Any]], None] | None = None,
    resource_accounting_observer: Any | None = None,
    untrusted_metadata_observer: UntrustedMetadataObserver | None = None,
) -> SearchServiceOutput:
    """Run the default internal search pipeline."""

    lineage_kwargs = (
        {"result_lineage_callback": result_lineage_callback}
        if result_lineage_callback is not None
        else {}
    )
    resource_kwargs = (
        {"resource_accounting_observer": resource_accounting_observer}
        if resource_accounting_observer is not None
        else {}
    )
    isolation_kwargs = (
        {"untrusted_metadata_observer": untrusted_metadata_observer}
        if untrusted_metadata_observer is not None
        else {}
    )
    return SearchService().run_search(
        query,
        top_k=top_k,
        run_profile=run_profile,
        enable_refchain=enable_refchain,
        enable_semantic_seed_expansion=enable_semantic_seed_expansion,
        enable_query_evolution=enable_query_evolution,
        query_evolution_policy=query_evolution_policy,
        query_planning_policy=query_planning_policy,
        ranking_policy=ranking_policy,
        enable_synthesis=enable_synthesis,
        current_year=current_year,
        enable_llm_query_understanding=enable_llm_query_understanding,
        enable_llm_judgement=enable_llm_judgement,
        sources_override=sources_override,
        explicit_constraints=explicit_constraints,
        budget=budget,
        event_callback=event_callback,
        should_cancel=should_cancel,
        collect_diagnostics=collect_diagnostics,
        **lineage_kwargs,
        **resource_kwargs,
        **isolation_kwargs,
    )


def _rerank_all_and_top(
    query_analysis: QueryAnalysis,
    judgements: list[JudgementResult],
    top_k: int,
    *,
    ranking_policy: RankingPolicy = "current_rules",
    retrieval_outputs: list[RetrievalOutput] | None = None,
) -> tuple[list[RankedPaper], list[RankedPaper]]:
    all_ranked_papers = rerank_papers(
        query_analysis,
        judgements,
        top_k=len(judgements),
    )
    if ranking_policy == "rrf_fusion":
        all_ranked_papers = fuse_ranked_papers(
            all_ranked_papers,
            build_retrieval_ranked_lists(retrieval_outputs or []),
            top_k=top_k,
        )
    if top_k <= 0:
        return all_ranked_papers, []
    return all_ranked_papers, all_ranked_papers[:top_k]


def _query_evolution_budget_stop_reason(
    runtime: SearchBudgetRuntime,
    candidate_count: int,
    *,
    check_rounds: bool = True,
) -> str | None:
    if (
        check_rounds
        and runtime.completed_search_rounds >= runtime.budget.max_search_rounds
    ):
        return runtime.stop("budget_stop:max_search_rounds")
    return (
        runtime.candidate_stop_reason(candidate_count)
        or runtime.latency_stop_reason()
    )


def _apply_candidate_budget(
    papers: list[Paper],
    runtime: SearchBudgetRuntime,
    *,
    stage: str,
    source_order: list[str],
) -> list[Paper]:
    limit = runtime.budget.max_candidate_papers
    if len(papers) <= limit:
        runtime.record_candidate_count(len(papers))
        return papers

    truncated = _stable_source_coverage_truncate(
        papers,
        limit=limit,
        source_order=source_order,
    )
    runtime.record_candidate_truncation(
        stage=stage,
        before_count=len(papers),
        after_count=len(truncated),
    )
    runtime.record_candidate_count(len(truncated))
    return truncated


def _stable_source_coverage_truncate(
    papers: list[Paper],
    *,
    limit: int,
    source_order: list[str],
) -> list[Paper]:
    """Round-robin stable source buckets only when truncation is necessary."""

    ordered_sources = _dedupe_warnings(
        [*source_order, *SUPPORTED_SEARCH_SOURCES, "other"]
    )
    buckets: dict[str, list[Paper]] = {source: [] for source in ordered_sources}
    source_positions = {source: index for index, source in enumerate(ordered_sources)}
    for paper in papers:
        normalized_sources = {
            str(source).strip().casefold().replace("-", "_").replace(" ", "_")
            for source in paper.sources
        }
        bucket = min(
            (source for source in ordered_sources if source in normalized_sources),
            key=lambda source: source_positions[source],
            default="other",
        )
        buckets[bucket].append(paper)

    selected: list[Paper] = []
    offsets = {source: 0 for source in ordered_sources}
    while len(selected) < limit:
        added = False
        for source in ordered_sources:
            offset = offsets[source]
            if offset >= len(buckets[source]):
                continue
            selected.append(buckets[source][offset])
            offsets[source] += 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def _env_flag(env_name: str, *, default: bool) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _add_stage_latency(
    stage_latencies: dict[str, float],
    stage: str,
    latency_seconds: float,
) -> None:
    stage_latencies[stage] = stage_latencies.get(stage, 0.0) + max(
        0.0,
        latency_seconds,
    )


def _apply_sources_override(
    search_plan: SearchPlan,
    sources_override: list[str] | None,
) -> SearchPlan:
    if sources_override is None:
        return search_plan

    selected_sources = _normalize_sources_override(sources_override)
    subqueries = [
        subquery.model_copy(update={"source_hints": list(selected_sources)})
        for subquery in search_plan.subqueries
    ]
    return search_plan.model_copy(
        update={
            "selected_sources": selected_sources,
            "subqueries": subqueries,
        }
    )


def _normalize_sources_override(raw_sources: list[str]) -> list[str]:
    selected_sources = normalize_search_sources(raw_sources)
    if not selected_sources:
        raise ValueError("source_preferences must contain at least one supported source")
    return selected_sources


def _limit_sources_to_selected(
    source_hints: list[str],
    selected_sources: list[str],
) -> list[str]:
    selected = set(selected_sources)
    return [source for source in source_hints if source in selected]


def _judgement_warnings(judgements: list[JudgementResult]) -> list[str]:
    warnings: list[str] = []
    for judgement in judgements:
        warnings.extend(judgement.warnings)
    return warnings


def _collect_retrieval_outputs(
    outputs: list[RetrievalOutput],
) -> tuple[list[Paper], list[SourceStats], list[str]]:
    papers: list[Paper] = []
    source_stats: list[SourceStats] = []
    warnings: list[str] = []
    for output in outputs:
        papers.extend(output.papers)
        source_stats.extend(output.source_stats)
        warnings.extend(output.warnings)
    return papers, source_stats, warnings


def _result_lineage_source_terminals(
    source_stats: list[SourceStats],
    selected_sources: list[str],
    source_records: list[Paper],
) -> list[dict[str, object]]:
    """Collapse per-subquery terminals without retaining connector error text."""

    rows: list[dict[str, object]] = []
    for source in sorted(set(selected_sources)):
        observed = [item for item in source_stats if item.source == source]
        contributed = sum(source in paper.sources for paper in source_records)
        executed = [item for item in observed if item.logical_call_executed]
        failed = [
            item
            for item in executed
            if item.error_message is not None
            or item.terminal_status
            in {"failed", "missing", "source_failure", "source_outage"}
        ]
        if not executed:
            status = "not_started"
            reason = "no_logical_source_call"
        elif failed and contributed:
            status = "partial_completion"
            reason = "one_or_more_source_calls_failed"
        elif failed:
            status = "failed"
            reason = "all_executed_source_calls_failed"
        elif contributed:
            status = "success"
            reason = None
        else:
            status = "success_empty"
            reason = None
        rows.append(
            {
                "source": source,
                "status": status,
                "reason": reason,
                "contributed_record_count": contributed,
            }
        )
    return rows


def _first_round_source_statuses(
    outputs: list[RetrievalOutput],
    selected_sources: list[str],
) -> dict[str, str]:
    """Summarize first-round terminals without treating skipped adapter calls as failures."""

    statuses: dict[str, str] = {}
    for source in selected_sources:
        stats = [
            item
            for output in outputs
            for item in output.source_stats
            if item.source == source
        ]
        succeeded = any(
            item.logical_call_executed and item.error_message is None
            for item in stats
        )
        failed = any(
            item.logical_call_executed and item.error_message is not None
            for item in stats
        )
        if succeeded and failed:
            statuses[source] = "partial_failure"
        elif succeeded:
            statuses[source] = "success"
        elif failed:
            statuses[source] = "failed"
        elif stats:
            statuses[source] = "not_started"
        else:
            statuses[source] = "missing"
    return statuses


def _skipped_supplemental_outputs(
    subqueries: list[SearchSubquery],
    *,
    selected_sources: list[str],
    reason: str,
    warning_prefix: str,
) -> list[RetrievalOutput]:
    """保留未执行补充查询的稳定诊断，不产生 connector 调用。"""

    outputs: list[RetrievalOutput] = []
    for subquery in subqueries[:1]:
        sources = subquery.source_hints or selected_sources
        outputs.append(
            RetrievalOutput(
                query=subquery.query,
                requested_sources=list(sources),
                raw_count=0,
                deduplicated_count=0,
                source_stats=[
                    SourceStats(
                        source=source,
                        query=subquery.query,
                        combination_mode=subquery.combination_mode,
                        logical_call_executed=False,
                        source_skipped_reason=reason,
                    )
                    for source in sources
                ],
                warnings=[
                    f"{warning_prefix}_skipped:{reason}"
                ],
            )
        )
    return outputs


def _filter_new_evolved_queries(
    evolved_queries: list[EvolvedSubquery],
    used_queries: set[str],
) -> list[EvolvedSubquery]:
    seen = {_query_key(query) for query in used_queries}
    filtered: list[EvolvedSubquery] = []
    for query in evolved_queries:
        key = _query_key(query.query)
        if not key or key in seen:
            continue
        filtered.append(query)
        seen.add(key)
    return filtered


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        item = warning.strip()
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped
