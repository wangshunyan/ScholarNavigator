"""Multi-source retrieval aggregation."""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import RLock

from pydantic import BaseModel, Field

from scholar_agent.connectors import (
    ConnectorSearchResult,
    search_arxiv_detailed,
    search_local_bm25_detailed,
    search_local_hybrid_detailed,
    search_openalex_detailed,
    search_pubmed_detailed,
    search_semantic_scholar_detailed,
)
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    CombinationMode,
    DEFAULT_SEARCH_SOURCES,
    QueryConstraint,
    SUPPORTED_SEARCH_SOURCES,
)
from scholar_agent.retrieval.query_adapter import (
    DEFAULT_QUERY_ADAPTER_POLICY,
    MIN_COMPACT_RETENTION_RATIO,
    AdaptedQuery,
    QueryAdapterPolicy,
    adapt_queries_for_source,
)


SUPPORTED_SOURCES = SUPPORTED_SEARCH_SOURCES
DEFAULT_CACHE_TTL_SECONDS = 15 * 60
DEFAULT_CACHE_MAX_ENTRIES = 256
DEFAULT_SOURCE_COOLDOWN_SECONDS = 60
DEFAULT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS = 60
RUN_TRANSIENT_FAILURE_THRESHOLD = 2
CACHE_DISABLED_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}
ADAPTIVE_MIN_UNIQUE_CANDIDATES = 8
ADAPTIVE_MIN_CANDIDATE_RATIO = 0.5
ADAPTIVE_MIN_CORE_TERM_COVERAGE = 0.6
ADAPTIVE_MIN_CONSTRAINT_COVERAGE = 0.5
ADAPTIVE_MIN_METADATA_COVERAGE = 0.5
ADAPTIVE_COMPLEX_CORE_TERM_COUNT = 5
ADAPTIVE_COMPLEX_DIMENSION_COUNT = 2


_CacheKey = tuple[str, str, int]
_RecordedTerminalLookup = Callable[[str, str, int], bool]
_RETRIEVAL_CACHE: OrderedDict[_CacheKey, tuple[float, ConnectorSearchResult]] = (
    OrderedDict()
)
_RETRIEVAL_CACHE_LOCK = RLock()
_SOURCE_FAILURE_COOLDOWNS: dict[str, float] = {}
_SOURCE_FAILURE_COOLDOWNS_LOCK = RLock()


class QueryAdaptationProvenance(BaseModel):
    origin_subquery: str
    adaptation_strategy: str
    purpose: str | None = None
    combination_mode: CombinationMode = "all"


class RetrievalSufficiency(BaseModel):
    sufficient: bool
    unique_candidate_count: int = 0
    core_term_coverage: float = 0.0
    constraint_coverage: float = 0.0
    metadata_coverage: float = 0.0
    missing_dimensions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SourceStats(BaseModel):
    source: str
    terminal_status: str | None = None
    query: str | None = None
    returned_count: int = 0
    latency_seconds: float = 0.0
    error_message: str | None = None
    cache_hit: bool = False
    run_dedupe_hit: bool = False
    adapted_query: str | None = None
    adaptation_strategy: str | None = None
    combination_mode: CombinationMode = "all"
    query_provenance: list[QueryAdaptationProvenance] = Field(default_factory=list)
    dropped_terms: list[str] = Field(default_factory=list)
    original_information_terms: list[str] = Field(default_factory=list)
    retained_information_terms: list[str] = Field(default_factory=list)
    retention_ratio: float | None = None
    protected_terms: list[str] = Field(default_factory=list)
    logical_call_executed: bool = True
    triggered_by: list[str] = Field(default_factory=list)
    safe_original_candidate_count: int | None = None
    safe_original_core_term_coverage: float | None = None
    safe_original_constraint_coverage: float | None = None
    sufficiency_reasons: list[str] = Field(default_factory=list)
    compact_query_executed: bool | None = None
    compact_query_skipped_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    source_skipped_reason: str | None = None
    remaining_subquery_count: int = 0
    diagnostic_papers: list[Paper] = Field(default_factory=list, exclude=True)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    snapshot_provenance: str = "live"
    snapshot_key: str | None = None
    snapshot_hit: bool = False
    recorded_diagnostics: ConnectorDiagnostics | None = None
    recorded_latency_seconds: float = 0.0


class RetrievalOutput(BaseModel):
    query: str
    requested_sources: list[str]
    raw_count: int
    deduplicated_count: int
    papers: list[Paper] = Field(default_factory=list)
    source_stats: list[SourceStats] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0


@dataclass
class RetrievalRunContext:
    """一次 SearchRun 内共享的调用去重、结果复用与来源降级状态。"""

    _lock: RLock = field(default_factory=RLock)
    _source_locks: dict[str, RLock] = field(default_factory=dict)
    _results: dict[_CacheKey, ConnectorSearchResult] = field(default_factory=dict)
    _query_provenance: dict[_CacheKey, list[QueryAdaptationProvenance]] = field(
        default_factory=dict
    )
    _blocked_sources: dict[str, str] = field(default_factory=dict)
    _transient_failure_counts: dict[str, int] = field(default_factory=dict)

    def source_lock(self, source: str) -> RLock:
        with self._lock:
            return self._source_locks.setdefault(source, RLock())

    def reused_result(self, key: _CacheKey) -> ConnectorSearchResult | None:
        with self._lock:
            result = self._results.get(key)
            return result.model_copy(deep=True) if result is not None else None

    def store_result(self, key: _CacheKey, result: ConnectorSearchResult) -> None:
        with self._lock:
            self._results[key] = result.model_copy(deep=True)

    def register_query_provenance(
        self,
        key: _CacheKey,
        values: list[QueryAdaptationProvenance],
    ) -> list[QueryAdaptationProvenance]:
        with self._lock:
            stored = self._query_provenance.setdefault(key, [])
            for value in values:
                if value not in stored:
                    stored.append(value.model_copy(deep=True))
            return [item.model_copy(deep=True) for item in stored]

    def block_source(self, source: str, reason: str) -> None:
        with self._lock:
            self._blocked_sources[source] = reason

    def blocked_reason(self, source: str) -> str | None:
        with self._lock:
            return self._blocked_sources.get(source)

    def record_transient_outcome(
        self,
        source: str,
        error_message: str | None,
    ) -> None:
        with self._lock:
            if not error_message:
                self._transient_failure_counts.pop(source, None)
                return
            if not _is_transient_error_message(error_message):
                return
            count = self._transient_failure_counts.get(source, 0) + 1
            self._transient_failure_counts[source] = count
            if count >= RUN_TRANSIENT_FAILURE_THRESHOLD:
                self._blocked_sources[source] = "run_transient_failure_circuit_open"


def retrieve_papers(
    query: str,
    limit_per_source: int = 20,
    sources: list[str] | None = None,
    connector_event_callback: Callable[[str, dict[str, object]], None] | None = None,
    constraints: QueryConstraint | None = None,
    run_context: RetrievalRunContext | None = None,
    remaining_subquery_count: int = 0,
    query_adapter_policy: QueryAdapterPolicy = DEFAULT_QUERY_ADAPTER_POLICY,
    query_purpose: str | None = None,
    combination_mode: CombinationMode = "all",
    adaptive_budget_check: Callable[[list[Paper]], str | None] | None = None,
    connector_result_provider: Callable[
        [str, str, int, QueryAdapterPolicy, Callable[[str, int], ConnectorSearchResult]],
        ConnectorSearchResult,
    ]
    | None = None,
    replay_recorded_terminals: bool = False,
    recorded_terminal_lookup: _RecordedTerminalLookup | None = None,
) -> RetrievalOutput:
    """Retrieve papers from supported sources and deduplicate them."""

    requested_sources = _normalize_sources(sources)
    start = time.perf_counter()
    warnings: list[str] = []
    source_stats: list[SourceStats] = []
    raw_papers: list[Paper] = []

    if not query.strip():
        warnings.append("empty_query")

    for source in requested_sources:
        search = _source_registry().get(source)
        if search is None:
            message = f"unsupported_source:{source}"
            warnings.append(message)
            source_stats.append(
                SourceStats(
                    source=source,
                    query=query,
                    returned_count=0,
                    latency_seconds=0.0,
                    error_message=message,
                    combination_mode=combination_mode,
                    source_skipped_reason="unsupported_source",
                    remaining_subquery_count=remaining_subquery_count,
                )
            )
            _emit_connector_completed(connector_event_callback, source_stats[-1])
            continue

        effective_search = search
        if connector_result_provider is not None:
            effective_search = lambda adapted_query, limit, *, _source=source, _search=search: (
                connector_result_provider(
                    _source,
                    adapted_query,
                    limit,
                    query_adapter_policy,
                    _search,
                )
            )

        adapted_queries = adapt_queries_for_source(
            query,
            source,
            constraints=constraints,
            policy=query_adapter_policy,
            combination_mode=combination_mode,
        )
        if query_adapter_policy == "adaptive":
            adaptive_stats = _retrieve_adaptive_source(
                original_query=query,
                source=source,
                adapted_queries=adapted_queries,
                constraints=constraints,
                limit_per_source=limit_per_source,
                search=effective_search,
                run_context=run_context,
                remaining_subquery_count=remaining_subquery_count,
                query_purpose=query_purpose,
                combination_mode=combination_mode,
                connector_event_callback=connector_event_callback,
                adaptive_budget_check=adaptive_budget_check,
                replay_recorded_terminals=replay_recorded_terminals,
                recorded_terminal_lookup=recorded_terminal_lookup,
            )
            for stats in adaptive_stats:
                source_stats.append(stats)
                if stats.logical_call_executed and not stats.run_dedupe_hit:
                    raw_papers.extend(stats.diagnostic_papers)
                warnings.extend(stats.warnings)
                if stats.cache_hit:
                    warnings.append(f"retrieval_cache_hit:{source}")
                if stats.error_message and stats.error_message not in warnings:
                    warnings.append(stats.error_message)
            continue
        for adapted in adapted_queries:
            if connector_event_callback is not None:
                connector_event_callback(
                    "connector_started",
                    {
                        "connector": source,
                        "source": source,
                        "adapted_query": adapted.query,
                        "adaptation_strategy": adapted.strategy,
                    },
                )
            stats = _retrieve_adapted_query(
                original_query=query,
                source=source,
                adapted=adapted,
                limit_per_source=limit_per_source,
                search=effective_search,
                run_context=run_context,
                remaining_subquery_count=remaining_subquery_count,
                query_purpose=query_purpose,
                combination_mode=combination_mode,
                replay_recorded_terminals=replay_recorded_terminals,
                recorded_terminal_lookup=recorded_terminal_lookup,
            )
            source_stats.append(stats)
            if stats.source_skipped_reason and not stats.error_message:
                message = (
                    f"source_skipped:{source}:{stats.source_skipped_reason}:"
                    f"remaining={remaining_subquery_count}"
                )
                warnings.append(message)
            if not stats.run_dedupe_hit:
                raw_papers.extend(stats.diagnostic_papers)
            warnings.extend(adapted.warnings)
            warnings.extend(stats.warnings)
            if stats.cache_hit:
                warnings.append(f"retrieval_cache_hit:{source}")
            if stats.error_message and stats.error_message not in warnings:
                warnings.append(stats.error_message)
            _emit_connector_completed(connector_event_callback, stats)

    deduplicated = deduplicate_papers(raw_papers)
    return RetrievalOutput(
        query=query,
        requested_sources=requested_sources,
        raw_count=len(raw_papers),
        deduplicated_count=len(deduplicated),
        papers=deduplicated,
        source_stats=source_stats,
        warnings=warnings,
        latency_seconds=time.perf_counter() - start,
    )


def evaluate_retrieval_sufficiency(
    papers: list[Paper],
    *,
    compact_query: AdaptedQuery,
    constraints: QueryConstraint | None,
    limit_per_source: int,
    source_succeeded: bool,
    safe_original_truncated: bool = False,
) -> RetrievalSufficiency:
    """仅根据查询、约束和 safe-original 候选判断是否需要补充检索。"""

    unique = deduplicate_papers(papers)
    candidate_count = len(unique)
    corpus = [_paper_search_text(paper) for paper in unique]
    core_terms = _stable_strings(compact_query.original_information_terms)
    matched_core = [
        term for term in core_terms if _term_in_any_candidate(term, corpus)
    ]
    core_coverage = len(matched_core) / len(core_terms) if core_terms else 1.0
    dimensions = _constraint_dimensions(constraints)
    missing_dimensions = [
        name
        for name, terms in dimensions.items()
        if terms and not any(_term_in_any_candidate(term, corpus) for term in terms)
    ]
    has_time_dimension = bool(constraints and constraints.time_range)
    if has_time_dimension and not any(
        _paper_in_time_range(paper, constraints) for paper in unique
    ):
        missing_dimensions.append("time_range")
    dimension_count = len(dimensions) + int(has_time_dimension)
    constraint_coverage = (
        (dimension_count - len(missing_dimensions)) / dimension_count
        if dimension_count
        else 1.0
    )
    metadata_coverage = (
        sum(bool(paper.title.strip() and paper.abstract.strip()) for paper in unique)
        / candidate_count
        if candidate_count
        else 0.0
    )
    required_candidates = min(
        ADAPTIVE_MIN_UNIQUE_CANDIDATES,
        max(1, round(limit_per_source * ADAPTIVE_MIN_CANDIDATE_RATIO)),
    )
    complex_query = (
        len(core_terms) >= ADAPTIVE_COMPLEX_CORE_TERM_COUNT
        or dimension_count >= ADAPTIVE_COMPLEX_DIMENSION_COUNT
    )
    reasons: list[str] = []
    if not source_succeeded:
        reasons.append("adaptive_source_failed")
    if candidate_count == 0:
        reasons.append("adaptive_empty_results")
    elif candidate_count < required_candidates:
        reasons.append("adaptive_low_candidate_count")
    for dimension in ("must_have_terms", "methods", "datasets"):
        if dimension in missing_dimensions:
            reasons.append(f"adaptive_missing_{dimension}")
    if complex_query and core_coverage < ADAPTIVE_MIN_CORE_TERM_COVERAGE:
        reasons.append("adaptive_low_core_term_coverage")
    if (
        complex_query
        and constraint_coverage < ADAPTIVE_MIN_CONSTRAINT_COVERAGE
    ):
        reasons.append("adaptive_low_constraint_coverage")
    if candidate_count and metadata_coverage < ADAPTIVE_MIN_METADATA_COVERAGE:
        reasons.append("adaptive_low_metadata_coverage")
    if safe_original_truncated:
        reasons.append("adaptive_safe_original_truncated")
    return RetrievalSufficiency(
        sufficient=not reasons,
        unique_candidate_count=candidate_count,
        core_term_coverage=core_coverage,
        constraint_coverage=constraint_coverage,
        metadata_coverage=metadata_coverage,
        missing_dimensions=missing_dimensions,
        reasons=reasons or ["adaptive_sufficient_results"],
    )


def _retrieve_adaptive_source(
    *,
    original_query: str,
    source: str,
    adapted_queries: list[AdaptedQuery],
    constraints: QueryConstraint | None,
    limit_per_source: int,
    search: Callable[[str, int], ConnectorSearchResult],
    run_context: RetrievalRunContext | None,
    remaining_subquery_count: int,
    query_purpose: str | None,
    combination_mode: CombinationMode,
    connector_event_callback: Callable[[str, dict[str, object]], None] | None,
    adaptive_budget_check: Callable[[list[Paper]], str | None] | None,
    replay_recorded_terminals: bool,
    recorded_terminal_lookup: _RecordedTerminalLookup | None,
) -> list[SourceStats]:
    if not adapted_queries:
        return []
    safe = adapted_queries[0]
    _emit_connector_started(connector_event_callback, source, safe)
    safe_stats = _retrieve_adapted_query(
        original_query=original_query,
        source=source,
        adapted=safe,
        limit_per_source=limit_per_source,
        search=search,
        run_context=run_context,
        remaining_subquery_count=remaining_subquery_count,
        query_purpose=query_purpose,
        combination_mode=combination_mode,
        replay_recorded_terminals=replay_recorded_terminals,
        recorded_terminal_lookup=recorded_terminal_lookup,
    )
    _emit_connector_completed(connector_event_callback, safe_stats)
    if len(adapted_queries) < 2:
        return [safe_stats]

    compact = adapted_queries[1]
    safe_papers = list(safe_stats.diagnostic_papers)
    sufficiency = evaluate_retrieval_sufficiency(
        safe_papers,
        compact_query=compact,
        constraints=constraints,
        limit_per_source=limit_per_source,
        source_succeeded=safe_stats.error_message is None,
        safe_original_truncated="safe_original_truncated" in safe.warnings,
    )
    budget_reason = (
        adaptive_budget_check(safe_papers)
        if adaptive_budget_check is not None
        else None
    )
    guard = run_context.source_lock(source) if run_context is not None else nullcontext()
    with guard:
        skip_reason = _adaptive_compact_skip_reason(
            source=source,
            safe=safe,
            compact=compact,
            safe_stats=safe_stats,
            sufficiency=sufficiency,
            budget_reason=budget_reason,
            limit_per_source=limit_per_source,
            run_context=run_context,
            replay_recorded_terminals=replay_recorded_terminals,
        )
        triggered_by = list(sufficiency.reasons)
        if budget_reason:
            triggered_by.append(budget_reason)
        if skip_reason is not None:
            skipped = _adaptive_skipped_stats(
                original_query=original_query,
                source=source,
                compact=compact,
                query_purpose=query_purpose,
                combination_mode=combination_mode,
                run_context=run_context,
                remaining_subquery_count=remaining_subquery_count,
                sufficiency=sufficiency,
                triggered_by=triggered_by,
                skip_reason=skip_reason,
                limit_per_source=limit_per_source,
            )
            _emit_adaptive_decision(connector_event_callback, skipped)
            return [safe_stats, skipped]

        _emit_adaptive_decision_payload(
            connector_event_callback,
            source=source,
            compact=compact,
            sufficiency=sufficiency,
            triggered_by=triggered_by,
            executed=True,
            skip_reason=None,
        )
        _emit_connector_started(connector_event_callback, source, compact)
        compact_stats = _retrieve_adapted_query(
            original_query=original_query,
            source=source,
            adapted=compact,
            limit_per_source=limit_per_source,
            search=search,
            run_context=run_context,
            remaining_subquery_count=remaining_subquery_count,
            query_purpose=query_purpose,
            combination_mode=combination_mode,
            replay_recorded_terminals=replay_recorded_terminals,
            recorded_terminal_lookup=recorded_terminal_lookup,
        ).model_copy(
            update=_adaptive_stats_update(
                sufficiency,
                triggered_by=triggered_by,
                executed=True,
                skip_reason=None,
            )
        )
        _emit_connector_completed(connector_event_callback, compact_stats)
        if adaptive_budget_check is not None:
            adaptive_budget_check(list(compact_stats.diagnostic_papers))
        return [safe_stats, compact_stats]


def _adaptive_compact_skip_reason(
    *,
    source: str,
    safe: AdaptedQuery,
    compact: AdaptedQuery,
    safe_stats: SourceStats,
    sufficiency: RetrievalSufficiency,
    budget_reason: str | None,
    limit_per_source: int,
    run_context: RetrievalRunContext | None,
    replay_recorded_terminals: bool,
) -> str | None:
    if not compact.query or compact.retention_ratio < MIN_COMPACT_RETENTION_RATIO:
        return "adaptive_low_information_retention"
    if "compact_query_protected_terms_removed" in compact.warnings:
        return "adaptive_low_information_retention"
    if _cache_key(source, safe.query, limit_per_source) == _cache_key(
        source, compact.query, limit_per_source
    ):
        return "adaptive_equivalent_query"
    if run_context is not None:
        key = _cache_key(source, compact.query, limit_per_source)
        if run_context.reused_result(key) is not None:
            return "adaptive_equivalent_query"
        if (
            not replay_recorded_terminals
            and run_context.blocked_reason(source) is not None
        ):
            return "adaptive_source_cooldown"
    if not replay_recorded_terminals and _is_source_in_cooldown(source):
        return "adaptive_source_cooldown"
    if safe_stats.error_message is not None:
        return "adaptive_source_failed"
    if budget_reason is not None:
        return "adaptive_budget_exhausted"
    if sufficiency.sufficient:
        return "adaptive_sufficient_results"
    return None


def _adaptive_skipped_stats(
    *,
    original_query: str,
    source: str,
    compact: AdaptedQuery,
    query_purpose: str | None,
    combination_mode: CombinationMode,
    run_context: RetrievalRunContext | None,
    remaining_subquery_count: int,
    sufficiency: RetrievalSufficiency,
    triggered_by: list[str],
    skip_reason: str,
    limit_per_source: int,
) -> SourceStats:
    provenance = [
        QueryAdaptationProvenance(
            origin_subquery=original_query,
            adaptation_strategy=compact.strategy,
            purpose=query_purpose,
            combination_mode=combination_mode,
        )
    ]
    if run_context is not None and compact.query:
        provenance = run_context.register_query_provenance(
            _cache_key(source, compact.query, limit_per_source),
            provenance,
        )
    return SourceStats(
        source=source,
        query=original_query,
        adapted_query=compact.query,
        adaptation_strategy=compact.strategy,
        combination_mode=combination_mode,
        query_provenance=provenance,
        dropped_terms=list(compact.dropped_terms),
        original_information_terms=list(compact.original_information_terms),
        retained_information_terms=list(compact.retained_information_terms),
        retention_ratio=compact.retention_ratio,
        protected_terms=list(compact.protected_terms),
        logical_call_executed=False,
        source_skipped_reason=skip_reason,
        remaining_subquery_count=max(0, remaining_subquery_count),
        **_adaptive_stats_update(
            sufficiency,
            triggered_by=triggered_by,
            executed=False,
            skip_reason=skip_reason,
        ),
    )


def _adaptive_stats_update(
    sufficiency: RetrievalSufficiency,
    *,
    triggered_by: list[str],
    executed: bool,
    skip_reason: str | None,
) -> dict[str, object]:
    return {
        "triggered_by": _stable_strings(triggered_by),
        "safe_original_candidate_count": sufficiency.unique_candidate_count,
        "safe_original_core_term_coverage": sufficiency.core_term_coverage,
        "safe_original_constraint_coverage": sufficiency.constraint_coverage,
        "sufficiency_reasons": list(sufficiency.reasons),
        "compact_query_executed": executed,
        "compact_query_skipped_reason": skip_reason,
    }


def _emit_connector_started(
    callback: Callable[[str, dict[str, object]], None] | None,
    source: str,
    adapted: AdaptedQuery,
) -> None:
    if callback is None:
        return
    callback(
        "connector_started",
        {
            "connector": source,
            "source": source,
            "adapted_query": adapted.query,
            "adaptation_strategy": adapted.strategy,
        },
    )


def _emit_adaptive_decision(
    callback: Callable[[str, dict[str, object]], None] | None,
    stats: SourceStats,
) -> None:
    _emit_adaptive_decision_payload(
        callback,
        source=stats.source,
        compact=AdaptedQuery(
            original_query=stats.query or "",
            source=stats.source,
            query=stats.adapted_query or "",
            strategy=stats.adaptation_strategy or "compact_core",
        ),
        sufficiency=RetrievalSufficiency(
            sufficient=stats.compact_query_skipped_reason
            == "adaptive_sufficient_results",
            unique_candidate_count=stats.safe_original_candidate_count or 0,
            core_term_coverage=stats.safe_original_core_term_coverage or 0.0,
            constraint_coverage=stats.safe_original_constraint_coverage or 0.0,
            reasons=list(stats.sufficiency_reasons),
        ),
        triggered_by=list(stats.triggered_by),
        executed=False,
        skip_reason=stats.compact_query_skipped_reason,
    )


def _emit_adaptive_decision_payload(
    callback: Callable[[str, dict[str, object]], None] | None,
    *,
    source: str,
    compact: AdaptedQuery,
    sufficiency: RetrievalSufficiency,
    triggered_by: list[str],
    executed: bool,
    skip_reason: str | None,
) -> None:
    if callback is None:
        return
    callback(
        "adaptive_query_decision",
        {
            "connector": source,
            "source": source,
            "adapted_query": compact.query,
            "adaptation_strategy": compact.strategy,
            "triggered_by": list(triggered_by),
            "safe_original_candidate_count": sufficiency.unique_candidate_count,
            "safe_original_core_term_coverage": sufficiency.core_term_coverage,
            "safe_original_constraint_coverage": sufficiency.constraint_coverage,
            "sufficiency_reasons": list(sufficiency.reasons),
            "compact_query_executed": executed,
            "compact_query_skipped_reason": skip_reason,
        },
    )


def _constraint_dimensions(
    constraints: QueryConstraint | None,
) -> dict[str, list[str]]:
    if constraints is None:
        return {}
    candidates = {
        "must_have_terms": constraints.must_include_terms,
        "methods": constraints.methods,
        "datasets": constraints.datasets,
        "paper_types": list(constraints.paper_types),
        "venues": constraints.venues,
        "domains": [value.replace("_", " ") for value in constraints.domains],
    }
    return {
        name: _stable_strings(values)
        for name, values in candidates.items()
        if values
    }


def _paper_in_time_range(paper: Paper, constraints: QueryConstraint) -> bool:
    time_range = constraints.time_range
    if time_range is None or paper.year is None:
        return False
    if time_range.start_year is not None and paper.year < time_range.start_year:
        return False
    if time_range.end_year is not None and paper.year > time_range.end_year:
        return False
    return True


def _paper_search_text(paper: Paper) -> str:
    return _normalize_match_text(
        " ".join(
            [
                paper.title,
                paper.abstract,
                paper.venue or "",
                " ".join(paper.authors),
            ]
        )
    )


def _term_in_any_candidate(term: str, corpus: list[str]) -> bool:
    normalized = _normalize_match_text(term)
    return bool(normalized and any(normalized in text for text in corpus))


def _normalize_match_text(value: str) -> str:
    return " ".join(
        re.sub(r"[^\w\u4e00-\u9fff]+", " ", value.casefold()).split()
    )


def _retrieve_adapted_query(
    *,
    original_query: str,
    source: str,
    adapted: AdaptedQuery,
    limit_per_source: int,
    search: Callable[[str, int], ConnectorSearchResult],
    run_context: RetrievalRunContext | None,
    remaining_subquery_count: int,
    query_purpose: str | None,
    combination_mode: CombinationMode,
    replay_recorded_terminals: bool,
    recorded_terminal_lookup: _RecordedTerminalLookup | None,
) -> SourceStats:
    source_start = time.perf_counter()
    base = {
        "source": source,
        "query": original_query,
        "adapted_query": adapted.query,
        "adaptation_strategy": adapted.strategy,
        "combination_mode": combination_mode,
        "dropped_terms": list(adapted.dropped_terms),
        "original_information_terms": list(adapted.original_information_terms),
        "retained_information_terms": list(adapted.retained_information_terms),
        "retention_ratio": adapted.retention_ratio,
        "protected_terms": list(adapted.protected_terms),
        "remaining_subquery_count": max(0, remaining_subquery_count),
    }
    if not adapted.query:
        return SourceStats(
            **base,
            source_skipped_reason="empty_adapted_query",
        )

    guard = run_context.source_lock(source) if run_context is not None else nullcontext()
    with guard:
        key = _cache_key(source, adapted.query, limit_per_source)
        strategies = _stable_strings(
            [adapted.strategy, *adapted.equivalent_strategies]
        )
        current_provenance = [
            QueryAdaptationProvenance(
                origin_subquery=original_query,
                adaptation_strategy=strategy,
                purpose=query_purpose,
                combination_mode=combination_mode,
            )
            for strategy in strategies
        ]
        query_provenance = (
            run_context.register_query_provenance(key, current_provenance)
            if run_context is not None
            else current_provenance
        )
        base["query_provenance"] = query_provenance
        if (
            replay_recorded_terminals
            and recorded_terminal_lookup is not None
            and not recorded_terminal_lookup(
                source,
                adapted.query,
                limit_per_source,
            )
        ):
            return SourceStats(
                **base,
                terminal_status="not_started",
                logical_call_executed=False,
                source_skipped_reason="snapshot_key_not_recorded",
                snapshot_provenance="snapshot_replay",
                latency_seconds=time.perf_counter() - source_start,
            )
        reused = run_context.reused_result(key) if run_context is not None else None
        if reused is not None:
            return SourceStats(
                **base,
                run_dedupe_hit=True,
                source_skipped_reason="duplicate_adapted_query",
                diagnostic_papers=list(reused.papers),
                latency_seconds=time.perf_counter() - source_start,
            )

        blocked_reason = (
            run_context.blocked_reason(source) if run_context is not None else None
        )
        if not replay_recorded_terminals and blocked_reason is not None:
            return SourceStats(
                **base,
                error_message=f"source_cooldown_skip:{source}",
                source_skipped_reason=blocked_reason,
                latency_seconds=time.perf_counter() - source_start,
            )
        if not replay_recorded_terminals and _is_source_in_cooldown(source):
            return SourceStats(
                **base,
                error_message=f"source_cooldown_skip:{source}",
                source_skipped_reason="source_cooldown",
                latency_seconds=time.perf_counter() - source_start,
            )

        try:
            result, cache_hit = _search_with_cache(
                source,
                adapted.query,
                limit_per_source,
                search,
            )
        except Exception as exc:  # noqa: BLE001 - isolate connector failures
            safe_error = sanitize_connector_error_text(str(exc))
            message = f"{source} failed: {safe_error}"
            if not replay_recorded_terminals:
                if _is_final_rate_limit(message):
                    _record_source_cooldown(source)
                    if run_context is not None:
                        run_context.block_source(source, "run_rate_limit_cooldown")
                elif run_context is not None:
                    run_context.record_transient_outcome(source, message)
            return SourceStats(
                **base,
                terminal_status=(
                    "missing"
                    if replay_recorded_terminals
                    and "snapshot_missing:" in str(exc)
                    else "source_failure"
                ),
                error_message=safe_error,
                warnings=[message],
                latency_seconds=time.perf_counter() - source_start,
                diagnostics=ConnectorDiagnostics(
                    error_count=1,
                    latency_seconds=time.perf_counter() - source_start,
                ),
                snapshot_provenance=(
                    "snapshot_replay" if replay_recorded_terminals else "live"
                ),
            )

        if run_context is not None:
            run_context.store_result(key, result)
        if (
            not replay_recorded_terminals
            and _has_cooldown_trigger(result.error_message, result.warnings)
        ):
            _record_source_cooldown(
                source,
                minimum_seconds=result.diagnostics.retry_after_seconds,
            )
            if run_context is not None and _is_final_rate_limit(
                result.error_message
            ):
                run_context.block_source(source, "run_rate_limit_cooldown")
        elif not replay_recorded_terminals and run_context is not None:
            run_context.record_transient_outcome(source, result.error_message)
        safe_result_error = (
            sanitize_connector_error_text(result.error_message)
            if result.error_message is not None
            else None
        )
        safe_result_warnings = [
            sanitize_connector_error_text(item) for item in result.warnings
        ]
        return SourceStats(
            **base,
            terminal_status=(
                _snapshot_terminal_status(result)
                if replay_recorded_terminals
                else None
            ),
            returned_count=len(result.papers),
            latency_seconds=time.perf_counter() - source_start,
            error_message=safe_result_error,
            cache_hit=cache_hit,
            warnings=safe_result_warnings,
            diagnostic_papers=list(result.papers),
            diagnostics=result.diagnostics,
            snapshot_provenance=result.snapshot_provenance,
            snapshot_key=result.snapshot_key,
            snapshot_hit=result.snapshot_hit,
            recorded_diagnostics=result.recorded_diagnostics,
            recorded_latency_seconds=result.recorded_latency_seconds,
        )


def _snapshot_terminal_status(result: ConnectorSearchResult) -> str:
    if result.error_message is None:
        return "success"
    if "source_outage" in result.error_message.casefold():
        return "source_outage"
    return "failed"


def _emit_connector_completed(
    callback: Callable[[str, dict[str, object]], None] | None,
    stats: SourceStats,
) -> None:
    if callback is None:
        return
    callback(
        "connector_completed",
        {
            "connector": stats.source,
            "source": stats.source,
            "returned_count": stats.returned_count,
            "latency_seconds": stats.latency_seconds,
            "request_count": stats.diagnostics.request_count,
            "retry_count": stats.diagnostics.retry_count,
            "error_count": stats.diagnostics.error_count,
            "cache_hit": stats.cache_hit,
            "cache_hit_count": stats.diagnostics.cache_hit_count,
            "rate_limit_wait_seconds": stats.diagnostics.rate_limit_wait_seconds,
            "retry_after_seconds": stats.diagnostics.retry_after_seconds,
            "adapted_query": stats.adapted_query,
            "adaptation_strategy": stats.adaptation_strategy,
            "combination_mode": stats.combination_mode,
            "query_provenance": [
                item.model_dump(mode="json") for item in stats.query_provenance
            ],
            "retention_ratio": stats.retention_ratio,
            "protected_terms": list(stats.protected_terms),
            "run_dedupe_hit": stats.run_dedupe_hit,
            "logical_call_executed": stats.logical_call_executed,
            "triggered_by": list(stats.triggered_by),
            "safe_original_candidate_count": stats.safe_original_candidate_count,
            "safe_original_core_term_coverage": (
                stats.safe_original_core_term_coverage
            ),
            "safe_original_constraint_coverage": (
                stats.safe_original_constraint_coverage
            ),
            "sufficiency_reasons": list(stats.sufficiency_reasons),
            "compact_query_executed": stats.compact_query_executed,
            "compact_query_skipped_reason": stats.compact_query_skipped_reason,
            "source_skipped_reason": stats.source_skipped_reason,
            "remaining_subquery_count": stats.remaining_subquery_count,
            "error_message": stats.error_message,
        },
    )


def clear_retrieval_cache() -> None:
    """Clear the process-local retrieval cache.

    This is primarily intended for tests and local debugging. The cache is
    intentionally in-memory only and is not shared across worker processes.
    """

    with _RETRIEVAL_CACHE_LOCK:
        _RETRIEVAL_CACHE.clear()


def clear_source_cooldowns() -> None:
    """Clear process-local source cooldown state for tests and local debugging."""

    with _SOURCE_FAILURE_COOLDOWNS_LOCK:
        _SOURCE_FAILURE_COOLDOWNS.clear()


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if sources is None:
        return list(DEFAULT_SEARCH_SOURCES)

    normalized: list[str] = []
    seen: set[str] = set()
    for source in sources:
        key = source.strip().lower()
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def _source_registry() -> dict[str, Callable[[str, int], ConnectorSearchResult]]:
    return {
        "openalex": search_openalex_detailed,
        "arxiv": search_arxiv_detailed,
        "semantic_scholar": search_semantic_scholar_detailed,
        "pubmed": search_pubmed_detailed,
        "local_bm25": search_local_bm25_detailed,
        "local_hybrid": search_local_hybrid_detailed,
    }


def _search_with_cache(
    source: str,
    query: str,
    limit_per_source: int,
    search: Callable[[str, int], ConnectorSearchResult],
) -> tuple[ConnectorSearchResult, bool]:
    config = _cache_config()
    if not config.enabled:
        return search(query, limit_per_source), False

    key = _cache_key(source, query, limit_per_source)
    cached = _get_cached_result(key, config.ttl_seconds)
    if cached is not None:
        return (
            cached.model_copy(
                update={
                    "latency_seconds": 0.0,
                    "diagnostics": ConnectorDiagnostics(cache_hit_count=1),
                }
            ),
            True,
        )

    result = search(query, limit_per_source)
    if result.error_message is None:
        _store_cached_result(key, result, config.max_entries)
    return result, False


class _CacheConfig(BaseModel):
    enabled: bool
    ttl_seconds: float
    max_entries: int


def _cache_config() -> _CacheConfig:
    enabled_value = os.getenv("SCHOLAR_AGENT_RETRIEVAL_CACHE", "1")
    enabled = enabled_value not in CACHE_DISABLED_VALUES
    ttl_seconds = _float_env(
        "SCHOLAR_AGENT_RETRIEVAL_CACHE_TTL_SECONDS",
        DEFAULT_CACHE_TTL_SECONDS,
    )
    max_entries = _int_env(
        "SCHOLAR_AGENT_RETRIEVAL_CACHE_MAX_ENTRIES",
        DEFAULT_CACHE_MAX_ENTRIES,
    )
    if ttl_seconds <= 0 or max_entries <= 0:
        enabled = False
    return _CacheConfig(
        enabled=enabled,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
    )


def _cache_key(source: str, query: str, limit_per_source: int) -> _CacheKey:
    return retrieval_cache_identity(source, query, limit_per_source)


def retrieval_cache_identity(
    source: str, query: str, limit_per_source: int
) -> _CacheKey:
    """Return the production cache identity without consulting runtime state."""

    return (
        source.strip().lower(),
        " ".join(query.casefold().split()),
        int(limit_per_source),
    )


def _stable_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        result.append(item)
        seen.add(key)
    return result


def _get_cached_result(
    key: _CacheKey,
    ttl_seconds: float,
) -> ConnectorSearchResult | None:
    now = time.monotonic()
    with _RETRIEVAL_CACHE_LOCK:
        cached = _RETRIEVAL_CACHE.get(key)
        if cached is None:
            return None

        stored_at, result = cached
        if now - stored_at > ttl_seconds:
            _RETRIEVAL_CACHE.pop(key, None)
            return None

        _RETRIEVAL_CACHE.move_to_end(key)
        return result.model_copy(deep=True)


def _store_cached_result(
    key: _CacheKey,
    result: ConnectorSearchResult,
    max_entries: int,
) -> None:
    with _RETRIEVAL_CACHE_LOCK:
        _RETRIEVAL_CACHE[key] = (time.monotonic(), result.model_copy(deep=True))
        _RETRIEVAL_CACHE.move_to_end(key)
        while len(_RETRIEVAL_CACHE) > max_entries:
            _RETRIEVAL_CACHE.popitem(last=False)


def _is_source_in_cooldown(source: str) -> bool:
    cooldown_seconds = _source_cooldown_seconds(source)
    if cooldown_seconds <= 0:
        return False

    now = time.monotonic()
    with _SOURCE_FAILURE_COOLDOWNS_LOCK:
        expires_at = _SOURCE_FAILURE_COOLDOWNS.get(source)
        if expires_at is None:
            return False
        if now >= expires_at:
            _SOURCE_FAILURE_COOLDOWNS.pop(source, None)
            return False
        return True


def _record_source_cooldown(
    source: str,
    *,
    minimum_seconds: float | None = None,
) -> None:
    cooldown_seconds = max(
        _source_cooldown_seconds(source),
        float(minimum_seconds or 0.0),
    )
    if cooldown_seconds <= 0:
        return

    with _SOURCE_FAILURE_COOLDOWNS_LOCK:
        _SOURCE_FAILURE_COOLDOWNS[source] = time.monotonic() + cooldown_seconds


def _source_cooldown_seconds(source: str | None = None) -> float:
    if source == "semantic_scholar":
        value = _float_env(
            "SCHOLAR_AGENT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS",
            DEFAULT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS,
        )
        return max(0.0, value)

    value = _float_env(
        "SCHOLAR_AGENT_SOURCE_COOLDOWN_SECONDS",
        DEFAULT_SOURCE_COOLDOWN_SECONDS,
    )
    return max(0.0, value)


def _has_cooldown_trigger(
    error_message: str | None,
    warnings: list[str],
) -> bool:
    # Connector warnings may include recovered transient failures, for example
    # "HTTP Error 429; retried" followed by a successful response. Cooldown must
    # only be recorded for final connector failures, which are surfaced through
    # error_message.
    return bool(error_message and _is_final_rate_limit(error_message))


def _is_transient_error_message(message: str) -> bool:
    normalized = message.casefold()
    return (
        _has_5xx_status(normalized)
        or "timeout" in normalized
        or "timed out" in normalized
        or "connection reset" in normalized
        or "ssl" in normalized
    )


def _is_final_rate_limit(message: str | None) -> bool:
    if not message:
        return False
    normalized = message.casefold()
    return "http error 429" in normalized or "status: 429" in normalized


def _has_5xx_status(normalized_message: str) -> bool:
    return any(
        f"http error {status}" in normalized_message
        or f"status: {status}" in normalized_message
        for status in range(500, 600)
    )


def sanitize_connector_error_text(value: str, *, max_length: int = 500) -> str:
    """Remove credentials, local paths, and raw control data from connector errors."""

    sanitized = (
        str(value)
        .replace("\x00", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )
    sanitized = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        "Bearer [redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(authorization|proxy-authorization|x-api-key|api[_-]?key|"
        r"access[_-]?token|refresh[_-]?token)(\s*[:=]\s*)[^\s,;&]+",
        r"\1\2[redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token)=)[^&#\s]+",
        r"\1[redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)(?<![\w.])\.env(?:\.[^\s/\\]+)?(?![\w.])",
        "[environment-file]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?<![:\w])/(?:Users|home|tmp|private|var|etc|opt)/[^\s,;]+",
        "[absolute-path]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b[a-z]:\\(?:[^\\\s]+\\)*[^\s,;]+",
        "[absolute-path]",
        sanitized,
    )
    return sanitized[:max_length]


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)
