from __future__ import annotations

import time
import threading
from types import SimpleNamespace

import pytest

from scholar_agent.connectors import ConnectorDiagnostics, ConnectorSearchResult
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.judgement_config import (
    CALIBRATED_RULES_V1_CONFIG,
    CURRENT_RULES_CONFIG,
    judgement_config_hash,
)
from scholar_agent.agents.retriever import (
    RetrievalOutput,
    RetrievalRunContext,
    SourceStats,
    clear_retrieval_cache,
    clear_source_cooldowns,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvolvedSubquery,
    QueryConstraint,
    QueryEvolutionRecord,
    SearchBudget,
    SearchSubquery,
)
from scholar_agent.services import search_service
from scholar_agent.services.search_service import (
    SearchCancelled,
    SearchService,
    _ExecutionSignals,
)


def make_paper(
    title: str,
    *,
    doi: str | None = None,
    openalex_id: str | None = None,
    year: int | None = 2024,
    citation_count: int = 0,
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=year,
        venue="ACL",
        abstract="This paper studies LLM reranking for scientific literature retrieval.",
        identifiers=PaperIdentifiers(doi=doi, openalex_id=openalex_id),
        sources=sources or ["openalex"],
        citation_count=citation_count,
    )


def make_output(
    query: str,
    papers: list[Paper],
    *,
    warnings: list[str] | None = None,
) -> RetrievalOutput:
    return RetrievalOutput(
        query=query,
        requested_sources=["openalex", "arxiv"],
        raw_count=len(papers),
        deduplicated_count=len(papers),
        papers=papers,
        source_stats=[
            SourceStats(
                source="openalex",
                query=query,
                returned_count=len(papers),
                latency_seconds=0.01,
                diagnostics=ConnectorDiagnostics(
                    request_count=1,
                    latency_seconds=0.01,
                ),
            )
        ],
        warnings=warnings or [],
        latency_seconds=0.01,
    )


class _SpawnSafeBlockingRetriever:
    def __call__(
        self,
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source, sources
        if query == "block":
            while True:
                time.sleep(0.01)
        return RetrievalOutput(
            query=query,
            requested_sources=["openalex"],
            raw_count=0,
            deduplicated_count=0,
            papers=[],
            source_stats=[SourceStats(source="openalex", query=query)],
        )


class _SpawnSafeLargeRetriever:
    def __call__(
        self,
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source, sources
        papers = [
            make_paper(f"large-{index}", doi=f"10.123/large-{index}")
            for index in range(300)
        ]
        return RetrievalOutput(
            query=query,
            requested_sources=["openalex"],
            raw_count=len(papers),
            deduplicated_count=len(papers),
            papers=papers,
            source_stats=[SourceStats(source="openalex", query=query, returned_count=len(papers))],
        )


def _batch_signals(seconds: float) -> _ExecutionSignals:
    signals = _ExecutionSignals(None, None)
    signals.set_deadline(time.perf_counter() + seconds)
    return signals


def test_uncooperative_retriever_is_terminated_and_next_task_completes() -> None:
    service = SearchService(retriever=_SpawnSafeBlockingRetriever(), max_workers=2)
    before = {thread.ident for thread in threading.enumerate() if thread.ident}
    started = time.perf_counter()
    outputs = service._retrieve_query_batch(  # noqa: SLF001
        [
            SearchSubquery(query="block", purpose="original_query"),
            SearchSubquery(query="ok", purpose="original_query"),
        ],
        selected_sources=["openalex"],
        limit_per_source=20,
        failure_prefix="test_failed",
        failure_source="test",
        signals=_batch_signals(0.8),
        constraints=QueryConstraint(),
        run_context=RetrievalRunContext(),
        query_adapter_policy="adaptive",
        adaptive_budget_check=lambda papers: None,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 1.5
    assert [item.query for item in outputs] == ["block", "ok"]
    assert outputs[0].source_stats[0].error_message == "timeout:test:0"
    assert outputs[0].source_stats[0].terminal_status == "timeout"
    assert outputs[1].source_stats[0].error_message is None
    time.sleep(0.1)
    leaked = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and "ThreadPoolExecutor" in thread.name
    ]
    assert leaked == []


def test_isolated_retriever_drains_large_payload_before_join() -> None:
    service = SearchService(retriever=_SpawnSafeLargeRetriever(), max_workers=1)
    outputs = service._retrieve_query_batch(  # noqa: SLF001
        [SearchSubquery(query="large", purpose="original_query")],
        selected_sources=["openalex"],
        limit_per_source=20,
        failure_prefix="test_failed",
        failure_source="test",
        signals=_batch_signals(2.0),
        constraints=QueryConstraint(),
        run_context=RetrievalRunContext(),
        query_adapter_policy="adaptive",
        adaptive_budget_check=lambda papers: None,
    )
    assert len(outputs) == 1
    assert outputs[0].raw_count == 300


def test_deadline_marks_queued_work_not_started() -> None:
    service = SearchService(retriever=_SpawnSafeBlockingRetriever(), max_workers=1)
    outputs = service._retrieve_query_batch(  # noqa: SLF001
        [
            SearchSubquery(query="block", purpose="original_query"),
            SearchSubquery(query="queued", purpose="original_query"),
        ],
        selected_sources=["openalex"],
        limit_per_source=20,
        failure_prefix="test_failed",
        failure_source="test",
        signals=_batch_signals(0.8),
        constraints=QueryConstraint(),
        run_context=RetrievalRunContext(),
        query_adapter_policy="adaptive",
        adaptive_budget_check=lambda papers: None,
    )
    assert [item.source_stats[0].error_message for item in outputs] == [
        "timeout:test:0",
        "not_started:test:1",
    ]
    assert outputs[1].source_stats[0].terminal_status == "not_started"


def test_cancellation_terminates_isolated_children_without_new_tasks() -> None:
    cancel_at = time.perf_counter() + 0.1

    def should_cancel() -> bool:
        return time.perf_counter() >= cancel_at

    service = SearchService(retriever=_SpawnSafeBlockingRetriever(), max_workers=1)
    signals = _ExecutionSignals(None, should_cancel)
    signals.set_deadline(time.perf_counter() + 2.0)
    with pytest.raises(SearchCancelled):
        service._retrieve_query_batch(  # noqa: SLF001
            [SearchSubquery(query="block", purpose="original_query")],
            selected_sources=["openalex"],
            limit_per_source=20,
            failure_prefix="test_failed",
            failure_source="test",
            signals=signals,
            constraints=QueryConstraint(),
            run_context=RetrievalRunContext(),
            query_adapter_policy="adaptive",
            adaptive_budget_check=lambda papers: None,
        )
    assert not service._isolated_processes  # noqa: SLF001


def test_run_search_complete_pipeline_with_injected_retriever() -> None:
    calls: list[tuple[str, int, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, limit_per_source, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"LLM Reranking for Retrieval {len(calls)}",
                    doi=f"10.123/{len(calls)}",
                    citation_count=10,
                )
            ],
        )

    def failing_reference_fetcher(paper: Paper, limit: int) -> list[Paper]:
        raise AssertionError("reference_fetcher should not run when refchain is disabled")

    output = SearchService(
        retriever=fake_retriever,
        reference_fetcher=failing_reference_fetcher,
    ).run_search(
        "latest LLM reranking retrieval papers",
        top_k=5,
        current_year=2026,
    )

    assert output.search_plan.subqueries
    assert len(calls) == len(output.search_plan.subqueries)
    assert len(output.retrieval_outputs) == len(output.search_plan.subqueries)
    assert output.query_evolution_records == []
    assert output.refchain_output is None
    assert output.raw_count == len(output.search_plan.subqueries)
    assert output.deduplicated_count == len(output.judgements)
    assert output.ranked_papers
    assert output.synthesis_output is not None
    assert output.synthesis_output.evidence_table
    assert output.latency_seconds >= 0
    assert output.llm_call_count == 0
    assert set(output.stage_latencies) >= {
        "query_understanding",
        "retrieval",
        "judgement",
        "reranking",
        "synthesis",
    }
    assert all(seconds >= 0 for seconds in output.stage_latencies.values())
    assert all(call[1] == output.search_plan.limit_per_source for call in calls)
    assert all(call[2] == ["arxiv", "openalex"] for call in calls)


def test_per_run_judgement_policy_does_not_inherit_other_policy_config() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(query, [make_paper("Graph Retrieval", doi="10.123/policy")])

    service = SearchService(
        retriever=fake_retriever,
        judgement_policy="calibrated_rules_v1",
        judgement_config=CALIBRATED_RULES_V1_CONFIG,
    )

    output = service.run_search(
        "graph retrieval",
        judgement_policy="current_rules",
        enable_refchain=False,
        enable_synthesis=False,
    )

    assert output.judgement_policy == "current_rules"
    assert output.judgement_config_hash == judgement_config_hash(
        CURRENT_RULES_CONFIG
    )


def test_run_search_respects_sources_override_and_filters_unimplemented_sources() -> None:
    calls: list[list[str] | None] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append(sources)
        return make_output(
            query,
            [
                make_paper(
                    "Source Override Paper",
                    doi=f"10.123/source-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking retrieval papers",
        top_k=5,
        current_year=2026,
        sources_override=["arxiv", "semantic_scholar", "pubmed", "openalex", "arxiv"],
    )

    assert output.search_plan.selected_sources == [
        "arxiv",
        "semantic_scholar",
        "pubmed",
        "openalex",
    ]
    assert output.search_plan.subqueries
    assert all(
        subquery.source_hints
        == ["arxiv", "semantic_scholar", "pubmed", "openalex"]
        for subquery in output.search_plan.subqueries
    )
    assert calls
    assert all(
        call == ["arxiv", "semantic_scholar", "pubmed", "openalex"]
        for call in calls
    )
    assert "source_preference_not_implemented:pubmed" not in output.warnings


def test_arxiv_only_fast_allows_first_two_subqueries() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    "Fast arXiv LLM Reranking Paper",
                    doi="10.123/fast-arxiv",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking methods for scientific literature retrieval",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert len(calls) == 2
    assert calls[0] == (output.search_plan.subqueries[0].query, ["arxiv"])
    assert calls[1] == (output.search_plan.subqueries[1].query, ["arxiv"])
    assert len(output.retrieval_outputs) == 2
    assert "fast_arxiv_subquery_skipped_by_limit:1" not in output.warnings


def test_arxiv_only_balanced_does_not_limit_initial_retrieval() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Balanced arXiv Paper {len(calls)}",
                    doi=f"10.123/balanced-arxiv-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking methods for scientific literature retrieval",
        run_profile="balanced",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert len(calls) == len(output.search_plan.subqueries)
    assert all(call[1] == ["arxiv"] for call in calls)
    assert not any(
        warning.startswith("fast_arxiv_subquery_skipped_by_limit:")
        for warning in output.warnings
    )


def test_fast_both_sources_does_not_limit_initial_retrieval() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Fast Both Sources Paper {len(calls)}",
                    doi=f"10.123/fast-both-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking methods for scientific literature retrieval",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["openalex", "arxiv"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert len(calls) == len(output.search_plan.subqueries)
    assert all(call[1] == ["openalex", "arxiv"] for call in calls)
    assert not any(
        warning.startswith("fast_arxiv_subquery_skipped_by_limit:")
        for warning in output.warnings
    )


def test_fast_profile_sends_each_subquery_to_all_selected_sources() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Fast Recommended Paper {len(calls)}",
                    doi=f"10.123/fast-recommended-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "benchmark datasets for scientific literature search agents",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert len(calls) == len(output.search_plan.subqueries)
    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])


def test_search_service_does_not_inject_source_specific_override_query() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        paper = make_paper(
            f"Citation Recommendation Paper {len(calls)}",
            doi=f"10.123/citation-recommendation-{len(calls)}",
            sources=sources or [],
        )
        return RetrievalOutput(
            query=query,
            requested_sources=sources or [],
            raw_count=1,
            deduplicated_count=1,
            papers=[paper],
            source_stats=[
                SourceStats(
                    source=source,
                    query=query,
                    returned_count=1,
                    latency_seconds=0.01,
                )
                for source in sources or []
            ],
            warnings=[],
            latency_seconds=0.01,
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "citation graph methods for paper recommendation",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])
    assert all(call[1] != ["semantic_scholar"] for call in calls)


def test_fast_profile_keeps_source_preferences_uniform_across_subqueries() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"LLM Literature Retrieval Paper {len(calls)}",
                    doi=f"10.123/llm-lit-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking methods for scientific literature retrieval",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])


def test_fast_profile_uses_only_queries_from_search_plan() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Academic Neural Ranking Paper {len(calls)}",
                    doi=f"10.123/academic-neural-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "neural ranking methods for academic search",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])


def test_single_source_preference_runs_all_planned_subqueries() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    "Fast Semantic Scholar Paper",
                    doi="10.123/fast-semantic-only",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "neural ranking methods for academic search",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert sorted(calls) == sorted([
        (subquery.query, ["semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])
    assert len(output.retrieval_outputs) == len(output.search_plan.subqueries)


def test_fast_semantic_scholar_single_subquery_uses_first_subquery() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    "Single Query Semantic Scholar Paper",
                    doi="10.123/single-semantic",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "hello",
        run_profile="fast",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) == 1
    assert calls == [(output.search_plan.subqueries[0].query, ["semantic_scholar"])]
    assert not any(
        warning.startswith("fast_semantic_scholar_subquery_skipped_by_limit:")
        for warning in output.warnings
    )


def test_balanced_semantic_scholar_does_not_limit_subqueries() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Balanced Semantic Scholar Paper {len(calls)}",
                    doi=f"10.123/balanced-semantic-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "benchmark datasets for scientific literature search agents",
        run_profile="balanced",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert len(calls) == len(output.search_plan.subqueries)
    assert all(call[1] == ["arxiv", "semantic_scholar"] for call in calls)
    assert not any(
        warning.startswith("fast_semantic_scholar_subquery_skipped_by_limit:")
        for warning in output.warnings
    )


def test_high_recall_executes_only_generic_planned_queries() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"RAG Evaluation Paper {len(calls)}",
                    doi=f"10.123/rag-eval-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "retrieval augmented generation evaluation benchmark papers",
        run_profile="high_recall",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])
    assert all(call[1] != ["semantic_scholar"] for call in calls)


def test_high_recall_method_query_uses_source_neutral_plan() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Academic Neural Ranking Paper {len(calls)}",
                    doi=f"10.123/academic-neural-high-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "neural ranking methods for academic search",
        run_profile="high_recall",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])


def test_mixed_query_does_not_create_source_only_target_query() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append((query, sources))
        return make_output(
            query,
            [
                make_paper(
                    f"Mixed Target Paper {len(calls)}",
                    doi=f"10.123/mixed-target-{len(calls)}",
                    sources=sources or [],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "RAG evaluation benchmark for academic search neural ranking information retrieval",
        run_profile="high_recall",
        enable_query_evolution=False,
        enable_refchain=False,
        sources_override=["arxiv", "semantic_scholar"],
        current_year=2026,
    )

    semantic_only_calls = [call for call in calls if call[1] == ["semantic_scholar"]]
    assert semantic_only_calls == []
    assert sorted(calls) == sorted([
        (subquery.query, ["arxiv", "semantic_scholar"])
        for subquery in output.search_plan.subqueries
    ])


def test_run_search_deduplicates_across_subqueries() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper("Shared LLM Reranking Paper", doi="10.123/shared"),
                make_paper(
                    f"Unique LLM Reranking Paper {query[:8]}",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                ),
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "latest LLM reranking retrieval benchmark papers",
        run_profile="high_recall",
        current_year=2026,
    )

    assert len(output.search_plan.subqueries) > 1
    assert output.raw_count == len(output.search_plan.subqueries) * 2
    assert output.deduplicated_count < output.raw_count
    assert sum(
        1
        for judgement in output.judgements
        if judgement.paper.identifiers.doi == "10.123/shared"
    ) == 1


def test_run_search_aggregates_warnings() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [make_paper("Clinical LLM Retrieval Paper", doi="10.123/clinical")],
            warnings=["mock_retriever_warning"],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "recent clinical gene therapy PubMed retrieval papers",
        current_year=2026,
    )

    assert "pubmed_not_implemented" not in output.warnings
    assert "mock_retriever_warning" in output.warnings
    assert output.warnings.count("mock_retriever_warning") == 1


def test_run_search_preserves_retrieval_cache_hit_stats() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return RetrievalOutput(
            query=query,
            requested_sources=["openalex"],
            raw_count=1,
            deduplicated_count=1,
            papers=[
                make_paper("LLM Reranking Retrieval Paper", doi="10.123/cache-hit")
            ],
            source_stats=[
                SourceStats(
                    source="openalex",
                    returned_count=1,
                    latency_seconds=0.0,
                    cache_hit=True,
                )
            ],
            warnings=["retrieval_cache_hit:openalex"],
            latency_seconds=0.0,
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        current_year=2026,
    )

    assert output.source_stats[0].source == "openalex"
    assert output.source_stats[0].cache_hit is True
    assert "retrieval_cache_hit:openalex" in output.warnings


def test_run_search_can_disable_synthesis() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [make_paper("LLM Reranking Retrieval Paper", doi="10.123/no-synthesis")],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        enable_synthesis=False,
        current_year=2026,
    )

    assert output.ranked_papers
    assert output.synthesis_output is None


def test_synthesis_output_includes_source_warnings_and_errors() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return RetrievalOutput(
            query=query,
            requested_sources=["openalex", "arxiv"],
            raw_count=1,
            deduplicated_count=1,
            papers=[
                make_paper(
                    "LLM Reranking Retrieval Paper",
                    doi="10.123/source-error",
                )
            ],
            source_stats=[
                SourceStats(
                    source="openalex",
                    returned_count=0,
                    latency_seconds=0.01,
                    error_message="HTTP Error 503: Service Unavailable",
                ),
                SourceStats(
                    source="arxiv",
                    returned_count=1,
                    latency_seconds=0.01,
                ),
            ],
            warnings=["retriever_warning"],
            latency_seconds=0.02,
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        current_year=2026,
    )

    assert output.synthesis_output is not None
    assert "retriever_warning" in output.synthesis_output.limitations
    assert (
        "source_error:openalex:HTTP Error 503: Service Unavailable"
        in output.synthesis_output.limitations
    )
    expected_error_count = sum(
        1 for stats in output.source_stats if stats.error_message
    )
    assert (
        output.synthesis_output.citation_coverage.source_error_count
        == expected_error_count
    )


def test_run_search_top_k_is_applied() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Reranking for Literature Retrieval",
                    doi="10.123/a",
                ),
                make_paper(
                    "Neural Retrieval with Large Language Models",
                    doi="10.123/b",
                ),
                make_paper(
                    "Transformer Ranking for Scientific Search",
                    doi="10.123/c",
                ),
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        top_k=2,
        current_year=2026,
    )

    assert len(output.ranked_papers) == 2
    assert [paper.rank for paper in output.ranked_papers] == [1, 2]


def test_run_search_preserves_pre_top_k_ranked_candidates() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    title,
                    doi=f"10.123/pretop-{index}",
                    citation_count=index,
                )
                for index, title in enumerate(
                    [
                        "LLM Reranking for Scientific Literature Retrieval",
                        "Neural Ranking Methods for Academic Search",
                        "Transformer Retrieval Models for Papers",
                        "Scientific Search Benchmark for Literature Agents",
                        "Citation Graph Retrieval for Scholarly Papers",
                        "RAG Evaluation for Scientific Literature",
                        "Machine Learning Paper Recommendation",
                        "Semantic Search for Academic Documents",
                    ]
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        top_k=5,
        current_year=2026,
        enable_synthesis=False,
    )

    assert len(output.ranked_papers) == 5
    assert len(output.all_ranked_papers) == 8
    assert [paper.rank for paper in output.ranked_papers] == [1, 2, 3, 4, 5]
    assert [paper.rank for paper in output.all_ranked_papers] == list(range(1, 9))


def test_run_search_with_query_evolution_retrieves_evolved_queries() -> None:
    query = "latest LLM reranking retrieval papers"
    expected_plan = analyze_query(query, current_year=2026)
    initial_queries = {subquery.query for subquery in expected_plan.subqueries}
    calls: list[str] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append(query)
        if query in initial_queries:
            return make_output(
                query,
                [
                    make_paper(
                        "LLM Reranking for Scientific Literature Retrieval",
                        doi=f"10.123/initial-{len(calls)}",
                        citation_count=1,
                    )
                ],
            )
        return make_output(
            query,
            [
                make_paper(
                    "Advanced LLM Reranking for Scientific Literature Retrieval",
                    doi=f"10.123/evolved-{len(calls)}",
                    citation_count=500,
                    sources=["openalex", "arxiv"],
                )
            ],
            warnings=["evolved_retriever_warning"],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        query,
        top_k=5,
        enable_query_evolution=True,
        query_evolution_policy="seed_expansion",
        current_year=2026,
    )

    assert output.query_evolution_records
    assert output.query_evolution_records[0].generated_queries
    assert len(calls) > len(initial_queries)
    assert any(call not in initial_queries for call in calls)
    assert output.raw_count == len(output.retrieval_outputs)
    assert output.search_diagnostics.request_count == len(calls)
    assert "evolved_retriever_warning" in output.warnings


def test_query_evolution_results_participate_in_final_ranking() -> None:
    query = "latest LLM reranking retrieval papers"
    expected_plan = analyze_query(query, current_year=2026)
    initial_queries = {subquery.query for subquery in expected_plan.subqueries}

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        if query in initial_queries:
            return make_output(
                query,
                [
                    make_paper(
                        "LLM Reranking for Scientific Literature Retrieval",
                        doi=f"10.123/initial-{sum(ord(char) for char in query)}",
                        citation_count=0,
                    )
                ],
            )
        return make_output(
            query,
            [
                make_paper(
                    "High Authority LLM Reranking Retrieval Paper",
                    doi=f"10.123/evolved-{sum(ord(char) for char in query)}",
                    citation_count=1000,
                    sources=["openalex", "arxiv"],
                )
            ],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        query,
        top_k=3,
        enable_query_evolution=True,
        query_evolution_policy="seed_expansion",
        current_year=2026,
    )

    assert output.query_evolution_records[0].generated_queries
    assert any(
        ranked.paper.title == "High Authority LLM Reranking Retrieval Paper"
        for ranked in output.ranked_papers
    )
    assert output.ranked_papers[0].paper.title == "High Authority LLM Reranking Retrieval Paper"


def test_query_evolution_used_queries_skip_duplicate_retrieval(monkeypatch) -> None:
    query = "LLM reranking retrieval papers"
    calls: list[str] = []
    captured_used_queries: set[str] = set()

    def fake_evolve_queries(
        query_analysis,
        search_plan,
        judgements,
        ranked_papers,
        used_queries,
        options=None,
    ) -> QueryEvolutionRecord:
        captured_used_queries.update(used_queries)
        duplicate_query = search_plan.subqueries[0].query
        return QueryEvolutionRecord(
            seed_count=1,
            generated_queries=[
                EvolvedSubquery(
                    query=duplicate_query,
                    source_hints=["openalex", "arxiv"],
                    priority=1,
                    purpose="duplicate_for_test",
                ),
                EvolvedSubquery(
                    query="unique evolved LLM reranking retrieval query",
                    source_hints=["openalex", "arxiv"],
                    priority=2,
                    purpose="unique_for_test",
                ),
            ],
        )

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append(query)
        return make_output(
            query,
            [
                make_paper(
                    f"Paper for {query[:24]}",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                )
            ],
        )

    monkeypatch.setattr(search_service, "evolve_queries", fake_evolve_queries)

    output = SearchService(retriever=fake_retriever).run_search(
        query,
        enable_query_evolution=True,
        current_year=2026,
    )

    initial_queries = {subquery.query for subquery in output.search_plan.subqueries}
    duplicate_query = output.search_plan.subqueries[0].query
    assert captured_used_queries == initial_queries
    assert calls.count(duplicate_query) == 1
    assert "unique evolved LLM reranking retrieval query" in calls
    assert "duplicate_evolved_query_skipped" in output.warnings


def test_query_evolution_adapted_duplicate_does_not_repeat_external_call(
    monkeypatch,
) -> None:
    calls: list[str] = []
    clear_retrieval_cache()
    clear_source_cooldowns()

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    "Graph Retrieval Paper",
                    doi="10.123/graph-retrieval",
                    sources=["openalex"],
                )
            ],
            diagnostics=ConnectorDiagnostics(request_count=1),
        )

    def fake_evolve_queries(
        query_analysis,
        search_plan,
        judgements,
        ranked_papers,
        used_queries,
        options=None,
    ) -> QueryEvolutionRecord:
        return QueryEvolutionRecord(
            seed_count=1,
            generated_queries=[
                EvolvedSubquery(
                    query="Could you list some papers about graph retrieval?",
                    source_hints=["openalex"],
                    priority=1,
                    purpose="generic_rephrasing",
                )
            ],
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        fake_openalex,
    )
    monkeypatch.setattr(search_service, "evolve_queries", fake_evolve_queries)

    output = SearchService().run_search(
        "graph retrieval papers",
        sources_override=["openalex"],
        enable_query_evolution=True,
        enable_synthesis=False,
        collect_diagnostics=True,
        query_adapter_policy="hybrid",
    )

    assert len(calls) == len(set(calls))
    assert calls.count("graph retrieval machine learning") == 1
    assert "Could you list some papers about graph retrieval" in calls
    evolved = next(
        item
        for item in output.stage_snapshots
        if item.stage == "query_evolution_retrieval"
    )
    assert any(
        call.run_dedupe_hit
        and call.adapted_query == "graph retrieval machine learning"
        and call.source_skipped_reason == "duplicate_adapted_query"
        for call in evolved.retrieval_calls
    )


def test_coverage_gap_policy_applies_candidate_quality_gate(monkeypatch) -> None:
    query = "graph neural networks for molecular prediction"
    expected_plan = analyze_query(
        query,
        explicit_constraints=QueryConstraint(
            methods=["message passing"],
            explicit_fields=["methods"],
        ),
    )
    initial_queries = {item.query for item in expected_plan.subqueries}

    def fake_evolve_queries(
        query_analysis,
        search_plan,
        judgements,
        ranked_papers,
        used_queries,
        options=None,
    ) -> QueryEvolutionRecord:
        assert options.policy == "coverage_gap"
        return QueryEvolutionRecord(
            policy="coverage_gap",
            seed_count=1,
            generated_queries=[
                EvolvedSubquery(
                    query="graph neural networks message passing molecular",
                    source_hints=["arxiv"],
                    priority=1,
                    purpose="query_evolution_coverage_gap_method",
                    generation_policy="coverage_gap",
                    gap_dimensions=["method"],
                )
            ],
        )

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        if query in initial_queries:
            return make_output(
                query,
                [
                    make_paper(
                        "Graph Neural Networks for Molecular Prediction",
                        doi="10.123/quality-initial",
                    )
                ],
            )
        return make_output(
            query,
            [
                make_paper(
                    "Message Passing for Molecular Graphs",
                    doi="10.123/quality-accepted",
                ),
                make_paper(
                    "Message Passing for Road Traffic Forecasting",
                    doi="10.123/quality-filtered",
                ),
            ],
        )

    monkeypatch.setattr(search_service, "evolve_queries", fake_evolve_queries)
    output = SearchService(retriever=fake_retriever).run_search(
        query,
        enable_query_evolution=True,
        query_evolution_policy="coverage_gap",
        sources_override=["arxiv"],
        explicit_constraints=QueryConstraint(
            methods=["message passing"],
            explicit_fields=["methods"],
        ),
        enable_synthesis=False,
    )

    record = output.query_evolution_records[0]
    assert output.search_plan.query_evolution_policy == "coverage_gap"
    assert record.quality_gate.raw_candidate_count == 2
    assert record.quality_gate.accepted_candidate_count == 1
    assert record.quality_gate.filtered_reason_counts == {"no_topic_match": 1}
    assert all(
        ranked.paper.identifiers.doi != "10.123/quality-filtered"
        for ranked in output.all_ranked_papers
    )


def test_run_search_with_refchain_calls_reference_fetcher() -> None:
    calls: list[tuple[str, int]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Reranking for Scientific Literature Retrieval",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                    openalex_id=f"W{sum(ord(char) for char in query)}",
                )
            ],
        )

    def fake_reference_fetcher(paper: Paper, limit: int) -> list[Paper]:
        calls.append((paper.title, limit))
        return [
            make_paper(
                "Reference LLM Reranking Retrieval Paper",
                doi="10.123/refchain-reference",
                openalex_id="WREFCHAIN",
            )
        ]

    output = SearchService(
        retriever=fake_retriever,
        reference_fetcher=fake_reference_fetcher,
    ).run_search(
        "LLM reranking retrieval papers",
        enable_refchain=True,
        current_year=2026,
    )

    assert calls
    assert output.refchain_output is not None
    assert output.refchain_output.references
    assert output.raw_count == (
        sum(item.raw_count for item in output.retrieval_outputs) + len(calls)
    )
    assert output.source_stats[-1].source == "refchain"
    assert output.source_stats[-1].returned_count == len(calls)


def test_refchain_references_participate_in_final_ranking() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Reranking for Scientific Literature Retrieval",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                    openalex_id=f"W{sum(ord(char) for char in query)}",
                    citation_count=0,
                )
            ],
        )

    def fake_reference_fetcher(paper: Paper, limit: int) -> list[Paper]:
        return [
            make_paper(
                "High Authority LLM Reranking Retrieval Reference",
                doi="10.123/high-authority-reference",
                openalex_id="WHIGHREF",
                citation_count=2000,
                sources=["openalex", "arxiv"],
            )
        ]

    output = SearchService(
        retriever=fake_retriever,
        reference_fetcher=fake_reference_fetcher,
    ).run_search(
        "LLM reranking retrieval papers",
        top_k=3,
        enable_refchain=True,
        current_year=2026,
    )

    assert output.refchain_output is not None
    assert any(
        ranked.paper.title == "High Authority LLM Reranking Retrieval Reference"
        for ranked in output.ranked_papers
    )
    assert output.ranked_papers[0].paper.title == (
        "High Authority LLM Reranking Retrieval Reference"
    )


def test_refchain_missing_seed_identifier_warning_is_aggregated() -> None:
    called = {"value": False}

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Reranking for Scientific Literature Retrieval",
                    citation_count=10,
                )
            ],
        )

    def fake_reference_fetcher(paper: Paper, limit: int) -> list[Paper]:
        called["value"] = True
        return []

    output = SearchService(
        retriever=fake_retriever,
        reference_fetcher=fake_reference_fetcher,
    ).run_search(
        "LLM reranking retrieval papers",
        enable_refchain=True,
        current_year=2026,
    )

    assert called["value"] is False
    assert output.refchain_output is not None
    assert "refchain_seed_missing_supported_identifier:1" in output.warnings
    assert output.source_stats[-1].source == "refchain"
    assert output.source_stats[-1].returned_count == 0


def test_query_evolution_and_refchain_can_run_together() -> None:
    query = "latest LLM reranking retrieval papers"
    expected_plan = analyze_query(query, current_year=2026)
    initial_queries = {subquery.query for subquery in expected_plan.subqueries}
    retriever_calls: list[str] = []
    reference_calls: list[str] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        retriever_calls.append(query)
        if query in initial_queries:
            return make_output(
                query,
                [
                    make_paper(
                        "LLM Reranking for Scientific Literature Retrieval",
                        doi=f"10.123/initial-{len(retriever_calls)}",
                        openalex_id=f"WINITIAL{len(retriever_calls)}",
                    )
                ],
            )
        return make_output(
            query,
            [
                make_paper(
                    "Evolved LLM Reranking Retrieval Paper",
                    doi=f"10.123/evolved-{len(retriever_calls)}",
                    openalex_id=f"WEVOLVED{len(retriever_calls)}",
                    citation_count=100,
                )
            ],
        )

    def fake_reference_fetcher(
        paper: Paper,
        limit: int,
    ) -> ConnectorSearchResult:
        reference_calls.append(paper.title)
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    "RefChain LLM Reranking Retrieval Reference",
                    doi=f"10.123/ref-{len(reference_calls)}",
                    openalex_id=f"WREF{len(reference_calls)}",
                )
            ],
            diagnostics=ConnectorDiagnostics(request_count=2),
        )

    output = SearchService(
        retriever=fake_retriever,
        reference_fetcher=fake_reference_fetcher,
    ).run_search(
        query,
        enable_query_evolution=True,
        query_evolution_policy="seed_expansion",
        enable_refchain=True,
        current_year=2026,
    )

    assert output.query_evolution_records
    assert output.refchain_output is not None
    assert any(call not in initial_queries for call in retriever_calls)
    assert reference_calls
    assert output.refchain_output.references
    assert output.ranked_papers
    assert output.synthesis_output is not None
    assert output.synthesis_output.evidence_table
    assert output.stage_latencies["query_evolution"] >= 0
    assert output.stage_latencies["refchain"] >= 0
    assert output.search_diagnostics.request_count == len(retriever_calls)
    assert output.reference_diagnostics.request_count == 2 * len(reference_calls)


def test_run_search_can_use_llm_query_understanding_with_injected_client() -> None:
    llm_client = FakeLLMClient()
    calls: list[str] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append(query)
        return make_output(
            query,
            [
                make_paper(
                    "LLM Planned Reranking Retrieval Paper",
                    doi="10.123/llm-plan",
                )
            ],
        )

    output = SearchService(
        retriever=fake_retriever,
        llm_client=llm_client,
    ).run_search(
        "latest LLM reranking retrieval papers",
        current_year=2026,
        enable_llm_query_understanding=True,
    )

    assert llm_client.calls == 1
    assert output.llm_call_count == 1
    assert output.llm_prompt_tokens == 11
    assert output.llm_completion_tokens == 7
    assert output.llm_total_tokens == 18
    assert output.search_plan.subqueries[0].query == "LLM reranking scientific retrieval"
    assert calls == ["LLM reranking scientific retrieval"]
    assert "llm_query_understanding_used" in output.warnings
    assert output.ranked_papers


def test_run_search_can_use_llm_judgement_with_injected_client() -> None:
    llm_client = FakeJudgementLLMClient()

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Judged Reranking Retrieval Paper",
                    doi="10.123/llm-judgement",
                )
            ],
        )

    output = SearchService(
        retriever=fake_retriever,
        llm_client=llm_client,
    ).run_search(
        "LLM reranking retrieval papers",
        current_year=2026,
        enable_llm_judgement=True,
    )

    assert llm_client.calls == 1
    assert output.llm_call_count == 1
    assert output.llm_prompt_tokens == 23
    assert output.llm_completion_tokens == 9
    assert output.llm_total_tokens == 32
    assert output.judgements[0].score == 0.94
    assert output.judgements[0].category == "highly_relevant"
    assert "llm_judgement_used" in output.warnings
    assert output.ranked_papers


def test_run_search_respects_llm_judgement_max_papers_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_MAX_PAPERS", "1")
    llm_client = FakeJudgementLLMClient()

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper(
                    "LLM Judged Reranking Retrieval Paper",
                    doi="10.123/llm-judgement",
                ),
                make_paper(
                    "Rule Judged Retrieval Paper",
                    doi="10.123/rule-judgement",
                ),
                make_paper(
                    "Skipped LLM Judgement Paper",
                    doi="10.123/skipped-judgement",
                ),
            ],
        )

    output = SearchService(
        retriever=fake_retriever,
        llm_client=llm_client,
    ).run_search(
        "LLM reranking retrieval papers",
        current_year=2026,
        enable_llm_judgement=True,
    )

    assert llm_client.calls == 1
    assert output.llm_call_count == 1
    assert "llm_judgement_used" in output.warnings
    assert "llm_judgement_skipped_by_limit:1" in output.warnings
    assert "llm_judgement_skipped_by_limit:2" in output.warnings
    assert output.ranked_papers


def test_run_search_empty_query_raises_value_error() -> None:
    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        raise AssertionError("retriever should not be called for an empty query")

    with pytest.raises(ValueError):
        SearchService(retriever=fake_retriever).run_search("   ")


def test_run_search_uses_injected_retriever_without_network(monkeypatch) -> None:
    monkeypatch.setattr(
        "scholar_agent.services.search_service.retrieve_papers",
        lambda *args, **kwargs: pytest.fail("default retriever should not be used"),
    )
    called = {"value": False}

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        called["value"] = True
        return make_output(
            query,
            [make_paper("LLM Reranking Retrieval Paper", doi="10.123/no-network")],
        )

    output = SearchService(retriever=fake_retriever).run_search(
        "LLM reranking retrieval papers",
        current_year=2026,
    )

    assert called["value"] is True
    assert output.llm_prompt_tokens == 0
    assert output.llm_completion_tokens == 0
    assert output.llm_total_tokens == 0
    assert output.ranked_papers


def test_run_search_concurrent_mode_preserves_retrieval_output_order() -> None:
    query = "latest LLM reranking retrieval benchmark papers"
    expected_plan = analyze_query(
        query,
        run_profile="high_recall",
        current_year=2026,
    )
    expected_queries = [subquery.query for subquery in expected_plan.subqueries]
    delay_by_query = {
        subquery: (len(expected_queries) - index) * 0.01
        for index, subquery in enumerate(expected_queries)
    }

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        time.sleep(delay_by_query[query])
        return make_output(
            query,
            [
                make_paper(
                    f"Paper for {query[:24]}",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                )
            ],
        )

    output = SearchService(retriever=fake_retriever, max_workers=4).run_search(
        query,
        run_profile="high_recall",
        current_year=2026,
    )

    assert [item.query for item in output.retrieval_outputs] == expected_queries
    assert len(output.ranked_papers) > 0


def test_run_search_subquery_failure_keeps_other_results_and_warnings() -> None:
    query = "latest LLM reranking retrieval benchmark papers"
    expected_plan = analyze_query(
        query,
        run_profile="high_recall",
        current_year=2026,
    )
    failing_query = expected_plan.subqueries[1].query

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        if query == failing_query:
            raise RuntimeError("mock subquery outage")
        return make_output(
            query,
            [
                make_paper(
                    f"Recovered Paper {query[:16]}",
                    doi=f"10.123/{sum(ord(char) for char in query)}",
                )
            ],
            warnings=["retriever_warning"],
        )

    output = SearchService(retriever=fake_retriever, max_workers=4).run_search(
        query,
        run_profile="high_recall",
        current_year=2026,
    )

    assert [item.query for item in output.retrieval_outputs] == [
        subquery.query for subquery in expected_plan.subqueries
    ]
    assert output.retrieval_outputs[1].raw_count == 0
    assert output.retrieval_outputs[1].source_stats[0].source == "subquery"
    assert output.retrieval_outputs[1].source_stats[0].terminal_status == "source_failure"
    assert output.retrieval_outputs[1].source_stats[0].error_message is not None
    assert "subquery_failed:1:mock subquery outage" in output.warnings
    assert output.warnings.count("retriever_warning") == 1
    assert output.raw_count == len(expected_plan.subqueries) - 1
    assert output.ranked_papers


def test_run_search_emits_events_in_real_stage_order() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    output = SearchService(
        retriever=lambda query, limit_per_source=20, sources=None: make_output(
            query,
            [make_paper("Event Paper", doi="10.123/event")],
        ),
        max_workers=1,
    ).run_search(
        "LLM reranking retrieval papers",
        enable_query_evolution=False,
        enable_refchain=False,
        event_callback=lambda name, payload: events.append((name, payload)),
    )

    names = [name for name, _ in events]
    required_order = [
        "query_understanding_started",
        "query_understanding_completed",
        "retrieval_started",
        "connector_started",
        "connector_completed",
        "retrieval_completed",
        "deduplication_completed",
        "judgement_started",
        "judgement_completed",
        "reranking_started",
        "reranking_completed",
        "query_evolution_skipped",
        "refchain_skipped",
        "synthesis_started",
        "synthesis_completed",
    ]
    positions = [names.index(name) for name in required_order]
    assert positions == sorted(positions)
    assert output.ranked_papers
    assert names.index("connector_started") < names.index("connector_completed")


def test_event_callback_failure_becomes_warning_without_failing_search() -> None:
    def broken_callback(name: str, payload: dict[str, object]) -> None:
        del name, payload
        raise RuntimeError("callback unavailable")

    output = SearchService(
        retriever=lambda query, limit_per_source=20, sources=None: make_output(
            query,
            [make_paper("Callback Paper", doi="10.123/callback")],
        )
    ).run_search(
        "LLM reranking retrieval papers",
        event_callback=broken_callback,
    )

    assert output.ranked_papers
    assert any(item.startswith("event_callback_failed:") for item in output.warnings)


def test_cancellation_between_subqueries_stops_new_retrieval_and_synthesis() -> None:
    calls: list[str] = []
    events: list[str] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        calls.append(query)
        return make_output(query, [make_paper("Cancel Paper", doi="10.123/cancel")])

    with pytest.raises(SearchCancelled):
        SearchService(retriever=fake_retriever, max_workers=1).run_search(
            "latest LLM reranking retrieval benchmark papers",
            run_profile="high_recall",
            should_cancel=lambda: len(calls) >= 1,
            event_callback=lambda name, payload: events.append(name),
        )

    assert len(calls) == 1
    assert "synthesis_started" not in events
    assert "synthesis_completed" not in events


def test_budget_stop_event_is_emitted_at_actual_stop() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    SearchService(
        retriever=lambda query, limit_per_source=20, sources=None: make_output(
            query,
            [make_paper("Budget Paper", doi="10.123/budget")],
        ),
        max_workers=1,
    ).run_search(
        "latest LLM reranking retrieval papers",
        enable_query_evolution=True,
        budget=SearchBudget(max_search_rounds=1),
        event_callback=lambda name, payload: events.append((name, payload)),
    )

    budget_events = [payload for name, payload in events if name == "budget_stop"]
    assert budget_events
    assert budget_events[0]["reason"] == "budget_stop:max_search_rounds"
    assert any(name == "query_evolution_skipped" for name, _ in events)


def test_run_search_collects_compact_stage_snapshots_only_when_requested() -> None:
    paper = make_paper("Diagnostic Paper", doi="10.123/diagnostic")

    def retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(query, [paper], sources=sources)

    without = SearchService(retriever=retriever).run_search(
        "diagnostic query",
        enable_synthesis=False,
    )
    with_diagnostics = SearchService(retriever=retriever).run_search(
        "diagnostic query",
        enable_synthesis=False,
        collect_diagnostics=True,
    )

    assert without.stage_snapshots == []
    snapshots = {item.stage: item for item in with_diagnostics.stage_snapshots}
    assert snapshots["initial_retrieval"].status == "completed"
    assert snapshots["initial_deduplicated"].status == "completed"
    assert snapshots["initial_judged"].status == "completed"
    assert snapshots["initial_reranked"].status == "completed"
    assert snapshots["query_evolution_retrieval"].status == "skipped"
    assert snapshots["refchain_retrieval"].status == "skipped"
    assert snapshots["final_ranked"].status == "completed"
    serialized = str(with_diagnostics.stage_snapshots)
    assert paper.abstract not in serialized


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls += 1
        self.token_usage.prompt_tokens += 11
        self.token_usage.completion_tokens += 7
        self.token_usage.total_tokens += 18
        return {
            "language": "en",
            "intent": "recent_progress",
            "domain": "machine_learning",
            "selected_sources": ["openalex", "arxiv"],
            "subqueries": [
                {
                    "query": "LLM reranking scientific retrieval",
                    "source_hints": ["openalex", "arxiv"],
                    "purpose": "llm_test_plan",
                }
            ],
            "warnings": [],
        }


class FakeJudgementLLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls += 1
        self.token_usage.prompt_tokens += 23
        self.token_usage.completion_tokens += 9
        self.token_usage.total_tokens += 32
        return {
            "judgements": [
                {
                    "paper_index": 0,
                    "score": 0.94,
                    "category": "highly_relevant",
                    "reasoning": "Strong metadata match for the requested topic.",
                    "evidence": [
                        {
                            "source": "title",
                            "text": "LLM Judged Reranking Retrieval Paper",
                            "confidence": 0.92,
                        }
                    ],
                    "matched_terms": ["LLM", "reranking", "retrieval"],
                    "warnings": [],
                }
            ],
            "warnings": [],
        }


def test_cancellation_between_llm_batches_stops_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_BATCH_SIZE", "1")
    llm_client = FakeJudgementLLMClient()

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        return make_output(
            query,
            [
                make_paper("LLM Batch One", doi="10.123/batch-one"),
                make_paper("LLM Batch Two", doi="10.123/batch-two"),
            ],
        )

    with pytest.raises(SearchCancelled):
        SearchService(
            retriever=fake_retriever,
            llm_client=llm_client,
        ).run_search(
            "LLM reranking retrieval papers",
            enable_llm_judgement=True,
            should_cancel=lambda: llm_client.calls >= 1,
        )

    assert llm_client.calls == 1
