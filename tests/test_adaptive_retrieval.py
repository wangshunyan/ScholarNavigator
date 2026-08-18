from __future__ import annotations

import inspect

import pytest

from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.retriever import (
    clear_retrieval_cache,
    clear_source_cooldowns,
    evaluate_retrieval_sufficiency,
    retrieve_papers,
)
from scholar_agent.connectors import ConnectorDiagnostics, ConnectorSearchResult
from scholar_agent.core.api_schemas import SearchRunCreateRequest
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryConstraint, SearchBudget
from scholar_agent.retrieval.query_adapter import adapt_queries_for_source
from scholar_agent.services.search_service import SearchService


@pytest.fixture(autouse=True)
def reset_retrieval_state(monkeypatch: pytest.MonkeyPatch):
    clear_retrieval_cache()
    clear_source_cooldowns()
    monkeypatch.setenv("SCHOLAR_AGENT_RETRIEVAL_CACHE", "0")
    yield
    clear_retrieval_cache()
    clear_source_cooldowns()


def _paper(index: int, text: str, *, abstract: bool = True) -> Paper:
    distinctive = [
        "Albatross",
        "Borealis",
        "Cypress",
        "Dragonfly",
        "Equinox",
        "Firefly",
        "Granite",
        "Harbor",
        "Ionian",
        "Juniper",
        "Keystone",
        "Lantern",
    ][index % 12]
    return Paper(
        title=f"{distinctive}: {text} study {index}",
        authors=["Researcher"],
        year=2024,
        abstract=f"Evidence about {text}." if abstract else "",
        identifiers=PaperIdentifiers(doi=f"10.1000/{index}"),
        sources=["openalex"],
    )


def _install_openalex(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ConnectorSearchResult],
) -> list[str]:
    calls: list[str] = []

    def fake(query: str, limit: int) -> ConnectorSearchResult:
        del limit
        calls.append(query)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed", fake
    )
    return calls


def test_adaptive_skips_compact_for_sufficient_safe_results(monkeypatch) -> None:
    papers = [_paper(index, "dense retrieval methods") for index in range(10)]
    calls = _install_openalex(
        monkeypatch, [ConnectorSearchResult(papers=papers)]
    )

    output = retrieve_papers(
        "Could you list papers about dense retrieval methods?",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 1
    compact = output.source_stats[1]
    assert compact.logical_call_executed is False
    assert compact.compact_query_executed is False
    assert compact.compact_query_skipped_reason == "adaptive_sufficient_results"


def test_adaptive_executes_compact_after_empty_safe_result(monkeypatch) -> None:
    calls = _install_openalex(
        monkeypatch,
        [
            ConnectorSearchResult(),
            ConnectorSearchResult(papers=[_paper(1, "dense retrieval")]),
        ],
    )

    output = retrieve_papers(
        "Please find dense retrieval papers",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 2
    assert output.source_stats[1].compact_query_executed is True
    assert "adaptive_empty_results" in output.source_stats[1].triggered_by


def test_adaptive_executes_compact_for_low_candidate_count(monkeypatch) -> None:
    calls = _install_openalex(
        monkeypatch,
        [ConnectorSearchResult(papers=[_paper(1, "dense retrieval")])],
    )

    output = retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 2
    assert "adaptive_low_candidate_count" in output.source_stats[1].triggered_by


@pytest.mark.parametrize(
    ("constraints", "reason"),
    [
        (
            QueryConstraint(must_include_terms=["causal evidence"]),
            "adaptive_missing_must_have_terms",
        ),
        (
            QueryConstraint(datasets=["NovelCorpus"]),
            "adaptive_missing_datasets",
        ),
    ],
)
def test_adaptive_executes_when_required_dimension_is_uncovered(
    monkeypatch: pytest.MonkeyPatch,
    constraints: QueryConstraint,
    reason: str,
) -> None:
    papers = [_paper(index, "dense retrieval methods") for index in range(10)]
    calls = _install_openalex(
        monkeypatch, [ConnectorSearchResult(papers=papers)]
    )

    output = retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        constraints=constraints,
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 2
    assert reason in output.source_stats[1].sufficiency_reasons


def test_adaptive_executes_when_candidate_metadata_is_insufficient(monkeypatch) -> None:
    papers = [
        _paper(index, "dense retrieval methods", abstract=False)
        for index in range(10)
    ]
    calls = _install_openalex(
        monkeypatch, [ConnectorSearchResult(papers=papers)]
    )

    output = retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 2
    assert "adaptive_low_metadata_coverage" in output.source_stats[1].triggered_by


def test_adaptive_equivalent_query_only_requests_once(monkeypatch) -> None:
    calls = _install_openalex(monkeypatch, [ConnectorSearchResult()])

    output = retrieve_papers(
        "graph retrieval",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 1
    assert output.source_stats[1].compact_query_skipped_reason == (
        "adaptive_equivalent_query"
    )


def test_adaptive_low_retention_compact_is_not_requested(monkeypatch) -> None:
    query = " ".join(f"SpecializedTerm{index}" for index in range(30))
    calls = _install_openalex(monkeypatch, [ConnectorSearchResult()])

    output = retrieve_papers(
        query,
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 1
    assert output.source_stats[1].compact_query_skipped_reason == (
        "adaptive_low_information_retention"
    )


def test_adaptive_executes_compact_after_safe_query_is_truncated(monkeypatch) -> None:
    query = " ".join(f"Topic{index}" for index in range(24))
    papers = [_paper(index, "retrieval evidence") for index in range(10)]
    calls = _install_openalex(
        monkeypatch, [ConnectorSearchResult(papers=papers)]
    )

    output = retrieve_papers(
        query,
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 2
    assert "adaptive_safe_original_truncated" in output.source_stats[1].triggered_by


@pytest.mark.parametrize(
    "budget_reason",
    ["max_latency_seconds_reached", "max_candidate_papers_reached"],
)
def test_adaptive_budget_exhaustion_skips_compact(
    monkeypatch: pytest.MonkeyPatch,
    budget_reason: str,
) -> None:
    calls = _install_openalex(monkeypatch, [ConnectorSearchResult()])

    output = retrieve_papers(
        "Please find dense retrieval papers",
        sources=["openalex"],
        query_adapter_policy="adaptive",
        adaptive_budget_check=lambda papers: budget_reason,
    )

    assert len(calls) == 1
    assert output.source_stats[1].compact_query_skipped_reason == (
        "adaptive_budget_exhausted"
    )
    assert budget_reason in output.source_stats[1].triggered_by


def test_search_service_candidate_budget_blocks_adaptive_second_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(query: str, limit: int) -> ConnectorSearchResult:
        del limit
        calls.append(query)
        return ConnectorSearchResult(
            papers=[_paper(1, "dense retrieval methods")],
            diagnostics=ConnectorDiagnostics(request_count=1),
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed", fake
    )
    output = SearchService(max_workers=1).run_search(
        "Could you list papers about dense retrieval methods?",
        sources_override=["openalex"],
        enable_synthesis=False,
        budget=SearchBudget(
            max_search_rounds=1,
            max_candidate_papers=1,
            max_llm_calls=0,
            max_total_tokens=0,
            max_latency_seconds=90,
        ),
        query_adapter_policy="adaptive",
    )

    compact = [
        item for item in output.source_stats if item.adaptation_strategy == "compact_core"
    ]
    assert compact
    assert all(item.logical_call_executed is False for item in compact)
    assert all(
        item.compact_query_skipped_reason == "adaptive_budget_exhausted"
        for item in compact
    )
    assert sum(item.diagnostics.request_count for item in output.source_stats) == len(
        calls
    )
    assert output.budget_status.stop_reasons == [
        "budget_stop:max_candidate_papers"
    ]


def test_adaptive_arxiv_uses_second_query_only_when_needed(monkeypatch) -> None:
    calls: list[str] = []

    def fake(query: str, limit: int) -> ConnectorSearchResult:
        del limit
        calls.append(query)
        return ConnectorSearchResult(
            papers=[_paper(index, "dense retrieval methods") for index in range(10)]
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_arxiv_detailed", fake)

    retrieve_papers(
        "Please find dense retrieval methods",
        sources=["arxiv"],
        query_adapter_policy="adaptive",
    )

    assert len(calls) == 1


def test_semantic_scholar_429_does_not_schedule_compact(monkeypatch) -> None:
    calls: list[str] = []

    def fake(query: str, limit: int) -> ConnectorSearchResult:
        del query, limit
        calls.append("called")
        return ConnectorSearchResult(error_message="HTTP Error 429: Too Many Requests")

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed", fake
    )

    output = retrieve_papers(
        "Please find dense retrieval papers",
        sources=["semantic_scholar"],
        query_adapter_policy="adaptive",
    )

    assert calls == ["called"]
    assert output.source_stats[1].compact_query_skipped_reason == (
        "adaptive_source_cooldown"
    )


def test_adaptive_skipped_event_has_no_fake_connector_completion(monkeypatch) -> None:
    papers = [_paper(index, "dense retrieval methods") for index in range(10)]
    _install_openalex(monkeypatch, [ConnectorSearchResult(papers=papers)])
    events: list[str] = []

    retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        query_adapter_policy="adaptive",
        connector_event_callback=lambda name, payload: events.append(name),
    )

    assert events == [
        "connector_started",
        "connector_completed",
        "adaptive_query_decision",
    ]


def test_adaptive_executed_event_order_and_provenance(monkeypatch) -> None:
    _install_openalex(monkeypatch, [ConnectorSearchResult()])
    events: list[tuple[str, dict[str, object]]] = []

    output = retrieve_papers(
        "Please find dense retrieval papers",
        sources=["openalex"],
        query_adapter_policy="adaptive",
        query_purpose="topic_expansion",
        connector_event_callback=lambda name, payload: events.append((name, payload)),
    )

    assert [name for name, _ in events] == [
        "connector_started",
        "connector_completed",
        "adaptive_query_decision",
        "connector_started",
        "connector_completed",
    ]
    compact = output.source_stats[1]
    assert compact.triggered_by
    assert compact.safe_original_candidate_count == 0
    assert compact.safe_original_core_term_coverage == 0.0
    assert compact.safe_original_constraint_coverage == 1.0
    assert compact.sufficiency_reasons
    assert compact.compact_query_executed is True
    assert compact.query_provenance[0].origin_subquery.startswith("Please find")
    assert compact.query_provenance[0].purpose == "topic_expansion"


def test_adaptive_same_input_is_deterministic(monkeypatch) -> None:
    papers = [_paper(index, "dense retrieval methods") for index in range(10)]
    _install_openalex(monkeypatch, [ConnectorSearchResult(papers=papers)])

    first = retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )
    second = retrieve_papers(
        "Please find dense retrieval methods",
        sources=["openalex"],
        query_adapter_policy="adaptive",
    )

    comparable = lambda output: [
        (
            item.compact_query_executed,
            item.compact_query_skipped_reason,
            item.sufficiency_reasons,
        )
        for item in output.source_stats
    ]
    assert comparable(first) == comparable(second)


def test_adaptive_decision_code_has_no_gold_or_benchmark_dependency() -> None:
    source = inspect.getsource(evaluate_retrieval_sufficiency).casefold()

    assert "gold" not in source
    assert "benchmark" not in source


def test_default_sources_are_domain_specific_and_semantic_scholar_is_explicit() -> None:
    request = SearchRunCreateRequest(query="dense retrieval")
    cs = analyze_query("dense retrieval methods")
    biomedical = analyze_query("clinical gene therapy trials")
    explicit_request = SearchRunCreateRequest(
        query="dense retrieval methods",
        source_preferences=["semantic_scholar", "arxiv"],
    )

    assert request.source_preferences is None
    assert cs.selected_sources == ["arxiv", "openalex"]
    assert biomedical.selected_sources == ["pubmed", "openalex"]
    assert explicit_request.source_preferences == ["semantic_scholar", "arxiv"]


def test_default_query_adapter_policy_is_adaptive() -> None:
    variants = adapt_queries_for_source(
        "Please find dense retrieval papers", "openalex"
    )

    assert [item.strategy for item in variants] == [
        "safe_original",
        "compact_core",
    ]
