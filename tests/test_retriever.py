from __future__ import annotations

import pytest

from scholar_agent.connectors import ConnectorDiagnostics, ConnectorSearchResult
from scholar_agent.agents.retriever import (
    RetrievalRunContext,
    clear_retrieval_cache,
    clear_source_cooldowns,
    retrieve_papers,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


@pytest.fixture(autouse=True)
def reset_retrieval_cache(monkeypatch: pytest.MonkeyPatch):
    clear_retrieval_cache()
    clear_source_cooldowns()
    monkeypatch.delenv("SCHOLAR_AGENT_RETRIEVAL_CACHE", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_RETRIEVAL_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_RETRIEVAL_CACHE_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_SOURCE_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS", raising=False)
    yield
    clear_retrieval_cache()
    clear_source_cooldowns()


def make_paper(
    title: str,
    *,
    doi: str | None = None,
    sources: list[str] | None = None,
    citation_count: int = 0,
    abstract: str = "",
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=2024,
        abstract=abstract,
        identifiers=PaperIdentifiers(doi=doi),
        sources=sources or [],
        citation_count=citation_count,
    )


def test_retrieve_papers_aggregates_and_deduplicates(monkeypatch) -> None:
    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        assert query == "llm reranking"
        assert limit == 5
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    "Shared Paper",
                    doi="10.123/shared",
                    sources=["openalex"],
                    citation_count=2,
                    abstract="short",
                ),
                make_paper("OpenAlex Only", sources=["openalex"], citation_count=1),
            ]
        )

    def fake_arxiv(query: str, limit: int) -> ConnectorSearchResult:
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    "Shared Paper Extended",
                    doi="https://doi.org/10.123/shared",
                    sources=["arxiv"],
                    citation_count=9,
                    abstract="This abstract is longer and should be retained.",
                ),
                make_paper("arXiv Only", sources=["arxiv"], citation_count=0),
            ]
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", fake_openalex)
    monkeypatch.setattr("scholar_agent.agents.retriever.search_arxiv_detailed", fake_arxiv)

    output = retrieve_papers(
        "llm reranking",
        limit_per_source=5,
        sources=["openalex", "arxiv"],
        query_adapter_policy="hybrid",
    )

    assert output.query == "llm reranking"
    assert output.requested_sources == ["openalex", "arxiv"]
    assert output.raw_count == 6
    assert output.deduplicated_count == 3
    assert len(output.papers) == 3
    assert output.papers[0].sources == ["openalex", "arxiv"]
    assert output.papers[0].citation_count == 9
    assert output.papers[0].abstract == "This abstract is longer and should be retained."
    assert [stat.source for stat in output.source_stats] == [
        "openalex",
        "arxiv",
        "arxiv",
    ]
    assert [stat.query for stat in output.source_stats] == [
        "llm reranking",
        "llm reranking",
        "llm reranking",
    ]
    assert [stat.returned_count for stat in output.source_stats] == [2, 2, 2]
    assert all(stat.error_message is None for stat in output.source_stats)
    assert all(stat.cache_hit is False for stat in output.source_stats)
    assert output.warnings == []
    assert output.latency_seconds >= 0


def test_connector_events_wrap_each_actual_connector_call(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        events.append(("call", "openalex"))
        return ConnectorSearchResult()

    def fake_arxiv(query: str, limit: int) -> ConnectorSearchResult:
        events.append(("call", "arxiv"))
        return ConnectorSearchResult()

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        fake_openalex,
    )
    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_arxiv_detailed",
        fake_arxiv,
    )

    retrieve_papers(
        "event order",
        sources=["openalex", "arxiv"],
        query_adapter_policy="hybrid",
        connector_event_callback=lambda name, payload: events.append(
            (name, str(payload["source"]))
        ),
    )

    assert events == [
        ("connector_started", "openalex"),
        ("call", "openalex"),
        ("connector_completed", "openalex"),
        ("connector_started", "arxiv"),
        ("call", "arxiv"),
        ("connector_completed", "arxiv"),
        ("connector_started", "arxiv"),
        ("call", "arxiv"),
        ("connector_completed", "arxiv"),
    ]


def test_retrieve_papers_single_source_error_keeps_other_results(monkeypatch) -> None:
    def failing_openalex(query: str, limit: int) -> ConnectorSearchResult:
        return ConnectorSearchResult(
            error_message="OpenAlex search failed: HTTP Error 503: Service Unavailable",
            warnings=["OpenAlex search failed: HTTP Error 503: Service Unavailable"],
        )

    def fake_arxiv(query: str, limit: int) -> ConnectorSearchResult:
        return ConnectorSearchResult(
            papers=[make_paper("arXiv Result", sources=["arxiv"])]
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", failing_openalex)
    monkeypatch.setattr("scholar_agent.agents.retriever.search_arxiv_detailed", fake_arxiv)

    output = retrieve_papers(
        "llm reranking",
        sources=["openalex", "arxiv"],
        query_adapter_policy="hybrid",
    )

    assert output.raw_count == 2
    assert output.deduplicated_count == 1
    assert output.papers[0].title == "arXiv Result"
    assert len(output.source_stats) == 3
    assert output.source_stats[0].source == "openalex"
    assert output.source_stats[0].query == "llm reranking"
    assert output.source_stats[0].returned_count == 0
    assert output.source_stats[0].cache_hit is False
    assert (
        output.source_stats[0].error_message
        == "OpenAlex search failed: HTTP Error 503: Service Unavailable"
    )
    assert output.source_stats[1].source == "arxiv"
    assert output.source_stats[1].query == "llm reranking"
    assert output.source_stats[1].returned_count == 1
    assert output.source_stats[1].cache_hit is False
    assert output.warnings == [
        "OpenAlex search failed: HTTP Error 503: Service Unavailable"
    ]


def test_retrieve_papers_supports_semantic_scholar_source(monkeypatch) -> None:
    def fake_semantic_scholar(query: str, limit: int) -> ConnectorSearchResult:
        assert query == "llm reranking"
        assert limit == 7
        return ConnectorSearchResult(
            papers=[
                Paper(
                    title="Semantic Scholar Result",
                    authors=["S2 Author"],
                    year=2025,
                    abstract="Semantic Scholar paper.",
                    identifiers=PaperIdentifiers(semantic_scholar_id="S2-1"),
                    sources=["semantic_scholar"],
                    citation_count=3,
                )
            ]
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        fake_semantic_scholar,
    )

    output = retrieve_papers(
        "llm reranking",
        limit_per_source=7,
        sources=["semantic_scholar"],
    )

    assert output.requested_sources == ["semantic_scholar"]
    assert output.raw_count == 1
    assert output.deduplicated_count == 1
    assert output.papers[0].identifiers.semantic_scholar_id == "S2-1"
    assert output.source_stats[0].source == "semantic_scholar"
    assert output.source_stats[0].query == "llm reranking"
    assert output.source_stats[0].returned_count == 1
    assert output.warnings == []


def test_retrieve_papers_supports_pubmed_source(monkeypatch) -> None:
    def fake_pubmed(query: str, limit: int) -> ConnectorSearchResult:
        assert query == "gene therapy"
        assert limit == 4
        return ConnectorSearchResult(
            papers=[
                Paper(
                    title="PubMed Result",
                    authors=["Clinical Author"],
                    year=2024,
                    abstract="Clinical retrieval paper.",
                    identifiers=PaperIdentifiers(pubmed_id="12345"),
                    sources=["pubmed"],
                )
            ]
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_pubmed_detailed",
        fake_pubmed,
    )

    output = retrieve_papers(
        "gene therapy",
        limit_per_source=4,
        sources=["pubmed"],
    )

    assert output.requested_sources == ["pubmed"]
    assert output.raw_count == 1
    assert output.deduplicated_count == 1
    assert output.papers[0].identifiers.pubmed_id == "12345"
    assert output.papers[0].sources == ["pubmed"]
    assert output.source_stats[0].source == "pubmed"
    assert output.source_stats[0].query == "gene therapy"
    assert output.source_stats[0].returned_count == 1
    assert output.warnings == []


def test_retrieve_papers_unknown_source_warning(monkeypatch) -> None:
    def fake_arxiv(query: str, limit: int) -> ConnectorSearchResult:
        return ConnectorSearchResult(
            papers=[make_paper("arXiv Result", sources=["arxiv"])]
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_arxiv_detailed", fake_arxiv)

    output = retrieve_papers("llm reranking", sources=["unknown", "arxiv"])

    assert output.requested_sources == ["unknown", "arxiv"]
    assert output.raw_count == 2
    assert output.deduplicated_count == 1
    assert output.source_stats[0].source == "unknown"
    assert output.source_stats[0].query == "llm reranking"
    assert output.source_stats[0].error_message == "unsupported_source:unknown"
    assert output.warnings == ["unsupported_source:unknown"]


def test_retrieve_papers_connector_warning_without_error_is_aggregated(monkeypatch) -> None:
    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        return ConnectorSearchResult(
            papers=[make_paper("OpenAlex Result", sources=["openalex"])],
            warnings=["OpenAlex parse warning"],
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", fake_openalex)

    output = retrieve_papers("llm reranking", sources=["openalex"])

    assert output.source_stats[0].source == "openalex"
    assert output.source_stats[0].returned_count == 1
    assert output.source_stats[0].error_message is None
    assert output.warnings == ["OpenAlex parse warning"]


def test_retrieve_papers_cache_hit_reuses_successful_connector_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"openalex": 0}

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls["openalex"] += 1
        return ConnectorSearchResult(
            papers=[make_paper("Cached OpenAlex Result", sources=["openalex"])],
            warnings=["OpenAlex parse warning"],
            diagnostics=ConnectorDiagnostics(request_count=1),
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", fake_openalex)

    first = retrieve_papers("llm reranking cache", limit_per_source=5, sources=["openalex"])
    second = retrieve_papers("llm reranking cache", limit_per_source=5, sources=["openalex"])

    assert calls["openalex"] == 1
    assert first.source_stats[0].cache_hit is False
    assert second.source_stats[0].cache_hit is True
    assert first.source_stats[0].diagnostics.request_count == 1
    assert first.source_stats[0].diagnostics.cache_hit_count == 0
    assert second.source_stats[0].diagnostics.request_count == 0
    assert second.source_stats[0].diagnostics.cache_hit_count == 1
    assert second.papers[0].title == "Cached OpenAlex Result"
    assert "OpenAlex parse warning" in second.warnings
    assert "retrieval_cache_hit:openalex" in second.warnings


def test_retrieve_papers_cache_key_includes_query_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls.append((query, limit))
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    f"Result {query} {limit}",
                    doi=f"10.123/{len(calls)}",
                    sources=["openalex"],
                )
            ]
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", fake_openalex)

    retrieve_papers("query one", limit_per_source=5, sources=["openalex"])
    retrieve_papers("query two", limit_per_source=5, sources=["openalex"])
    retrieve_papers("query one", limit_per_source=10, sources=["openalex"])
    cached = retrieve_papers("query one", limit_per_source=5, sources=["openalex"])

    assert calls == [("query one", 5), ("query two", 5), ("query one", 10)]
    assert cached.source_stats[0].cache_hit is True


def test_retrieve_papers_cache_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"openalex": 0}

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls["openalex"] += 1
        return ConnectorSearchResult(
            papers=[make_paper("No Cache Result", sources=["openalex"])]
        )

    monkeypatch.setenv("SCHOLAR_AGENT_RETRIEVAL_CACHE", "0")
    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", fake_openalex)

    first = retrieve_papers("llm reranking no cache", sources=["openalex"])
    second = retrieve_papers("llm reranking no cache", sources=["openalex"])

    assert calls["openalex"] == 2
    assert first.source_stats[0].cache_hit is False
    assert second.source_stats[0].cache_hit is False


def test_retrieve_papers_does_not_cache_connector_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"openalex": 0}

    def flaky_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls["openalex"] += 1
        if calls["openalex"] == 1:
            return ConnectorSearchResult(
                error_message="OpenAlex search failed: timeout",
                warnings=["OpenAlex search failed: timeout"],
            )
        return ConnectorSearchResult(
            papers=[make_paper("Recovered OpenAlex Result", sources=["openalex"])]
        )

    monkeypatch.setenv("SCHOLAR_AGENT_SOURCE_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr("scholar_agent.agents.retriever.search_openalex_detailed", flaky_openalex)

    first = retrieve_papers("llm reranking flaky", sources=["openalex"])
    second = retrieve_papers("llm reranking flaky", sources=["openalex"])

    assert calls["openalex"] == 2
    assert first.source_stats[0].cache_hit is False
    assert first.source_stats[0].error_message == "OpenAlex search failed: timeout"
    assert second.source_stats[0].cache_hit is False
    assert second.source_stats[0].error_message is None
    assert second.papers[0].title == "Recovered OpenAlex Result"


def test_retrieve_papers_cooldown_skips_source_after_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"semantic_scholar": 0}
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def rate_limited_then_recovered_semantic_scholar(
        query: str,
        limit: int,
    ) -> ConnectorSearchResult:
        calls["semantic_scholar"] += 1
        if calls["semantic_scholar"] > 1:
            return ConnectorSearchResult(
                papers=[make_paper("Recovered S2 Result", sources=["semantic_scholar"])]
            )
        return ConnectorSearchResult(
            error_message="Semantic Scholar search failed: HTTP Error 429: ",
            warnings=["Semantic Scholar search failed: HTTP Error 429:"],
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.time.monotonic", fake_monotonic)
    monkeypatch.setenv("SCHOLAR_AGENT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS", "2")
    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        rate_limited_then_recovered_semantic_scholar,
    )

    first = retrieve_papers("llm reranking cooldown", sources=["semantic_scholar"])
    second = retrieve_papers("llm reranking cooldown", sources=["semantic_scholar"])
    clock[0] = 3.0
    third = retrieve_papers("llm reranking cooldown", sources=["semantic_scholar"])

    assert calls["semantic_scholar"] == 2
    assert first.source_stats[0].error_message == (
        "Semantic Scholar search failed: HTTP Error 429: "
    )
    assert second.source_stats[0].source == "semantic_scholar"
    assert second.source_stats[0].returned_count == 0
    assert second.source_stats[0].error_message == (
        "source_cooldown_skip:semantic_scholar"
    )
    assert second.warnings == ["source_cooldown_skip:semantic_scholar"]
    assert third.source_stats[0].error_message is None
    assert third.papers[0].title == "Recovered S2 Result"


def test_retrieve_papers_recovered_429_warning_does_not_trigger_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"semantic_scholar": 0}

    def recovered_semantic_scholar(
        query: str,
        limit: int,
    ) -> ConnectorSearchResult:
        calls["semantic_scholar"] += 1
        return ConnectorSearchResult(
            papers=[
                make_paper(
                    f"Recovered S2 Result {calls['semantic_scholar']}",
                    sources=["semantic_scholar"],
                )
            ],
            warnings=[
                "Semantic Scholar search transient error on attempt 1/2: "
                "HTTP Error 429: ; retried"
            ],
        )

    monkeypatch.setenv("SCHOLAR_AGENT_RETRIEVAL_CACHE", "0")
    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        recovered_semantic_scholar,
    )

    first = retrieve_papers("llm reranking recovered", sources=["semantic_scholar"])
    second = retrieve_papers("llm reranking recovered", sources=["semantic_scholar"])

    assert calls["semantic_scholar"] == 2
    assert first.source_stats[0].error_message is None
    assert second.source_stats[0].error_message is None
    assert "HTTP Error 429" in first.warnings[0]
    assert not any("source_cooldown_skip" in warning for warning in second.warnings)
    assert second.papers[0].title == "Recovered S2 Result 2"


def test_retrieve_papers_does_not_global_cooldown_after_one_5xx_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"semantic_scholar": 0}

    def failed_semantic_scholar(
        query: str,
        limit: int,
    ) -> ConnectorSearchResult:
        calls["semantic_scholar"] += 1
        return ConnectorSearchResult(
            error_message="Semantic Scholar search failed: HTTP Error 503: ",
            warnings=["Semantic Scholar search failed: HTTP Error 503:"],
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        failed_semantic_scholar,
    )

    first = retrieve_papers("llm reranking 5xx", sources=["semantic_scholar"])
    second = retrieve_papers("llm reranking 5xx", sources=["semantic_scholar"])

    assert calls["semantic_scholar"] == 2
    assert first.source_stats[0].error_message == (
        "Semantic Scholar search failed: HTTP Error 503: "
    )
    assert second.source_stats[0].error_message == (
        "Semantic Scholar search failed: HTTP Error 503: "
    )


def test_retrieve_papers_does_not_global_cooldown_after_one_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"arxiv": 0}

    def timeout_then_skipped(query: str, limit: int) -> ConnectorSearchResult:
        calls["arxiv"] += 1
        return ConnectorSearchResult(
            error_message="arXiv search failed: request timed out",
            warnings=["arXiv search failed: request timed out"],
        )

    monkeypatch.setattr("scholar_agent.agents.retriever.search_arxiv_detailed", timeout_then_skipped)

    first = retrieve_papers(
        "llm reranking timeout",
        sources=["arxiv"],
        query_adapter_policy="hybrid",
    )
    second = retrieve_papers(
        "llm reranking timeout",
        sources=["arxiv"],
        query_adapter_policy="hybrid",
    )

    assert calls["arxiv"] == 4
    assert first.source_stats[0].error_message == "arXiv search failed: request timed out"
    assert second.source_stats[0].error_message == "arXiv search failed: request timed out"


def test_run_circuit_opens_after_two_consecutive_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    context = RetrievalRunContext()

    def timeout_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult(
            error_message="OpenAlex search failed: request timed out",
            warnings=["OpenAlex search failed: request timed out"],
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        timeout_openalex,
    )

    retrieve_papers("first graph query", sources=["openalex"], run_context=context)
    retrieve_papers("second graph query", sources=["openalex"], run_context=context)
    third = retrieve_papers(
        "third graph query",
        sources=["openalex"],
        run_context=context,
    )

    assert calls == ["first graph query", "second graph query"]
    assert third.source_stats[0].source_skipped_reason == (
        "run_transient_failure_circuit_open"
    )
    assert third.source_stats[0].diagnostics.request_count == 0


def test_retrieve_papers_source_cooldown_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"semantic_scholar": 0}

    def flaky_semantic_scholar(query: str, limit: int) -> ConnectorSearchResult:
        calls["semantic_scholar"] += 1
        if calls["semantic_scholar"] == 1:
            return ConnectorSearchResult(
                error_message="Semantic Scholar search failed: HTTP Error 429: ",
                warnings=["Semantic Scholar search failed: HTTP Error 429:"],
            )
        return ConnectorSearchResult(
            papers=[make_paper("Recovered S2 Result", sources=["semantic_scholar"])]
        )

    monkeypatch.setenv("SCHOLAR_AGENT_SEMANTIC_SCHOLAR_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        flaky_semantic_scholar,
    )

    first = retrieve_papers("llm reranking no cooldown", sources=["semantic_scholar"])
    second = retrieve_papers("llm reranking no cooldown", sources=["semantic_scholar"])

    assert calls["semantic_scholar"] == 2
    assert first.source_stats[0].error_message == (
        "Semantic Scholar search failed: HTTP Error 429: "
    )
    assert second.source_stats[0].error_message is None
    assert second.papers[0].title == "Recovered S2 Result"


def test_same_adapted_query_is_called_once_per_run(monkeypatch) -> None:
    calls: list[str] = []
    context = RetrievalRunContext()

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult(
            papers=[make_paper("Shared", sources=["openalex"])]
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        fake_openalex,
    )

    first = retrieve_papers(
        "Could you list papers about graph retrieval?",
        sources=["openalex"],
        run_context=context,
        query_adapter_policy="hybrid",
    )
    second = retrieve_papers(
        "graph retrieval",
        sources=["openalex"],
        run_context=context,
        query_adapter_policy="hybrid",
    )

    assert calls == [
        "Could you list papers about graph retrieval",
        "graph retrieval",
    ]
    assert first.papers[0].title == "Shared"
    assert second.papers == []
    assert second.source_stats[0].run_dedupe_hit is True
    assert second.source_stats[0].source_skipped_reason == "duplicate_adapted_query"
    assert second.source_stats[0].diagnostic_papers[0].title == "Shared"
    assert {
        (
            item.origin_subquery,
            item.adaptation_strategy,
        )
        for item in second.source_stats[0].query_provenance
    } == {
        (
            "Could you list papers about graph retrieval?",
            "compact_core",
        ),
        ("graph retrieval", "safe_original"),
        ("graph retrieval", "compact_core"),
    }


def test_run_dedupe_preserves_word_order_and_arxiv_field_syntax(monkeypatch) -> None:
    calls: list[str] = []
    context = RetrievalRunContext()

    def fake_arxiv(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult()

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_arxiv_detailed",
        fake_arxiv,
    )

    retrieve_papers("graph retrieval", sources=["arxiv"], run_context=context)
    retrieve_papers("retrieval graph", sources=["arxiv"], run_context=context)

    assert len(calls) == 4
    assert "all:graph retrieval" in calls
    assert "all:retrieval graph" in calls
    assert any(query.startswith("((ti:graph") for query in calls)
    assert any(query.startswith("((ti:retrieval") for query in calls)


def test_hybrid_retains_safe_original_candidate_when_compact_returns_none(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_openalex(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        if query.startswith("Could you list"):
            return ConnectorSearchResult(
                papers=[make_paper("Safe Original Target", sources=["openalex"])]
            )
        return ConnectorSearchResult()

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        fake_openalex,
    )

    output = retrieve_papers(
        "Could you list papers about rare graph retrieval?",
        sources=["openalex"],
        query_adapter_policy="hybrid",
    )

    assert len(calls) == 2
    assert [paper.title for paper in output.papers] == ["Safe Original Target"]
    assert [item.adaptation_strategy for item in output.source_stats] == [
        "safe_original",
        "compact_core",
    ]


def test_query_adapter_policy_does_not_change_source_preferences(monkeypatch) -> None:
    calls: list[str] = []

    def fake_search(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult()

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_pubmed_detailed",
        fake_search,
    )
    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_openalex_detailed",
        fake_search,
    )

    output = retrieve_papers(
        "gene retrieval",
        sources=["pubmed", "openalex"],
        query_adapter_policy="safe_original",
    )

    assert output.requested_sources == ["pubmed", "openalex"]
    assert calls == ["gene retrieval", "gene retrieval"]


def test_semantic_scholar_final_429_skips_later_run_queries(monkeypatch) -> None:
    calls: list[str] = []
    context = RetrievalRunContext()

    def rate_limited(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult(
            error_message="Semantic Scholar search failed: HTTP Error 429: ",
            warnings=["Semantic Scholar search failed: HTTP Error 429:"],
            diagnostics=ConnectorDiagnostics(
                request_count=2,
                retry_count=1,
                error_count=1,
                retry_after_seconds=120,
            ),
        )

    monkeypatch.setattr(
        "scholar_agent.agents.retriever.search_semantic_scholar_detailed",
        rate_limited,
    )

    first = retrieve_papers(
        "first semantic query",
        sources=["semantic_scholar"],
        run_context=context,
        remaining_subquery_count=2,
    )
    second = retrieve_papers(
        "different semantic query",
        sources=["semantic_scholar"],
        run_context=context,
        remaining_subquery_count=1,
    )

    assert len(calls) == 1
    assert first.source_stats[0].diagnostics.retry_after_seconds == 120
    assert second.source_stats[0].source_skipped_reason == "run_rate_limit_cooldown"
    assert second.source_stats[0].remaining_subquery_count == 1
    assert second.source_stats[0].diagnostics.request_count == 0
