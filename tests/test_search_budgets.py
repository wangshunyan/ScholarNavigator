from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.app.main import app
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvolvedSubquery,
    QueryEvolutionRecord,
    SearchBudget,
)
from scholar_agent.services.api_mapper import map_search_service_output_to_api_result
from scholar_agent.services.search_service import SearchService


client = TestClient(app)


def test_default_budget_is_used_by_direct_calls() -> None:
    output = SearchService(retriever=_one_paper_retriever).run_search(
        "LLM retrieval",
        enable_synthesis=False,
    )

    defaults = SearchBudget()
    status = output.budget_status
    assert status.max_search_rounds == defaults.max_search_rounds
    assert status.max_candidate_papers == defaults.max_candidate_papers
    assert status.max_llm_calls == defaults.max_llm_calls
    assert status.max_total_tokens == defaults.max_total_tokens
    assert status.max_latency_seconds == defaults.max_latency_seconds
    assert status.completed_search_rounds == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_search_rounds", 0),
        ("max_candidate_papers", 0),
        ("max_llm_calls", -1),
        ("max_total_tokens", -1),
        ("max_latency_seconds", 0),
    ],
)
def test_invalid_api_budgets_return_422(field: str, value: int) -> None:
    response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM retrieval", "budgets": {field: value}},
    )

    assert response.status_code == 422
    assert field in response.text


def test_one_round_budget_skips_query_evolution() -> None:
    calls: list[str] = []

    def retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        calls.append(query)
        return _retrieval(query, [_paper("Initial", "1")])

    output = SearchService(retriever=retriever).run_search(
        "LLM retrieval",
        enable_query_evolution=True,
        enable_synthesis=False,
        budget=SearchBudget(max_search_rounds=1),
    )

    assert output.budget_status.completed_search_rounds == 1
    assert output.budget_status.stop_reasons == ["budget_stop:max_search_rounds"]
    assert output.query_evolution_records[0].skipped_reasons == [
        "budget_stop:max_search_rounds"
    ]
    assert len(calls) == len(output.retrieval_outputs)


def test_many_initial_subqueries_are_one_logical_round() -> None:
    output = SearchService(retriever=_one_paper_retriever).run_search(
        "latest LLM reranking retrieval methods survey",
        run_profile="high_recall",
        enable_synthesis=False,
    )

    assert len(output.retrieval_outputs) > 1
    assert output.budget_status.completed_search_rounds == 1


def test_refchain_does_not_increment_search_rounds() -> None:
    def fetcher(paper: Paper, limit: int) -> list[Paper]:
        return [_paper("Reference LLM retrieval", "ref", openalex_id="WREF")]

    output = SearchService(
        retriever=_one_paper_retriever,
        reference_fetcher=fetcher,
    ).run_search(
        "LLM retrieval",
        enable_refchain=True,
        enable_synthesis=False,
    )

    assert output.refchain_output is not None
    assert output.budget_status.completed_search_rounds == 1


def test_candidate_truncation_is_source_covering_and_stable() -> None:
    papers = [
        _paper("OpenAlex A", "oa-a", sources=["openalex"]),
        _paper("OpenAlex B", "oa-b", sources=["openalex"]),
        _paper("arXiv A", "ax-a", sources=["arxiv"]),
        _paper("Semantic A", "ss-a", sources=["semantic_scholar"]),
        _paper("PubMed A", "pm-a", sources=["pubmed"]),
    ]

    def retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        return _retrieval(query, papers)

    service = SearchService(retriever=retriever)
    kwargs = {
        "query": "LLM retrieval",
        "enable_synthesis": False,
        "budget": SearchBudget(max_candidate_papers=3),
    }
    first = service.run_search(**kwargs)
    second = service.run_search(**kwargs)

    first_titles = [result.paper.title for result in first.judgements]
    second_titles = [result.paper.title for result in second.judgements]
    assert first_titles == ["arXiv A", "OpenAlex A", "Semantic A"]
    assert second_titles == first_titles
    truncation = first.budget_status.candidate_truncations[0]
    assert (truncation.before_count, truncation.after_count, truncation.truncated_count) == (
        5,
        3,
        2,
    )
    assert truncation.stage == "initial_retrieval"
    assert "budget_stop:max_candidate_papers" in first.warnings


def test_candidate_budget_is_reapplied_after_query_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolved_query = "evolved LLM retrieval"

    def fake_evolve(*args, **kwargs):  # noqa: ANN002, ANN003
        return QueryEvolutionRecord(
            generated_queries=[
                EvolvedSubquery(
                    query=evolved_query,
                    purpose="budget_test",
                    source_hints=["openalex"],
                )
            ]
        )

    monkeypatch.setattr("scholar_agent.services.search_service.evolve_queries", fake_evolve)

    def retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        if query == evolved_query:
            return _retrieval(
                query,
                [_paper(f"Evolved {index}", f"e{index}") for index in range(4)],
            )
        return _retrieval(query, [_paper("Initial", "initial")])

    output = SearchService(retriever=retriever).run_search(
        "LLM retrieval",
        enable_query_evolution=True,
        enable_synthesis=False,
        budget=SearchBudget(max_candidate_papers=3),
    )

    assert output.budget_status.completed_search_rounds == 2
    assert len(output.judgements) == 3
    assert output.budget_status.candidate_truncations[-1].stage == "query_evolution"


def test_refchain_obeys_remaining_candidate_capacity() -> None:
    limits: list[int] = []

    def fetcher(paper: Paper, limit: int) -> list[Paper]:
        limits.append(limit)
        return [
            _paper("Reference One", "ref-1", openalex_id="WREF1"),
            _paper("Reference Two", "ref-2", openalex_id="WREF2"),
        ]

    output = SearchService(
        retriever=_one_paper_retriever,
        reference_fetcher=fetcher,
    ).run_search(
        "LLM retrieval",
        enable_refchain=True,
        enable_synthesis=False,
        budget=SearchBudget(max_candidate_papers=2),
    )

    assert limits == [1]
    assert output.refchain_output is not None
    assert len(output.refchain_output.references) == 1
    assert len(output.judgements) == 2


def test_refchain_stops_before_seed_when_candidate_budget_is_full() -> None:
    calls = 0

    def fetcher(paper: Paper, limit: int) -> list[Paper]:
        nonlocal calls
        calls += 1
        return []

    output = SearchService(
        retriever=_one_paper_retriever,
        reference_fetcher=fetcher,
    ).run_search(
        "LLM retrieval",
        enable_refchain=True,
        enable_synthesis=False,
        budget=SearchBudget(max_candidate_papers=1),
    )

    assert calls == 0
    assert output.refchain_output is not None
    assert "budget_stop:max_candidate_papers" in (
        output.refchain_output.record.skipped_reasons
    )


def test_query_understanding_and_judgement_share_llm_call_budget() -> None:
    llm = SequencedLLMClient(tokens_per_call=5)
    output = SearchService(
        retriever=_one_paper_retriever,
        llm_client=llm,
    ).run_search(
        "LLM retrieval",
        enable_llm_query_understanding=True,
        enable_llm_judgement=True,
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=1),
    )

    assert llm.calls == 1
    assert output.budget_status.used_llm_calls == 1
    assert "budget_stop:max_llm_calls" in output.warnings
    assert output.judgements
    assert output.judgements[0].score != pytest.approx(0.99)


def test_judgement_batches_never_exceed_global_llm_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_BATCH_SIZE", "1")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_MAX_PAPERS", "4")
    llm = SequencedLLMClient(tokens_per_call=3)

    def retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        return _retrieval(
            query,
            [_paper(f"Paper {index}", str(index)) for index in range(4)],
        )

    output = SearchService(retriever=retriever, llm_client=llm).run_search(
        "LLM retrieval",
        enable_llm_judgement=True,
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=2),
    )

    assert llm.calls == 2
    assert output.budget_status.used_llm_calls == 2
    assert len(output.judgements) == 4
    assert "budget_stop:max_llm_calls" in output.warnings


def test_token_budget_stops_subsequent_llm_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_BATCH_SIZE", "1")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_JUDGEMENT_MAX_PAPERS", "3")
    llm = SequencedLLMClient(tokens_per_call=10)

    def retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        return _retrieval(
            query,
            [_paper(f"Paper {index}", str(index)) for index in range(3)],
        )

    output = SearchService(retriever=retriever, llm_client=llm).run_search(
        "LLM retrieval",
        enable_llm_judgement=True,
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=5, max_total_tokens=10),
    )

    assert llm.calls == 1
    assert output.budget_status.used_total_tokens == 10
    assert output.budget_status.used_llm_calls == 1
    assert "budget_stop:max_total_tokens" in output.warnings


def test_missing_provider_usage_returns_explicit_diagnostic() -> None:
    llm = NoUsageLLMClient()
    output = SearchService(
        retriever=_one_paper_retriever,
        llm_client=llm,
    ).run_search(
        "LLM retrieval",
        enable_llm_judgement=True,
        enable_synthesis=False,
    )

    assert llm.calls == 1
    assert output.budget_status.used_total_tokens == 0
    assert output.budget_status.token_usage_precise is False
    assert output.budget_status.diagnostics == [
        "budget_diagnostic:llm_usage_unavailable"
    ]


def test_latency_budget_stops_future_stages_and_keeps_results() -> None:
    reference_calls = 0

    def slow_retriever(query: str, limit_per_source=20, sources=None):  # noqa: ANN001
        time.sleep(0.02)
        return _retrieval(query, [_paper("Slow result", "slow")])

    def fetcher(paper: Paper, limit: int) -> list[Paper]:
        nonlocal reference_calls
        reference_calls += 1
        return []

    output = SearchService(
        retriever=slow_retriever,
        reference_fetcher=fetcher,
        max_workers=1,
    ).run_search(
        "LLM retrieval",
        enable_query_evolution=True,
        enable_refchain=True,
        enable_synthesis=True,
        budget=SearchBudget(max_latency_seconds=0.005),
    )

    assert "budget_stop:max_latency_seconds" in output.warnings
    assert output.ranked_papers
    assert output.synthesis_output is None
    assert output.budget_status.completed_search_rounds == 1
    assert reference_calls == 0
    assert output.query_evolution_records[0].skipped_reasons == [
        "budget_stop:max_latency_seconds"
    ]


def test_api_mapping_reports_budget_stop_as_successful_partial_result() -> None:
    output = SearchService(retriever=_one_paper_retriever).run_search(
        "LLM retrieval",
        enable_query_evolution=True,
        enable_synthesis=False,
        budget=SearchBudget(max_search_rounds=1),
    )
    result = map_search_service_output_to_api_result("run_budget", output)

    assert result.status == "succeeded"
    assert result.partial is True
    assert result.budget_status.stop_reasons == ["budget_stop:max_search_rounds"]
    assert result.cost_report.search_rounds == 1
    assert result.search_plan.max_rounds == 1
    assert "budget_stop:max_search_rounds" in result.warnings
    assert all("budget_" not in item for item in result.missing_evidence)


def _one_paper_retriever(
    query: str,
    limit_per_source: int = 20,
    sources: list[str] | None = None,
) -> RetrievalOutput:
    return _retrieval(
        query,
        [_paper("LLM retrieval method", "seed", openalex_id="WSEED")],
    )


def _retrieval(query: str, papers: list[Paper]) -> RetrievalOutput:
    return RetrievalOutput(
        query=query,
        requested_sources=["openalex", "arxiv"],
        raw_count=len(papers),
        deduplicated_count=len(papers),
        papers=papers,
        source_stats=[
            SourceStats(source="openalex", query=query, returned_count=len(papers))
        ],
    )


def _paper(
    title: str,
    identifier: str,
    *,
    sources: list[str] | None = None,
    openalex_id: str | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["A"],
        year=2025,
        venue="ACL",
        abstract="LLM retrieval method for scientific literature search.",
        identifiers=PaperIdentifiers(
            doi=f"10.123/{identifier}",
            openalex_id=openalex_id,
        ),
        sources=sources or ["openalex"],
    )


class SequencedLLMClient:
    def __init__(self, *, tokens_per_call: int) -> None:
        self.calls = 0
        self.tokens_per_call = tokens_per_call
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls += 1
        self.token_usage.prompt_tokens += self.tokens_per_call
        self.token_usage.total_tokens += self.tokens_per_call
        if self.calls == 1 and any(
            "query understanding" in message["content"].casefold()
            for message in messages
        ):
            return {
                "language": "en",
                "intent": "general",
                "domain": "machine_learning",
                "selected_sources": ["openalex"],
                "subqueries": [
                    {
                        "query": "LLM retrieval",
                        "source_hints": ["openalex"],
                        "purpose": "budget_test",
                    }
                ],
            }
        return {
            "judgements": [
                {
                    "paper_index": 0,
                    "score": 0.99,
                    "category": "highly_relevant",
                    "reasoning": "LLM judgement",
                    "evidence": [],
                    "matched_terms": ["LLM", "retrieval"],
                    "warnings": [],
                }
            ]
        }


class NoUsageLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls += 1
        return {
            "judgements": [
                {
                    "paper_index": 0,
                    "score": 0.9,
                    "category": "highly_relevant",
                    "reasoning": "LLM judgement without usage",
                    "evidence": [],
                    "matched_terms": ["LLM"],
                    "warnings": [],
                }
            ]
        }
