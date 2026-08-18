from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.app.api import routes
from scholar_agent.app.main import app
from scholar_agent.core.api_schemas import SearchRunCreateRequest
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    QueryConstraint,
    TimeRange,
)
from scholar_agent.services.search_service import SearchService


client = TestClient(app)


def test_explicit_constraints_override_rule_parsing_and_build_subquery() -> None:
    explicit = QueryConstraint(
        time_range=TimeRange(start_year=2023, end_year=2025),
        venues=["SIGIR"],
        datasets=["LitSearch"],
        must_include_terms=["causal retrieval"],
        exclude_terms=["computer vision"],
        paper_types=["Systematic Review"],
    )

    plan = analyze_query(
        "ACL benchmark dataset papers from 2020 to 2022",
        current_year=2026,
        explicit_constraints=explicit,
    )

    constraints = plan.query_analysis.constraints
    assert constraints.time_range == explicit.time_range
    assert constraints.venues == ["SIGIR"]
    assert constraints.datasets == ["LitSearch"]
    assert constraints.must_include_terms == ["causal retrieval"]
    assert constraints.exclude_terms == ["computer vision"]
    assert constraints.paper_types == ["review"]
    assert set(constraints.explicit_fields) == {
        "time_range",
        "venues",
        "datasets",
        "must_include_terms",
        "exclude_terms",
        "paper_types",
    }
    constraint_query = plan.subqueries[1]
    assert constraint_query.purpose == "constraint_expansion"
    assert "LitSearch" in constraint_query.query
    assert "review" in constraint_query.query
    assert "SIGIR" in constraint_query.query
    assert "2023-2025" in constraint_query.query


def test_explicit_constraints_override_llm_constraints() -> None:
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "paper_finding",
            "domain": "machine_learning",
            "constraints": {
                "time_range": {"start_year": 2018, "end_year": 2020},
                "venues": ["ACL"],
                "datasets": ["MS MARCO"],
                "must_include_terms": ["dense retrieval"],
                "excluded_terms": ["survey"],
                "paper_types": ["method"],
            },
            "selected_sources": ["arxiv"],
            "subqueries": ["LLM generated query"],
        }
    )
    explicit = QueryConstraint(
        time_range=TimeRange(start_year=2022, end_year=2026),
        venues=["SIGIR"],
        datasets=["LitSearch"],
        must_include_terms=["reranking"],
        exclude_terms=["vision"],
        paper_types=["review"],
    )

    plan = analyze_query(
        "find retrieval papers",
        use_llm=True,
        llm_client=client,
        explicit_constraints=explicit,
        current_year=2026,
    )

    constraints = plan.query_analysis.constraints
    assert constraints.time_range == explicit.time_range
    assert constraints.venues == ["SIGIR"]
    assert constraints.datasets == ["LitSearch"]
    assert constraints.must_include_terms == ["reranking"]
    assert constraints.exclude_terms == ["vision"]
    assert constraints.paper_types == ["review"]
    assert plan.subqueries[0].query == "LLM generated query"
    assert plan.subqueries[1].purpose == "constraint_expansion"


def test_no_explicit_constraints_preserves_query_understanding_behavior() -> None:
    first = analyze_query(
        "latest LLM reranking methods for scientific literature retrieval",
        current_year=2026,
    )
    second = analyze_query(
        "latest LLM reranking methods for scientific literature retrieval",
        current_year=2026,
        explicit_constraints=None,
    )

    assert first == second
    assert first.query_analysis.constraints.explicit_fields == []


def test_must_have_excluded_dataset_and_paper_type_constraints_are_executed() -> None:
    plan = analyze_query(
        "find LLM retrieval papers",
        current_year=2026,
        explicit_constraints=QueryConstraint(
            must_include_terms=["causal"],
            exclude_terms=["computer vision"],
            datasets=["LitSearch"],
            paper_types=["review"],
        ),
    )
    matching = _paper(
        "Causal LLM Retrieval Review",
        abstract="A review evaluated retrieval on the LitSearch dataset.",
    )
    missing_required = _paper(
        "LLM Retrieval Review",
        abstract="A review evaluated retrieval on the LitSearch dataset.",
    )
    excluded = _paper(
        "Causal Computer Vision Retrieval Review",
        abstract="A review evaluated retrieval on the LitSearch dataset.",
    )
    wrong_dataset_and_type = _paper(
        "Causal Retrieval Method",
        abstract="A method evaluated retrieval on another corpus.",
    )

    matching_result, missing_result, excluded_result, wrong_result = judge_papers(
        plan.query_analysis,
        [matching, missing_required, excluded, wrong_dataset_and_type],
    )

    assert "causal" in {item.casefold() for item in matching_result.matched_terms}
    assert "LitSearch" in matching_result.matched_terms
    assert "review" in matching_result.matched_terms
    assert missing_result.category != "highly_relevant"
    assert "missing_must_have_terms:causal" in missing_result.warnings
    assert excluded_result.category == "irrelevant"
    assert excluded_result.score == 0.0
    assert "excluded_terms_matched:computer vision" in excluded_result.warnings
    assert "dataset_terms_not_matched:LitSearch" in wrong_result.warnings
    assert "paper_types_not_matched:review" in wrong_result.warnings
    assert matching_result.score > wrong_result.score


def test_time_and_venue_constraints_affect_judgement_and_reranking() -> None:
    plan = analyze_query(
        "LLM retrieval",
        current_year=2026,
        explicit_constraints=QueryConstraint(
            time_range=TimeRange(start_year=2022, end_year=2025),
            venues=["SI GIR"],
        ),
    )
    matching = _paper("LLM Retrieval", year=2024, venue="SIGIR")
    outside = _paper("LLM Retrieval", year=2020, venue="KDD")

    matching_result, outside_result = judge_papers(
        plan.query_analysis,
        [matching, outside],
    )
    ranked = rerank_papers(plan.query_analysis, [matching_result, outside_result])

    assert any(item.source == "venue" for item in matching_result.evidence)
    assert outside_result.category != "highly_relevant"
    assert "outside_time_range:2020" in outside_result.warnings
    outside_ranked = next(item for item in ranked if item.paper.year == 2020)
    matching_ranked = next(item for item in ranked if item.paper.year == 2024)
    assert outside_ranked.score_breakdown.timeliness_score == 0.0
    assert (
        matching_ranked.score_breakdown.authority_score
        > outside_ranked.score_breakdown.authority_score
    )


def test_search_service_direct_call_is_compatible_and_deterministic() -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source
        calls.append((query, sources))
        return _retrieval_output(query, sources)

    service = SearchService(retriever=fake_retriever)
    explicit = QueryConstraint(
        datasets=["LitSearch"],
        paper_types=["review"],
    )
    first = service.run_search(
        "LLM retrieval",
        top_k=5,
        current_year=2026,
        sources_override=["arxiv", "arxiv"],
        explicit_constraints=explicit,
    )
    first_calls = list(calls)
    calls.clear()
    second = service.run_search(
        "LLM retrieval",
        top_k=5,
        current_year=2026,
        sources_override=["arxiv", "arxiv"],
        explicit_constraints=explicit,
    )

    assert first.search_plan == second.search_plan
    assert first.judgements == second.judgements
    assert first.ranked_papers == second.ranked_papers
    assert sorted(first_calls) == sorted(calls)
    assert first.search_plan.selected_sources == ["arxiv"]

    legacy = SearchService(retriever=fake_retriever).run_search(
        "plain query",
        top_k=1,
        current_year=2026,
    )
    assert legacy.search_plan.query_analysis.original_query == "plain query"


@pytest.mark.parametrize("sources", [[], ["arxiv", "unknown"]])
def test_search_service_rejects_invalid_source_overrides(sources: list[str]) -> None:
    with pytest.raises(ValueError):
        SearchService(retriever=lambda *_args, **_kwargs: None).run_search(  # type: ignore[arg-type]
            "query",
            sources_override=sources,
        )


@pytest.mark.parametrize("sources", [[], ["arxiv", "unknown"]])
def test_api_rejects_empty_or_unsupported_source_preferences(
    sources: list[str],
) -> None:
    response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "retrieval", "source_preferences": sources},
    )

    assert response.status_code == 422


def test_api_rejects_unknown_paper_type() -> None:
    response = client.post(
        "/api/v1/real/search/runs",
        json={
            "query": "retrieval",
            "constraints": {"paper_types": ["position_paper"]},
        },
    )

    assert response.status_code == 422


def test_request_schema_normalizes_paper_types_and_sources() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "retrieval",
            "constraints": {
                "paper_types": ["Systematic Review", "METHODS", "review"],
            },
            "source_preferences": [
                "Semantic Scholar",
                "arxiv",
                "semantic_scholar",
            ],
        }
    )

    assert request.constraints.paper_types == ["review", "method"]
    assert request.source_preferences == ["semantic_scholar", "arxiv"]

    with pytest.raises(ValidationError):
        SearchRunCreateRequest.model_validate(
            {
                "query": "retrieval",
                "constraints": {"paper_types": ["unknown"]},
            }
        )


def test_real_api_result_contains_final_merged_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackedSearchService(SearchService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                retriever=lambda query, limit_per_source=20, sources=None: (
                    _retrieval_output(query, sources)
                ),
                max_workers=kwargs.get("max_workers", 1),
            )

    monkeypatch.setattr(routes, "SearchService", FakeBackedSearchService)
    response = client.post(
        "/api/v1/real/search/runs",
        json={
            "query": "find retrieval papers",
            "top_k": 5,
            "constraints": {
                "time_range": {"start_year": 2022, "end_year": 2025},
                "venues": ["SIGIR"],
                "must_have_terms": ["causal"],
                "excluded_terms": ["vision"],
                "datasets": ["LitSearch"],
                "paper_types": ["Systematic Review"],
            },
            "source_preferences": ["arxiv"],
            "options": {
                "enable_query_evolution": False,
                "enable_refchain": False,
                "enable_llm_query_understanding": False,
                "enable_llm_judgement": False,
            },
        },
    )
    assert response.status_code == 201
    run_id = response.json()["run_id"]
    _wait_for_terminal_run(run_id)

    result_response = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    constraints = result["query_analysis"]["constraints"]
    assert constraints["time_range"] == {
        "start_year": 2022,
        "end_year": 2025,
        "label": "explicit",
    }
    assert constraints["venues"] == ["SIGIR"]
    assert constraints["datasets"] == ["LitSearch"]
    assert constraints["must_have_terms"] == ["causal"]
    assert constraints["excluded_terms"] == ["vision"]
    assert constraints["paper_types"] == ["review"]
    assert result["search_plan"]["source_preferences"] == ["arxiv"]


class FakeLLMClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, object]:
        del messages, temperature, timeout
        return self.response


def _paper(
    title: str,
    *,
    abstract: str = "A paper about LLM retrieval.",
    year: int = 2024,
    venue: str = "SIGIR",
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(
            doi=f"10.123/{title.casefold().replace(' ', '-')}"
        ),
        sources=["arxiv"],
        citation_count=10,
    )


def _retrieval_output(
    query: str,
    sources: list[str] | None,
) -> RetrievalOutput:
    paper = _paper(
        "Causal LLM Retrieval Review",
        abstract="A review evaluates retrieval on the LitSearch dataset.",
    )
    return RetrievalOutput(
        query=query,
        requested_sources=list(sources or []),
        raw_count=1,
        deduplicated_count=1,
        papers=[paper],
        source_stats=[
            SourceStats(
                source=(sources or ["fixture"])[0],
                query=query,
                returned_count=1,
                latency_seconds=0.0,
            )
        ],
        latency_seconds=0.0,
    )


def _wait_for_terminal_run(run_id: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/real/search/runs/{run_id}")
        assert response.status_code == 200
        if response.json()["status"] in {"succeeded", "failed", "cancelled"}:
            assert response.json()["status"] == "succeeded"
            return
        time.sleep(0.01)
    raise AssertionError(f"run did not finish: {run_id}")
