from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scholar_agent.agents.llm_feedback_evolution import evolve_with_llm_feedback
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.connectors import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    RankedPaper,
    RerankScoreBreakdown,
    SearchBudget,
    SearchPlan,
    SearchSubquery,
)
from scholar_agent.services.search_service import SearchService


QUERY = "graph neural networks for molecular property prediction"
FEEDBACK_QUERY = "graph neural networks molecular property prediction equivariant"


class FakeLLMClient:
    provider = "test_provider"
    model = "feedback-test-v1"

    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else _response(FEEDBACK_QUERY)
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        self.last_call_diagnostics = None

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls.append(
            {"messages": messages, "temperature": temperature, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        self.token_usage.prompt_tokens += 11
        self.token_usage.completion_tokens += 7
        self.token_usage.total_tokens += 18
        self.last_call_diagnostics = SimpleNamespace(
            http_attempts=1,
            http_429_count=0,
            retry_after_seconds=(),
            retry_wait_seconds=0.0,
            failure_class=None,
            cache_hit=False,
        )
        return deepcopy(self.response)


def _response(query: str) -> dict[str, object]:
    return {
        "intent_summary": "molecular graph learning",
        "facets": [
            {
                "facet_type": "method",
                "original_terms": ["message passing"],
                "normalized_terms": ["message passing"],
                "confidence": 0.9,
            }
        ],
        "supplemental_queries": [
            {
                "query": query,
                "purpose": "method coverage gap",
                "covered_facets": ["method"],
                "retained_must_have_terms": ["molecular"],
                "terminology_expansions": [],
            }
        ],
        "warnings": [],
    }


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query=QUERY,
        language="en",
        intent="general",
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["message passing"],
            must_include_terms=["molecular"],
            explicit_fields=["methods", "must_include_terms"],
        ),
    )


def _plan() -> SearchPlan:
    analysis = _analysis()
    return SearchPlan(
        query_analysis=analysis,
        subqueries=[
            SearchSubquery(
                query=QUERY,
                source_hints=["arxiv"],
                purpose="original_query",
            )
        ],
        selected_sources=["arxiv"],
        enable_query_evolution=True,
        query_evolution_policy="llm_feedback",
    )


def _paper(
    title: str = "Graph Neural Networks for Molecular Prediction",
    *,
    doi: str = "10.1000/feedback",
) -> Paper:
    return Paper(
        title=title,
        authors=["A. Researcher"],
        year=2025,
        venue="NeurIPS",
        abstract="Graph neural networks for molecular property prediction.",
        identifiers=PaperIdentifiers(doi=doi),
        sources=["arxiv"],
    )


def _judgement(paper: Paper | None = None) -> JudgementResult:
    return JudgementResult(
        paper=paper or _paper(),
        score=0.8,
        category="highly_relevant",
        reasoning="rule judgement",
        evidence=[EvidenceItem(source="title", text="graph neural networks", confidence=0.9)],
        matched_terms=["graph", "molecular"],
    )


def _ranked(judgement: JudgementResult) -> RankedPaper:
    return RankedPaper(
        rank=1,
        paper=judgement.paper,
        final_score=0.8,
        category=judgement.category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.8,
            authority_score=0.5,
            timeliness_score=0.8,
            metadata_score=0.9,
            final_score=0.8,
            relevance_weight=0.65,
            authority_weight=0.08,
            timeliness_weight=0.22,
            metadata_weight=0.05,
        ),
        ranking_reason="rule ranking",
        evidence=judgement.evidence,
        matched_terms=judgement.matched_terms,
    )


def _record(client: FakeLLMClient):
    judgement = _judgement()
    return evolve_with_llm_feedback(
        _analysis(),
        _plan(),
        [judgement],
        [_ranked(judgement)],
        {QUERY},
        llm_client=client,
    )


def test_feedback_calls_llm_once_with_untrusted_metadata_and_temperature_zero() -> None:
    client = FakeLLMClient()

    record = _record(client)

    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0
    rendered = str(client.calls[0]["messages"]).casefold()
    assert "untrusted_metadata_isolation_v1" in rendered
    assert "gold" not in rendered
    assert "qrels" not in rendered
    assert record.llm_feedback is not None
    assert record.llm_feedback.fallback_used is False
    assert record.llm_feedback.total_tokens == 18


def test_feedback_preserves_core_topic_and_explicit_must_have_terms() -> None:
    record = _record(FakeLLMClient())

    assert record.generated_queries[0].query == FEEDBACK_QUERY
    assert "molecular" in record.generated_queries[0].query
    assert "graph" in record.generated_queries[0].query
    assert record.generated_queries[0].purpose.startswith("llm_feedback:")


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_response(QUERY), "all_queries_rejected"),
        (_response("quantum optics photon entanglement"), "all_queries_rejected"),
        ({"unexpected": True}, "invalid_schema"),
    ],
)
def test_feedback_rejects_invalid_or_duplicate_queries(
    response: object,
    reason: str,
) -> None:
    record = _record(FakeLLMClient(response))

    assert record.generated_queries == []
    assert record.llm_feedback is not None
    assert record.llm_feedback.skipped_reason == reason
    assert record.llm_feedback.fallback_used is False


def test_feedback_provider_failure_does_not_create_second_round_query() -> None:
    record = _record(FakeLLMClient(error=TimeoutError("timeout")))

    assert record.generated_queries == []
    assert record.llm_feedback is not None
    assert record.llm_feedback.fallback_reason == "llm_timeout"


def test_coverage_sufficient_is_a_normal_skip_not_a_provider_fallback() -> None:
    analysis = _analysis()
    paper = _paper(
        "Message Passing Graph Neural Networks for Molecular Property Prediction"
    )
    judgement = _judgement(paper)
    client = FakeLLMClient()

    record = evolve_with_llm_feedback(
        analysis,
        _plan(),
        [judgement],
        [_ranked(judgement)],
        {QUERY},
        llm_client=client,
    )

    assert client.calls == []
    assert record.llm_feedback is not None
    assert record.llm_feedback.eligible_for_feedback is False
    assert record.llm_feedback.skipped_reason == "coverage_sufficient"
    assert record.llm_feedback.fallback_used is False


def _retrieval(query: str, **kwargs: object) -> RetrievalOutput:  # noqa: ARG001
    if query == FEEDBACK_QUERY:
        paper = _paper(
            "Message Passing Graph Neural Networks for Molecular Prediction",
            doi="10.1000/feedback-round-two",
        )
    else:
        paper = _paper()
    return RetrievalOutput(
        query=query,
        requested_sources=["arxiv"],
        raw_count=1,
        deduplicated_count=1,
        papers=[paper],
        source_stats=[
            SourceStats(
                source="arxiv",
                query=query,
                returned_count=1,
                diagnostics=ConnectorDiagnostics(request_count=1),
            )
        ],
    )


def test_search_service_uses_one_feedback_query_for_second_round() -> None:
    client = FakeLLMClient()
    service = SearchService(retriever=_retrieval, llm_client=client, max_workers=1)

    output = service.run_search(
        QUERY,
        sources_override=["arxiv"],
        enable_query_evolution=True,
        query_evolution_policy="llm_feedback",
        explicit_constraints=_analysis().constraints,
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=1, max_search_rounds=3),
    )

    assert len(client.calls) == 1
    assert output.llm_call_count == 1
    assert output.query_evolution_records[0].policy == "llm_feedback"
    assert any(item.query == FEEDBACK_QUERY for item in output.retrieval_outputs)


def test_search_service_budget_exhaustion_prevents_feedback_call() -> None:
    client = FakeLLMClient()
    service = SearchService(retriever=_retrieval, llm_client=client, max_workers=1)

    output = service.run_search(
        QUERY,
        sources_override=["arxiv"],
        enable_query_evolution=True,
        query_evolution_policy="llm_feedback",
        explicit_constraints=_analysis().constraints,
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=0, max_search_rounds=3),
    )

    assert client.calls == []
    assert output.query_evolution_records[0].llm_feedback is not None
    assert output.query_evolution_records[0].llm_feedback.fallback_reason == "budget_exhausted"


def test_feedback_policy_rejects_multiple_llm_stages() -> None:
    service = SearchService(retriever=_retrieval, llm_client=FakeLLMClient())

    with pytest.raises(ValueError, match="rule-only initial planning"):
        service.run_search(
            QUERY,
            enable_query_evolution=True,
            query_evolution_policy="llm_feedback",
            query_planning_policy="llm_semantic",
            enable_synthesis=False,
        )
