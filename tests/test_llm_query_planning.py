from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import QueryConstraint, SearchBudget
from scholar_agent.services.search_service import SearchService


QUERY = "graph neural networks for molecule property prediction"


def _valid_output(*queries: str) -> dict[str, object]:
    values = queries or (
        "graph neural networks molecular property prediction benchmark",
    )
    return {
        "intent_summary": "molecular property prediction",
        "facets": [
            {
                "facet_type": "synonym",
                "original_terms": ["molecule"],
                "normalized_terms": ["molecular"],
                "confidence": 0.9,
            }
        ],
        "supplemental_queries": [
            {
                "query": query,
                "purpose": f"semantic expansion {index}",
                "covered_facets": ["topic", "task"],
                "retained_must_have_terms": [],
                "terminology_expansions": ["molecule -> molecular"],
            }
            for index, query in enumerate(values)
        ],
        "warnings": [],
    }


class FakeLLMClient:
    provider = "test_provider"
    model = "semantic-test-v1"

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = _valid_output() if response is None else response
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls.append(
            {"messages": messages, "temperature": temperature, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        self.token_usage.prompt_tokens += 11
        self.token_usage.completion_tokens += 7
        self.token_usage.total_tokens += 18
        return deepcopy(self.response)


def _plan(client: FakeLLMClient | None, **kwargs):  # noqa: ANN003, ANN202
    return analyze_query(
        QUERY,
        query_planning_policy="llm_semantic",
        llm_client=client,
        **kwargs,
    )


def test_llm_semantic_always_retains_original_query() -> None:
    plan = _plan(FakeLLMClient())

    assert plan.subqueries[0].query == QUERY
    assert plan.query_planning.original_query_retained is True


def test_llm_semantic_accepts_at_most_two_supplemental_queries() -> None:
    output = _valid_output(
        "graph neural networks molecular property prediction benchmark",
        "graph neural network molecule property estimation comparison",
    )
    plan = _plan(FakeLLMClient(output), run_profile="high_recall")

    assert len(plan.subqueries) == 3
    assert plan.query_planning.accepted_query_count == 2


def test_llm_semantic_calls_model_once_with_temperature_zero() -> None:
    client = FakeLLMClient()
    _plan(client)

    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0


def test_llm_semantic_prompt_input_excludes_evaluation_and_candidates() -> None:
    client = FakeLLMClient()
    _plan(client)
    rendered = str(client.calls[0]["messages"]).casefold()

    assert "gold" not in rendered
    assert "qrels" not in rendered
    assert "case_id" not in rendered
    assert "candidate_papers" not in rendered


def test_llm_semantic_valid_output_is_accepted_with_provenance() -> None:
    plan = _plan(FakeLLMClient())

    assert plan.query_planning.output_valid is True
    assert plan.query_planning.fallback_used is False
    assert plan.subqueries[1].provenance[0] == "llm_semantic"
    assert "prompt:llm_query_planning@1.0.0" in plan.subqueries[1].provenance


def test_llm_semantic_invalid_schema_falls_back_to_current_rules() -> None:
    plan = _plan(FakeLLMClient({"unexpected": True}))

    assert plan.query_planning.fallback_used is True
    assert plan.query_planning.fallback_reason == "invalid_schema"
    assert len(plan.subqueries) >= 1


def test_llm_semantic_empty_output_falls_back() -> None:
    empty = _valid_output()
    empty["supplemental_queries"] = []
    plan = _plan(FakeLLMClient(empty))

    assert plan.query_planning.fallback_reason == "all_queries_rejected"


def test_llm_semantic_timeout_falls_back_without_failing_search() -> None:
    plan = _plan(FakeLLMClient(error=TimeoutError("request timeout")))

    assert plan.query_planning.fallback_reason == "llm_timeout"
    assert plan.subqueries


def test_llm_semantic_unconfigured_falls_back_without_call() -> None:
    plan = _plan(None)

    assert plan.query_planning.fallback_reason == "llm_unconfigured"
    assert plan.query_planning.llm_call_attempted is False


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("quantum optics photon entanglement", "missing_core_topic"),
        ("graph neural networks 10.1234/secret", "suspicious_identifier"),
        ("graph neural networks Smith et al. (2024)", "suspicious_citation"),
        (
            'graph neural networks paper titled "A Specific Molecular Method"',
            "suspicious_citation",
        ),
        ("graph neural networks " + "molecule " * 40, "query_too_long"),
        (
            "graph neural networks "
            + " ".join(f"t{index}" for index in range(30)),
            "too_many_terms",
        ),
        (QUERY, "duplicate_query"),
    ],
)
def test_llm_semantic_rejects_unsafe_or_low_quality_query(
    query: str,
    reason: str,
) -> None:
    plan = _plan(FakeLLMClient(_valid_output(query)))

    assert plan.query_planning.fallback_used is True
    assert plan.query_planning.rejection_reasons[reason] == 1


def test_llm_semantic_requires_explicit_must_have_terms() -> None:
    constraints = QueryConstraint(
        must_include_terms=["clinical evidence"],
        explicit_fields=["must_include_terms"],
    )
    plan = _plan(FakeLLMClient(), explicit_constraints=constraints)

    assert plan.query_planning.rejection_reasons["missing_must_have"] == 1


def test_llm_semantic_rejects_explicit_excluded_terms() -> None:
    constraints = QueryConstraint(
        exclude_terms=["benchmark"],
        explicit_fields=["exclude_terms"],
    )
    plan = _plan(FakeLLMClient(), explicit_constraints=constraints)

    assert plan.query_planning.rejection_reasons["contains_excluded_term"] == 1


def test_llm_semantic_rejects_empty_query_with_stable_reason() -> None:
    plan = _plan(FakeLLMClient(_valid_output("   ")))

    assert plan.query_planning.rejection_reasons["empty_query"] == 1


def test_llm_semantic_rejects_unrelated_expansion_when_no_core_topic() -> None:
    output = _valid_output("quantum optics photon entanglement")
    plan = analyze_query(
        "papers",
        query_planning_policy="llm_semantic",
        llm_client=FakeLLMClient(output),
    )

    assert plan.query_planning.rejection_reasons["unrelated_expansion"] == 1


def test_llm_semantic_keeps_valid_query_when_a_sibling_is_rejected() -> None:
    output = _valid_output(
        "quantum optics photon entanglement",
        "graph neural networks molecular property prediction benchmark",
    )
    plan = _plan(FakeLLMClient(output), run_profile="high_recall")

    assert plan.query_planning.accepted_query_count == 1
    assert plan.query_planning.rejected_query_count == 1
    assert plan.query_planning.fallback_used is False


def test_llm_semantic_records_token_and_latency_diagnostics() -> None:
    plan = _plan(FakeLLMClient())

    assert plan.query_planning.llm_prompt_tokens == 11
    assert plan.query_planning.llm_completion_tokens == 7
    assert plan.query_planning.llm_total_tokens == 18
    assert plan.query_planning.recorded_llm_latency_seconds >= 0


def test_current_rules_never_calls_llm_semantic_planner() -> None:
    client = FakeLLMClient()
    plan = analyze_query(QUERY, query_planning_policy="current_rules", llm_client=client)

    assert not client.calls
    assert plan.query_planning.policy == "current_rules"


def test_llm_semantic_ignores_general_query_understanding_toggle_to_avoid_two_calls() -> None:
    client = FakeLLMClient()
    plan = analyze_query(
        QUERY,
        query_planning_policy="llm_semantic",
        use_llm=True,
        llm_client=client,
    )

    assert len(client.calls) == 1
    assert plan.query_planning.policy == "llm_semantic"


def test_llm_semantic_output_is_stable_for_same_input() -> None:
    first = _plan(FakeLLMClient())
    second = _plan(FakeLLMClient())

    first.query_planning.recorded_llm_latency_seconds = 0
    second.query_planning.recorded_llm_latency_seconds = 0
    assert first.model_dump() == second.model_dump()


def test_search_service_survives_llm_planning_failure() -> None:
    service = SearchService(
        retriever=lambda query, **kwargs: _empty_retrieval(query),  # noqa: ARG005
        llm_client=FakeLLMClient(error=RuntimeError("provider failed")),
        max_workers=1,
    )

    output = service.run_search(
        QUERY,
        query_planning_policy="llm_semantic",
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=1),
    )

    assert output.search_plan.query_planning.fallback_used is True
    assert output.search_plan.subqueries


def test_search_service_llm_budget_stop_falls_back_before_model_call() -> None:
    client = FakeLLMClient()
    service = SearchService(
        retriever=lambda query, **kwargs: _empty_retrieval(query),  # noqa: ARG005
        llm_client=client,
        max_workers=1,
    )

    output = service.run_search(
        QUERY,
        query_planning_policy="llm_semantic",
        enable_synthesis=False,
        budget=SearchBudget(max_llm_calls=0),
    )

    assert output.search_plan.query_planning.fallback_reason == "budget_exhausted"
    assert output.search_plan.query_planning.llm_call_attempted is False
    assert client.calls == []


def _empty_retrieval(query: str):  # noqa: ANN202
    from scholar_agent.agents.retriever import RetrievalOutput

    return RetrievalOutput(query=query, requested_sources=[], papers=[])
