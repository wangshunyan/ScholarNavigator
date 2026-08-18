from __future__ import annotations

from scholar_agent.agents.query_planning import plan_concept_projection
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.retriever import RetrievalOutput
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint, SearchSubquery
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key
from scholar_agent.services.search_service import SearchService


QUERY = "Find three recent papers about graph graph retrieval without recommendation"


def _constraints() -> QueryConstraint:
    return QueryConstraint(
        must_include_terms=["graph", "retrieval", "recommendation"],
        exclude_terms=["recommendation"],
        explicit_fields=["must_include_terms", "exclude_terms"],
    )


def test_projection_preserves_order_deduplicates_and_excludes_constraints() -> None:
    plan = analyze_query(
        QUERY,
        query_planning_policy="concept_projection",
        explicit_constraints=_constraints(),
    )

    diagnostics = plan.query_planning
    assert diagnostics.concept_projection_input_concepts == [
        "three",
        "graph",
        "retrieval",
        "without",
        "recommendation",
    ]
    assert diagnostics.concept_projection_selected_concepts == [
        "graph",
        "retrieval",
    ]
    assert diagnostics.concept_projection_query == "graph retrieval"
    assert plan.subqueries[-1].query == "graph retrieval"
    assert plan.subqueries[-1].purpose == "concept_projection"
    assert "three" not in plan.subqueries[-1].query
    assert "recent" not in plan.subqueries[-1].query
    assert "without" not in plan.subqueries[-1].query
    assert "recommendation" not in plan.subqueries[-1].query


def test_projection_uses_original_unicode_surface_without_rewriting() -> None:
    query = "请查找 αvβ8 与 PTEN graph retrieval 的论文"

    plan = analyze_query(query, query_planning_policy="concept_projection")

    projection = plan.query_planning.concept_projection_query
    assert projection == "PTEN graph retrieval"
    assert all(term in query for term in projection.split())
    assert "论文" not in projection


def test_projection_deduplicates_boundary_punctuation_and_filters_constraints() -> None:
    analysis = QueryAnalysis(
        original_query="Find three papers. about anxiety prevalence.",
        constraints=QueryConstraint(
            must_include_terms=["papers.", "prevalence.", "prevalence"],
        ),
    )
    original = SearchSubquery(
        query=analysis.original_query,
        source_hints=["arxiv"],
        priority=1,
        purpose="original_query",
    )
    derived = SearchSubquery(
        query="anxiety papers",
        source_hints=["arxiv"],
        priority=2,
        purpose="normalized_keywords",
    )

    subqueries, diagnostics = plan_concept_projection(
        analysis,
        current_subqueries=[original, derived],
        selected_sources=["arxiv"],
        max_subqueries=2,
    )

    assert diagnostics.concept_projection_selected_concepts == [
        "anxiety",
        "prevalence.",
    ]
    assert diagnostics.concept_projection_query == "anxiety prevalence."
    assert subqueries[-1].query == "anxiety prevalence."


def test_projection_replaces_lowest_priority_derived_query_without_budget_growth() -> None:
    current = analyze_query(
        QUERY,
        query_planning_policy="current_rules",
        explicit_constraints=_constraints(),
    )
    projected = analyze_query(
        QUERY,
        query_planning_policy="concept_projection",
        explicit_constraints=_constraints(),
    )

    assert len(projected.subqueries) == len(current.subqueries)
    assert projected.subqueries[0] == current.subqueries[0]
    assert projected.subqueries[:-1] == current.subqueries[:-1]
    assert projected.subqueries[-1].priority == current.subqueries[-1].priority
    assert projected.query_planning.concept_projection_replaced_query == (
        current.subqueries[-1].query
    )
    assert projected.query_planning.concept_projection_replaced_purpose == (
        current.subqueries[-1].purpose
    )


def test_projection_skips_empty_and_equivalent_queries() -> None:
    empty = analyze_query(
        "recent papers",
        query_planning_policy="concept_projection",
    )
    analysis = QueryAnalysis(
        original_query="Could you find graph retrieval papers?",
        constraints=QueryConstraint(must_include_terms=["graph", "retrieval"]),
    )
    original = SearchSubquery(
        query=analysis.original_query,
        source_hints=["arxiv"],
        priority=1,
        purpose="original_query",
    )
    equivalent_query = SearchSubquery(
        query="graph retrieval",
        source_hints=["arxiv"],
        priority=2,
        purpose="normalized_keywords",
    )
    _, equivalent = plan_concept_projection(
        analysis,
        current_subqueries=[original, equivalent_query],
        selected_sources=["arxiv"],
        max_subqueries=2,
    )

    assert empty.query_planning.concept_projection_query is None
    assert empty.query_planning.concept_projection_skip_reason == (
        "no_must_have_or_topic_concepts"
    )
    assert equivalent.concept_projection_skip_reason == "equivalent_existing_query"


def test_projection_skips_when_no_derived_query_can_be_replaced() -> None:
    analysis = QueryAnalysis(
        original_query="Could you find graph retrieval papers?",
        constraints=QueryConstraint(must_include_terms=["graph", "retrieval"]),
    )
    original = SearchSubquery(
        query=analysis.original_query,
        source_hints=["arxiv"],
        priority=1,
        purpose="original_query",
    )

    subqueries, diagnostics = plan_concept_projection(
        analysis,
        current_subqueries=[original],
        selected_sources=["arxiv"],
        max_subqueries=1,
    )

    assert subqueries == [original]
    assert diagnostics.concept_projection_query == "graph retrieval"
    assert diagnostics.concept_projection_skip_reason == "no_derived_query_to_replace"


def test_current_rules_remains_the_default_and_has_no_projection_diagnostics() -> None:
    plan = analyze_query(QUERY, explicit_constraints=_constraints())

    assert plan.query_planning_policy == "current_rules"
    assert all(item.purpose != "concept_projection" for item in plan.subqueries)
    assert plan.query_planning.concept_projection_input_concepts == []
    assert plan.query_planning.concept_projection_query is None


def test_projection_and_source_execution_order_are_deterministic() -> None:
    expected = analyze_query(
        QUERY,
        query_planning_policy="concept_projection",
        explicit_constraints=_constraints(),
    )
    calls: list[tuple[str, list[str]]] = []

    def retrieve(
        query: str,
        limit_per_source: int,
        sources: list[str],
    ) -> RetrievalOutput:
        del limit_per_source
        calls.append((query, list(sources)))
        return RetrievalOutput(
            query=query,
            requested_sources=list(sources),
            raw_count=0,
            deduplicated_count=0,
        )

    output = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        query_planning_policy="concept_projection",
        explicit_constraints=_constraints(),
        sources_override=["pubmed", "arxiv"],
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )

    assert [query for query, _ in calls] == [item.query for item in expected.subqueries]
    assert all(sources == ["pubmed", "arxiv"] for _, sources in calls)
    assert [item.query for item in output.search_plan.subqueries] == [
        item.query for item in expected.subqueries
    ]
    assert [item.purpose for item in output.search_plan.subqueries] == [
        item.purpose for item in expected.subqueries
    ]
    second = analyze_query(
        QUERY,
        query_planning_policy="concept_projection",
        explicit_constraints=_constraints(),
    )
    assert second.model_dump() == expected.model_dump()


def test_projection_snapshot_namespace_is_distinct_from_current_rules() -> None:
    common = {
        "source": "arxiv",
        "adapted_query": "graph retrieval",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": "test",
        "query_planner_version": "1.8.1",
    }

    current, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="current_rules",
    )
    projected, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="concept_projection",
    )

    assert projected != current
