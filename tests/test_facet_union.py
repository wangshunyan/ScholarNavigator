from __future__ import annotations

import inspect
from threading import Lock

import pytest

from scholar_agent.agents.query_planning import plan_facet_union
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    QUERY_PLANNER_VERSION,
    QueryAnalysis,
    QueryConstraint,
    QueryPlanningResult,
    SearchBudget,
    SearchSubquery,
)
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key
from scholar_agent.services.search_service import SearchService


QUERY = "contrastive graph representation learning for document ranking"


def test_facet_union_retains_current_queries_and_adds_at_most_one() -> None:
    constraints = QueryConstraint(
        methods=["contrastive learning"],
        datasets=["MS MARCO"],
        explicit_fields=["methods", "datasets"],
    )
    current = analyze_query(
        QUERY,
        query_planning_policy="current_rules",
        explicit_constraints=constraints,
    )
    candidate = analyze_query(
        QUERY,
        query_planning_policy="facet_union",
        explicit_constraints=constraints,
    )

    assert candidate.subqueries[:-1] == current.subqueries
    assert len(candidate.subqueries) == len(current.subqueries) + 1
    supplemental = candidate.subqueries[-1]
    assert supplemental.purpose == "facet_union_dataset"
    assert supplemental.combination_mode == "all"
    assert supplemental.facet_types == ["dataset"]
    assert "MS MARCO" in supplemental.query
    assert candidate.query_planning.policy == "facet_union"
    assert candidate.query_planning.planner_version == QUERY_PLANNER_VERSION == "1.9.0"


@pytest.mark.parametrize(
    ("query", "constraints", "expected_purpose"),
    [
        (
            QUERY,
            QueryConstraint(
                methods=["contrastive learning"],
                datasets=["inferred corpus"],
                explicit_fields=["methods"],
            ),
            "facet_union_method",
        ),
        (
            "robust image segmentation under domain shift",
            QueryConstraint(),
            "facet_union_task",
        ),
        (
            "variational manifold alignment latent geometry",
            QueryConstraint(),
            "facet_union_topic",
        ),
    ],
)
def test_facet_selection_priority_is_deterministic(
    query: str,
    constraints: QueryConstraint,
    expected_purpose: str,
) -> None:
    subqueries, _ = _direct_plan(query, constraints)

    assert subqueries[-1].purpose == expected_purpose
    assert subqueries[-1].combination_mode == "all"
    assert " OR " not in subqueries[-1].query


def test_explicit_must_have_is_retained_but_inferred_term_is_not_hard() -> None:
    explicit = analyze_query(
        QUERY,
        query_planning_policy="facet_union",
        explicit_constraints=QueryConstraint(
            methods=["contrastive learning"],
            must_include_terms=["zero-shot"],
            explicit_fields=["methods", "must_include_terms"],
        ),
    )
    inferred, _ = _direct_plan(
        QUERY,
        QueryConstraint(
            methods=["contrastive learning"],
            must_include_terms=["inferred-hard"],
            explicit_fields=["methods"],
        ),
    )

    assert "zero-shot" in explicit.subqueries[-1].query
    assert "inferred-hard" not in inferred[-1].query
    assert "graph" not in inferred[-1].query.casefold()


def test_baseline_finishes_before_facet_query_execution() -> None:
    plan = analyze_query(QUERY, query_planning_policy="facet_union")
    supplemental = plan.subqueries[-1].query
    baseline = {item.query for item in plan.subqueries[:-1]}
    calls: list[str] = []
    lock = Lock()

    def retrieve(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        with lock:
            calls.append(query)
        return _retrieval(query, [_paper(f"paper:{query}")])

    output = SearchService(retriever=retrieve, max_workers=4).run_search(
        QUERY,
        query_planning_policy="facet_union",
        sources_override=["arxiv"],
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )

    assert calls[-1] == supplemental
    assert set(calls[:-1]) == baseline
    assert output.retrieval_outputs[-1].query == supplemental


def test_facet_query_uses_only_remaining_candidate_budget() -> None:
    baseline = [_paper("baseline one"), _paper("baseline two")]
    supplemental_papers = [_paper("facet one"), _paper("facet two")]
    plan = analyze_query(QUERY, query_planning_policy="facet_union")
    facet_query = plan.subqueries[-1].query
    observed_limits: dict[str, int] = {}

    def by_query(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        observed_limits[query] = limit_per_source
        papers = supplemental_papers if query == facet_query else baseline
        return _retrieval(query, papers)

    output = SearchService(retriever=by_query, max_workers=1).run_search(
        QUERY,
        query_planning_policy="facet_union",
        sources_override=["arxiv"],
        budget=SearchBudget(max_candidate_papers=3),
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
        collect_diagnostics=True,
    )
    snapshot = next(
        item for item in output.stage_snapshots if item.stage == "initial_deduplicated"
    )

    assert [item.title for item in snapshot.candidates] == [
        "baseline one",
        "baseline two",
        "facet one",
    ]
    assert observed_limits[facet_query] == 1


def test_facet_query_is_skipped_after_baseline_exhausts_budget() -> None:
    calls: list[str] = []

    def retrieve(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        calls.append(query)
        return _retrieval(query, [_paper("baseline")])

    output = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        query_planning_policy="facet_union",
        sources_override=["arxiv"],
        budget=SearchBudget(max_candidate_papers=1),
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )
    facet_query = output.search_plan.subqueries[-1].query
    skipped = output.retrieval_outputs[-1]

    assert facet_query not in calls
    assert skipped.query == facet_query
    assert skipped.source_stats[0].logical_call_executed is False
    assert skipped.source_stats[0].source_skipped_reason == (
        "budget_stop:max_candidate_papers"
    )
    assert "facet_union_skipped:budget_stop:max_candidate_papers" in output.warnings


def test_snapshot_key_is_isolated_and_planner_is_gold_free() -> None:
    common = {
        "source": "arxiv",
        "adapted_query": "contrastive learning",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": "search-v1",
        "query_planner_version": QUERY_PLANNER_VERSION,
    }
    current, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="current_rules",
    )
    candidate, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="facet_union",
    )
    parameters = inspect.signature(plan_facet_union).parameters
    source = inspect.getsource(plan_facet_union).casefold()

    assert current != candidate
    assert "gold" not in parameters
    assert "case_id" not in parameters
    assert "gold" not in source
    assert "case_id" not in source


def test_facet_union_is_deterministic() -> None:
    first = analyze_query(QUERY, query_planning_policy="facet_union")
    second = analyze_query(QUERY, query_planning_policy="facet_union")

    assert first == second


def _retrieval(query: str, papers: list[Paper]) -> RetrievalOutput:
    return RetrievalOutput(
        query=query,
        requested_sources=["arxiv"],
        raw_count=len(papers),
        deduplicated_count=len(papers),
        papers=papers,
        source_stats=[
            SourceStats(
                source="arxiv",
                query=query,
                returned_count=len(papers),
                diagnostic_papers=papers,
            )
        ],
    )


def _direct_plan(
    query: str,
    constraints: QueryConstraint,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    analysis = QueryAnalysis(
        original_query=query,
        language="en",
        intent="general",
        domain="computer_science",
        constraints=constraints,
    )
    return plan_facet_union(
        analysis,
        current_subqueries=[
            SearchSubquery(
                query=query,
                source_hints=["arxiv"],
                priority=1,
                purpose="original_query",
            )
        ],
        selected_sources=["arxiv"],
        max_subqueries=3,
    )


def _paper(title: str) -> Paper:
    return Paper(
        title=title,
        abstract=f"abstract for {title}",
        year=2024,
        identifiers=PaperIdentifiers(arxiv_id=f"fixture:{title}"),
        sources=["arxiv"],
    )
