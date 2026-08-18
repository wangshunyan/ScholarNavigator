from __future__ import annotations

import inspect
from threading import Lock

from scholar_agent.agents.query_planning import plan_current_plus_disjunctive
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QUERY_PLANNER_VERSION, SearchBudget
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key
from scholar_agent.services.search_service import SearchService


QUERY = (
    "contrastive graph neural retrieval with MS MARCO dataset "
    "for document ranking"
)


def test_current_queries_are_retained_before_single_or_query() -> None:
    current = analyze_query(QUERY, query_planning_policy="current_rules")
    candidate = analyze_query(
        QUERY,
        query_planning_policy="current_plus_disjunctive",
    )

    assert [
        (item.query, item.purpose, item.priority, item.combination_mode)
        for item in candidate.subqueries[:-1]
    ] == [
        (item.query, item.purpose, item.priority, item.combination_mode)
        for item in current.subqueries
    ]
    assert len(candidate.subqueries) == len(current.subqueries) + 1
    assert candidate.subqueries[-1].purpose == "current_plus_disjunctive_any"
    assert candidate.subqueries[-1].combination_mode == "any"
    assert candidate.subqueries[-1].priority == len(candidate.subqueries)
    assert candidate.query_planning.policy == "current_plus_disjunctive"
    assert candidate.query_planning.planner_version == QUERY_PLANNER_VERSION == "1.9.0"


def test_current_queries_finish_before_or_execution() -> None:
    plan = analyze_query(QUERY, query_planning_policy="current_plus_disjunctive")
    or_query = plan.subqueries[-1].query
    baseline_queries = {item.query for item in plan.subqueries[:-1]}
    calls: list[str] = []
    lock = Lock()

    def retrieve(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        with lock:
            calls.append(query)
        return _retrieval(query, [_paper(f"paper:{query}")])

    output = SearchService(retriever=retrieve, max_workers=4).run_search(
        QUERY,
        query_planning_policy="current_plus_disjunctive",
        sources_override=["arxiv"],
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )

    assert calls[-1] == or_query
    assert set(calls[:-1]) == baseline_queries
    assert output.retrieval_outputs[-1].query == or_query


def test_or_is_skipped_when_current_candidates_exhaust_budget() -> None:
    calls: list[str] = []
    baseline = _paper("baseline")

    def retrieve(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        calls.append(query)
        return _retrieval(query, [baseline])

    output = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        query_planning_policy="current_plus_disjunctive",
        sources_override=["arxiv"],
        budget=SearchBudget(max_candidate_papers=1),
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )
    or_query = output.search_plan.subqueries[-1].query
    skipped = output.retrieval_outputs[-1]

    assert or_query not in calls
    assert skipped.query == or_query
    assert skipped.source_stats[0].logical_call_executed is False
    assert (
        skipped.source_stats[0].source_skipped_reason
        == "budget_stop:max_candidate_papers"
    )
    assert "current_plus_disjunctive_skipped:budget_stop:max_candidate_papers" in (
        output.warnings
    )


def test_or_candidates_only_fill_remaining_candidate_budget() -> None:
    baseline = [_paper("baseline one"), _paper("baseline two")]
    or_papers = [_paper("or one"), _paper("or two")]
    plan = analyze_query(QUERY, query_planning_policy="current_plus_disjunctive")
    or_query = plan.subqueries[-1].query
    observed_limits: dict[str, int] = {}

    def by_query(query: str, limit_per_source: int, sources: list[str]) -> RetrievalOutput:
        observed_limits[query] = limit_per_source
        return _retrieval(query, or_papers if query == or_query else baseline)

    output = SearchService(retriever=by_query, max_workers=1).run_search(
        QUERY,
        query_planning_policy="current_plus_disjunctive",
        sources_override=["arxiv"],
        budget=SearchBudget(max_candidate_papers=3),
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
        collect_diagnostics=True,
    )
    deduplicated = next(
        item for item in output.stage_snapshots if item.stage == "initial_deduplicated"
    )
    titles = [item.title for item in deduplicated.candidates]

    assert len(titles) == 3
    assert titles[:2] == ["baseline one", "baseline two"]
    assert titles[2] == "or one"
    assert observed_limits[or_query] == 1


def test_or_duplicate_and_insufficient_terms_are_stably_skipped() -> None:
    candidate = analyze_query(
        QUERY,
        query_planning_policy="current_plus_disjunctive",
    )
    or_subquery = candidate.subqueries[-1]
    duplicated, result = plan_current_plus_disjunctive(
        candidate.query_analysis,
        current_subqueries=[*candidate.subqueries[:-1], or_subquery],
        selected_sources=["arxiv"],
        max_subqueries=4,
    )
    short = analyze_query(
        "graph",
        query_planning_policy="current_plus_disjunctive",
    )

    assert duplicated == candidate.subqueries
    assert result.skipped_facets == ["duplicate:current_plus_disjunctive_any"]
    assert all(item.combination_mode == "all" for item in short.subqueries)
    assert short.query_planning.skipped_facets == [
        "insufficient_high_confidence_terms:current_plus_disjunctive_any"
    ]


def test_snapshot_key_and_planner_are_policy_isolated_and_gold_free() -> None:
    common = {
        "source": "arxiv",
        "adapted_query": "(all:graph OR all:retrieval)",
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
        query_planning_policy="current_plus_disjunctive",
    )
    parameters = inspect.signature(plan_current_plus_disjunctive).parameters
    source = inspect.getsource(plan_current_plus_disjunctive).casefold()

    assert current != candidate
    assert "gold" not in parameters
    assert "case_id" not in parameters
    assert "gold" not in source
    assert "case_id" not in source


def test_current_plus_planning_is_deterministic() -> None:
    first = analyze_query(QUERY, query_planning_policy="current_plus_disjunctive")
    second = analyze_query(QUERY, query_planning_policy="current_plus_disjunctive")

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


def _paper(title: str) -> Paper:
    return Paper(
        title=title,
        abstract=f"abstract for {title}",
        year=2024,
        identifiers=PaperIdentifiers(arxiv_id=f"fixture:{title}"),
        sources=["arxiv"],
    )
