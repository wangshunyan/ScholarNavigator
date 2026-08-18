from __future__ import annotations

import inspect
import re

import pytest
from pydantic import ValidationError

from scholar_agent.agents import retriever as retriever_module
from scholar_agent.agents.query_planning import plan_disjunctive_facets
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.search_schemas import (
    QUERY_PLANNER_VERSION,
    QueryAnalysis,
    QueryConstraint,
    SearchSubquery,
)
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key
from scholar_agent.retrieval.query_adapter import (
    MAX_ARXIV_QUERY_LENGTH,
    adapt_queries_for_source,
    adapt_query_for_source,
)
from scholar_agent.services.search_service import SearchService


QUERY = (
    "contrastive graph neural retrieval with MS MARCO dataset "
    "for document ranking"
)


def test_disjunctive_plan_retains_original_and_bounds_queries() -> None:
    plan = analyze_query(QUERY, query_planning_policy="disjunctive_facets")

    assert plan.subqueries[0].query == QUERY
    assert plan.subqueries[0].combination_mode == "all"
    assert len(plan.subqueries) <= 3
    any_query = next(
        item for item in plan.subqueries if item.combination_mode == "any"
    )
    assert any_query.purpose == "disjunctive_facet_any"
    assert set(any_query.facet_types).issubset(
        {"topic", "method", "dataset", "task"}
    )
    adapted = adapt_query_for_source(
        any_query.query,
        "arxiv",
        constraints=plan.query_analysis.constraints,
        combination_mode="any",
    )
    assert 4 <= adapted.query.count("all:") <= 8
    assert " OR " in adapted.query
    assert plan.query_planning.policy == "disjunctive_facets"
    assert plan.query_planning.planner_version == QUERY_PLANNER_VERSION == "1.9.0"


def test_explicit_must_have_is_hard_outside_or_group() -> None:
    constraints = QueryConstraint(
        methods=["contrastive"],
        datasets=["MS MARCO"],
        must_include_terms=["document ranking"],
        explicit_fields=["must_include_terms"],
    )
    plan = analyze_query(
        QUERY,
        query_planning_policy="disjunctive_facets",
        explicit_constraints=constraints,
    )
    any_query = next(item for item in plan.subqueries if item.combination_mode == "any")

    adapted = adapt_queries_for_source(
        any_query.query,
        "arxiv",
        constraints=plan.query_analysis.constraints,
        combination_mode=any_query.combination_mode,
    )

    assert len(adapted) == 1
    assert adapted[0].strategy == "disjunctive_any"
    assert adapted[0].query.endswith('AND all:"document ranking"')
    assert 'all:"document ranking" OR' not in adapted[0].query
    assert adapted[0].protected_terms == ["document ranking"]


def test_inferred_must_have_remains_a_soft_facet() -> None:
    analysis = QueryAnalysis(
        original_query=QUERY,
        constraints=QueryConstraint(
            methods=["contrastive"],
            datasets=["MS MARCO"],
            must_include_terms=["document ranking"],
        ),
    )
    subqueries, _ = plan_disjunctive_facets(
        analysis,
        selected_sources=["arxiv"],
        max_subqueries=3,
    )
    any_query = next(item for item in subqueries if item.combination_mode == "any")
    adapted = adapt_query_for_source(
        any_query.query,
        "arxiv",
        constraints=analysis.constraints,
        combination_mode="any",
    )

    assert " AND " not in adapted.query
    assert adapted.protected_terms == []


def test_arxiv_any_escapes_phrases_parentheses_and_special_characters() -> None:
    adapted = adapt_query_for_source(
        '"graph (neural)" retrieval ranking C++ [survey]',
        "arxiv",
        combination_mode="any",
    )

    assert adapted.query.startswith("(") and adapted.query.endswith(")")
    assert 'all:"graph neural"' in adapted.query
    assert "all:retrieval" in adapted.query
    assert "all:ranking" in adapted.query
    assert "all:C++" in adapted.query
    assert not re.search(r"[\[\]{}]", adapted.query)
    assert adapted.query.count('"') % 2 == 0
    assert len(adapted.query) <= MAX_ARXIV_QUERY_LENGTH


def test_retriever_executes_any_as_one_arxiv_or_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def search(query: str, limit: int) -> ConnectorSearchResult:
        del limit
        captured.append(query)
        return ConnectorSearchResult()

    monkeypatch.setattr(
        retriever_module,
        "_source_registry",
        lambda: {"arxiv": search},
    )
    output = retriever_module.retrieve_papers(
        'graph "neural retrieval" ranking benchmark',
        sources=["arxiv"],
        combination_mode="any",
        query_adapter_policy="adaptive",
    )

    assert len(captured) == 1
    assert " OR " in captured[0]
    assert output.source_stats[0].combination_mode == "any"
    assert output.source_stats[0].adaptation_strategy == "disjunctive_any"


def test_search_service_propagates_any_mode_to_default_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def search(query: str, limit: int) -> ConnectorSearchResult:
        del limit
        captured.append(query)
        return ConnectorSearchResult()

    retriever_module.clear_retrieval_cache()
    retriever_module.clear_source_cooldowns()
    monkeypatch.setattr(
        retriever_module,
        "_source_registry",
        lambda: {"arxiv": search},
    )

    output = SearchService(max_workers=1).run_search(
        QUERY,
        query_planning_policy="disjunctive_facets",
        sources_override=["arxiv"],
        query_adapter_policy="adaptive",
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )

    any_queries = [
        item for item in output.search_plan.subqueries if item.combination_mode == "any"
    ]
    assert len(any_queries) == 1
    disjunctive_requests = {
        query for query in captured if " OR " in query and "all:" in query
    }
    assert len(disjunctive_requests) == 1
    any_stats = [
        stats for stats in output.source_stats if stats.combination_mode == "any"
    ]
    assert len(any_stats) == 1
    assert any_stats[0].adaptation_strategy == "disjunctive_any"


def test_disjunctive_planning_is_deterministic_and_gold_independent() -> None:
    first = analyze_query(QUERY, query_planning_policy="disjunctive_facets")
    second = analyze_query(QUERY, query_planning_policy="disjunctive_facets")
    signature = inspect.signature(plan_disjunctive_facets)
    source = inspect.getsource(plan_disjunctive_facets).casefold()

    assert first == second
    assert list(signature.parameters) == [
        "query_analysis",
        "selected_sources",
        "max_subqueries",
    ]
    assert "gold" not in source
    assert "case_id" not in source
    assert "arxiv" not in source


def test_combination_mode_validation_and_current_rules_regression() -> None:
    with pytest.raises(ValidationError):
        SearchSubquery(
            query="graph retrieval",
            combination_mode="some",  # type: ignore[arg-type]
            purpose="invalid",
        )

    default = analyze_query(QUERY, query_planning_policy="current_rules")
    assert all(item.combination_mode == "all" for item in default.subqueries)
    assert default.query_planning_policy == "current_rules"


def test_disjunctive_snapshot_key_is_policy_and_version_isolated() -> None:
    common = {
        "source": "arxiv",
        "adapted_query": "(all:graph OR all:retrieval)",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": "search-v1",
    }
    current, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="current_rules",
        query_planner_version=QUERY_PLANNER_VERSION,
    )
    disjunctive, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="disjunctive_facets",
        query_planner_version=QUERY_PLANNER_VERSION,
    )

    assert current != disjunctive
