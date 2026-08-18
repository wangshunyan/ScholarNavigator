from __future__ import annotations

import inspect

import pytest

from scholar_agent.agents.query_planning import plan_controlled_relaxation
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import QueryConstraint


def test_original_query_is_first_and_supplemental_count_is_bounded() -> None:
    query = (
        "Which studies designed a siamese network framework using AlexNet "
        "for feature extraction in visual object tracking?"
    )

    plan = analyze_query(
        query,
        run_profile="high_recall",
        query_planning_policy="controlled_relaxation",
    )

    assert plan.subqueries[0].query == query
    assert plan.subqueries[0].purpose == "original_query"
    assert len(plan.subqueries) <= 3
    assert sum(item.purpose != "original_query" for item in plan.subqueries) <= 2


def test_explicit_must_have_is_retained_in_every_supplemental_query() -> None:
    plan = analyze_query(
        "adaptive retrieval for clinical reasoning",
        query_planning_policy="controlled_relaxation",
        explicit_constraints=QueryConstraint(
            must_include_terms=["clinical question answering", "evidence grounding"],
            methods=["contrastive learning"],
            explicit_fields=["must_include_terms", "methods"],
        ),
    )

    assert len(plan.subqueries) >= 2
    for subquery in plan.subqueries[1:]:
        assert "clinical question answering" in subquery.query
        assert "evidence grounding" in subquery.query
        assert "must_have:explicit" in subquery.provenance


def test_rule_inferred_must_have_terms_are_soft_not_all_hard_conditions() -> None:
    plan = analyze_query(
        "Which papers focused on locally aligning fixed patches with textual words?",
        query_planning_policy="controlled_relaxation",
    )

    inferred = {
        term.casefold()
        for term in plan.query_analysis.constraints.must_include_terms
    }
    supplemental_terms = {
        term.casefold()
        for subquery in plan.subqueries[1:]
        for term in subquery.query.split()
    }

    assert inferred
    assert "must_include_terms" not in plan.query_analysis.constraints.explicit_fields
    assert not inferred.issubset(supplemental_terms)


def test_only_one_reliable_facet_is_added_without_strong_boolean_and() -> None:
    plan = analyze_query(
        "adaptive evidence retrieval for clinical question answering",
        query_planning_policy="controlled_relaxation",
        explicit_constraints=QueryConstraint(
            methods=["contrastive reranking"],
            datasets=["NovaSet"],
            paper_types=["comparison"],
            explicit_fields=["methods", "datasets", "paper_types"],
        ),
    )

    facet_queries = [
        item
        for item in plan.subqueries
        if item.purpose.startswith("controlled_core_plus_")
    ]
    assert len(facet_queries) == 1
    assert facet_queries[0].purpose == "controlled_core_plus_dataset"
    assert facet_queries[0].facet_types == ["topic", "dataset"]
    assert all(" AND " not in item.query.upper() for item in plan.subqueries)


def test_false_substring_method_is_not_added_as_a_facet() -> None:
    plan = analyze_query(
        "How can paragraph representations learn a proxy reward function?",
        query_planning_policy="controlled_relaxation",
    )

    assert plan.query_analysis.constraints.methods == ["rag"]
    assert all(item.purpose != "controlled_core_plus_method" for item in plan.subqueries)


@pytest.mark.parametrize("profile,maximum", [("fast", 2), ("balanced", 3)])
def test_profile_budget_is_respected(profile: str, maximum: int) -> None:
    plan = analyze_query(
        "semantic segmentation with deep learning methods",
        run_profile=profile,
        query_planning_policy="controlled_relaxation",
    )

    assert len(plan.subqueries) <= maximum


def test_controlled_relaxation_is_deterministic_and_source_independent() -> None:
    query = "ray based rendering for novel view synthesis"
    first = analyze_query(query, query_planning_policy="controlled_relaxation")
    second = analyze_query(query, query_planning_policy="controlled_relaxation")
    signature = inspect.signature(plan_controlled_relaxation)
    source = inspect.getsource(plan_controlled_relaxation).casefold()

    assert first.model_dump() == second.model_dump()
    assert "gold" not in signature.parameters
    assert "gold" not in source
    assert "paper_title" not in source
    assert "arxiv_id" not in source
    assert "arxiv" not in source
    assert "openalex" not in source
    assert "semantic_scholar" not in source
    assert "pubmed" not in source
    assert "case_id" not in source


def test_current_rules_output_is_unchanged_when_policy_is_not_selected() -> None:
    query = "graph representation learning survey"
    implicit = analyze_query(query)
    explicit = analyze_query(query, query_planning_policy="current_rules")

    assert implicit.model_dump() == explicit.model_dump()
