from __future__ import annotations

import inspect

import pytest

from scholar_agent.agents.query_planning import plan_facet_balanced
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint


def _explicit_constraints() -> QueryConstraint:
    return QueryConstraint(
        methods=["contrastive reranking"],
        datasets=["NovaSet"],
        must_include_terms=["clinical question answering"],
        paper_types=["comparison"],
        venues=["ACL"],
        explicit_fields=[
            "methods",
            "datasets",
            "must_include_terms",
            "paper_types",
            "venues",
        ],
    )


@pytest.mark.parametrize(
    "policy",
    ["current_rules", "prf_v1", "concept_projection", "facet_balanced"],
)
def test_original_query_is_always_preserved(policy: str) -> None:
    query = "hybrid retrieval for clinical question answering"

    plan = analyze_query(query, query_planning_policy=policy)  # type: ignore[arg-type]

    assert plan.subqueries[0].query == query
    assert plan.subqueries[0].purpose == "original_query"


def test_explicit_method_and_dataset_facets_have_priority() -> None:
    plan = analyze_query(
        "adaptive evidence retrieval for clinical question answering",
        query_planning_policy="facet_balanced",
        explicit_constraints=_explicit_constraints(),
    )

    assert [item.purpose for item in plan.subqueries] == [
        "original_query",
        "facet_method",
        "facet_dataset",
    ]
    assert "contrastive reranking" in plan.subqueries[1].query
    assert "NovaSet" in plan.subqueries[2].query


def test_explicit_must_have_is_retained_in_every_facet_query() -> None:
    plan = analyze_query(
        "adaptive evidence retrieval",
        query_planning_policy="facet_balanced",
        explicit_constraints=_explicit_constraints(),
    )

    assert all(
        "clinical question answering" in item.query
        for item in plan.subqueries[1:]
    )


def test_topic_compaction_removes_question_boilerplate_without_losing_topic() -> None:
    query = "Could you provide papers about voxel geometry for scene representation?"

    plan = analyze_query(query, query_planning_policy="facet_balanced")

    compact = next(
        item for item in plan.subqueries if item.purpose == "facet_topic_compact"
    )
    assert "voxel" in compact.query.casefold()
    assert "geometry" in compact.query.casefold()
    assert "scene" in compact.query.casefold()
    assert "provide" not in compact.query.casefold()
    assert compact.facet_types == ["topic"]


def test_inferred_method_is_separated_from_compact_topic_dimension() -> None:
    plan = analyze_query(
        "white-box cyber-security scenarios in machine learning",
        query_planning_policy="facet_balanced",
    )

    compact = next(
        item for item in plan.subqueries if item.purpose == "facet_topic_compact"
    )
    method = next(item for item in plan.subqueries if item.purpose == "facet_method")
    assert "machine learning" not in compact.query.casefold()
    assert "machine learning" in method.query.casefold()


def test_paper_type_facet_is_selected_when_higher_facets_are_absent() -> None:
    plan = analyze_query(
        "graph representation learning papers",
        query_planning_policy="facet_balanced",
        explicit_constraints=QueryConstraint(
            paper_types=["survey"],
            explicit_fields=["paper_types"],
        ),
    )

    assert any(item.purpose == "facet_paper_type" for item in plan.subqueries)
    assert any("survey" in item.query.casefold() for item in plan.subqueries[1:])


def test_venue_and_time_never_become_standalone_topic_queries() -> None:
    plan = analyze_query(
        "graph representation learning 2021-2024",
        query_planning_policy="facet_balanced",
        explicit_constraints=QueryConstraint(
            venues=["NeurIPS"],
            explicit_fields=["venues"],
        ),
    )

    supplemental = plan.subqueries[1:]
    assert supplemental
    assert all("topic" in item.facet_types for item in supplemental)
    assert all(len(item.query.split()) > 1 for item in supplemental)


def test_chinese_and_mixed_queries_keep_original_text() -> None:
    chinese = "图神经网络在分子属性预测中的方法"
    mixed = "请查找 LLM 在学术检索中的重排序方法"

    chinese_plan = analyze_query(chinese, query_planning_policy="facet_balanced")
    mixed_plan = analyze_query(mixed, query_planning_policy="facet_balanced")

    assert chinese_plan.subqueries[0].query == chinese
    assert mixed_plan.subqueries[0].query == mixed
    assert mixed_plan.query_analysis.language == "mixed"


@pytest.mark.parametrize(
    ("profile", "maximum"),
    [("fast", 2), ("balanced", 3), ("high_recall", 5), ("evaluation", 3)],
)
def test_profile_caps_total_subqueries(profile: str, maximum: int) -> None:
    plan = analyze_query(
        "adaptive evidence retrieval for clinical question answering 2021-2024",
        run_profile=profile,
        query_planning_policy="facet_balanced",
        explicit_constraints=_explicit_constraints(),
    )

    assert len(plan.subqueries) <= maximum
    assert plan.query_planning.skipped_by_budget_count >= 0


def test_facet_planner_does_not_generate_cartesian_product() -> None:
    plan = analyze_query(
        "adaptive evidence retrieval for clinical question answering 2021-2024",
        run_profile="high_recall",
        query_planning_policy="facet_balanced",
        explicit_constraints=_explicit_constraints(),
    )

    assert len(plan.subqueries) <= 5
    assert all(len(item.facet_types) <= 2 for item in plan.subqueries[1:])


def test_similar_same_facet_queries_merge_provenance_deterministically() -> None:
    analysis = QueryAnalysis(
        original_query="protein ranking",
        constraints=QueryConstraint(methods=["ranking"]),
    )

    queries, result = plan_facet_balanced(
        analysis,
        selected_sources=["arxiv"],
        max_subqueries=5,
    )

    assert len(queries) == 1
    assert result.duplicate_subquery_count == 2
    assert set(queries[0].facet_types) == {"topic", "method", "task"}
    assert queries[0].provenance == [
        "original_query",
        "topic:rules",
        "method:rules",
        "task:rules",
    ]


def test_different_facets_are_not_merged_only_for_token_overlap() -> None:
    plan = analyze_query(
        "protein representation learning",
        run_profile="high_recall",
        query_planning_policy="facet_balanced",
        explicit_constraints=QueryConstraint(
            methods=["contrastive learning"],
            datasets=["ProteinSet"],
            explicit_fields=["methods", "datasets"],
        ),
    )

    purposes = {item.purpose for item in plan.subqueries}
    assert "facet_method" in purposes
    assert "facet_dataset" in purposes


def test_planner_keeps_source_adapter_boundary_and_has_no_gold_input() -> None:
    signature = inspect.signature(plan_facet_balanced)
    source = inspect.getsource(plan_facet_balanced).casefold()

    assert "gold" not in signature.parameters
    assert "arxiv" not in source
    assert "openalex" not in source
    assert "semantic_scholar" not in source
    assert "pubmed" not in source


def test_facet_planning_is_stable() -> None:
    kwargs = {
        "query_planning_policy": "facet_balanced",
        "explicit_constraints": _explicit_constraints(),
    }

    first = analyze_query("adaptive evidence retrieval", **kwargs)  # type: ignore[arg-type]
    second = analyze_query("adaptive evidence retrieval", **kwargs)  # type: ignore[arg-type]

    assert first.model_dump() == second.model_dump()
