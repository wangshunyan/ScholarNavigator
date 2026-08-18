from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.core.api_schemas import SearchRunCreateRequest


def test_api_defaults_to_current_rules_planning() -> None:
    request = SearchRunCreateRequest(query="graph retrieval")

    assert request.options.query_planning_policy == "current_rules"


def test_api_accepts_facet_balanced_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "facet_balanced"},
        }
    )

    assert request.options.query_planning_policy == "facet_balanced"


def test_api_accepts_concept_projection_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "concept_projection"},
        }
    )

    assert request.options.query_planning_policy == "concept_projection"


def test_api_accepts_prf_v1_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "prf_v1"},
        }
    )

    assert request.options.query_planning_policy == "prf_v1"


def test_api_accepts_controlled_relaxation_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "controlled_relaxation"},
        }
    )

    assert request.options.query_planning_policy == "controlled_relaxation"


def test_api_accepts_disjunctive_facets_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "disjunctive_facets"},
        }
    )

    assert request.options.query_planning_policy == "disjunctive_facets"


def test_api_accepts_current_plus_disjunctive_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {
                "query_planning_policy": "current_plus_disjunctive"
            },
        }
    )

    assert request.options.query_planning_policy == "current_plus_disjunctive"


def test_api_accepts_facet_union_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "facet_union"},
        }
    )

    assert request.options.query_planning_policy == "facet_union"


def test_api_accepts_llm_semantic_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "llm_semantic"},
        }
    )

    assert request.options.query_planning_policy == "llm_semantic"


def test_api_accepts_llm_constrained_rewrite_planning() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"query_planning_policy": "llm_constrained_rewrite"},
        }
    )

    assert request.options.query_planning_policy == "llm_constrained_rewrite"


def test_api_rejects_unknown_planning_policy() -> None:
    with pytest.raises(ValidationError):
        SearchRunCreateRequest.model_validate(
            {
                "query": "graph retrieval",
                "options": {"query_planning_policy": "unknown"},
            }
        )


def test_api_defaults_to_current_judgement_rules() -> None:
    request = SearchRunCreateRequest(query="graph retrieval")

    assert request.options.judgement_policy == "current_rules"


def test_api_accepts_calibrated_judgement_rules() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {"judgement_policy": "calibrated_rules_v1"},
        }
    )

    assert request.options.judgement_policy == "calibrated_rules_v1"


def test_api_rejects_unknown_judgement_policy() -> None:
    with pytest.raises(ValidationError):
        SearchRunCreateRequest.model_validate(
            {
                "query": "graph retrieval",
                "options": {"judgement_policy": "unknown"},
            }
        )
