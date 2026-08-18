from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.core.api_schemas import SearchRunCreateRequest


def test_api_defaults_query_evolution_off_with_coverage_gap_policy() -> None:
    request = SearchRunCreateRequest(query="graph retrieval")

    assert request.options.enable_query_evolution is False
    assert request.options.query_evolution_policy == "coverage_gap"


def test_api_accepts_supported_query_evolution_policy() -> None:
    request = SearchRunCreateRequest.model_validate(
        {
            "query": "graph retrieval",
            "options": {
                "enable_query_evolution": True,
                "query_evolution_policy": "seed_expansion",
            },
        }
    )

    assert request.options.enable_query_evolution is True
    assert request.options.query_evolution_policy == "seed_expansion"


def test_api_rejects_unknown_query_evolution_policy() -> None:
    with pytest.raises(ValidationError):
        SearchRunCreateRequest.model_validate(
            {
                "query": "graph retrieval",
                "options": {"query_evolution_policy": "unknown"},
            }
        )
