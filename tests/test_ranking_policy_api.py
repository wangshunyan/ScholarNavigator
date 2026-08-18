from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.core.api_schemas import SearchRunCreateRequest
from scholar_agent.core.search_schemas import SearchPlan, QueryAnalysis


def test_ranking_policy_defaults_off_in_api_and_internal_plan() -> None:
    request = SearchRunCreateRequest(query="graph retrieval")
    plan = SearchPlan(query_analysis=QueryAnalysis(original_query="graph retrieval"))

    assert request.options.ranking_policy == "current_rules"
    assert plan.ranking_policy == "current_rules"


def test_api_accepts_rrf_and_rejects_unknown_ranking_policy() -> None:
    request = SearchRunCreateRequest.model_validate(
        {"query": "graph retrieval", "options": {"ranking_policy": "rrf_fusion"}}
    )
    assert request.options.ranking_policy == "rrf_fusion"

    with pytest.raises(ValidationError):
        SearchRunCreateRequest.model_validate(
            {"query": "graph retrieval", "options": {"ranking_policy": "unknown"}}
        )
