from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.agents.synthesis import synthesize_answer  # noqa: E402
from scholar_agent.agents.retriever import SourceStats  # noqa: E402
from scholar_agent.app.main import app  # noqa: E402
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers  # noqa: E402
from scholar_agent.core.search_schemas import (  # noqa: E402
    EvolvedSubquery,
    EvidenceItem,
    QueryAnalysis,
    QueryConstraint,
    QueryEvolutionRecord,
    RankedPaper,
    RefChainOutput,
    RefChainRecord,
    RefChainSeed,
    ReferenceEdge,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
)
from scholar_agent.services.search_service import SearchServiceOutput  # noqa: E402


client = TestClient(app)


def test_internal_search_preview_maps_search_service_output(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delenv("REAL_PREVIEW_MAX_WORKERS", raising=False)

    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            captured["max_workers"] = kwargs.get("max_workers")

        def run_search(
            self,
            query: str,
            top_k: int = 20,
            run_profile: str = "balanced",
            enable_refchain: bool = False,
                enable_query_evolution: bool = False,
                query_evolution_policy: str = "coverage_gap",
                query_planning_policy: str = "current_rules",
            enable_llm_query_understanding: bool | None = None,
            enable_llm_judgement: bool | None = None,
            current_year: int | None = None,
        ) -> SearchServiceOutput:
            captured.update(
                {
                    "query": query,
                    "top_k": top_k,
                    "run_profile": run_profile,
                    "enable_refchain": enable_refchain,
                    "enable_query_evolution": enable_query_evolution,
                        "query_evolution_policy": query_evolution_policy,
                        "query_planning_policy": query_planning_policy,
                    "enable_llm_query_understanding": enable_llm_query_understanding,
                    "enable_llm_judgement": enable_llm_judgement,
                    "current_year": current_year,
                }
            )
            return _fake_output(query, top_k)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 3,
            "run_profile": "high_recall",
            "enable_refchain": False,
            "enable_query_evolution": False,
        "query_evolution_policy": "coverage_gap",
        "query_planning_policy": "current_rules",
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert captured == {
        "query": "latest LLM reranking retrieval papers",
        "top_k": 3,
        "run_profile": "high_recall",
            "enable_refchain": False,
        "enable_query_evolution": False,
        "query_evolution_policy": "coverage_gap",
        "query_planning_policy": "current_rules",
        "enable_llm_query_understanding": None,
            "enable_llm_judgement": None,
            "current_year": 2026,
            "max_workers": 2,
        }
    assert body["query_analysis"]["original_query"] == captured["query"]
    assert body["search_plan"]["selected_sources"] == ["openalex", "arxiv"]
    assert body["query_evolution_records"] == []
    assert body["refchain_output"] is None
    assert body["synthesis_output"] is not None
    assert body["synthesis_output"]["evidence_table"][0]["citation_key"] == "R1"
    assert body["ranked_papers"][0]["rank"] == 1
    assert body["ranked_papers"][0]["paper"]["title"] == "Preview Paper"
    assert body["raw_count"] == 4
    assert body["deduplicated_count"] == 2
    assert body["warnings"] == ["mock_warning"]
    assert body["source_stats"][0]["source"] == "openalex"
    assert body["latency_seconds"] == 0.123


def test_internal_search_preview_includes_query_evolution_records(monkeypatch) -> None:
    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(
            self,
            query: str,
            top_k: int = 20,
            run_profile: str = "balanced",
            enable_refchain: bool = False,
            enable_query_evolution: bool = False,
            query_evolution_policy: str = "coverage_gap",
            query_planning_policy: str = "current_rules",
            enable_llm_query_understanding: bool | None = None,
            enable_llm_judgement: bool | None = None,
            current_year: int | None = None,
        ) -> SearchServiceOutput:
            assert enable_query_evolution is True
            assert query_evolution_policy == "coverage_gap"
            assert query_planning_policy == "current_rules"
            assert enable_refchain is False
            assert enable_llm_query_understanding is None
            assert enable_llm_judgement is None
            return _fake_output(query, top_k, include_query_evolution=True)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 3,
            "enable_query_evolution": True,
            "enable_refchain": False,
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["query_evolution_records"]) == 1
    record = body["query_evolution_records"][0]
    assert record["seed_count"] == 1
    assert record["generated_queries"][0]["query"] == "LLM reranking recent advances"
    assert record["generated_queries"][0]["source_hints"] == ["openalex", "arxiv"]
    assert body["refchain_output"] is None


def test_internal_search_preview_includes_refchain_output(monkeypatch) -> None:
    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(
            self,
            query: str,
            top_k: int = 20,
            run_profile: str = "balanced",
            enable_refchain: bool = False,
            enable_query_evolution: bool = False,
            query_evolution_policy: str = "coverage_gap",
            query_planning_policy: str = "current_rules",
            enable_llm_query_understanding: bool | None = None,
            enable_llm_judgement: bool | None = None,
            current_year: int | None = None,
        ) -> SearchServiceOutput:
            assert enable_refchain is True
            assert enable_llm_query_understanding is None
            assert enable_llm_judgement is None
            return _fake_output(query, top_k, include_refchain=True)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview",
        json={
            "query": "LLM reranking retrieval papers",
            "top_k": 3,
            "enable_refchain": True,
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query_evolution_records"] == []
    assert body["refchain_output"] is not None
    assert body["refchain_output"]["references"][0]["title"] == "Preview Reference"
    assert (
        body["refchain_output"]["reference_edges"][0]["seed_paper_id"]
        == "openalex:wpreview"
    )
    assert body["refchain_output"]["record"]["returned_reference_count"] == 1


def test_internal_search_preview_returns_400_for_service_value_error(monkeypatch) -> None:
    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, *args, **kwargs) -> SearchServiceOutput:
            raise ValueError("query must not be empty")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview",
        json={
            "query": " ",
            "top_k": 3,
            "run_profile": "balanced",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query must not be empty"


def test_internal_search_preview_uses_real_preview_max_workers_env(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            captured["max_workers"] = kwargs.get("max_workers")

        def run_search(
            self,
            query: str,
            top_k: int = 20,
            run_profile: str = "balanced",
            enable_refchain: bool = False,
            enable_query_evolution: bool = False,
            query_evolution_policy: str = "coverage_gap",
            query_planning_policy: str = "current_rules",
            enable_llm_query_understanding: bool | None = None,
            enable_llm_judgement: bool | None = None,
            current_year: int | None = None,
        ) -> SearchServiceOutput:
            return _fake_output(query, top_k)

    monkeypatch.setenv("REAL_PREVIEW_MAX_WORKERS", "1")
    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 3,
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    assert captured["max_workers"] == 1


def test_legacy_mock_search_runs_endpoint_is_removed(monkeypatch) -> None:
    class FailingSearchService:
        def run_search(self, *args, **kwargs) -> SearchServiceOutput:
            raise AssertionError("legacy mock run endpoint must not call SearchService")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FailingSearchService)

    response = client.post(
        "/api/v1/search/runs",
        json={"query": "请帮我搜索关于 LLM reranking 的代表性论文"},
    )

    assert response.status_code in {404, 405}


def _fake_output(
    query: str,
    top_k: int,
    *,
    include_query_evolution: bool = False,
    include_refchain: bool = False,
) -> SearchServiceOutput:
    query_analysis = QueryAnalysis(
        original_query=query,
        language="en",
        intent="recent_progress",
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["reranking"],
            must_include_terms=["LLM", "retrieval"],
        ),
    )
    search_plan = SearchPlan(
        query_analysis=query_analysis,
        subqueries=[
            SearchSubquery(
                query=query,
                source_hints=["openalex", "arxiv"],
                priority=1,
                purpose="original_query",
            )
        ],
        selected_sources=["openalex", "arxiv"],
        limit_per_source=20,
        top_k=top_k,
        run_profile="high_recall",
        warnings=["mock_warning"],
    )
    paper = Paper(
        title="Preview Paper",
        authors=["Alice"],
        year=2024,
        venue="ACL",
        abstract="A preview paper about LLM reranking and retrieval.",
        identifiers=PaperIdentifiers(doi="10.123/preview", openalex_id="WPREVIEW"),
        sources=["openalex"],
        citation_count=12,
    )
    ranked_paper = RankedPaper(
        rank=1,
        paper=paper,
        final_score=0.88,
        category="highly_relevant",
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.9,
            authority_score=0.5,
            timeliness_score=0.8,
            metadata_score=0.9,
            final_score=0.88,
            relevance_weight=0.65,
            authority_weight=0.08,
            timeliness_weight=0.22,
            metadata_weight=0.05,
        ),
        ranking_reason="metadata-only preview ranking",
        evidence=[
            EvidenceItem(source="title", text="Preview Paper", confidence=0.9),
        ],
        matched_terms=["LLM", "reranking"],
    )
    query_evolution_records = []
    if include_query_evolution:
        query_evolution_records.append(
            QueryEvolutionRecord(
                seed_count=1,
                generated_queries=[
                    EvolvedSubquery(
                        query="LLM reranking recent advances",
                        source_hints=["openalex", "arxiv"],
                        priority=1,
                        purpose="query_evolution_recent_progress",
                        seed_paper_titles=["Preview Paper"],
                    )
                ],
            )
        )

    refchain_output = None
    if include_refchain:
        reference = Paper(
            title="Preview Reference",
            authors=["Bob"],
            year=2022,
            venue="SIGIR",
            abstract="A reference paper about LLM reranking and retrieval.",
            identifiers=PaperIdentifiers(doi="10.123/reference", openalex_id="WREF"),
            sources=["openalex"],
            citation_count=100,
        )
        edge = ReferenceEdge(
            seed_paper_id="openalex:wpreview",
            reference_paper_id="openalex:wref",
            source="openalex",
        )
        record = RefChainRecord(
            seeds=[
                RefChainSeed(
                    paper=paper,
                    rank=1,
                    score=0.88,
                    reason="metadata-only preview ranking",
                )
            ],
            reference_edges=[edge],
            raw_reference_count=1,
            returned_reference_count=1,
        )
        refchain_output = RefChainOutput(
            references=[reference],
            reference_edges=[edge],
            record=record,
        )

    output = SearchServiceOutput(
        search_plan=search_plan,
        query_evolution_records=query_evolution_records,
        refchain_output=refchain_output,
        raw_count=4,
        deduplicated_count=2,
        ranked_papers=[ranked_paper],
        warnings=["mock_warning"],
        source_stats=[
            SourceStats(
                source="openalex",
                returned_count=2,
                latency_seconds=0.02,
            )
        ],
        latency_seconds=0.123,
    )
    output.synthesis_output = synthesize_answer(output)
    return output
