from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats  # noqa: E402
from scholar_agent.agents.synthesis import synthesize_answer  # noqa: E402
from scholar_agent.app.main import app  # noqa: E402
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers  # noqa: E402
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics  # noqa: E402
from scholar_agent.core.search_schemas import (  # noqa: E402
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
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


def test_internal_search_preview_api_result_returns_existing_api_shape(
    monkeypatch,
) -> None:
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
            return _fake_output(
                query,
                include_refchain=enable_refchain,
            )

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview/api-result",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 5,
            "run_profile": "high_recall",
            "enable_refchain": True,
            "enable_query_evolution": True,
        "query_evolution_policy": "coverage_gap",
        "query_planning_policy": "current_rules",
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"].startswith("run_internal_")
    assert body["status"] == "succeeded"
    assert body["partial"] is False
    assert body["query_analysis"]["intent_type"] == "recent_progress"
    assert body["search_plan"]["expanded_queries"] == [
        "latest LLM reranking retrieval papers"
    ]
    assert body["highly_relevant_papers"][0]["paper"]["title"] == "Highly Relevant"
    assert body["partially_relevant_papers"][0]["paper"]["title"] == "Partially Relevant"
    assert body["synthesis"] is not None
    assert body["synthesis"]["evidence_table"][0]["citation_key"] == "R1"
    assert body["synthesis"]["citation_coverage"]["ranked_paper_count"] == 2
    assert body["method_clusters"]
    assert body["timeline"]
    assert body["citation_graph"]["edges"][0]["source"] == "openalex:whigh"
    assert "mock_warning" in body["missing_evidence"]
    assert "source_error:openalex:HTTP 503" in body["missing_evidence"]
    assert "refchain:seed_count=1:returned_reference_count=1" in body["missing_evidence"]
    assert body["cost_report"]["llm_call_count"] == 0
    assert body["cost_report"]["cache_hit_count"] == 0
    assert body["cost_report"]["search_api_call_count"] == 2

    assert captured == {
        "query": "latest LLM reranking retrieval papers",
        "top_k": 5,
        "run_profile": "high_recall",
            "enable_refchain": True,
        "enable_query_evolution": True,
        "query_evolution_policy": "coverage_gap",
        "query_planning_policy": "current_rules",
        "enable_llm_query_understanding": None,
            "enable_llm_judgement": None,
            "current_year": 2026,
            "max_workers": 2,
        }


def test_internal_search_preview_api_result_uses_real_preview_max_workers_env(
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
            return _fake_output(query)

    monkeypatch.setenv("REAL_PREVIEW_MAX_WORKERS", "1")
    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/internal/search/preview/api-result",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 5,
            "current_year": 2026,
        },
    )

    assert response.status_code == 200
    assert captured["max_workers"] == 1


def test_legacy_mock_search_runs_api_is_unavailable(monkeypatch) -> None:
    class FailingSearchService:
        def run_search(self, *args, **kwargs) -> SearchServiceOutput:
            raise AssertionError("legacy mock API must not call SearchService")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FailingSearchService)

    create_response = client.post(
        "/api/v1/search/runs",
        json={"query": "请帮我搜索关于 LLM reranking 的代表性论文"},
    )
    status_response = client.get("/api/v1/search/runs/run_missing")
    result_response = client.get("/api/v1/search/runs/run_missing/result")

    assert create_response.status_code in {404, 405}
    assert status_response.status_code in {404, 405}
    assert result_response.status_code in {404, 405}


def _fake_output(query: str, *, include_refchain: bool = False) -> SearchServiceOutput:
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
        top_k=5,
        run_profile="high_recall",
    )
    high = _ranked(
        _paper("Highly Relevant", doi="10.123/high", openalex_id="WHIGH"),
        rank=1,
        category="highly_relevant",
        final_score=0.91,
    )
    partial = _ranked(
        _paper("Partially Relevant", doi="10.123/partial", openalex_id="WPARTIAL"),
        rank=2,
        category="partially_relevant",
        final_score=0.66,
    )
    refchain_output = None
    if include_refchain:
        reference = _paper("RefChain Reference", doi="10.123/ref", openalex_id="WREF")
        edge = ReferenceEdge(
            seed_paper_id="openalex:whigh",
            reference_paper_id="openalex:wref",
            source="openalex",
        )
        refchain_output = RefChainOutput(
            references=[reference],
            reference_edges=[edge],
            record=RefChainRecord(
                seeds=[
                    RefChainSeed(
                        paper=high.paper,
                        rank=1,
                        score=0.91,
                        reason="metadata ranking",
                    )
                ],
                reference_edges=[edge],
                raw_reference_count=1,
                returned_reference_count=1,
            ),
        )

    output = SearchServiceOutput(
        search_plan=search_plan,
        retrieval_outputs=[
            RetrievalOutput(
                query=query,
                requested_sources=["openalex", "arxiv"],
                raw_count=2,
                deduplicated_count=2,
                papers=[high.paper, partial.paper],
                source_stats=[],
                warnings=[],
                latency_seconds=0.01,
            )
        ],
        refchain_output=refchain_output,
        raw_count=2,
        deduplicated_count=2,
        judgements=[
            _judgement(high),
            _judgement(partial),
        ],
        ranked_papers=[high, partial],
        warnings=["mock_warning"],
        source_stats=[
            SourceStats(
                source="openalex",
                returned_count=0,
                latency_seconds=0.1,
                error_message="HTTP 503",
                diagnostics=ConnectorDiagnostics(
                    request_count=1,
                    error_count=1,
                    latency_seconds=0.1,
                ),
            ),
            SourceStats(
                source="arxiv",
                returned_count=2,
                latency_seconds=0.1,
                diagnostics=ConnectorDiagnostics(
                    request_count=1,
                    latency_seconds=0.1,
                ),
            ),
        ],
        latency_seconds=0.25,
    )
    output.synthesis_output = synthesize_answer(output)
    return output


def _paper(title: str, *, doi: str, openalex_id: str) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=2025,
        venue="ACL",
        abstract="A paper about LLM reranking and scientific literature retrieval.",
        identifiers=PaperIdentifiers(doi=doi, openalex_id=openalex_id),
        sources=["fixture"],
        citation_count=12,
    )


def _ranked(
    paper: Paper,
    *,
    rank: int,
    category: str,
    final_score: float,
) -> RankedPaper:
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=final_score,
        category=category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=final_score,
            authority_score=0.5,
            timeliness_score=0.7,
            metadata_score=0.8,
            final_score=final_score,
            relevance_weight=0.65,
            authority_weight=0.1,
            timeliness_weight=0.2,
            metadata_weight=0.05,
        ),
        ranking_reason="metadata ranking",
        evidence=[EvidenceItem(source="title", text=paper.title, confidence=0.9)],
        matched_terms=["LLM", "retrieval", "reranking"],
    )


def _judgement(ranked: RankedPaper) -> JudgementResult:
    return JudgementResult(
        paper=ranked.paper,
        score=ranked.final_score,
        category=ranked.category,
        reasoning="metadata judgement",
        evidence=ranked.evidence,
        matched_terms=ranked.matched_terms,
    )
