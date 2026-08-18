from __future__ import annotations

from scholar_agent.agents.query_evolution import evolve_queries
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvolvedSubquery,
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    QueryEvolutionOptions,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
)


SEED_EXPANSION = QueryEvolutionOptions(policy="seed_expansion")


def make_query_analysis(intent: str = "recent_progress") -> QueryAnalysis:
    return QueryAnalysis(
        original_query="latest LLM reranking methods for scientific literature retrieval",
        language="en",
        intent=intent,
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["reranking"],
            datasets=[],
            domains=["machine_learning"],
            must_include_terms=["LLM", "reranking", "retrieval"],
        ),
    )


def make_search_plan(query_analysis: QueryAnalysis | None = None) -> SearchPlan:
    analysis = query_analysis or make_query_analysis()
    return SearchPlan(
        query_analysis=analysis,
        subqueries=[
            SearchSubquery(
                query=analysis.original_query,
                source_hints=["openalex", "arxiv"],
                priority=1,
                purpose="original_query",
            )
        ],
        selected_sources=["openalex", "arxiv"],
        limit_per_source=20,
        top_k=10,
        run_profile="balanced",
        enable_query_evolution=True,
    )


def make_paper(
    title: str,
    *,
    identifier_suffix: str,
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=2025,
        venue="SIGIR",
        abstract="A paper about LLM reranking and scientific literature retrieval.",
        identifiers=PaperIdentifiers(doi=f"10.123/{identifier_suffix}"),
        sources=sources or ["openalex"],
        citation_count=10,
    )


def make_judgement(
    paper: Paper,
    *,
    score: float = 0.82,
    category: str = "highly_relevant",
    matched_terms: list[str] | None = None,
) -> JudgementResult:
    return JudgementResult(
        paper=paper,
        score=score,
        category=category,
        reasoning="metadata judgement",
        evidence=[EvidenceItem(source="title", text=paper.title, confidence=0.9)],
        matched_terms=matched_terms or ["LLM", "reranking", "retrieval"],
        warnings=[],
    )


def make_ranked(
    judgement: JudgementResult,
    *,
    rank: int = 1,
    final_score: float = 0.8,
) -> RankedPaper:
    return RankedPaper(
        rank=rank,
        paper=judgement.paper,
        final_score=final_score,
        category=judgement.category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=judgement.score,
            authority_score=0.5,
            timeliness_score=0.8,
            metadata_score=0.9,
            final_score=final_score,
            relevance_weight=0.65,
            authority_weight=0.08,
            timeliness_weight=0.22,
            metadata_weight=0.05,
        ),
        ranking_reason="metadata ranking",
        evidence=judgement.evidence,
        matched_terms=judgement.matched_terms,
        warnings=[],
    )


def test_relevant_seed_generates_evolved_queries() -> None:
    query_analysis = make_query_analysis()
    search_plan = make_search_plan(query_analysis)
    judgement = make_judgement(
        make_paper(
            "LLM Reranking for Scientific Literature Retrieval",
            identifier_suffix="relevant",
        )
    )

    record = evolve_queries(
        query_analysis,
        search_plan,
        [judgement],
        [make_ranked(judgement)],
        used_queries=set(),
        options=SEED_EXPANSION,
    )

    assert record.seed_count == 1
    assert record.generated_queries
    assert record.generated_queries[0].generated_by == "rules"
    assert "LLM" in record.generated_queries[0].query
    assert record.warnings == []


def test_no_relevant_seed_returns_warning() -> None:
    query_analysis = make_query_analysis()
    search_plan = make_search_plan(query_analysis)
    irrelevant = make_judgement(
        make_paper("Crystal Growth in Volcanic Rocks", identifier_suffix="bad"),
        score=0.1,
        category="irrelevant",
        matched_terms=[],
    )
    insufficient = make_judgement(
        make_paper("", identifier_suffix="empty"),
        score=0.0,
        category="insufficient_evidence",
        matched_terms=[],
    )

    record = evolve_queries(
        query_analysis,
        search_plan,
        [irrelevant, insufficient],
        [make_ranked(irrelevant), make_ranked(insufficient)],
        used_queries=set(),
        options=SEED_EXPANSION,
    )

    assert record.generated_queries == []
    assert record.seed_count == 0
    assert "no_relevant_seed" in record.warnings


def test_used_queries_are_deduplicated() -> None:
    query_analysis = make_query_analysis()
    search_plan = make_search_plan(query_analysis)
    judgement = make_judgement(
        make_paper("LLM Reranking for Retrieval", identifier_suffix="dedupe")
    )
    used_queries = {
        "LLM reranking retrieval recent advances",
        "LLM reranking retrieval",
    }

    record = evolve_queries(
        query_analysis,
        search_plan,
        [judgement],
        [make_ranked(judgement)],
        used_queries=used_queries,
        options=SEED_EXPANSION,
    )

    normalized_used = {query.casefold() for query in used_queries}
    assert record.generated_queries
    assert all(
        query.query.casefold() not in normalized_used
        for query in record.generated_queries
    )


def test_max_evolved_queries_is_respected() -> None:
    query_analysis = make_query_analysis()
    search_plan = make_search_plan(query_analysis)
    first = make_judgement(
        make_paper("LLM Reranking for Retrieval", identifier_suffix="first")
    )
    second = make_judgement(
        make_paper("Neural Reranking for Academic Search", identifier_suffix="second")
    )

    record = evolve_queries(
        query_analysis,
        search_plan,
        [first, second],
        [make_ranked(first, rank=1), make_ranked(second, rank=2)],
        used_queries=set(),
        options=QueryEvolutionOptions(
            policy="seed_expansion",
            max_evolved_queries=1,
        ),
    )

    assert len(record.generated_queries) == 1


def test_source_hints_only_include_supported_sources() -> None:
    query_analysis = make_query_analysis()
    search_plan = make_search_plan(query_analysis)
    judgement = make_judgement(
        make_paper("LLM Reranking for Retrieval", identifier_suffix="sources")
    )

    record = evolve_queries(
        query_analysis,
        search_plan,
        [judgement],
        [make_ranked(judgement)],
        used_queries=set(),
        options=SEED_EXPANSION,
    )

    assert record.generated_queries
    for subquery in record.generated_queries:
        assert isinstance(subquery, EvolvedSubquery)
        assert set(subquery.source_hints) <= {"openalex", "arxiv"}


def test_output_is_stable_without_llm_or_network() -> None:
    query_analysis = make_query_analysis(intent="survey")
    search_plan = make_search_plan(query_analysis)
    judgement = make_judgement(
        make_paper("LLM Reranking Survey for Retrieval", identifier_suffix="stable")
    )
    ranked = [make_ranked(judgement)]

    first = evolve_queries(
        query_analysis,
        search_plan,
        [judgement],
        ranked,
        used_queries=set(),
        options=SEED_EXPANSION,
    )
    second = evolve_queries(
        query_analysis,
        search_plan,
        [judgement],
        ranked,
        used_queries=set(),
        options=SEED_EXPANSION,
    )

    assert first.model_dump() == second.model_dump()
    assert first.latency_seconds == 0.0
