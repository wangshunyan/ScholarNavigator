from __future__ import annotations

from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    TimeRange,
)


def make_query_analysis(
    *,
    intent: str = "general",
    time_range: TimeRange | None = None,
) -> QueryAnalysis:
    return QueryAnalysis(
        original_query="LLM reranking for scientific literature retrieval",
        language="en",
        intent=intent,
        domain="machine_learning",
        constraints=QueryConstraint(
            time_range=time_range,
            methods=["reranking"],
            must_include_terms=["LLM", "retrieval"],
        ),
    )


def make_paper(
    title: str,
    *,
    year: int | None = 2024,
    citation_count: int = 0,
    sources: list[str] | None = None,
    venue: str | None = "ACL",
    identifiers: PaperIdentifiers | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=year,
        venue=venue,
        abstract="A paper about LLM reranking and retrieval.",
        identifiers=identifiers
        or PaperIdentifiers(doi=f"10.123/{title.casefold().replace(' ', '-')}"),
        sources=sources or ["openalex"],
        citation_count=citation_count,
    )


def make_judgement(
    paper: Paper,
    *,
    score: float,
    category: str,
) -> JudgementResult:
    return JudgementResult(
        paper=paper,
        score=score,
        category=category,
        reasoning="metadata judgement",
        evidence=[
            EvidenceItem(source="title", text=paper.title, confidence=0.9),
        ],
        matched_terms=["LLM", "reranking"],
        warnings=[],
    )


def test_relevance_dominates_sorting() -> None:
    query_analysis = make_query_analysis()
    highly_relevant = make_judgement(
        make_paper("Highly Relevant Low Citation", citation_count=0),
        score=0.82,
        category="highly_relevant",
    )
    less_relevant = make_judgement(
        make_paper("Partial High Citation", citation_count=500),
        score=0.55,
        category="partially_relevant",
    )

    ranked = rerank_papers(query_analysis, [less_relevant, highly_relevant])

    assert ranked[0].paper.title == "Highly Relevant Low Citation"
    assert ranked[0].score_breakdown.relevance_score == 0.82
    assert ranked[0].score_breakdown.category_multiplier == 1.0
    assert ranked[1].score_breakdown.category_multiplier == 0.92


def test_high_citation_irrelevant_cannot_outrank_highly_relevant() -> None:
    query_analysis = make_query_analysis()
    irrelevant = make_judgement(
        make_paper("Irrelevant Famous Paper", citation_count=100_000),
        score=0.1,
        category="irrelevant",
    )
    relevant = make_judgement(
        make_paper("Relevant Paper", citation_count=0),
        score=0.78,
        category="highly_relevant",
    )

    ranked = rerank_papers(query_analysis, [irrelevant, relevant])

    assert ranked[0].paper.title == "Relevant Paper"
    assert ranked[-1].category == "irrelevant"


def test_recent_query_rewards_newer_paper() -> None:
    query_analysis = make_query_analysis(
        intent="recent_progress",
        time_range=TimeRange(start_year=2020, end_year=2026),
    )
    older = make_judgement(
        make_paper("Older Relevant Paper", year=2020),
        score=0.74,
        category="highly_relevant",
    )
    newer = make_judgement(
        make_paper("Newer Relevant Paper", year=2025),
        score=0.74,
        category="highly_relevant",
    )

    ranked = rerank_papers(query_analysis, [older, newer])

    assert ranked[0].paper.title == "Newer Relevant Paper"
    assert (
        ranked[0].score_breakdown.timeliness_score
        > ranked[1].score_breakdown.timeliness_score
    )
    assert ranked[0].score_breakdown.timeliness_weight == 0.22


def test_survey_query_rewards_authority() -> None:
    query_analysis = make_query_analysis(intent="survey")
    low_citation = make_judgement(
        make_paper("Low Citation Survey Candidate", citation_count=1),
        score=0.76,
        category="highly_relevant",
    )
    high_citation = make_judgement(
        make_paper("High Citation Survey Candidate", citation_count=800),
        score=0.76,
        category="highly_relevant",
    )

    ranked = rerank_papers(query_analysis, [low_citation, high_citation])

    assert ranked[0].paper.title == "High Citation Survey Candidate"
    assert ranked[0].score_breakdown.authority_weight == 0.25
    assert (
        ranked[0].score_breakdown.authority_score
        > ranked[1].score_breakdown.authority_score
    )


def test_multi_source_and_identifier_completeness_add_bonus() -> None:
    query_analysis = make_query_analysis()
    sparse = make_judgement(
        make_paper(
            "Sparse Metadata",
            sources=["openalex"],
            identifiers=PaperIdentifiers(),
        ),
        score=0.76,
        category="highly_relevant",
    )
    complete = make_judgement(
        make_paper(
            "Complete Metadata",
            sources=["openalex", "arxiv"],
            identifiers=PaperIdentifiers(
                doi="10.123/complete",
                arxiv_id="2401.00001",
                semantic_scholar_id="S2",
                openalex_id="W1",
            ),
        ),
        score=0.76,
        category="highly_relevant",
    )

    ranked = rerank_papers(query_analysis, [sparse, complete])

    assert ranked[0].paper.title == "Complete Metadata"
    assert (
        ranked[0].score_breakdown.authority_score
        > ranked[1].score_breakdown.authority_score
    )
    assert (
        ranked[0].score_breakdown.metadata_score
        > ranked[1].score_breakdown.metadata_score
    )


def test_top_k_rank_score_range_and_tie_breaker_are_stable() -> None:
    query_analysis = make_query_analysis()
    papers = [
        make_judgement(make_paper("Gamma Paper"), score=0.76, category="highly_relevant"),
        make_judgement(make_paper("Alpha Paper"), score=0.76, category="highly_relevant"),
        make_judgement(make_paper("Beta Paper"), score=0.76, category="highly_relevant"),
    ]

    first_run = rerank_papers(query_analysis, papers, top_k=2)
    second_run = rerank_papers(query_analysis, papers, top_k=2)

    assert [item.rank for item in first_run] == [1, 2]
    assert [item.paper.title for item in first_run] == ["Alpha Paper", "Beta Paper"]
    assert [item.paper.title for item in second_run] == ["Alpha Paper", "Beta Paper"]
    assert all(0 <= item.final_score <= 1 for item in first_run)
    assert all(item.ranking_reason for item in first_run)


def test_reranker_does_not_promote_candidate_by_source_name() -> None:
    query_analysis = make_query_analysis()
    single_source_s2 = make_judgement(
        make_paper(
            "Single Source Semantic Scholar Partial",
            sources=["semantic_scholar"],
            identifiers=PaperIdentifiers(semantic_scholar_id="S2-CLOSE"),
        ),
        score=0.56,
        category="partially_relevant",
    )
    arxiv_candidate = make_judgement(
        make_paper(
            "ArXiv Partial",
            sources=["arxiv"],
            identifiers=PaperIdentifiers(arxiv_id="2401.00001"),
        ),
        score=0.55,
        category="partially_relevant",
    )

    ranked = rerank_papers(query_analysis, [single_source_s2, arxiv_candidate])

    assert ranked[0].paper.title == "Single Source Semantic Scholar Partial"
    assert ranked[1].paper.title == "ArXiv Partial"
    assert ranked[0].final_score > ranked[1].final_score
    assert ranked[0].final_score - ranked[1].final_score <= 0.02


def test_score_gap_controls_order_regardless_of_source() -> None:
    query_analysis = make_query_analysis()
    single_source_s2 = make_judgement(
        make_paper(
            "Clearly Higher Semantic Scholar Partial",
            sources=["semantic_scholar"],
            identifiers=PaperIdentifiers(semantic_scholar_id="S2-LARGE"),
        ),
        score=0.66,
        category="partially_relevant",
    )
    arxiv_candidate = make_judgement(
        make_paper(
            "Lower ArXiv Partial",
            sources=["arxiv"],
            identifiers=PaperIdentifiers(arxiv_id="2401.00002"),
        ),
        score=0.55,
        category="partially_relevant",
    )

    ranked = rerank_papers(query_analysis, [single_source_s2, arxiv_candidate])

    assert ranked[0].paper.title == "Clearly Higher Semantic Scholar Partial"
    assert ranked[1].paper.title == "Lower ArXiv Partial"
    assert ranked[0].final_score - ranked[1].final_score > 0.02


def test_multi_source_metadata_contributes_without_fixed_source_priority() -> None:
    query_analysis = make_query_analysis()
    multi_source = make_judgement(
        make_paper(
            "Multi Source Semantic Scholar ArXiv Partial",
            sources=["semantic_scholar", "arxiv"],
            identifiers=PaperIdentifiers(
                semantic_scholar_id="S2-MULTI",
                arxiv_id="2401.00003",
            ),
        ),
        score=0.56,
        category="partially_relevant",
    )
    arxiv_candidate = make_judgement(
        make_paper(
            "ArXiv Partial Behind Multi Source",
            sources=["arxiv"],
            identifiers=PaperIdentifiers(arxiv_id="2401.00004"),
        ),
        score=0.55,
        category="partially_relevant",
    )

    ranked = rerank_papers(query_analysis, [multi_source, arxiv_candidate])

    assert ranked[0].paper.title == "Multi Source Semantic Scholar ArXiv Partial"
    assert ranked[1].paper.title == "ArXiv Partial Behind Multi Source"


def test_relevance_category_controls_order_regardless_of_source() -> None:
    query_analysis = make_query_analysis()
    highly_relevant_s2 = make_judgement(
        make_paper(
            "Highly Relevant Semantic Scholar",
            sources=["semantic_scholar"],
            identifiers=PaperIdentifiers(semantic_scholar_id="S2-HIGH"),
        ),
        score=0.76,
        category="highly_relevant",
    )
    arxiv_candidate = make_judgement(
        make_paper(
            "ArXiv Highly Relevant Behind S2",
            sources=["arxiv"],
            identifiers=PaperIdentifiers(arxiv_id="2401.00005"),
        ),
        score=0.75,
        category="highly_relevant",
    )

    ranked = rerank_papers(query_analysis, [highly_relevant_s2, arxiv_candidate])

    assert ranked[0].paper.title == "Highly Relevant Semantic Scholar"
    assert ranked[1].paper.title == "ArXiv Highly Relevant Behind S2"


def test_three_candidates_follow_deterministic_score_order() -> None:
    query_analysis = make_query_analysis()
    single_source_s2 = make_judgement(
        make_paper(
            "Single Source Semantic Scholar Partial",
            sources=["semantic_scholar"],
            identifiers=PaperIdentifiers(semantic_scholar_id="S2-NONADJACENT"),
        ),
        score=0.57,
        category="partially_relevant",
    )
    middle_openalex = make_judgement(
        make_paper(
            "Middle OpenAlex Partial",
            sources=["openalex"],
            identifiers=PaperIdentifiers(openalex_id="W-NONADJACENT"),
        ),
        score=0.565,
        category="partially_relevant",
    )
    arxiv_candidate = make_judgement(
        make_paper(
            "Non Adjacent ArXiv Partial",
            sources=["arxiv"],
            identifiers=PaperIdentifiers(arxiv_id="2401.00006"),
        ),
        score=0.56,
        category="partially_relevant",
    )

    ranked = rerank_papers(
        query_analysis,
        [single_source_s2, middle_openalex, arxiv_candidate],
    )

    assert [item.paper.title for item in ranked] == [
        "Single Source Semantic Scholar Partial",
        "Middle OpenAlex Partial",
        "Non Adjacent ArXiv Partial",
    ]
