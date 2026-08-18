from __future__ import annotations

from scholar_agent.agents.synthesis import synthesize_answer
from scholar_agent.agents.retriever import SourceStats
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    QueryAnalysis,
    QueryConstraint,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
)
from scholar_agent.core.synthesis_schemas import SynthesisOptions
from scholar_agent.services.search_service import SearchServiceOutput


def make_query_analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="latest LLM reranking methods for scientific literature retrieval",
        language="en",
        intent="recent_progress",
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["reranking"],
            must_include_terms=["LLM", "retrieval"],
        ),
    )


def make_search_output(
    ranked_papers: list[RankedPaper],
    *,
    warnings: list[str] | None = None,
    source_stats: list[SourceStats] | None = None,
) -> SearchServiceOutput:
    query_analysis = make_query_analysis()
    return SearchServiceOutput(
        search_plan=SearchPlan(query_analysis=query_analysis, top_k=10),
        ranked_papers=ranked_papers,
        warnings=warnings or [],
        source_stats=source_stats or [],
        raw_count=len(ranked_papers),
        deduplicated_count=len(ranked_papers),
        latency_seconds=0.01,
    )


def make_ranked_paper(
    title: str,
    *,
    rank: int = 1,
    category: str = "highly_relevant",
    final_score: float = 0.86,
    evidence: list[EvidenceItem] | None = None,
    matched_terms: list[str] | None = None,
) -> RankedPaper:
    paper = Paper(
        title=title,
        authors=["Alice", "Bob"],
        year=2025,
        venue="SIGIR",
        abstract="This paper studies LLM reranking for scientific literature retrieval.",
        identifiers=PaperIdentifiers(
            doi=f"10.123/{rank}",
            arxiv_id=f"2501.0000{rank}",
            openalex_id=f"W{rank}",
        ),
        sources=["openalex", "arxiv"],
        citation_count=20,
    )
    score = RerankScoreBreakdown(
        relevance_score=final_score,
        authority_score=0.6,
        timeliness_score=0.8,
        metadata_score=0.9,
        final_score=final_score,
        relevance_weight=0.7,
        authority_weight=0.1,
        timeliness_weight=0.1,
        metadata_weight=0.1,
    )
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=final_score,
        category=category,
        score_breakdown=score,
        ranking_reason="Strong metadata match.",
        evidence=evidence
        or [
            EvidenceItem(
                source="title",
                text=title,
                confidence=0.95,
            ),
            EvidenceItem(
                source="abstract",
                text="The abstract discusses LLM reranking for literature retrieval.",
                confidence=0.9,
            ),
        ],
        matched_terms=matched_terms or ["LLM", "reranking", "retrieval"],
    )


def test_no_ranked_papers_returns_insufficient_evidence() -> None:
    output = make_search_output([])

    synthesis = synthesize_answer(output)

    assert synthesis.status == "insufficient_evidence"
    assert "Insufficient evidence" in synthesis.answer_summary
    assert synthesis.evidence_table == []
    assert synthesis.key_findings == []
    assert "insufficient_evidence:no_supported_evidence_rows" in synthesis.limitations
    assert synthesis.citation_coverage.ranked_paper_count == 0


def test_evidence_generates_citation_keys_and_findings() -> None:
    output = make_search_output(
        [
            make_ranked_paper("CoRank: LLM-Based Compact Reranking", rank=1),
            make_ranked_paper("Scientific Paper Retrieval with LLM Ranking", rank=2),
        ]
    )

    synthesis = synthesize_answer(output)

    assert synthesis.status == "succeeded"
    assert [row.citation_key for row in synthesis.evidence_table[:2]] == ["R1", "R1"]
    assert any(row.citation_key == "R2" for row in synthesis.evidence_table)
    assert synthesis.key_findings
    assert "[R1]" in synthesis.answer_summary


def test_every_finding_has_legal_citation_key() -> None:
    output = make_search_output([make_ranked_paper("Relevant Reranking Paper", rank=1)])

    synthesis = synthesize_answer(output)

    valid_keys = {row.citation_key for row in synthesis.evidence_table}
    assert valid_keys
    for finding in synthesis.key_findings:
        assert finding.citation_keys
        assert set(finding.citation_keys).issubset(valid_keys)


def test_source_errors_and_warnings_enter_limitations() -> None:
    output = make_search_output(
        [make_ranked_paper("Relevant Reranking Paper", rank=1)],
        warnings=["mock_warning"],
        source_stats=[
            SourceStats(
                source="openalex",
                returned_count=0,
                latency_seconds=0.01,
                error_message="HTTP Error 503: Service Unavailable",
            )
        ],
    )

    synthesis = synthesize_answer(output)

    assert "mock_warning" in synthesis.limitations
    assert (
        "source_error:openalex:HTTP Error 503: Service Unavailable"
        in synthesis.limitations
    )
    assert synthesis.citation_coverage.source_error_count == 1


def test_citation_coverage_is_computed() -> None:
    paper_with_evidence = make_ranked_paper("Relevant Reranking Paper", rank=1)
    paper_without_supported_evidence = make_ranked_paper(
        "Unsupported Evidence Paper",
        rank=2,
        evidence=[
            EvidenceItem.model_construct(
                source="full_text",
                text="A full text snippet that is not allowed in MVP.",
                confidence=0.9,
            )
        ],
    )
    output = make_search_output([paper_with_evidence, paper_without_supported_evidence])

    synthesis = synthesize_answer(output)

    assert synthesis.citation_coverage.ranked_paper_count == 2
    assert synthesis.citation_coverage.cited_paper_count == 1
    assert synthesis.citation_coverage.evidence_row_count == 2
    assert synthesis.citation_coverage.cited_evidence_row_count == 2
    assert synthesis.citation_coverage.coverage_ratio == 0.5


def test_unsupported_evidence_source_is_filtered() -> None:
    ranked = make_ranked_paper(
        "Mixed Evidence Paper",
        rank=1,
        evidence=[
            EvidenceItem(source="title", text="LLM reranking paper", confidence=0.9),
            EvidenceItem.model_construct(
                source="full_text",
                text="Unsupported full text evidence.",
                confidence=0.9,
            ),
        ],
    )

    synthesis = synthesize_answer(make_search_output([ranked]))

    assert [row.evidence_source for row in synthesis.evidence_table] == ["title"]
    assert "unsupported_evidence_filtered:rank=1:count=1" in synthesis.warnings


def test_metadata_only_evidence_is_marked_as_limitation() -> None:
    ranked = make_ranked_paper(
        "Metadata Evidence Paper",
        rank=1,
        evidence=[
            EvidenceItem(
                source="metadata",
                text="Metadata shows this paper is from SIGIR 2025.",
                confidence=0.8,
            )
        ],
    )

    synthesis = synthesize_answer(make_search_output([ranked]))

    assert (
        "metadata_only_evidence:no_abstract_or_full_text_evidence_used"
        in synthesis.limitations
    )
    assert "full_text_evidence_unavailable" in synthesis.limitations


def test_output_is_stable_and_local(monkeypatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    output = make_search_output([make_ranked_paper("Stable Reranking Paper", rank=1)])

    first = synthesize_answer(output)
    second = synthesize_answer(output)

    assert first.model_dump() == second.model_dump()
    assert first.citation_coverage.source_error_count == 0


def test_options_limit_cited_papers_and_findings() -> None:
    output = make_search_output(
        [
            make_ranked_paper("First Reranking Paper", rank=1),
            make_ranked_paper("Second Reranking Paper", rank=2),
        ]
    )

    synthesis = synthesize_answer(
        output,
        SynthesisOptions(max_cited_papers=1, max_findings=1),
    )

    assert {row.citation_key for row in synthesis.evidence_table} == {"R1"}
    assert len(synthesis.key_findings) == 1
