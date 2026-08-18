from __future__ import annotations

import pytest

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.agents.rrf_fusion import (
    RetrievalListEntry,
    RetrievalRankedList,
    build_retrieval_ranked_lists,
    fuse_ranked_papers,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.pipeline_diagnostics import PipelineDiagnosticsCollector
from scholar_agent.core.search_schemas import RankedPaper, RerankScoreBreakdown
from scholar_agent.core.search_schemas import JudgementResult, QueryAnalysis
from scholar_agent.services.search_service import _rerank_all_and_top


def _paper(
    title: str,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    year: int = 2020,
) -> Paper:
    return Paper(
        title=title,
        authors=["Ada Lovelace"],
        year=year,
        identifiers=PaperIdentifiers(doi=doi, arxiv_id=arxiv_id),
        sources=["arxiv"],
    )


def _ranked(paper: Paper, rank: int, score: float = 0.5) -> RankedPaper:
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=score,
        category="partially_relevant",
        score_breakdown=RerankScoreBreakdown(
            relevance_score=score,
            authority_score=0.0,
            timeliness_score=0.0,
            metadata_score=0.0,
            final_score=score,
            relevance_weight=1.0,
            authority_weight=0.0,
            timeliness_weight=0.0,
            metadata_weight=0.0,
        ),
        ranking_reason="fixture",
    )


def _list(
    source: str,
    query: str,
    entries: list[tuple[Paper, int]],
) -> RetrievalRankedList:
    return RetrievalRankedList(
        source=source,
        subquery=query,
        entries=tuple(
            RetrievalListEntry(paper=paper, rank=rank) for paper, rank in entries
        ),
    )


def test_rrf_sums_cross_list_contributions_and_uses_best_same_list_rank() -> None:
    first = _paper("First", doi="10.1/first")
    second = _paper("Second", doi="10.1/second")

    fused = fuse_ranked_papers(
        [_ranked(second, 1, 0.9), _ranked(first, 2, 0.4)],
        [
            _list("arxiv", "q1", [(first, 5), (first, 2), (second, 1)]),
            _list("openalex", "q2", [(first, 1)]),
        ],
    )

    assert [item.paper.title for item in fused] == ["First", "Second"]
    assert [item.rank for item in fused[0].rrf_contributions] == [2, 1]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].original_rank == 2
    assert fused[0].rrf_top_20_change == "retained_top_20"
    assert "contributing_lists=2" in (fused[0].rrf_rank_change_reason or "")


def test_rrf_uses_unified_identity_and_rejects_missing_or_illegal_ranks() -> None:
    candidate = _paper("Unicode—Title", doi="10.1/shared")
    same_identity = _paper("Different metadata", doi="https://doi.org/10.1/SHARED")
    fused = fuse_ranked_papers(
        [_ranked(candidate, 1)],
        [_list("openalex", "q", [(same_identity, 3)])],
    )
    assert fused[0].rrf_contributions[0].rank == 3

    with pytest.raises(ValueError, match="rrf_provenance_incomplete"):
        fuse_ranked_papers(
            [_ranked(candidate, 1)],
            [_list("openalex", "q", [(same_identity, 0)])],
        )


def test_rrf_ties_are_input_order_independent_and_use_current_score() -> None:
    alpha = _paper("Alpha", doi="10.1/alpha")
    beta = _paper("Beta", doi="10.1/beta")
    lists = [_list("arxiv", "q", [(alpha, 1), (beta, 1)])]

    forward = fuse_ranked_papers(
        [_ranked(alpha, 2, 0.4), _ranked(beta, 1, 0.8)], lists
    )
    reverse = fuse_ranked_papers(
        [_ranked(beta, 1, 0.8), _ranked(alpha, 2, 0.4)], lists
    )

    assert [item.paper.title for item in forward] == ["Beta", "Alpha"]
    assert [item.paper.title for item in reverse] == ["Beta", "Alpha"]


def test_build_lists_keeps_distinct_queries_and_merges_run_dedupe_observation() -> None:
    first = _paper("First", doi="10.1/first")
    second = _paper("Second", doi="10.1/second")
    outputs = [
        RetrievalOutput(
            query="logical one",
            requested_sources=["arxiv"],
            raw_count=1,
            deduplicated_count=1,
            papers=[first],
            source_stats=[
                SourceStats(
                    source="arxiv",
                    adapted_query="shared",
                    diagnostic_papers=[first],
                ),
                SourceStats(
                    source="arxiv",
                    adapted_query="other",
                    diagnostic_papers=[second],
                ),
                SourceStats(
                    source="pubmed",
                    adapted_query="shared",
                    terminal_status="source_failure",
                    error_message="fixture_failure",
                ),
            ],
        ),
        RetrievalOutput(
            query="logical two",
            requested_sources=["arxiv"],
            raw_count=1,
            deduplicated_count=1,
            papers=[first],
            source_stats=[
                SourceStats(
                    source="arxiv",
                    adapted_query="shared",
                    run_dedupe_hit=True,
                    diagnostic_papers=[first],
                )
            ],
        ),
    ]

    ranked_lists = build_retrieval_ranked_lists(outputs)

    assert [(item.source, item.subquery) for item in ranked_lists] == [
        ("arxiv", "shared"),
        ("arxiv", "other"),
        ("pubmed", "shared"),
    ]
    assert [entry.rank for entry in ranked_lists[0].entries] == [1, 1]
    assert ranked_lists[2].entries == ()

    collector = PipelineDiagnosticsCollector(True)
    collector.register_retrieval(
        "initial_retrieval",
        outputs[:1],
        origin_kind_by_query={"logical one": "initial_query"},
    )
    assert [
        item.source_rank
        for item in collector.snapshots[0].candidates[0].provenance
        if item.source == "arxiv" and item.adapted_query == "shared"
    ] == [1]


def test_default_ranking_policy_preserves_current_rules_output() -> None:
    papers = [
        _paper("Lower", doi="10.1/lower"),
        _paper("Higher", doi="10.1/higher"),
    ]
    judgements = [
        JudgementResult(
            paper=papers[0],
            score=0.2,
            category="weakly_relevant",
            reasoning="fixture",
        ),
        JudgementResult(
            paper=papers[1],
            score=0.8,
            category="highly_relevant",
            reasoning="fixture",
        ),
    ]

    all_ranked, top = _rerank_all_and_top(QueryAnalysis(original_query="q"), judgements, 1)

    assert [item.paper.title for item in all_ranked] == ["Higher", "Lower"]
    assert [item.paper.title for item in top] == ["Higher"]
    assert all(item.rrf_score is None for item in all_ranked)
