from __future__ import annotations

from scholar_agent.agents.pseudo_relevance_feedback import (
    build_prf_plan,
    extract_prf_feedback,
)
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    RankedPaper,
    RerankScoreBreakdown,
)
from scholar_agent.services.search_service import SearchService
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key


QUERY = "graph retrieval"


def _paper(index: int, title: str, abstract: str = "") -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        identifiers=PaperIdentifiers(doi=f"10.1000/{index}"),
        sources=["arxiv"],
    )


def _ranked(paper: Paper, rank: int) -> RankedPaper:
    score = max(0.1, 1.0 - rank / 10)
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=score,
        category="partially_relevant",
        score_breakdown=RerankScoreBreakdown(
            relevance_score=score,
            authority_score=0.0,
            timeliness_score=0.0,
            metadata_score=0.5,
            final_score=score,
            relevance_weight=0.65,
            authority_weight=0.08,
            timeliness_weight=0.22,
            metadata_weight=0.05,
        ),
        ranking_reason="test",
        evidence=[EvidenceItem(source="title", text=paper.title, confidence=1.0)],
    )


def _seeds() -> list[RankedPaper]:
    return [
        _ranked(
            _paper(1, "Graph retrieval with neural ranking", "dense evidence fusion"),
            1,
        ),
        _ranked(
            _paper(2, "Neural ranking for graph search", "dense evidence retrieval"),
            2,
        ),
        _ranked(
            _paper(3, "Graph search via neural encoders", "evidence fusion"),
            3,
        ),
    ]


def test_prf_feedback_is_deterministic_and_excludes_query_and_identifiers() -> None:
    seeds = _seeds()
    seeds[0].paper.abstract += " https://example.org 10.1000/xyz arXiv:2401.00001 2024"

    first = extract_prf_feedback(QUERY, seeds)
    second = extract_prf_feedback(QUERY, list(seeds))

    assert first == second
    assert first
    assert all(item.document_frequency >= 2 for item in first)
    terms = {item.term for item in first}
    assert not terms & {"graph", "retrieval", "2024", "arxiv", "doi"}
    assert all("example" not in item for item in terms)


def test_prf_handles_duplicate_seed_identity_empty_abstract_and_unicode() -> None:
    duplicate = _paper(1, "图神经 网络 表征", "")
    ranked = [
        _ranked(duplicate, 1),
        _ranked(duplicate.model_copy(deep=True), 2),
        _ranked(_paper(2, "图神经 网络 学习", "网络 表征"), 3),
    ]

    terms = extract_prf_feedback("图检索", ranked)
    plan = build_prf_plan(
        "图检索",
        analyze_query("图检索", query_planning_policy="current_rules").subqueries,
        ranked,
        first_round_succeeded=True,
    )

    assert any(item.term == "网络" for item in terms)
    assert all(item.document_frequency == 2 for item in terms)
    assert [item.title for item in plan.seeds] == [
        "图神经 网络 表征",
        "图神经 网络 学习",
    ]


def test_prf_ties_are_lexically_deterministic_and_limited_to_six() -> None:
    ranked = [
        _ranked(_paper(1, "alpha beta gamma delta epsilon zeta eta"), 1),
        _ranked(_paper(2, "alpha beta gamma delta epsilon zeta eta"), 2),
    ]

    first = extract_prf_feedback("unrelated", ranked)
    second = extract_prf_feedback("unrelated", list(reversed(ranked)))

    assert len(first) == 6
    assert [item.term for item in first] == [item.term for item in second]


def test_prf_replaces_lowest_priority_derived_query_without_budget_growth() -> None:
    current = analyze_query(QUERY, query_planning_policy="current_rules")

    outcome = build_prf_plan(
        QUERY,
        current.subqueries,
        _seeds(),
        first_round_succeeded=True,
    )

    assert outcome.skip_reason is None
    assert len(outcome.subqueries) == len(current.subqueries)
    assert outcome.subqueries[0] == current.subqueries[0]
    assert outcome.replaced_index == len(current.subqueries) - 1
    assert outcome.replaced_query == current.subqueries[-1].query
    assert outcome.subqueries[-1].purpose == "prf_v1"
    assert outcome.subqueries[-1].priority == current.subqueries[-1].priority
    assert outcome.query and outcome.query.startswith(f"{QUERY} ")


def test_prf_falls_back_for_no_feedback_or_failed_first_round() -> None:
    current = analyze_query(QUERY, query_planning_policy="current_rules")
    no_feedback = [_ranked(_paper(1, "graph retrieval"), 1)]

    empty = build_prf_plan(
        QUERY,
        current.subqueries,
        no_feedback,
        first_round_succeeded=True,
    )
    failed = build_prf_plan(
        QUERY,
        current.subqueries,
        _seeds(),
        first_round_succeeded=False,
    )

    assert empty.subqueries == current.subqueries
    assert empty.skip_reason == "no_eligible_feedback_terms"
    assert empty.fallback_used is True
    assert failed.subqueries == current.subqueries
    assert failed.skip_reason == "first_round_failed"


def test_search_service_prf_runs_original_first_and_preserves_call_budget() -> None:
    calls: list[str] = []
    seed_papers = [item.paper for item in _seeds()]

    def retrieve(query: str, **kwargs: object) -> RetrievalOutput:
        calls.append(query)
        sources = list(kwargs.get("sources") or ["arxiv"])
        papers = seed_papers if query == QUERY else []
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=len(papers),
            deduplicated_count=len(papers),
            papers=papers,
            source_stats=[
                SourceStats(
                    source=source,
                    terminal_status="success",
                    query=query,
                    returned_count=len(papers),
                )
                for source in sources
            ],
        )

    baseline_plan = analyze_query(QUERY, query_planning_policy="current_rules")
    output = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        query_planning_policy="prf_v1",
        sources_override=["arxiv"],
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        enable_synthesis=False,
    )

    assert calls[0] == QUERY
    assert len(calls) == len(baseline_plan.subqueries)
    assert output.search_plan.subqueries[0].purpose == "original_query"
    assert output.search_plan.subqueries[-1].purpose == "prf_v1"
    assert output.search_plan.query_planning.prf_seed_candidates
    assert output.search_plan.query_planning.prf_feedback_terms
    assert output.search_plan.query_planning.prf_fallback_used is False
    assert output.search_plan.query_planning.prf_first_round_source_statuses == {
        "arxiv": "success"
    }


def test_prf_default_is_off_and_failed_round_falls_back_in_original_order() -> None:
    calls: list[str] = []

    def retrieve(query: str, **kwargs: object) -> RetrievalOutput:
        calls.append(query)
        sources = list(kwargs.get("sources") or ["arxiv"])
        failed = query == QUERY
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=0,
            deduplicated_count=0,
            source_stats=[
                SourceStats(
                    source=source,
                    terminal_status="failed" if failed else "success",
                    query=query,
                    error_message="timeout" if failed else None,
                )
                for source in sources
            ],
        )

    current = analyze_query(QUERY, query_planning_policy="current_rules")
    default = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        sources_override=["arxiv"],
        enable_synthesis=False,
    )
    assert default.search_plan.query_planning_policy == "current_rules"
    assert default.search_plan.query_planning.prf_seed_candidates == []

    calls.clear()
    failed = SearchService(retriever=retrieve, max_workers=1).run_search(
        QUERY,
        query_planning_policy="prf_v1",
        sources_override=["arxiv"],
        enable_synthesis=False,
    )
    assert calls == [item.query for item in current.subqueries]
    assert failed.search_plan.query_planning.prf_skip_reason == "first_round_failed"
    assert failed.search_plan.query_planning.prf_fallback_used is True


def test_prf_reuses_identical_current_rules_snapshot_requests() -> None:
    common = {
        "source": "arxiv",
        "adapted_query": "graph retrieval neural ranking",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": "test",
        "query_planner_version": "1.8.1",
    }
    current, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="current_rules",
    )
    prf, _ = retrieval_snapshot_key(
        **common,
        query_planning_policy="prf_v1",
    )

    assert prf == current
