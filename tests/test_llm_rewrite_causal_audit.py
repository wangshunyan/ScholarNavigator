from __future__ import annotations

from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    _PreparedCounterfactual,
    _QueryList,
    _rank_pool,
    build_source_counterfactual,
    classify_counterfactual,
    equivalent_paper_sequences,
    query_list_observations,
    rewrite_acceptance,
    source_comparability,
    stable_source_coverage_truncate,
    write_causal_audit,
)


def _paper(
    title: str,
    doi: str,
    *,
    source: str = "arxiv",
    citation_count: int = 0,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice Example"],
        year=2024,
        venue="TestConf",
        abstract=f"Evidence about {title}",
        identifiers=PaperIdentifiers(doi=doi),
        sources=[source],
        citation_count=citation_count,
    )


def _query_list(
    query: str,
    source: str,
    papers: list[Paper],
    *,
    raw_count: int | None = None,
    status: str = "success",
) -> _QueryList:
    return _QueryList(
        query,
        source,
        papers,
        len(papers) if raw_count is None else raw_count,
        [
            {
                "adapted_query": query,
                "adaptation_strategy": "safe_original",
                "status": status,
                "key": "a" * 64,
                "paper_count": len(papers),
            }
        ],
    )


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="causal bandit interventions",
        language="en",
        intent="paper_finding",
        domain="machine_learning",
        constraints=QueryConstraint(
            domains=["machine_learning"],
            must_include_terms=["causal", "bandit", "interventions"],
        ),
    )


def test_query_list_observations_cover_overlap_independent_gold_and_duplicates() -> None:
    shared = _paper("Shared", "10.1000/shared")
    shared_variant = _paper("Shared", "HTTPS://DOI.ORG/10.1000/SHARED")
    replaced_gold = _paper("Replaced gold", "10.1000/replaced")
    rewrite_gold = _paper("Rewrite gold", "10.1000/rewrite")
    gold = [
        EvalGoldPaper(doi="10.1000/replaced"),
        EvalGoldPaper(doi="10.1000/rewrite"),
    ]
    observations = query_list_observations(
        [
            _query_list("original", "arxiv", [shared], raw_count=1),
            _query_list(
                "replaced",
                "arxiv",
                [shared_variant, replaced_gold],
                raw_count=3,
            ),
            _query_list("rewrite", "arxiv", [rewrite_gold], raw_count=1),
        ],
        gold,
    )

    assert observations[1].unique_candidate_count == 2
    assert observations[1].duplicate_ratio == 1 / 3
    assert observations[1].independent_gold_ids == ["doi:10.1000/replaced"]
    assert observations[2].independent_gold_ids == ["doi:10.1000/rewrite"]
    assert observations[1].first_gold_rank == 2


def test_source_comparability_rejects_failure_and_original_result_drift() -> None:
    paper = _paper("Same", "10.1000/same")
    original = _query_list("original", "arxiv", [paper])
    failed_rewrite = _query_list("rewrite", "arxiv", [], status="failed")
    status, reasons = source_comparability(
        original,
        original,
        _query_list("replaced", "arxiv", []),
        failed_rewrite,
    )
    assert status == "source_terminal_inconsistent"
    assert reasons == ["rewrite_request_failed"]

    status, reasons = source_comparability(
        original,
        _query_list("original", "arxiv", [_paper("Drift", "10.1000/drift")]),
        _query_list("replaced", "arxiv", []),
        _query_list("rewrite", "arxiv", []),
    )
    assert status == "source_terminal_inconsistent"
    assert "original_candidate_list_mismatch" in reasons


def test_source_counterfactual_attributes_replacement_gold_loss() -> None:
    other = _paper("Causal bandit overview", "10.1000/other", citation_count=3)
    gold_paper = _paper(
        "Causal bandit interventions",
        "10.1000/gold",
        citation_count=20,
    )
    gold = [EvalGoldPaper(doi="10.1000/gold")]
    baseline_metrics, baseline_ranked, _ = _rank_pool(
        _analysis(), [other, gold_paper], gold
    )
    prepared = _PreparedCounterfactual(
        [other, gold_paper],
        [
            {"provenance": [{"origin_subquery": "original", "source": "arxiv"}]},
            {"provenance": [{"origin_subquery": "replaced", "source": "arxiv"}]},
        ],
        _analysis(),
        baseline_metrics,
        baseline_ranked,
    )
    rewritten = _query_list(
        "rewrite",
        "arxiv",
        [_paper("Unrelated rewrite result", "10.1000/new")],
    )
    first = build_source_counterfactual(
        prepared,
        gold,
        {"sources": ["arxiv"], "budgets": {"max_candidate_papers": 20}},
        replaced_query="replaced",
        source="arxiv",
        rewritten=rewritten,
    )
    second = build_source_counterfactual(
        prepared,
        gold,
        {"sources": ["arxiv"], "budgets": {"max_candidate_papers": 20}},
        replaced_query="replaced",
        source="arxiv",
        rewritten=rewritten,
    )

    assert first == second
    assert first["replacement_lost_gold_ids"] == ["doi:10.1000/gold"]
    assert first["rewrite_added_gold_ids"] == []
    assert classify_counterfactual(first) == "replacement_lost_gold"


def test_identity_sequence_and_source_budget_are_deterministic() -> None:
    doi = _paper("Unicode – title", "10.1000/same", source="openalex")
    doi_variant = _paper(
        "Unicode - title",
        "https://doi.org/10.1000/SAME",
        source="openalex",
    )
    assert equivalent_paper_sequences([doi], [doi_variant])

    papers = [
        _paper("A1", "10.1000/a1", source="arxiv"),
        _paper("A2", "10.1000/a2", source="arxiv"),
        _paper("S1", "10.1000/s1", source="semantic_scholar"),
        _paper("S2", "10.1000/s2", source="semantic_scholar"),
    ]
    first = stable_source_coverage_truncate(
        papers,
        limit=3,
        source_order=["arxiv", "semantic_scholar"],
    )
    second = stable_source_coverage_truncate(
        papers,
        limit=3,
        source_order=["arxiv", "semantic_scholar"],
    )
    assert [paper.title for paper in first] == ["A1", "S1", "A2"]
    assert first == second


def test_fallback_classification_and_output_bytes_are_deterministic(tmp_path: Path) -> None:
    assert rewrite_acceptance(
        {
            "accepted_query_count": 0,
            "fallback_used": True,
            "fallback_reason": "rewrite_rejected",
        }
    ) == (False, "rewrite_rejected")
    counterfactual = {
        "rewrite_added_gold_ids": [],
        "replacement_lost_gold_ids": [],
        "baseline": {"candidate_ids": ["doi:a"]},
        "remove_replaced_add_rewrite": {"candidate_ids": ["doi:b"]},
    }
    assert classify_counterfactual(counterfactual) == "candidate_only_change"

    rows = [
        {
            "case_id": "case-1",
            "classification": "fallback_or_rejected_unattributable",
            "fallback_reason": "rewrite_rejected",
        }
    ]
    aggregate = {"network_request_count": 0, "snapshot_write_count": 0}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_causal_audit(first, rows, aggregate)
    write_causal_audit(second, rows, aggregate)
    assert (first / "case_audit.jsonl").read_bytes() == (
        second / "case_audit.jsonl"
    ).read_bytes()
    assert (first / "aggregate.json").read_bytes() == (
        second / "aggregate.json"
    ).read_bytes()
