from __future__ import annotations

from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
from scholar_agent.evaluation.current_rules_source_audit import (
    _cross_dataset_removal,
    _metrics_non_decreasing,
    _rank_pool,
    build_source_pool,
    source_candidate_observations,
    source_terminal_status,
    write_source_audit,
)
from scholar_agent.evaluation.current_rules_subquery_audit import _Cell, _QueryList


def _paper(
    title: str,
    doi: str,
    *,
    source: str,
    citation_count: int = 0,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice Example"],
        year=2024,
        abstract=title,
        identifiers=PaperIdentifiers(doi=doi),
        sources=[source],
        citation_count=citation_count,
    )


def _cell(
    query: str,
    purpose: str,
    priority: int,
    papers: list[Paper],
    *,
    source: str,
    status: str = "completed",
) -> _Cell:
    terminal_status = "success" if status == "completed" else status
    return _Cell(
        "case",
        source,
        query,
        purpose,
        priority,
        _QueryList(
            query,
            source,
            papers,
            len(papers),
            [{"status": terminal_status, "key": str(priority) * 64}],
        ),
        status,  # type: ignore[arg-type]
        [status] if status not in {"completed", "not_started"} else [],
        {
            "snapshot_key_count": int(status != "not_started"),
            "request_count": int(status != "not_started"),
            "retry_count": 0,
            "error_count": int(status == "source_failure"),
            "cache_hit_count": 0,
            "latency_seconds": 0.1,
            "rate_limit_wait_seconds": 0.0,
        },
    )


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="causal evidence",
        language="en",
        intent="general",
        domain="general_science",
        constraints=QueryConstraint(must_include_terms=["causal", "evidence"]),
    )


def test_source_status_keeps_failure_not_started_and_missing_distinct() -> None:
    success = _cell("q", "original_query", 1, [], source="arxiv")
    skipped = _cell(
        "d", "normalized_keywords", 2, [], source="arxiv", status="not_started"
    )
    failed = _cell(
        "d", "normalized_keywords", 2, [], source="arxiv", status="source_failure"
    )
    missing = _cell(
        "d", "normalized_keywords", 2, [], source="arxiv", status="missing_snapshot"
    )

    assert source_terminal_status([success, skipped]) == ("success", [])
    assert source_terminal_status([skipped])[0] == "not_started"
    assert source_terminal_status([success, failed])[0] == "failed"
    assert source_terminal_status([success, missing])[0] == "missing_snapshot"


def test_source_observations_merge_identity_and_find_independent_gold() -> None:
    shared = _paper("Shared", "10.1000/shared", source="arxiv")
    shared_variant = _paper(
        "Shared variant", "https://doi.org/10.1000/SHARED", source="pubmed"
    )
    gold_paper = _paper("Gold", "10.1000/gold", source="pubmed")
    observations = source_candidate_observations(
        ["arxiv", "pubmed"],
        {"arxiv": [shared], "pubmed": [shared_variant, gold_paper]},
        {"arxiv": "success", "pubmed": "success"},
        [EvalGoldPaper(doi="10.1000/gold")],
    )

    assert observations["arxiv"]["observed_independent_candidate_ids"] == []
    assert observations["pubmed"]["observed_independent_gold_ids"] == [
        "doi:10.1000/gold"
    ]
    assert observations["pubmed"]["strict_independent_gold_ids"] == [
        "doi:10.1000/gold"
    ]
    assert observations["pubmed"]["first_gold_rank"] == 2


def test_failed_competing_source_disables_strict_independence() -> None:
    observations = source_candidate_observations(
        ["arxiv", "openalex"],
        {
            "arxiv": [_paper("Only observed", "10.1000/only", source="arxiv")],
            "openalex": [],
        },
        {"arxiv": "success", "openalex": "failed"},
        [],
    )
    assert observations["arxiv"]["observed_independent_candidate_ids"] == [
        "doi:10.1000/only"
    ]
    assert observations["arxiv"]["strict_independent_candidate_ids"] is None


def test_source_pool_and_leave_one_out_rerank_are_deterministic() -> None:
    selected = [
        {"query": "derived", "purpose": "normalized_keywords", "priority": 2},
        {"query": "original", "purpose": "original_query", "priority": 1},
    ]
    original = _paper("Causal evidence", "10.1000/shared", source="arxiv")
    duplicate = _paper(
        "Causal evidence duplicate", "10.1000/SHARED", source="pubmed"
    )
    gold_paper = _paper(
        "Causal gold evidence",
        "10.1000/gold",
        source="pubmed",
        citation_count=20,
    )
    cells = {
        ("original", "arxiv"): _cell(
            "original", "original_query", 1, [original], source="arxiv"
        ),
        ("derived", "arxiv"): _cell(
            "derived", "normalized_keywords", 2, [], source="arxiv"
        ),
        ("original", "pubmed"): _cell(
            "original", "original_query", 1, [duplicate], source="pubmed"
        ),
        ("derived", "pubmed"): _cell(
            "derived", "normalized_keywords", 2, [gold_paper], source="pubmed"
        ),
    }
    full = build_source_pool(
        selected,
        ["arxiv", "pubmed"],
        cells,
        included_sources=["arxiv", "pubmed"],
        candidate_limit=20,
    )
    reversed_input = build_source_pool(
        list(reversed(selected)),
        ["arxiv", "pubmed"],
        cells,
        included_sources=["arxiv", "pubmed"],
        candidate_limit=20,
    )
    leave_pubmed = build_source_pool(
        selected,
        ["arxiv", "pubmed"],
        cells,
        included_sources=["arxiv"],
        candidate_limit=20,
    )
    gold = [EvalGoldPaper(doi="10.1000/gold")]
    full_metrics, _, _ = _rank_pool(_analysis(), full, gold)
    leave_metrics, _, _ = _rank_pool(_analysis(), leave_pubmed, gold)

    assert full == reversed_input
    assert len(full) == 2
    assert full_metrics["candidate_recall"] == 1.0
    assert leave_metrics["candidate_recall"] == 0.0


def test_non_decreasing_requires_all_three_metrics() -> None:
    baseline = {"candidate_recall": 0.1, "recall_at_20": 0.1, "f1_at_20": 0.1}
    tied = {"candidate_recall": 0.1, "recall_at_20": 0.1, "f1_at_20": 0.1}
    mixed = {"candidate_recall": 0.2, "recall_at_20": 0.0, "f1_at_20": 0.0}
    assert _metrics_non_decreasing(baseline, tied)
    assert not _metrics_non_decreasing(baseline, mixed)


def test_source_without_strict_cases_cannot_be_marked_safe_to_remove() -> None:
    unavailable_summary = {
        "case_count": 0,
        "leave_one_out_outcomes": {"degraded": 0, "improved": 0, "tied": 0},
        "baseline": {
            "candidate_recall": None,
            "recall_at_20": None,
            "f1_at_20": None,
        },
        "leave_one_out": {
            "candidate_recall": None,
            "recall_at_20": None,
            "f1_at_20": None,
        },
        "request_savings": {"request_count": 0},
    }
    datasets = {
        "dev": {
            "sources": {
                "openalex": {"strict_source_success": unavailable_summary}
            }
        },
        "val": {
            "sources": {
                "openalex": {"strict_source_success": unavailable_summary}
            }
        },
    }
    result = _cross_dataset_removal(datasets)
    assert result[0]["strict_evidence_in_all_datasets"] is False
    assert result[0]["safe_removal_supported"] is False


def test_source_audit_output_is_byte_deterministic(tmp_path: Path) -> None:
    rows = [{"dataset": "dev", "case_order": 0, "case_id": "case"}]
    aggregate = {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_source_audit(first, rows, aggregate)
    write_source_audit(second, rows, aggregate)
    assert (first / "case_source_audit.jsonl").read_bytes() == (
        second / "case_source_audit.jsonl"
    ).read_bytes()
    assert (first / "aggregate.json").read_bytes() == (
        second / "aggregate.json"
    ).read_bytes()
