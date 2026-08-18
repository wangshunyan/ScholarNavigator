from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.current_rules_subquery_audit import (
    _Cell,
    _QueryList,
    _read_cell,
    _request_savings,
    _average_metrics,
    _metric_outcome,
    build_query_type_pool,
    classify_terminals,
    source_list_contributions,
    write_subquery_audit,
)
from scholar_agent.evaluation.metrics import matched_paper_ids
from scholar_agent.evaluation.snapshots.store import SnapshotMissingError


def _paper(title: str, doi: str, *, source: str = "arxiv") -> Paper:
    return Paper(
        title=title,
        authors=["Alice Example"],
        year=2024,
        identifiers=PaperIdentifiers(doi=doi),
        sources=[source],
    )


def _cell(
    query: str,
    purpose: str,
    priority: int,
    papers: list[Paper],
    *,
    status: str = "completed",
    raw_count: int | None = None,
    source: str = "arxiv",
    key: str | None = None,
) -> _Cell:
    terminal_status = "success" if status == "completed" else status
    terminal = {
        "status": terminal_status,
        "key": key or (str(priority) * 64),
        "recorded_diagnostics": {
            "request_count": 1,
            "retry_count": 0,
            "error_count": int(status == "source_failure"),
            "cache_hit_count": 0,
            "latency_seconds": 0.5,
            "rate_limit_wait_seconds": 0.0,
        },
    }
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
            len(papers) if raw_count is None else raw_count,
            [terminal],
        ),
        status,  # type: ignore[arg-type]
        ["timeout"] if status == "source_failure" else [],
        {
            "snapshot_key_count": 1,
            "request_count": 1,
            "retry_count": 0,
            "error_count": int(status == "source_failure"),
            "cache_hit_count": 0,
            "latency_seconds": 0.5,
            "rate_limit_wait_seconds": 0.0,
        },
    )


def test_terminal_classification_keeps_failure_and_not_started_distinct() -> None:
    assert classify_terminals(
        [{"status": "failed", "error_type": "timeout"}]
    ) == ("source_failure", ["timeout"])
    assert classify_terminals([{"status": "not_started"}]) == (
        "not_started",
        ["no_adapted_query_was_started"],
    )
    assert classify_terminals(
        [{"status": "success"}, {"status": "not_started"}]
    ) == ("completed", [])


def test_source_contribution_handles_overlap_independent_gold_and_plan_order() -> None:
    shared = _paper("Shared", "10.1000/shared")
    shared_variant = _paper("Shared variant", "https://doi.org/10.1000/SHARED")
    gold_paper = _paper("Independent gold", "10.1000/gold")
    gold = [EvalGoldPaper(doi="10.1000/gold")]
    rows = source_list_contributions(
        [
            _cell("original", "original_query", 1, [shared]),
            _cell(
                "derived",
                "normalized_keywords",
                2,
                [shared_variant, gold_paper],
                raw_count=3,
            ),
        ],
        gold,
    )

    assert rows[1]["duplicate_ratio"] == pytest.approx(1 / 3)
    assert rows[1]["independent_candidate_ids"] == ["doi:10.1000/gold"]
    assert rows[1]["independent_gold_ids"] == ["doi:10.1000/gold"]
    assert rows[1]["plan_order_marginal_candidate_count"] == 1
    assert rows[1]["plan_order_marginal_gold_ids"] == ["doi:10.1000/gold"]
    assert rows[1]["first_gold_rank"] == 2


def test_source_failure_makes_later_marginal_incomparable_not_zero() -> None:
    rows = source_list_contributions(
        [
            _cell(
                "original",
                "original_query",
                1,
                [],
                status="source_failure",
            ),
            _cell(
                "derived",
                "normalized_keywords",
                2,
                [_paper("Observed", "10.1000/observed")],
            ),
        ],
        [],
    )

    assert rows[0]["contribution_comparable"] is False
    assert rows[1]["contribution_comparable"] is False
    assert rows[1]["plan_order_marginal_candidate_count"] is None
    assert rows[1]["independent_candidate_ids"] is None


def test_query_type_pool_reuses_identity_and_removal_is_order_deterministic() -> None:
    original = _paper("Original", "10.1000/original")
    duplicate = _paper("Original duplicate", "HTTPS://DOI.ORG/10.1000/ORIGINAL")
    derived = _paper("Derived", "10.1000/derived")
    selected = [
        {"query": "derived", "purpose": "normalized_keywords", "priority": 2},
        {"query": "original", "purpose": "original_query", "priority": 1},
    ]
    cells = {
        ("original", "arxiv"): _cell(
            "original", "original_query", 1, [original]
        ),
        ("derived", "arxiv"): _cell(
            "derived", "normalized_keywords", 2, [duplicate, derived]
        ),
    }
    first = build_query_type_pool(
        selected,
        ["arxiv"],
        cells,
        candidate_limit=20,
        remove_purpose="normalized_keywords",
    )
    second = build_query_type_pool(
        list(reversed(selected)),
        ["arxiv"],
        cells,
        candidate_limit=20,
        remove_purpose="normalized_keywords",
    )
    full = build_query_type_pool(
        selected,
        ["arxiv"],
        cells,
        candidate_limit=20,
    )

    assert [item.title for item in first] == ["Original"]
    assert first == second
    assert len(full) == 2
    assert matched_paper_ids(full, [EvalGoldPaper(doi="10.1000/derived")]) == [
        "doi:10.1000/derived"
    ]


def test_shared_request_key_is_saved_only_when_all_owners_are_removed() -> None:
    shared_key = "a" * 64
    original = _cell(
        "same", "original_query", 1, [], key=shared_key
    )
    derived = _cell(
        "same", "normalized_keywords", 2, [], key=shared_key
    )
    savings = _request_savings(
        [original, derived], remove_purpose="normalized_keywords"
    )
    assert savings["snapshot_key_count"] == 0
    assert savings["request_count"] == 0


def test_mixed_counterfactual_change_is_not_reported_as_improvement() -> None:
    before = {"candidate_recall": 0.1, "recall_at_20": 0.1, "f1_at_20": 0.1}
    after = {"candidate_recall": 0.2, "recall_at_20": 0.0, "f1_at_20": 0.0}
    assert _metric_outcome(before, after) == "degraded"


def test_metric_average_excludes_identity_unevaluable_cases() -> None:
    metrics = _average_metrics(
        [
            {
                "candidate_recall": 1.0,
                "recall_at_20": 1.0,
                "f1_at_20": 0.5,
                "candidate_gold_ids": ["doi:gold"],
                "returned_gold_ids": ["doi:gold"],
            },
            {
                "candidate_recall": None,
                "recall_at_20": 0.0,
                "f1_at_20": 0.0,
                "candidate_gold_ids": [],
                "returned_gold_ids": [],
            },
        ]
    )
    assert metrics["input_case_count"] == 2
    assert metrics["evaluable_case_count"] == 1
    assert metrics["recall_at_20"] == 1.0


def test_missing_snapshot_cell_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> _QueryList:
        raise SnapshotMissingError("missing")

    monkeypatch.setattr(
        "scholar_agent.evaluation.current_rules_subquery_audit._query_list",
        missing,
    )
    cell = _read_cell(
        "case",
        {},
        {},
        object(),  # type: ignore[arg-type]
        query="query",
        source="arxiv",
        purpose="original_query",
        priority=1,
    )
    assert cell.status == "missing_snapshot"
    assert cell.observation.papers == []


def test_snapshot_request_mismatch_is_terminal_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inconsistent(*args: object, **kwargs: object) -> _QueryList:
        raise ValueError("snapshot request mismatch")

    monkeypatch.setattr(
        "scholar_agent.evaluation.current_rules_subquery_audit._query_list",
        inconsistent,
    )
    cell = _read_cell(
        "case",
        {},
        {},
        object(),  # type: ignore[arg-type]
        query="query",
        source="arxiv",
        purpose="original_query",
        priority=1,
    )
    assert cell.status == "terminal_inconsistent"


def test_audit_output_is_byte_deterministic(tmp_path: Path) -> None:
    rows = [{"dataset": "dev", "case_order": 0, "case_id": "case"}]
    aggregate = {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_subquery_audit(first, rows, aggregate)
    write_subquery_audit(second, rows, aggregate)
    assert (first / "case_audit.jsonl").read_bytes() == (
        second / "case_audit.jsonl"
    ).read_bytes()
    assert (first / "aggregate.json").read_bytes() == (
        second / "aggregate.json"
    ).read_bytes()
