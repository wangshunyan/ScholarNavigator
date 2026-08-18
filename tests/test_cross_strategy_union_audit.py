from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
from scholar_agent.evaluation.cross_strategy_union_audit import (
    CoverOption,
    _Observation,
    _query_groups,
    _product_candidate_pool,
    _strategy_rows,
    _strict_reasons,
    exact_minimum_cover,
    gold_priority_oracle,
    rank_candidate_union,
    write_cross_strategy_union_audit,
)


def _paper(
    title: str,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    year: int = 2020,
) -> Paper:
    return Paper(
        title=title,
        authors=["Ada Author"],
        year=year,
        identifiers=PaperIdentifiers(doi=doi, arxiv_id=arxiv_id),
        sources=["arxiv"],
    )


def _observation(
    strategy: str,
    source: str,
    query: str,
    key: str,
    papers: list[Paper],
    *,
    status: str = "success",
    order: int = 1,
) -> _Observation:
    return _Observation(
        strategy,
        order,
        "initial_retrieval",
        source,
        query,
        query,
        key,
        status,
        papers,
        {
            "request_count": 1,
            "retry_count": 0,
            "error_count": 0,
            "cache_hit_count": 0,
            "latency_seconds": 0.5,
            "rate_limit_wait_seconds": 0.0,
        },
        None,
    )


def test_exact_minimum_cover_deduplicates_coverage_and_is_deterministic() -> None:
    options = [
        CoverOption("query-b", frozenset({"gold-1", "gold-2"}), 2, 2.0),
        CoverOption("query-a", frozenset({"gold-1"}), 1, 0.5),
        CoverOption("query-c", frozenset({"gold-2"}), 1, 0.5),
        CoverOption("duplicate", frozenset({"gold-1"}), 3, 9.0),
    ]

    first = exact_minimum_cover({"gold-1", "gold-2"}, options)
    second = exact_minimum_cover({"gold-2", "gold-1"}, list(reversed(options)))

    assert first == second
    assert first["complete"] is True
    assert first["selected_query_ids"] == ["query-b"]
    assert first["selected_query_count"] == 1
    assert first["request_count"] == 2


def test_query_groups_close_duplicate_queries_and_detect_response_drift() -> None:
    shared = _paper("Shared", doi="10.1/shared")
    changed = _paper("Changed", doi="10.1/changed")
    values = {
        "current_rules": [
            _observation("current_rules", "arxiv", "same query", "a", [shared])
        ],
        "concept_projection": [
            _observation(
                "concept_projection", "arxiv", "same query", "b", [shared]
            ),
            _observation(
                "concept_projection", "openalex", "other query", "c", [changed]
            ),
        ],
    }
    states = {
        "current_rules": {"state": "baseline", "reason": None},
        "concept_projection": {"state": "action_executed", "reason": None},
    }

    groups = _query_groups(values, states)

    assert [(row["source"], row["query"]) for row in groups] == [
        ("arxiv", "same query"),
        ("openalex", "other query"),
    ]
    assert groups[0]["owner_count"] == 2
    assert groups[0]["consistent"] is True
    values["concept_projection"][0] = _observation(
        "concept_projection", "arxiv", "same query", "b", [changed]
    )
    assert _query_groups(values, states)[0]["consistent"] is False


def test_strategy_contribution_uses_unified_identity_and_unique_queries() -> None:
    left = _paper("Unicode：Paper", doi="https://doi.org/10.1/SAME")
    same = _paper("Unicode Paper", doi="10.1/same")
    unique = _paper("Unique", arxiv_id="2401.00001v2")
    strategies = ["current_rules", "concept_projection"]
    pools = {"current_rules": [left], "concept_projection": [same, unique]}
    candidate_sets = {
        "current_rules": {"doi:10.1/same"},
        "concept_projection": {"doi:10.1/same", "arxiv:2401.00001"},
    }
    gold_sets = {
        "current_rules": {"doi:10.1/same"},
        "concept_projection": {"doi:10.1/same", "arxiv:2401.00001"},
    }
    observations = {
        "current_rules": [
            _observation("current_rules", "arxiv", "shared", "a", [left])
        ],
        "concept_projection": [
            _observation("concept_projection", "arxiv", "shared", "b", [same]),
            _observation("concept_projection", "arxiv", "projected", "c", [unique]),
        ],
    }
    states = {
        "current_rules": {"state": "baseline", "reason": None},
        "concept_projection": {"state": "action_executed", "reason": None},
    }
    gold = [
        EvalGoldPaper(doi="10.1/same"),
        EvalGoldPaper(arxiv_id="2401.00001"),
    ]

    rows = _strategy_rows(
        strategies,
        pools,
        candidate_sets,
        gold_sets,
        observations,
        states,
        gold,
    )

    assert rows["concept_projection"]["observed_independent_gold_ids"] == [
        "arxiv:2401.00001"
    ]
    assert rows["concept_projection"]["unique_query_attributable_gold_ids"] == [
        "arxiv:2401.00001"
    ]
    assert rows["current_rules"]["unique_query_attributable_gold_ids"] == []


def test_gold_priority_oracle_handles_multi_gold_duplicates_and_top20_boundary() -> None:
    duplicate_a = _paper("A", doi="10.1/a")
    duplicate_b = _paper("A second record", doi="10.1/a")
    noise = [_paper(f"Noise {index}", doi=f"10.1/noise-{index}") for index in range(20)]
    gold_b = _paper("B", doi="10.1/b")
    candidates = [duplicate_a, duplicate_b, *noise, gold_b]
    gold = [EvalGoldPaper(doi="10.1/a"), EvalGoldPaper(doi="10.1/b")]

    oracle = gold_priority_oracle(candidates, gold)

    assert oracle["candidate_gold_count"] == 2
    assert oracle["candidate_recall"] == 1.0
    assert len(oracle["top20_gold_ids"]) == 2
    assert oracle["recall_at_20"] == 1.0
    assert oracle["oracle_only_not_achieved_score"] is True


def test_product_pool_accepts_canonical_succeeded_terminal() -> None:
    paper = _paper("Recorded", doi="10.1/recorded")
    row = {
        "status": "succeeded",
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": "initial_deduplicated",
                    "status": "completed",
                    "candidates": [
                        {
                            "title": paper.title,
                            "authors": paper.authors,
                            "year": paper.year,
                            "identifiers": paper.identifiers.model_dump(),
                        }
                    ],
                }
            ]
        },
    }
    observation = _observation(
        "current_rules", "arxiv", "query", "key", [paper]
    )

    assert _product_candidate_pool(row, [observation]) == [paper]


def test_union_rerank_merges_identity_and_uses_current_filtering() -> None:
    duplicate_a = _paper("LLM retrieval method", doi="10.1/gold")
    duplicate_b = _paper("LLM retrieval method extended", doi="10.1/gold")
    unrelated = _paper("Unrelated chemistry", doi="10.1/noise")
    analysis = QueryAnalysis(
        original_query="LLM retrieval method",
        language="en",
        intent="paper_finding",
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["retrieval"],
            must_include_terms=["LLM", "retrieval", "method"],
        ),
    )

    candidates, metrics = rank_candidate_union(
        analysis,
        [[duplicate_a, unrelated], [duplicate_b]],
        [EvalGoldPaper(doi="10.1/gold")],
    )

    assert len(candidates) == 2
    assert metrics["candidate_recall"] == 1.0
    assert metrics["recall_at_20"] == 1.0


def test_strict_subset_rejects_failure_fallback_missing_and_inconsistent_query() -> None:
    observations = {
        "current_rules": [
            _observation("current_rules", "arxiv", "q", "a", [], status="failed")
        ],
        "llm_semantic": [],
    }
    states = {
        "current_rules": {"state": "baseline", "reason": None},
        "llm_semantic": {"state": "fallback_or_rejected", "reason": "schema"},
    }
    products = {
        "current_rules": {"product_reconstruction_error": None},
        "llm_semantic": {"product_reconstruction_error": "candidate_stage_unavailable"},
    }

    reasons = _strict_reasons(
        ["current_rules", "llm_semantic"],
        states,
        observations,
        products,
        [{"consistent": False}],
    )

    assert reasons == [
        "current_rules:terminal_failed",
        "duplicate_query_response_inconsistent",
        "llm_semantic:fallback_or_rejected",
        "llm_semantic:product_reconstruction_failed",
    ]


def test_audit_writer_is_byte_deterministic(tmp_path: Path) -> None:
    rows = [{"case_id": "二", "values": [2, 1]}, {"case_id": "a", "values": []}]
    aggregate = {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_cross_strategy_union_audit(first, rows, aggregate)
    write_cross_strategy_union_audit(second, rows, aggregate)

    assert (first / "case_union_audit.jsonl").read_bytes() == (
        second / "case_union_audit.jsonl"
    ).read_bytes()
    assert (first / "aggregate.json").read_bytes() == (
        second / "aggregate.json"
    ).read_bytes()
    assert json.loads((first / "aggregate.json").read_text()) == aggregate
