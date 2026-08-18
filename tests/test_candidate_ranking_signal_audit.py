from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.evaluation.candidate_ranking_signal_audit import (
    evaluate_case_signals,
    extract_candidate_signal_row,
    rank_candidate_signal_rows,
    write_candidate_ranking_signal_audit,
)


def _candidate(
    doi: str,
    *,
    rank: int,
    final_score: float,
    provenance: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "identifiers": {"doi": doi},
        "title": doi,
        "year": 2024,
        "sources": sorted(
            {
                str(item.get("source") or "")
                for item in provenance
                if item.get("source")
            }
        ),
        "provenance": provenance,
        "rank": rank,
        "final_score": final_score,
        "judgement_score": final_score,
        "category": "partially_relevant",
    }


def _provenance(
    source: str,
    query: str,
    source_rank: object,
    *,
    purpose: str = "original_query",
) -> dict[str, object]:
    return {
        "source": source,
        "adapted_query": query,
        "source_rank": source_rank,
        "purpose": purpose,
        "origin_kind": "initial_query",
    }


def test_extracts_best_rank_and_deduplicates_same_list_support() -> None:
    row = extract_candidate_signal_row(
        _candidate(
            "10.1/shared",
            rank=1,
            final_score=0.6,
            provenance=[
                _provenance("arxiv", "query", 8),
                _provenance("arxiv", "query", 3),
                _provenance(
                    "openalex", "derived", 5, purpose="normalized_keywords"
                ),
            ],
        ),
        score_breakdown={
            "relevance_score": 0.6,
            "authority_score": 0.2,
            "final_score": 0.5,
        },
    )
    assert row["best_source_rank"] == 3
    assert row["best_reciprocal_rank"] == 1 / 3
    assert row["support_list_count"] == 2
    assert row["support_source_count"] == 2
    assert row["original_query_list_count"] == 1
    assert row["derived_query_list_count"] == 1
    assert row["provenance_status"] == "complete"
    assert row["score_breakdown"]["relevance_score"] == 0.6


def test_missing_rank_is_explicit_and_sorts_after_valid_rank() -> None:
    missing = extract_candidate_signal_row(
        _candidate(
            "10.1/missing",
            rank=1,
            final_score=0.8,
            provenance=[_provenance("arxiv", "query", None)],
        )
    )
    valid = extract_candidate_signal_row(
        _candidate(
            "10.1/valid",
            rank=2,
            final_score=0.7,
            provenance=[_provenance("arxiv", "query", 10)],
        )
    )
    assert missing["provenance_status"] == "incomplete"
    assert missing["best_reciprocal_rank"] == 0.0
    ranked = rank_candidate_signal_rows(
        [missing, valid], "best_reciprocal_rank"
    )
    assert [item["candidate_id"] for item in ranked] == [
        "doi:10.1/valid",
        "doi:10.1/missing",
    ]


def test_tied_provenance_signal_is_input_order_independent() -> None:
    first = extract_candidate_signal_row(
        _candidate(
            "10.1/a",
            rank=2,
            final_score=0.5,
            provenance=[_provenance("arxiv", "q", 4)],
        )
    )
    second = extract_candidate_signal_row(
        _candidate(
            "10.1/b",
            rank=1,
            final_score=0.6,
            provenance=[_provenance("openalex", "q", 4)],
        )
    )
    forward = rank_candidate_signal_rows(
        [first, second], "best_reciprocal_rank"
    )
    reverse = rank_candidate_signal_rows(
        [second, first], "best_reciprocal_rank"
    )
    assert [item["candidate_id"] for item in forward] == [
        item["candidate_id"] for item in reverse
    ] == ["doi:10.1/a", "doi:10.1/b"]


def test_cross_source_merged_identity_matches_gold_once() -> None:
    merged = extract_candidate_signal_row(
        _candidate(
            "https://doi.org/10.1/MERGED",
            rank=1,
            final_score=0.2,
            provenance=[
                _provenance("arxiv", "q", 2),
                _provenance("openalex", "q", 1),
            ],
        )
    )
    evaluated = evaluate_case_signals(
        [merged], [EvalGoldPaper(doi="10.1/merged")]
    )
    for signal in evaluated["signals"].values():
        assert signal["captures"]["20"]["matched_gold_relation_count"] == 1
        assert signal["gold_match_ranks"] == [1]


def test_unevaluable_gold_is_excluded_and_top_20_boundary_is_exact() -> None:
    rows = [
        extract_candidate_signal_row(
            _candidate(
                f"10.1/{index:02d}",
                rank=index,
                final_score=1 - index / 100,
                provenance=[_provenance("arxiv", "q", index)],
            )
        )
        for index in range(1, 22)
    ]
    evaluated = evaluate_case_signals(
        rows,
        [EvalGoldPaper(doi="10.1/20"), EvalGoldPaper()],
    )
    baseline = evaluated["signals"]["existing_composite_score"]
    assert evaluated["evaluable_gold_count"] == 1
    assert baseline["captures"]["20"]["matched_gold_relation_count"] == 1
    assert baseline["gold_rank_distribution"]["1_20"] == 1
    assert baseline["gold_rank_distribution"]["candidate_miss"] == 0


def test_manifest_freezes_signals_before_metrics() -> None:
    manifest = json.loads(
        Path("benchmark/candidate_ranking_signal_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["name"] for item in manifest["signals"]] == [
        "existing_composite_score",
        "best_reciprocal_rank",
        "support_list_count",
        "support_source_count",
    ]
    assert manifest["cutoffs"] == [20, 50, 100]
    assert manifest["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
    }


def test_audit_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"dataset": "x", "case_order": 0, "case_id": "one"}]
    candidates = [
        {
            "dataset": "x",
            "case_order": 0,
            "candidate_id": "doi:10.1/one",
        }
    ]
    aggregate = {"schema_version": "1", "signals": ["one"]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_candidate_ranking_signal_audit(first, cases, candidates, aggregate)
    write_candidate_ranking_signal_audit(second, cases, candidates, aggregate)
    for name in (
        "case_signal_audit.jsonl",
        "candidate_signal_audit.jsonl",
        "aggregate.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
