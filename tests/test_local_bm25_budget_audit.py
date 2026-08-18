from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.local_bm25_budget_audit import (
    OfflineLocalBM25Index,
    RankedDocument,
    classify_gold_budget_gap,
    merge_ranked_lists,
    scope_depth_curve,
    write_budget_audit,
)


def _paper(corpus_id: str, *, title: str | None = None) -> Paper:
    return Paper(
        title=title or f"Paper {corpus_id}",
        sources=["local_bm25"],
        identifiers=PaperIdentifiers(s2orc_corpus_id=corpus_id),
    )


def _ranked(corpus_id: str, rank: int, *, score: float = 1.0) -> RankedDocument:
    return RankedDocument(
        paper=_paper(corpus_id), corpus_id=corpus_id, rank=rank, score=score
    )


def test_merge_ranked_lists_honors_quota_identity_and_source_cap() -> None:
    first = [_ranked("1", 1), _ranked("2", 2), _ranked("3", 3)]
    second = [_ranked("2", 1), _ranked("4", 2), _ranked("5", 3)]
    pool = merge_ranked_lists(
        [first, second], per_list_depth=2, candidate_limit=3
    )
    assert [paper.identifiers.s2orc_corpus_id for paper in pool.papers] == [
        "1",
        "2",
        "4",
    ]
    assert pool.raw_count == 4
    assert pool.deduplicated_count == 3
    assert pool.duplicate_count == 1
    assert pool.truncated_count == 0

    capped = merge_ranked_lists(
        [first, second], per_list_depth=3, candidate_limit=4
    )
    assert capped.raw_count == 6
    assert capped.deduplicated_count == 5
    assert capped.duplicate_count == 1
    assert capped.truncated_count == 1


def test_merge_preserves_conflicting_stable_identities() -> None:
    first = _ranked("1", 1)
    second = RankedDocument(
        paper=_paper("2", title=first.paper.title),
        corpus_id="2",
        rank=1,
        score=1.0,
    )
    pool = merge_ranked_lists(
        [[first], [second]], per_list_depth=1, candidate_limit=20
    )
    assert pool.deduplicated_count == 2
    assert pool.duplicate_count == 0


def test_offline_index_uses_stable_tie_break_and_is_deterministic(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id":"2","title":"same","text":"text"}\n'
        '{"_id":"1","title":"same","text":"text"}\n'
        '{"_id":"3","title":"Unicode β","text":"response"}\n',
        encoding="utf-8",
    )
    index = OfflineLocalBM25Index(corpus)
    first = index.rank("absent token", limit=3)
    second = index.rank("absent token", limit=3)
    assert [item.corpus_id for item in first] == ["1", "2", "3"]
    assert first == second
    assert index.rank("β response", limit=1)[0].corpus_id == "3"


def test_scope_curve_covers_multi_gold_and_top_k_boundaries() -> None:
    papers = [_paper(str(index)) for index in range(1, 202)]
    queries = {
        "q": EvalQuery(
            query_id="q",
            query="query",
            gold_papers=[
                EvalGoldPaper(s2orc_corpus_id="20"),
                EvalGoldPaper(s2orc_corpus_id="50"),
                EvalGoldPaper(s2orc_corpus_id="100"),
                EvalGoldPaper(s2orc_corpus_id="200"),
            ],
        )
    }
    for gold in queries["q"].gold_papers:
        gold.metadata["evaluator_crosswalk"] = {"status": "success"}
    curve = scope_depth_curve({"q": papers}, queries)
    assert [
        curve[str(depth)]["matched_gold_relation_count"]
        for depth in (20, 50, 100, 200)
    ] == [1, 2, 3, 4]
    assert curve["200"]["micro_candidate_recall"] == 1.0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"current_scope_hit": True, "formal_candidate_hit": True},
            "current_connector_candidate",
        ),
        (
            {"original_top_200_hit": True},
            "per_query_adapter_quota",
        ),
        (
            {"any_rank_within_adapter_limit": True},
            "cross_query_identity_deduplication",
        ),
        (
            {
                "current_scope_hit": True,
                "current_merged_position": 201,
            },
            "local_source_pool_truncation",
        ),
        (
            {
                "current_scope_hit": True,
                "formal_retrieval_hit": True,
                "formal_deduplicated_hit": False,
            },
            "cross_query_identity_deduplication",
        ),
        (
            {
                "current_scope_hit": True,
                "formal_deduplicated_hit": True,
            },
            "global_candidate_budget",
        ),
        (
            {"exact_corpus_identity_available": False},
            "identity_matching_gap",
        ),
        ({}, "query_mismatch_top_200"),
    ],
)
def test_gold_budget_terminal_classification_is_closed(
    overrides: dict[str, object], expected: str
) -> None:
    values: dict[str, object] = {
        "current_scope_hit": False,
        "formal_candidate_hit": False,
        "any_rank_within_adapter_limit": False,
        "current_merged_position": None,
        "local_source_limit": 200,
        "formal_retrieval_hit": False,
        "formal_deduplicated_hit": False,
        "original_top_200_hit": False,
        "all_subqueries_top_200_hit": False,
        "exact_corpus_identity_available": True,
    }
    values.update(overrides)
    assert classify_gold_budget_gap(**values) == expected  # type: ignore[arg-type]


def test_budget_audit_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"case_id": "a", "scopes": {"current": {"raw_count": 2}}}]
    gold = [{"case_id": "a", "gold_index": 0, "terminal_class": "success"}]
    aggregate = {"schema_version": "1", "depths": [20, 50, 100, 200]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_budget_audit(first, cases, gold, aggregate)
    write_budget_audit(second, cases, gold, aggregate)
    for name in ("case_audit.jsonl", "gold_chains.jsonl", "aggregate.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
