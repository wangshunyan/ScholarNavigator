from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.local_bm25_budget_audit import (
    OfflineLocalBM25Index,
    RankedDocument,
)
from scholar_agent.evaluation.local_bm25_original_deepening import (
    FrozenCall,
    OriginalDeepeningPolicy,
    build_candidate_pool,
    classify_gold_conversion,
    effective_local_list_depth,
    write_original_deepening_artifacts,
)


def _local_paper(value: int | str) -> Paper:
    corpus_id = str(value)
    return Paper(
        title=f"Local {corpus_id}",
        abstract="shared terms",
        sources=["local_bm25"],
        identifiers=PaperIdentifiers(s2orc_corpus_id=corpus_id),
    )


def _external_paper(value: int | str) -> Paper:
    identifier = str(value)
    return Paper(
        title=f"External {identifier}",
        sources=["openalex"],
        identifiers=PaperIdentifiers(doi=f"10.1/{identifier}"),
    )


class _FakeIndex:
    def __init__(self, size: int = 200) -> None:
        self.values = tuple(
            RankedDocument(
                paper=_local_paper(rank),
                corpus_id=str(rank),
                rank=rank,
                score=float(size - rank),
            )
            for rank in range(1, size + 1)
        )

    def rank(self, query: str, *, limit: int = 200):  # noqa: ANN201
        assert query == "original"
        return self.values[:limit]


def _call(
    *,
    order: int,
    source: str,
    query: str,
    purpose: str,
    papers: list[Paper],
    strategy: str = "safe_original",
) -> FrozenCall:
    return FrozenCall(
        order=order,
        source=source,
        adapted_query=query,
        adaptation_strategy=strategy,
        purposes=(purpose,),
        origin_subqueries=(query,),
        snapshot_key=f"{order:064d}",
        status="success",
        limit=20,
        papers=tuple(papers),
    )


def test_policy_is_disabled_by_default_and_only_deepens_eligible_list() -> None:
    default = OriginalDeepeningPolicy()
    assert not default.enabled
    assert effective_local_list_depth(
        source="local_bm25",
        purpose="original_query",
        adaptation_strategy="safe_original",
        recorded_limit=20,
    ) == 20
    enabled = OriginalDeepeningPolicy(enabled=True)
    assert effective_local_list_depth(
        enabled,
        source="local_bm25",
        purpose="original_query",
        adaptation_strategy="safe_original",
        recorded_limit=20,
    ) == 200
    for source, purpose, strategy in (
        ("arxiv", "original_query", "safe_original"),
        ("local_bm25", "normalized_keywords", "safe_original"),
        ("local_bm25", "original_query", "compact_core"),
    ):
        assert effective_local_list_depth(
            enabled,
            source=source,
            purpose=purpose,
            adaptation_strategy=strategy,
            recorded_limit=20,
        ) == 20


def test_deepening_uses_old_snapshot_prefix_and_honors_depth_boundary() -> None:
    index = _FakeIndex()
    original = _call(
        order=0,
        source="local_bm25",
        query="original",
        purpose="original_query",
        papers=[_local_paper(value) for value in range(1, 21)],
    )
    selected = [{"query": "original"}]
    baseline = build_candidate_pool(
        calls=[original],
        selected_subqueries=selected,
        source_order=["local_bm25"],
        index=index,  # type: ignore[arg-type]
    )
    experiment = build_candidate_pool(
        calls=[original],
        selected_subqueries=selected,
        source_order=["local_bm25"],
        index=index,  # type: ignore[arg-type]
        policy=OriginalDeepeningPolicy(enabled=True),
    )
    assert len(baseline.candidates) == 20
    assert len(experiment.candidates) == 200
    assert experiment.candidates[-1].identifiers.s2orc_corpus_id == "200"


def test_cross_query_duplicate_is_merged_without_changing_derived_list() -> None:
    index = _FakeIndex()
    original = _call(
        order=0,
        source="local_bm25",
        query="original",
        purpose="original_query",
        papers=[_local_paper(value) for value in range(1, 21)],
    )
    derived = _call(
        order=1,
        source="local_bm25",
        query="derived",
        purpose="normalized_keywords",
        papers=[_local_paper(21), _local_paper(201)],
    )
    pool = build_candidate_pool(
        calls=[original, derived],
        selected_subqueries=[{"query": "original"}, {"query": "derived"}],
        source_order=["local_bm25"],
        index=index,  # type: ignore[arg-type]
        policy=OriginalDeepeningPolicy(
            enabled=True,
            local_source_candidate_limit=201,
            global_candidate_limit=201,
        ),
    )
    ids = [paper.identifiers.s2orc_corpus_id for paper in pool.candidates]
    assert ids.count("21") == 1
    assert ids[-1] == "201"
    assert derived.papers == (_local_paper(21), _local_paper(201))


def test_global_budget_uses_stable_source_round_robin() -> None:
    index = _FakeIndex(size=4)
    local = _call(
        order=0,
        source="local_bm25",
        query="original",
        purpose="original_query",
        papers=[_local_paper(value) for value in range(1, 5)],
    )
    external = _call(
        order=1,
        source="openalex",
        query="original",
        purpose="original_query",
        papers=[_external_paper(value) for value in range(1, 4)],
    )
    pool = build_candidate_pool(
        calls=[local, external],
        selected_subqueries=[{"query": "original"}],
        source_order=["openalex", "local_bm25"],
        index=index,  # type: ignore[arg-type]
        policy=OriginalDeepeningPolicy(
            enabled=True,
            original_depth=4,
            local_source_candidate_limit=4,
            global_candidate_limit=4,
        ),
    )
    assert [paper.title for paper in pool.candidates] == [
        "External 1",
        "Local 1",
        "External 2",
        "Local 2",
    ]
    assert pool.global_truncated_count == 3


def test_offline_index_has_stable_tie_order(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id":"2","title":"same","text":"words"}\n'
        '{"_id":"1","title":"same","text":"words"}\n',
        encoding="utf-8",
    )
    index = OfflineLocalBM25Index(corpus)
    assert [item.corpus_id for item in index.rank("missing", limit=2)] == [
        "1",
        "2",
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"baseline_candidate": True}, "baseline_candidate_preserved"),
        (
            {"local_source_retained": False, "original_rank": 21},
            "deep_candidate_local_source_truncated",
        ),
        (
            {
                "pre_global_retained": False,
                "experimental_candidate": False,
                "original_rank": 21,
            },
            "deep_candidate_global_budget_truncated",
        ),
        (
            {"judgement_category": "irrelevant", "original_rank": 21},
            "deep_candidate_relevance_filtered",
        ),
        (
            {
                "judgement_category": "partially_relevant",
                "final_rank": 21,
                "original_rank": 21,
            },
            "deep_candidate_ranked_outside_top_20",
        ),
        (
            {
                "judgement_category": "partially_relevant",
                "final_rank": 20,
                "returned": True,
                "original_rank": 21,
            },
            "newly_returned",
        ),
        ({"original_rank": None}, "query_mismatch_top_200"),
    ],
)
def test_gold_conversion_terminal_is_closed(
    values: dict[str, object], expected: str
) -> None:
    inputs: dict[str, object] = {
        "baseline_candidate": False,
        "experimental_candidate": True,
        "original_rank": None,
        "local_source_retained": True,
        "pre_global_retained": True,
        "judgement_category": None,
        "final_rank": None,
        "returned": False,
    }
    inputs.update(values)
    assert classify_gold_conversion(**inputs) == expected  # type: ignore[arg-type]


def test_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"case_id": "1", "baseline": {"candidate_count": 2}}]
    candidates = [{"case_id": "1", "candidate_id": "s2orc:1"}]
    gold = [{"case_id": "1", "terminal_class": "query_mismatch_top_200"}]
    aggregate = {"benchmark": "test", "values": [2, 1]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_original_deepening_artifacts(
        first, cases, candidates, gold, aggregate
    )
    write_original_deepening_artifacts(
        second, cases, candidates, gold, aggregate
    )
    for name in (
        "case_comparison.jsonl",
        "deep_candidates.jsonl",
        "gold_conversion.jsonl",
        "aggregate.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
