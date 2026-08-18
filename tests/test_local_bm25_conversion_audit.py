from __future__ import annotations

from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    FrozenLocalBM25Scorer,
    classify_gold_terminal,
    gold_first_oracle,
    rank_by_local_best,
    write_conversion_audit,
)


def _candidate(
    corpus_id: str,
    *,
    rank: int,
    query: str = "q",
    list_rank: int | None = None,
    extra_provenance: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    provenance: list[dict[str, object]] = []
    if list_rank is not None:
        provenance.append(
            {
                "source": "local_bm25",
                "adapted_query": query,
                "source_rank": list_rank,
            }
        )
    provenance.extend(extra_provenance or [])
    return {
        "identifiers": {"s2orc_corpus_id": corpus_id},
        "title": f"Paper {corpus_id}",
        "rank": rank,
        "provenance": provenance,
    }


def test_classification_closes_identity_budget_filter_rank_and_success() -> None:
    common = {"retrieval_match": True, "top_k": 20}
    assert classify_gold_terminal(
        **common,
        deduplicated_match=False,
        identity_merge_evidence=True,
        budget_truncated=False,
        category=None,
        current_rank=None,
        final_returned=False,
    ) == "identity_merge_loss"
    assert classify_gold_terminal(
        **common,
        deduplicated_match=False,
        identity_merge_evidence=False,
        budget_truncated=True,
        category=None,
        current_rank=None,
        final_returned=False,
    ) == "candidate_budget_truncation"
    assert classify_gold_terminal(
        **common,
        deduplicated_match=True,
        identity_merge_evidence=False,
        budget_truncated=False,
        category="weakly_relevant",
        current_rank=3,
        final_returned=False,
    ) == "weak_or_irrelevant_filter"
    assert classify_gold_terminal(
        **common,
        deduplicated_match=True,
        identity_merge_evidence=False,
        budget_truncated=False,
        category="partially_relevant",
        current_rank=21,
        final_returned=False,
    ) == "ranking_outside_top_20"
    assert classify_gold_terminal(
        **common,
        deduplicated_match=True,
        identity_merge_evidence=False,
        budget_truncated=False,
        category="partially_relevant",
        current_rank=20,
        final_returned=True,
    ) == "successfully_returned"


def test_local_rank_uses_best_list_hit_and_is_input_order_independent() -> None:
    duplicate = _candidate(
        "2",
        rank=7,
        query="second",
        list_rank=1,
        extra_provenance=[
            {
                "source": "local_bm25",
                "adapted_query": "first",
                "source_rank": 4,
            }
        ],
    )
    tied_a = _candidate("1", rank=3, query="first", list_rank=1)
    tied_b = _candidate("3", rank=1, query="first", list_rank=1)
    expected = ["s2orc:1", "s2orc:3", "s2orc:2"]
    for values in ([duplicate, tied_b, tied_a], [tied_a, duplicate, tied_b]):
        ranked = rank_by_local_best(
            values, query_order={"first": 0, "second": 1}, top_k=20
        )
        assert [
            f"s2orc:{item['identifiers']['s2orc_corpus_id']}" for item in ranked
        ] == expected


def test_local_rank_excludes_non_local_and_honors_top20_boundary() -> None:
    values = [
        _candidate(str(index), rank=index, list_rank=index)
        for index in range(1, 22)
    ]
    values.append(_candidate("external", rank=1))
    ranked = rank_by_local_best(values, query_order={"q": 0}, top_k=20)
    assert len(ranked) == 20
    assert ranked[-1]["identifiers"]["s2orc_corpus_id"] == "20"


def test_gold_first_oracle_handles_duplicates_and_top20_boundary() -> None:
    values = [_candidate(str(index), rank=index) for index in range(1, 22)]
    gold = [EvalGoldPaper(s2orc_corpus_id="21")]
    ranked = gold_first_oracle(values, gold, top_k=20)
    assert ranked[0]["identifiers"]["s2orc_corpus_id"] == "21"
    assert len(ranked) == 20


def test_frozen_bm25_score_is_deterministic_for_unicode_and_duplicates(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id":"2","title":"β response","text":"immune"}\n'
        '{"_id":"1","title":"alpha","text":"response"}\n'
        '{"_id":"1","title":"alpha","text":"response"}\n',
        encoding="utf-8",
    )
    first = FrozenLocalBM25Scorer(corpus)
    second = FrozenLocalBM25Scorer(corpus)
    assert first.document_ids == ["1", "2"]
    assert first.score("β immune", "2") == second.score("β immune", 2)


def test_audit_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    case_rows = [{"case_id": "1", "variants": {"current": {"value": 1}}}]
    gold_rows = [{"case_id": "1", "gold_id": "s2orc:1"}]
    aggregate = {"schema_version": "1", "value": [2, 1]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_conversion_audit(first, case_rows, gold_rows, aggregate)
    write_conversion_audit(second, case_rows, gold_rows, aggregate)
    for name in ("case_audit.jsonl", "gold_chains.jsonl", "aggregate.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
