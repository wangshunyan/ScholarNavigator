from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.scifact_bm25_audit import (
    DeterministicBM25Index,
    _audit_query,
    aggregate_rows,
    load_corpus,
    load_external_candidate_hits,
    tokenize,
    write_artifacts,
)


def _gold(corpus_id: str, *, status: str = "success") -> EvalGoldPaper:
    return EvalGoldPaper(
        title="Gold title",
        s2orc_corpus_id=corpus_id,
        metadata={"evaluator_crosswalk": {"status": status}},
    )


def test_tokenizer_and_title_only_document_support_unicode_and_missing_abstract() -> None:
    assert tokenize("Café、病毒 COVID-19") == ["café", "病毒", "covid", "19"]
    index = DeterministicBM25Index(
        [
            {"corpus_id": "1", "title": "病毒 response", "abstract": ""},
            {"corpus_id": "2", "title": "unrelated", "abstract": ""},
        ]
    )
    assert index.rank("病毒")[0][0] == "1"
    assert index.config()["parameters"] == index.config()["library_defaults"] == {
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
    }


def test_duplicate_content_is_retained_and_zero_score_ties_use_document_id() -> None:
    index = DeterministicBM25Index(
        [
            {"corpus_id": "2", "title": "same", "abstract": "text"},
            {"corpus_id": "1", "title": "same", "abstract": "text"},
            {"corpus_id": "3", "title": "different", "abstract": ""},
        ]
    )
    assert [item[0] for item in index.rank("absent token")] == ["1", "2", "3"]


def test_load_corpus_rejects_duplicate_document_id(tmp_path: Path) -> None:
    root = tmp_path / "scifact"
    root.mkdir()
    (root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "title": "a", "text": ""}),
                json.dumps({"_id": "1", "title": "b", "text": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate SciFact corpus ID"):
        load_corpus(root)


def test_multi_gold_depth_boundaries_and_mutually_exclusive_classification() -> None:
    query = EvalQuery(
        query_id="q",
        query="original query only",
        gold_papers=[
            _gold("at20"),
            _gold("at50"),
            _gold("at201"),
            _gold("never"),
            _gold("unavailable", status="failed"),
        ],
    )
    filler = [
        (f"filler-{rank:03d}", float(400 - rank))
        for rank in range(1, 301)
    ]
    for corpus_id, _score, target_rank in (
        ("at20", 0.0, 20),
        ("at50", 0.0, 50),
        ("at201", 0.0, 201),
        ("never", 0.0, 300),
    ):
        filler[target_rank - 1] = (corpus_id, float(400 - target_rank))
    external = {
        ("q", 0): True,
        ("q", 1): False,
        ("q", 2): True,
        ("q", 3): False,
        ("q", 4): False,
    }
    row = _audit_query(query, filler, external)
    aggregate = aggregate_rows([row])
    assert [
        row["depths"][str(k)]["matched_gold_count"]
        for k in (20, 50, 100, 200)
    ] == [1, 2, 2, 2]
    assert row["first_hit_rank"] == 20
    assert row["reciprocal_rank"] == 0.05
    assert aggregate["classification_counts"] == {
        "bm25_top_200_only": 1,
        "both_hit": 1,
        "external_only": 1,
        "identity_unresolvable": 1,
        "neither_hit": 1,
    }
    assert aggregate["evaluable_gold_count"] == 4
    assert aggregate["gold_first_hit_distribution"]["not_in_top_200"] == 2


def test_external_candidate_matching_requires_stable_identity(tmp_path: Path) -> None:
    query = EvalQuery(
        query_id="q",
        query="query",
        gold_papers=[
            EvalGoldPaper(
                title="Same visible title",
                doi="10.1000/exact",
                s2orc_corpus_id="10",
                metadata={"evaluator_crosswalk": {"status": "success"}},
            ),
            _gold("11"),
        ],
    )
    row = {
        "case_id": "q",
        "status": "succeeded",
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": "initial_retrieval",
                    "status": "completed",
                    "candidates": [
                        {
                            "title": "Different title",
                            "identifiers": {"doi": "10.1000/exact"},
                        },
                        {
                            "title": "Gold title",
                            "year": 2020,
                            "identifiers": {},
                        },
                    ],
                }
            ]
        },
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_external_candidate_hits(run, [query]) == {
        ("q", 0): True,
        ("q", 1): False,
    }


def test_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    config = {"bm25": {"k1": 1.5}, "query": "原始 query"}
    rows = [{"query_id": "1", "top_200": [{"corpus_id": "d", "score": 1.0}]}]
    aggregate = {"depths": {"20": {"matched_gold_count": 1}}}
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = write_artifacts(first, config, rows, aggregate)
    second_hashes = write_artifacts(second, config, rows, aggregate)
    assert first_hashes == second_hashes
    assert {
        path.name: path.read_bytes() for path in first.iterdir()
    } == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
