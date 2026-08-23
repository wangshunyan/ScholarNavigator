from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from scholar_agent.connectors.local_bm25 import (
    LocalBM25Config,
    LocalBM25FieldConfig,
    configure_local_bm25,
    local_bm25_connector_version,
    tokenize_local_bm25_query,
    search_local_bm25_detailed,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    DEFAULT_SEARCH_SOURCES,
    SUPPORTED_SEARCH_SOURCES,
    normalize_search_sources,
)
from scholar_agent.evaluation.snapshots import (
    SnapshotManifest,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.snapshots.store import connector_version, utc_now
from scholar_agent.services.search_service import _stable_source_coverage_truncate


@pytest.fixture(autouse=True)
def clear_local_connector() -> None:
    configure_local_bm25(None)
    yield
    configure_local_bm25(None)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(corpus: Path, cache: Path) -> LocalBM25Config:
    return LocalBM25Config(
        corpus_path=corpus,
        cache_dir=cache,
        fields=LocalBM25FieldConfig(
            document_id="_id",
            title="title",
            abstract="text",
            document_id_identity="s2orc_corpus_id",
            doi="metadata.doi",
        ),
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "_id": "2",
            "title": "β细胞与 immune response",
            "text": "Unicode 炎症 signaling",
            "authors": ["A. Author", "B. Author"],
            "year": 2024,
            "venue": "NeurIPS",
            "metadata": {"doi": "10.1/two"},
        },
        {
            "_id": "1",
            "title": "Immune response in beta cells",
            "text": "inflammation signaling pathway",
            "metadata": {},
        },
        {
            "_id": "3",
            "title": "Document without abstract",
            "metadata": {},
        },
    ]


def test_local_bm25_is_default_off_and_explicitly_supported() -> None:
    assert "local_bm25" not in DEFAULT_SEARCH_SOURCES
    assert "local_bm25" in SUPPORTED_SEARCH_SOURCES
    assert normalize_search_sources(None) == list(DEFAULT_SEARCH_SOURCES)
    assert normalize_search_sources(["local-bm25"]) == ["local_bm25"]
    config_fields = {item.name for item in dataclass_fields(LocalBM25Config)}
    assert config_fields == {"corpus_path", "cache_dir", "fields", "k1", "b", "epsilon"}
    assert config_fields.isdisjoint({"qrels", "gold", "case_id", "crosswalk"})


def test_local_bm25_builds_deterministically_and_preserves_identity(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _rows())
    metadata = configure_local_bm25(_config(corpus, tmp_path / "cache"))
    first = search_local_bm25_detailed("immune response 炎症", 3)
    second = search_local_bm25_detailed("immune response 炎症", 3)

    assert metadata is not None
    assert metadata.document_count == 3
    assert metadata.field_completeness == {
        "title": 1.0,
        "abstract": 2 / 3,
        "authors": 1 / 3,
        "year": 1 / 3,
        "venue": 1 / 3,
        "doi": 1 / 3,
    }
    assert [paper.model_dump(mode="json") for paper in first.papers] == [
        paper.model_dump(mode="json") for paper in second.papers
    ]
    assert [paper.identifiers.s2orc_corpus_id for paper in first.papers] == [
        "2",
        "1",
        "3",
    ]
    assert first.papers[0].identifiers.doi == "10.1/two"
    assert first.papers[0].authors == ["A. Author", "B. Author"]
    assert first.papers[0].year == 2024
    assert first.papers[0].venue == "NeurIPS"
    assert first.papers[0].sources == ["local_bm25"]
    assert first.papers[-1].abstract == ""


def test_local_bm25_normalizes_structured_author_names(tmp_path: Path) -> None:
    corpus = tmp_path / "authors.jsonl"
    _write_jsonl(corpus, [{
        "_id": "1",
        "title": "Structured author paper",
        "authors": [{"name": "Ada Lovelace"}, {"full_name": "Alan Turing"}],
        "year": "2023",
        "venue": "ICLR",
    }])
    configure_local_bm25(LocalBM25Config(corpus_path=corpus, cache_dir=tmp_path / "cache"))
    paper = search_local_bm25_detailed("structured author", 1).papers[0]
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.year == 2023
    assert paper.venue == "ICLR"


def test_local_bm25_query_tokenizer_removes_question_filler() -> None:
    tokens = tokenize_local_bm25_query(
        "Could you list papers about graph retrieval methods?"
    )

    assert tokens == ["graph", "retrieval", "methods"]


def test_local_bm25_query_tokenizer_keeps_short_or_all_stopword_query() -> None:
    assert tokenize_local_bm25_query("AI") == ["ai"]
    assert tokenize_local_bm25_query("papers") == ["papers"]


def test_local_bm25_query_filler_does_not_outweigh_topic_terms(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"_id": "noise", "title": "Can you tell me about papers", "text": ""},
            {"_id": "topic", "title": "Graph Retrieval Methods", "text": ""},
        ],
    )
    configure_local_bm25(_config(corpus, tmp_path / "cache"))

    result = search_local_bm25_detailed(
        "Could you tell me about graph retrieval papers?",
        2,
    )

    assert result.papers[0].identifiers.s2orc_corpus_id == "topic"


def test_local_bm25_cache_hit_and_corpus_change_invalidate_fingerprint(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    cache = tmp_path / "cache"
    _write_jsonl(corpus, _rows())
    first = configure_local_bm25(_config(corpus, cache))
    configure_local_bm25(None)
    second = configure_local_bm25(_config(corpus, cache))
    assert first is not None and second is not None
    assert first.fingerprint == second.fingerprint
    assert second.cache_hit is True
    assert second.field_completeness == first.field_completeness

    changed = [*_rows(), {"_id": "4", "title": "new paper", "text": "new"}]
    _write_jsonl(corpus, changed)
    third = configure_local_bm25(_config(corpus, cache))
    assert third is not None
    assert third.fingerprint != first.fingerprint
    assert third.cache_hit is False
    assert Path(first.cache_path).is_file()
    assert Path(third.cache_path).is_file()


def test_local_bm25_duplicate_rows_are_stable_and_conflicts_fail(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = _rows()
    _write_jsonl(corpus, [rows[0], rows[0]])
    metadata = configure_local_bm25(_config(corpus, tmp_path / "cache"))
    assert metadata is not None and metadata.document_count == 1

    conflicting = dict(rows[0])
    conflicting["title"] = "different title"
    _write_jsonl(corpus, [rows[0], conflicting])
    with pytest.raises(ValueError, match="conflicting_document"):
        configure_local_bm25(_config(corpus, tmp_path / "other-cache"))


def test_local_bm25_document_identity_conflict_is_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [{"_id": "1", "title": "paper", "text": "text", "cid": "2"}],
    )
    config = LocalBM25Config(
        corpus_path=corpus,
        cache_dir=tmp_path / "cache",
        fields=LocalBM25FieldConfig(
            abstract="text",
            document_id_identity="s2orc_corpus_id",
            s2orc_corpus_id="cid",
        ),
    )
    with pytest.raises(ValueError, match="document_identity_conflict"):
        configure_local_bm25(config)


def test_local_bm25_snapshot_record_replay_never_calls_live_search(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _rows())
    configure_local_bm25(_config(corpus, tmp_path / "cache"))
    snapshot_root = tmp_path / "snapshots"
    now = utc_now()
    store = SnapshotStore(snapshot_root)
    store.ensure_manifest(
        SnapshotManifest(
            snapshot_name=snapshot_root.name,
            dataset="synthetic",
            split="test",
            offset=0,
            sources=["local_bm25"],
            adapter_policy="safe_original",
            run_profile="balanced",
            budgets={"max_candidate_papers": 200},
            llm_enabled=False,
            query_understanding_prompt={},
            judgement_prompt={},
            connector_versions={"local_bm25": connector_version("local_bm25")},
            code_hash="a" * 64,
            dirty_worktree=True,
            created_at=now,
            updated_at=now,
        )
    )
    record = SnapshotRuntime(store, mode="record", group_name="baseline")
    recorded = record.search(
        "local_bm25",
        "immune response",
        2,
        "safe_original",
        search_local_bm25_detailed,
    )
    record.finish_group(completed=True)
    replay = SnapshotRuntime(store, mode="replay", group_name="baseline")
    replayed = replay.search(
        "local_bm25",
        "immune response",
        2,
        "safe_original",
        lambda query, limit: pytest.fail("Replay must not invoke local search"),
    )

    assert [paper.model_dump() for paper in replayed.papers] == [
        paper.model_dump() for paper in recorded.papers
    ]
    assert replayed.snapshot_hit is True
    assert replayed.diagnostics.request_count == 0
    assert replayed.recorded_diagnostics is not None


def test_local_bm25_candidates_share_the_frozen_global_budget() -> None:
    papers = [
        Paper(
            title=f"Paper {index}",
            identifiers=PaperIdentifiers(s2orc_corpus_id=str(index)),
            sources=["local_bm25" if index % 2 else "openalex"],
        )
        for index in range(225)
    ]
    selected = _stable_source_coverage_truncate(
        papers,
        limit=200,
        source_order=["openalex", "arxiv", "semantic_scholar", "pubmed", "local_bm25"],
    )
    assert len(selected) == 200
    assert {source for paper in selected for source in paper.sources} == {
        "openalex",
        "local_bm25",
    }


def test_local_bm25_connector_version_binds_fields_and_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _rows())
    configure_local_bm25(_config(corpus, tmp_path / "cache"), build_index=False)
    first = local_bm25_connector_version()
    changed = LocalBM25Config(
        corpus_path=corpus,
        cache_dir=tmp_path / "cache",
        fields=LocalBM25FieldConfig(abstract="text", title="title", document_id="_id"),
    )
    configure_local_bm25(changed, build_index=False)
    assert local_bm25_connector_version() != first
