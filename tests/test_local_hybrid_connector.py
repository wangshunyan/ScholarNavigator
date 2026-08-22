from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scholar_agent.connectors.local_hybrid as local_hybrid_module
from scholar_agent.connectors.local_bm25 import LocalBM25Config
from scholar_agent.connectors.local_bm25 import LocalBM25FieldConfig
from scholar_agent.connectors.local_hybrid import LocalHybridConfig
from scholar_agent.connectors.local_hybrid import build_local_hybrid_index
from scholar_agent.connectors.local_hybrid import _fuse_ranked_lists
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.agents.neural_reranker import NeuralRerankResult


def _paper(
    arxiv_id: str,
    title: str,
    *,
    abstract: str = "",
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=sources or [],
    )


def test_local_hybrid_rrf_merges_channels_and_prefers_shared_candidates() -> None:
    bm25 = [
        _paper("a", "Graph retrieval methods", sources=["local_bm25"]),
        _paper("b", "Causal bandits", sources=["local_bm25"]),
    ]
    semantic = [
        _paper(
            "b",
            "Causal bandits",
            abstract="Interventions selected through causal inference.",
            sources=["local_semantic"],
        ),
        _paper("c", "Graph neural retrieval"),
    ]

    fused = _fuse_ranked_lists(bm25, semantic, limit=3, rrf_k=60)

    assert [item.identifiers.arxiv_id for item in fused] == ["b", "a", "c"]
    assert fused[0].abstract.startswith("Interventions")
    assert fused[0].sources == ["local_hybrid", "local_bm25", "local_semantic"]


def test_local_hybrid_rrf_is_deterministic_for_equal_scores() -> None:
    left = [_paper("b", "B"), _paper("a", "A")]
    right = [_paper("a", "A"), _paper("b", "B")]

    first = _fuse_ranked_lists(left, right, limit=2, rrf_k=60)
    second = _fuse_ranked_lists(left, right, limit=2, rrf_k=60)

    assert [item.identifiers.arxiv_id for item in first] == [
        item.identifiers.arxiv_id for item in second
    ]


def test_semantic_row_preserves_extended_metadata() -> None:
    paper = local_hybrid_module._paper_from_semantic_row(
        {
            "_id": "2401.00001",
            "title": "Metadata paper",
            "abstract": "Abstract",
            "authors": [{"name": "Alice"}, "Bob", {"full_name": "Alice"}],
            "year": "2024",
            "venue": "NeurIPS",
            "doi": "https://doi.org/10.1234/example",
        },
        sources=["local_semantic"],
    )
    assert paper.authors == ["Alice", "Bob"]
    assert paper.year == 2024
    assert paper.venue == "NeurIPS"
    assert paper.identifiers.doi == "10.1234/example"


def test_semantic_corpus_requires_unique_stable_arxiv_ids(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"title":"No identity","abstract":"A"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing_stable_arxiv_id"):
        local_hybrid_module._read_semantic_rows(missing)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"_id":"2501.00001","title":"A"}\n'
        '{"_id":"arxiv:2501.00001v2","title":"B"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_arxiv_id"):
        local_hybrid_module._read_semantic_rows(duplicate)


class _FakeEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            value = float(sum(ord(char) for char in text) % 11 + 1)
            vector = np.array([value, value + 1.0, value + 2.0], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


@pytest.mark.skipif(
    pytest.importorskip("faiss", reason="Faiss optional dependency") is None,
    reason="Faiss optional dependency",
)
def test_build_persists_faiss_and_reports_recall(tmp_path: Path, monkeypatch) -> None:
    semantic_corpus = tmp_path / "semantic.jsonl"
    semantic_corpus.write_text(
        "".join(
            json.dumps(
                {
                    "_id": f"2501.0000{i}",
                    "arxiv_id": f"2501.0000{i}",
                    "title": f"Paper {i}",
                    "abstract": f"Abstract {i}",
                }
            )
            + "\n"
            for i in range(1, 6)
        ),
        encoding="utf-8",
    )
    bm25_corpus = tmp_path / "bm25.jsonl"
    bm25_corpus.write_text(
        '{"_id":"2501.00001","title":"Paper 1","abstract":""}\n',
        encoding="utf-8",
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(local_hybrid_module, "_load_model_from_path", lambda _path: _FakeEncoder())
    monkeypatch.setattr(local_hybrid_module, "_model_fingerprint", lambda _path: "fake-model")

    metadata = build_local_hybrid_index(
        LocalHybridConfig(
            bm25_config=LocalBM25Config(
                corpus_path=bm25_corpus,
                cache_dir=tmp_path / "bm25-cache",
                fields=LocalBM25FieldConfig(
                    document_id="_id",
                    title="title",
                    abstract="abstract",
                    document_id_identity="arxiv_id",
                    arxiv_id="arxiv_id",
                ),
            ),
            semantic_corpus_path=semantic_corpus,
            semantic_index_dir=tmp_path / "index",
            model_path=model_path,
            recall_sample_size=5,
            recall_k=3,
        )
    )

    index_dir = tmp_path / "index"
    persisted = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "embeddings.npy").is_file()
    assert persisted["schema_version"] == "2"
    assert persisted["index_type"] == "hnsw_ip"
    assert persisted["recall_query_count"] == 5
    assert persisted["field_completeness"] == {
        "title": 1.0,
        "abstract": 1.0,
        "authors": 0.0,
        "year": 0.0,
        "venue": 0.0,
        "doi": 0.0,
    }
    assert 0.0 <= persisted["ann_recall_at_k"] <= 1.0
    assert metadata.index_fingerprint == persisted["index_fingerprint"]


def test_old_index_schema_is_rejected(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "metadata.json").write_text(
        json.dumps({"schema_version": "1"}), encoding="utf-8"
    )
    config = LocalHybridConfig(
        bm25_config=LocalBM25Config(
            corpus_path=tmp_path / "bm25.jsonl",
            cache_dir=tmp_path / "bm25-cache",
        ),
        semantic_corpus_path=tmp_path / "semantic.jsonl",
        semantic_index_dir=index_dir,
        model_path=tmp_path / "model",
    )
    with pytest.raises(ValueError, match="schema_mismatch"):
        local_hybrid_module._load_index_metadata(config)


def test_reranker_configuration_requires_a_local_model_directory(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic.jsonl"
    bm25 = tmp_path / "bm25.jsonl"
    encoder = tmp_path / "encoder"
    semantic.write_text('{"arxiv_id":"2501.00001","title":"Paper","abstract":"A"}\n')
    bm25.write_text('{"_id":"2501.00001","title":"Paper","abstract":""}\n')
    encoder.mkdir()
    config = LocalHybridConfig(
        bm25_config=LocalBM25Config(corpus_path=bm25, cache_dir=tmp_path / "cache"),
        semantic_corpus_path=semantic,
        semantic_index_dir=tmp_path / "index",
        model_path=encoder,
        reranker_model_path=tmp_path / "missing-reranker",
    )

    with pytest.raises(ValueError, match="reranker_model_not_found"):
        local_hybrid_module._normalize_config(config)


def test_reranker_configuration_accepts_explicit_cuda_index(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic.jsonl"
    bm25 = tmp_path / "bm25.jsonl"
    encoder = tmp_path / "encoder"
    reranker_model = tmp_path / "reranker"
    semantic.write_text('{"arxiv_id":"2501.00001","title":"Paper","abstract":"A"}\n')
    bm25.write_text('{"_id":"2501.00001","title":"Paper","abstract":""}\n')
    encoder.mkdir()
    reranker_model.mkdir()
    config = LocalHybridConfig(
        bm25_config=LocalBM25Config(corpus_path=bm25, cache_dir=tmp_path / "cache"),
        semantic_corpus_path=semantic,
        semantic_index_dir=tmp_path / "index",
        model_path=encoder,
        reranker_model_path=reranker_model,
        reranker_device="cuda:1",
    )

    assert local_hybrid_module._normalize_config(config).reranker_device == "cuda:1"


def test_connector_reports_reranker_diagnostics_and_uses_candidate_pool(
    tmp_path: Path, monkeypatch
) -> None:
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    reranker_model = tmp_path / "reranker"
    reranker_model.mkdir()
    config = LocalHybridConfig(
        bm25_config=LocalBM25Config(corpus_path=tmp_path / "bm25.jsonl", cache_dir=tmp_path / "cache"),
        semantic_corpus_path=tmp_path / "semantic.jsonl",
        semantic_index_dir=tmp_path / "index",
        model_path=encoder,
        reranker_model_path=reranker_model,
        reranker_candidate_limit=3,
    )
    papers = [_paper(str(index), f"Paper {index}") for index in range(4)]
    observed: list[int] = []

    class FakeReranker:
        def rerank(self, _query, candidates, *, limit):
            observed.append(len(candidates))
            return NeuralRerankResult(
                papers=list(reversed(candidates))[:limit],
                latency_seconds=0.25,
                model_fingerprint="reranker-fingerprint",
                batch_count=2,
            )

    monkeypatch.setattr(local_hybrid_module, "_ACTIVE_CONFIG", config)
    monkeypatch.setattr(local_hybrid_module, "_ACTIVE_RERANKER", FakeReranker())
    monkeypatch.setattr(
        local_hybrid_module,
        "_active_config_and_index",
        lambda: (config, object()),
    )
    monkeypatch.setattr(
        local_hybrid_module,
        "search_local_bm25_detailed",
        lambda *_args, **_kwargs: ConnectorSearchResult(papers=papers),
    )
    monkeypatch.setattr(
        local_hybrid_module,
        "_search_semantic",
        lambda *_args, **_kwargs: ConnectorSearchResult(papers=papers),
    )

    result = local_hybrid_module.search_local_hybrid_detailed("query", limit=2)

    assert result.error_message is None
    assert observed == [3]
    assert result.diagnostics.local_model_latency_seconds == 0.25
    assert result.diagnostics.local_model_batch_count == 2
    assert result.diagnostics.local_model_fingerprint == "reranker-fingerprint"
