"""Offline BM25 plus semantic vector retrieval over a local paper corpus."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from scholar_agent.agents.neural_reranker import (
    NeuralReranker,
    NeuralRerankerConfig,
    NeuralRerankResult,
)
from scholar_agent.connectors.local_bm25 import (
    LocalBM25Config,
    configure_local_bm25,
    search_local_bm25_detailed,
)
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


LOCAL_HYBRID_CONNECTOR_VERSION = "local-hybrid-v2"
LOCAL_HYBRID_INDEX_SCHEMA_VERSION = "2"
DEFAULT_BM25_CANDIDATE_LIMIT = 60
DEFAULT_SEMANTIC_CANDIDATE_LIMIT = 60
DEFAULT_RRF_K = 60
_EMBEDDINGS_FILE = "embeddings.npy"
_METADATA_FILE = "metadata.json"
_FAISS_INDEX_FILE = "index.faiss"
_PARTIAL_EMBEDDINGS_FILE = "embeddings.partial.npy"
_PARTIAL_FAISS_INDEX_FILE = "index.partial.faiss"
_BUILD_PROGRESS_FILE = "build_progress.json"
_BUILD_SCHEMA_VERSION = "2"
_ENCODE_BATCH_SIZE = 128
_FAISS_BATCH_SIZE = 4096
_DEFAULT_HNSW_M = 32
_DEFAULT_EF_CONSTRUCTION = 80
_DEFAULT_EF_SEARCH = 64
_DEFAULT_RECALL_SAMPLE_SIZE = 100
_DEFAULT_RECALL_K = 10
_CONFIG_LOCK = RLock()
_MODEL_LOCK = RLock()
_ACTIVE_CONFIG: "LocalHybridConfig | None" = None
_ACTIVE_METADATA: "LocalHybridIndexMetadata | None" = None
_ACTIVE_INDEX: "_SemanticIndex | None" = None
_ACTIVE_MODEL: Any | None = None
_ACTIVE_MODEL_PATH: Path | None = None
_ACTIVE_RERANKER: NeuralReranker | None = None


@dataclass(frozen=True)
class LocalHybridConfig:
    """Explicit hybrid retrieval inputs.

    The BM25 corpus and semantic corpus are separate on purpose. The former
    can remain the full PaSa title database while the latter is an enriched
    title+abstract subset. Neither input contains evaluator labels.
    """

    bm25_config: LocalBM25Config
    semantic_corpus_path: Path
    semantic_index_dir: Path
    model_path: Path
    bm25_candidate_limit: int = DEFAULT_BM25_CANDIDATE_LIMIT
    semantic_candidate_limit: int = DEFAULT_SEMANTIC_CANDIDATE_LIMIT
    rrf_k: int = DEFAULT_RRF_K
    semantic_search_mode: str = "ann"
    hnsw_m: int = _DEFAULT_HNSW_M
    hnsw_ef_construction: int = _DEFAULT_EF_CONSTRUCTION
    hnsw_ef_search: int = _DEFAULT_EF_SEARCH
    recall_sample_size: int = _DEFAULT_RECALL_SAMPLE_SIZE
    recall_k: int = _DEFAULT_RECALL_K
    reranker_model_path: Path | None = None
    reranker_candidate_limit: int = 120
    reranker_batch_size: int = 8
    reranker_device: str = "auto"


@dataclass(frozen=True)
class LocalHybridIndexMetadata:
    semantic_corpus_sha256: str
    semantic_corpus_size_bytes: int
    document_count: int
    abstract_document_count: int
    embedding_dimension: int
    model_path: str
    model_fingerprint: str
    index_dir: str
    index_fingerprint: str
    index_type: str
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    build_seconds: float
    peak_memory_bytes: int
    ann_recall_at_k: float
    recall_query_count: int
    recall_k: int
    ann_query_seconds: float
    exact_query_seconds: float
    cache_hit: bool
    index_load_seconds: float


@dataclass(frozen=True)
class _SemanticIndex:
    embeddings_path: Path
    corpus_path: Path
    rows: tuple[dict[str, Any], ...]
    document_count: int
    embedding_dimension: int
    index_fingerprint: str
    faiss_index: Any | None


def configure_local_hybrid(
    config: LocalHybridConfig | None,
    *,
    build_bm25_index: bool = True,
) -> LocalHybridIndexMetadata | None:
    """Configure the hybrid connector and validate its persisted vector index."""

    global _ACTIVE_CONFIG, _ACTIVE_METADATA, _ACTIVE_INDEX, _ACTIVE_MODEL
    global _ACTIVE_MODEL_PATH, _ACTIVE_RERANKER
    with _CONFIG_LOCK:
        _ACTIVE_CONFIG = None
        _ACTIVE_METADATA = None
        _ACTIVE_INDEX = None
        with _MODEL_LOCK:
            _ACTIVE_MODEL = None
            _ACTIVE_MODEL_PATH = None
            _ACTIVE_RERANKER = None
        if config is None:
            configure_local_bm25(None)
            return None

        normalized = _normalize_config(config)
        metadata = _load_index_metadata(normalized)
        cache_path = normalized.semantic_index_dir / _METADATA_FILE
        if metadata.semantic_corpus_sha256 != _sha256_file(
            normalized.semantic_corpus_path
        )[0]:
            raise ValueError("local_hybrid_semantic_corpus_changed")
        if metadata.model_path != str(normalized.model_path):
            raise ValueError("local_hybrid_model_path_changed")
        if metadata.document_count <= 0:
            raise ValueError("local_hybrid_empty_index")
        embeddings_path = normalized.semantic_index_dir / _EMBEDDINGS_FILE
        if not embeddings_path.is_file():
            raise ValueError("local_hybrid_embeddings_not_found")
        shape = _embedding_shape(embeddings_path)
        if shape != (metadata.document_count, metadata.embedding_dimension):
            raise ValueError("local_hybrid_embedding_shape_invalid")
        rows = tuple(_read_semantic_rows(normalized.semantic_corpus_path))
        if len(rows) != metadata.document_count:
            raise ValueError("local_hybrid_semantic_corpus_count_changed")
        faiss_index = None
        if metadata.index_type == "hnsw_ip":
            faiss_index = _read_faiss_index(
                normalized.semantic_index_dir / _FAISS_INDEX_FILE,
                metadata.embedding_dimension,
                metadata.document_count,
            )
            _set_faiss_ef_search(faiss_index, normalized.hnsw_ef_search)
        elif metadata.index_type != "exact_flat":
            raise ValueError("local_hybrid_index_type_invalid")

        configure_local_bm25(
            normalized.bm25_config,
            build_index=build_bm25_index,
        )
        if normalized.reranker_model_path is not None:
            _ACTIVE_RERANKER = NeuralReranker(
                NeuralRerankerConfig(
                    model_path=normalized.reranker_model_path,
                    candidate_limit=normalized.reranker_candidate_limit,
                    batch_size=normalized.reranker_batch_size,
                    device=normalized.reranker_device,
                )
            )
        _ACTIVE_CONFIG = normalized
        _ACTIVE_METADATA = metadata
        _ACTIVE_INDEX = _SemanticIndex(
            embeddings_path=embeddings_path,
            corpus_path=normalized.semantic_corpus_path,
            rows=rows,
            document_count=metadata.document_count,
            embedding_dimension=metadata.embedding_dimension,
            index_fingerprint=metadata.index_fingerprint,
            faiss_index=faiss_index,
        )
        return metadata


def local_hybrid_metadata() -> LocalHybridIndexMetadata:
    with _CONFIG_LOCK:
        if _ACTIVE_METADATA is None:
            raise ValueError("local_hybrid_not_configured")
        return _ACTIVE_METADATA


def local_hybrid_connector_version() -> str:
    metadata = local_hybrid_metadata()
    return f"{LOCAL_HYBRID_CONNECTOR_VERSION}:{metadata.index_fingerprint}"


def build_local_hybrid_index(
    config: LocalHybridConfig,
    *,
    resume: bool = False,
) -> LocalHybridIndexMetadata:
    """Encode the enriched local corpus with resumable batch checkpoints."""

    started = time.perf_counter()
    normalized = _normalize_config(config)
    rows = _read_semantic_rows(normalized.semantic_corpus_path)
    if not rows:
        raise ValueError("local_hybrid_semantic_corpus_empty")
    import torch

    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    model = _load_model_from_path(normalized.model_path)
    normalized.semantic_index_dir.mkdir(parents=True, exist_ok=True)
    model_fingerprint = _model_fingerprint(normalized.model_path)
    embedding_dimension = int(model.get_sentence_embedding_dimension() or 0)
    if embedding_dimension <= 0:
        raise ValueError("local_hybrid_embedding_dimension_invalid")
    faiss = _load_faiss()

    embeddings_path = normalized.semantic_index_dir / _EMBEDDINGS_FILE
    metadata_path = normalized.semantic_index_dir / _METADATA_FILE
    faiss_index_path = normalized.semantic_index_dir / _FAISS_INDEX_FILE
    partial_embeddings_path = normalized.semantic_index_dir / _PARTIAL_EMBEDDINGS_FILE
    partial_faiss_path = normalized.semantic_index_dir / _PARTIAL_FAISS_INDEX_FILE
    progress_path = normalized.semantic_index_dir / _BUILD_PROGRESS_FILE
    corpus_sha, corpus_size = _sha256_file(normalized.semantic_corpus_path)
    expected = {
        "schema_version": _BUILD_SCHEMA_VERSION,
        "semantic_corpus_sha256": corpus_sha,
        "model_fingerprint": model_fingerprint,
        "document_count": len(rows),
        "embedding_dimension": embedding_dimension,
        "hnsw_m": normalized.hnsw_m,
        "hnsw_ef_construction": normalized.hnsw_ef_construction,
        "hnsw_ef_search": normalized.hnsw_ef_search,
    }
    progress = _load_build_progress(progress_path)
    if not resume:
        _discard_partial_build(partial_embeddings_path, partial_faiss_path, progress_path)
        progress = None
    elif progress is not None and not _progress_matches(progress, expected):
        raise ValueError("local_hybrid_resume_configuration_changed")
    elif resume and progress is None and (
        partial_embeddings_path.exists() or partial_faiss_path.exists()
    ):
        raise ValueError("local_hybrid_resume_progress_missing")

    peak_memory = _current_rss_bytes()
    embeddings_complete = bool(progress and progress.get("phase") == "faiss")
    if embeddings_complete:
        if _embedding_shape(embeddings_path) != (len(rows), embedding_dimension):
            raise ValueError("local_hybrid_resume_embeddings_invalid")
    else:
        embedding_start = (
            int(progress.get("next_row") or 0)
            if progress and progress.get("phase") == "embeddings"
            else 0
        )
        if embedding_start == 0:
            partial_embeddings_path.unlink(missing_ok=True)
            memmap = np.lib.format.open_memmap(
                partial_embeddings_path, mode="w+", dtype=np.float32,
                shape=(len(rows), embedding_dimension),
            )
            memmap.flush()
            del memmap
        elif _embedding_shape(partial_embeddings_path) != (
            len(rows), embedding_dimension
        ):
            raise ValueError("local_hybrid_resume_embeddings_invalid")

        memmap = np.lib.format.open_memmap(
            partial_embeddings_path, mode="r+", dtype=np.float32,
            shape=(len(rows), embedding_dimension),
        )
        _write_build_progress(
            progress_path,
            {**expected, "phase": "embeddings", "next_row": embedding_start},
        )
        try:
            for start in range(embedding_start, len(rows), _ENCODE_BATCH_SIZE):
                end = min(len(rows), start + _ENCODE_BATCH_SIZE)
                texts = [
                    f"passage: {row['title']}\n{row.get('abstract') or ''}".strip()
                    for row in rows[start:end]
                ]
                batch = np.asarray(
                    model.encode(
                        texts, batch_size=_ENCODE_BATCH_SIZE, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True,
                    ),
                    dtype=np.float32,
                )
                if batch.shape != (end - start, embedding_dimension):
                    raise ValueError("local_hybrid_embedding_generation_invalid")
                memmap[start:end] = batch
                memmap.flush()
                peak_memory = max(peak_memory, _current_rss_bytes())
                _write_build_progress(
                    progress_path,
                    {**expected, "phase": "embeddings", "next_row": end},
                )
            memmap.flush()
        finally:
            del memmap
        os.replace(partial_embeddings_path, embeddings_path)

    index_start = int(progress.get("next_row") or 0) if progress and progress.get("phase") == "faiss" else 0
    if index_start == 0:
        partial_faiss_path.unlink(missing_ok=True)
        ann_index = faiss.IndexHNSWFlat(
            embedding_dimension, normalized.hnsw_m, faiss.METRIC_INNER_PRODUCT
        )
        ann_index.hnsw.efConstruction = normalized.hnsw_ef_construction
    else:
        ann_index = _read_faiss_index(
            partial_faiss_path, embedding_dimension, index_start
        )
    _write_build_progress(progress_path, {**expected, "phase": "faiss", "next_row": index_start})
    matrix = np.load(embeddings_path, mmap_mode="r")
    for start in range(index_start, len(rows), _FAISS_BATCH_SIZE):
        end = min(len(rows), start + _FAISS_BATCH_SIZE)
        ann_index.add(np.asarray(matrix[start:end], dtype=np.float32))
        faiss.write_index(ann_index, str(partial_faiss_path))
        peak_memory = max(peak_memory, _current_rss_bytes())
        _write_build_progress(progress_path, {**expected, "phase": "faiss", "next_row": end})
    if int(ann_index.ntotal) != len(rows):
        raise ValueError("local_hybrid_faiss_document_count_invalid")
    _set_faiss_ef_search(ann_index, normalized.hnsw_ef_search)
    os.replace(partial_faiss_path, faiss_index_path)
    recall, recall_count, ann_query_seconds, exact_query_seconds = _measure_index_recall(
        ann_index, matrix, normalized.recall_sample_size, normalized.recall_k
    )
    index_fingerprint = _index_fingerprint(
        corpus_sha=corpus_sha, model_fingerprint=model_fingerprint,
        shape=(len(rows), embedding_dimension), hnsw_m=normalized.hnsw_m,
        hnsw_ef_construction=normalized.hnsw_ef_construction,
        hnsw_ef_search=normalized.hnsw_ef_search,
    )
    build_seconds = time.perf_counter() - started
    metadata_payload = {
        **expected,
        "schema_version": LOCAL_HYBRID_INDEX_SCHEMA_VERSION,
        "connector_version": LOCAL_HYBRID_CONNECTOR_VERSION,
        "semantic_corpus_size_bytes": corpus_size,
        "abstract_document_count": sum(bool(str(row.get("abstract") or "").strip()) for row in rows),
        "embedding_dimension": embedding_dimension,
        "model_path": str(normalized.model_path),
        "index_fingerprint": index_fingerprint,
        "index_type": "hnsw_ip",
        "embedding_dtype": "float32",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "build_batch_size": _ENCODE_BATCH_SIZE,
        "faiss_batch_size": _FAISS_BATCH_SIZE,
        "build_seconds": build_seconds,
        "peak_memory_bytes": peak_memory,
        "ann_recall_at_k": recall,
        "recall_query_count": recall_count,
        "recall_k": normalized.recall_k,
        "ann_query_seconds": ann_query_seconds,
        "exact_query_seconds": exact_query_seconds,
        "resumable_build": True,
    }
    _atomic_write_text(metadata_path, json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return _metadata_from_payload(metadata_payload, normalized.semantic_index_dir, cache_hit=False)


def search_local_hybrid(
    query: str,
    limit: int = 20,
) -> list[Paper]:
    return search_local_hybrid_detailed(query, limit).papers


def search_local_hybrid_detailed(
    query: str,
    limit: int = 20,
) -> ConnectorSearchResult:
    """Search BM25 and semantic channels, then fuse with deterministic RRF."""

    started = time.perf_counter()
    normalized_query = str(query).strip()
    if not normalized_query or limit <= 0:
        latency = time.perf_counter() - started
        return ConnectorSearchResult(
            warnings=["local_hybrid_empty_query"] if not normalized_query else [],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(latency_seconds=latency),
        )
    try:
        config, index = _active_config_and_index()
        bm25_result = search_local_bm25_detailed(
            normalized_query,
            max(int(limit), config.bm25_candidate_limit),
        )
        semantic_result = _search_semantic(
            normalized_query,
            max(int(limit), config.semantic_candidate_limit),
            config=config,
            index=index,
        )
        fusion_limit = max(
            int(limit),
            config.reranker_candidate_limit
            if config.reranker_model_path is not None
            else int(limit),
        )
        fused = _fuse_ranked_lists(
            bm25_result.papers,
            semantic_result.papers,
            limit=fusion_limit,
            rrf_k=config.rrf_k,
        )
        rerank_outcome = _rerank_if_configured(
            normalized_query,
            fused,
            limit=int(limit),
        )
        fused = rerank_outcome.papers
        warnings = [
            "local_hybrid_rrf_fusion",
            f"local_hybrid_bm25_candidates:{len(bm25_result.papers)}",
            f"local_hybrid_semantic_candidates:{len(semantic_result.papers)}",
        ]
        warnings.extend(rerank_outcome.warnings)
        warnings.extend(bm25_result.warnings)
        warnings.extend(semantic_result.warnings)
        latency = time.perf_counter() - started
        diagnostics = merge_connector_diagnostics(
            [
                bm25_result.diagnostics,
                semantic_result.diagnostics,
                rerank_outcome.diagnostics,
            ]
        ).model_copy(update={"latency_seconds": latency})
        return ConnectorSearchResult(
            papers=fused,
            warnings=_dedupe(warnings),
            latency_seconds=latency,
            diagnostics=diagnostics,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        latency = time.perf_counter() - started
        message = f"local_hybrid_failed:{type(exc).__name__}"
        return ConnectorSearchResult(
            error_message=message,
            warnings=[message],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(
                error_count=1,
                latency_seconds=latency,
            ),
        )


def _active_config_and_index() -> tuple[LocalHybridConfig, _SemanticIndex]:
    with _CONFIG_LOCK:
        if _ACTIVE_CONFIG is None or _ACTIVE_INDEX is None:
            raise ValueError("local_hybrid_not_configured")
        return _ACTIVE_CONFIG, _ACTIVE_INDEX


@dataclass(frozen=True)
class _RerankOutcome:
    papers: list[Paper]
    warnings: list[str]
    diagnostics: ConnectorDiagnostics


def _rerank_if_configured(
    query: str,
    papers: list[Paper],
    *,
    limit: int,
) -> _RerankOutcome:
    with _CONFIG_LOCK:
        reranker = _ACTIVE_RERANKER
        config = _ACTIVE_CONFIG
    if reranker is None or config is None or config.reranker_model_path is None:
        return _RerankOutcome(
            papers=papers,
            warnings=[],
            diagnostics=ConnectorDiagnostics(),
        )
    result: NeuralRerankResult = reranker.rerank(query, papers, limit=limit)
    warnings = [
        "local_hybrid_neural_reranker",
        f"local_hybrid_reranker_model:{result.model_fingerprint[:16]}",
        f"local_hybrid_reranker_latency_ms:{result.latency_seconds * 1000:.3f}",
        f"local_hybrid_reranker_prompt_version:{result.prompt_version}",
        f"local_hybrid_reranker_device:{result.device or 'unknown'}",
        f"local_hybrid_reranker_max_length:{result.max_length}",
    ]
    if result.fallback_used:
        warnings.append("local_hybrid_reranker_fallback_current_order")
        if result.error:
            warnings.append(f"local_hybrid_reranker_error:{result.error}")
    return _RerankOutcome(
        papers=result.papers,
        warnings=warnings,
        diagnostics=ConnectorDiagnostics(
            local_model_latency_seconds=result.latency_seconds,
            local_model_batch_count=result.batch_count,
            local_model_fallback_count=1 if result.fallback_used else 0,
            local_model_fingerprint=result.model_fingerprint,
            local_model_prompt_version=result.prompt_version,
            local_model_device=result.device,
            local_model_max_length=result.max_length,
            local_model_candidate_count=result.candidate_count,
            local_model_inference_success_count=(
                1 if result.inference_success else 0
            ),
            local_model_batch_size=result.batch_size or None,
            local_model_candidate_limit=result.candidate_limit or None,
            local_model_peak_vram_bytes=result.peak_vram_bytes,
        ),
    )


def _search_semantic(
    query: str,
    limit: int,
    *,
    config: LocalHybridConfig,
    index: _SemanticIndex,
) -> ConnectorSearchResult:
    started = time.perf_counter()
    model = _cached_model(config.model_path)
    encoded = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_vector = np.asarray(encoded[0], dtype=np.float32)
    count = min(max(0, int(limit)), index.document_count)
    if config.semantic_search_mode == "exact":
        matrix = np.load(index.embeddings_path, mmap_mode="r")
        scores = np.asarray(matrix @ query_vector).reshape(-1)
        candidate_indices = np.argpartition(-scores, count - 1)[:count]
        scored_indices = [
            (int(value), float(scores[value])) for value in candidate_indices
        ]
    else:
        if index.faiss_index is None:
            raise ValueError("local_hybrid_faiss_index_unavailable")
        distances, indices = index.faiss_index.search(
            query_vector.reshape(1, -1), count
        )
        scored_indices = [
            (int(value), float(distances[0][offset]))
            for offset, value in enumerate(indices[0])
            if int(value) >= 0
        ]
    rows = index.rows
    ranked_indices = [
        value
        for value, _score in sorted(
            scored_indices,
            key=lambda item: (
                -item[1],
                str(rows[item[0]].get("arxiv_id") or rows[item[0]].get("_id") or ""),
            ),
        )
    ]
    papers = [
        _paper_from_semantic_row(
            rows[value],
            sources=["local_hybrid", "local_semantic"],
        )
        for value in ranked_indices
    ]
    latency = time.perf_counter() - started
    return ConnectorSearchResult(
        papers=papers,
        warnings=[
            "local_hybrid_semantic_model",
            f"local_hybrid_semantic_mode:{config.semantic_search_mode}",
        ],
        latency_seconds=latency,
        diagnostics=ConnectorDiagnostics(latency_seconds=latency),
    )


def _fuse_ranked_lists(
    bm25_papers: list[Paper],
    semantic_papers: list[Paper],
    *,
    limit: int,
    rrf_k: int,
) -> list[Paper]:
    paper_by_key: dict[str, Paper] = {}
    score_by_key: dict[str, float] = {}
    for ranked in (bm25_papers, semantic_papers):
        for rank, paper in enumerate(ranked, start=1):
            key = _paper_key(paper)
            score_by_key[key] = score_by_key.get(key, 0.0) + 1.0 / (
                rrf_k + rank
            )
            tagged = paper.model_copy(
                update={
                    "sources": _merge_sources(
                        ["local_hybrid"],
                        list(paper.sources),
                    )
                }
            )
            if key not in paper_by_key:
                paper_by_key[key] = tagged
            else:
                paper_by_key[key] = deduplicate_papers(
                    [paper_by_key[key], tagged]
                )[0]
    ranked_keys = sorted(
        paper_by_key,
        key=lambda key: (-score_by_key[key], key),
    )
    return [paper_by_key[key] for key in ranked_keys[: max(0, limit)]]


def _paper_key(paper: Paper) -> str:
    arxiv_id = str(paper.identifiers.arxiv_id or "").strip().casefold()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return "title:" + " ".join(paper.title.casefold().split())


def _paper_from_semantic_row(
    row: dict[str, Any],
    *,
    sources: list[str],
) -> Paper:
    arxiv_id = _string_value(row.get("arxiv_id") or row.get("_id"))
    return Paper(
        title=_string_value(row.get("title")) or "",
        abstract=_string_value(row.get("abstract")) or "",
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=sources,
    )


def _normalize_config(config: LocalHybridConfig) -> LocalHybridConfig:
    semantic_corpus = Path(config.semantic_corpus_path).expanduser().resolve()
    index_dir = Path(config.semantic_index_dir).expanduser().resolve()
    model_path = Path(config.model_path).expanduser().resolve()
    if not semantic_corpus.is_file():
        raise ValueError("local_hybrid_semantic_corpus_not_found")
    if not model_path.is_dir():
        raise ValueError("local_hybrid_model_not_found")
    if config.bm25_candidate_limit < 20 or config.semantic_candidate_limit < 20:
        raise ValueError("local_hybrid_candidate_limit_too_small")
    if config.rrf_k <= 0:
        raise ValueError("local_hybrid_rrf_k_invalid")
    if config.semantic_search_mode not in {"ann", "exact"}:
        raise ValueError("local_hybrid_semantic_search_mode_invalid")
    if config.hnsw_m <= 0 or config.hnsw_ef_construction <= 0 or config.hnsw_ef_search <= 0:
        raise ValueError("local_hybrid_hnsw_parameter_invalid")
    if config.recall_sample_size < 0 or config.recall_k <= 0:
        raise ValueError("local_hybrid_recall_parameter_invalid")
    reranker_model = (
        Path(config.reranker_model_path).expanduser().resolve()
        if config.reranker_model_path is not None
        else None
    )
    if reranker_model is not None and not reranker_model.is_dir():
        raise ValueError("local_hybrid_reranker_model_not_found")
    if config.reranker_candidate_limit <= 0 or config.reranker_batch_size <= 0:
        raise ValueError("local_hybrid_reranker_limits_invalid")
    if config.reranker_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("local_hybrid_reranker_device_invalid")
    bm25 = config.bm25_config
    return LocalHybridConfig(
        bm25_config=bm25,
        semantic_corpus_path=semantic_corpus,
        semantic_index_dir=index_dir,
        model_path=model_path,
        bm25_candidate_limit=config.bm25_candidate_limit,
        semantic_candidate_limit=config.semantic_candidate_limit,
        rrf_k=config.rrf_k,
        semantic_search_mode=config.semantic_search_mode,
        hnsw_m=config.hnsw_m,
        hnsw_ef_construction=config.hnsw_ef_construction,
        hnsw_ef_search=config.hnsw_ef_search,
        recall_sample_size=config.recall_sample_size,
        recall_k=config.recall_k,
        reranker_model_path=reranker_model,
        reranker_candidate_limit=config.reranker_candidate_limit,
        reranker_batch_size=config.reranker_batch_size,
        reranker_device=config.reranker_device,
    )


def _load_index_metadata(config: LocalHybridConfig) -> LocalHybridIndexMetadata:
    path = config.semantic_index_dir / _METADATA_FILE
    if not path.is_file():
        raise ValueError("local_hybrid_index_metadata_not_found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LOCAL_HYBRID_INDEX_SCHEMA_VERSION:
        raise ValueError("local_hybrid_index_schema_mismatch")
    return _metadata_from_payload(payload, config.semantic_index_dir, cache_hit=True)


def _metadata_from_payload(
    payload: Mapping[str, Any],
    index_dir: Path,
    *,
    cache_hit: bool,
) -> LocalHybridIndexMetadata:
    try:
        return LocalHybridIndexMetadata(
            semantic_corpus_sha256=str(payload["semantic_corpus_sha256"]),
            semantic_corpus_size_bytes=int(payload["semantic_corpus_size_bytes"]),
            document_count=int(payload["document_count"]),
            abstract_document_count=int(payload["abstract_document_count"]),
            embedding_dimension=int(payload["embedding_dimension"]),
            model_path=str(payload["model_path"]),
            model_fingerprint=str(payload["model_fingerprint"]),
            index_dir=str(index_dir),
            index_fingerprint=str(payload["index_fingerprint"]),
            index_type=str(payload["index_type"]),
            hnsw_m=int(payload["hnsw_m"]),
            hnsw_ef_construction=int(payload["hnsw_ef_construction"]),
            hnsw_ef_search=int(payload["hnsw_ef_search"]),
            build_seconds=float(payload["build_seconds"]),
            peak_memory_bytes=int(payload["peak_memory_bytes"]),
            ann_recall_at_k=float(payload["ann_recall_at_k"]),
            recall_query_count=int(payload["recall_query_count"]),
            recall_k=int(payload["recall_k"]),
            ann_query_seconds=float(payload["ann_query_seconds"]),
            exact_query_seconds=float(payload["exact_query_seconds"]),
            cache_hit=cache_hit,
            index_load_seconds=0.0,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("local_hybrid_index_metadata_invalid") from exc


def _embedding_shape(path: Path) -> tuple[int, int]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError("local_hybrid_embedding_rank_invalid")
    return int(array.shape[0]), int(array.shape[1])


def _load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss_required_for_local_hybrid") from exc
    return faiss


def _read_faiss_index(
    path: Path,
    dimension: int,
    expected_count: int,
) -> Any:
    if not path.is_file():
        raise ValueError("local_hybrid_faiss_index_not_found")
    index = _load_faiss().read_index(str(path))
    if int(index.d) != dimension or int(index.ntotal) != expected_count:
        raise ValueError("local_hybrid_faiss_index_shape_invalid")
    return index


def _set_faiss_ef_search(index: Any, ef_search: int) -> None:
    if not hasattr(index, "hnsw"):
        raise ValueError("local_hybrid_faiss_index_type_invalid")
    index.hnsw.efSearch = ef_search


def _measure_index_recall(
    ann_index: Any,
    matrix: np.ndarray,
    sample_size: int,
    recall_k: int,
) -> tuple[float, int, float, float]:
    count = min(max(0, sample_size), int(matrix.shape[0]))
    k = min(max(1, recall_k), int(matrix.shape[0]))
    if count == 0:
        return 0.0, 0, 0.0, 0.0
    queries = np.asarray(matrix[:count], dtype=np.float32)
    ann_started = time.perf_counter()
    _distances, ann_indices = ann_index.search(queries, k)
    ann_seconds = time.perf_counter() - ann_started
    exact_started = time.perf_counter()
    scores = np.asarray(queries @ matrix.T)
    exact_indices = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    exact_seconds = time.perf_counter() - exact_started
    overlap = sum(
        len(set(map(int, ann_indices[row])) & set(map(int, exact_indices[row])))
        for row in range(count)
    )
    return overlap / (count * k), count, ann_seconds, exact_seconds


def _current_rss_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


def _read_semantic_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"local_hybrid_invalid_row:{line_number}")
            if not _string_value(payload.get("title")):
                raise ValueError(f"local_hybrid_missing_title:{line_number}")
            rows.append(payload)
    return rows


def _load_model_from_path(path: Path) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as exc:
        raise ImportError("sentence_transformers_required_for_local_hybrid") from exc
    requested_device = os.environ.get(
        "SCHOLARNAVIGATOR_LOCAL_HYBRID_DEVICE", "auto"
    ).strip().casefold()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("local_hybrid_model_device_invalid")
    return SentenceTransformer(
        str(path),
        device=requested_device,
        local_files_only=True,
    )


def _cached_model(path: Path) -> Any:
    global _ACTIVE_MODEL, _ACTIVE_MODEL_PATH
    with _MODEL_LOCK:
        if _ACTIVE_MODEL is None or _ACTIVE_MODEL_PATH != path:
            _ACTIVE_MODEL = _load_model_from_path(path)
            _ACTIVE_MODEL_PATH = path
        return _ACTIVE_MODEL


def _model_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    selected = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and item.name
        in {
            "config.json",
            "modules.json",
            "sentence_bert_config.json",
            "tokenizer.json",
            "vocab.txt",
            "model.safetensors",
        }
    ]
    if not selected:
        raise ValueError("local_hybrid_model_files_not_found")
    for item in selected:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _index_fingerprint(
    *,
    corpus_sha: str,
    model_fingerprint: str,
    shape: tuple[int, int],
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
) -> str:
    payload = {
        "schema_version": LOCAL_HYBRID_INDEX_SCHEMA_VERSION,
        "connector_version": LOCAL_HYBRID_CONNECTOR_VERSION,
        "corpus_sha256": corpus_sha,
        "model_fingerprint": model_fingerprint,
        "shape": list(shape),
        "dtype": "float32",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "index_type": "hnsw_ip",
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
        "hnsw_ef_search": hnsw_ef_search,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _temporary_path(path: Path) -> Path:
    descriptor = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(descriptor.name)
    descriptor.close()
    temporary.unlink(missing_ok=True)
    return temporary


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_build_progress(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_build_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress_matches(progress: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(progress.get(key) == value for key, value in expected.items())


def _discard_partial_build(
    partial_embeddings_path: Path,
    partial_faiss_path: Path,
    progress_path: Path,
) -> None:
    partial_embeddings_path.unlink(missing_ok=True)
    partial_faiss_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace("\r", " ").replace("\n", " ").strip()
    return normalized or None


def _merge_sources(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for source in group:
            value = str(source).strip()
            if value and value not in seen:
                merged.append(value)
                seen.add(value)
    return merged


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
