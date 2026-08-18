"""Environment-driven configuration for the local hybrid connector."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from scholar_agent.connectors.local_hybrid import (
    LocalHybridConfig,
    LocalHybridIndexMetadata,
    configure_local_hybrid,
    local_hybrid_metadata,
)
from scholar_agent.core.local_bm25_env import local_bm25_config_from_env


LOCAL_HYBRID_SEMANTIC_CORPUS_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CORPUS"
LOCAL_HYBRID_INDEX_DIR_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_INDEX_DIR"
LOCAL_HYBRID_MODEL_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_MODEL"
LOCAL_HYBRID_BM25_CANDIDATE_LIMIT_ENV = (
    "SCHOLAR_AGENT_LOCAL_HYBRID_BM25_CANDIDATE_LIMIT"
)
LOCAL_HYBRID_SEMANTIC_CANDIDATE_LIMIT_ENV = (
    "SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CANDIDATE_LIMIT"
)
LOCAL_HYBRID_RRF_K_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_RRF_K"
LOCAL_HYBRID_SEARCH_MODE_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_SEARCH_MODE"
LOCAL_HYBRID_HNSW_M_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_HNSW_M"
LOCAL_HYBRID_HNSW_EF_CONSTRUCTION_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_HNSW_EF_CONSTRUCTION"
LOCAL_HYBRID_HNSW_EF_SEARCH_ENV = "SCHOLAR_AGENT_LOCAL_HYBRID_HNSW_EF_SEARCH"

_DEFAULT_INDEX_DIR = Path("outputs") / "benchmark_cache" / "local_hybrid"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK = RLock()
_ACTIVE_SIGNATURE: tuple[object, ...] | None = None
_ACTIVE_BUILT_INDEX = False


def local_hybrid_config_from_env(
    repository_root: Path | None = None,
) -> LocalHybridConfig | None:
    """Build the local hybrid config from env vars, or return None."""

    raw_corpus = _optional_env(LOCAL_HYBRID_SEMANTIC_CORPUS_ENV)
    raw_model = _optional_env(LOCAL_HYBRID_MODEL_ENV)
    if raw_corpus is None or raw_model is None:
        return None
    root = (repository_root or _REPO_ROOT).resolve()
    bm25_config = local_bm25_config_from_env(root)
    if bm25_config is None:
        return None
    return LocalHybridConfig(
        bm25_config=bm25_config,
        semantic_corpus_path=_resolve_env_path(raw_corpus, root),
        semantic_index_dir=_resolve_env_path(
            _env_or_default(
                LOCAL_HYBRID_INDEX_DIR_ENV,
                str(_DEFAULT_INDEX_DIR),
            ),
            root,
        ),
        model_path=_resolve_env_path(raw_model, root),
        bm25_candidate_limit=_int_env(
            LOCAL_HYBRID_BM25_CANDIDATE_LIMIT_ENV,
            60,
        ),
        semantic_candidate_limit=_int_env(
            LOCAL_HYBRID_SEMANTIC_CANDIDATE_LIMIT_ENV,
            60,
        ),
        rrf_k=_int_env(LOCAL_HYBRID_RRF_K_ENV, 60),
        semantic_search_mode=_env_or_default(LOCAL_HYBRID_SEARCH_MODE_ENV, "ann"),
        hnsw_m=_int_env(LOCAL_HYBRID_HNSW_M_ENV, 32),
        hnsw_ef_construction=_int_env(LOCAL_HYBRID_HNSW_EF_CONSTRUCTION_ENV, 80),
        hnsw_ef_search=_int_env(LOCAL_HYBRID_HNSW_EF_SEARCH_ENV, 64),
    )


def configure_local_hybrid_from_env(
    *,
    repository_root: Path | None = None,
    build_index: bool = False,
) -> LocalHybridIndexMetadata | None:
    """Configure the process-local hybrid connector from env idempotently."""

    global _ACTIVE_BUILT_INDEX, _ACTIVE_SIGNATURE
    config = local_hybrid_config_from_env(repository_root)
    with _LOCK:
        if config is None:
            if _ACTIVE_SIGNATURE is not None:
                configure_local_hybrid(None)
                _ACTIVE_SIGNATURE = None
                _ACTIVE_BUILT_INDEX = False
            return None

        signature = _config_signature(config)
        if _ACTIVE_SIGNATURE == signature and (_ACTIVE_BUILT_INDEX or not build_index):
            return local_hybrid_metadata()

        if build_index:
            metadata = configure_local_hybrid(config, build_bm25_index=True)
        else:
            metadata = configure_local_hybrid(config, build_bm25_index=False)
        _ACTIVE_SIGNATURE = signature
        _ACTIVE_BUILT_INDEX = build_index
        return metadata


def _config_signature(config: LocalHybridConfig) -> tuple[object, ...]:
    return (
        tuple(sorted(asdict(config.bm25_config.fields).items())),
        str(Path(config.bm25_config.corpus_path).expanduser()),
        str(Path(config.bm25_config.cache_dir).expanduser()),
        str(Path(config.semantic_corpus_path).expanduser()),
        str(Path(config.semantic_index_dir).expanduser()),
        str(Path(config.model_path).expanduser()),
        config.bm25_candidate_limit,
        config.semantic_candidate_limit,
        config.rrf_k,
        config.semantic_search_mode,
        config.hnsw_m,
        config.hnsw_ef_construction,
        config.hnsw_ef_search,
    )


def _resolve_env_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _env_or_default(name: str, default: str) -> str:
    return _optional_env(name) or default


def _int_env(name: str, default: int) -> int:
    raw = _optional_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
