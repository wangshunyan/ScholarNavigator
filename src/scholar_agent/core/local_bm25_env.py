"""Environment-driven configuration for the local BM25 connector."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from scholar_agent.connectors.local_bm25 import (
    IdentityField,
    LocalBM25Config,
    LocalBM25FieldConfig,
    LocalBM25IndexMetadata,
    configure_local_bm25,
    local_bm25_metadata,
)


LOCAL_BM25_CORPUS_ENV = "SCHOLAR_AGENT_LOCAL_BM25_CORPUS"
LOCAL_BM25_CACHE_DIR_ENV = "SCHOLAR_AGENT_LOCAL_BM25_CACHE_DIR"
LOCAL_BM25_DOCUMENT_ID_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_ID_FIELD"
LOCAL_BM25_TITLE_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_TITLE_FIELD"
LOCAL_BM25_ABSTRACT_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_ABSTRACT_FIELD"
LOCAL_BM25_DOCUMENT_IDENTITY_ENV = "SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY"
LOCAL_BM25_DOI_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_DOI_FIELD"
LOCAL_BM25_ARXIV_ID_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_ARXIV_ID_FIELD"
LOCAL_BM25_SEMANTIC_SCHOLAR_ID_FIELD_ENV = (
    "SCHOLAR_AGENT_LOCAL_BM25_SEMANTIC_SCHOLAR_ID_FIELD"
)
LOCAL_BM25_S2ORC_CORPUS_ID_FIELD_ENV = (
    "SCHOLAR_AGENT_LOCAL_BM25_S2ORC_CORPUS_ID_FIELD"
)
LOCAL_BM25_OPENALEX_ID_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_OPENALEX_ID_FIELD"
LOCAL_BM25_PUBMED_ID_FIELD_ENV = "SCHOLAR_AGENT_LOCAL_BM25_PUBMED_ID_FIELD"

_DEFAULT_CACHE_DIR = Path("outputs") / "benchmark_cache" / "local_bm25"
_IDENTITY_FIELDS: tuple[str, ...] = (
    "doi",
    "arxiv_id",
    "semantic_scholar_id",
    "s2orc_corpus_id",
    "openalex_id",
    "pubmed_id",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK = RLock()
_ACTIVE_SIGNATURE: tuple[object, ...] | None = None
_ACTIVE_BUILT_INDEX = False


def local_bm25_config_from_env(
    repository_root: Path | None = None,
) -> LocalBM25Config | None:
    """Build the local BM25 config from env vars, or return None when disabled."""

    raw_corpus = _optional_env(LOCAL_BM25_CORPUS_ENV)
    if raw_corpus is None:
        return None
    root = (repository_root or _REPO_ROOT).resolve()
    identity = _identity_from_env()
    fields = LocalBM25FieldConfig(
        document_id=_env_or_default(LOCAL_BM25_DOCUMENT_ID_FIELD_ENV, "_id"),
        title=_env_or_default(LOCAL_BM25_TITLE_FIELD_ENV, "title"),
        abstract=_env_or_default(LOCAL_BM25_ABSTRACT_FIELD_ENV, "abstract"),
        document_id_identity=identity,
        doi=_optional_env(LOCAL_BM25_DOI_FIELD_ENV),
        arxiv_id=_optional_env(LOCAL_BM25_ARXIV_ID_FIELD_ENV),
        semantic_scholar_id=_optional_env(
            LOCAL_BM25_SEMANTIC_SCHOLAR_ID_FIELD_ENV
        ),
        s2orc_corpus_id=_optional_env(LOCAL_BM25_S2ORC_CORPUS_ID_FIELD_ENV),
        openalex_id=_optional_env(LOCAL_BM25_OPENALEX_ID_FIELD_ENV),
        pubmed_id=_optional_env(LOCAL_BM25_PUBMED_ID_FIELD_ENV),
    )
    return LocalBM25Config(
        corpus_path=_resolve_env_path(raw_corpus, root),
        cache_dir=_resolve_env_path(
            _env_or_default(LOCAL_BM25_CACHE_DIR_ENV, str(_DEFAULT_CACHE_DIR)),
            root,
        ),
        fields=fields,
    )


def configure_local_bm25_from_env(
    *,
    repository_root: Path | None = None,
    build_index: bool = False,
) -> LocalBM25IndexMetadata | None:
    """Configure the process-local BM25 connector from env with idempotent caching."""

    global _ACTIVE_BUILT_INDEX, _ACTIVE_SIGNATURE
    config = local_bm25_config_from_env(repository_root)
    with _LOCK:
        if config is None:
            if _ACTIVE_SIGNATURE is not None:
                configure_local_bm25(None)
                _ACTIVE_SIGNATURE = None
                _ACTIVE_BUILT_INDEX = False
            return None

        signature = _config_signature(config)
        if _ACTIVE_SIGNATURE == signature and (_ACTIVE_BUILT_INDEX or not build_index):
            return local_bm25_metadata()

        metadata = configure_local_bm25(config, build_index=build_index)
        _ACTIVE_SIGNATURE = signature
        _ACTIVE_BUILT_INDEX = build_index
        return metadata


def _identity_from_env() -> IdentityField:
    value = _env_or_default(
        LOCAL_BM25_DOCUMENT_IDENTITY_ENV,
        "s2orc_corpus_id",
    )
    if value not in _IDENTITY_FIELDS:
        raise ValueError(f"unsupported local_bm25 identity: {value}")
    return value  # type: ignore[return-value]


def _config_signature(config: LocalBM25Config) -> tuple[object, ...]:
    return (
        str(Path(config.corpus_path).expanduser()),
        str(Path(config.cache_dir).expanduser()),
        tuple(sorted(asdict(config.fields).items())),
        config.k1,
        config.b,
        config.epsilon,
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
