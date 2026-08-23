"""Deterministic disk-backed BM25 connector for local JSONL corpora."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Literal

from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.retrieval.query_adapter import ACADEMIC_QUERY_STOPWORDS


LOCAL_BM25_CONNECTOR_VERSION = "local-bm25-v4"
LOCAL_BM25_CACHE_SCHEMA_VERSION = "4"
LOCAL_BM25_TOKENIZER_VERSION = "unicode-word-casefold-v1"
LOCAL_BM25_QUERY_TOKENIZER_VERSION = "academic-stopword-filter-v1"
LOCAL_BM25_ENGINE = "sqlite-fts5-bm25-v1"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25
_SQLITE_TABLE = "papers"
_SQLITE_META_TABLE = "local_bm25_metadata"
_IDENTIFIER_COLUMNS = tuple(PaperIdentifiers.model_fields)
_COMPLETENESS_FIELDS = ("title", "abstract", "authors", "year", "venue", "doi")

IdentityField = Literal[
    "doi",
    "arxiv_id",
    "semantic_scholar_id",
    "s2orc_corpus_id",
    "openalex_id",
    "pubmed_id",
]


@dataclass(frozen=True)
class LocalBM25FieldConfig:
    """JSONL field paths and the stable identity carried by the document ID."""

    document_id: str = "_id"
    title: str = "title"
    abstract: str = "abstract"
    authors: str = "authors"
    year: str = "year"
    venue: str = "venue"
    document_id_identity: IdentityField = "s2orc_corpus_id"
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    s2orc_corpus_id: str | None = None
    openalex_id: str | None = None
    pubmed_id: str | None = None


@dataclass(frozen=True)
class LocalBM25Config:
    """Explicit local corpus configuration; no dataset or evaluator inputs exist."""

    corpus_path: Path
    cache_dir: Path
    fields: LocalBM25FieldConfig = LocalBM25FieldConfig()
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True)
class LocalBM25IndexMetadata:
    corpus_sha256: str
    corpus_size_bytes: int
    document_count: int
    fingerprint: str
    cache_path: str
    cache_hit: bool
    index_load_seconds: float
    field_completeness: dict[str, float] | None = None


@dataclass(frozen=True)
class _LocalBM25Index:
    fingerprint: str
    database_path: Path
    document_count: int
    field_completeness: dict[str, float]


_CONFIG_LOCK = RLock()
_ACTIVE_CONFIG: LocalBM25Config | None = None
_ACTIVE_METADATA: LocalBM25IndexMetadata | None = None
_ACTIVE_INDEX: _LocalBM25Index | None = None


def tokenize_local_bm25(value: str | None) -> list[str]:
    """Match the frozen SciFact offline audit tokenizer exactly."""

    return TOKEN_PATTERN.findall(str(value or "").casefold())


def tokenize_local_bm25_query(value: str | None) -> list[str]:
    """Remove natural-language filler before querying a title-heavy corpus.

    The PaSa paper database available locally only contains titles. Searching
    every word in a question would otherwise let terms such as ``tell`` and
    ``papers`` dominate FTS matches. The corpus tokenizer stays frozen for
    audit compatibility; this is intentionally a query-only normalization.
    """

    tokens = tokenize_local_bm25(value)
    focused = [token for token in tokens if token not in ACADEMIC_QUERY_STOPWORDS]
    return focused or tokens


def configure_local_bm25(
    config: LocalBM25Config | None,
    *,
    build_index: bool = True,
) -> LocalBM25IndexMetadata | None:
    """Set or clear the process-local connector configuration.

    The FTS5 index remains on disk so PaSa-sized corpora do not require a
    second full in-memory index when the API and benchmark run concurrently.
    """

    global _ACTIVE_CONFIG, _ACTIVE_INDEX, _ACTIVE_METADATA
    with _CONFIG_LOCK:
        _ACTIVE_CONFIG = None
        _ACTIVE_INDEX = None
        _ACTIVE_METADATA = None
        if config is None:
            return None
        normalized = _normalize_config(config)
        fingerprint, corpus_sha, corpus_size = _fingerprint(normalized)
        cache_path = _cache_path(normalized, fingerprint)
        _ACTIVE_CONFIG = normalized
        _ACTIVE_METADATA = LocalBM25IndexMetadata(
            corpus_sha256=corpus_sha,
            corpus_size_bytes=corpus_size,
            document_count=_nonempty_line_count(normalized.corpus_path),
            fingerprint=fingerprint,
            cache_path=str(cache_path),
            cache_hit=False,
            index_load_seconds=0.0,
            field_completeness=None,
        )
        if build_index:
            _ACTIVE_INDEX, _ACTIVE_METADATA = _load_or_build_index(
                normalized,
                fingerprint=fingerprint,
                corpus_sha=corpus_sha,
                corpus_size=corpus_size,
            )
        elif cache_path.is_file():
            try:
                cached = _open_index(cache_path, fingerprint)
                _ACTIVE_METADATA = LocalBM25IndexMetadata(
                    corpus_sha256=corpus_sha,
                    corpus_size_bytes=corpus_size,
                    document_count=cached.document_count,
                    fingerprint=fingerprint,
                    cache_path=str(cache_path),
                    cache_hit=True,
                    index_load_seconds=0.0,
                    field_completeness=dict(cached.field_completeness),
                )
            except (OSError, sqlite3.Error, ValueError):
                pass
        return _ACTIVE_METADATA


def local_bm25_metadata() -> LocalBM25IndexMetadata:
    with _CONFIG_LOCK:
        if _ACTIVE_METADATA is None:
            raise ValueError("local_bm25_not_configured")
        return _ACTIVE_METADATA


def local_bm25_connector_version() -> str:
    metadata = local_bm25_metadata()
    return f"{LOCAL_BM25_CONNECTOR_VERSION}:{metadata.fingerprint}"


def search_local_bm25(query: str, limit: int = 20) -> list[Paper]:
    return search_local_bm25_detailed(query, limit).papers


def search_local_bm25_detailed(
    query: str,
    limit: int = 20,
) -> ConnectorSearchResult:
    """Search the configured disk-backed FTS5 BM25 index."""

    started = time.perf_counter()
    normalized_query = str(query).strip()
    if not normalized_query or limit <= 0:
        latency = time.perf_counter() - started
        return ConnectorSearchResult(
            warnings=["local_bm25_empty_query"] if not normalized_query else [],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(latency_seconds=latency),
        )
    try:
        index, metadata = _active_index()
        papers = _search_index(
            index,
            tokenize_local_bm25_query(normalized_query),
            limit,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        latency = time.perf_counter() - started
        return ConnectorSearchResult(
            error_message=f"local_bm25_failed:{type(exc).__name__}",
            warnings=[f"local_bm25_failed:{type(exc).__name__}"],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(
                error_count=1,
                latency_seconds=latency,
            ),
        )

    latency = time.perf_counter() - started
    return ConnectorSearchResult(
        papers=papers,
        warnings=[
            "local_bm25_index_cache_hit"
            if metadata.cache_hit
            else "local_bm25_index_built"
        ],
        latency_seconds=latency,
        diagnostics=ConnectorDiagnostics(
            cache_hit_count=int(metadata.cache_hit),
            latency_seconds=latency,
        ),
    )


def _active_index() -> tuple[_LocalBM25Index, LocalBM25IndexMetadata]:
    global _ACTIVE_INDEX, _ACTIVE_METADATA
    with _CONFIG_LOCK:
        if _ACTIVE_CONFIG is None or _ACTIVE_METADATA is None:
            raise ValueError("local_bm25_not_configured")
        if _ACTIVE_INDEX is None:
            _ACTIVE_INDEX, _ACTIVE_METADATA = _load_or_build_index(
                _ACTIVE_CONFIG,
                fingerprint=_ACTIVE_METADATA.fingerprint,
                corpus_sha=_ACTIVE_METADATA.corpus_sha256,
                corpus_size=_ACTIVE_METADATA.corpus_size_bytes,
            )
        return _ACTIVE_INDEX, _ACTIVE_METADATA


def _normalize_config(config: LocalBM25Config) -> LocalBM25Config:
    corpus_path = Path(config.corpus_path).expanduser().resolve()
    cache_dir = Path(config.cache_dir).expanduser().resolve()
    if not corpus_path.is_file():
        raise ValueError("local_bm25_corpus_not_found")
    if corpus_path.suffix.casefold() not in {".jsonl", ".json"}:
        raise ValueError("local_bm25_corpus_must_be_jsonl")
    for name, value in asdict(config.fields).items():
        if value is not None and not str(value).strip():
            raise ValueError(f"local_bm25_empty_field:{name}")
    if (config.k1, config.b, config.epsilon) != (
        DEFAULT_K1,
        DEFAULT_B,
        DEFAULT_EPSILON,
    ):
        raise ValueError("local_bm25_parameters_are_frozen")
    return LocalBM25Config(
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        fields=config.fields,
        k1=config.k1,
        b=config.b,
        epsilon=config.epsilon,
    )


def _fingerprint(config: LocalBM25Config) -> tuple[str, str, int]:
    corpus_sha, corpus_size = _sha256_file(config.corpus_path)
    descriptor = {
        "cache_schema_version": LOCAL_BM25_CACHE_SCHEMA_VERSION,
        "connector_version": LOCAL_BM25_CONNECTOR_VERSION,
        "engine": LOCAL_BM25_ENGINE,
        "corpus_sha256": corpus_sha,
        "fields": asdict(config.fields),
        "parameters": {
            "b": config.b,
            "epsilon": config.epsilon,
            "k1": config.k1,
        },
        "tokenizer": LOCAL_BM25_TOKENIZER_VERSION,
        "query_tokenizer": LOCAL_BM25_QUERY_TOKENIZER_VERSION,
    }
    encoded = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), corpus_sha, corpus_size


def _cache_path(config: LocalBM25Config, fingerprint: str) -> Path:
    return config.cache_dir / f"{fingerprint}.sqlite3"


def _load_or_build_index(
    config: LocalBM25Config,
    *,
    fingerprint: str,
    corpus_sha: str,
    corpus_size: int,
) -> tuple[_LocalBM25Index, LocalBM25IndexMetadata]:
    started = time.perf_counter()
    cache_path = _cache_path(config, fingerprint)
    index: _LocalBM25Index | None = None
    cache_hit = False
    if cache_path.is_file():
        try:
            index = _open_index(cache_path, fingerprint)
            cache_hit = True
        except (OSError, sqlite3.Error, ValueError):
            cache_path.unlink(missing_ok=True)
    if index is None:
        index = _build_index(config, fingerprint, cache_path)
    metadata = LocalBM25IndexMetadata(
        corpus_sha256=corpus_sha,
        corpus_size_bytes=corpus_size,
        document_count=index.document_count,
        fingerprint=fingerprint,
        cache_path=str(cache_path),
        cache_hit=cache_hit,
        index_load_seconds=time.perf_counter() - started,
        field_completeness=dict(index.field_completeness),
    )
    return index, metadata


def _open_index(cache_path: Path, fingerprint: str) -> _LocalBM25Index:
    with _connect(cache_path, read_only=True) as connection:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                f"SELECT key, value FROM {_SQLITE_META_TABLE}"
            )
        }
        if (
            metadata.get("schema_version") != LOCAL_BM25_CACHE_SCHEMA_VERSION
            or metadata.get("fingerprint") != fingerprint
            or metadata.get("engine") != LOCAL_BM25_ENGINE
        ):
            raise ValueError("local_bm25_cache_incompatible")
        document_count = int(metadata["document_count"])
        try:
            field_completeness = json.loads(metadata["field_completeness"])
        except (KeyError, json.JSONDecodeError, TypeError):
            raise ValueError("local_bm25_cache_missing_field_completeness")
        if (
            not isinstance(field_completeness, dict)
            or set(field_completeness) != set(_COMPLETENESS_FIELDS)
            or any(
                not isinstance(value, (int, float)) or not 0 <= float(value) <= 1
                for value in field_completeness.values()
            )
        ):
            raise ValueError("local_bm25_cache_invalid_field_completeness")
        actual_count = int(
            connection.execute(f"SELECT count(*) FROM {_SQLITE_TABLE}").fetchone()[0]
        )
        if document_count <= 0 or document_count != actual_count:
            raise ValueError("local_bm25_cache_invalid_document_count")
    return _LocalBM25Index(
        fingerprint=fingerprint,
        database_path=cache_path,
        document_count=document_count,
        field_completeness={
            key: float(value) for key, value in field_completeness.items()
        },
    )


def _build_index(
    config: LocalBM25Config,
    fingerprint: str,
    cache_path: Path,
) -> _LocalBM25Index:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        prefix=f".{fingerprint}.",
        suffix=".sqlite3.tmp",
        dir=cache_path.parent,
        delete=False,
    )
    temporary = Path(descriptor.name)
    descriptor.close()
    temporary.unlink(missing_ok=True)
    try:
        with _connect(temporary, read_only=False) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            _create_schema(connection)
            seen: set[str] = set()
            completeness_counts = {key: 0 for key in _COMPLETENESS_FIELDS}
            with connection:
                with config.corpus_path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        payload = _parse_jsonl_row(line, line_number)
                        document_id = _required_value(
                            payload,
                            config.fields.document_id,
                            line_number,
                        )
                        title = _string_value(payload, config.fields.title)
                        abstract = _string_value(payload, config.fields.abstract) or ""
                        if not title:
                            raise ValueError(f"local_bm25_missing_title:{line_number}")
                        identifiers = _identifiers(
                            payload,
                            document_id,
                            config.fields,
                        )
                        authors_json, year, venue = _metadata_values(payload, config.fields)
                        values = (
                            document_id,
                            title,
                            abstract,
                            *(
                                getattr(identifiers, name)
                                for name in _IDENTIFIER_COLUMNS
                            ),
                            authors_json,
                            year,
                            venue,
                        )
                        if document_id in seen:
                            _validate_duplicate_row(connection, values)
                            continue
                        seen.add(document_id)
                        completeness_counts["title"] += int(bool(title))
                        completeness_counts["abstract"] += int(bool(abstract))
                        completeness_counts["authors"] += int(
                            json.loads(authors_json) != []
                        )
                        completeness_counts["year"] += int(year is not None)
                        completeness_counts["venue"] += int(venue is not None)
                        completeness_counts["doi"] += int(identifiers.doi is not None)
                        placeholders = ",".join("?" for _ in values)
                        columns = (
                            "document_id,title,abstract,"
                            + ",".join(_IDENTIFIER_COLUMNS)
                            + ",authors_json,year,venue"
                        )
                        connection.execute(
                            f"INSERT INTO {_SQLITE_TABLE} ({columns}) "
                            f"VALUES ({placeholders})",
                            values,
                        )
            if not seen:
                raise ValueError("local_bm25_empty_corpus")
            with connection:
                connection.execute(
                    f"INSERT INTO {_SQLITE_META_TABLE} (key, value) VALUES (?, ?)",
                    ("schema_version", LOCAL_BM25_CACHE_SCHEMA_VERSION),
                )
                connection.execute(
                    f"INSERT INTO {_SQLITE_META_TABLE} (key, value) VALUES (?, ?)",
                    ("fingerprint", fingerprint),
                )
                connection.execute(
                    f"INSERT INTO {_SQLITE_META_TABLE} (key, value) VALUES (?, ?)",
                    ("engine", LOCAL_BM25_ENGINE),
                )
                connection.execute(
                    f"INSERT INTO {_SQLITE_META_TABLE} (key, value) VALUES (?, ?)",
                    ("document_count", str(len(seen))),
                )
                field_completeness = {
                    key: completeness_counts[key] / len(seen)
                    for key in _COMPLETENESS_FIELDS
                }
                connection.execute(
                    f"INSERT INTO {_SQLITE_META_TABLE} (key, value) VALUES (?, ?)",
                    (
                        "field_completeness",
                        json.dumps(
                            field_completeness,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
                connection.execute(f"INSERT INTO {_SQLITE_TABLE}({_SQLITE_TABLE}) VALUES('optimize')")
            connection.execute("PRAGMA optimize")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return _open_index(cache_path, fingerprint)


def _create_schema(connection: sqlite3.Connection) -> None:
    identifier_columns = ",".join(f"{name} UNINDEXED" for name in _IDENTIFIER_COLUMNS)
    connection.execute(
        f"CREATE VIRTUAL TABLE {_SQLITE_TABLE} USING fts5("
        f"document_id UNINDEXED,title,abstract,{identifier_columns},"
        "authors_json UNINDEXED,year UNINDEXED,venue UNINDEXED,"
        "tokenize='unicode61')"
    )
    connection.execute(
        f"CREATE TABLE {_SQLITE_META_TABLE} (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _validate_duplicate_row(
    connection: sqlite3.Connection,
    values: tuple[str | None, ...],
) -> None:
    document_id = str(values[0])
    columns = (
        "document_id,title,abstract,"
        + ",".join(_IDENTIFIER_COLUMNS)
        + ",authors_json,year,venue"
    )
    prior = connection.execute(
        f"SELECT {columns} FROM {_SQLITE_TABLE} WHERE document_id = ? LIMIT 1",
        (document_id,),
    ).fetchone()
    if prior is None or tuple(prior) != values:
        raise ValueError(f"local_bm25_conflicting_document:{document_id}")


def _search_index(
    index: _LocalBM25Index,
    tokens: list[str],
    limit: int,
) -> list[Paper]:
    if not tokens:
        return []
    match_expression = " OR ".join(
        '"' + token.replace('"', '""') + '"' for token in tokens
    )
    columns = (
        "rowid,document_id,title,abstract,"
        + ",".join(_IDENTIFIER_COLUMNS)
        + ",authors_json,year,venue"
    )
    with _connect(index.database_path, read_only=True) as connection:
        matched = connection.execute(
            f"SELECT {columns} FROM {_SQLITE_TABLE} "
            f"WHERE {_SQLITE_TABLE} MATCH ? "
            f"ORDER BY bm25({_SQLITE_TABLE}), document_id LIMIT ?",
            (match_expression, min(int(limit), index.document_count)),
        ).fetchall()
        rows = list(matched)
        remaining = min(int(limit), index.document_count) - len(rows)
        if remaining > 0:
            excluded = [int(row[0]) for row in rows]
            if excluded:
                placeholders = ",".join("?" for _ in excluded)
                fallback = connection.execute(
                    f"SELECT {columns} FROM {_SQLITE_TABLE} "
                    f"WHERE rowid NOT IN ({placeholders}) "
                    "ORDER BY document_id LIMIT ?",
                    (*excluded, remaining),
                ).fetchall()
            else:
                fallback = connection.execute(
                    f"SELECT {columns} FROM {_SQLITE_TABLE} "
                    "ORDER BY document_id LIMIT ?",
                    (remaining,),
                ).fetchall()
            rows.extend(fallback)
    return [_paper_from_row(row) for row in rows]


def _paper_from_row(row: sqlite3.Row | tuple[Any, ...]) -> Paper:
    values = tuple(row)
    identifiers = PaperIdentifiers(
        **{
            name: str(value) if value is not None else None
            for name, value in zip(
                _IDENTIFIER_COLUMNS,
                values[4 : 4 + len(_IDENTIFIER_COLUMNS)],
                strict=True,
            )
        }
    )
    authors_json = str(values[4 + len(_IDENTIFIER_COLUMNS)] or "[]")
    try:
        authors_value = json.loads(authors_json)
    except json.JSONDecodeError:
        authors_value = []
    authors = (
        [str(item) for item in authors_value if str(item).strip()]
        if isinstance(authors_value, list)
        else []
    )
    raw_year = values[5 + len(_IDENTIFIER_COLUMNS)]
    try:
        year = int(raw_year) if raw_year not in (None, "") else None
    except (TypeError, ValueError):
        year = None
    venue = str(values[6 + len(_IDENTIFIER_COLUMNS)] or "") or None
    return Paper(
        title=str(values[2]),
        authors=authors,
        year=year,
        venue=venue,
        abstract=str(values[3] or ""),
        identifiers=identifiers,
        sources=["local_bm25"],
    )


@contextmanager
def _connect(path: Path, *, read_only: bool) -> Iterator[sqlite3.Connection]:
    if read_only:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def _parse_jsonl_row(line: str, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"local_bm25_invalid_jsonl:{line_number}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"local_bm25_invalid_row:{line_number}")
    return payload


def _identifiers(
    payload: dict[str, Any],
    document_id: str,
    fields: LocalBM25FieldConfig,
) -> PaperIdentifiers:
    values: dict[str, str | None] = {}
    for name in PaperIdentifiers.model_fields:
        field_path = getattr(fields, name)
        values[name] = _string_value(payload, field_path) if field_path else None
    existing = values[fields.document_id_identity]
    if existing is not None and existing != document_id:
        raise ValueError(
            f"local_bm25_document_identity_conflict:{fields.document_id_identity}"
        )
    values[fields.document_id_identity] = document_id
    return PaperIdentifiers(**values)


def _metadata_values(
    payload: dict[str, Any], fields: LocalBM25FieldConfig
) -> tuple[str, str | None, str | None]:
    raw_authors = _value_at_path(payload, fields.authors)
    if isinstance(raw_authors, list):
        authors = []
        for item in raw_authors:
            if isinstance(item, dict):
                item = item.get("name") or item.get("full_name") or item.get("author")
            normalized = str(item or "").strip()
            if normalized:
                authors.append(normalized)
    elif raw_authors is None:
        authors = []
    else:
        authors = [
            part.strip() for part in str(raw_authors).split(",") if part.strip()
        ]
    raw_year = _value_at_path(payload, fields.year)
    year = str(raw_year).strip() if raw_year not in (None, "") else None
    if year is not None:
        try:
            year = str(int(year))
        except ValueError:
            year = None
    raw_venue = _value_at_path(payload, fields.venue)
    venue = str(raw_venue).strip() if raw_venue not in (None, "") else None
    return json.dumps(authors, ensure_ascii=False, separators=(",", ":")), year, venue


def _value_at_path(payload: dict[str, Any], field_path: str | None) -> Any:
    if not field_path:
        return None
    value: Any = payload
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _required_value(
    payload: dict[str, Any],
    field_path: str,
    line_number: int,
) -> str:
    value = _string_value(payload, field_path)
    if value is None:
        raise ValueError(f"local_bm25_missing_document_id:{line_number}")
    return value


def _string_value(payload: dict[str, Any], field_path: str | None) -> str | None:
    if not field_path:
        return None
    value: Any = payload
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)
