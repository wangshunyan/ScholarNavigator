"""Evaluator-only Semantic Scholar crosswalk for BEIR SciFact gold papers.

The crosswalk is deliberately isolated from retrieval.  It accepts only an
exact S2ORC Corpus ID and retains only stable identifiers returned by the
official Semantic Scholar Graph API.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError

from scholar_agent.connectors.semantic_scholar import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    _retry_after_seconds,
    _semantic_scholar_headers,
    _throttle_semantic_scholar_request,
)
from scholar_agent.core.identity import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_s2orc_corpus_id,
    normalize_simple_id,
)


CROSSWALK_SCHEMA_VERSION = "1"
CROSSWALK_CONNECTOR_VERSION = "semantic-scholar-corpusid-v1"
CROSSWALK_SOURCE = "Semantic Scholar Graph API /paper/CorpusId:{id}"
CROSSWALK_FIELDS = ("paperId", "corpusId", "externalIds")
SEMANTIC_SCHOLAR_PAPER_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/CorpusId:{corpus_id}"
)
MAX_RETRY_WAIT_SECONDS = 5.0

CrosswalkStatus = Literal["success", "unavailable", "failed"]


class SciFactCrosswalkEntry(BaseModel):
    """One deterministic evaluator identity result."""

    s2orc_corpus_id: str
    status: CrosswalkStatus
    semantic_scholar_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pubmed_id: str | None = None
    external_id_fields: list[str] = Field(default_factory=list)
    error_type: str | None = None
    http_status: int | None = None


class SciFactCrosswalkArtifact(BaseModel):
    """Versioned, deterministic crosswalk consumed by the dataset adapter."""

    schema_version: str = CROSSWALK_SCHEMA_VERSION
    dataset: str = "beir_scifact"
    source: str = CROSSWALK_SOURCE
    connector_version: str = CROSSWALK_CONNECTOR_VERSION
    requested_fields: list[str] = Field(
        default_factory=lambda: list(CROSSWALK_FIELDS)
    )
    entries: list[SciFactCrosswalkEntry]


class SciFactCrosswalkSnapshot(BaseModel):
    """Auditable terminal response for one exact Corpus ID lookup."""

    schema_version: str = CROSSWALK_SCHEMA_VERSION
    connector_version: str = CROSSWALK_CONNECTOR_VERSION
    key: str = Field(min_length=64, max_length=64)
    s2orc_corpus_id: str
    requested_fields: list[str] = Field(
        default_factory=lambda: list(CROSSWALK_FIELDS)
    )
    status: CrosswalkStatus
    semantic_scholar_id: str | None = None
    returned_corpus_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    error_type: str | None = None
    http_status: int | None = None
    request_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)


def crosswalk_snapshot_key(corpus_id: Any) -> str:
    normalized = _required_corpus_id(corpus_id)
    return _stable_hash(
        {
            "schema_version": CROSSWALK_SCHEMA_VERSION,
            "connector_version": CROSSWALK_CONNECTOR_VERSION,
            "s2orc_corpus_id": normalized,
            "requested_fields": list(CROSSWALK_FIELDS),
        }
    )


def crosswalk_content_hash(snapshot: SciFactCrosswalkSnapshot | dict[str, Any]) -> str:
    payload = (
        snapshot.model_dump(mode="json")
        if isinstance(snapshot, SciFactCrosswalkSnapshot)
        else dict(snapshot)
    )
    payload.pop("content_hash", None)
    payload.pop("recorded_at", None)
    return _stable_hash(payload)


class SciFactCrosswalkStore:
    """Strict snapshot store for bounded, serial exact lookups."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries_dir = self.root / "entries"

    def path_for(self, corpus_id: Any) -> Path:
        return self.entries_dir / f"{crosswalk_snapshot_key(corpus_id)}.json"

    def read(self, corpus_id: Any) -> SciFactCrosswalkSnapshot:
        expected_id = _required_corpus_id(corpus_id)
        path = self.path_for(expected_id)
        try:
            snapshot = SciFactCrosswalkSnapshot.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ValueError(f"invalid SciFact crosswalk snapshot: {path.name}") from exc
        expected_key = crosswalk_snapshot_key(expected_id)
        if (
            snapshot.schema_version != CROSSWALK_SCHEMA_VERSION
            or snapshot.connector_version != CROSSWALK_CONNECTOR_VERSION
            or snapshot.requested_fields != list(CROSSWALK_FIELDS)
            or snapshot.key != expected_key
            or snapshot.s2orc_corpus_id != expected_id
        ):
            raise ValueError(f"SciFact crosswalk snapshot request mismatch: {path.name}")
        if snapshot.content_hash != crosswalk_content_hash(snapshot):
            raise ValueError(f"SciFact crosswalk snapshot hash mismatch: {path.name}")
        if snapshot.status == "success" and snapshot.returned_corpus_id != expected_id:
            raise ValueError(f"SciFact crosswalk snapshot response mismatch: {path.name}")
        return snapshot

    def write(self, snapshot: SciFactCrosswalkSnapshot) -> None:
        path = self.path_for(snapshot.s2orc_corpus_id)
        if path.exists():
            self.read(snapshot.s2orc_corpus_id)
            return
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, snapshot.model_dump(mode="json"))


def fetch_exact_corpus_id(
    corpus_id: Any,
    *,
    opener: Callable[..., Any] = urlopen,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    throttle: Callable[..., float] = _throttle_semantic_scholar_request,
) -> SciFactCrosswalkSnapshot:
    """Fetch one official exact CorpusId record without logging request details."""

    normalized = _required_corpus_id(corpus_id)
    key = crosswalk_snapshot_key(normalized)
    started = time.perf_counter()
    request_count = 0
    retry_count = 0
    terminal: dict[str, Any] | None = None
    attempts = max(0, int(max_retries)) + 1
    for attempt in range(attempts):
        throttle(sleep=sleep)
        request_count += 1
        retry_count += int(attempt > 0)
        request = Request(
            _request_url(normalized),
            headers=_semantic_scholar_headers(),
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", getattr(response, "code", 200)))
                if status < 200 or status >= 300:
                    terminal = _http_terminal(status)
                    if _retryable_status(status) and attempt < attempts - 1:
                        sleep(_bounded_retry_wait(response, attempt))
                        continue
                    break
                try:
                    payload = json.loads(response.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    terminal = {
                        "status": "failed",
                        "error_type": "invalid_json",
                        "http_status": status,
                    }
                    break
                terminal = _parse_success_payload(normalized, payload, status)
                break
        except HTTPError as exc:
            terminal = _http_terminal(int(exc.code))
            if _retryable_status(exc.code) and attempt < attempts - 1:
                sleep(_bounded_retry_wait(exc, attempt))
                continue
            break
        except (TimeoutError, socket.timeout):
            terminal = {
                "status": "failed",
                "error_type": "network_timeout",
                "http_status": None,
            }
            if attempt < attempts - 1:
                sleep(min(MAX_RETRY_WAIT_SECONDS, 0.5 * (attempt + 1)))
                continue
            break
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            error_type = (
                "dns_error"
                if isinstance(reason, socket.gaierror)
                else "network_error"
            )
            terminal = {"status": "failed", "error_type": error_type, "http_status": None}
            if attempt < attempts - 1:
                sleep(min(MAX_RETRY_WAIT_SECONDS, 0.5 * (attempt + 1)))
                continue
            break
        except OSError:
            terminal = {"status": "failed", "error_type": "network_error", "http_status": None}
            if attempt < attempts - 1:
                sleep(min(MAX_RETRY_WAIT_SECONDS, 0.5 * (attempt + 1)))
                continue
            break

    if terminal is None:
        terminal = {
            "status": "failed",
            "error_type": "attempts_exhausted",
            "http_status": None,
        }
    snapshot = SciFactCrosswalkSnapshot(
        key=key,
        s2orc_corpus_id=normalized,
        request_count=request_count,
        retry_count=retry_count,
        latency_seconds=time.perf_counter() - started,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        content_hash="0" * 64,
        **terminal,
    )
    return snapshot.model_copy(update={"content_hash": crosswalk_content_hash(snapshot)})


def record_missing_crosswalk(
    corpus_ids: Iterable[Any],
    store: SciFactCrosswalkStore,
    *,
    fetcher: Callable[[Any], SciFactCrosswalkSnapshot] = fetch_exact_corpus_id,
) -> dict[str, int]:
    """Record each unique request serially; one failure never aborts later IDs."""

    counts = {"planned": 0, "existing": 0, "written": 0}
    for corpus_id in _unique_corpus_ids(corpus_ids):
        counts["planned"] += 1
        path = store.path_for(corpus_id)
        if path.exists():
            store.read(corpus_id)
            counts["existing"] += 1
            continue
        store.write(fetcher(corpus_id))
        counts["written"] += 1
    return counts


def replay_crosswalk(
    corpus_ids: Iterable[Any],
    store: SciFactCrosswalkStore,
) -> SciFactCrosswalkArtifact:
    """Strictly replay every expected key without HTTP or writes."""

    entries = [
        _artifact_entry(store.read(corpus_id))
        for corpus_id in _unique_corpus_ids(corpus_ids)
    ]
    return SciFactCrosswalkArtifact(entries=entries)


def load_crosswalk(path: str | Path) -> SciFactCrosswalkArtifact:
    try:
        artifact = SciFactCrosswalkArtifact.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid SciFact crosswalk artifact: {Path(path).name}") from exc
    if (
        artifact.schema_version != CROSSWALK_SCHEMA_VERSION
        or artifact.dataset != "beir_scifact"
        or artifact.source != CROSSWALK_SOURCE
        or artifact.connector_version != CROSSWALK_CONNECTOR_VERSION
        or artifact.requested_fields != list(CROSSWALK_FIELDS)
    ):
        raise ValueError(f"incompatible SciFact crosswalk artifact: {Path(path).name}")
    seen: dict[str, SciFactCrosswalkEntry] = {}
    for entry in artifact.entries:
        corpus_id = _required_corpus_id(entry.s2orc_corpus_id)
        prior = seen.get(corpus_id)
        if prior is not None and prior != entry:
            raise ValueError(f"conflicting duplicate SciFact crosswalk ID: {corpus_id}")
        seen[corpus_id] = entry
    return artifact


def write_crosswalk(path: str | Path, artifact: SciFactCrosswalkArtifact) -> None:
    entries = sorted(
        artifact.entries,
        key=lambda item: _corpus_sort_key(item.s2orc_corpus_id),
    )
    normalized = artifact.model_copy(update={"entries": entries})
    _atomic_write_json(Path(path), normalized.model_dump(mode="json"))


def crosswalk_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_success_payload(
    requested_corpus_id: str,
    payload: Any,
    http_status: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "error_type": "invalid_schema",
            "http_status": http_status,
        }
    external = payload.get("externalIds")
    external = external if isinstance(external, dict) else {}
    returned = normalize_s2orc_corpus_id(
        payload.get("corpusId") or external.get("CorpusId")
    )
    if returned is None or returned != requested_corpus_id:
        return {
            "status": "failed",
            "error_type": "corpus_id_mismatch",
            "http_status": http_status,
            "returned_corpus_id": returned,
        }
    stable_external_ids = _normalized_external_ids(external)
    return {
        "status": "success",
        "http_status": http_status,
        "semantic_scholar_id": normalize_simple_id(payload.get("paperId")),
        "returned_corpus_id": returned,
        "external_ids": stable_external_ids,
    }


def _normalized_external_ids(external_ids: dict[str, Any]) -> dict[str, str]:
    normalizers: tuple[tuple[str, Callable[[Any], str | None]], ...] = (
        ("DOI", normalize_doi),
        ("ArXiv", normalize_arxiv_id),
        ("PubMed", normalize_simple_id),
    )
    normalized: dict[str, str] = {}
    for field, normalizer in normalizers:
        value = normalizer(external_ids.get(field))
        if value:
            normalized[field] = value
    return normalized


def _artifact_entry(snapshot: SciFactCrosswalkSnapshot) -> SciFactCrosswalkEntry:
    external = snapshot.external_ids
    return SciFactCrosswalkEntry(
        s2orc_corpus_id=snapshot.s2orc_corpus_id,
        status=snapshot.status,
        semantic_scholar_id=snapshot.semantic_scholar_id,
        doi=external.get("DOI"),
        arxiv_id=external.get("ArXiv"),
        pubmed_id=external.get("PubMed"),
        external_id_fields=sorted(external),
        error_type=snapshot.error_type,
        http_status=snapshot.http_status,
    )


def _http_terminal(status: int) -> dict[str, Any]:
    if status == 404:
        return {
            "status": "unavailable",
            "error_type": "status_404",
            "http_status": status,
        }
    if status in {401, 403, 429}:
        error_type = f"status_{status}"
    elif 500 <= status <= 599:
        error_type = "status_5xx"
    else:
        error_type = "other_http_status"
    return {"status": "failed", "error_type": error_type, "http_status": status}


def _retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _bounded_retry_wait(response_or_error: Any, attempt: int) -> float:
    retry_after = _retry_after_seconds(response_or_error)
    if retry_after is None:
        retry_after = 0.5 * (attempt + 1)
    return min(MAX_RETRY_WAIT_SECONDS, max(0.0, retry_after))


def _request_url(corpus_id: str) -> str:
    params = urlencode({"fields": ",".join(CROSSWALK_FIELDS)})
    return f"{SEMANTIC_SCHOLAR_PAPER_URL.format(corpus_id=quote(corpus_id, safe=''))}?{params}"


def _unique_corpus_ids(values: Iterable[Any]) -> list[str]:
    return sorted(
        {_required_corpus_id(value) for value in values},
        key=_corpus_sort_key,
    )


def _required_corpus_id(value: Any) -> str:
    normalized = normalize_s2orc_corpus_id(value)
    if not normalized:
        raise ValueError("SciFact crosswalk requires a Corpus ID")
    return normalized


def _corpus_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
