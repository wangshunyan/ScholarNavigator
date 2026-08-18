"""Exact-identifier source-indexability audit for BEIR SciFact.

This module is evaluator-only.  Gold identifiers are used solely to issue
isolated oracle lookups and never enter product retrieval, ranking, or APIs.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError

from scholar_agent.connectors.arxiv import _throttle_arxiv_request
from scholar_agent.connectors.pubmed import _throttle_pubmed_request
from scholar_agent.connectors.semantic_scholar import (
    _semantic_scholar_headers,
    _throttle_semantic_scholar_request,
)
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    identity_evidence,
    normalize_arxiv_id,
    normalize_doi,
    normalize_s2orc_corpus_id,
    normalize_simple_id,
)
from scholar_agent.evaluation.datasets.beir_scifact import (
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.metrics import gold_crosswalk_status


AUDIT_SCHEMA_VERSION = "1"
AUDIT_CONNECTOR_VERSION = "scifact-source-index-v1"
SOURCES = ("arxiv", "openalex", "semantic_scholar", "pubmed")
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1
MAX_RETRY_WAIT_SECONDS = 5.0

ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_PAPER_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/{identifier}"
)
PUBMED_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

LookupStatus = Literal[
    "success", "not_found", "failed", "source_outage", "not_applicable"
]
SourceTerminal = Literal[
    "exact_hit",
    "not_found",
    "source_unavailable",
    "not_applicable",
    "identity_evidence_insufficient",
]
GoldClassification = Literal[
    "current_query_recalled",
    "source_exactly_locatable_query_miss",
    "applicable_sources_not_located",
    "source_unavailable",
    "identity_evidence_insufficient",
]


class ExactLookupRequest(BaseModel):
    """Canonical request key for one source-specific exact identifier lookup."""

    schema_version: str = AUDIT_SCHEMA_VERSION
    connector_version: str = AUDIT_CONNECTOR_VERSION
    key: str = Field(min_length=64, max_length=64)
    source: Literal["arxiv", "openalex", "semantic_scholar", "pubmed"]
    audit_subject_id: str
    identifier_type: str | None = None
    identifier_value: str | None = None
    applicable: bool
    request_kind: str = "exact_identifier"


class ExactLookupResponse(BaseModel):
    """Auditable terminal response without credentials, headers, or URLs."""

    status: LookupStatus
    requested: bool
    returned_identifiers: dict[str, str] = Field(default_factory=dict)
    error_type: str | None = None
    http_status: int | None = None
    request_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    preflight_evidence: dict[str, Any] | None = None
    preflight_evidence_hash: str | None = None


class ExactLookupSnapshot(BaseModel):
    schema_version: str = AUDIT_SCHEMA_VERSION
    request: ExactLookupRequest
    response: ExactLookupResponse
    content_hash: str = Field(min_length=64, max_length=64)


class SourcePreflightEvidence(BaseModel):
    schema_version: str = AUDIT_SCHEMA_VERSION
    connector_version: str = AUDIT_CONNECTOR_VERSION
    source: Literal["arxiv", "openalex", "semantic_scholar", "pubmed"]
    timestamp: str
    status: LookupStatus
    identifier_type: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    request_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    content_hash: str = Field(min_length=64, max_length=64)


class ExactLookupStore:
    """Strict immutable store for exact-identifier oracle snapshots."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries_dir = self.root / "entries"

    def path_for(self, request: ExactLookupRequest) -> Path:
        return self.entries_dir / f"{request.key}.json"

    def read(self, request: ExactLookupRequest) -> ExactLookupSnapshot:
        path = self.path_for(request)
        try:
            snapshot = ExactLookupSnapshot.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ValueError(f"invalid source-index snapshot: {path.name}") from exc
        if snapshot.request != request or snapshot.request.key != request.key:
            raise ValueError(f"source-index snapshot request mismatch: {path.name}")
        if snapshot.content_hash != snapshot_content_hash(snapshot):
            raise ValueError(f"source-index snapshot hash mismatch: {path.name}")
        _validate_response(snapshot.response, request)
        return snapshot

    def write(
        self, request: ExactLookupRequest, response: ExactLookupResponse
    ) -> None:
        path = self.path_for(request)
        if path.exists():
            self.read(request)
            return
        snapshot = ExactLookupSnapshot(
            request=request,
            response=response,
            content_hash="0" * 64,
        )
        snapshot = snapshot.model_copy(
            update={"content_hash": snapshot_content_hash(snapshot)}
        )
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, snapshot.model_dump(mode="json"))


def snapshot_content_hash(
    snapshot: ExactLookupSnapshot | dict[str, Any],
) -> str:
    payload = (
        snapshot.model_dump(mode="json")
        if isinstance(snapshot, ExactLookupSnapshot)
        else dict(snapshot)
    )
    payload.pop("content_hash", None)
    response = dict(payload.get("response") or {})
    response.pop("recorded_at", None)
    payload["response"] = response
    return _stable_hash(payload)


def build_request_plan(
    queries: Sequence[EvalQuery],
) -> tuple[dict[str, ExactLookupRequest], list[dict[str, Any]]]:
    """Build one relation record per evaluable gold and deduplicate real calls."""

    requests: dict[str, ExactLookupRequest] = {}
    records: list[dict[str, Any]] = []
    for query in queries:
        for gold_index, gold in enumerate(query.gold_papers):
            if gold_crosswalk_status(gold) != "success":
                continue
            subject_id = normalize_s2orc_corpus_id(gold.s2orc_corpus_id)
            if not subject_id:
                raise ValueError("evaluable SciFact gold requires a Corpus ID")
            source_keys: dict[str, str] = {}
            for source in SOURCES:
                request = build_exact_request(source, gold)
                prior = requests.get(request.key)
                if prior is not None and prior != request:
                    raise ValueError("conflicting source-index request key")
                requests[request.key] = request
                source_keys[source] = request.key
            records.append(
                {
                    "case_id": query.query_id,
                    "gold_index": gold_index,
                    "audit_subject_id": subject_id,
                    "gold": gold.model_dump(mode="json"),
                    "requests": source_keys,
                }
            )
    return requests, records


def build_exact_request(source: str, gold: EvalGoldPaper) -> ExactLookupRequest:
    subject_id = normalize_s2orc_corpus_id(gold.s2orc_corpus_id)
    if not subject_id:
        raise ValueError("source-index request requires a Corpus ID audit subject")
    identifier_type, identifier_value = _select_identifier(source, gold)
    core = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "connector_version": AUDIT_CONNECTOR_VERSION,
        "source": source,
        "audit_subject_id": subject_id,
        "identifier_type": identifier_type,
        "identifier_value": identifier_value,
        "applicable": identifier_value is not None,
        "request_kind": "exact_identifier",
    }
    return ExactLookupRequest(key=_stable_hash(core), **core)


def write_preflight(path: str | Path, evidence: SourcePreflightEvidence) -> None:
    _atomic_write_json(Path(path), evidence.model_dump(mode="json"))


def read_preflight(path: str | Path, source: str) -> SourcePreflightEvidence:
    try:
        evidence = SourcePreflightEvidence.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid source preflight evidence: {source}") from exc
    if evidence.source != source or evidence.content_hash != preflight_content_hash(
        evidence
    ):
        raise ValueError(f"source preflight evidence mismatch: {source}")
    return evidence


def preflight_content_hash(
    evidence: SourcePreflightEvidence | dict[str, Any],
) -> str:
    payload = (
        evidence.model_dump(mode="json")
        if isinstance(evidence, SourcePreflightEvidence)
        else dict(evidence)
    )
    payload.pop("content_hash", None)
    return _stable_hash(payload)


def run_preflight(
    source: str,
    requests: Iterable[ExactLookupRequest],
    *,
    runner: Callable[[ExactLookupRequest, float], ExactLookupResponse],
    wall_timeout_seconds: float = DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
) -> SourcePreflightEvidence:
    applicable = sorted(
        (item for item in requests if item.source == source and item.applicable),
        key=lambda item: item.key,
    )
    if not applicable:
        response = _terminal_response("not_applicable", requested=False)
        identifier_type = None
    else:
        request = applicable[0]
        _parent_throttle(source)
        response = runner(request, wall_timeout_seconds)
        identifier_type = request.identifier_type
    evidence = SourcePreflightEvidence(
        source=source,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=response.status,
        identifier_type=identifier_type,
        error_type=response.error_type,
        http_status=response.http_status,
        request_count=response.request_count,
        retry_count=response.retry_count,
        content_hash="0" * 64,
    )
    return evidence.model_copy(
        update={"content_hash": preflight_content_hash(evidence)}
    )


def record_missing(
    requests: Iterable[ExactLookupRequest],
    store: ExactLookupStore,
    preflights: dict[str, SourcePreflightEvidence],
    *,
    source: str | None = None,
    runner: Callable[[ExactLookupRequest, float], ExactLookupResponse],
    wall_timeout_seconds: float = DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Record serial terminals; an individual failure never stops later calls."""

    selected = sorted(
        (
            item
            for item in requests
            if source is None or item.source == source
        ),
        key=lambda item: (SOURCES.index(item.source), item.key),
    )
    counts: Counter[str] = Counter()
    for request in selected:
        counts["planned"] += 1
        path = store.path_for(request)
        if path.exists():
            store.read(request)
            counts["existing"] += 1
            continue
        if not request.applicable:
            response = _terminal_response("not_applicable", requested=False)
        else:
            preflight = preflights.get(request.source)
            if preflight is None:
                raise ValueError(f"missing source preflight: {request.source}")
            if preflight.status == "failed":
                response = _source_outage_response(preflight)
            else:
                _parent_throttle(request.source)
                response = runner(request, wall_timeout_seconds)
        store.write(request, response)
        counts[response.status] += 1
        counts["written"] += 1
    return dict(sorted(counts.items()))


def replay_audit(
    *,
    queries: Sequence[EvalQuery],
    requests: dict[str, ExactLookupRequest],
    gold_plan: Sequence[dict[str, Any]],
    store: ExactLookupStore,
    external_run_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay every expected key with zero HTTP and deterministic output."""

    snapshots = {key: store.read(request) for key, request in requests.items()}
    external_rows = _load_external_rows(external_run_dir, queries)
    records: list[dict[str, Any]] = []
    for planned in gold_plan:
        gold = EvalGoldPaper.model_validate(planned["gold"])
        current_sources = _current_recalled_sources(
            external_rows[planned["case_id"]], gold
        )
        source_rows: dict[str, dict[str, Any]] = {}
        for source in SOURCES:
            key = planned["requests"][source]
            snapshot = snapshots[key]
            source_rows[source] = _source_result(snapshot.response, gold)
        classification = classify_gold(bool(current_sources), source_rows)
        records.append(
            {
                "case_id": planned["case_id"],
                "gold_index": planned["gold_index"],
                "audit_subject_id": planned["audit_subject_id"],
                "current_query_recalled": bool(current_sources),
                "current_candidate_sources": current_sources,
                "classification": classification,
                "sources": source_rows,
            }
        )
    return records, aggregate_records(records, requests, snapshots)


def classify_gold(
    current_recalled: bool, source_rows: dict[str, dict[str, Any]]
) -> GoldClassification:
    terminals = [row["terminal"] for row in source_rows.values()]
    if current_recalled:
        return "current_query_recalled"
    if "exact_hit" in terminals:
        return "source_exactly_locatable_query_miss"
    if "source_unavailable" in terminals:
        return "source_unavailable"
    applicable = [item for item in terminals if item != "not_applicable"]
    if not applicable or "identity_evidence_insufficient" in applicable:
        return "identity_evidence_insufficient"
    if all(item == "not_found" for item in applicable):
        return "applicable_sources_not_located"
    return "identity_evidence_insufficient"


def aggregate_records(
    records: Sequence[dict[str, Any]],
    requests: dict[str, ExactLookupRequest],
    snapshots: dict[str, ExactLookupSnapshot],
) -> dict[str, Any]:
    classifications = Counter(row["classification"] for row in records)
    classification_names = (
        "current_query_recalled",
        "source_exactly_locatable_query_miss",
        "applicable_sources_not_located",
        "source_unavailable",
        "identity_evidence_insufficient",
    )
    by_source: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        terminals = Counter(row["sources"][source]["terminal"] for row in records)
        exact_subjects = {
            row["audit_subject_id"]
            for row in records
            if row["sources"][source]["terminal"] == "exact_hit"
        }
        completed = terminals["exact_hit"] + terminals["not_found"]
        by_source[source] = {
            "gold_source_pair_count": len(records),
            "applicable_gold_count": len(records) - terminals["not_applicable"],
            "completed_gold_count": completed,
            "exact_hit_gold_count": terminals["exact_hit"],
            "exact_hit_unique_subject_count": len(exact_subjects),
            "not_found_gold_count": terminals["not_found"],
            "unavailable_gold_count": terminals["source_unavailable"],
            "not_applicable_gold_count": terminals["not_applicable"],
            "identity_evidence_insufficient_gold_count": terminals[
                "identity_evidence_insufficient"
            ],
            "exact_coverage_rate": (
                terminals["exact_hit"] / completed if completed else None
            ),
        }
    exact_union = [
        row
        for row in records
        if any(
            item["terminal"] == "exact_hit" for item in row["sources"].values()
        )
    ]
    current = [row for row in records if row["current_query_recalled"]]
    current_plus_exact = [
        row
        for row in records
        if row["current_query_recalled"]
        or any(
            item["terminal"] == "exact_hit" for item in row["sources"].values()
        )
    ]
    response_status = Counter(
        snapshot.response.status for snapshot in snapshots.values()
    )
    source_requests: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        source_snapshots = [
            snapshots[key]
            for key, request in requests.items()
            if request.source == source
        ]
        source_requests[source] = {
            "unique_snapshot_count": len(source_snapshots),
            "attempted_http_count": sum(
                item.response.request_count for item in source_snapshots
            ),
            "retry_count": sum(item.response.retry_count for item in source_snapshots),
            "latency_seconds": sum(
                item.response.latency_seconds for item in source_snapshots
            ),
            "status_counts": dict(
                sorted(Counter(item.response.status for item in source_snapshots).items())
            ),
        }
    unique_subjects = {row["audit_subject_id"] for row in records}
    return {
        "gold_count": len(records),
        "unique_gold_subject_count": len(unique_subjects),
        "source_pair_count": len(records) * len(SOURCES),
        "classification_counts": {
            name: classifications[name] for name in classification_names
        },
        "by_source": by_source,
        "joint_exact_coverage": {
            "matched_gold_count": len(exact_union),
            "gold_denominator": len(records),
            "coverage_rate": len(exact_union) / len(records) if records else None,
            "unique_subject_count": len(
                {row["audit_subject_id"] for row in exact_union}
            ),
        },
        "current_rules_candidate": {
            "matched_gold_count": len(current),
            "gold_denominator": len(records),
            "coverage_rate": len(current) / len(records) if records else None,
        },
        "current_plus_exact_coverage_upper_bound": {
            "matched_gold_count": len(current_plus_exact),
            "gold_denominator": len(records),
            "coverage_rate": (
                len(current_plus_exact) / len(records) if records else None
            ),
        },
        "strategy_space": {
            "query_expression_theoretical_max_new_gold_count": classifications[
                "source_exactly_locatable_query_miss"
            ],
            "additional_or_replacement_source_residual_gold_count": classifications[
                "applicable_sources_not_located"
            ],
            "indeterminate_gold_count": classifications["source_unavailable"]
            + classifications["identity_evidence_insufficient"],
        },
        "request_status_counts": dict(sorted(response_status.items())),
        "requests_by_source": source_requests,
        "replay_http_request_count": 0,
        "replay_write_count": 0,
    }


def run_exact_lookup_isolated(
    request: ExactLookupRequest,
    wall_timeout_seconds: float = DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
) -> ExactLookupResponse:
    """Run one potentially blocking HTTP lookup in a killable spawn process."""

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_lookup_process_entry, args=(request, child))
    process.start()
    child.close()
    try:
        if parent.poll(wall_timeout_seconds):
            payload = parent.recv()
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
            return ExactLookupResponse.model_validate(payload)
        _terminate_process(process)
        return _terminal_response(
            "failed",
            requested=True,
            error_type="audit_wall_clock_timeout",
            request_count=1,
            latency_seconds=wall_timeout_seconds,
        )
    except (EOFError, OSError, ValidationError):
        _terminate_process(process)
        return _terminal_response(
            "failed",
            requested=True,
            error_type="audit_worker_failure",
            request_count=1,
        )
    finally:
        parent.close()


def _lookup_process_entry(request: ExactLookupRequest, connection: Any) -> None:
    try:
        response = fetch_exact_lookup(request)
    except BaseException as exc:  # noqa: BLE001 - isolate worker terminal
        response = _terminal_response(
            "failed",
            requested=True,
            error_type=f"connector_exception:{type(exc).__name__}",
            request_count=1,
        )
    try:
        connection.send(response.model_dump(mode="json"))
    finally:
        connection.close()


def fetch_exact_lookup(
    request: ExactLookupRequest,
    *,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExactLookupResponse:
    if not request.applicable:
        return _terminal_response("not_applicable", requested=False)
    url, headers = _request_target(request)
    started = time.perf_counter()
    payload, terminal = _request_bytes(
        url,
        headers=headers,
        opener=opener,
        sleep=sleep,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    if payload is None:
        return ExactLookupResponse(
            recorded_at=datetime.now(timezone.utc).isoformat(),
            latency_seconds=time.perf_counter() - started,
            **terminal,
        )
    identifiers, parse_error = _parse_identifiers(request.source, payload)
    if parse_error == "not_found":
        status: LookupStatus = "not_found"
    elif parse_error:
        status = "failed"
    else:
        status = "success"
    return ExactLookupResponse(
        status=status,
        requested=True,
        returned_identifiers=identifiers,
        error_type=parse_error if parse_error != "not_found" else None,
        http_status=terminal["http_status"],
        request_count=terminal["request_count"],
        retry_count=terminal["retry_count"],
        latency_seconds=time.perf_counter() - started,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )


def write_replay_artifacts(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "config.json", config)
    (root / "gold_audit.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    _write_json(root / "aggregate.json", aggregate)
    hashes = {
        name: file_sha256(root / name)
        for name in ("config.json", "gold_audit.jsonl", "aggregate.json")
    }
    _write_json(root / "artifact_hashes.json", hashes)
    return hashes


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_identifier(
    source: str, gold: EvalGoldPaper
) -> tuple[str | None, str | None]:
    values: dict[str, str | None] = {
        "doi": normalize_doi(gold.doi),
        "arxiv_id": normalize_arxiv_id(gold.arxiv_id),
        "openalex_id": normalize_simple_id(gold.openalex_id),
        "semantic_scholar_id": normalize_simple_id(gold.semantic_scholar_id),
        "s2orc_corpus_id": normalize_s2orc_corpus_id(gold.s2orc_corpus_id),
        "pubmed_id": normalize_simple_id(gold.pubmed_id),
    }
    priority = {
        "arxiv": ("arxiv_id",),
        "openalex": ("openalex_id", "doi", "pubmed_id", "arxiv_id"),
        "semantic_scholar": (
            "s2orc_corpus_id",
            "semantic_scholar_id",
            "doi",
            "arxiv_id",
            "pubmed_id",
        ),
        "pubmed": ("pubmed_id",),
    }[source]
    return next(((field, values[field]) for field in priority if values[field]), (None, None))


def _request_target(request: ExactLookupRequest) -> tuple[str, dict[str, str]]:
    value = request.identifier_value or ""
    if request.source == "arxiv":
        return (
            f"{ARXIV_QUERY_URL}?{urlencode({'id_list': value, 'max_results': '1'})}",
            {"User-Agent": "ScholarNavigator"},
        )
    if request.source == "openalex":
        locator = {
            "openalex_id": value.upper(),
            "doi": f"https://doi.org/{value}",
            "pubmed_id": f"pmid:{value}",
            "arxiv_id": f"arxiv:{value}",
        }[str(request.identifier_type)]
        return (
            f"{OPENALEX_WORKS_URL}/{quote(locator, safe=':/')}",
            {"User-Agent": "ScholarNavigator"},
        )
    if request.source == "semantic_scholar":
        prefix = {
            "s2orc_corpus_id": "CorpusId:",
            "semantic_scholar_id": "",
            "doi": "DOI:",
            "arxiv_id": "ARXIV:",
            "pubmed_id": "PMID:",
        }[str(request.identifier_type)]
        identifier = quote(f"{prefix}{value}", safe=":")
        params = urlencode({"fields": "paperId,corpusId,externalIds"})
        return (
            f"{SEMANTIC_SCHOLAR_PAPER_URL.format(identifier=identifier)}?{params}",
            _semantic_scholar_headers(),
        )
    return (
        f"{PUBMED_EFETCH_URL}?{urlencode({'db': 'pubmed', 'id': value, 'retmode': 'xml'})}",
        {"User-Agent": "ScholarNavigator"},
    )


def _request_bytes(
    url: str,
    *,
    headers: dict[str, str],
    opener: Callable[..., Any],
    sleep: Callable[[float], None],
    timeout_seconds: float,
    max_retries: int,
) -> tuple[bytes | None, dict[str, Any]]:
    attempts = max(0, int(max_retries)) + 1
    request_count = 0
    retry_count = 0
    for attempt in range(attempts):
        request_count += 1
        retry_count += int(attempt > 0)
        try:
            with opener(Request(url, headers=headers), timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", getattr(response, "code", 200)))
                if status < 200 or status >= 300:
                    terminal = _http_terminal(status, request_count, retry_count)
                    if _retryable_status(status) and attempt < attempts - 1:
                        sleep(_retry_wait(response, attempt))
                        continue
                    return None, terminal
                return response.read(), {
                    "status": "success",
                    "requested": True,
                    "error_type": None,
                    "http_status": status,
                    "request_count": request_count,
                    "retry_count": retry_count,
                }
        except HTTPError as exc:
            terminal = _http_terminal(int(exc.code), request_count, retry_count)
            if _retryable_status(exc.code) and attempt < attempts - 1:
                sleep(_retry_wait(exc, attempt))
                continue
            return None, terminal
        except (TimeoutError, socket.timeout):
            error_type = "network_timeout"
        except URLError as exc:
            error_type = (
                "dns_error"
                if isinstance(getattr(exc, "reason", None), socket.gaierror)
                else "network_error"
            )
        except OSError:
            error_type = "network_error"
        if attempt < attempts - 1:
            sleep(min(MAX_RETRY_WAIT_SECONDS, 0.5 * (attempt + 1)))
            continue
        return None, {
            "status": "failed",
            "requested": True,
            "error_type": error_type,
            "http_status": None,
            "request_count": request_count,
            "retry_count": retry_count,
        }
    raise AssertionError("unreachable HTTP terminal")


def _http_terminal(status: int, request_count: int, retry_count: int) -> dict[str, Any]:
    if status == 404:
        terminal: LookupStatus = "not_found"
        error_type = None
    else:
        terminal = "failed"
        error_type = (
            f"status_{status}"
            if status in {401, 403, 429}
            else "status_5xx"
            if 500 <= status <= 599
            else "other_http_status"
        )
    return {
        "status": terminal,
        "requested": True,
        "error_type": error_type,
        "http_status": status,
        "request_count": request_count,
        "retry_count": retry_count,
    }


def _retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _retry_wait(response: Any, attempt: int) -> float:
    raw = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("Retry-After")
        except (AttributeError, TypeError):
            raw = None
    try:
        wait = float(raw) if raw is not None else 0.5 * (attempt + 1)
    except (TypeError, ValueError):
        wait = 0.5 * (attempt + 1)
    return min(MAX_RETRY_WAIT_SECONDS, max(0.0, wait))


def _parse_identifiers(source: str, payload: bytes) -> tuple[dict[str, str], str | None]:
    try:
        if source == "arxiv":
            return _parse_arxiv_identifiers(payload)
        if source == "openalex":
            return _parse_openalex_identifiers(json.loads(payload.decode("utf-8")))
        if source == "semantic_scholar":
            return _parse_semantic_identifiers(json.loads(payload.decode("utf-8")))
        return _parse_pubmed_identifiers(payload)
    except (ET.ParseError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {}, "invalid_response_schema"


def _parse_arxiv_identifiers(payload: bytes) -> tuple[dict[str, str], str | None]:
    root = ET.fromstring(payload)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return {}, "not_found"
    raw_id = (entry.findtext("{http://www.w3.org/2005/Atom}id") or "").strip()
    identifiers: dict[str, str] = {}
    arxiv_id = normalize_arxiv_id(raw_id)
    doi = normalize_doi(entry.findtext("{http://arxiv.org/schemas/atom}doi"))
    if arxiv_id:
        identifiers["arxiv_id"] = arxiv_id
    if doi:
        identifiers["doi"] = doi
    return (identifiers, None) if identifiers else ({}, "invalid_response_schema")


def _parse_openalex_identifiers(payload: Any) -> tuple[dict[str, str], str | None]:
    if not isinstance(payload, dict):
        return {}, "invalid_response_schema"
    ids = payload.get("ids") if isinstance(payload.get("ids"), dict) else {}
    identifiers = _normalized_identifiers(
        openalex_id=payload.get("id") or ids.get("openalex"),
        doi=payload.get("doi") or ids.get("doi"),
        pubmed_id=ids.get("pmid"),
    )
    return (identifiers, None) if identifiers else ({}, "invalid_response_schema")


def _parse_semantic_identifiers(payload: Any) -> tuple[dict[str, str], str | None]:
    if not isinstance(payload, dict):
        return {}, "invalid_response_schema"
    external = payload.get("externalIds")
    external = external if isinstance(external, dict) else {}
    identifiers = _normalized_identifiers(
        semantic_scholar_id=payload.get("paperId"),
        s2orc_corpus_id=payload.get("corpusId") or external.get("CorpusId"),
        doi=external.get("DOI"),
        arxiv_id=external.get("ArXiv"),
        pubmed_id=external.get("PubMed"),
    )
    return (identifiers, None) if identifiers else ({}, "invalid_response_schema")


def _parse_pubmed_identifiers(payload: bytes) -> tuple[dict[str, str], str | None]:
    root = ET.fromstring(payload)
    article = root.find(".//PubmedArticle")
    if article is None:
        return {}, "not_found"
    raw: dict[str, str] = {}
    # Restrict IDs to the article's own metadata.  ReferenceList also contains
    # nested ArticleIdList nodes and must never overwrite the requested work.
    pmid = article.findtext("./MedlineCitation/PMID")
    if pmid:
        raw["pubmed_id"] = pmid
    for item in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        kind = str(item.attrib.get("IdType") or "").casefold()
        value = item.text or ""
        if kind == "pubmed":
            raw["pubmed_id"] = value
        elif kind == "doi":
            raw["doi"] = value
    identifiers = _normalized_identifiers(**raw)
    return (identifiers, None) if identifiers else ({}, "invalid_response_schema")


def _normalized_identifiers(**values: Any) -> dict[str, str]:
    normalizers: dict[str, Callable[[Any], str | None]] = {
        "doi": normalize_doi,
        "arxiv_id": normalize_arxiv_id,
        "openalex_id": normalize_simple_id,
        "semantic_scholar_id": normalize_simple_id,
        "s2orc_corpus_id": normalize_s2orc_corpus_id,
        "pubmed_id": normalize_simple_id,
    }
    return {
        field: normalized
        for field, value in values.items()
        if (normalized := normalizers[field](value))
    }


def _source_result(response: ExactLookupResponse, gold: EvalGoldPaper) -> dict[str, Any]:
    if response.status == "not_applicable":
        terminal: SourceTerminal = "not_applicable"
        evidence_rule = None
        conflicts: list[str] = []
        shared: list[str] = []
    elif response.status in {"failed", "source_outage"}:
        terminal = "source_unavailable"
        evidence_rule = None
        conflicts = []
        shared = []
    elif response.status == "not_found":
        terminal = "not_found"
        evidence_rule = None
        conflicts = []
        shared = []
    else:
        evidence = identity_evidence(gold, {"identifiers": response.returned_identifiers})
        terminal = "exact_hit" if evidence.equivalent else "identity_evidence_insufficient"
        evidence_rule = evidence.rule
        conflicts = list(evidence.conflicting_identifiers)
        shared = list(evidence.shared_identifiers)
    return {
        "terminal": terminal,
        "lookup_status": response.status,
        "requested": response.requested,
        "http_status": response.http_status,
        "error_type": response.error_type,
        "request_count": response.request_count,
        "retry_count": response.retry_count,
        "identity_rule": evidence_rule,
        "shared_identifiers": shared,
        "conflicting_identifiers": conflicts,
    }


def _current_recalled_sources(row: dict[str, Any], gold: EvalGoldPaper) -> list[str]:
    diagnostics = row.get("stage_diagnostics") or {}
    snapshots = diagnostics.get("snapshots") or []
    initial = next(
        (
            item
            for item in snapshots
            if isinstance(item, dict) and item.get("stage") == "initial_retrieval"
        ),
        None,
    )
    if not isinstance(initial, dict) or initial.get("status") != "completed":
        raise ValueError("external Replay lacks completed initial retrieval")
    sources: set[str] = set()
    for candidate in initial.get("candidates") or []:
        evidence = identity_evidence(gold, candidate)
        if evidence.equivalent and evidence.shared_identifiers:
            sources.update(str(item) for item in candidate.get("sources") or [])
    return sorted(sources)


def _load_external_rows(
    run_dir: str | Path, queries: Sequence[EvalQuery]
) -> dict[str, dict[str, Any]]:
    path = Path(run_dir) / "results.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in rows:
            raise ValueError("invalid external Replay case IDs")
        if row.get("status") != "succeeded":
            raise ValueError("external Replay contains non-success terminal")
        rows[case_id] = row
    expected = {query.query_id for query in queries}
    if set(rows) != expected:
        raise ValueError("external Replay query set mismatch")
    return rows


def load_inputs(
    dataset_path: str | Path,
    sample_manifest_path: str | Path,
    crosswalk_path: str | Path,
) -> list[EvalQuery]:
    queries = load_beir_scifact_enriched(dataset_path, crosswalk_path=crosswalk_path)
    manifest = json.loads(Path(sample_manifest_path).read_text(encoding="utf-8"))
    expected = [str(item) for item in manifest.get("query_ids") or []]
    if [query.query_id for query in queries] != expected:
        raise ValueError("SciFact query order does not match fixed manifest")
    return queries


def build_config(
    *,
    dataset_path: str | Path,
    sample_manifest_path: str | Path,
    crosswalk_path: str | Path,
    external_run_dir: str | Path,
    request_count: int,
    gold_count: int,
    request_wall_timeout_seconds: float = DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "connector_version": AUDIT_CONNECTOR_VERSION,
        "dataset": "beir_scifact",
        "split": "test",
        "sources": list(SOURCES),
        "gold_count": gold_count,
        "request_key_count": request_count,
        "oracle_query": "source_specific_exact_stable_identifier_only",
        "title_matching": False,
        "product_path_access": False,
        "request_policy": {
            "serial": True,
            "http_timeout_seconds": DEFAULT_HTTP_TIMEOUT_SECONDS,
            "request_wall_timeout_seconds": request_wall_timeout_seconds,
            "max_retries": DEFAULT_MAX_RETRIES,
            "max_retry_wait_seconds": MAX_RETRY_WAIT_SECONDS,
        },
        "source_identifier_priority": {
            "arxiv": ["arxiv_id"],
            "openalex": ["openalex_id", "doi", "pubmed_id", "arxiv_id"],
            "semantic_scholar": [
                "s2orc_corpus_id",
                "semantic_scholar_id",
                "doi",
                "arxiv_id",
                "pubmed_id",
            ],
            "pubmed": ["pubmed_id"],
        },
        "source_exact_endpoints": {
            "arxiv": "Atom API id_list",
            "openalex": "Works single-entity lookup",
            "semantic_scholar": "Graph API paper stable-ID lookup",
            "pubmed": "E-utilities EFetch PMID lookup",
        },
        "inputs": {
            "dataset_sha256": file_sha256(dataset_path),
            "sample_manifest_sha256": file_sha256(sample_manifest_path),
            "crosswalk_sha256": file_sha256(crosswalk_path),
            "external_results_sha256": file_sha256(
                Path(external_run_dir) / "results.jsonl"
            ),
        },
    }


def _source_outage_response(
    evidence: SourcePreflightEvidence,
) -> ExactLookupResponse:
    payload = evidence.model_dump(mode="json")
    return _terminal_response(
        "source_outage",
        requested=False,
        error_type=evidence.error_type,
        http_status=evidence.http_status,
        preflight_evidence=payload,
        preflight_evidence_hash=evidence.content_hash,
    )


def _terminal_response(
    status: LookupStatus,
    *,
    requested: bool,
    error_type: str | None = None,
    http_status: int | None = None,
    request_count: int = 0,
    retry_count: int = 0,
    latency_seconds: float = 0.0,
    preflight_evidence: dict[str, Any] | None = None,
    preflight_evidence_hash: str | None = None,
) -> ExactLookupResponse:
    return ExactLookupResponse(
        status=status,
        requested=requested,
        error_type=error_type,
        http_status=http_status,
        request_count=request_count,
        retry_count=retry_count,
        latency_seconds=latency_seconds,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        preflight_evidence=preflight_evidence,
        preflight_evidence_hash=preflight_evidence_hash,
    )


def _validate_response(
    response: ExactLookupResponse, request: ExactLookupRequest
) -> None:
    if response.status == "not_applicable":
        if request.applicable or response.requested or response.request_count:
            raise ValueError("invalid not-applicable source-index terminal")
    if response.status == "source_outage":
        evidence = response.preflight_evidence
        if (
            response.requested
            or response.request_count
            or not isinstance(evidence, dict)
            or response.preflight_evidence_hash != _stable_hash_without_hash(evidence)
            or evidence.get("source") != request.source
            or evidence.get("status") != "failed"
        ):
            raise ValueError("invalid source-outage source-index terminal")
    if response.status in {"success", "not_found", "failed"} and not response.requested:
        raise ValueError("attempted source-index terminal marked unrequested")


def _stable_hash_without_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_hash", None)
    return _stable_hash(value)


def _parent_throttle(source: str) -> float:
    throttle = {
        "arxiv": _throttle_arxiv_request,
        "semantic_scholar": _throttle_semantic_scholar_request,
        "pubmed": _throttle_pubmed_request,
    }.get(source)
    return throttle() if throttle else 0.0


def _terminate_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(2)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(2)


def _stable_hash(payload: Any) -> str:
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
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
