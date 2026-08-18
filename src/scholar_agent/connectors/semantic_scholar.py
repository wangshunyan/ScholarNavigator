"""Semantic Scholar Graph API connector."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scholar_agent.connectors.schemas import ConnectorRequestSpec, ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls
from scholar_agent.retrieval.query_adapter import adapt_query_for_source


logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers"
)
SEMANTIC_SCHOLAR_PAPER_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
)
SEMANTIC_SCHOLAR_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS_ENV = (
    "SCHOLAR_AGENT_SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS"
)
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 2.0
DEFAULT_MIN_INTERVAL_SECONDS = 1.5
DEFAULT_MIN_INTERVAL_WITH_KEY_SECONDS = 0.1
MAX_SEMANTIC_SCHOLAR_LIMIT = 100
MAX_SEMANTIC_SCHOLAR_RECOMMENDATION_SEEDS = 3
MAX_SEMANTIC_SCHOLAR_BATCH_IDS = 100
SEARCH_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "externalIds",
        "url",
        "citationCount",
    ]
)

_REQUEST_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC: float | None = None


def search_semantic_scholar(query: str, limit: int = 20) -> list[Paper]:
    """Search papers from Semantic Scholar and return parsed papers only."""

    return search_semantic_scholar_detailed(query, limit).papers


def describe_semantic_scholar_search_request(
    query: str, limit: int = 20
) -> ConnectorRequestSpec:
    """Describe the exact search request without reading the API-key environment."""

    adapted = adapt_query_for_source(query, "semantic_scholar")
    return ConnectorRequestSpec(
        source="semantic_scholar",
        adapted_query=adapted.query,
        endpoint_alias="semantic_scholar_graph_search",
        method="GET",
        parameters={
            "query": adapted.query,
            "limit": str(min(limit, MAX_SEMANTIC_SCHOLAR_LIMIT)),
            "fields": SEARCH_FIELDS,
        },
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        auth_scope_alias="semantic_scholar_api_key_optional",
        accept="application/json",
        response_media_type="application/json",
        page_budget=1,
        pagination_strategy="fixed_first_page_no_cursor",
    )


def search_semantic_scholar_detailed(
    query: str,
    limit: int = 20,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> ConnectorSearchResult:
    """Search papers from Semantic Scholar with diagnostic details."""

    start = time.perf_counter()
    adapted = adapt_query_for_source(query, "semantic_scholar")
    spec = describe_semantic_scholar_search_request(query, limit)
    query = spec.adapted_query
    if not query or limit <= 0:
        latency = time.perf_counter() - start
        return ConnectorSearchResult(
            warnings=list(adapted.warnings),
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(latency_seconds=latency),
        )

    request = Request(
        f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{urlencode(spec.parameters)}",
        headers=_semantic_scholar_headers(),
    )

    payload, error_message, warnings, diagnostics = _request_json_detailed(
        request,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
    )
    if payload is None:
        return ConnectorSearchResult(
            error_message=error_message,
            warnings=[*adapted.warnings, *warnings],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    results = payload.get("data", [])
    if not isinstance(results, list):
        message = "Semantic Scholar search response missing list data"
        return ConnectorSearchResult(
            error_message=message,
            warnings=[*adapted.warnings, *warnings, message],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(
                diagnostics.model_copy(
                    update={"error_count": diagnostics.error_count + 1}
                ),
                start,
            ),
        )

    papers: list[Paper] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            paper = _parse_paper(item)
        except Exception as exc:  # noqa: BLE001 - isolate malformed records
            message = f"Failed to parse Semantic Scholar paper: {exc}"
            logger.warning(message)
            warnings.append(message)
            continue
        if paper is not None:
            papers.append(paper)

    return ConnectorSearchResult(
        papers=papers,
        warnings=[*adapted.warnings, *warnings],
        latency_seconds=time.perf_counter() - start,
        diagnostics=_with_total_latency(diagnostics, start),
    )


def recommend_semantic_scholar_papers_detailed(
    seed_paper_ids: list[str],
    limit: int = 100,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> ConnectorSearchResult:
    """Return recommendations for up to three exact Semantic Scholar paper IDs."""

    start = time.perf_counter()
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in seed_paper_ids:
        paper_id = str(raw_id).strip()
        key = paper_id.casefold()
        if not paper_id or key in seen:
            continue
        normalized_ids.append(paper_id)
        seen.add(key)
        if len(normalized_ids) >= MAX_SEMANTIC_SCHOLAR_RECOMMENDATION_SEEDS:
            break
    if not normalized_ids or limit <= 0:
        latency = time.perf_counter() - start
        return ConnectorSearchResult(
            error_message="Semantic Scholar recommendations require a seed paper ID",
            warnings=["semantic_scholar_recommendations_missing_seed"],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(error_count=1, latency_seconds=latency),
        )

    params = {
        "limit": str(min(limit, MAX_SEMANTIC_SCHOLAR_LIMIT)),
        "fields": SEARCH_FIELDS,
    }
    request = Request(
        f"{SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL}?{urlencode(params)}",
        data=json.dumps(
            {"positivePaperIds": normalized_ids, "negativePaperIds": []},
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={
            **_semantic_scholar_headers(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    payload, error_message, warnings, diagnostics = _request_json_detailed(
        request,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
        operation="recommendations",
    )
    if payload is None:
        return ConnectorSearchResult(
            error_message=error_message,
            warnings=warnings,
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    results = payload.get("recommendedPapers")
    if not isinstance(results, list):
        message = "Semantic Scholar recommendations response missing list recommendedPapers"
        return ConnectorSearchResult(
            error_message=message,
            warnings=[*warnings, message],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(
                diagnostics.model_copy(
                    update={"error_count": diagnostics.error_count + 1}
                ),
                start,
            ),
        )

    papers: list[Paper] = []
    malformed_count = 0
    for item in results:
        if not isinstance(item, dict):
            malformed_count += 1
            continue
        try:
            paper = _parse_paper(item)
        except Exception:  # noqa: BLE001 - isolate malformed recommendation rows
            malformed_count += 1
            continue
        if paper is not None:
            papers.append(paper)
    if malformed_count:
        warnings.append(
            f"semantic_scholar_recommendations_malformed_rows:{malformed_count}"
        )
    return ConnectorSearchResult(
        papers=papers,
        warnings=warnings,
        latency_seconds=time.perf_counter() - start,
        diagnostics=_with_total_latency(diagnostics, start),
    )


def resolve_semantic_scholar_paper_ids_detailed(
    paper_ids: list[str],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> ConnectorSearchResult:
    """Resolve exact DOI/arXiv/PMID/CorpusId values with the official batch API."""

    start = time.perf_counter()
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in paper_ids:
        paper_id = str(raw_id).strip()
        key = paper_id.casefold()
        if not paper_id or key in seen:
            continue
        normalized_ids.append(paper_id)
        seen.add(key)
        if len(normalized_ids) >= MAX_SEMANTIC_SCHOLAR_BATCH_IDS:
            break
    if not normalized_ids:
        latency = time.perf_counter() - start
        return ConnectorSearchResult(
            error_message="Semantic Scholar paper batch requires an exact identifier",
            warnings=["semantic_scholar_paper_batch_missing_identifier"],
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(error_count=1, latency_seconds=latency),
            reference_batch_status="failed",
        )

    request = Request(
        f"{SEMANTIC_SCHOLAR_PAPER_BATCH_URL}?{urlencode({'fields': SEARCH_FIELDS})}",
        data=json.dumps(
            {"ids": normalized_ids},
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={
            **_semantic_scholar_headers(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    payload, error_message, warnings, diagnostics = _request_json_detailed(
        request,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
        operation="paper batch",
    )
    if payload is None:
        return ConnectorSearchResult(
            error_message=error_message,
            warnings=warnings,
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
            reference_batch_status="failed",
            missing_reference_ids=list(normalized_ids),
            reference_batch_count=1,
        )
    if not isinstance(payload, list):
        message = "Semantic Scholar paper batch response must be a list"
        return ConnectorSearchResult(
            error_message=message,
            warnings=[*warnings, message],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(
                diagnostics.model_copy(
                    update={"error_count": diagnostics.error_count + 1}
                ),
                start,
            ),
            reference_batch_status="failed",
            missing_reference_ids=list(normalized_ids),
            reference_batch_count=1,
        )

    papers: list[Paper] = []
    missing_ids: list[str] = []
    malformed_count = 0
    for index, requested_id in enumerate(normalized_ids):
        item = payload[index] if index < len(payload) else None
        if item is None:
            missing_ids.append(requested_id)
            continue
        if not isinstance(item, dict):
            malformed_count += 1
            missing_ids.append(requested_id)
            continue
        try:
            paper = _parse_paper(item)
        except Exception:  # noqa: BLE001 - isolate malformed batch rows
            paper = None
        if paper is None or not paper.identifiers.semantic_scholar_id:
            malformed_count += 1
            missing_ids.append(requested_id)
            continue
        papers.append(paper)
    if len(payload) > len(normalized_ids):
        malformed_count += len(payload) - len(normalized_ids)
    if malformed_count:
        warnings.append(f"semantic_scholar_paper_batch_malformed_rows:{malformed_count}")
    status = (
        "success"
        if not missing_ids and not malformed_count
        else "partial_success"
        if papers
        else "missing_id"
    )
    return ConnectorSearchResult(
        papers=papers,
        warnings=warnings,
        latency_seconds=time.perf_counter() - start,
        diagnostics=_with_total_latency(diagnostics, start),
        reference_batch_status=status,
        missing_reference_ids=missing_ids,
        reference_batch_count=1,
    )


def _semantic_scholar_headers() -> dict[str, str]:
    headers = {"User-Agent": "ScholarNavigator"}
    api_key = os.getenv(SEMANTIC_SCHOLAR_API_KEY_ENV, "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _request_json_detailed(
    request: Request,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    operation: str = "search",
) -> tuple[
    Any | None,
    str | None,
    list[str],
    ConnectorDiagnostics,
]:
    started_at = time.perf_counter()
    warnings: list[str] = []
    attempts = max(0, max_retries) + 1
    sleep = retry_sleep or time.sleep
    request_count = 0
    retry_count = 0
    rate_limit_wait_seconds = 0.0
    retry_after_seen: float | None = None

    for attempt in range(attempts):
        try:
            rate_limit_wait_seconds += _throttle_semantic_scholar_request(
                sleep=throttle_sleep,
                monotonic=monotonic,
            )
            request_count += 1
            retry_count += int(attempt > 0)
            with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", getattr(response, "code", 200))
                if status < 200 or status >= 300:
                    message = f"Semantic Scholar {operation} returned non-2xx status: {status}"
                    retry_after_seen = _max_optional(
                        retry_after_seen,
                        _retry_after_seconds(response),
                    )
                    if _should_retry_status(status) and attempt < attempts - 1:
                        _record_retry_warning(
                            warnings,
                            attempt=attempt,
                            attempts=attempts,
                            reason=message,
                            operation=operation,
                        )
                        wait_seconds = _retry_backoff_seconds(attempt, response)
                        sleep(wait_seconds)
                        if _is_rate_limit_wait(status, response):
                            rate_limit_wait_seconds += wait_seconds
                        continue
                    logger.warning(message)
                    return None, message, warnings + [message], ConnectorDiagnostics(
                        request_count=request_count,
                        retry_count=retry_count,
                        error_count=1,
                        rate_limit_wait_seconds=rate_limit_wait_seconds,
                        retry_after_seconds=retry_after_seen,
                        latency_seconds=time.perf_counter() - started_at,
                    )
                return (
                    json.loads(response.read().decode("utf-8")),
                    None,
                    warnings,
                    ConnectorDiagnostics(
                        request_count=request_count,
                        retry_count=retry_count,
                        rate_limit_wait_seconds=rate_limit_wait_seconds,
                        retry_after_seconds=retry_after_seen,
                        latency_seconds=time.perf_counter() - started_at,
                    ),
                )
        except HTTPError as exc:
            retry_after_seen = _max_optional(
                retry_after_seen,
                _retry_after_seconds(exc),
            )
            if _should_retry_status(exc.code) and attempt < attempts - 1:
                _record_retry_warning(
                    warnings,
                    attempt=attempt,
                    attempts=attempts,
                    reason=str(exc),
                    operation=operation,
                )
                wait_seconds = _retry_backoff_seconds(attempt, exc)
                sleep(wait_seconds)
                if _is_rate_limit_wait(exc.code, exc):
                    rate_limit_wait_seconds += wait_seconds
                continue
            message = f"Semantic Scholar {operation} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                retry_after_seconds=retry_after_seen,
                latency_seconds=time.perf_counter() - started_at,
            )
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < attempts - 1:
                _record_retry_warning(
                    warnings,
                    attempt=attempt,
                    attempts=attempts,
                    reason=str(exc),
                    operation=operation,
                )
                sleep(_retry_backoff_seconds(attempt))
                continue
            message = f"Semantic Scholar {operation} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                latency_seconds=time.perf_counter() - started_at,
            )
        except json.JSONDecodeError as exc:
            message = f"Semantic Scholar {operation} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                latency_seconds=time.perf_counter() - started_at,
            )

    message = f"Semantic Scholar {operation} failed: retry attempts exhausted"
    logger.warning(message)
    return None, message, warnings + [message], ConnectorDiagnostics(
        request_count=request_count,
        retry_count=retry_count,
        error_count=1,
        rate_limit_wait_seconds=rate_limit_wait_seconds,
        latency_seconds=time.perf_counter() - started_at,
    )


def _semantic_scholar_min_interval_seconds() -> float:
    raw_value = os.getenv(SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS_ENV)
    if raw_value is None:
        return (
            DEFAULT_MIN_INTERVAL_WITH_KEY_SECONDS
            if _semantic_scholar_has_api_key()
            else DEFAULT_MIN_INTERVAL_SECONDS
        )
    try:
        value = float(raw_value)
    except ValueError:
        return (
            DEFAULT_MIN_INTERVAL_WITH_KEY_SECONDS
            if _semantic_scholar_has_api_key()
            else DEFAULT_MIN_INTERVAL_SECONDS
        )
    return value if value > 0 else 0.0


def _semantic_scholar_has_api_key() -> bool:
    return bool(os.getenv(SEMANTIC_SCHOLAR_API_KEY_ENV, "").strip())


def _throttle_semantic_scholar_request(
    *,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> float:
    min_interval = _semantic_scholar_min_interval_seconds()
    if min_interval <= 0:
        return 0.0

    sleep_fn = sleep or time.sleep
    monotonic_fn = monotonic or time.monotonic

    global _LAST_REQUEST_MONOTONIC
    waited = 0.0
    with _REQUEST_THROTTLE_LOCK:
        now = monotonic_fn()
        if _LAST_REQUEST_MONOTONIC is not None:
            wait_seconds = min_interval - (now - _LAST_REQUEST_MONOTONIC)
            if wait_seconds > 0:
                sleep_fn(wait_seconds)
                waited = wait_seconds
                now = monotonic_fn()
        _LAST_REQUEST_MONOTONIC = now
    return waited


def _reset_semantic_scholar_throttle_for_tests() -> None:
    global _LAST_REQUEST_MONOTONIC
    with _REQUEST_THROTTLE_LOCK:
        _LAST_REQUEST_MONOTONIC = None


def _record_retry_warning(
    warnings: list[str],
    *,
    attempt: int,
    attempts: int,
    reason: str,
    operation: str = "search",
) -> None:
    message = (
        f"Semantic Scholar {operation} transient error on attempt {attempt + 1}/{attempts}: "
        f"{reason}; retried"
    )
    logger.warning(message)
    warnings.append(message)


def _retry_backoff_seconds(attempt: int, response_or_error: Any | None = None) -> float:
    retry_after = _retry_after_seconds(response_or_error)
    if retry_after is not None:
        return retry_after
    if _status_code(response_or_error) == 429:
        return DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
    return DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1)


def _retry_after_seconds(response_or_error: Any | None) -> float | None:
    headers = getattr(response_or_error, "headers", None)
    if headers is None:
        return None
    try:
        raw_value = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw_value is None:
        return None
    try:
        value = float(str(raw_value).strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _is_rate_limit_wait(
    status: int | None,
    response_or_error: Any | None,
) -> bool:
    return status == 429 or _retry_after_seconds(response_or_error) is not None


def _max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _with_total_latency(
    diagnostics: ConnectorDiagnostics,
    started_at: float,
) -> ConnectorDiagnostics:
    return diagnostics.model_copy(
        update={"latency_seconds": time.perf_counter() - started_at}
    )


def _status_code(response_or_error: Any | None) -> int | None:
    status = getattr(response_or_error, "status", None)
    if status is None:
        status = getattr(response_or_error, "code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _should_retry_status(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status <= 599)


def _parse_paper(item: dict[str, Any]) -> Paper | None:
    title = _normalize_space(item.get("title")) or "Untitled Semantic Scholar Paper"
    external_ids = item.get("externalIds")
    external_ids = external_ids if isinstance(external_ids, dict) else {}
    paper_id = _normalize_space(item.get("paperId"))
    landing_page = _normalize_space(item.get("url"))
    if not landing_page and paper_id:
        landing_page = f"https://www.semanticscholar.org/paper/{paper_id}"

    authors = [
        name
        for author in item.get("authors") or []
        if isinstance(author, dict)
        if (name := _normalize_space(author.get("name")))
    ]

    return Paper(
        title=title,
        authors=authors,
        year=_parse_year(item.get("year")),
        venue=_normalize_space(item.get("venue")),
        abstract=_normalize_space(item.get("abstract")) or "",
        identifiers=PaperIdentifiers(
            doi=_normalize_doi(external_ids.get("DOI")),
            arxiv_id=_normalize_space(external_ids.get("ArXiv")),
            semantic_scholar_id=paper_id,
            s2orc_corpus_id=_normalize_space(
                external_ids.get("CorpusId") or item.get("corpusId")
            ),
            pubmed_id=_normalize_pubmed_id(external_ids.get("PubMed")),
        ),
        urls=PaperUrls(landing_page=landing_page),
        sources=["semantic_scholar"],
        citation_count=_parse_int(item.get("citationCount")),
    )


def _normalize_space(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _normalize_doi(value: Any) -> str | None:
    text = _normalize_space(value)
    if not text:
        return None
    text = text.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    text = text.removeprefix("doi:")
    return text.strip()


def _normalize_pubmed_id(value: Any) -> str | None:
    text = _normalize_space(value)
    if not text:
        return None
    return text.rstrip("/").split("/")[-1]


def _parse_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1800 <= year <= 2200 else None


def _parse_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
