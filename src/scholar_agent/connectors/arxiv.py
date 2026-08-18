"""arXiv public API connector."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from scholar_agent.connectors.schemas import ConnectorRequestSpec, ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls
from scholar_agent.retrieval.query_adapter import adapt_query_for_source


logger = logging.getLogger(__name__)

ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
ARXIV_MIN_INTERVAL_SECONDS_ENV = "SCHOLAR_AGENT_ARXIV_MIN_INTERVAL_SECONDS"
DEFAULT_MIN_INTERVAL_SECONDS = 3.0
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_REQUEST_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC: float | None = None


def search_arxiv(query: str, limit: int = 20) -> list[Paper]:
    """Search papers from the arXiv public API."""

    return search_arxiv_detailed(query, limit).papers


def describe_arxiv_search_request(
    query: str, limit: int = 20
) -> ConnectorRequestSpec:
    """Describe the exact credential-free request assembled by this connector."""

    adapted = adapt_query_for_source(query, "arxiv")
    return ConnectorRequestSpec(
        source="arxiv",
        adapted_query=adapted.query,
        endpoint_alias="arxiv_api_query",
        method="GET",
        parameters={
            "search_query": adapted.query,
            "start": "0",
            "max_results": str(limit),
        },
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        auth_scope_alias="public_anonymous",
        accept="application/atom+xml",
        response_media_type="application/atom+xml",
        page_budget=1,
        pagination_strategy="fixed_first_page_no_cursor",
    )


def search_arxiv_detailed(
    query: str,
    limit: int = 20,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> ConnectorSearchResult:
    """Search papers from the arXiv public API with diagnostic details."""

    start = time.perf_counter()
    adapted = adapt_query_for_source(query, "arxiv")
    spec = describe_arxiv_search_request(query, limit)
    query = spec.adapted_query
    if not query or limit <= 0:
        return ConnectorSearchResult(
            warnings=list(adapted.warnings),
            diagnostics=ConnectorDiagnostics(
                latency_seconds=time.perf_counter() - start
            )
        )

    request = Request(
        f"{ARXIV_QUERY_URL}?{urlencode(spec.parameters)}",
        headers={"User-Agent": "ScholarNavigator"},
    )

    payload, error_message, warnings, diagnostics = _request_feed_detailed(
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

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        message = f"arXiv search failed: {exc}"
        logger.warning(message)
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
    for entry in root.findall(f"{ATOM_NS}entry"):
        try:
            paper = _parse_entry(entry)
        except Exception as exc:  # noqa: BLE001 - isolate malformed records
            message = f"Failed to parse arXiv entry: {exc}"
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


def _request_feed_detailed(
    request: Request,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> tuple[bytes | None, str | None, list[str], ConnectorDiagnostics]:
    started_at = time.perf_counter()
    warnings: list[str] = []
    attempts = max(0, max_retries) + 1
    sleep = retry_sleep or time.sleep

    request_count = 0
    retry_count = 0
    rate_limit_wait_seconds = 0.0
    retry_after_seen: float | None = None
    for attempt in range(attempts):
        rate_limit_wait_seconds += _throttle_arxiv_request(
            sleep=throttle_sleep,
            monotonic=monotonic,
        )
        request_count += 1
        retry_count += int(attempt > 0)
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", getattr(response, "code", 200))
                if status < 200 or status >= 300:
                    message = f"arXiv search returned non-2xx status: {status}"
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
                        )
                        sleep(_retry_backoff_seconds(attempt, response))
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
                return response.read(), None, warnings, ConnectorDiagnostics(
                    request_count=request_count,
                    retry_count=retry_count,
                    rate_limit_wait_seconds=rate_limit_wait_seconds,
                    retry_after_seconds=retry_after_seen,
                    latency_seconds=time.perf_counter() - started_at,
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
                )
                sleep(_retry_backoff_seconds(attempt, exc))
                continue
            message = f"arXiv search failed: {exc}"
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
                )
                sleep(_retry_backoff_seconds(attempt))
                continue
            message = f"arXiv search failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                retry_after_seconds=retry_after_seen,
                latency_seconds=time.perf_counter() - started_at,
            )

    message = "arXiv search failed: retry attempts exhausted"
    logger.warning(message)
    return None, message, warnings + [message], ConnectorDiagnostics(
        request_count=request_count,
        retry_count=retry_count,
        error_count=1,
        rate_limit_wait_seconds=rate_limit_wait_seconds,
        retry_after_seconds=retry_after_seen,
        latency_seconds=time.perf_counter() - started_at,
    )


def _with_total_latency(
    diagnostics: ConnectorDiagnostics,
    started_at: float,
) -> ConnectorDiagnostics:
    return diagnostics.model_copy(
        update={"latency_seconds": time.perf_counter() - started_at}
    )


def _record_retry_warning(
    warnings: list[str],
    *,
    attempt: int,
    attempts: int,
    reason: str,
) -> None:
    message = (
        f"arXiv search transient error on attempt {attempt + 1}/{attempts}: "
        f"{reason}; retried"
    )
    logger.warning(message)
    warnings.append(message)


def _retry_backoff_seconds(
    attempt: int,
    response_or_error: Any | None = None,
) -> float:
    retry_after = _retry_after_seconds(response_or_error)
    if retry_after is not None:
        return retry_after
    return DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1)


def _arxiv_min_interval_seconds() -> float:
    raw_value = os.getenv(ARXIV_MIN_INTERVAL_SECONDS_ENV)
    if raw_value is None:
        return DEFAULT_MIN_INTERVAL_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_MIN_INTERVAL_SECONDS
    return value if value > 0 else 0.0


def _throttle_arxiv_request(
    *,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> float:
    min_interval = _arxiv_min_interval_seconds()
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


def _reset_arxiv_throttle_for_tests() -> None:
    global _LAST_REQUEST_MONOTONIC
    with _REQUEST_THROTTLE_LOCK:
        _LAST_REQUEST_MONOTONIC = None


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


def _max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _should_retry_status(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status <= 599)


def _parse_entry(entry: ET.Element) -> Paper | None:
    landing_page = _text(entry.find(f"{ATOM_NS}id"))
    title = _normalize_space(_text(entry.find(f"{ATOM_NS}title"))) or "Untitled arXiv Paper"
    abstract = _normalize_space(_text(entry.find(f"{ATOM_NS}summary"))) or ""
    published = _text(entry.find(f"{ATOM_NS}published")) or _text(entry.find(f"{ATOM_NS}updated"))
    year = _parse_year(published)
    authors = [
        name
        for author in entry.findall(f"{ATOM_NS}author")
        if (name := _normalize_space(_text(author.find(f"{ATOM_NS}name"))))
    ]
    doi = _normalize_space(_text(entry.find(f"{ARXIV_NS}doi")))
    venue = _normalize_space(_text(entry.find(f"{ARXIV_NS}journal_ref"))) or "arXiv"
    arxiv_id = _extract_arxiv_id(landing_page)
    pdf_url = _extract_pdf_url(entry, landing_page)

    return Paper(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(
            doi=doi,
            arxiv_id=arxiv_id,
        ),
        urls=PaperUrls(
            landing_page=landing_page,
            pdf=pdf_url,
        ),
        sources=["arxiv"],
        citation_count=0,
    )


def _extract_pdf_url(entry: ET.Element, landing_page: str | None) -> str | None:
    for link in entry.findall(f"{ATOM_NS}link"):
        href = link.attrib.get("href")
        if not href:
            continue
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return href
    if landing_page and "/abs/" in landing_page:
        return landing_page.replace("/abs/", "/pdf/")
    return None


def _extract_arxiv_id(landing_page: str | None) -> str | None:
    if not landing_page:
        return None
    parsed = urlparse(landing_page)
    raw_id = parsed.path.rstrip("/").split("/")[-1] or landing_page.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", raw_id)


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).year
    except ValueError:
        match = re.match(r"(\d{4})", value)
        if match:
            return int(match.group(1))
    return None


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


def _normalize_space(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None
