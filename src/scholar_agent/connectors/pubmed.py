"""PubMed E-utilities connector."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scholar_agent.connectors.schemas import ConnectorRequestSpec, ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls
from scholar_agent.retrieval.query_adapter import adapt_query_for_source


logger = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY_ENV = "NCBI_API_KEY"
PUBMED_API_KEY_ENV = "PUBMED_API_KEY"
PUBMED_MIN_INTERVAL_SECONDS_ENV = "SCHOLAR_AGENT_PUBMED_MIN_INTERVAL_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MIN_INTERVAL_SECONDS = 0.34
MAX_PUBMED_LIMIT = 100

_REQUEST_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC: float | None = None


def search_pubmed(query: str, limit: int = 20) -> list[Paper]:
    """Search papers from PubMed and return parsed papers only."""

    return search_pubmed_detailed(query, limit).papers


def describe_pubmed_search_request(
    query: str, limit: int = 20
) -> ConnectorRequestSpec:
    """Describe esearch plus its response-dependent efetch child."""

    adapted = adapt_query_for_source(query, "pubmed")
    return ConnectorRequestSpec(
        source="pubmed",
        adapted_query=adapted.query,
        endpoint_alias="pubmed_esearch",
        method="GET",
        parameters=_pubmed_esearch_parameters(adapted.query, limit),
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=0,
        auth_scope_alias="ncbi_api_key_optional",
        accept="application/json",
        response_media_type="application/json",
        page_budget=1,
        pagination_strategy="fixed_esearch_then_response_dependent_fetch",
        response_dependent_children=[
            {
                "endpoint_alias": "pubmed_efetch",
                "method": "GET",
                "parent_field": "esearchresult.idlist",
                "parameter_rule": "response_order_pmids_joined_by_comma",
                "cursor_or_ids": "not_materialized_before_response",
                "accept": "application/xml",
                "response_media_type": "application/xml",
                "max_retries": 0,
            }
        ],
    )


def search_pubmed_detailed(
    query: str,
    limit: int = 20,
    *,
    throttle_sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> ConnectorSearchResult:
    """Search papers from PubMed with diagnostic details."""

    start = time.perf_counter()
    adapted = adapt_query_for_source(query, "pubmed")
    spec = describe_pubmed_search_request(query, limit)
    query = spec.adapted_query
    if not query or limit <= 0:
        latency = time.perf_counter() - start
        return ConnectorSearchResult(
            warnings=list(adapted.warnings),
            latency_seconds=latency,
            diagnostics=ConnectorDiagnostics(latency_seconds=latency),
        )

    ids, search_error, search_warnings, search_diagnostics = _esearch_ids(
        query,
        limit=limit,
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
    )
    if search_error is not None:
        return ConnectorSearchResult(
            error_message=search_error,
            warnings=[*adapted.warnings, *search_warnings],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(search_diagnostics, start),
        )
    if not ids:
        return ConnectorSearchResult(
            warnings=[*adapted.warnings, *search_warnings],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(search_diagnostics, start),
        )

    payload, fetch_error, fetch_warnings, fetch_diagnostics = _efetch_articles(
        ids,
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
    )
    warnings = [*adapted.warnings, *search_warnings, *fetch_warnings]
    diagnostics = merge_connector_diagnostics(
        [search_diagnostics, fetch_diagnostics]
    )
    if fetch_error is not None:
        return ConnectorSearchResult(
            error_message=fetch_error,
            warnings=warnings,
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    papers: list[Paper] = []
    try:
        root = ET.fromstring(payload or b"")
    except ET.ParseError as exc:
        message = f"PubMed efetch parse failed: {exc}"
        logger.warning(message)
        return ConnectorSearchResult(
            error_message=message,
            warnings=warnings + [message],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(
                diagnostics.model_copy(
                    update={"error_count": diagnostics.error_count + 1}
                ),
                start,
            ),
        )

    for article in root.findall(".//PubmedArticle"):
        try:
            paper = _parse_article(article)
        except Exception as exc:  # noqa: BLE001 - isolate malformed records
            message = f"Failed to parse PubMed article: {exc}"
            logger.warning(message)
            warnings.append(message)
            continue
        if paper is not None:
            papers.append(paper)

    return ConnectorSearchResult(
        papers=papers,
        warnings=warnings,
        latency_seconds=time.perf_counter() - start,
        diagnostics=_with_total_latency(diagnostics, start),
    )


def _esearch_ids(
    query: str,
    *,
    limit: int,
    throttle_sleep: Callable[[float], None] | None,
    monotonic: Callable[[], float] | None,
) -> tuple[list[str], str | None, list[str], ConnectorDiagnostics]:
    params = _with_api_key(_pubmed_esearch_parameters(query, limit))
    request = Request(
        f"{PUBMED_ESEARCH_URL}?{urlencode(params)}",
        headers={"User-Agent": "ScholarNavigator"},
    )
    payload, error_message, warnings, diagnostics = _request_bytes(
        request,
        label="PubMed esearch",
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
    )
    if payload is None:
        return [], error_message, warnings, diagnostics

    try:
        data = json.loads(payload.decode("utf-8"))
        raw_ids = data.get("esearchresult", {}).get("idlist", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        message = f"PubMed esearch parse failed: {exc}"
        logger.warning(message)
        return [], message, warnings + [message], diagnostics.model_copy(
            update={"error_count": diagnostics.error_count + 1}
        )

    if not isinstance(raw_ids, list):
        message = "PubMed esearch response missing idlist"
        logger.warning(message)
        return [], message, warnings + [message], diagnostics.model_copy(
            update={"error_count": diagnostics.error_count + 1}
        )

    ids = [_normalize_pmid(item) for item in raw_ids]
    return [pmid for pmid in ids if pmid], None, warnings, diagnostics


def _pubmed_esearch_parameters(query: str, limit: int) -> dict[str, str]:
    return {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(min(limit, MAX_PUBMED_LIMIT)),
        "sort": "relevance",
    }


def _efetch_articles(
    pmids: list[str],
    *,
    throttle_sleep: Callable[[float], None] | None,
    monotonic: Callable[[], float] | None,
) -> tuple[bytes | None, str | None, list[str], ConnectorDiagnostics]:
    params = _with_api_key(
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
    )
    request = Request(
        f"{PUBMED_EFETCH_URL}?{urlencode(params)}",
        headers={"User-Agent": "ScholarNavigator"},
    )
    return _request_bytes(
        request,
        label="PubMed efetch",
        throttle_sleep=throttle_sleep,
        monotonic=monotonic,
    )


def _request_bytes(
    request: Request,
    *,
    label: str,
    throttle_sleep: Callable[[float], None] | None,
    monotonic: Callable[[], float] | None,
) -> tuple[bytes | None, str | None, list[str], ConnectorDiagnostics]:
    started_at = time.perf_counter()
    warnings: list[str] = []
    rate_limit_wait_seconds = 0.0
    try:
        rate_limit_wait_seconds = _throttle_pubmed_request(
            sleep=throttle_sleep,
            monotonic=monotonic,
        )
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", getattr(response, "code", 200))
            if status < 200 or status >= 300:
                message = f"{label} returned non-2xx status: {status}"
                logger.warning(message)
                return None, message, warnings + [message], ConnectorDiagnostics(
                    request_count=1,
                    error_count=1,
                    rate_limit_wait_seconds=rate_limit_wait_seconds,
                    latency_seconds=time.perf_counter() - started_at,
                )
            return response.read(), None, warnings, ConnectorDiagnostics(
                request_count=1,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                latency_seconds=time.perf_counter() - started_at,
            )
    except HTTPError as exc:
        message = f"{label} failed: {exc}"
        logger.warning(message)
        return None, message, warnings + [message], ConnectorDiagnostics(
            request_count=1,
            error_count=1,
            rate_limit_wait_seconds=rate_limit_wait_seconds,
            latency_seconds=time.perf_counter() - started_at,
        )
    except (URLError, TimeoutError, OSError) as exc:
        message = f"{label} failed: {exc}"
        logger.warning(message)
        return None, message, warnings + [message], ConnectorDiagnostics(
            request_count=1,
            error_count=1,
            rate_limit_wait_seconds=rate_limit_wait_seconds,
            latency_seconds=time.perf_counter() - started_at,
        )


def _with_api_key(params: dict[str, str]) -> dict[str, str]:
    api_key = os.getenv(NCBI_API_KEY_ENV, "").strip() or os.getenv(
        PUBMED_API_KEY_ENV,
        "",
    ).strip()
    if api_key:
        params = dict(params)
        params["api_key"] = api_key
    return params


def _pubmed_min_interval_seconds() -> float:
    raw_value = os.getenv(PUBMED_MIN_INTERVAL_SECONDS_ENV)
    if raw_value is None:
        return DEFAULT_MIN_INTERVAL_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_MIN_INTERVAL_SECONDS
    return value if value > 0 else 0.0


def _throttle_pubmed_request(
    *,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> float:
    min_interval = _pubmed_min_interval_seconds()
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


def _with_total_latency(
    diagnostics: ConnectorDiagnostics,
    started_at: float,
) -> ConnectorDiagnostics:
    return diagnostics.model_copy(
        update={"latency_seconds": time.perf_counter() - started_at}
    )


def _reset_pubmed_throttle_for_tests() -> None:
    global _LAST_REQUEST_MONOTONIC
    with _REQUEST_THROTTLE_LOCK:
        _LAST_REQUEST_MONOTONIC = None


def _parse_article(article: ET.Element) -> Paper | None:
    pmid = _normalize_pmid(_text(article.find(".//MedlineCitation/PMID")))
    title = _normalize_space(_text(article.find(".//Article/ArticleTitle")))
    if not title:
        title = "Untitled PubMed Paper"

    journal_title = _normalize_space(_text(article.find(".//Journal/Title")))
    journal_iso = _normalize_space(_text(article.find(".//Journal/ISOAbbreviation")))
    abstract_parts = [
        _normalize_space("".join(abstract.itertext()))
        for abstract in article.findall(".//Abstract/AbstractText")
    ]
    abstract = " ".join(part for part in abstract_parts if part)
    doi = _article_doi(article)

    return Paper(
        title=title,
        authors=_article_authors(article),
        year=_article_year(article),
        venue=journal_title or journal_iso or "PubMed",
        abstract=abstract,
        identifiers=PaperIdentifiers(
            doi=doi,
            pubmed_id=pmid,
        ),
        urls=PaperUrls(
            landing_page=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        ),
        sources=["pubmed"],
        citation_count=0,
    )


def _article_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        collective_name = _normalize_space(_text(author.find("CollectiveName")))
        if collective_name:
            authors.append(collective_name)
            continue
        last_name = _normalize_space(_text(author.find("LastName")))
        fore_name = _normalize_space(_text(author.find("ForeName")))
        initials = _normalize_space(_text(author.find("Initials")))
        if last_name and fore_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name and initials:
            authors.append(f"{initials} {last_name}")
        elif last_name:
            authors.append(last_name)
    return authors


def _article_year(article: ET.Element) -> int | None:
    for path in (
        ".//ArticleDate/Year",
        ".//JournalIssue/PubDate/Year",
        ".//PubMedPubDate/Year",
    ):
        year = _parse_year(_text(article.find(path)))
        if year is not None:
            return year
    medline_date = _text(article.find(".//JournalIssue/PubDate/MedlineDate"))
    return _parse_year(medline_date)


def _article_doi(article: ET.Element) -> str | None:
    for element in article.findall(".//ArticleIdList/ArticleId"):
        if element.attrib.get("IdType", "").casefold() == "doi":
            return _normalize_doi(_text(element))
    for element in article.findall(".//ELocationID"):
        if element.attrib.get("EIdType", "").casefold() == "doi":
            return _normalize_doi(_text(element))
    return None


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


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


def _normalize_pmid(value: Any) -> str | None:
    text = _normalize_space(value)
    if not text:
        return None
    return text.rstrip("/").split("/")[-1]


def _parse_year(value: Any) -> int | None:
    text = _normalize_space(value)
    if not text:
        return None
    for token in text.replace("-", " ").split():
        if len(token) >= 4 and token[:4].isdigit():
            year = int(token[:4])
            return year if 1800 <= year <= 2200 else None
    return None
