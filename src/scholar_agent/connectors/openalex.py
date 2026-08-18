"""OpenAlex Works API connector."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from scholar_agent.connectors.schemas import ConnectorRequestSpec, ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls
from scholar_agent.retrieval.query_adapter import adapt_query_for_source


logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
MAX_OPENALEX_LIMIT = 200
MAX_OPENALEX_BATCH_IDS = 100


def search_openalex(query: str, limit: int = 20) -> list[Paper]:
    """Search papers from OpenAlex Works.

    The connector is deliberately fail-closed: external failures return an
    empty list and malformed individual records are skipped without aborting the
    full response.
    """

    return search_openalex_detailed(query, limit).papers


def describe_openalex_search_request(
    query: str, limit: int = 20
) -> ConnectorRequestSpec:
    """Describe the request before optional runtime polite-pool identity."""

    adapted = adapt_query_for_source(query, "openalex")
    return ConnectorRequestSpec(
        source="openalex",
        adapted_query=adapted.query,
        endpoint_alias="openalex_works_search",
        method="GET",
        parameters={
            "search": adapted.query,
            "per-page": str(min(limit, MAX_OPENALEX_LIMIT)),
        },
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        auth_scope_alias="openalex_polite_pool_optional",
        accept="application/json",
        response_media_type="application/json",
        page_budget=1,
        pagination_strategy="fixed_first_page_no_cursor",
    )


def search_openalex_detailed(
    query: str,
    limit: int = 20,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
) -> ConnectorSearchResult:
    """Search papers from OpenAlex Works with diagnostic details."""

    start = time.perf_counter()
    adapted = adapt_query_for_source(query, "openalex")
    spec = describe_openalex_search_request(query, limit)
    query = spec.adapted_query
    if not query or limit <= 0:
        return ConnectorSearchResult(
            warnings=list(adapted.warnings),
            diagnostics=ConnectorDiagnostics(
                latency_seconds=time.perf_counter() - start
            )
        )

    params = dict(spec.parameters)
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto

    payload, error_message, warnings, diagnostics = _request_json_detailed(
        f"{OPENALEX_WORKS_URL}?{urlencode(params)}",
        context="OpenAlex search",
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    if payload is None:
        return ConnectorSearchResult(
            error_message=error_message,
            warnings=[*adapted.warnings, *warnings],
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    results = payload.get("results", [])
    if not isinstance(results, list):
        message = "OpenAlex search response missing list results"
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
            paper = _parse_work(item)
        except Exception as exc:  # noqa: BLE001 - isolate malformed records
            message = f"Failed to parse OpenAlex work: {exc}"
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


def fetch_openalex_references(paper: Paper, limit: int = 20) -> list[Paper]:
    """Fetch references for a seed paper from OpenAlex metadata.

    The function uses only metadata available through the OpenAlex Works API.
    It does not read full text or infer references. Failures return an empty
    list or the successfully parsed subset of references.
    """

    return fetch_openalex_references_detailed(paper, limit).papers


def fetch_openalex_references_detailed(
    paper: Paper,
    limit: int = 20,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
) -> ConnectorSearchResult:
    """批量获取一层 OpenAlex 引用，并返回每次真实请求的诊断。"""

    start = time.perf_counter()
    if limit <= 0:
        return ConnectorSearchResult(
            diagnostics=ConnectorDiagnostics(
                latency_seconds=time.perf_counter() - start
            )
        )

    seed_work, seed_error, warnings, seed_diagnostics = _fetch_seed_work_detailed(
        paper,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    diagnostics = seed_diagnostics
    if seed_work is None:
        return ConnectorSearchResult(
            error_message=seed_error,
            warnings=warnings,
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    reference_ids = _referenced_work_ids(seed_work.get("referenced_works"))[
        : min(limit, MAX_OPENALEX_LIMIT)
    ]
    if not reference_ids:
        return ConnectorSearchResult(
            warnings=warnings,
            latency_seconds=time.perf_counter() - start,
            diagnostics=_with_total_latency(diagnostics, start),
        )

    works_by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    batch_count = 0

    def add_works(payload: dict[str, Any] | None) -> bool:
        results = (payload or {}).get("results", [])
        if not isinstance(results, list):
            return False
        for work in results:
            if not isinstance(work, dict):
                continue
            work_id = _normalize_openalex_id(work.get("id"))
            if work_id:
                works_by_id.setdefault(work_id.casefold(), work)
        return True

    for offset in range(0, len(reference_ids), MAX_OPENALEX_BATCH_IDS):
        batch_ids = reference_ids[offset : offset + MAX_OPENALEX_BATCH_IDS]
        batch_count += 1
        payload, error, batch_warnings, batch_diagnostics = (
            _fetch_works_by_openalex_ids_detailed(
                batch_ids,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        )
        diagnostics = merge_connector_diagnostics(
            [diagnostics, batch_diagnostics]
        )
        warnings.extend(batch_warnings)
        if error is not None:
            errors.append(error)
            continue
        if not add_works(payload):
            message = "OpenAlex reference batch response missing list results"
            warnings.append(message)
            errors.append(message)
            diagnostics = diagnostics.model_copy(
                update={"error_count": diagnostics.error_count + 1}
            )
            continue
    missing_ids = [
        reference_id
        for reference_id in reference_ids
        if reference_id.casefold() not in works_by_id
    ]
    supplemental_request_count = 0
    supplemental_success_count = 0
    for reference_id in missing_ids:
        supplemental_request_count += 1
        payload, error, supplement_warnings, supplement_diagnostics = (
            _fetch_works_by_openalex_ids_detailed(
                [reference_id],
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        )
        diagnostics = merge_connector_diagnostics(
            [diagnostics, supplement_diagnostics]
        )
        warnings.extend(supplement_warnings)
        if error is None and add_works(payload):
            if reference_id.casefold() in works_by_id:
                supplemental_success_count += 1
                continue
        if error:
            errors.append(error)

    missing_ids = [
        reference_id
        for reference_id in reference_ids
        if reference_id.casefold() not in works_by_id
    ]
    if missing_ids:
        missing_message = "OpenAlex reference missing work ids:" + ",".join(
            missing_ids
        )
        warnings.append(missing_message)
        errors.extend(
            f"OpenAlex reference missing work id:{reference_id}"
            for reference_id in missing_ids
        )
        diagnostics = diagnostics.model_copy(
            update={
                "error_count": diagnostics.error_count + len(missing_ids),
            }
        )
    if supplemental_request_count:
        warnings.append(
            "OpenAlex reference supplemental requests:"
            f"{supplemental_request_count}"
            f" (success:{supplemental_success_count})"
        )

    papers: list[Paper] = []
    for reference_id in reference_ids:
        work = works_by_id.get(reference_id.casefold())
        if work is None:
            continue
        try:
            reference_paper = _parse_work(work)
        except Exception as exc:  # noqa: BLE001 - isolate malformed references
            message = f"Failed to parse OpenAlex reference work: {exc}"
            logger.warning(message)
            warnings.append(message)
            continue
        if reference_paper is not None:
            papers.append(reference_paper)

    return ConnectorSearchResult(
        papers=papers[:limit],
        error_message=";".join(errors) or None,
        warnings=warnings,
        latency_seconds=time.perf_counter() - start,
        diagnostics=_with_total_latency(diagnostics, start),
        reference_batch_status=(
            "success"
            if not missing_ids and not errors
            else "partial_success"
            if papers
            else "failed"
        ),
        missing_reference_ids=missing_ids,
        reference_batch_count=batch_count,
        supplemental_request_count=supplemental_request_count,
    )


def _fetch_seed_work_detailed(
    paper: Paper,
    *,
    max_retries: int,
    retry_sleep: Callable[[float], None] | None,
) -> tuple[
    dict[str, Any] | None,
    str | None,
    list[str],
    ConnectorDiagnostics,
]:
    openalex_id = _normalize_openalex_id(paper.identifiers.openalex_id)
    if openalex_id:
        url = _url_with_mailto(
            f"{OPENALEX_WORKS_URL}/{quote(openalex_id, safe='')}",
            {},
        )
        return _request_json_detailed(
            url,
            context="OpenAlex seed work",
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

    doi = _normalize_doi(paper.identifiers.doi)
    if doi:
        payload, error, warnings, diagnostics = _request_json_detailed(
            _url_with_mailto(
                OPENALEX_WORKS_URL,
                {"filter": f"doi:{doi}", "per-page": "1"},
            ),
            context="OpenAlex DOI seed work",
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        results = (payload or {}).get("results", [])
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0], None, warnings, diagnostics
        if error is None:
            error = "OpenAlex DOI seed work not found"
            warnings.append(error)
            diagnostics = diagnostics.model_copy(
                update={"error_count": diagnostics.error_count + 1}
            )
        return None, error, warnings, diagnostics

    return None, None, [], ConnectorDiagnostics()


def _fetch_works_by_openalex_ids_detailed(
    openalex_ids: list[str],
    *,
    max_retries: int,
    retry_sleep: Callable[[float], None] | None,
) -> tuple[dict[str, Any] | None, str | None, list[str], ConnectorDiagnostics]:
    return _request_json_detailed(
        _url_with_mailto(
            OPENALEX_WORKS_URL,
            {
                "filter": "openalex_id:" + "|".join(openalex_ids),
                "per-page": str(len(openalex_ids)),
            },
        ),
        context="OpenAlex reference batch",
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )


def _referenced_work_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    reference_ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _normalize_openalex_id(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        reference_ids.append(normalized)
        seen.add(key)
    return reference_ids


def _request_json(url: str) -> dict[str, Any] | None:
    payload, _, _, _ = _request_json_detailed(url, context="OpenAlex request")
    return payload


def _request_json_detailed(
    url: str,
    *,
    context: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] | None = None,
) -> tuple[
    dict[str, Any] | None,
    str | None,
    list[str],
    ConnectorDiagnostics,
]:
    started_at = time.perf_counter()
    request = Request(url, headers=_openalex_headers())
    warnings: list[str] = []
    attempts = max(0, max_retries) + 1
    sleep = retry_sleep or time.sleep
    payload: Any = None
    request_count = 0
    retry_count = 0

    for attempt in range(attempts):
        request_count += 1
        retry_count += int(attempt > 0)
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", getattr(response, "code", 200))
                if status < 200 or status >= 300:
                    message = f"{context} returned non-2xx status: {status}"
                    if _should_retry_status(status) and attempt < attempts - 1:
                        _record_retry_warning(
                            warnings,
                            context=context,
                            attempt=attempt,
                            attempts=attempts,
                            reason=message,
                        )
                        sleep(_retry_backoff_seconds(attempt))
                        continue
                    logger.warning(message)
                    return None, message, warnings + [message], ConnectorDiagnostics(
                        request_count=request_count,
                        retry_count=retry_count,
                        error_count=1,
                        latency_seconds=time.perf_counter() - started_at,
                    )
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if _should_retry_status(exc.code) and attempt < attempts - 1:
                _record_retry_warning(
                    warnings,
                    context=context,
                    attempt=attempt,
                    attempts=attempts,
                    reason=str(exc),
                )
                sleep(_retry_backoff_seconds(attempt))
                continue
            message = f"{context} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                latency_seconds=time.perf_counter() - started_at,
            )
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < attempts - 1:
                _record_retry_warning(
                    warnings,
                    context=context,
                    attempt=attempt,
                    attempts=attempts,
                    reason=str(exc),
                )
                sleep(_retry_backoff_seconds(attempt))
                continue
            message = f"{context} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                latency_seconds=time.perf_counter() - started_at,
            )
        except json.JSONDecodeError as exc:
            message = f"{context} failed: {exc}"
            logger.warning(message)
            return None, message, warnings + [message], ConnectorDiagnostics(
                request_count=request_count,
                retry_count=retry_count,
                error_count=1,
                latency_seconds=time.perf_counter() - started_at,
            )
        break

    if not isinstance(payload, dict):
        message = f"{context} returned non-object JSON payload"
        logger.warning(message)
        return None, message, warnings + [message], ConnectorDiagnostics(
            request_count=request_count,
            retry_count=retry_count,
            error_count=1,
            latency_seconds=time.perf_counter() - started_at,
        )
    return payload, None, warnings, ConnectorDiagnostics(
        request_count=request_count,
        retry_count=retry_count,
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
    context: str,
    attempt: int,
    attempts: int,
    reason: str,
) -> None:
    message = (
        f"{context} transient error on attempt {attempt + 1}/{attempts}: "
        f"{reason}; retried"
    )
    logger.warning(message)
    warnings.append(message)


def _retry_backoff_seconds(attempt: int) -> float:
    return DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1)


def _should_retry_status(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status <= 599)


def _url_with_mailto(base_url: str, params: dict[str, str]) -> str:
    query_params = dict(params)
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        query_params["mailto"] = mailto
    if not query_params:
        return base_url
    return f"{base_url}?{urlencode(query_params)}"


def _openalex_headers() -> dict[str, str]:
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        return {"User-Agent": f"ScholarNavigator (mailto: {mailto})"}
    return {"User-Agent": "ScholarNavigator (mailto: unavailable)"}


def _parse_work(work: dict[str, Any]) -> Paper | None:
    title = _clean_text(work.get("display_name")) or "Untitled OpenAlex Work"
    authors = _parse_authors(work.get("authorships"))
    year = _as_int(work.get("publication_year"))
    primary_location = _as_dict(work.get("primary_location"))
    source = _as_dict(primary_location.get("source"))
    venue = _clean_text(source.get("display_name"))
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    ids = _as_dict(work.get("ids"))

    doi = _normalize_doi(ids.get("doi") or work.get("doi"))
    openalex_id = _normalize_openalex_id(ids.get("openalex") or work.get("id"))
    arxiv_id = _normalize_arxiv_id(ids.get("arxiv"))
    pubmed_id = _normalize_pubmed_id(ids.get("pmid"))

    landing_page = _clean_text(primary_location.get("landing_page_url"))
    if not landing_page:
        landing_page = _clean_text(ids.get("openalex") or work.get("id"))
    pdf_url = _clean_text(primary_location.get("pdf_url"))

    return Paper(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            pubmed_id=pubmed_id,
        ),
        urls=PaperUrls(
            landing_page=landing_page,
            pdf=pdf_url,
        ),
        sources=["openalex"],
        citation_count=max(_as_int(work.get("cited_by_count")) or 0, 0),
    )


def _parse_authors(authorships: Any) -> list[str]:
    if not isinstance(authorships, list):
        return []
    authors: list[str] = []
    for authorship in authorships:
        author = _as_dict(_as_dict(authorship).get("author"))
        name = _clean_text(author.get("display_name"))
        if name:
            authors.append(name)
    return authors


def _reconstruct_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""

    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            int_position = _as_int(position)
            if int_position is not None:
                positioned_words.append((int_position, word))

    if not positioned_words:
        return ""

    return " ".join(word for _, word in sorted(positioned_words))


def _normalize_doi(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    lower = text.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lower.startswith(prefix):
            return text[len(prefix) :]
    return text


def _normalize_openalex_id(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme and parsed.path:
        return parsed.path.rstrip("/").split("/")[-1] or text
    return text


def _normalize_pubmed_id(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme and parsed.path:
        return parsed.path.rstrip("/").split("/")[-1] or text
    return text.removeprefix("pmid:")


def _normalize_arxiv_id(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme and parsed.path:
        text = parsed.path.rstrip("/").split("/")[-1] or text
    return text.removeprefix("arXiv:").removeprefix("arxiv:")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
