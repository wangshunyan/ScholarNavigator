from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import unquote

import pytest

from scholar_agent.connectors.openalex import (
    fetch_openalex_references,
    fetch_openalex_references_detailed,
    search_openalex,
    search_openalex_detailed,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


class MockResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch) -> None:
    monkeypatch.setattr("scholar_agent.connectors.openalex.time.sleep", lambda _: None)


def test_search_openalex_parses_normal_response(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return MockResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Test OpenAlex Paper",
                        "publication_year": 2024,
                        "cited_by_count": 17,
                        "doi": "https://doi.org/10.1234/test",
                        "ids": {
                            "openalex": "https://openalex.org/W123",
                            "doi": "https://doi.org/10.1234/test",
                            "pmid": "https://pubmed.ncbi.nlm.nih.gov/987654/",
                        },
                        "authorships": [
                            {"author": {"display_name": "Alice Chen"}},
                            {"author": {"display_name": "Bob Smith"}},
                        ],
                        "primary_location": {
                            "landing_page_url": "https://example.org/paper",
                            "pdf_url": "https://example.org/paper.pdf",
                            "source": {"display_name": "ACL"},
                        },
                        "abstract_inverted_index": {
                            "A": [0],
                            "mock": [1],
                            "abstract": [2],
                        },
                    }
                ]
            }
        )

    monkeypatch.setenv("OPENALEX_MAILTO", "team@example.org")
    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    papers = search_openalex("llm reranking", limit=5)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Test OpenAlex Paper"
    assert paper.authors == ["Alice Chen", "Bob Smith"]
    assert paper.year == 2024
    assert paper.venue == "ACL"
    assert paper.abstract == "A mock abstract"
    assert paper.identifiers.doi == "10.1234/test"
    assert paper.identifiers.openalex_id == "W123"
    assert paper.identifiers.pubmed_id == "987654"
    assert paper.urls.landing_page == "https://example.org/paper"
    assert paper.urls.pdf == "https://example.org/paper.pdf"
    assert paper.sources == ["openalex"]
    assert paper.citation_count == 17
    assert "mailto=team%40example.org" in captured["url"]
    assert captured["timeout"] == 10.0


def test_search_openalex_detailed_normal_response_has_no_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Detailed OpenAlex Paper",
                    }
                ]
            }
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking", limit=5)

    assert len(result.papers) == 1
    assert result.papers[0].title == "Detailed OpenAlex Paper"
    assert result.error_message is None
    assert result.warnings == []
    assert result.diagnostics.request_count == 1
    assert result.diagnostics.retry_count == 0
    assert result.diagnostics.error_count == 0


def test_search_openalex_detailed_retries_transient_error_then_succeeds(
    monkeypatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                hdrs=None,
                fp=None,
            )
        return MockResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/WRETRY",
                        "display_name": "Recovered OpenAlex Paper",
                    }
                ]
            }
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed(
        "llm reranking",
        retry_sleep=lambda seconds: sleeps.append(seconds),
    )

    assert calls == 2
    assert sleeps == [0.5]
    assert result.error_message is None
    assert [paper.title for paper in result.papers] == ["Recovered OpenAlex Paper"]
    assert any("retried" in warning for warning in result.warnings)
    assert any("HTTP Error 503" in warning for warning in result.warnings)
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 1
    assert result.diagnostics.error_count == 0


def test_search_openalex_detailed_retry_failure_keeps_diagnostics(
    monkeypatch,
) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking")

    assert calls == 2
    assert result.papers == []
    assert result.error_message is not None
    assert "HTTP Error 503" in result.error_message
    assert result.error_message in result.warnings
    assert any("retried" in warning for warning in result.warnings)
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 1
    assert result.diagnostics.error_count == 1


def test_search_openalex_safe_original_is_not_dropped_as_stopwords(monkeypatch) -> None:
    calls = 0
    requested_url = ""

    def fake_urlopen(request, timeout):
        nonlocal calls, requested_url
        calls += 1
        requested_url = request.full_url
        return MockResponse({"results": []})

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("Could you list some papers???")

    assert calls == 1
    assert "Could+you+list+some+papers" in requested_url
    assert result.papers == []
    assert result.diagnostics.request_count == 1
    assert result.error_message is None


def test_search_openalex_400_is_not_retried(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 400, "Bad Request", hdrs=None, fp=None)

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("query with unsafe? punctuation", max_retries=3)

    assert calls == 1
    assert result.diagnostics.request_count == 1
    assert result.diagnostics.retry_count == 0
    assert "HTTP Error 400" in (result.error_message or "")


@pytest.mark.parametrize("status", [429, 503])
def test_search_openalex_transient_statuses_still_retry(monkeypatch, status: int) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(request.full_url, status, "transient", hdrs=None, fp=None)
        return MockResponse({"results": []})

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("stable retrieval query")

    assert calls == 2
    assert result.error_message is None
    assert result.diagnostics.retry_count == 1


def test_search_openalex_exception_returns_empty(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("timeout")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    assert search_openalex("llm reranking") == []


def test_search_openalex_detailed_url_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("timeout")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message is not None
    assert "timeout" in result.error_message
    assert result.error_message in result.warnings


def test_search_openalex_detailed_timeout_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise TimeoutError("request timed out")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message is not None
    assert "request timed out" in result.error_message
    assert result.error_message in result.warnings


def test_search_openalex_detailed_http_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message is not None
    assert "HTTP Error 503" in result.error_message
    assert result.error_message in result.warnings


def test_search_openalex_detailed_non_2xx_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse({}, status=503)

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    result = search_openalex_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message == "OpenAlex search returned non-2xx status: 503"
    assert result.error_message in result.warnings
    assert any("retried" in warning for warning in result.warnings)


def test_search_openalex_missing_fields_returns_available_result(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse({"results": [{"id": "https://openalex.org/W999"}]})

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)

    papers = search_openalex("minimal")

    assert len(papers) == 1
    assert papers[0].title == "Untitled OpenAlex Work"
    assert papers[0].authors == []
    assert papers[0].abstract == ""
    assert papers[0].identifiers.openalex_id == "W999"
    assert papers[0].sources == ["openalex"]


def test_fetch_openalex_references_with_openalex_id_seed(monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/WREF1",
                        "https://openalex.org/WREF2",
                    ],
                }
            )
        decoded = unquote(request.full_url)
        if "filter=openalex_id:WREF1|WREF2" in decoded:
            return MockResponse(
                {
                    "results": [
                        _openalex_work("WREF2", "Reference Two"),
                        _openalex_work("WREF1", "Reference One"),
                    ]
                }
            )
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(
        title="Seed",
        identifiers=PaperIdentifiers(openalex_id="WSEED"),
    )

    result = fetch_openalex_references_detailed(seed, limit=20)
    references = result.papers

    assert [paper.title for paper in references] == ["Reference One", "Reference Two"]
    assert [paper.identifiers.openalex_id for paper in references] == ["WREF1", "WREF2"]
    assert all(paper.sources == ["openalex"] for paper in references)
    assert requested_urls[0].endswith("/WSEED")
    assert len(requested_urls) == 2
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 0


def test_fetch_openalex_references_with_doi_seed(monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        decoded = unquote(request.full_url)
        if "filter=doi:10.555/seed" in decoded:
            return MockResponse(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/WSEED",
                            "referenced_works": ["https://openalex.org/WREFDOI"],
                        }
                    ]
                }
            )
        if "filter=openalex_id:WREFDOI" in decoded:
            return MockResponse(
                {"results": [_openalex_work("WREFDOI", "DOI Reference")]}
            )
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(
        title="Seed",
        identifiers=PaperIdentifiers(doi="https://doi.org/10.555/seed"),
    )

    result = fetch_openalex_references_detailed(seed)
    references = result.papers

    assert len(references) == 1
    assert references[0].title == "DOI Reference"
    assert references[0].identifiers.openalex_id == "WREFDOI"
    assert "filter=doi:10.555/seed" in unquote(requested_urls[0])
    assert result.diagnostics.request_count == 2


def test_fetch_openalex_references_limit_is_applied(monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/WREF1",
                        "https://openalex.org/WREF2",
                    ],
                }
            )
        decoded = unquote(request.full_url)
        if "filter=openalex_id:WREF1" in decoded:
            assert "WREF2" not in decoded
            return MockResponse(
                {"results": [_openalex_work("WREF1", "Reference One")]}
            )
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(
        title="Seed",
        identifiers=PaperIdentifiers(openalex_id="WSEED"),
    )

    references = fetch_openalex_references(seed, limit=1)

    assert len(references) == 1
    assert references[0].title == "Reference One"
    assert not any("WREF2" in unquote(url) for url in requested_urls[1:])


def test_fetch_openalex_references_without_supported_identifier_returns_empty(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        raise AssertionError("OpenAlex should not be called without an identifier")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(title="Seed", identifiers=PaperIdentifiers())

    assert fetch_openalex_references(seed) == []


def test_fetch_openalex_references_timeout_and_non_2xx_return_empty(monkeypatch) -> None:
    seed = Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))

    def timeout_urlopen(request, timeout):
        raise URLError("timeout")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", timeout_urlopen)
    assert fetch_openalex_references(seed) == []

    def non_2xx_urlopen(request, timeout):
        return MockResponse({}, status=503)

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", non_2xx_urlopen)
    assert fetch_openalex_references(seed) == []


def test_fetch_openalex_references_missing_fields_are_tolerated(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/WMINIMAL",
                        None,
                    ],
                }
            )
        if "filter=openalex_id:WMINIMAL" in unquote(request.full_url):
            return MockResponse(
                {"results": [{"id": "https://openalex.org/WMINIMAL"}]}
            )
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))

    references = fetch_openalex_references(seed)

    assert len(references) == 1
    assert references[0].title == "Untitled OpenAlex Work"
    assert references[0].authors == []
    assert references[0].abstract == ""
    assert references[0].identifiers.openalex_id == "WMINIMAL"
    assert references[0].sources == ["openalex"]


def test_fetch_openalex_references_detailed_counts_batch_retry(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": ["https://openalex.org/WREF1"],
                }
            )
        if calls == 2:
            raise HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                hdrs=None,
                fp=None,
            )
        return MockResponse(
            {"results": [_openalex_work("WREF1", "Reference One")]}
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    seed = Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))

    result = fetch_openalex_references_detailed(seed)

    assert [paper.title for paper in result.papers] == ["Reference One"]
    assert result.diagnostics.request_count == 3
    assert result.diagnostics.retry_count == 1
    assert result.diagnostics.error_count == 0


def test_fetch_openalex_references_keeps_partial_batch_and_supplements_missing(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        del timeout
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/W1",
                        "https://openalex.org/W2",
                        "https://openalex.org/W3",
                    ],
                }
            )
        decoded = unquote(request.full_url)
        if "filter=openalex_id:W1|W2|W3" in decoded:
            return MockResponse(
                {
                    "results": [
                        _openalex_work("W3", "Third"),
                        _openalex_work("W1", "First"),
                    ]
                }
            )
        if "filter=openalex_id:W2" in decoded:
            return MockResponse({"results": [_openalex_work("W2", "Second")]})
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    result = fetch_openalex_references_detailed(
        Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))
    )

    assert [paper.title for paper in result.papers] == ["First", "Second", "Third"]
    assert result.reference_batch_status == "success"
    assert result.missing_reference_ids == []
    assert result.reference_batch_count == 1
    assert result.supplemental_request_count == 1
    assert result.diagnostics.request_count == 3


def test_fetch_openalex_references_marks_partial_success_and_terminal_missing(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        del timeout
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/W1",
                        "https://openalex.org/W2",
                    ],
                }
            )
        decoded = unquote(request.full_url)
        if "filter=openalex_id:W1|W2" in decoded:
            return MockResponse({"results": [_openalex_work("W2", "Second")]})
        if "filter=openalex_id:W1" in decoded:
            return MockResponse({"results": []})
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    result = fetch_openalex_references_detailed(
        Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))
    )

    assert [paper.title for paper in result.papers] == ["Second"]
    assert result.reference_batch_status == "partial_success"
    assert result.missing_reference_ids == ["W1"]
    assert "missing work id:W1" in result.error_message
    assert result.supplemental_request_count == 1


def test_fetch_openalex_references_all_missing_is_terminal_per_id(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del timeout
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/W1",
                        "https://openalex.org/W2",
                    ],
                }
            )
        return MockResponse({"results": []})

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    result = fetch_openalex_references_detailed(
        Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))
    )

    assert result.papers == []
    assert result.reference_batch_status == "failed"
    assert result.missing_reference_ids == ["W1", "W2"]
    assert result.diagnostics.error_count == 2
    assert result.supplemental_request_count == 2
    assert "missing work id:W1" in result.error_message
    assert "missing work id:W2" in result.error_message


def test_fetch_openalex_references_deduplicates_reference_ids_and_records(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del timeout
        if request.full_url.endswith("/WSEED"):
            return MockResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "referenced_works": [
                        "https://openalex.org/W1",
                        "https://openalex.org/W1",
                    ],
                }
            )
        return MockResponse(
            {
                "results": [
                    _openalex_work("W1", "First"),
                    _openalex_work("W1", "First duplicate"),
                ]
            }
        )

    monkeypatch.setattr("scholar_agent.connectors.openalex.urlopen", fake_urlopen)
    result = fetch_openalex_references_detailed(
        Paper(title="Seed", identifiers=PaperIdentifiers(openalex_id="WSEED"))
    )

    assert [paper.title for paper in result.papers] == ["First"]
    assert result.reference_batch_status == "success"
    assert result.reference_batch_count == 1
    assert result.supplemental_request_count == 0


def _openalex_work(openalex_id: str, title: str) -> dict:
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "display_name": title,
        "publication_year": 2023,
        "cited_by_count": 5,
        "ids": {
            "openalex": f"https://openalex.org/{openalex_id}",
            "doi": f"https://doi.org/10.123/{openalex_id.casefold()}",
        },
        "authorships": [{"author": {"display_name": "Reference Author"}}],
        "primary_location": {
            "landing_page_url": f"https://example.org/{openalex_id}",
            "source": {"display_name": "OpenAlex Venue"},
        },
        "abstract_inverted_index": {
            "Reference": [0],
            "abstract": [1],
        },
    }
