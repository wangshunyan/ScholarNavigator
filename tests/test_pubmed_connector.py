from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

import scholar_agent.connectors.pubmed as pubmed_connector
from scholar_agent.connectors.pubmed import (
    search_pubmed,
    search_pubmed_detailed,
)


PUBMED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>PubMed Test Paper</ArticleTitle>
        <Abstract>
          <AbstractText>This paper studies biomedical retrieval.</AbstractText>
          <AbstractText>It includes an evidence search benchmark.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Journal of Test Medicine</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
        <AuthorList>
          <Author><ForeName>Alice</ForeName><LastName>Chen</LastName></Author>
          <Author><ForeName>Bob</ForeName><LastName>Smith</LastName></Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.1234/pubmed</ELocationID>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345</ArticleId>
        <ArticleId IdType="doi">10.1234/pubmed</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class MockResponse:
    def __init__(self, payload: bytes | dict, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self) -> "MockResponse":
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture(autouse=True)
def reset_pubmed(monkeypatch: pytest.MonkeyPatch):
    pubmed_connector._reset_pubmed_throttle_for_tests()
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("PUBMED_API_KEY", raising=False)
    monkeypatch.setenv("SCHOLAR_AGENT_PUBMED_MIN_INTERVAL_SECONDS", "0")
    yield
    pubmed_connector._reset_pubmed_throttle_for_tests()


def test_search_pubmed_esearch_and_efetch_parse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_urls: list[str] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        captured_urls.append(request.full_url)
        parsed = urlparse(request.full_url)
        if "esearch.fcgi" in parsed.path:
            return MockResponse({"esearchresult": {"idlist": ["12345"]}})
        if "efetch.fcgi" in parsed.path:
            return MockResponse(PUBMED_XML)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setenv("NCBI_API_KEY", "ncbi-secret")
    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    result = search_pubmed_detailed("biomedical retrieval", limit=3)

    assert result.error_message is None
    assert result.warnings == []
    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.title == "PubMed Test Paper"
    assert paper.authors == ["Alice Chen", "Bob Smith"]
    assert paper.year == 2024
    assert paper.venue == "Journal of Test Medicine"
    assert paper.abstract == (
        "This paper studies biomedical retrieval. "
        "It includes an evidence search benchmark."
    )
    assert paper.identifiers.doi == "10.1234/pubmed"
    assert paper.identifiers.pubmed_id == "12345"
    assert paper.urls.landing_page == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert paper.sources == ["pubmed"]
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 0
    assert result.diagnostics.error_count == 0

    esearch_query = parse_qs(urlparse(captured_urls[0]).query)
    efetch_query = parse_qs(urlparse(captured_urls[1]).query)
    assert esearch_query["db"] == ["pubmed"]
    assert esearch_query["term"] == ["biomedical retrieval"]
    assert esearch_query["retmax"] == ["3"]
    assert esearch_query["api_key"] == ["ncbi-secret"]
    assert efetch_query["id"] == ["12345"]
    assert efetch_query["api_key"] == ["ncbi-secret"]


def test_search_pubmed_wrapper_returns_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        parsed = urlparse(request.full_url)
        if "esearch.fcgi" in parsed.path:
            return MockResponse({"esearchresult": {"idlist": ["12345"]}})
        return MockResponse(PUBMED_XML)

    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    papers = search_pubmed("biomedical retrieval", limit=1)

    assert len(papers) == 1
    assert papers[0].identifiers.pubmed_id == "12345"


def test_search_pubmed_empty_result_skips_efetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        calls.append(request.full_url)
        return MockResponse({"esearchresult": {"idlist": []}})

    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    result = search_pubmed_detailed("no matches", limit=5)

    assert result.error_message is None
    assert result.papers == []
    assert len(calls) == 1
    assert "esearch.fcgi" in urlparse(calls[0]).path
    assert result.diagnostics.request_count == 1
    assert result.diagnostics.error_count == 0


def test_search_pubmed_http_error_returns_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        raise HTTPError(
            url=request.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    result = search_pubmed_detailed("biomedical retrieval")

    assert result.papers == []
    assert result.error_message is not None
    assert "PubMed esearch failed: HTTP Error 503" in result.error_message
    assert result.warnings == [result.error_message]
    assert result.diagnostics.request_count == 1
    assert result.diagnostics.error_count == 1


def test_search_pubmed_efetch_failure_keeps_two_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request, timeout):  # noqa: ANN001
        nonlocal calls
        calls += 1
        if calls == 1:
            return MockResponse({"esearchresult": {"idlist": ["12345"]}})
        raise HTTPError(
            url=request.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    result = search_pubmed_detailed("biomedical retrieval")

    assert result.papers == []
    assert "PubMed efetch failed" in str(result.error_message)
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 0
    assert result.diagnostics.error_count == 1


def test_search_pubmed_throttles_consecutive_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter([10.0, 10.1, 10.34])
    slept: list[float] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        parsed = urlparse(request.full_url)
        if "esearch.fcgi" in parsed.path:
            return MockResponse({"esearchresult": {"idlist": ["12345"]}})
        return MockResponse(PUBMED_XML)

    monkeypatch.delenv("SCHOLAR_AGENT_PUBMED_MIN_INTERVAL_SECONDS", raising=False)
    monkeypatch.setattr("scholar_agent.connectors.pubmed.urlopen", fake_urlopen)

    result = search_pubmed_detailed(
        "biomedical retrieval",
        throttle_sleep=slept.append,
        monotonic=lambda: next(monotonic_values),
    )

    assert result.error_message is None
    assert slept == [pytest.approx(0.24)]
    assert result.diagnostics.rate_limit_wait_seconds == pytest.approx(0.24)
