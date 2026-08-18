from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

import scholar_agent.connectors.arxiv as arxiv_connector
from scholar_agent.connectors.arxiv import search_arxiv, search_arxiv_detailed


class MockResponse:
    def __init__(self, payload: str, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload.encode("utf-8")


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2407.18940v2</id>
    <updated>2024-07-30T00:00:00Z</updated>
    <published>2024-07-28T00:00:00Z</published>
    <title> LitSearch: A Retrieval Benchmark </title>
    <summary>
      A benchmark for scientific literature search.
    </summary>
    <author><name>Alice Chen</name></author>
    <author><name>Bob Smith</name></author>
    <arxiv:doi>10.48550/arXiv.2407.18940</arxiv:doi>
    <arxiv:journal_ref>EMNLP 2024</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2407.18940v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2407.18940v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch) -> None:
    arxiv_connector._reset_arxiv_throttle_for_tests()
    monkeypatch.setattr("scholar_agent.connectors.arxiv.time.sleep", lambda _: None)
    monkeypatch.delenv("SCHOLAR_AGENT_ARXIV_MIN_INTERVAL_SECONDS", raising=False)
    yield
    arxiv_connector._reset_arxiv_throttle_for_tests()


def test_search_arxiv_parses_normal_response(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return MockResponse(ARXIV_FEED)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    papers = search_arxiv("scientific literature search", limit=3)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "LitSearch: A Retrieval Benchmark"
    assert paper.authors == ["Alice Chen", "Bob Smith"]
    assert paper.year == 2024
    assert paper.venue == "EMNLP 2024"
    assert paper.abstract == "A benchmark for scientific literature search."
    assert paper.identifiers.doi == "10.48550/arXiv.2407.18940"
    assert paper.identifiers.arxiv_id == "2407.18940"
    assert paper.urls.landing_page == "http://arxiv.org/abs/2407.18940v2"
    assert paper.urls.pdf == "http://arxiv.org/pdf/2407.18940v2"
    assert paper.sources == ["arxiv"]
    assert paper.citation_count == 0
    assert "max_results=3" in captured["url"]
    assert captured["timeout"] == 10.0


def test_search_arxiv_detailed_normal_response_has_no_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse(ARXIV_FEED)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("scientific literature search", limit=3)

    assert len(result.papers) == 1
    assert result.error_message is None
    assert result.warnings == []
    assert result.diagnostics.request_count == 1
    assert result.diagnostics.retry_count == 0


def test_search_arxiv_preserves_pre_adapted_expression(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return MockResponse(ARXIV_FEED)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)
    expression = '(ti:"graph retrieval" OR abs:"graph retrieval")'

    search_arxiv_detailed(expression, limit=2)

    params = parse_qs(urlparse(captured["url"]).query)
    assert params["search_query"] == [expression]
    assert "all:" not in params["search_query"][0]


def test_arxiv_throttle_uses_official_three_second_default(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(
        "scholar_agent.connectors.arxiv.urlopen",
        lambda request, timeout: MockResponse(ARXIV_FEED),
    )

    first = search_arxiv_detailed(
        "first query",
        max_retries=0,
        throttle_sleep=fake_sleep,
        monotonic=lambda: clock[0],
    )
    second = search_arxiv_detailed(
        "second query",
        max_retries=0,
        throttle_sleep=fake_sleep,
        monotonic=lambda: clock[0],
    )

    assert first.error_message is None
    assert second.error_message is None
    assert sleeps == [3.0]
    assert second.diagnostics.rate_limit_wait_seconds == pytest.approx(3.0)


def test_search_arxiv_detailed_retries_transient_error_then_succeeds(
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
                429,
                "Too Many Requests",
                hdrs=None,
                fp=None,
            )
        return MockResponse(ARXIV_FEED)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed(
        "scientific literature search",
        retry_sleep=lambda seconds: sleeps.append(seconds),
    )

    assert calls == 2
    assert sleeps == [0.5]
    assert result.error_message is None
    assert len(result.papers) == 1
    assert any("retried" in warning for warning in result.warnings)
    assert any("HTTP Error 429" in warning for warning in result.warnings)
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 1
    assert result.diagnostics.error_count == 0


def test_search_arxiv_detailed_retry_failure_keeps_diagnostics(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise TimeoutError("read timed out")

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("llm reranking")

    assert calls == 2
    assert result.papers == []
    assert result.error_message is not None
    assert "read timed out" in result.error_message
    assert result.error_message in result.warnings
    assert any("retried" in warning for warning in result.warnings)
    assert result.diagnostics.request_count == 2
    assert result.diagnostics.retry_count == 1
    assert result.diagnostics.error_count == 1


def test_search_arxiv_exception_returns_empty(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("timeout")

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    assert search_arxiv("llm reranking") == []


def test_search_arxiv_detailed_url_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("timeout")

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message is not None
    assert "timeout" in result.error_message
    assert result.error_message in result.warnings


def test_search_arxiv_detailed_http_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message is not None
    assert "HTTP Error 503" in result.error_message
    assert result.error_message in result.warnings


def test_search_arxiv_detailed_non_2xx_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse("", status=503)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("llm reranking")

    assert result.papers == []
    assert result.error_message == "arXiv search returned non-2xx status: 503"
    assert result.error_message in result.warnings
    assert any("retried" in warning for warning in result.warnings)


def test_search_arxiv_missing_fields_returns_available_result(monkeypatch) -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001</id>
  </entry>
</feed>
"""

    def fake_urlopen(request, timeout):
        return MockResponse(feed)

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    papers = search_arxiv("minimal")

    assert len(papers) == 1
    assert papers[0].title == "Untitled arXiv Paper"
    assert papers[0].authors == []
    assert papers[0].abstract == ""
    assert papers[0].venue == "arXiv"
    assert papers[0].identifiers.arxiv_id == "2501.00001"
    assert papers[0].sources == ["arxiv"]


def test_search_arxiv_xml_parse_error_returns_empty(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse("<feed><broken></feed>")

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    assert search_arxiv("bad xml") == []


def test_search_arxiv_detailed_xml_parse_error_returns_error_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return MockResponse("<feed><broken></feed>")

    monkeypatch.setattr("scholar_agent.connectors.arxiv.urlopen", fake_urlopen)

    result = search_arxiv_detailed("bad xml")

    assert result.papers == []
    assert result.error_message is not None
    assert "mismatched tag" in result.error_message
    assert result.error_message in result.warnings
