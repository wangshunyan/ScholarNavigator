from __future__ import annotations

import pytest

from scholar_agent.core.full_text_evidence import (
    FullTextLicenseError,
    build_paragraph_evidence,
    fetch_open_full_text,
)


TEXT = "First paragraph.\r\ncontinued.\r\n\r\nSecond paragraph.\r\n"


def test_paragraph_evidence_is_stable_and_locatable() -> None:
    first = build_paragraph_evidence(
        TEXT,
        source_url="https://example.test/paper.txt",
        license_id="CC-BY-4.0",
        license_verified=True,
    )
    second = build_paragraph_evidence(
        TEXT.replace("\r\n", "\n"),
        source_url="https://example.test/paper.txt",
        license_id="CC-BY-4.0",
        license_verified=True,
    )

    assert first.source.content_sha256 == second.source.content_sha256
    assert [item.evidence_id for item in first.paragraphs] == [
        item.evidence_id for item in second.paragraphs
    ]
    assert [item.text for item in first.paragraphs] == [
        "First paragraph.\ncontinued.",
        "Second paragraph.",
    ]
    for item in first.paragraphs:
        assert first.paragraphs[item.paragraph_index].text == item.text


def test_unverified_license_fails_closed() -> None:
    with pytest.raises(FullTextLicenseError, match="license_unverified"):
        build_paragraph_evidence(
            "licensed-looking text",
            source_url="https://example.test/paper.txt",
            license_id="CC-BY-4.0",
            license_verified=False,
        )


def test_empty_full_text_does_not_create_evidence() -> None:
    with pytest.raises(ValueError, match="full_text_empty"):
        build_paragraph_evidence(
            " \r\n \n",
            source_url="https://example.test/paper.txt",
            license_id="CC-BY-4.0",
            license_verified=True,
        )


class _Response:
    def __init__(self, body: bytes, content_type: str, *, status: int = 200, length: str | None = None) -> None:
        self.body = body
        self.status = status
        self.headers = {"Content-Type": content_type}
        if length is not None:
            self.headers["Content-Length"] = length

    def __enter__(self):  # noqa: ANN201
        return self

    def __exit__(self, *args):  # noqa: ANN002, ANN201
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _opener(response: _Response):
    def open_request(request, *, timeout: float):  # noqa: ANN001
        assert request.full_url == "https://open.example.test/paper"
        assert timeout == 2
        return response

    return open_request


def test_allowlisted_html_fetch_builds_visible_paragraph_evidence() -> None:
    result = fetch_open_full_text(
        source_url="https://open.example.test/paper",
        license_id="CC-BY-4.0",
        license_verified=True,
        allowed_hosts={"open.example.test"},
        timeout_seconds=2,
        opener=_opener(
            _Response(
                b"<article><p>First evidence.</p><script>ignore()</script><p>Second evidence.</p></article>",
                "text/html; charset=utf-8",
            )
        ),
    )

    assert result.status == "succeeded"
    assert result.document is not None
    assert [item.text for item in result.document.paragraphs] == [
        "First evidence.",
        "Second evidence.",
    ]


def test_fetch_requires_verified_license_and_allowlisted_https_host() -> None:
    unverified = fetch_open_full_text(
        source_url="https://open.example.test/paper",
        license_id="CC-BY-4.0",
        license_verified=False,
        allowed_hosts={"open.example.test"},
    )
    blocked = fetch_open_full_text(
        source_url="http://other.example.test/paper",
        license_id="CC-BY-4.0",
        license_verified=True,
        allowed_hosts={"open.example.test"},
    )

    assert unverified.status == "license_unverified"
    assert blocked.status == "url_not_allowed"


def test_fetch_limits_unsupported_types_and_pdf_fail_closed() -> None:
    common = {
        "source_url": "https://open.example.test/paper",
        "license_id": "CC-BY-4.0",
        "license_verified": True,
        "allowed_hosts": {"open.example.test"},
        "timeout_seconds": 2,
    }
    too_large = fetch_open_full_text(
        **common,
        max_bytes=4,
        opener=_opener(_Response(b"ignored", "text/plain", length="999")),
    )
    unsupported = fetch_open_full_text(
        **common,
        opener=_opener(_Response(b"binary", "application/octet-stream")),
    )
    pdf = fetch_open_full_text(
        **common,
        opener=_opener(_Response(b"%PDF", "application/pdf")),
    )

    assert too_large.status == "response_too_large"
    assert unsupported.status == "unsupported_media_type"
    assert pdf.status == "parser_unavailable"
