from __future__ import annotations

import pytest

from scholar_agent.core.full_text_evidence import (
    FullTextLicenseError,
    build_paragraph_evidence,
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
