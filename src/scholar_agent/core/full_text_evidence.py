"""Deterministic, license-gated paragraph evidence for open full text."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FullTextEvidenceError(ValueError):
    """Base error for closed full-text evidence handling."""


class FullTextLicenseError(FullTextEvidenceError):
    """Raised when the caller has not established a usable license basis."""


class FullTextSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_verified: bool = False
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ParagraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^paragraph:[0-9a-f]{64}:[0-9]+$")
    paragraph_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class FullTextEvidenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["full-text-evidence-v1"] = "full-text-evidence-v1"
    source: FullTextSource
    paragraphs: list[ParagraphEvidence] = Field(default_factory=list)


def build_paragraph_evidence(
    text: str,
    *,
    source_url: str,
    license_id: str,
    license_verified: bool,
) -> FullTextEvidenceDocument:
    """Normalize licensed plain text and create stable paragraph locations.

    This function intentionally accepts text that has already been obtained by a
    caller. Network retrieval, HTML/PDF parsing and license discovery belong to
    a later task; an unverified license fails closed here.
    """

    if not license_verified or not license_id.strip():
        raise FullTextLicenseError("full_text_license_unverified")
    normalized = _normalize_text(text)
    if not normalized:
        raise FullTextEvidenceError("full_text_empty")

    content_hash = _sha256(normalized)
    source = FullTextSource(
        source_url=source_url.strip(),
        license_id=license_id.strip(),
        license_verified=True,
        content_sha256=content_hash,
    )
    paragraphs: list[ParagraphEvidence] = []
    cursor = 0
    for chunk in re.split(r"\n{2,}", normalized):
        paragraph = chunk.strip()
        start = normalized.find(paragraph, cursor)
        if start < 0:  # pragma: no cover - normalization makes this unreachable.
            raise FullTextEvidenceError("full_text_offset_error")
        if not paragraph:
            cursor = start + len(chunk)
            continue
        end = start + len(paragraph)
        paragraph_index = len(paragraphs)
        paragraph_hash = _sha256(paragraph)
        paragraphs.append(
            ParagraphEvidence(
                evidence_id=f"paragraph:{content_hash}:{paragraph_index}",
                paragraph_index=paragraph_index,
                text=paragraph,
                text_sha256=paragraph_hash,
                start_char=start,
                end_char=end,
            )
        )
        cursor = start + len(chunk)
    return FullTextEvidenceDocument(source=source, paragraphs=paragraphs)


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise FullTextEvidenceError("full_text_not_text")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
