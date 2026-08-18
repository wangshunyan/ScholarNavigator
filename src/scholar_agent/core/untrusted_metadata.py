"""Trust-boundary helpers for connector-supplied academic metadata.

The authoritative :class:`Paper` remains untouched.  These helpers derive
bounded representations for LLM transport, diagnostics, links, and exports,
while recording only hashes and transformation names for optional audit use.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.core.paper_schemas import Paper


CONTRACT_VERSION = "untrusted_metadata_isolation_v1"
SCHEMA_VERSION = "1"
NORMALIZATION_VERSION = "untrusted_metadata_nfkc_escape_v1"
ARTIFACT_NAME = "untrusted_metadata_isolation.jsonl"

FIELD_LIMITS = {
    "paper.title": 512,
    "paper.abstract": 4000,
    "paper.author": 256,
    "paper.venue": 256,
    "paper.urls.landing_page": 2048,
    "paper.urls.pdf": 2048,
    "source.error_message": 240,
}
SAFE_URL_SCHEMES = frozenset({"http", "https"})
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_INJECTION_MARKERS = re.compile(
    r"(?i)(ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"(?:system|developer|tool)\s*(?:message|role|call)|"
    r"authorization|api[_ -]?key|access[_ -]?token|\.env|"
    r"<\/?(?:system|tool|script)|javascript:|data:text/html|file:)"
)
_SAFE_DIAGNOSTIC = re.compile(r"^[A-Za-z0-9_.:-]+(?: [A-Za-z0-9_.:-]+){0,5}$")


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def run_manifest_output_spec() -> dict[str, str]:
    """Return the generic output registration used by run_manifest_v1."""

    return {"path": ARTIFACT_NAME, "role": CONTRACT_VERSION, "format": "jsonl"}


class MetadataIsolationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_identity: str = Field(pattern=r"^query:[0-9a-f]{64}$")
    result_identity: str = Field(pattern=r"^record:[0-9a-f]{64}$")
    field: str
    status: Literal["preserved", "normalized", "escaped", "truncated", "rejected"]
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transformations: list[str] = Field(default_factory=list)
    raw_codepoint_count: int = Field(ge=0)
    derived_codepoint_count: int = Field(ge=0)


class UntrustedMetadataIsolationDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["untrusted_metadata_isolation_v1"] = CONTRACT_VERSION
    schema_version: Literal["1"] = SCHEMA_VERSION
    normalization_version: Literal[
        "untrusted_metadata_nfkc_escape_v1"
    ] = NORMALIZATION_VERSION
    query_identity: str = Field(pattern=r"^query:[0-9a-f]{64}$")
    records: list[MetadataIsolationRecord]
    record_count: int = Field(ge=0)
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class UntrustedMetadataObserver:
    """Collect hash-only observations without retaining connector text."""

    def __init__(self) -> None:
        self._records: list[MetadataIsolationRecord] = []

    def observe(
        self,
        *,
        query_identity: str,
        result_identity: str,
        field: str,
        raw: str,
        derived: str | None,
        status: str,
        transformations: list[str],
    ) -> None:
        self._records.append(
            MetadataIsolationRecord(
                query_identity=query_identity,
                result_identity=result_identity,
                field=field,
                status=status,
                raw_sha256=_text_sha256(raw),
                derived_sha256=_text_sha256(derived) if derived is not None else None,
                transformations=sorted(set(transformations)),
                raw_codepoint_count=len(raw),
                derived_codepoint_count=len(derived or ""),
            )
        )

    def document(self, query_identity: str) -> UntrustedMetadataIsolationDocument:
        rows = sorted(
            (item for item in self._records if item.query_identity == query_identity),
            key=lambda item: (
                item.result_identity,
                item.field,
                item.raw_sha256,
                item.derived_sha256 or "",
            ),
        )
        payload = [item.model_dump(mode="json") for item in rows]
        return UntrustedMetadataIsolationDocument(
            query_identity=query_identity,
            records=rows,
            record_count=len(rows),
            records_sha256=stable_sha256(payload),
        )


@dataclass(frozen=True)
class ProtectedText:
    value: str
    status: str
    transformations: tuple[str, ...]


def opaque_record_identity(paper: Paper) -> str:
    return f"record:{stable_sha256(paper.model_dump(mode='json'))}"


def protect_text(
    value: object,
    *,
    field: str,
    query_identity: str,
    result_identity: str,
    observer: UntrustedMetadataObserver | None = None,
    limit: int | None = None,
) -> str:
    """Return a bounded data representation and optionally record its lineage."""

    if field not in FIELD_LIMITS:
        raise ValueError(f"unsupported_untrusted_field:{field}")
    raw = "" if value is None else str(value)
    declared_limit = FIELD_LIMITS[field]
    effective_limit = declared_limit if limit is None else min(declared_limit, limit)
    if effective_limit < 1:
        raise ValueError("untrusted_field_limit_must_be_positive")
    protected = _protect_text(raw, limit=effective_limit)
    if observer is not None:
        observer.observe(
            query_identity=query_identity,
            result_identity=result_identity,
            field=field,
            raw=raw,
            derived=protected.value,
            status=protected.status,
            transformations=list(protected.transformations),
        )
    return protected.value


def protect_url(
    value: object,
    *,
    field: str,
    query_identity: str,
    result_identity: str,
    observer: UntrustedMetadataObserver | None = None,
) -> str | None:
    """Accept only bounded, credential-free HTTP(S) URLs for active links."""

    if field not in {"paper.urls.landing_page", "paper.urls.pdf"}:
        raise ValueError(f"unsupported_untrusted_url_field:{field}")
    raw = "" if value is None else str(value)
    text_result = _protect_text(raw, limit=FIELD_LIMITS[field])
    protected = text_result.value
    accepted: str | None = protected
    transformations = list(text_result.transformations)
    try:
        parsed = urlsplit(protected)
        if (
            parsed.scheme.casefold() not in SAFE_URL_SCHEMES
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or _INJECTION_MARKERS.search(protected)
        ):
            accepted = None
            transformations.append("reject_unsafe_url_v1")
    except ValueError:
        accepted = None
        transformations.append("reject_malformed_url_v1")
    if observer is not None:
        observer.observe(
            query_identity=query_identity,
            result_identity=result_identity,
            field=field,
            raw=raw,
            derived=accepted,
            status="rejected" if accepted is None else (
                text_result.status
            ),
            transformations=transformations,
        )
    return accepted


def safe_diagnostic_message(value: object) -> str:
    """Return a non-executable diagnostic code without echoing unsafe text."""

    raw = "" if value is None else str(value)
    protected = _protect_text(raw, limit=FIELD_LIMITS["source.error_message"])
    candidate = protected.value.strip()
    if (
        candidate
        and _SAFE_DIAGNOSTIC.fullmatch(candidate)
        and not _INJECTION_MARKERS.search(candidate)
        and not _looks_like_absolute_path(candidate)
    ):
        return candidate
    return f"untrusted_source_error:{_text_sha256(raw)[:16]}"


def protect_source_error(
    value: object,
    *,
    source: str,
    query_identity: str,
    observer: UntrustedMetadataObserver | None = None,
) -> str:
    """Derive a public diagnostic and retain only hash-level source lineage."""

    raw = "" if value is None else str(value)
    derived = safe_diagnostic_message(raw)
    if observer is not None:
        observer.observe(
            query_identity=query_identity,
            result_identity=f"record:{stable_sha256({'source': source})}",
            field="source.error_message",
            raw=raw,
            derived=derived,
            status=("preserved" if derived == raw else "rejected"),
            transformations=(
                [] if derived == raw else ["unsafe_diagnostic_to_hash_identity_v1"]
            ),
        )
    return derived


def build_llm_paper_payload(
    paper: Paper,
    *,
    query_identity: str,
    observer: UntrustedMetadataObserver | None = None,
) -> dict[str, Any]:
    """Build the relevance-judgement paper payload from bounded data fields."""

    record_identity = opaque_record_identity(paper)
    return {
        "metadata_role": "untrusted_data",
        "instruction_capability": False,
        "field_roles": {
            field: "untrusted_data"
            for field in (
                "title",
                "authors",
                "year",
                "venue",
                "abstract",
                "identifiers",
                "sources",
                "citation_count",
            )
        },
        "title": protect_text(
            paper.title,
            field="paper.title",
            query_identity=query_identity,
            result_identity=record_identity,
            observer=observer,
        ),
        "authors": [
            protect_text(
                author,
                field="paper.author",
                query_identity=query_identity,
                result_identity=record_identity,
                observer=observer,
            )
            for author in paper.authors
        ],
        "year": paper.year,
        "venue": protect_text(
            paper.venue or "",
            field="paper.venue",
            query_identity=query_identity,
            result_identity=record_identity,
            observer=observer,
        ),
        "abstract": protect_text(
            paper.abstract,
            field="paper.abstract",
            query_identity=query_identity,
            result_identity=record_identity,
            observer=observer,
            limit=1200,
        ),
        "identifiers": paper.identifiers.model_dump(mode="json"),
        "sources": list(paper.sources),
        "citation_count": paper.citation_count,
    }


def _protect_text(raw: str, *, limit: int) -> ProtectedText:
    transformations: list[str] = []
    normalized = unicodedata.normalize("NFKC", raw)
    if normalized != raw:
        transformations.append("unicode_nfkc_v1")
    output: list[str] = []
    for character in normalized:
        if character in {"\r", "\n", "\t"}:
            output.append(" ")
            transformations.append("line_break_or_tab_to_space_v1")
        elif character in _BIDI_CONTROLS or unicodedata.category(character) in {
            "Cc",
            "Cf",
        }:
            output.append(_visible_escape(character))
            transformations.append("control_or_bidi_visible_escape_v1")
        else:
            output.append(character)
    value = "".join(output)
    if len(value) > limit:
        value = value[:limit] + "…"
        transformations.append(f"truncate_codepoints:{limit}")
    transformations = sorted(set(transformations))
    if any(item.startswith("truncate_codepoints:") for item in transformations):
        status = "truncated"
    elif "control_or_bidi_visible_escape_v1" in transformations:
        status = "escaped"
    elif transformations:
        status = "normalized"
    else:
        status = "preserved"
    return ProtectedText(value, status, tuple(transformations))


def _visible_escape(character: str) -> str:
    codepoint = ord(character)
    return f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.search(r"(?:^|\s)(?:/[A-Za-z]|[A-Za-z]:\\)", value))


def _text_sha256(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()
