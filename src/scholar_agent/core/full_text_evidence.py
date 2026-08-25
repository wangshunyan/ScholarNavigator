"""Deterministic, license-gated paragraph evidence for open full text."""

from __future__ import annotations

import hashlib
from io import BytesIO
import re
from html.parser import HTMLParser
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from pydantic import BaseModel, ConfigDict, Field


class FullTextEvidenceError(ValueError):
    """Base error for closed full-text evidence handling."""


class FullTextLicenseError(FullTextEvidenceError):
    """Raised when the caller has not established a usable license basis."""


class FullTextFetchError(FullTextEvidenceError):
    """Raised for a bounded full-text fetch or parse failure."""


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


class FullTextFetchResult(BaseModel):
    """Outcome for one allow-listed, license-verified full-text attempt."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "succeeded",
        "license_unverified",
        "url_not_allowed",
        "fetch_failed",
        "response_too_large",
        "unsupported_media_type",
        "parser_unavailable",
        "parse_failed",
    ]
    document: FullTextEvidenceDocument | None = None
    source_url: str | None = None
    license_id: str | None = None
    response_media_type: str | None = None
    failure_reason: str | None = None


DEFAULT_FULL_TEXT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_FULL_TEXT_BYTES = 2_000_000
DEFAULT_MAX_PDF_PAGES = 100
_TEXT_MEDIA_TYPES = {
    "text/plain",
    "text/html",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
}


class _AllowlistedRedirectHandler(HTTPRedirectHandler):
    """Re-check every redirect target against the caller's host allow-list."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self._allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        target = urljoin(request.full_url, new_url)
        if not _is_allowed_url(target, self._allowed_hosts):
            raise FullTextEvidenceError("full_text_redirect_not_allowed")
        return super().redirect_request(request, fp, code, msg, headers, target)


def fetch_open_full_text(
    *,
    source_url: str,
    license_id: str,
    license_verified: bool,
    allowed_hosts: set[str],
    timeout_seconds: float = DEFAULT_FULL_TEXT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FULL_TEXT_BYTES,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    opener: Callable[..., Any] = urlopen,
) -> FullTextFetchResult:
    """Fetch one explicit open-text URL under a closed allow-list policy.

    Callers must supply a prior license decision. This helper never discovers a
    license, follows a landing page, or falls back to an arbitrary URL.
    """

    clean_url = source_url.strip()
    if not license_verified or not license_id.strip():
        return FullTextFetchResult(
            status="license_unverified",
            source_url=clean_url or None,
            license_id=license_id.strip() or None,
            failure_reason="full_text_license_unverified",
        )
    if not _is_allowed_url(clean_url, allowed_hosts):
        return FullTextFetchResult(
            status="url_not_allowed",
            source_url=clean_url or None,
            license_id=license_id.strip(),
            failure_reason="full_text_url_not_allowed",
        )
    if timeout_seconds <= 0 or max_bytes <= 0 or max_pdf_pages <= 0:
        raise ValueError("full_text_fetch_limits_invalid")

    request = Request(clean_url, headers={"User-Agent": "ScholarNavigator"})
    try:
        open_function = opener
        if opener is urlopen:
            # A custom opener is deliberately preserved for deterministic tests
            # and callers that already enforce their own redirect policy.
            open_function = build_opener(
                _AllowlistedRedirectHandler(set(allowed_hosts))
            ).open
        with open_function(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", getattr(response, "code", 200)))
            if status < 200 or status >= 300:
                return _failure("fetch_failed", clean_url, license_id, f"http_status:{status}")
            headers = getattr(response, "headers", None)
            content_type = _media_type(_header(headers, "Content-Type"))
            declared_length = _header(headers, "Content-Length")
            if declared_length is not None and _content_length_exceeds(declared_length, max_bytes):
                return _failure("response_too_large", clean_url, license_id, "content_length_exceeded", content_type)
            if content_type not in _TEXT_MEDIA_TYPES and content_type != "application/pdf":
                return _failure("unsupported_media_type", clean_url, license_id, "media_type_not_supported", content_type)
            payload = response.read(max_bytes + 1)
    except FullTextEvidenceError as exc:
        return _failure("url_not_allowed", clean_url, license_id, str(exc))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return _failure("fetch_failed", clean_url, license_id, f"fetch_error:{type(exc).__name__}")
    if len(payload) > max_bytes:
        return _failure("response_too_large", clean_url, license_id, "response_bytes_exceeded", content_type)
    try:
        if content_type == "application/pdf":
            text = _pdf_to_text(payload, max_pages=max_pdf_pages)
        else:
            text = _decode_text(payload, content_type)
        if content_type in {
            "text/html",
            "application/xhtml+xml",
            "application/xml",
            "text/xml",
        }:
            text = _html_to_text(text)
            if _looks_like_browser_challenge(text):
                raise FullTextEvidenceError("full_text_challenge_page")
        document = build_paragraph_evidence(
            text,
            source_url=clean_url,
            license_id=license_id,
            license_verified=True,
        )
    except FullTextEvidenceError as exc:
        status = "parser_unavailable" if str(exc) == "pdf_parser_unavailable" else "parse_failed"
        return _failure(status, clean_url, license_id, str(exc), content_type)
    return FullTextFetchResult(
        status="succeeded",
        document=document,
        source_url=clean_url,
        license_id=license_id.strip(),
        response_media_type=content_type,
    )


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


def _failure(
    status: Literal[
        "fetch_failed", "response_too_large", "unsupported_media_type", "parser_unavailable", "parse_failed"
    ],
    source_url: str,
    license_id: str,
    reason: str,
    media_type: str | None = None,
) -> FullTextFetchResult:
    return FullTextFetchResult(
        status=status,
        source_url=source_url,
        license_id=license_id.strip(),
        response_media_type=media_type,
        failure_reason=reason,
    )


def _is_allowed_url(value: str, allowed_hosts: set[str]) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and not parsed.username
        and not parsed.password
        and port in {None, 443}
        and host in {item.casefold() for item in allowed_hosts}
    )


def _header(headers: Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    value = getter(name) if callable(getter) else None
    return str(value).strip() if value else None


def _media_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().casefold()


def _content_length_exceeds(value: str, maximum: int) -> bool:
    try:
        return int(value) > maximum
    except ValueError:
        return False


def _decode_text(payload: bytes, media_type: str) -> str:
    if not payload:
        raise FullTextEvidenceError("full_text_empty")
    return payload.decode("utf-8", errors="strict")


def _looks_like_browser_challenge(text: str) -> bool:
    """Reject anti-bot interstitials instead of exposing them as evidence."""

    normalized = re.sub(r"\s+", " ", text).casefold()
    markers = (
        "checking your browser",
        "verify you are human",
        "enable javascript and cookies",
        "captcha",
    )
    return any(marker in normalized for marker in markers)


def _pdf_to_text(payload: bytes, *, max_pages: int) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise FullTextEvidenceError("pdf_parser_unavailable") from exc
    try:
        reader = PdfReader(BytesIO(payload), strict=True)
        if reader.is_encrypted:
            raise FullTextEvidenceError("pdf_encrypted")
        if len(reader.pages) > max_pages:
            raise FullTextEvidenceError("pdf_page_limit_exceeded")
        pages = [page.extract_text() or "" for page in reader.pages]
    except FullTextEvidenceError:
        raise
    except (PdfReadError, OSError, ValueError) as exc:
        raise FullTextEvidenceError("pdf_parse_failed") from exc
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise FullTextEvidenceError("pdf_text_unavailable")
    return text


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        if tag.casefold() in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _html_to_text(value: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:  # HTMLParser errors must become a terminal status.
        raise FullTextEvidenceError("full_text_html_parse_failed") from exc
    return parser.text()
