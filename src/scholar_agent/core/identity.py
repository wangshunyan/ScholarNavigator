"""保守、可审计的论文身份归一化与等价判断。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_IDENTIFIER_FIELDS = (
    ("doi", "doi"),
    ("arxiv_id", "arxiv"),
    ("openalex_id", "openalex"),
    ("semantic_scholar_id", "s2"),
    ("s2orc_corpus_id", "s2orc"),
    ("pubmed_id", "pubmed"),
)
_S2ORC_FIELD_ALIASES = (
    "s2orc_corpus_id",
    "s2orc_id",
    "corpus_id",
    "corpusId",
    "CorpusId",
)
_S2ORC_PREFIX_RE = re.compile(
    r"^(?:s2orc(?:[_ -]?corpus)?[_ -]?id|corpus[_ -]?id)\s*:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IdentityEvidence:
    """等价判断的结果及其可审计依据。"""

    equivalent: bool
    rule: str
    shared_identifiers: tuple[str, ...] = ()
    conflicting_identifiers: tuple[str, ...] = ()
    title: str | None = None
    author_overlap: tuple[str, ...] = ()
    year: int | None = None


@dataclass(frozen=True)
class IdentityProfile:
    """一次性规范化的论文身份字段，供批量比较复用。"""

    identifiers: frozenset[str]
    field_values: tuple[tuple[str, str | None], ...]
    title: str
    authors: frozenset[str]
    year: int | None


def normalize_doi(value: str | None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = text.casefold().split("?", 1)[0].rstrip("/.")
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip().rstrip("/.") or None


def normalize_arxiv_id(value: str | None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = text.casefold().split("?", 1)[0].rstrip("/")
    if "arxiv.org/abs/" in text:
        text = text.split("arxiv.org/abs/", 1)[1]
    elif "arxiv.org/pdf/" in text:
        text = text.split("arxiv.org/pdf/", 1)[1]
    for prefix in ("arxiv:", "abs/", "pdf/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.endswith(".pdf"):
        text = text[:-4]
    return _ARXIV_VERSION_RE.sub("", text).strip() or None


def normalize_simple_id(value: str | None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = text.casefold().rstrip("/")
    if "://" in text:
        text = urlparse(text).path.rstrip("/").rsplit("/", 1)[-1]
    else:
        text = text.rsplit("/", 1)[-1]
    for prefix in (
        "openalex:",
        "s2:",
        "pmid:",
        "pubmed:",
        "semantic_scholar:",
        "semantic-scholar:",
        "semanticscholar:",
        "corpusid:",
        "paper:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip() or None


def normalize_s2orc_corpus_id(value: Any | None) -> str | None:
    """Normalize only representation details around an exact S2ORC Corpus ID."""

    text = _clean(value)
    if not text:
        return None
    text = _S2ORC_PREFIX_RE.sub("", text).strip()
    return text or None


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).casefold()
    text = _LATEX_COMMAND_RE.sub(lambda match: match.group(1) or " ", text)
    text = text.replace("$", " ").replace("\\", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("^", " ").replace("_", " ").replace("~", " ")
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def paper_identifier_set(paper: Any) -> set[str]:
    return set(build_identity_profile(paper).identifiers)


def paper_title_year_key(paper: Any) -> str | None:
    title = normalize_title(_value(paper, "title"))
    year = _value(paper, "year")
    if not title or year is None:
        return None
    return f"title_year:{title}:{year}"


def identity_evidence(left: Any, right: Any) -> IdentityEvidence:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    )


def build_identity_profile(paper: Any) -> IdentityProfile:
    values = _identifier_values(paper)
    identifiers = {
        f"{prefix}:{value}"
        for (field, prefix), (_, value) in zip(_IDENTIFIER_FIELDS, values.items())
        if value
    }
    doi = values.get("doi")
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv = normalize_arxiv_id(doi.removeprefix("10.48550/arxiv."))
        if arxiv:
            identifiers.add(f"arxiv:{arxiv}")
    return IdentityProfile(
        identifiers=frozenset(identifiers),
        field_values=tuple(values.items()),
        title=normalize_title(_value(paper, "title")),
        authors=frozenset(_author_keys(paper)),
        year=_value(paper, "year"),
    )


def identity_evidence_from_profiles(
    left: IdentityProfile, right: IdentityProfile
) -> IdentityEvidence:
    left_values = dict(left.field_values)
    right_values = dict(right.field_values)
    conflicts = tuple(
        f"{field}:{left_values[field]}!={right_values[field]}"
        for field, _ in _IDENTIFIER_FIELDS
        if left_values[field]
        and right_values[field]
        and left_values[field] != right_values[field]
    )
    shared = tuple(sorted(left.identifiers & right.identifiers))
    if conflicts:
        return IdentityEvidence(False, "conflicting_stable_identifier", shared, conflicts)
    if shared:
        return IdentityEvidence(True, "shared_stable_identifier", shared)

    # A Corpus ID is dataset identity, not title evidence. If either side carries
    # one, equivalence must come from an exact shared stable identifier above.
    if left_values.get("s2orc_corpus_id") or right_values.get("s2orc_corpus_id"):
        return IdentityEvidence(False, "s2orc_requires_exact_identifier")

    left_title = left.title
    right_title = right.title
    left_year = left.year
    right_year = right.year
    overlap = tuple(sorted(left.authors & right.authors))
    if (
        left_title
        and left_title == right_title
        and left_year is not None
        and left_year == right_year
        and overlap
    ):
        return IdentityEvidence(
            True,
            "exact_title_author_year",
            title=left_title,
            author_overlap=overlap,
            year=left_year,
        )
    return IdentityEvidence(False, "no_identity_evidence")


def _identifier_values(paper: Any) -> dict[str, str | None]:
    return {
        field: (
            normalize_doi(_value(paper, field))
            if field == "doi"
            else normalize_arxiv_id(_value(paper, field))
            if field == "arxiv_id"
            else normalize_s2orc_corpus_id(_value(paper, field))
            if field == "s2orc_corpus_id"
            else normalize_simple_id(_value(paper, field))
        )
        for field, _ in _IDENTIFIER_FIELDS
    }


def _author_keys(paper: Any) -> set[str]:
    authors = _value(paper, "authors") or []
    keys: set[str] = set()
    for author in authors:
        key = normalize_title(str(author))
        if key:
            keys.add(key)
    return keys


def _value(paper: Any, field: str) -> Any:
    if hasattr(paper, "paper") and not hasattr(paper, field):
        paper = paper.paper
    if isinstance(paper, dict):
        if isinstance(paper.get("paper"), dict):
            paper = paper["paper"]
        if field in paper:
            return paper[field]
        identifiers = paper.get("identifiers") or {}
        if field == "s2orc_corpus_id":
            return _first_alias_value(paper, identifiers, paper.get("metadata") or {})
        return identifiers.get(field)
    value = getattr(paper, field, None)
    if value is not None:
        return value
    identifiers = getattr(paper, "identifiers", None)
    if field == "s2orc_corpus_id":
        metadata = getattr(paper, "metadata", None)
        return _first_alias_value(paper, identifiers, metadata)
    return getattr(identifiers, field, None) if identifiers is not None else None


def _first_alias_value(*containers: Any) -> Any:
    for container in containers:
        if container is None:
            continue
        for alias in _S2ORC_FIELD_ALIASES:
            if isinstance(container, dict):
                value = container.get(alias)
            else:
                value = getattr(container, alias, None)
            if value is not None and _clean(value):
                return value
    return None


def _clean(value: Any | None) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split()).strip() or None
