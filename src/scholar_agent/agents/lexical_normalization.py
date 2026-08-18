"""Conservative, versioned lexical normalization for relevance evidence.

This module intentionally performs no semantic expansion, fuzzy matching,
stemming beyond fixed singular/plural rules, or dataset-specific handling.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


LEXICAL_NORMALIZATION_VERSION = "lexical-normalization-v1"
LexicalNormalizationField = Literal["title", "abstract"]

_DOTTED_ABBREVIATION = re.compile(
    r"(?<![^\W_])(?:[^\W\d_]\.){2,}",
    flags=re.UNICODE,
)
_ENGLISH_POSSESSIVE = re.compile(r"(?<=\w)[\'’]s\b", flags=re.IGNORECASE)
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class LexicalNormalizationEvidence:
    original_term: str
    normalized_form: str
    field: LexicalNormalizationField


def normalize_lexical_tokens(value: str) -> tuple[str, ...]:
    """Normalize one term or field into conservative comparable tokens."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _ENGLISH_POSSESSIVE.sub("", normalized)
    normalized = _DOTTED_ABBREVIATION.sub(
        lambda match: match.group(0).replace(".", ""),
        normalized,
    )
    return tuple(_singularize(token) for token in _WORD.findall(normalized))


def find_lexical_normalization_match(
    term: str,
    *,
    title: str,
    abstract: str,
) -> LexicalNormalizationEvidence | None:
    """Return a whole-token normalized match, preferring title evidence."""

    expected = normalize_lexical_tokens(term)
    if not expected:
        return None
    normalized_form = " ".join(expected)
    for field, value in (("title", title), ("abstract", abstract)):
        observed = normalize_lexical_tokens(value)
        if _contains_sequence(observed, expected):
            return LexicalNormalizationEvidence(
                original_term=term,
                normalized_form=normalized_form,
                field=field,
            )
    return None


def _contains_sequence(
    observed: tuple[str, ...], expected: tuple[str, ...]
) -> bool:
    width = len(expected)
    return any(
        observed[index : index + width] == expected
        for index in range(len(observed) - width + 1)
    )


def _singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if len(token) > 4 and token.endswith("ies"):
        candidate = token[:-3] + "y"
        return candidate if len(candidate) > 3 else token
    if len(token) > 4 and token.endswith(
        ("ches", "shes", "xes", "zes", "sses")
    ):
        candidate = token[:-2]
        return candidate if len(candidate) > 3 else token
    if len(token) > 4 and token.endswith("s") and not token.endswith(
        ("ss", "us", "is")
    ):
        candidate = token[:-1]
        return candidate if len(candidate) > 3 else token
    return token
