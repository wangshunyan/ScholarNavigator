"""Conservative, opt-in collectors for independent paper-quality evidence."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from scholar_agent.core.identity import normalize_doi
from scholar_agent.core.paper_quality import VerifiedQualityEvidence


_CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
_MAX_RESPONSE_BYTES = 1_048_576

CrossrefEvidenceOutcome = Literal[
    "flagged",
    "no_explicit_retraction_relation",
    "not_found",
    "http_error",
    "network_error",
    "invalid_response",
]


@dataclass(frozen=True)
class CrossrefRetractionLookup:
    """One provenance-free result from an explicit Crossref metadata lookup."""

    paper_identifier: str
    outcome: CrossrefEvidenceOutcome


@dataclass(frozen=True)
class CrossrefRetractionCollection:
    """Flagged evidence plus a compact account of every attempted lookup."""

    evidence: tuple[VerifiedQualityEvidence, ...]
    lookups: tuple[CrossrefRetractionLookup, ...]

    def outcome_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(lookup.outcome for lookup in self.lookups).items()))


def collect_crossref_retraction_evidence(
    paper_identifiers: Sequence[str],
    *,
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> CrossrefRetractionCollection:
    """Collect only explicit Crossref retraction evidence for canonical DOI IDs.

    The collector never writes a ``clear`` record: a missing Crossref relation
    is not proof that a paper was never retracted.  Network, HTTP and schema
    errors remain per-paper ``unknown`` by producing no ledger entry.
    """

    if not 0 < timeout_seconds <= 30:
        raise ValueError("crossref_timeout_out_of_range")
    identifiers = _canonical_doi_identifiers(paper_identifiers)
    evidence: list[VerifiedQualityEvidence] = []
    lookups: list[CrossrefRetractionLookup] = []
    for paper_identifier in identifiers:
        doi = paper_identifier.removeprefix("doi:")
        request = Request(
            f"{_CROSSREF_WORKS_URL}{quote(doi, safe='')}",
            headers={
                "Accept": "application/json",
                "User-Agent": "ScholarNavigator-quality-evidence/1.0",
            },
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                payload = _load_crossref_payload(response)
        except HTTPError as exc:
            outcome = "not_found" if exc.code == 404 else "http_error"
        except (TimeoutError, URLError, OSError):
            outcome = "network_error"
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            outcome = "invalid_response"
        else:
            if _has_explicit_retraction_relation(payload):
                evidence.append(
                    VerifiedQualityEvidence(
                        paper_identifier=paper_identifier,
                        signal_name="retraction_status",
                        state="flagged",
                        source="crossref",
                        source_record_id=f"crossref-work:{doi}",
                    )
                )
                outcome = "flagged"
            else:
                outcome = "no_explicit_retraction_relation"
        lookups.append(CrossrefRetractionLookup(paper_identifier, outcome))
    return CrossrefRetractionCollection(tuple(evidence), tuple(lookups))


def _canonical_doi_identifiers(paper_identifiers: Sequence[str]) -> tuple[str, ...]:
    canonical: set[str] = set()
    for identifier in paper_identifiers:
        prefix, separator, raw_doi = identifier.partition(":")
        normalized = normalize_doi(raw_doi) if prefix == "doi" and separator else None
        value = f"doi:{normalized}" if normalized else None
        if value != identifier:
            raise ValueError("canonical_doi_paper_identifier_required")
        canonical.add(value)
    if not canonical:
        raise ValueError("crossref_paper_identifier_required")
    return tuple(sorted(canonical))


def _load_crossref_payload(response: Any) -> Mapping[str, Any]:
    content_type = response.headers.get_content_type()
    if content_type != "application/json":
        raise ValueError("crossref_json_response_required")
    raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("crossref_response_too_large")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("message"), dict):
        raise ValueError("crossref_message_required")
    return value["message"]


def _has_explicit_retraction_relation(payload: Mapping[str, Any]) -> bool:
    relation = payload.get("relation")
    if isinstance(relation, Mapping) and _nonempty_relation(relation.get("is-retracted-by")):
        return True
    updates = payload.get("update-to")
    if not isinstance(updates, list):
        return False
    return any(
        isinstance(update, Mapping)
        and str(update.get("type") or "").casefold() == "retraction"
        for update in updates
    )


def _nonempty_relation(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return isinstance(value, list) and bool(value)
