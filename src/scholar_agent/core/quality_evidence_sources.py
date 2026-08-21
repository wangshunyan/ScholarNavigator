"""Conservative, opt-in collectors for independent paper-quality evidence."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from scholar_agent.core.identity import normalize_arxiv_id, normalize_doi
from scholar_agent.core.paper_quality import VerifiedQualityEvidence


_CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
_ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_ARXIV_BATCH_SIZE = 20
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

CrossrefEvidenceOutcome = Literal[
    "flagged",
    "no_explicit_retraction_relation",
    "not_found",
    "http_error",
    "network_error",
    "invalid_response",
]
ArxivDoiResolutionOutcome = Literal[
    "resolved",
    "no_doi",
    "not_returned",
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


@dataclass(frozen=True)
class ArxivDoiResolution:
    """Exact arXiv-ID lookup outcome without storing metadata response text."""

    paper_identifier: str
    outcome: ArxivDoiResolutionOutcome
    doi: str | None = None


@dataclass(frozen=True)
class ArxivCrossrefRetractionCollection:
    """Evidence bound to arXiv IDs plus compact resolver/registry outcomes."""

    evidence: tuple[VerifiedQualityEvidence, ...]
    arxiv_resolutions: tuple[ArxivDoiResolution, ...]
    crossref_lookups: tuple[CrossrefRetractionLookup, ...]

    def outcome_counts(self) -> dict[str, int]:
        counts = Counter(
            f"arxiv:{item.outcome}" for item in self.arxiv_resolutions
        )
        counts.update(f"crossref:{item.outcome}" for item in self.crossref_lookups)
        return dict(sorted(counts.items()))


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
    evidence, lookups = _collect_crossref_retraction_for_dois(
        ((identifier, identifier.removeprefix("doi:")) for identifier in identifiers),
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    return CrossrefRetractionCollection(tuple(evidence), tuple(lookups))


def collect_arxiv_crossref_retraction_evidence(
    paper_identifiers: Sequence[str],
    *,
    timeout_seconds: float = 10.0,
    arxiv_opener: Callable[..., Any] = urlopen,
    crossref_opener: Callable[..., Any] = urlopen,
) -> ArxivCrossrefRetractionCollection:
    """Resolve explicit arXiv DOI metadata then query Crossref for risks.

    This accepts at most 20 exact arXiv identifiers in one public metadata
    request. A resolved DOI is only a lookup key: any Crossref risk evidence
    stays bound to the original ``arxiv:`` paper identifier so it matches the
    fixed P0 corpus without changing that corpus or online ranking inputs.
    """

    identifiers = _canonical_arxiv_identifiers(paper_identifiers)
    if not 0 < timeout_seconds <= 30:
        raise ValueError("arxiv_timeout_out_of_range")
    if len(identifiers) > _MAX_ARXIV_BATCH_SIZE:
        raise ValueError("arxiv_identifier_batch_too_large")
    request = Request(
        f"{_ARXIV_QUERY_URL}?{urlencode({'id_list': ','.join(item.removeprefix('arxiv:') for item in identifiers)})}",
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": "ScholarNavigator-quality-evidence/1.0",
        },
    )
    try:
        with arxiv_opener(request, timeout=timeout_seconds) as response:
            returned = _load_arxiv_doi_mapping(response)
    except HTTPError:
        resolutions = tuple(
            ArxivDoiResolution(identifier, "http_error") for identifier in identifiers
        )
    except (TimeoutError, URLError, OSError):
        resolutions = tuple(
            ArxivDoiResolution(identifier, "network_error") for identifier in identifiers
        )
    except (UnicodeDecodeError, ET.ParseError, ValueError, TypeError):
        resolutions = tuple(
            ArxivDoiResolution(identifier, "invalid_response") for identifier in identifiers
        )
    else:
        resolutions = tuple(
            ArxivDoiResolution(
                paper_identifier=identifier,
                outcome="resolved" if returned.get(identifier) else (
                    "no_doi" if identifier in returned else "not_returned"
                ),
                doi=returned.get(identifier),
            )
            for identifier in identifiers
        )
    pairs = tuple(
        (resolution.paper_identifier, resolution.doi)
        for resolution in resolutions
        if resolution.doi is not None
    )
    evidence, lookups = _collect_crossref_retraction_for_dois(
        pairs,
        timeout_seconds=timeout_seconds,
        opener=crossref_opener,
    )
    return ArxivCrossrefRetractionCollection(
        evidence=tuple(evidence),
        arxiv_resolutions=resolutions,
        crossref_lookups=tuple(lookups),
    )


def _collect_crossref_retraction_for_dois(
    paper_dois: Sequence[tuple[str, str]],
    *,
    timeout_seconds: float,
    opener: Callable[..., Any],
) -> tuple[list[VerifiedQualityEvidence], list[CrossrefRetractionLookup]]:
    evidence: list[VerifiedQualityEvidence] = []
    lookups: list[CrossrefRetractionLookup] = []
    for paper_identifier, doi in paper_dois:
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
    return evidence, lookups


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


def _canonical_arxiv_identifiers(paper_identifiers: Sequence[str]) -> tuple[str, ...]:
    canonical: set[str] = set()
    for identifier in paper_identifiers:
        prefix, separator, raw_arxiv_id = identifier.partition(":")
        normalized = (
            normalize_arxiv_id(raw_arxiv_id)
            if prefix == "arxiv" and separator
            else None
        )
        value = f"arxiv:{normalized}" if normalized else None
        if value != identifier:
            raise ValueError("canonical_arxiv_paper_identifier_required")
        canonical.add(value)
    if not canonical:
        raise ValueError("arxiv_paper_identifier_required")
    return tuple(sorted(canonical))


def _load_arxiv_doi_mapping(response: Any) -> dict[str, str | None]:
    content_type = response.headers.get_content_type()
    if content_type not in {"application/atom+xml", "application/xml", "text/xml"}:
        raise ValueError("arxiv_atom_response_required")
    raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("arxiv_response_too_large")
    root = ET.fromstring(raw)
    result: dict[str, str | None] = {}
    for entry in root.findall(f"{_ATOM_NS}entry"):
        raw_id = entry.findtext(f"{_ATOM_NS}id")
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id is None:
            continue
        doi = normalize_doi(entry.findtext(f"{_ARXIV_NS}doi"))
        identifier = f"arxiv:{arxiv_id}"
        previous = result.get(identifier)
        if identifier in result and previous != doi:
            raise ValueError("arxiv_conflicting_doi_metadata")
        result[identifier] = doi
    return result


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
