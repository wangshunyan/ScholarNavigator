"""Paper deduplication and merge helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

from scholar_agent.core.identity import (
    IdentityEvidence,
    IdentityProfile,
    build_identity_profile,
    identity_evidence,
    identity_evidence_from_profiles,
    normalize_arxiv_id,
    normalize_doi,
    normalize_simple_id,
    normalize_title,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers while preserving first-seen order."""

    deduplicated, _ = deduplicate_papers_with_audit(papers)
    return deduplicated


def deduplicate_papers_with_audit(
    papers: list[Paper],
    *,
    lineage_query_identity: str | None = None,
    lineage_source_terminals: Sequence[Mapping[str, Any]] | None = None,
    lineage_untrusted_metadata_isolation: Mapping[str, Any] | None = None,
    lineage_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Paper], list[dict[str, object]]]:
    """Deduplicate and return an evidence row for every accepted merge.

    ``lineage_sink`` is an opt-in observational hook.  It uses the exact same
    clusters and merged objects as the production path; with the hook omitted,
    behavior and serialized output are unchanged.
    """

    deduplicated: list[Paper] = []
    profiles: list[IdentityProfile] = []
    evidence_rows: list[dict[str, object]] = []
    capture_lineage = lineage_sink is not None
    cluster_indexes: list[list[int]] = []
    cluster_evidence: list[list[IdentityEvidence]] = []
    for input_index, paper in enumerate(papers):
        profile = build_identity_profile(paper)
        match_index = _find_duplicate_index(profiles, profile)
        if match_index is None:
            deduplicated.append(paper.model_copy(deep=True))
            profiles.append(profile)
            if capture_lineage:
                cluster_indexes.append([input_index])
                cluster_evidence.append(
                    [IdentityEvidence(False, "new_identity_cluster")]
                )
            continue
        evidence = identity_evidence_from_profiles(profiles[match_index], profile)
        propagated_identifiers = sorted(
            profile.identifiers - profiles[match_index].identifiers
        )
        evidence_rows.append(
            {
                "existing_index": match_index,
                "incoming_title": paper.title,
                "rule": evidence.rule,
                "shared_identifiers": list(evidence.shared_identifiers),
                "conflicting_identifiers": list(evidence.conflicting_identifiers),
                "propagated_identifiers": propagated_identifiers,
                "title": evidence.title,
                "author_overlap": list(evidence.author_overlap),
                "year": evidence.year,
            }
        )
        deduplicated[match_index] = _merge_papers(deduplicated[match_index], paper)
        profiles[match_index] = build_identity_profile(deduplicated[match_index])
        if capture_lineage:
            cluster_indexes[match_index].append(input_index)
            cluster_evidence[match_index].append(evidence)

    if lineage_sink is not None:
        if not lineage_query_identity:
            raise ValueError("lineage_query_identity_required")
        from scholar_agent.core.result_lineage import build_result_lineage_document

        document = build_result_lineage_document(
            query_identity=lineage_query_identity,
            input_papers=papers,
            final_papers=deduplicated,
            cluster_indexes=cluster_indexes,
            cluster_member_evidence=cluster_evidence,
            source_terminals=lineage_source_terminals,
            untrusted_metadata_isolation=lineage_untrusted_metadata_isolation,
        )
        lineage_sink(document.model_dump(mode="json"))

    return deduplicated, evidence_rows


def deduplicate_papers_with_lineage(
    papers: list[Paper],
    *,
    query_identity: str,
    source_terminals: Sequence[Mapping[str, Any]] | None = None,
    untrusted_metadata_isolation: Mapping[str, Any] | None = None,
) -> tuple[list[Paper], list[dict[str, object]], dict[str, Any]]:
    """Run production deduplication and return its observational lineage."""

    captured: list[dict[str, Any]] = []
    deduplicated, audit = deduplicate_papers_with_audit(
        papers,
        lineage_query_identity=query_identity,
        lineage_source_terminals=source_terminals,
        lineage_untrusted_metadata_isolation=untrusted_metadata_isolation,
        lineage_sink=captured.append,
    )
    if len(captured) != 1:
        raise RuntimeError("result_lineage_capture_failed")
    return deduplicated, audit, captured[0]


def _find_duplicate_index(
    existing: list[IdentityProfile], candidate: IdentityProfile
) -> int | None:
    for index, profile in enumerate(existing):
        if identity_evidence_from_profiles(profile, candidate).equivalent:
            return index
    return None


def _is_duplicate(left: Paper, right: Paper) -> bool:
    return identity_evidence(left, right).equivalent


def paper_identity_evidence(left: Paper, right: Paper) -> IdentityEvidence:
    """Return the shared identity rule used by production deduplication."""

    return identity_evidence(left, right)


def _merge_papers(existing: Paper, incoming: Paper) -> Paper:
    return Paper(
        title=_choose_title(existing.title, incoming.title),
        authors=_choose_authors(existing.authors, incoming.authors),
        year=existing.year if existing.year is not None else incoming.year,
        venue=existing.venue or incoming.venue,
        abstract=_choose_longer_text(existing.abstract, incoming.abstract),
        identifiers=_merge_identifiers(existing.identifiers, incoming.identifiers),
        urls=_merge_urls(existing.urls, incoming.urls),
        sources=_merge_unique(existing.sources, incoming.sources),
        citation_count=max(existing.citation_count or 0, incoming.citation_count or 0),
    )


def _merge_identifiers(left: PaperIdentifiers, right: PaperIdentifiers) -> PaperIdentifiers:
    return PaperIdentifiers(
        doi=left.doi or right.doi,
        arxiv_id=left.arxiv_id or right.arxiv_id,
        semantic_scholar_id=left.semantic_scholar_id or right.semantic_scholar_id,
        s2orc_corpus_id=left.s2orc_corpus_id or right.s2orc_corpus_id,
        openalex_id=left.openalex_id or right.openalex_id,
        pubmed_id=left.pubmed_id or right.pubmed_id,
    )


def _merge_urls(left: PaperUrls, right: PaperUrls) -> PaperUrls:
    return PaperUrls(
        landing_page=left.landing_page or right.landing_page,
        pdf=left.pdf or right.pdf,
    )


def _choose_title(left: str, right: str) -> str:
    if _is_placeholder_title(left) and not _is_placeholder_title(right):
        return right
    if _is_placeholder_title(right) and not _is_placeholder_title(left):
        return left
    return right if len(right.strip()) > len(left.strip()) else left


def _choose_authors(left: list[str], right: list[str]) -> list[str]:
    return right if len(right) > len(left) else left


def _choose_longer_text(left: str, right: str) -> str:
    return right if len(right.strip()) > len(left.strip()) else left


def _merge_unique(*values: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in values:
        for value in group:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            merged.append(value)
            seen.add(key)
    return merged


def _is_placeholder_title(value: str) -> bool:
    return normalize_title(value).startswith("untitled")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
