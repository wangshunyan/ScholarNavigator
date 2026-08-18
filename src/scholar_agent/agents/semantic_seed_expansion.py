"""Deterministic seed selection for Semantic Scholar recommendations."""

from __future__ import annotations

import time
from collections.abc import Callable

from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.identity import (
    identity_evidence,
    normalize_arxiv_id,
    normalize_doi,
    normalize_s2orc_corpus_id,
    normalize_simple_id,
    paper_identifier_set,
)
from scholar_agent.core.search_schemas import (
    RankedPaper,
    SemanticSeedExpansionOutput,
    SemanticSeedExpansionRecord,
    SemanticSeedResolutionRecord,
    SemanticSeedExpansionSeed,
)


RecommendationFetcher = Callable[[list[str], int], ConnectorSearchResult]
SeedResolver = Callable[[list[str]], ConnectorSearchResult]
MAX_SEED_RESOLUTION_CANDIDATES = 100


def expand_semantic_seeds(
    ranked_papers: list[RankedPaper],
    fetch_recommendations: RecommendationFetcher,
    *,
    resolve_seed_ids: SeedResolver | None = None,
    max_seeds: int = 3,
    limit: int = 100,
) -> SemanticSeedExpansionOutput:
    """Resolve exact stable IDs, then select ranked S2 seeds and recommend once."""

    started = time.perf_counter()
    requests = _resolution_requests(ranked_papers)
    resolution_result = ConnectorSearchResult()
    resolutions: list[SemanticSeedResolutionRecord] = []
    resolved_by_rank: dict[int, tuple[str, str]] = {}
    resolution_status = "not_needed" if not requests else "not_attempted"
    if requests and resolve_seed_ids is not None:
        try:
            resolution_result = resolve_seed_ids(
                [identifier for _, identifier in requests]
            )
        except Exception as exc:  # noqa: BLE001 - preserve direct seeds
            resolution_result = ConnectorSearchResult(
                error_message=f"seed_resolution_failed:{type(exc).__name__}",
                warnings=[f"semantic_seed_resolution_failure:{type(exc).__name__}"],
                diagnostics=ConnectorDiagnostics(error_count=1),
                reference_batch_status="failed",
            )
        resolutions, resolved_by_rank = _match_resolutions(
            requests,
            resolution_result,
        )
        if resolution_result.error_message:
            resolution_status = "source_failure"
        elif any(item.status in {"missing", "conflict"} for item in resolutions):
            resolution_status = "partial_success"
        else:
            resolution_status = "success"
    seeds = _select_seeds(
        ranked_papers,
        resolved_by_rank=resolved_by_rank,
        max_seeds=max_seeds,
    )
    resolution_recorded = (
        resolution_result.recorded_diagnostics or ConnectorDiagnostics()
    )
    resolution_fields = {
        "resolution_snapshot_key": resolution_result.snapshot_key,
        "resolution_status": resolution_status,
        "resolution_candidate_count": len(requests),
        "resolution_request_identifier_count": len(requests),
        "resolved_candidate_count": sum(
            item.status == "resolved" for item in resolutions
        ),
        "resolution_conflict_count": sum(
            item.status == "conflict" for item in resolutions
        ),
        "resolution_missing_count": sum(
            item.status in {"missing", "source_failure"} for item in resolutions
        ),
        "direct_seed_count": sum(seed.source == "direct" for seed in seeds),
        "resolved_seed_count": sum(seed.source == "resolved" for seed in seeds),
        "resolutions": resolutions,
        "resolution_diagnostics": resolution_result.diagnostics,
        "recorded_resolution_diagnostics": resolution_recorded,
        "resolution_latency_seconds": resolution_result.latency_seconds,
        "recorded_resolution_latency_seconds": (
            resolution_result.recorded_latency_seconds
        ),
    }
    if not seeds:
        warning = (
            "semantic_seed_expansion_seed_resolution_failure"
            if resolution_status == "source_failure"
            else "semantic_seed_expansion_no_eligible_seed"
        )
        record = SemanticSeedExpansionRecord(
            status=(
                "source_failure"
                if resolution_status == "source_failure"
                else "no_eligible_seed"
            ),
            seeds=[],
            skip_reason=(
                "seed_resolution_failure"
                if resolution_status == "source_failure"
                else "no_eligible_seed"
            ),
            warnings=_dedupe([*resolution_result.warnings, warning]),
            diagnostics=resolution_result.diagnostics,
            recorded_diagnostics=resolution_recorded,
            latency_seconds=time.perf_counter() - started,
            recorded_latency_seconds=resolution_result.recorded_latency_seconds,
            **resolution_fields,
        )
        return SemanticSeedExpansionOutput(
            record=record,
            warnings=record.warnings,
            diagnostics=record.diagnostics,
            recorded_diagnostics=record.recorded_diagnostics,
            latency_seconds=record.latency_seconds,
            recorded_latency_seconds=record.recorded_latency_seconds,
        )

    try:
        result = fetch_recommendations(
            [seed.semantic_scholar_id for seed in seeds],
            limit,
        )
    except Exception as exc:  # noqa: BLE001 - preserve initial results on one call failure
        warning = f"semantic_seed_expansion_source_failure:{type(exc).__name__}"
        elapsed = time.perf_counter() - started
        diagnostics = ConnectorDiagnostics(error_count=1, latency_seconds=elapsed)
        record = SemanticSeedExpansionRecord(
            status="source_failure",
            seeds=seeds,
            skip_reason="source_failure",
            warnings=[warning],
            diagnostics=merge_connector_diagnostics(
                [resolution_result.diagnostics, diagnostics]
            ),
            recorded_diagnostics=resolution_recorded,
            latency_seconds=elapsed,
            recorded_latency_seconds=resolution_result.recorded_latency_seconds,
            **resolution_fields,
        )
        return SemanticSeedExpansionOutput(
            record=record,
            warnings=record.warnings,
            diagnostics=record.diagnostics,
            recorded_diagnostics=record.recorded_diagnostics,
            latency_seconds=elapsed,
            recorded_latency_seconds=record.recorded_latency_seconds,
        )

    elapsed = time.perf_counter() - started
    recorded = result.recorded_diagnostics or ConnectorDiagnostics()
    status = "source_failure" if result.error_message else "success"
    warnings = [*resolution_result.warnings, *result.warnings]
    if result.error_message:
        warnings.append("semantic_seed_expansion_source_failure")
    record = SemanticSeedExpansionRecord(
        status=status,
        seeds=seeds,
        snapshot_key=result.snapshot_key,
        raw_recommendation_count=len(result.papers),
        skip_reason="source_failure" if result.error_message else None,
        warnings=_dedupe(warnings),
        diagnostics=merge_connector_diagnostics(
            [resolution_result.diagnostics, result.diagnostics]
        ),
        recorded_diagnostics=merge_connector_diagnostics(
            [resolution_recorded, recorded]
        ),
        latency_seconds=max(result.latency_seconds, elapsed),
        recorded_latency_seconds=(
            resolution_result.recorded_latency_seconds
            + result.recorded_latency_seconds
        ),
        **resolution_fields,
    )
    recommendations = [] if result.error_message else list(result.papers)
    return SemanticSeedExpansionOutput(
        recommendations=recommendations,
        record=record,
        warnings=record.warnings,
        diagnostics=record.diagnostics,
        recorded_diagnostics=record.recorded_diagnostics,
        latency_seconds=record.latency_seconds,
        recorded_latency_seconds=record.recorded_latency_seconds,
    )


def _select_seeds(
    ranked_papers: list[RankedPaper],
    *,
    resolved_by_rank: dict[int, tuple[str, str]] | None = None,
    max_seeds: int,
) -> list[SemanticSeedExpansionSeed]:
    resolved_by_rank = resolved_by_rank or {}
    seeds: list[SemanticSeedExpansionSeed] = []
    seen_ids: set[str] = set()
    for ranked in ranked_papers:
        value = ranked.paper.identifiers.semantic_scholar_id
        direct_id = str(value).strip() if value is not None else ""
        resolved = resolved_by_rank.get(ranked.rank)
        paper_id = direct_id or (resolved[0] if resolved else "")
        key = paper_id.casefold()
        if not paper_id or key in seen_ids:
            continue
        seeds.append(
            SemanticSeedExpansionSeed(
                semantic_scholar_id=paper_id,
                rank=ranked.rank,
                title=ranked.paper.title,
                source="direct" if direct_id else "resolved",
                resolution_identifier=(resolved[1] if resolved and not direct_id else None),
            )
        )
        seen_ids.add(key)
        if len(seeds) >= max(0, max_seeds):
            break
    return seeds


def _resolution_requests(
    ranked_papers: list[RankedPaper],
) -> list[tuple[RankedPaper, str]]:
    requests: list[tuple[RankedPaper, str]] = []
    seen: set[str] = set()
    for ranked in ranked_papers:
        if ranked.paper.identifiers.semantic_scholar_id:
            continue
        identifier = _official_query_identifier(ranked)
        key = identifier.casefold() if identifier else ""
        if not identifier or key in seen:
            continue
        requests.append((ranked, identifier))
        seen.add(key)
        if len(requests) >= MAX_SEED_RESOLUTION_CANDIDATES:
            break
    return requests


def _official_query_identifier(ranked: RankedPaper) -> str | None:
    identifiers = ranked.paper.identifiers
    doi = normalize_doi(identifiers.doi)
    if doi:
        return f"DOI:{doi}"
    arxiv_id = normalize_arxiv_id(identifiers.arxiv_id)
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"
    pubmed_id = normalize_simple_id(identifiers.pubmed_id)
    if pubmed_id:
        return f"PMID:{pubmed_id}"
    corpus_id = normalize_s2orc_corpus_id(identifiers.s2orc_corpus_id)
    if corpus_id:
        return f"CorpusId:{corpus_id}"
    return None


def _match_resolutions(
    requests: list[tuple[RankedPaper, str]],
    result: ConnectorSearchResult,
) -> tuple[list[SemanticSeedResolutionRecord], dict[int, tuple[str, str]]]:
    records: list[SemanticSeedResolutionRecord] = []
    resolved: dict[int, tuple[str, str]] = {}
    if result.error_message:
        return (
            [
                SemanticSeedResolutionRecord(
                    requested_identifier=identifier,
                    rank=ranked.rank,
                    status="source_failure",
                )
                for ranked, identifier in requests
            ],
            resolved,
        )
    for ranked, identifier in requests:
        canonical = _canonical_query_identifier(identifier)
        candidates = [
            paper
            for paper in result.papers
            if canonical is not None and canonical in paper_identifier_set(paper)
        ]
        if not candidates:
            records.append(
                SemanticSeedResolutionRecord(
                    requested_identifier=identifier,
                    rank=ranked.rank,
                    status="missing",
                )
            )
            continue
        mapped = candidates[0]
        evidence = identity_evidence(ranked.paper, mapped)
        semantic_id = mapped.identifiers.semantic_scholar_id
        if (
            not semantic_id
            or evidence.rule != "shared_stable_identifier"
            or evidence.conflicting_identifiers
        ):
            records.append(
                SemanticSeedResolutionRecord(
                    requested_identifier=identifier,
                    rank=ranked.rank,
                    status="conflict",
                    semantic_scholar_id=semantic_id,
                    identity_rule=evidence.rule,
                    shared_identifiers=list(evidence.shared_identifiers),
                    conflicting_identifiers=list(evidence.conflicting_identifiers),
                )
            )
            continue
        resolved[ranked.rank] = (semantic_id, identifier)
        records.append(
            SemanticSeedResolutionRecord(
                requested_identifier=identifier,
                rank=ranked.rank,
                status="resolved",
                semantic_scholar_id=semantic_id,
                identity_rule=evidence.rule,
                shared_identifiers=list(evidence.shared_identifiers),
            )
        )
    return records, resolved


def _canonical_query_identifier(identifier: str) -> str | None:
    prefix, _, value = identifier.partition(":")
    if not value:
        return None
    normalized_prefix = prefix.casefold()
    if normalized_prefix == "doi":
        normalized = normalize_doi(value)
        return f"doi:{normalized}" if normalized else None
    if normalized_prefix == "arxiv":
        normalized = normalize_arxiv_id(value)
        return f"arxiv:{normalized}" if normalized else None
    if normalized_prefix == "pmid":
        normalized = normalize_simple_id(value)
        return f"pubmed:{normalized}" if normalized else None
    if normalized_prefix == "corpusid":
        normalized = normalize_s2orc_corpus_id(value)
        return f"s2orc:{normalized}" if normalized else None
    return None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
