"""Deterministic reciprocal-rank fusion over recorded retrieval lists."""

from __future__ import annotations

from dataclasses import dataclass

from scholar_agent.agents.retriever import RetrievalOutput
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import RankedPaper, RRFListContribution


RRF_K = 60
RRF_TOP_K = 20


@dataclass(frozen=True)
class RetrievalListEntry:
    paper: Paper
    rank: int


@dataclass(frozen=True)
class RetrievalRankedList:
    source: str
    subquery: str
    entries: tuple[RetrievalListEntry, ...]


def build_retrieval_ranked_lists(
    outputs: list[RetrievalOutput],
) -> list[RetrievalRankedList]:
    """Reconstruct unique ``(source, adapted subquery)`` response lists.

    A run-dedupe hit may expose the same recorded response more than once. Such
    observations are one physical retrieval list and are merged by best rank.
    Empty/failed calls remain auditable in retrieval diagnostics but contribute
    no fabricated ranks.
    """

    list_entries: dict[tuple[str, str], list[RetrievalListEntry]] = {}
    list_order: list[tuple[str, str]] = []
    for output in outputs:
        for stats in output.source_stats:
            subquery = (stats.adapted_query or output.query).strip()
            source = stats.source.strip().casefold()
            if not source or not subquery:
                continue
            if not stats.logical_call_executed and not stats.run_dedupe_hit:
                continue
            key = (source, subquery)
            if key not in list_entries:
                list_entries[key] = []
                list_order.append(key)
            list_entries[key].extend(
                RetrievalListEntry(paper=paper, rank=rank)
                for rank, paper in enumerate(stats.diagnostic_papers, start=1)
            )
    return [
        RetrievalRankedList(
            source=source,
            subquery=subquery,
            entries=tuple(list_entries[(source, subquery)]),
        )
        for source, subquery in list_order
    ]


def fuse_ranked_papers(
    baseline_ranked: list[RankedPaper],
    ranked_lists: list[RetrievalRankedList],
    *,
    k: int = RRF_K,
    top_k: int = RRF_TOP_K,
) -> list[RankedPaper]:
    """Fuse candidates by RRF while retaining the current score as tie-breaker."""

    if k <= 0:
        raise ValueError("rrf_k_must_be_positive")
    candidate_profiles = [build_identity_profile(item.paper) for item in baseline_ranked]
    best_ranks: list[dict[tuple[str, str], int]] = [
        {} for _ in baseline_ranked
    ]
    for ranked_list in ranked_lists:
        list_key = (ranked_list.source, ranked_list.subquery)
        for entry in ranked_list.entries:
            if entry.rank <= 0:
                continue
            matched_index = _candidate_index(
                build_identity_profile(entry.paper), candidate_profiles
            )
            if matched_index is None:
                continue
            previous = best_ranks[matched_index].get(list_key)
            if previous is None or entry.rank < previous:
                best_ranks[matched_index][list_key] = entry.rank

    missing = [
        _candidate_label(item.paper)
        for index, item in enumerate(baseline_ranked)
        if not best_ranks[index]
    ]
    if missing:
        raise ValueError(
            "rrf_provenance_incomplete:" + ",".join(sorted(missing))
        )

    enriched: list[RankedPaper] = []
    for index, ranked in enumerate(baseline_ranked):
        contributions = [
            RRFListContribution(
                source=source,
                subquery=subquery,
                rank=rank,
                reciprocal_score=1.0 / (k + rank),
            )
            for (source, subquery), rank in sorted(best_ranks[index].items())
        ]
        enriched.append(
            ranked.model_copy(
                update={
                    "rrf_score": sum(item.reciprocal_score for item in contributions),
                    "rrf_contributions": contributions,
                    "original_rank": ranked.rank,
                }
            )
        )

    enriched.sort(key=_rrf_sort_key)
    result: list[RankedPaper] = []
    for fused_rank, ranked in enumerate(enriched, start=1):
        original_rank = ranked.original_rank or ranked.rank
        change = _top_k_change(original_rank, fused_rank, top_k=top_k)
        result.append(
            ranked.model_copy(
                update={
                    "rank": fused_rank,
                    "rrf_top_20_change": change,
                    "rrf_rank_change_reason": _rank_change_reason(
                        ranked,
                        change,
                    ),
                }
            )
        )
    return result


def _candidate_index(
    profile: IdentityProfile,
    candidates: list[IdentityProfile],
) -> int | None:
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if identity_evidence_from_profiles(candidate, profile).equivalent
    ]
    if len(matches) > 1:
        raise ValueError("rrf_provenance_ambiguous_identity")
    return matches[0] if matches else None


def _rrf_sort_key(ranked: RankedPaper) -> tuple[object, ...]:
    profile = build_identity_profile(ranked.paper)
    identity_key = tuple(sorted(profile.identifiers))
    return (
        -(ranked.rrf_score or 0.0),
        -ranked.final_score,
        profile.title,
        ranked.paper.year or 0,
        tuple(sorted(profile.authors)),
        identity_key,
    )


def _candidate_label(paper: Paper) -> str:
    profile = build_identity_profile(paper)
    if profile.identifiers:
        return sorted(profile.identifiers)[0]
    return f"title:{profile.title}:{paper.year or 0}"


def _top_k_change(original_rank: int, fused_rank: int, *, top_k: int) -> str:
    if original_rank <= top_k and fused_rank <= top_k:
        return "retained_top_20"
    if original_rank > top_k and fused_rank <= top_k:
        return "entered_top_20"
    if original_rank <= top_k and fused_rank > top_k:
        return "exited_top_20"
    return "remained_outside_top_20"


def _rank_change_reason(ranked: RankedPaper, change: str) -> str:
    best_rank = min(item.rank for item in ranked.rrf_contributions)
    return (
        f"{change}:rrf_score={ranked.rrf_score or 0.0:.12f};"
        f"contributing_lists={len(ranked.rrf_contributions)};"
        f"best_list_rank={best_rank};"
        f"current_score_tiebreak={ranked.final_score:.4f}"
    )
