"""Rule-based reranking for judged papers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

from scholar_agent.core.identity import build_identity_profile, normalize_title
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    JudgementCategory,
    JudgementResult,
    QueryAnalysis,
    RankedPaper,
    RerankScoreBreakdown,
)


CATEGORY_TIER: dict[str, int] = {
    "highly_relevant": 0,
    "partially_relevant": 1,
    "weakly_relevant": 2,
    "irrelevant": 3,
    "insufficient_evidence": 4,
}

CATEGORY_MULTIPLIER: dict[str, float] = {
    "highly_relevant": 1.0,
    "partially_relevant": 0.92,
    "weakly_relevant": 0.82,
    "irrelevant": 0.45,
    "insufficient_evidence": 0.25,
}

PRODUCTION_RERANK_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("category_tier", "ascending"),
    ("final_score", "descending"),
    ("relevance_score", "descending"),
    ("nonnegative_citation_count", "descending"),
    ("year_or_zero", "descending"),
    ("title_casefold", "ascending"),
    ("original_index", "ascending"),
)

TieBreakPolicy = Literal["original_index_v1", "deterministic_tiebreak_v2"]
DEFAULT_TIEBREAK_POLICY: TieBreakPolicy = "original_index_v1"
DETERMINISTIC_TIEBREAK_V2 = "deterministic_tiebreak_v2"


class DeterministicTieBreakUnavailable(ValueError):
    """A v2 exact-tie group lacks a unique, auditable stable key."""


class RerankerAgent:
    """Deterministic metadata-only reranker."""

    def rerank(
        self,
        query_analysis: QueryAnalysis,
        judged_papers: list[JudgementResult],
        *,
        top_k: int = 20,
        tie_break_policy: TieBreakPolicy = DEFAULT_TIEBREAK_POLICY,
        source_record_refs: Mapping[int, Sequence[str]] | None = None,
    ) -> list[RankedPaper]:
        if top_k <= 0:
            return []

        scored = [
            _score_judgement(query_analysis, judgement, original_index)
            for original_index, judgement in enumerate(judged_papers)
        ]
        if tie_break_policy == DEFAULT_TIEBREAK_POLICY:
            scored.sort(key=_sort_key)
        elif tie_break_policy == DETERMINISTIC_TIEBREAK_V2:
            refs = source_record_refs or {}
            stable_keys = {
                item.original_index: deterministic_tiebreak_v2_key(
                    item.judgement.paper,
                    source_record_refs=refs.get(item.original_index, ()),
                )
                for item in scored
            }
            _validate_v2_tie_keys(scored, stable_keys)
            scored.sort(
                key=lambda item: (
                    *_primary_sort_key(item),
                    stable_keys[item.original_index],
                )
            )
        else:
            raise ValueError("unsupported_tie_break_policy")
        return [
            _to_ranked_paper(rank=index + 1, scored=scored_item)
            for index, scored_item in enumerate(scored[:top_k])
        ]


def rerank_papers(
    query_analysis: QueryAnalysis,
    judged_papers: list[JudgementResult],
    *,
    top_k: int = 20,
    tie_break_policy: TieBreakPolicy = DEFAULT_TIEBREAK_POLICY,
    source_record_refs: Mapping[int, Sequence[str]] | None = None,
) -> list[RankedPaper]:
    """Rerank judged papers using deterministic metadata signals."""

    return RerankerAgent().rerank(
        query_analysis,
        judged_papers,
        top_k=top_k,
        tie_break_policy=tie_break_policy,
        source_record_refs=source_record_refs,
    )


class _ScoredJudgement:
    def __init__(
        self,
        *,
        judgement: JudgementResult,
        original_index: int,
        score_breakdown: RerankScoreBreakdown,
        ranking_reason: str,
    ) -> None:
        self.judgement = judgement
        self.original_index = original_index
        self.score_breakdown = score_breakdown
        self.ranking_reason = ranking_reason


def _score_judgement(
    query_analysis: QueryAnalysis,
    judgement: JudgementResult,
    original_index: int,
) -> _ScoredJudgement:
    paper = judgement.paper
    weights = _weights_for_intent(query_analysis.intent)
    relevance_score = _clamp(judgement.score)
    authority_score = _authority_score(query_analysis, paper)
    timeliness_score = _timeliness_score(query_analysis, paper)
    metadata_score = _metadata_score(paper)
    base_score = (
        relevance_score * weights["relevance"]
        + authority_score * weights["authority"]
        + timeliness_score * weights["timeliness"]
        + metadata_score * weights["metadata"]
    )
    category_multiplier = CATEGORY_MULTIPLIER.get(judgement.category, 0.5)
    final_score = round(
        _clamp(base_score * category_multiplier),
        4,
    )
    breakdown = RerankScoreBreakdown(
        relevance_score=round(relevance_score, 4),
        authority_score=round(authority_score, 4),
        timeliness_score=round(timeliness_score, 4),
        metadata_score=round(metadata_score, 4),
        category_multiplier=category_multiplier,
        final_score=final_score,
        relevance_weight=weights["relevance"],
        authority_weight=weights["authority"],
        timeliness_weight=weights["timeliness"],
        metadata_weight=weights["metadata"],
    )
    return _ScoredJudgement(
        judgement=judgement,
        original_index=original_index,
        score_breakdown=breakdown,
        ranking_reason=_ranking_reason(query_analysis, judgement, breakdown),
    )


def _weights_for_intent(intent: str) -> dict[str, float]:
    if intent == "recent_progress":
        return {
            "relevance": 0.65,
            "authority": 0.08,
            "timeliness": 0.22,
            "metadata": 0.05,
        }
    if intent == "survey":
        return {
            "relevance": 0.62,
            "authority": 0.25,
            "timeliness": 0.08,
            "metadata": 0.05,
        }
    return {
        "relevance": 0.72,
        "authority": 0.13,
        "timeliness": 0.10,
        "metadata": 0.05,
    }


def _authority_score(query_analysis: QueryAnalysis, paper: Paper) -> float:
    citation_component = min(
        math.log10(max(paper.citation_count, 0) + 1) / math.log10(1001),
        1.0,
    ) * 0.65
    source_component = min(len(set(paper.sources)) / 3, 1.0) * 0.15
    identifier_component = min(_identifier_count(paper) / 5, 1.0) * 0.12
    venue_component = _venue_component(query_analysis, paper)
    return _clamp(
        citation_component
        + source_component
        + identifier_component
        + venue_component
    )


def _timeliness_score(query_analysis: QueryAnalysis, paper: Paper) -> float:
    time_range = query_analysis.constraints.time_range
    if time_range is not None:
        if paper.year is None:
            return 0.0
        start_year = time_range.start_year
        end_year = time_range.end_year
        if start_year is not None and paper.year < start_year:
            return 0.0
        if end_year is not None and paper.year > end_year:
            return 0.0
        if start_year is not None and end_year is not None and end_year > start_year:
            position = (paper.year - start_year) / (end_year - start_year)
            return _clamp(0.7 + position * 0.3)
        return 0.85

    if paper.year is None:
        return 0.25

    current_year = date.today().year
    age = max(0, current_year - paper.year)
    return _clamp(1.0 - min(age, 12) / 12)


def _metadata_score(paper: Paper) -> float:
    score = 0.0
    if paper.title.strip():
        score += 0.2
    if paper.abstract.strip():
        score += 0.25
    if paper.year is not None:
        score += 0.15
    if paper.venue and paper.venue.strip():
        score += 0.15
    if paper.sources:
        score += 0.1
    score += min(_identifier_count(paper) / 5, 1.0) * 0.15
    return _clamp(score)


def _venue_component(query_analysis: QueryAnalysis, paper: Paper) -> float:
    venue = (paper.venue or "").strip()
    if not venue:
        return 0.0
    requested = query_analysis.constraints.venues
    if requested:
        venue_key = _normalize_venue(venue)
        if any(_normalize_venue(item) in venue_key for item in requested):
            return 0.12
        return 0.0
    return 0.08


def _normalize_venue(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.casefold())


def _identifier_count(paper: Paper) -> int:
    identifiers = paper.identifiers
    return sum(
        1
        for value in (
            identifiers.doi,
            identifiers.arxiv_id,
            identifiers.semantic_scholar_id,
            identifiers.openalex_id,
            identifiers.pubmed_id,
        )
        if value
    )


def _sort_key(scored: _ScoredJudgement) -> tuple[object, ...]:
    return (*_primary_sort_key(scored), scored.original_index)


def _primary_sort_key(scored: _ScoredJudgement) -> tuple[object, ...]:
    paper = scored.judgement.paper
    return (
        CATEGORY_TIER.get(scored.judgement.category, 5),
        -scored.score_breakdown.final_score,
        -scored.score_breakdown.relevance_score,
        -max(paper.citation_count, 0),
        -(paper.year or 0),
        paper.title.casefold(),
    )


def deterministic_tiebreak_v2_key(
    paper: Paper,
    *,
    source_record_refs: Sequence[str] = (),
) -> str:
    """Build the optional v2 key without input-position or execution state.

    Unified stable identifiers take precedence.  The conservative title,
    author and year identity is used only when all three are available.  If
    neither identity form is available, the caller must supply registered
    field-lineage record references; their raw records are never embedded in
    the key or audit output.
    """

    profile = build_identity_profile(paper)
    if profile.identifiers:
        payload = {
            "kind": "stable_identifiers",
            "identifiers": sorted(profile.identifiers),
        }
        return f"0:{_stable_digest(payload)}"
    if profile.title and profile.authors and profile.year is not None:
        payload = {
            "kind": "exact_title_author_year",
            "title": profile.title,
            "authors": sorted(profile.authors),
            "year": profile.year,
        }
        return f"1:{_stable_digest(payload)}"

    refs = sorted({str(value) for value in source_record_refs if str(value)})
    if not refs or any(not value.startswith("record:") for value in refs):
        raise DeterministicTieBreakUnavailable(
            "registered_source_record_reference_required"
        )
    payload = {
        "kind": "source_record_and_normalized_fields",
        "sources": sorted(
            {
                str(value).strip().casefold()
                for value in paper.sources
                if str(value).strip()
            }
        ),
        "source_record_refs": refs,
        "field_summary_sha256": _stable_digest(_normalized_field_summary(paper)),
    }
    return f"2:{_stable_digest(payload)}"


def deterministic_tiebreak_v2_catalog() -> dict[str, object]:
    """Describe the explicitly enabled v2 key for offline qualification."""

    return {
        "version": DETERMINISTIC_TIEBREAK_V2,
        "default_enabled": False,
        "exact_tie_key": [
            {"field": field, "direction": direction}
            for field, direction in PRODUCTION_RERANK_KEY_FIELDS[:-1]
        ],
        "equality": "exact_python_value_equality_no_float_tolerance",
        "stable_key_precedence": [
            "unified_stable_identifiers",
            "unified_exact_title_author_year",
            "registered_source_record_refs_and_normalized_field_summary_sha256",
        ],
        "fallback_requires_registered_source_record": True,
        "prohibited_inputs": [
            "input_position",
            "completion_order",
            "process_state",
            "gold",
            "quality_result",
            "case_id",
        ],
    }


def trace_deterministic_tiebreak_v2(
    query_analysis: QueryAnalysis,
    judgement: JudgementResult,
    original_index: int,
    *,
    source_record_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Observe the exact primary and v2 keys without changing default rerank."""

    scored = _score_judgement(query_analysis, judgement, original_index)
    stable_key = deterministic_tiebreak_v2_key(
        judgement.paper, source_record_refs=source_record_refs
    )
    return {
        "primary_sort_key": list(_primary_sort_key(scored)),
        "stable_key": stable_key,
        "stable_key_kind": stable_key.split(":", 1)[0],
    }


def _validate_v2_tie_keys(
    scored: Sequence[_ScoredJudgement], stable_keys: Mapping[int, str]
) -> None:
    groups: defaultdict[tuple[object, ...], list[_ScoredJudgement]] = defaultdict(list)
    for item in scored:
        groups[_primary_sort_key(item)].append(item)
    for values in groups.values():
        if len(values) < 2:
            continue
        keys = [stable_keys[item.original_index] for item in values]
        if len(keys) != len(set(keys)):
            raise DeterministicTieBreakUnavailable("stable_key_collision_in_exact_tie")


def _normalized_field_summary(paper: Paper) -> dict[str, object]:
    profile = build_identity_profile(paper)
    return {
        "title": profile.title,
        "authors": sorted(profile.authors),
        "year": profile.year,
        "venue": normalize_title(paper.venue),
        "abstract": _normalize_stable_text(paper.abstract),
        "landing_page": _normalize_stable_text(paper.urls.landing_page),
        "pdf": _normalize_stable_text(paper.urls.pdf),
        "citation_count": int(paper.citation_count),
    }


def _normalize_stable_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _stable_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def production_ranking_decision_catalog() -> dict[str, object]:
    """Describe the existing rerank key and numeric semantics for audits."""

    return {
        "version": "current-rules-rerank-decision-v1",
        "category_tiers": dict(CATEGORY_TIER),
        "category_multipliers": dict(CATEGORY_MULTIPLIER),
        "score_components": [
            "relevance_score",
            "authority_score",
            "timeliness_score",
            "metadata_score",
            "category_multiplier",
            "final_score",
            "relevance_weight",
            "authority_weight",
            "timeliness_weight",
            "metadata_weight",
        ],
        "component_decimals": 4,
        "final_score_decimals": 4,
        "sort_key": [
            {"field": field, "direction": direction}
            for field, direction in PRODUCTION_RERANK_KEY_FIELDS
        ],
        "missing_values": {
            "citation_count": "max(value, 0)",
            "year": "zero",
            "title": "Paper schema requires a string; casefold is exact",
        },
    }


def trace_ranking_decision(
    query_analysis: QueryAnalysis,
    judgement: JudgementResult,
    original_index: int,
) -> dict[str, object]:
    """Observe the exact existing score and sort key for one candidate."""

    scored = _score_judgement(query_analysis, judgement, original_index)
    return {
        "score_breakdown": scored.score_breakdown.model_dump(mode="json"),
        "sort_key": list(_sort_key(scored)),
        "tie_key": list(_sort_key(scored)[:-1]),
        "original_index": original_index,
    }


def _to_ranked_paper(rank: int, scored: _ScoredJudgement) -> RankedPaper:
    judgement = scored.judgement
    return RankedPaper(
        rank=rank,
        paper=judgement.paper,
        final_score=scored.score_breakdown.final_score,
        category=judgement.category,
        score_breakdown=scored.score_breakdown,
        ranking_reason=scored.ranking_reason,
        evidence=judgement.evidence,
        matched_terms=judgement.matched_terms,
        warnings=judgement.warnings,
    )


def _ranking_reason(
    query_analysis: QueryAnalysis,
    judgement: JudgementResult,
    breakdown: RerankScoreBreakdown,
) -> str:
    paper = judgement.paper
    details = [
        f"category={judgement.category}",
        f"judgement_score={breakdown.relevance_score:.4f}",
        f"citation_count={max(paper.citation_count, 0)}",
        f"sources={len(set(paper.sources))}",
        f"identifiers={_identifier_count(paper)}",
    ]
    if paper.venue:
        details.append(f"venue={paper.venue}")
    if paper.year is not None:
        details.append(f"year={paper.year}")
    if query_analysis.intent == "recent_progress":
        details.append("recent_progress weights timeliness more heavily")
    if query_analysis.intent == "survey":
        details.append("survey weights authority more heavily")
    details.append(
        "final_score combines relevance, authority, timeliness, and metadata"
    )
    return "; ".join(details)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
