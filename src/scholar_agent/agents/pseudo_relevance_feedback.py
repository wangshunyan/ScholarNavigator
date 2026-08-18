"""Deterministic pseudo-relevance feedback for the fixed-budget query plan."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from scholar_agent.agents.query_evolution import STOPWORDS
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.search_schemas import (
    PRFFeedbackTerm,
    PRFSeedCandidate,
    RankedPaper,
    SearchSubquery,
)
from scholar_agent.retrieval.query_adapter import (
    ACADEMIC_QUERY_STOPWORDS,
    tokenize_academic_text,
)


PRF_MAX_SEEDS = 5
PRF_MIN_DOCUMENT_FREQUENCY = 2
PRF_MAX_TERMS = 6

_URL_PATTERN = re.compile(r"(?i)\b(?:https?|ftp)://\S+|\bwww\.\S+")
_DOI_PATTERN = re.compile(r"(?i)\b10\.\d{4,9}/\S+")
_ARXIV_PATTERN = re.compile(
    r"(?i)\b(?:arxiv\s*:\s*)?(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?\b"
)
_YEAR_PATTERN = re.compile(r"^(?:18|19|20|21|22)\d{2}$")
_IDENTIFIER_WORDS = frozenset(
    {"arxiv", "doi", "pmid", "pubmed", "openalex", "corpusid", "s2orc"}
)
_STOPWORDS = frozenset(
    {*(item.casefold() for item in ACADEMIC_QUERY_STOPWORDS), *STOPWORDS}
)


@dataclass(frozen=True)
class PRFPlanOutcome:
    subqueries: list[SearchSubquery]
    seeds: list[PRFSeedCandidate]
    feedback_terms: list[PRFFeedbackTerm]
    query: str | None
    replaced_index: int | None
    replaced_query: str | None
    replaced_purpose: str | None
    skip_reason: str | None
    fallback_used: bool


def build_prf_plan(
    original_query: str,
    current_subqueries: list[SearchSubquery],
    ranked_papers: list[RankedPaper],
    *,
    first_round_succeeded: bool,
) -> PRFPlanOutcome:
    """Replace the lowest-priority derived query without increasing plan size."""

    selected = [item.model_copy(deep=True) for item in current_subqueries]
    unique_ranked = _unique_ranked_papers(ranked_papers)
    seeds = [
        PRFSeedCandidate(rank=index, title=item.paper.title)
        for index, item in enumerate(unique_ranked[:PRF_MAX_SEEDS], start=1)
    ]
    if not first_round_succeeded:
        return _skipped(selected, seeds, "first_round_failed")
    if not seeds:
        return _skipped(selected, seeds, "no_seed_candidates")

    replaceable = [
        (item.priority, index)
        for index, item in enumerate(selected)
        if item.purpose != "original_query"
    ]
    if not replaceable:
        return _skipped(selected, seeds, "no_derived_query_to_replace")

    feedback_terms = extract_prf_feedback(original_query, unique_ranked)
    if not feedback_terms:
        return _skipped(selected, seeds, "no_eligible_feedback_terms")
    query = " ".join(
        [original_query.strip(), *(item.term for item in feedback_terms)]
    ).strip()
    normalized_query = _query_key(query)
    if not normalized_query or normalized_query == _query_key(original_query):
        return _skipped(selected, seeds, "feedback_query_equivalent_to_original")
    if any(_query_key(item.query) == normalized_query for item in selected):
        return _skipped(selected, seeds, "feedback_query_duplicates_existing")

    _, replace_index = max(replaceable)
    replaced = selected[replace_index]
    selected[replace_index] = SearchSubquery(
        query=query,
        combination_mode=replaced.combination_mode,
        source_hints=list(replaced.source_hints),
        priority=replaced.priority,
        purpose="prf_v1",
        facet_types=list(replaced.facet_types),
        provenance=[
            "prf_v1:top5_unique_title_abstract",
            "prf_v1:min_document_frequency_2",
            "prf_v1:rank_discount_reciprocal",
            f"replaced:{replaced.purpose}",
        ],
    )
    return PRFPlanOutcome(
        subqueries=selected,
        seeds=seeds,
        feedback_terms=feedback_terms,
        query=query,
        replaced_index=replace_index,
        replaced_query=replaced.query,
        replaced_purpose=replaced.purpose,
        skip_reason=None,
        fallback_used=False,
    )


def extract_prf_feedback(
    original_query: str,
    ranked_papers: list[RankedPaper],
) -> list[PRFFeedbackTerm]:
    """Extract deterministic unigram/bigram feedback from at most five seeds."""

    query_terms = {
        token.casefold()
        for token in tokenize_academic_text(_strip_identifiers(original_query))
    }
    document_counts: list[Counter[str]] = []
    ngram_sizes: dict[str, int] = {}
    for ranked in _unique_ranked_papers(ranked_papers)[:PRF_MAX_SEEDS]:
        tokens = _feedback_tokens(
            f"{ranked.paper.title}\n{ranked.paper.abstract}",
            query_terms=query_terms,
        )
        counts: Counter[str] = Counter(token for token in tokens if token is not None)
        for left, right in zip(tokens, tokens[1:]):
            if left is None or right is None:
                continue
            counts[f"{left} {right}"] += 1
        for term in counts:
            ngram_sizes[term] = 2 if " " in term else 1
        document_counts.append(counts)

    document_frequency: Counter[str] = Counter()
    total_frequency: Counter[str] = Counter()
    discounted_frequency: Counter[str] = Counter()
    for rank, counts in enumerate(document_counts, start=1):
        for term, count in counts.items():
            document_frequency[term] += 1
            total_frequency[term] += count
            discounted_frequency[term] += count / rank

    eligible = [
        term
        for term, count in document_frequency.items()
        if count >= PRF_MIN_DOCUMENT_FREQUENCY
    ]
    eligible.sort(
        key=lambda term: (
            -discounted_frequency[term],
            -document_frequency[term],
            -total_frequency[term],
            ngram_sizes[term],
            term,
        )
    )
    return [
        PRFFeedbackTerm(
            term=term,
            ngram_size=ngram_sizes[term],
            document_frequency=document_frequency[term],
            term_frequency=total_frequency[term],
            rank_discounted_frequency=round(discounted_frequency[term], 12),
        )
        for term in eligible[:PRF_MAX_TERMS]
    ]


def _unique_ranked_papers(ranked_papers: list[RankedPaper]) -> list[RankedPaper]:
    """Keep the first ranked occurrence of each unified paper identity."""

    selected: list[RankedPaper] = []
    profiles: list[IdentityProfile] = []
    for ranked in ranked_papers:
        profile = build_identity_profile(ranked.paper)
        if any(
            identity_evidence_from_profiles(existing, profile).equivalent
            for existing in profiles
        ):
            continue
        selected.append(ranked)
        profiles.append(profile)
    return selected


def _feedback_tokens(value: str, *, query_terms: set[str]) -> list[str | None]:
    tokens: list[str | None] = []
    for raw in tokenize_academic_text(_strip_identifiers(value)):
        token = unicodedata.normalize("NFKC", raw).casefold().strip("._-+#")
        if (
            not token
            or token in query_terms
            or token in _STOPWORDS
            or token in _IDENTIFIER_WORDS
            or token.isdigit()
            or _YEAR_PATTERN.fullmatch(token)
        ):
            tokens.append(None)
            continue
        tokens.append(token)
    return tokens


def _strip_identifiers(value: str) -> str:
    without_urls = _URL_PATTERN.sub(" ", unicodedata.normalize("NFKC", value))
    without_doi = _DOI_PATTERN.sub(" ", without_urls)
    return _ARXIV_PATTERN.sub(" ", without_doi)


def _query_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _skipped(
    subqueries: list[SearchSubquery],
    seeds: list[PRFSeedCandidate],
    reason: str,
) -> PRFPlanOutcome:
    return PRFPlanOutcome(
        subqueries=subqueries,
        seeds=seeds,
        feedback_terms=[],
        query=None,
        replaced_index=None,
        replaced_query=None,
        replaced_purpose=None,
        skip_reason=reason,
        fallback_used=True,
    )
