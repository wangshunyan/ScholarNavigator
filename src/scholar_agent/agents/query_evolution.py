"""基于初始结果覆盖缺口的确定性查询演化。"""

from __future__ import annotations

import re
from collections import Counter

from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    DEFAULT_SEARCH_SOURCES,
    EvolvedSubquery,
    JudgementResult,
    QueryAnalysis,
    QueryCoverageGap,
    QueryEvolutionOptions,
    QueryEvolutionPolicy,
    QueryEvolutionQualityGate,
    QueryEvolutionRecord,
    RankedPaper,
    SearchPlan,
    SUPPORTED_SEARCH_SOURCES,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "latest",
    "method",
    "methods",
    "of",
    "on",
    "paper",
    "papers",
    "recent",
    "research",
    "review",
    "scientific",
    "search",
    "study",
    "studies",
    "survey",
    "the",
    "to",
    "using",
    "with",
}
GAP_QUERY_BOILERPLATE = {
    "about",
    "can",
    "could",
    "find",
    "give",
    "list",
    "looking",
    "me",
    "please",
    "show",
    "some",
    "tell",
    "want",
    "you",
}
BROAD_BACKGROUND_TERMS = {
    "analysis",
    "approach",
    "background",
    "evaluation",
    "framework",
    "literature",
    "model",
    "models",
    "paper",
    "papers",
    "research",
    "system",
    "systems",
}
PAPER_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "survey": ("survey",),
    "review": ("review", "literature review", "systematic review"),
    "method": ("method", "methodology", "approach", "framework"),
    "benchmark": ("benchmark", "evaluation"),
    "dataset": ("dataset", "data set", "corpus"),
    "application": ("application", "applied", "deployment"),
    "comparison": ("comparison", "comparative", "versus"),
}


class QueryEvolutionAgent:
    """保留旧 seed 扩展，并提供覆盖缺口驱动的受控策略。"""

    def evolve(
        self,
        query_analysis: QueryAnalysis,
        search_plan: SearchPlan,
        judgements: list[JudgementResult],
        ranked_papers: list[RankedPaper],
        used_queries: set[str],
        options: QueryEvolutionOptions | None = None,
    ) -> QueryEvolutionRecord:
        options = options or QueryEvolutionOptions()
        if options.policy == "off":
            return QueryEvolutionRecord(
                policy="off",
                skipped_reasons=["disabled"],
                warnings=["query_evolution_disabled"],
            )
        if options.policy == "seed_expansion":
            return _evolve_seed_expansion(
                query_analysis,
                search_plan,
                judgements,
                ranked_papers,
                used_queries,
                options,
            )
        return _evolve_coverage_gap(
            query_analysis,
            search_plan,
            judgements,
            ranked_papers,
            used_queries,
            options,
        )


def evolve_queries(
    query_analysis: QueryAnalysis,
    search_plan: SearchPlan,
    judgements: list[JudgementResult],
    ranked_papers: list[RankedPaper],
    used_queries: set[str],
    options: QueryEvolutionOptions | None = None,
) -> QueryEvolutionRecord:
    """不使用 LLM、网络或评测答案生成演化查询。"""

    return QueryEvolutionAgent().evolve(
        query_analysis=query_analysis,
        search_plan=search_plan,
        judgements=judgements,
        ranked_papers=ranked_papers,
        used_queries=used_queries,
        options=options,
    )


def analyze_query_coverage(
    query_analysis: QueryAnalysis,
    judgements: list[JudgementResult],
) -> QueryCoverageGap:
    """只根据查询、约束和初始规则判断结果计算覆盖缺口。"""

    relevant = [
        result
        for result in judgements
        if result.category in {"highly_relevant", "partially_relevant"}
    ]
    papers = [result.paper for result in relevant]
    constraints = query_analysis.constraints
    topic_terms = _topic_terms(query_analysis, coverage_gap=True)
    required_must = _required_must_terms(query_analysis)

    topic_coverage, missing_topics = _term_coverage(topic_terms, papers)
    method_coverage, missing_methods = _term_coverage(
        constraints.methods,
        papers,
    )
    dataset_coverage, missing_datasets = _term_coverage(
        constraints.datasets,
        papers,
    )
    must_coverage, missing_must = _term_coverage(required_must, papers)
    paper_type_coverage, missing_paper_types = _paper_type_coverage(
        constraints.paper_types,
        papers,
    )
    venue_coverage, missing_venues = _venue_coverage(
        constraints.venues,
        papers,
    )
    temporal_coverage = _temporal_coverage(query_analysis, papers)

    reasons: list[str] = []
    if not relevant:
        reasons.append("no_relevant_initial_results")
    if missing_methods:
        reasons.append("method_gap")
    if missing_datasets:
        reasons.append("dataset_gap")
    if missing_must:
        reasons.append("must_have_gap")
    if missing_paper_types:
        reasons.append("paper_type_gap")
    if missing_venues:
        reasons.append("venue_gap")
    if query_analysis.constraints.time_range is not None and temporal_coverage == 0.0:
        reasons.append("temporal_gap")

    actionable_topics = [
        term
        for term in missing_topics
        if term.casefold() not in BROAD_BACKGROUND_TERMS
    ]
    if actionable_topics and len(topic_terms) >= 2:
        reasons.append("topic_gap")

    actionable = bool(
        missing_methods
        or missing_datasets
        or missing_must
        or missing_paper_types
        or actionable_topics
    )
    if not actionable:
        reasons.append(
            "coverage_sufficient" if relevant else "no_actionable_gap"
        )
    return QueryCoverageGap(
        topic_coverage=topic_coverage,
        method_coverage=method_coverage,
        dataset_coverage=dataset_coverage,
        must_have_coverage=must_coverage,
        paper_type_coverage=paper_type_coverage,
        venue_coverage=venue_coverage,
        temporal_coverage=temporal_coverage,
        missing_topics=actionable_topics,
        missing_methods=missing_methods,
        missing_datasets=missing_datasets,
        missing_must_have_terms=missing_must,
        missing_paper_types=missing_paper_types,
        missing_venues=missing_venues,
        needs_evolution=actionable,
        reasons=_dedupe(reasons),
    )


def filter_evolved_candidates(
    query_analysis: QueryAnalysis,
    initial_papers: list[Paper],
    evolved_papers: list[Paper],
) -> tuple[list[Paper], QueryEvolutionQualityGate]:
    """复用同一组查询维度做低成本预过滤，不使用 gold。"""

    unique = deduplicate_papers(evolved_papers)
    initial_keys = {_paper_key(paper) for paper in initial_papers}
    topic_terms = _topic_terms(query_analysis, coverage_gap=True)
    structured_terms = _dedupe(
        query_analysis.constraints.methods
        + query_analysis.constraints.datasets
        + _required_must_terms(query_analysis)
    )
    excluded_terms = query_analysis.constraints.exclude_terms
    accepted: list[Paper] = []
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    accepted_source_counts: Counter[str] = Counter()
    duplicate_count = 0
    core_match_count = 0
    for paper in unique:
        source_counts.update(paper.sources or ["unknown"])
        if _paper_key(paper) in initial_keys:
            duplicate_count += 1
            reason_counts["duplicate_with_initial"] += 1
            continue
        text = _paper_text(paper)
        if not (paper.title.strip() or paper.abstract.strip()):
            reason_counts["missing_metadata"] += 1
            continue
        if any(_term_present(term, text) for term in excluded_terms):
            reason_counts["excluded_term_match"] += 1
            continue
        if topic_terms and not any(
            _term_present(term, text) for term in topic_terms
        ):
            reason_counts["no_topic_match"] += 1
            continue
        if structured_terms and not any(
            _term_present(term, text) for term in structured_terms
        ):
            reason_counts["no_structured_dimension_match"] += 1
            continue
        core_match_count += 1
        accepted.append(paper)
        accepted_source_counts.update(paper.sources or ["unknown"])
    return accepted, QueryEvolutionQualityGate(
        raw_candidate_count=len(evolved_papers),
        unique_candidate_count=len(unique),
        duplicate_candidate_count=len(evolved_papers) - len(unique),
        duplicate_with_initial_count=duplicate_count,
        accepted_candidate_count=len(accepted),
        filtered_candidate_count=len(evolved_papers) - len(accepted),
        core_dimension_match_count=core_match_count,
        filtered_reason_counts=dict(sorted(reason_counts.items())),
        source_candidate_counts=dict(sorted(source_counts.items())),
        accepted_source_counts=dict(sorted(accepted_source_counts.items())),
    )


class _Seed:
    def __init__(
        self,
        paper: Paper,
        score: float,
        matched_terms: list[str],
        category: str,
    ) -> None:
        self.paper = paper
        self.score = score
        self.matched_terms = matched_terms
        self.category = category


class _Candidate:
    def __init__(
        self,
        query: str,
        purpose: str,
        seed_titles: list[str],
        *,
        policy: QueryEvolutionPolicy,
        gap_dimensions: list[str] | None = None,
    ) -> None:
        self.query = query
        self.purpose = purpose
        self.seed_titles = seed_titles
        self.policy = policy
        self.gap_dimensions = gap_dimensions or []


def _evolve_seed_expansion(
    query_analysis: QueryAnalysis,
    search_plan: SearchPlan,
    judgements: list[JudgementResult],
    ranked_papers: list[RankedPaper],
    used_queries: set[str],
    options: QueryEvolutionOptions,
) -> QueryEvolutionRecord:
    max_queries = 3 if options.max_evolved_queries is None else options.max_evolved_queries
    max_seeds = 5 if options.max_seed_papers is None else options.max_seed_papers
    if max_queries == 0:
        return QueryEvolutionRecord(
            policy="seed_expansion",
            skipped_reasons=["max_evolved_queries_zero"],
            warnings=["max_evolved_queries_zero"],
        )
    seeds = _select_seed_papers(
        judgements=judgements,
        ranked_papers=ranked_papers,
        max_seed_papers=max_seeds,
        min_seed_score=options.min_seed_score,
    )
    eligible = _eligible_seeds(judgements, options.min_seed_score)
    if not seeds:
        return QueryEvolutionRecord(
            policy="seed_expansion",
            eligible_seed_count=len(eligible),
            eligible_seed_titles=_seed_titles(eligible),
            skipped_reasons=["no_relevant_seed"],
            warnings=["no_relevant_seed"],
        )
    generated = _materialize_candidates(
        _seed_expansion_candidates(query_analysis, seeds),
        search_plan,
        used_queries,
        max_queries=max_queries,
    )
    skipped = [] if generated else ["duplicate_query"]
    return QueryEvolutionRecord(
        policy="seed_expansion",
        eligible_seed_count=len(eligible),
        eligible_seed_titles=_seed_titles(eligible),
        seed_count=len(seeds),
        seed_paper_titles=_seed_titles(seeds),
        generated_queries=generated,
        skipped_reasons=skipped,
        warnings=[] if generated else list(skipped),
    )


def _evolve_coverage_gap(
    query_analysis: QueryAnalysis,
    search_plan: SearchPlan,
    judgements: list[JudgementResult],
    ranked_papers: list[RankedPaper],
    used_queries: set[str],
    options: QueryEvolutionOptions,
) -> QueryEvolutionRecord:
    max_queries = min(
        2,
        2 if options.max_evolved_queries is None else options.max_evolved_queries,
    )
    max_seeds = min(
        3,
        3 if options.max_seed_papers is None else options.max_seed_papers,
    )
    gap = analyze_query_coverage(query_analysis, judgements)
    eligible = _eligible_seeds(judgements, options.min_seed_score)
    if not gap.needs_evolution:
        reason = (
            "coverage_sufficient"
            if "coverage_sufficient" in gap.reasons
            else "no_actionable_gap"
        )
        return QueryEvolutionRecord(
            policy="coverage_gap",
            eligible_seed_count=len(eligible),
            eligible_seed_titles=_seed_titles(eligible),
            coverage_gap=gap,
            skipped_reasons=[reason],
        )
    if max_queries == 0:
        return QueryEvolutionRecord(
            policy="coverage_gap",
            eligible_seed_count=len(eligible),
            eligible_seed_titles=_seed_titles(eligible),
            coverage_gap=gap,
            skipped_reasons=["budget_stop"],
            warnings=["max_evolved_queries_zero"],
        )
    seeds = _select_gap_seeds(
        query_analysis,
        eligible,
        ranked_papers,
        max_seed_papers=max_seeds,
    )
    if not seeds:
        return QueryEvolutionRecord(
            policy="coverage_gap",
            eligible_seed_count=len(eligible),
            eligible_seed_titles=_seed_titles(eligible),
            coverage_gap=gap,
            skipped_reasons=["no_reliable_seed"],
            warnings=["no_reliable_seed"],
        )
    candidates, generation_skips = _coverage_gap_candidates(
        query_analysis,
        gap,
        seeds,
    )
    generated = _materialize_candidates(
        candidates,
        search_plan,
        used_queries,
        max_queries=max_queries,
    )
    skipped = list(generation_skips)
    if not generated:
        skipped.append(
            "duplicate_query" if candidates else "no_actionable_gap"
        )
    return QueryEvolutionRecord(
        policy="coverage_gap",
        eligible_seed_count=len(eligible),
        eligible_seed_titles=_seed_titles(eligible),
        seed_count=len(seeds),
        seed_paper_titles=_seed_titles(seeds),
        coverage_gap=gap,
        generated_queries=generated,
        skipped_reasons=_dedupe(skipped),
        warnings=[] if generated else _dedupe(skipped),
    )


def _eligible_seeds(
    judgements: list[JudgementResult],
    min_seed_score: float,
) -> list[_Seed]:
    return [
        _Seed(
            result.paper,
            result.score,
            list(result.matched_terms),
            result.category,
        )
        for result in judgements
        if _is_seed_category(result.category, result.score, min_seed_score)
        and (result.paper.title.strip() or result.paper.abstract.strip())
    ]


def _select_seed_papers(
    *,
    judgements: list[JudgementResult],
    ranked_papers: list[RankedPaper],
    max_seed_papers: int,
    min_seed_score: float,
) -> list[_Seed]:
    if max_seed_papers == 0:
        return []
    judgement_by_key = {_paper_key(item.paper): item for item in judgements}
    seeds: list[_Seed] = []
    seen: set[str] = set()
    for ranked in ranked_papers:
        key = _paper_key(ranked.paper)
        judgement = judgement_by_key.get(key)
        score = judgement.score if judgement is not None else ranked.final_score
        category = judgement.category if judgement is not None else ranked.category
        if not _is_seed_category(category, score, min_seed_score):
            continue
        seeds.append(
            _Seed(
                ranked.paper,
                score,
                list(ranked.matched_terms),
                category,
            )
        )
        seen.add(key)
        if len(seeds) >= max_seed_papers:
            return seeds
    for judgement in judgements:
        key = _paper_key(judgement.paper)
        if key in seen or not _is_seed_category(
            judgement.category,
            judgement.score,
            min_seed_score,
        ):
            continue
        seeds.append(
            _Seed(
                judgement.paper,
                judgement.score,
                list(judgement.matched_terms),
                judgement.category,
            )
        )
        if len(seeds) >= max_seed_papers:
            break
    return seeds


def _select_gap_seeds(
    query_analysis: QueryAnalysis,
    eligible: list[_Seed],
    ranked_papers: list[RankedPaper],
    *,
    max_seed_papers: int,
) -> list[_Seed]:
    if max_seed_papers == 0:
        return []
    by_key = {_paper_key(seed.paper): seed for seed in eligible}
    ordered: list[_Seed] = []
    for ranked in ranked_papers:
        seed = by_key.pop(_paper_key(ranked.paper), None)
        if seed is not None:
            ordered.append(seed)
    ordered.extend(by_key.values())
    core_terms = _dedupe(
        _topic_terms(query_analysis, coverage_gap=True)
        + query_analysis.constraints.methods
        + query_analysis.constraints.datasets
        + _required_must_terms(query_analysis)
    )
    ordered.sort(
        key=lambda seed: (
            0 if seed.category == "highly_relevant" else 1,
            0 if _paper_matches_any(seed.paper, core_terms) else 1,
            0 if seed.paper.title.strip() and seed.paper.abstract.strip() else 1,
            -seed.score,
            _paper_key(seed.paper),
        )
    )
    selected: list[_Seed] = []
    for seed in ordered:
        if seed.category == "partially_relevant" and not _paper_matches_any(
            seed.paper,
            core_terms,
        ):
            continue
        if any(_near_duplicate_title(seed.paper.title, item.paper.title) for item in selected):
            continue
        selected.append(seed)
        if len(selected) >= max_seed_papers:
            break
    return selected


def _is_seed_category(category: str, score: float, min_seed_score: float) -> bool:
    if category == "highly_relevant":
        return True
    return category == "partially_relevant" and score >= min_seed_score


def _seed_expansion_candidates(
    query_analysis: QueryAnalysis,
    seeds: list[_Seed],
) -> list[_Candidate]:
    core_terms = _core_terms(query_analysis, seeds)
    if not core_terms:
        core_terms = _tokenize_text(query_analysis.original_query)
    seed_titles = _seed_titles(seeds)
    candidates: list[_Candidate] = []
    base_query = _join_terms(core_terms[:6])
    if base_query:
        candidates.append(
            _Candidate(
                _intent_query(query_analysis.intent, base_query),
                f"query_evolution_{query_analysis.intent}",
                seed_titles[:3],
                policy="seed_expansion",
            )
        )
    method_terms = _dedupe(query_analysis.constraints.methods + _matched_terms(seeds))
    method_query = _join_terms((method_terms + core_terms)[:6])
    if method_query:
        candidates.append(
            _Candidate(
                _method_query(query_analysis.intent, method_query),
                "query_evolution_matched_methods",
                seed_titles[:3],
                policy="seed_expansion",
            )
        )
    for seed in seeds:
        terms = _dedupe(
            seed.matched_terms
            + _tokenize_text(seed.paper.title)
            + core_terms
        )
        query = _join_terms(terms[:7])
        if query:
            candidates.append(
                _Candidate(
                    query,
                    "query_evolution_from_seed_title",
                    [seed.paper.title] if seed.paper.title.strip() else [],
                    policy="seed_expansion",
                )
            )
    return candidates


def _coverage_gap_candidates(
    query_analysis: QueryAnalysis,
    gap: QueryCoverageGap,
    seeds: list[_Seed],
) -> tuple[list[_Candidate], list[str]]:
    topics = _topic_terms(query_analysis, coverage_gap=True)
    stable_topics = [
        term for term in topics if term.casefold() not in BROAD_BACKGROUND_TERMS
    ]
    if not stable_topics:
        return [], ["no_actionable_gap"]
    base = stable_topics[:3]
    required_must = _required_must_terms(query_analysis)
    retained = _dedupe(required_must + query_analysis.constraints.venues[:1])
    time_terms = _time_terms(query_analysis)
    seed_titles = _seed_titles(seeds)
    gap_terms: list[tuple[str, str]] = []
    gap_terms.extend(("method", term) for term in gap.missing_methods)
    gap_terms.extend(("dataset", term) for term in gap.missing_datasets)
    gap_terms.extend(("must_have", term) for term in gap.missing_must_have_terms)
    gap_terms.extend(("topic", term) for term in gap.missing_topics)
    gap_terms.extend(("paper_type", term) for term in gap.missing_paper_types)
    candidates: list[_Candidate] = []
    skips: list[str] = []
    for dimension, term in gap_terms:
        query = _join_terms(base + [term] + retained + time_terms)
        if not query:
            continue
        if not _retains_required_information(query, base, required_must):
            skips.append("low_information_retention")
            continue
        candidates.append(
            _Candidate(
                query,
                f"query_evolution_coverage_gap_{dimension}",
                seed_titles,
                policy="coverage_gap",
                gap_dimensions=[dimension],
            )
        )
    return candidates, _dedupe(skips)


def _materialize_candidates(
    candidates: list[_Candidate],
    search_plan: SearchPlan,
    used_queries: set[str],
    *,
    max_queries: int,
) -> list[EvolvedSubquery]:
    blocked = _used_query_keys(search_plan, used_queries)
    source_hints = _safe_source_hints(search_plan)
    generated: list[EvolvedSubquery] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_query(candidate.query)
        key = _query_key(normalized)
        if not normalized or key in blocked or key in seen:
            continue
        generated.append(
            EvolvedSubquery(
                query=normalized,
                source_hints=source_hints,
                priority=min(len(generated) + 1, 5),
                purpose=candidate.purpose,
                seed_paper_titles=candidate.seed_titles,
                generated_by="rules",
                generation_policy=candidate.policy,
                gap_dimensions=candidate.gap_dimensions,
            )
        )
        seen.add(key)
        if len(generated) >= max_queries:
            break
    return generated


def _topic_terms(
    query_analysis: QueryAnalysis,
    *,
    coverage_gap: bool = False,
) -> list[str]:
    ignored = STOPWORDS | (GAP_QUERY_BOILERPLATE if coverage_gap else set())
    constrained = _dedupe(query_analysis.constraints.must_include_terms)
    structural = {
        term.casefold()
        for term in (
            query_analysis.constraints.methods
            + query_analysis.constraints.datasets
            + query_analysis.constraints.paper_types
            + query_analysis.constraints.venues
        )
    }
    terms = [
        term
        for term in constrained
        if term.casefold() not in structural
        and term.casefold() not in ignored
    ]
    return _dedupe(
        terms
        + _tokenize_text(
            query_analysis.original_query,
            extra_stopwords=GAP_QUERY_BOILERPLATE if coverage_gap else None,
        )
    )


def _required_must_terms(query_analysis: QueryAnalysis) -> list[str]:
    constraints = query_analysis.constraints
    if "must_include_terms" not in constraints.explicit_fields:
        return []
    return list(constraints.must_include_terms)


def _term_coverage(
    terms: list[str],
    papers: list[Paper],
) -> tuple[float, list[str]]:
    expected = _dedupe(terms)
    if not expected:
        return 1.0, []
    matched = {
        term.casefold()
        for term in expected
        if any(_term_present(term, _paper_text(paper)) for paper in papers)
    }
    missing = [term for term in expected if term.casefold() not in matched]
    return (len(expected) - len(missing)) / len(expected), missing


def _paper_type_coverage(
    paper_types: list[str],
    papers: list[Paper],
) -> tuple[float, list[str]]:
    expected = _dedupe(paper_types)
    if not expected:
        return 1.0, []
    missing = [
        paper_type
        for paper_type in expected
        if not any(
            any(
                _term_present(alias, _paper_text(paper))
                for alias in PAPER_TYPE_TERMS.get(paper_type, (paper_type,))
            )
            for paper in papers
        )
    ]
    return (len(expected) - len(missing)) / len(expected), missing


def _venue_coverage(
    venues: list[str],
    papers: list[Paper],
) -> tuple[float, list[str]]:
    expected = _dedupe(venues)
    if not expected:
        return 1.0, []
    missing = [
        venue
        for venue in expected
        if not any(
            _normalized_venue(venue) in _normalized_venue(paper.venue or "")
            for paper in papers
            if paper.venue
        )
    ]
    return (len(expected) - len(missing)) / len(expected), missing


def _temporal_coverage(
    query_analysis: QueryAnalysis,
    papers: list[Paper],
) -> float:
    time_range = query_analysis.constraints.time_range
    if time_range is None:
        return 1.0
    if not papers:
        return 0.0
    matching = sum(
        paper.year is not None
        and (time_range.start_year is None or paper.year >= time_range.start_year)
        and (time_range.end_year is None or paper.year <= time_range.end_year)
        for paper in papers
    )
    return matching / len(papers)


def _time_terms(query_analysis: QueryAnalysis) -> list[str]:
    value = query_analysis.constraints.time_range
    if value is None:
        return []
    if value.start_year is not None and value.end_year is not None:
        return [f"{value.start_year}-{value.end_year}"]
    if value.start_year is not None:
        return [f"since {value.start_year}"]
    if value.end_year is not None:
        return [f"before {value.end_year}"]
    return []


def _retains_required_information(
    query: str,
    core_topics: list[str],
    required_must: list[str],
) -> bool:
    topic_matches = sum(_term_present(term, query) for term in core_topics)
    topic_ratio = topic_matches / len(core_topics) if core_topics else 0.0
    return topic_ratio >= 0.5 and all(
        _term_present(term, query) for term in required_must
    )


def _near_duplicate_title(left: str, right: str) -> bool:
    left_terms = set(_tokenize_text(left))
    right_terms = set(_tokenize_text(right))
    if not left_terms or not right_terms:
        return False
    return len(left_terms & right_terms) / len(left_terms | right_terms) >= 0.8


def _paper_matches_any(paper: Paper, terms: list[str]) -> bool:
    text = _paper_text(paper)
    return any(_term_present(term, text) for term in terms)


def _paper_text(paper: Paper) -> str:
    return " ".join(
        value for value in (paper.title, paper.abstract, paper.venue or "") if value
    )


def _term_present(term: str, text: str) -> bool:
    needle = _normalize_match_text(term)
    haystack = _normalize_match_text(text)
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9+.#-]+", needle):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack))
    return needle in haystack


def _normalize_match_text(value: str) -> str:
    return " ".join(
        re.sub(r"[^\w+.#-]+", " ", value.casefold(), flags=re.UNICODE).split()
    )


def _normalized_venue(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _core_terms(query_analysis: QueryAnalysis, seeds: list[_Seed]) -> list[str]:
    constraints = query_analysis.constraints
    return _dedupe(
        constraints.must_include_terms
        + constraints.methods
        + constraints.datasets
        + _matched_terms(seeds)
    )


def _matched_terms(seeds: list[_Seed]) -> list[str]:
    terms: list[str] = []
    for seed in seeds:
        terms.extend(seed.matched_terms)
    return _dedupe(terms)


def _seed_titles(seeds: list[_Seed]) -> list[str]:
    return [seed.paper.title for seed in seeds if seed.paper.title.strip()]


def _intent_query(intent: str, base_query: str) -> str:
    if intent == "survey":
        return f"{base_query} survey review"
    if intent == "recent_progress":
        return f"{base_query} recent advances"
    if intent == "method_comparison":
        return f"{base_query} comparison benchmark"
    if intent == "benchmark_or_dataset":
        return f"{base_query} dataset benchmark evaluation"
    if intent == "application":
        return f"{base_query} application deployment"
    return f"{base_query} representative papers"


def _method_query(intent: str, method_query: str) -> str:
    if intent == "benchmark_or_dataset":
        return f"{method_query} benchmark"
    if intent == "method_comparison":
        return f"{method_query} comparison"
    return method_query


def _used_query_keys(search_plan: SearchPlan, used_queries: set[str]) -> set[str]:
    keys = {_query_key(query) for query in used_queries}
    keys.add(_query_key(search_plan.query_analysis.original_query))
    for subquery in search_plan.subqueries:
        keys.add(_query_key(subquery.query))
    return {key for key in keys if key}


def _safe_source_hints(search_plan: SearchPlan) -> list[str]:
    supported = set(SUPPORTED_SEARCH_SOURCES)
    sources = [source for source in search_plan.selected_sources if source in supported]
    return sources or list(DEFAULT_SEARCH_SOURCES)


def _paper_key(paper: Paper) -> str:
    identifiers = paper.identifiers
    for prefix, value in (
        ("doi", identifiers.doi),
        ("arxiv", identifiers.arxiv_id),
        ("openalex", identifiers.openalex_id),
        ("semantic", identifiers.semantic_scholar_id),
        ("pubmed", identifiers.pubmed_id),
    ):
        if value:
            return f"{prefix}:{value.casefold()}"
    return f"title:{_query_key(paper.title)}"


def _tokenize_text(
    text: str,
    *,
    extra_stopwords: set[str] | None = None,
) -> list[str]:
    ignored = STOPWORDS | (extra_stopwords or set())
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", text):
        item = token.strip(".,;:()[]{}")
        key = item.casefold()
        if not item or key in ignored:
            continue
        if len(key) <= 2 and key not in {"ai", "ml", "cv"}:
            continue
        terms.append(_canonical_term(item))
    return _dedupe(terms)


def _canonical_term(term: str) -> str:
    upper_terms = {"ai", "cv", "llm", "ml", "nlp", "rag"}
    clean = term.strip()
    return clean.upper() if clean.casefold() in upper_terms else clean


def _join_terms(terms: list[str]) -> str:
    return " ".join(_dedupe(terms))


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _query_key(query: str) -> str:
    return _normalize_query(query).casefold()


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped
