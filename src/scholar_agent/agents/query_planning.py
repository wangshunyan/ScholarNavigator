"""确定性的初始检索子查询规划，不包含来源适配或评测信息。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from scholar_agent.core.search_schemas import (
    QUERY_PLANNER_VERSION,
    QueryAnalysis,
    QueryFacet,
    QueryFacetSource,
    QueryFacetType,
    QueryPlanningResult,
    SearchSubquery,
)


PLANNER_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "any",
    "as",
    "at",
    "based",
    "by",
    "can",
    "compare",
    "comparison",
    "could",
    "develop",
    "explored",
    "find",
    "for",
    "from",
    "give",
    "have",
    "in",
    "into",
    "is",
    "latest",
    "list",
    "looking",
    "me",
    "of",
    "on",
    "paper",
    "papers",
    "please",
    "present",
    "proposed",
    "provide",
    "recent",
    "research",
    "review",
    "search",
    "show",
    "some",
    "studies",
    "study",
    "survey",
    "tell",
    "that",
    "the",
    "there",
    "through",
    "to",
    "using",
    "want",
    "what",
    "where",
    "which",
    "with",
    "works",
    "you",
}
GENERIC_PAPER_TERMS = {
    "application",
    "benchmark",
    "comparison",
    "dataset",
    "evaluation",
    "method",
    "review",
    "survey",
}
TASK_TERMS = (
    "question answering",
    "causal inference",
    "information retrieval",
    "property prediction",
    "recommendation",
    "classification",
    "prediction",
    "detection",
    "segmentation",
    "generation",
    "retrieval",
    "ranking",
    "reranking",
    "translation",
    "summarization",
    "reconstruction",
    "forecasting",
    "optimization",
    "application",
    "问答",
    "因果推断",
    "信息检索",
    "推荐",
    "分类",
    "预测",
    "检测",
    "分割",
    "生成",
    "检索",
    "排序",
    "翻译",
    "摘要",
    "重建",
)
PURPOSE_FACETS: dict[str, tuple[QueryFacetType, ...]] = {
    "original_query": ("topic",),
    "concept_projection": ("topic",),
    "normalized_keywords": ("topic",),
    "method_dimension": ("topic", "method"),
    "dataset_dimension": ("topic", "dataset"),
    "venue_dimension": ("topic", "venue"),
    "paper_type_dimension": ("topic", "paper_type"),
    "application_expansion": ("topic", "task"),
    "benchmark_dataset_expansion": ("topic", "dataset", "paper_type"),
    "method_comparison_expansion": ("topic", "method", "paper_type"),
    "survey_expansion": ("topic", "paper_type"),
    "recent_progress_expansion": ("topic", "temporal"),
}
CONTROLLED_RELAXATION_MAX_SUPPLEMENTAL_QUERIES = 2
CONTROLLED_RELAXATION_MAX_CORE_TERMS = 8
DISJUNCTIVE_FACETS_MAX_SUPPLEMENTAL_QUERIES = 2
DISJUNCTIVE_FACETS_MIN_ANY_TERMS = 4
DISJUNCTIVE_FACETS_MAX_ANY_TERMS = 8
CURRENT_PLUS_DISJUNCTIVE_MAX_TOTAL_QUERIES = 5
FACET_UNION_MAX_TOTAL_QUERIES = 5
FACET_UNION_MAX_TERMS = 3
CONTROLLED_RELAXATION_STOPWORDS = PLANNER_STOPWORDS | {
    "advancement",
    "any",
    "applying",
    "been",
    "called",
    "contribute",
    "designed",
    "during",
    "enhancing",
    "focus",
    "focused",
    "formalised",
    "framework",
    "fundamental",
    "improve",
    "introduced",
    "introduce",
    "leveraged",
    "like",
    "only",
    "proposing",
    "recently",
    "reducing",
    "stage",
    "talked",
    "term",
    "tested",
    "used",
    "was",
    "while",
    "work",
    "working",
}
_UNRELIABLE_INFERRED_FACET_TERMS = {
    "approach",
    "benchmark",
    "corpus",
    "dataset",
    "method",
    "proposed",
    "study",
}
CONCEPT_PROJECTION_NON_CONCEPT_TERMS = PLANNER_STOPWORDS | {
    "current",
    "earliest",
    "eight",
    "excluding",
    "except",
    "few",
    "first",
    "five",
    "four",
    "last",
    "many",
    "new",
    "newest",
    "nine",
    "no",
    "not",
    "one",
    "previous",
    "recently",
    "second",
    "seven",
    "several",
    "six",
    "ten",
    "third",
    "three",
    "top",
    "two",
    "without",
    "year",
    "years",
    "不",
    "无",
    "最新",
    "近年",
    "论文",
    "研究",
}


def plan_facet_balanced(
    query_analysis: QueryAnalysis,
    *,
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """保留原查询，并在固定配额内选择互补 facet 查询。"""

    facets = identify_query_facets(query_analysis)
    topic = next((item for item in facets if item.facet_type == "topic"), None)
    topic_terms = list(topic.terms if topic is not None else [])
    required_terms = _explicit_must_have(query_analysis)
    original = SearchSubquery(
        query=query_analysis.original_query,
        source_hints=selected_sources,
        priority=1,
        purpose="original_query",
        facet_types=["topic"] if topic_terms else [],
        provenance=["original_query"],
    )
    candidates = _facet_candidates(
        facets,
        topic_terms=topic_terms,
        required_terms=required_terms,
        selected_sources=selected_sources,
    )
    selected = [original]
    duplicate_count = 0
    skipped_facets: list[str] = []
    for candidate in candidates:
        duplicate = _similar_subquery(selected, candidate)
        if duplicate is not None:
            duplicate_count += 1
            selected[selected.index(duplicate)] = _merge_subquery_provenance(
                duplicate,
                candidate,
            )
            skipped_facets.append(f"duplicate:{candidate.purpose}")
            continue
        if len(selected) >= max_subqueries:
            skipped_facets.append(f"budget:{candidate.purpose}")
            continue
        selected.append(candidate.model_copy(update={"priority": len(selected) + 1}))

    result = _planning_result(
        policy="facet_balanced",
        facets=facets,
        selected=selected,
        skipped_facets=skipped_facets,
        duplicate_count=duplicate_count,
    )
    return selected, result


def plan_controlled_relaxation(
    query_analysis: QueryAnalysis,
    *,
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """保留原查询，并用至多两条来源无关查询受控放宽召回。"""

    facets = identify_query_facets(query_analysis)
    original = SearchSubquery(
        query=query_analysis.original_query,
        source_hints=selected_sources,
        priority=1,
        purpose="original_query",
        facet_types=["topic"],
        provenance=["original_query"],
    )
    maximum = min(
        max(1, max_subqueries),
        1 + CONTROLLED_RELAXATION_MAX_SUPPLEMENTAL_QUERIES,
    )
    if maximum == 1:
        return [original], _planning_result(
            policy="controlled_relaxation",
            facets=facets,
            selected=[original],
            skipped_facets=["budget:controlled_core_topic"],
            duplicate_count=0,
        )

    explicit_must = _explicit_must_have(query_analysis)
    core_terms = _controlled_core_terms(query_analysis, facets, explicit_must)
    core_query = " ".join(_dedupe([*explicit_must, *core_terms])).strip()
    candidates: list[SearchSubquery] = []
    if core_query:
        candidates.append(
            SearchSubquery(
                query=core_query,
                source_hints=selected_sources,
                priority=2,
                purpose="controlled_core_topic",
                facet_types=["topic"],
                provenance=_dedupe(
                    [
                        "topic:controlled_relaxation",
                        *("must_have:explicit" for _ in explicit_must[:1]),
                    ]
                ),
            )
        )

    selected_facet = _most_reliable_relaxation_facet(
        facets,
        core_query,
        query_analysis.original_query,
    )
    if core_query and selected_facet is not None:
        facet_query = " ".join(
            _dedupe([*explicit_must, *core_terms, *selected_facet.terms])
        ).strip()
        candidates.append(
            SearchSubquery(
                query=facet_query,
                source_hints=selected_sources,
                priority=3,
                purpose=f"controlled_core_plus_{selected_facet.facet_type}",
                facet_types=["topic", selected_facet.facet_type],
                provenance=_dedupe(
                    [
                        "topic:controlled_relaxation",
                        f"{selected_facet.facet_type}:{selected_facet.source}",
                        *("must_have:explicit" for _ in explicit_must[:1]),
                    ]
                ),
            )
        )

    selected = [original]
    skipped: list[str] = []
    duplicate_count = 0
    for candidate in candidates:
        duplicate = _similar_subquery(selected, candidate)
        if duplicate is not None:
            duplicate_count += 1
            skipped.append(f"duplicate:{candidate.purpose}")
            continue
        if len(selected) >= maximum:
            skipped.append(f"budget:{candidate.purpose}")
            continue
        selected.append(candidate.model_copy(update={"priority": len(selected) + 1}))

    return selected, _planning_result(
        policy="controlled_relaxation",
        facets=facets,
        selected=selected,
        skipped_facets=skipped,
        duplicate_count=duplicate_count,
    )


def plan_disjunctive_facets(
    query_analysis: QueryAnalysis,
    *,
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """保留原查询，并用有界、来源无关的析取分面补充召回。"""

    facets = identify_query_facets(query_analysis)
    original = SearchSubquery(
        query=query_analysis.original_query,
        combination_mode="all",
        source_hints=selected_sources,
        priority=1,
        purpose="original_query",
        facet_types=["topic"],
        provenance=["original_query"],
    )
    maximum = min(
        max(1, max_subqueries),
        1 + DISJUNCTIVE_FACETS_MAX_SUPPLEMENTAL_QUERIES,
    )
    if maximum == 1:
        return [original], _planning_result(
            policy="disjunctive_facets",
            facets=facets,
            selected=[original],
            skipped_facets=["budget:disjunctive_facet_any"],
            duplicate_count=0,
        )

    explicit_must = _explicit_must_have(query_analysis)
    any_terms, any_types, any_provenance = _disjunctive_facet_terms(
        query_analysis,
        facets,
        explicit_must,
    )
    candidates: list[SearchSubquery] = []
    skipped: list[str] = []
    if len(any_terms) >= DISJUNCTIVE_FACETS_MIN_ANY_TERMS:
        candidates.append(
            SearchSubquery(
                query=_render_logical_terms(any_terms),
                combination_mode="any",
                source_hints=selected_sources,
                priority=2,
                purpose="disjunctive_facet_any",
                facet_types=any_types,
                provenance=_dedupe(
                    [
                        *any_provenance,
                        *("must_have:explicit_hard" for _ in explicit_must[:1]),
                    ]
                ),
            )
        )
    else:
        skipped.append("insufficient_high_confidence_terms:disjunctive_facet_any")

    topic = next((item for item in facets if item.facet_type == "topic"), None)
    topic_terms = _reliable_facet_terms(
        query_analysis,
        topic,
        excluded=explicit_must,
    )[:4]
    reliable_facet = _most_reliable_relaxation_facet(
        facets,
        " ".join(topic_terms),
        query_analysis.original_query,
    )
    if topic_terms and reliable_facet is not None:
        all_terms = _dedupe(
            [*explicit_must, *topic_terms, *reliable_facet.terms[:2]]
        )
        candidates.append(
            SearchSubquery(
                query=" ".join(all_terms),
                combination_mode="all",
                source_hints=selected_sources,
                priority=3,
                purpose=f"disjunctive_topic_plus_{reliable_facet.facet_type}",
                facet_types=["topic", reliable_facet.facet_type],
                provenance=_dedupe(
                    [
                        "topic:disjunctive_facets",
                        f"{reliable_facet.facet_type}:{reliable_facet.source}",
                        *("must_have:explicit_hard" for _ in explicit_must[:1]),
                    ]
                ),
            )
        )

    selected = [original]
    duplicate_count = 0
    for candidate in candidates:
        duplicate = _similar_subquery(selected, candidate)
        if duplicate is not None:
            duplicate_count += 1
            skipped.append(f"duplicate:{candidate.purpose}")
            continue
        if len(selected) >= maximum:
            skipped.append(f"budget:{candidate.purpose}")
            continue
        selected.append(candidate.model_copy(update={"priority": len(selected) + 1}))

    return selected, _planning_result(
        policy="disjunctive_facets",
        facets=facets,
        selected=selected,
        skipped_facets=skipped,
        duplicate_count=duplicate_count,
    )


def plan_current_plus_disjunctive(
    query_analysis: QueryAnalysis,
    *,
    current_subqueries: list[SearchSubquery],
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """完整保留旧规则查询，仅在额外配额中追加一条析取查询。"""

    facets = identify_query_facets(query_analysis)
    selected = [item.model_copy(deep=True) for item in current_subqueries]
    skipped: list[str] = []
    duplicate_count = 0
    maximum = min(
        CURRENT_PLUS_DISJUNCTIVE_MAX_TOTAL_QUERIES,
        max(1, max_subqueries) + 1,
    )
    if len(selected) >= maximum:
        skipped.append("budget:current_plus_disjunctive_any")
        return selected, _planning_result(
            policy="current_plus_disjunctive",
            facets=facets,
            selected=selected,
            skipped_facets=skipped,
            duplicate_count=duplicate_count,
        )

    explicit_must = _explicit_must_have(query_analysis)
    any_terms, any_types, any_provenance = _disjunctive_facet_terms(
        query_analysis,
        facets,
        explicit_must,
    )
    if len(any_terms) < DISJUNCTIVE_FACETS_MIN_ANY_TERMS:
        skipped.append(
            "insufficient_high_confidence_terms:current_plus_disjunctive_any"
        )
        return selected, _planning_result(
            policy="current_plus_disjunctive",
            facets=facets,
            selected=selected,
            skipped_facets=skipped,
            duplicate_count=duplicate_count,
        )

    candidate = SearchSubquery(
        query=_render_logical_terms(any_terms),
        combination_mode="any",
        source_hints=selected_sources,
        priority=min(len(selected) + 1, 5),
        purpose="current_plus_disjunctive_any",
        facet_types=any_types,
        provenance=_dedupe(
            [
                *any_provenance,
                *(
                    "must_have:explicit_hard"
                    for _ in explicit_must[:1]
                ),
                "execution_tier:supplemental_after_current_rules",
            ]
        ),
    )
    if any(_query_key(item.query) == _query_key(candidate.query) for item in selected):
        duplicate_count = 1
        skipped.append("duplicate:current_plus_disjunctive_any")
    else:
        selected.append(candidate)

    return selected, _planning_result(
        policy="current_plus_disjunctive",
        facets=facets,
        selected=selected,
        skipped_facets=skipped,
        duplicate_count=duplicate_count,
    )


def plan_facet_union(
    query_analysis: QueryAnalysis,
    *,
    current_subqueries: list[SearchSubquery],
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """完整保留旧规则查询，再追加至多一条独立分面查询。"""

    facets = identify_query_facets(query_analysis)
    selected = [item.model_copy(deep=True) for item in current_subqueries]
    skipped: list[str] = []
    duplicate_count = 0
    maximum = min(
        FACET_UNION_MAX_TOTAL_QUERIES,
        max(1, max_subqueries) + 1,
    )
    if len(selected) >= maximum:
        skipped.append("budget:facet_union")
        return selected, _planning_result(
            policy="facet_union",
            facets=facets,
            selected=selected,
            skipped_facets=skipped,
            duplicate_count=duplicate_count,
        )

    candidate = _facet_union_candidate(
        query_analysis,
        facets,
        selected_sources=selected_sources,
        priority=min(len(selected) + 1, 5),
    )
    if candidate is None:
        skipped.append("no_eligible_facet:facet_union")
    elif _similar_subquery(selected, candidate) is not None:
        duplicate_count = 1
        skipped.append(f"duplicate:{candidate.purpose}")
    else:
        selected.append(candidate)

    return selected, _planning_result(
        policy="facet_union",
        facets=facets,
        selected=selected,
        skipped_facets=skipped,
        duplicate_count=duplicate_count,
    )


def summarize_current_rules_planning(
    query_analysis: QueryAnalysis,
    subqueries: list[SearchSubquery],
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """为旧规划补充诊断元数据，但不改变查询、顺序或配额。"""

    facets = identify_query_facets(query_analysis)
    annotated = [
        _annotate_current_subquery(item, facets)
        for item in subqueries
    ]
    result = _planning_result(
        policy="current_rules",
        facets=facets,
        selected=annotated,
        skipped_facets=[],
        duplicate_count=max(0, len(annotated) - len({_query_key(i.query) for i in annotated})),
    )
    return annotated, result


def plan_concept_projection(
    query_analysis: QueryAnalysis,
    *,
    current_subqueries: list[SearchSubquery],
    selected_sources: list[str],
    max_subqueries: int,
) -> tuple[list[SearchSubquery], QueryPlanningResult]:
    """Replace one derived query with an ordered verbatim concept projection.

    The projection consumes only rule-derived topic facets and explicit must-have
    constraints. Every emitted concept must be recoverable verbatim from the
    original query; source syntax remains the query adapter's responsibility.
    """

    selected = list(current_subqueries[:max_subqueries])
    facets = identify_query_facets(query_analysis)
    input_concepts, projected_concepts = _concept_projection_terms(
        query_analysis,
        facets,
    )
    projection = " ".join(projected_concepts).strip()
    replaced_query: str | None = None
    replaced_purpose: str | None = None
    skip_reason: str | None = None

    if not input_concepts:
        skip_reason = "no_must_have_or_topic_concepts"
    elif not projection:
        skip_reason = "no_eligible_verbatim_concepts"
    elif _query_key(projection) == _query_key(query_analysis.original_query):
        skip_reason = "equivalent_to_original_query"
    elif any(_query_key(item.query) == _query_key(projection) for item in selected):
        skip_reason = "equivalent_existing_query"
    else:
        replaceable = [
            (item.priority, index)
            for index, item in enumerate(selected)
            if item.purpose != "original_query"
        ]
        if not replaceable:
            skip_reason = "no_derived_query_to_replace"
        else:
            _, replace_index = max(replaceable)
            replaced = selected[replace_index]
            replaced_query = replaced.query
            replaced_purpose = replaced.purpose
            selected[replace_index] = SearchSubquery(
                query=projection,
                source_hints=list(selected_sources),
                priority=replaced.priority,
                purpose="concept_projection",
                facet_types=["topic"],
                provenance=[
                    "concept_projection:verbatim_must_have_topic",
                    f"replaced:{replaced.purpose}",
                ],
            )

    skipped = [f"concept_projection:{skip_reason}"] if skip_reason else []
    result = _planning_result(
        policy="concept_projection",
        facets=facets,
        selected=selected,
        skipped_facets=skipped,
        duplicate_count=max(
            0,
            len(selected) - len({_query_key(item.query) for item in selected}),
        ),
    )
    result = result.model_copy(
        update={
            "concept_projection_input_concepts": input_concepts,
            "concept_projection_selected_concepts": projected_concepts,
            "concept_projection_query": projection or None,
            "concept_projection_replaced_query": replaced_query,
            "concept_projection_replaced_purpose": replaced_purpose,
            "concept_projection_skip_reason": skip_reason,
        }
    )
    return selected, result


def _concept_projection_terms(
    query_analysis: QueryAnalysis,
    facets: list[QueryFacet],
) -> tuple[list[str], list[str]]:
    topic = next((item for item in facets if item.facet_type == "topic"), None)
    raw_concepts = _dedupe(
        [
            *query_analysis.constraints.must_include_terms,
            *(topic.terms if topic is not None else []),
        ]
    )
    original = query_analysis.original_query
    located: list[tuple[int, int, str, str]] = []
    unlocated: list[tuple[int, str]] = []
    for index, concept in enumerate(raw_concepts):
        normalized = " ".join(str(concept).split()).strip()
        if not normalized:
            continue
        match = _verbatim_concept_match(original, normalized)
        if match is None:
            unlocated.append((index, normalized))
            continue
        located.append((match.start(), index, normalized, match.group(0)))

    located.sort(key=lambda item: (item[0], item[1]))
    input_concepts = [item[2] for item in located]
    input_concepts.extend(item[1] for item in unlocated)
    excluded = _casefold_set(query_analysis.constraints.exclude_terms)
    projected: list[str] = []
    seen: set[str] = set()
    for _, _, normalized, surface in located:
        key = _concept_projection_key(normalized)
        surface_key = _concept_projection_key(surface)
        if (
            not key
            or key in CONCEPT_PROJECTION_NON_CONCEPT_TERMS
            or re.fullmatch(r"\d+(?:st|nd|rd|th)?", key)
            or _concept_conflicts_with_exclusion(key, excluded)
            or surface_key in seen
        ):
            continue
        projected.append(surface)
        seen.add(surface_key)
    return input_concepts, projected


def _concept_projection_key(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _verbatim_concept_match(query: str, concept: str) -> re.Match[str] | None:
    pattern = re.escape(concept).replace(r"\ ", r"\s+")
    if concept[0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if concept[-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return re.search(pattern, query, flags=re.IGNORECASE)


def _concept_conflicts_with_exclusion(key: str, excluded: set[str]) -> bool:
    return any(
        key == value
        or _contains_term(key, value)
        or _contains_term(value, key)
        for value in excluded
    )


def identify_query_facets(query_analysis: QueryAnalysis) -> list[QueryFacet]:
    """从最终合并后的 QueryAnalysis 提取可追踪 facet。"""

    constraints = query_analysis.constraints
    llm_used = "llm_query_understanding" in query_analysis.reasoning
    structured_terms = _casefold_set(
        constraints.methods
        + constraints.datasets
        + constraints.venues
        + list(constraints.paper_types)
    )
    tasks = _task_terms(query_analysis.original_query)
    structured_terms.update(value.casefold() for value in tasks)
    structured_tokens = {
        token.casefold()
        for value in structured_terms
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", value)
    }
    topics = [
        term
        for term in _query_terms(query_analysis.original_query)
        if term.casefold() not in structured_terms
        and term.casefold() not in structured_tokens
        and term.casefold() not in GENERIC_PAPER_TERMS
    ]
    explicit_must = _explicit_must_have(query_analysis)
    topics = _dedupe([*explicit_must, *topics])[:12]
    facets: list[QueryFacet] = []
    if topics:
        source: QueryFacetSource = (
            "explicit"
            if explicit_must
            else "llm" if llm_used else "rules"
        )
        facets.append(
            QueryFacet(
                facet_type="topic",
                terms=topics,
                confidence=1.0 if explicit_must else 0.8,
                source=source,
                required=True,
            )
        )
    _append_constraint_facet(
        facets,
        query_analysis,
        "method",
        list(constraints.methods),
        "methods",
        llm_used,
    )
    _append_constraint_facet(
        facets,
        query_analysis,
        "dataset",
        list(constraints.datasets),
        "datasets",
        llm_used,
    )
    if tasks:
        facets.append(
            QueryFacet(
                facet_type="task",
                terms=tasks,
                confidence=0.75,
                source="llm" if llm_used else "rules",
            )
        )
    _append_constraint_facet(
        facets,
        query_analysis,
        "paper_type",
        list(constraints.paper_types),
        "paper_types",
        llm_used,
    )
    _append_constraint_facet(
        facets,
        query_analysis,
        "venue",
        list(constraints.venues),
        "venues",
        llm_used,
    )
    if constraints.time_range is not None:
        facets.append(
            QueryFacet(
                facet_type="temporal",
                terms=[_time_term(constraints.time_range)],
                confidence=1.0,
                source=_constraint_source(
                    query_analysis,
                    "time_range",
                    llm_used,
                ),
                required="time_range" in constraints.explicit_fields,
            )
        )
    return facets


def _append_constraint_facet(
    facets: list[QueryFacet],
    query_analysis: QueryAnalysis,
    facet_type: QueryFacetType,
    terms: list[str],
    field_name: str,
    llm_used: bool,
) -> None:
    terms = _dedupe(terms)
    if not terms:
        return
    source = _constraint_source(query_analysis, field_name, llm_used)
    facets.append(
        QueryFacet(
            facet_type=facet_type,
            terms=terms,
            confidence=1.0 if source == "explicit" else 0.8,
            source=source,
            required=source == "explicit",
        )
    )


def _constraint_source(
    query_analysis: QueryAnalysis,
    field_name: str,
    llm_used: bool,
) -> QueryFacetSource:
    if field_name in query_analysis.constraints.explicit_fields:
        return "explicit"
    return "llm" if llm_used else "rules"


def _facet_candidates(
    facets: list[QueryFacet],
    *,
    topic_terms: list[str],
    required_terms: list[str],
    selected_sources: list[str],
) -> list[SearchSubquery]:
    base = topic_terms[:10]
    if not base:
        return []
    candidates: list[tuple[int, SearchSubquery]] = []
    compact_terms = _dedupe([*base, *required_terms])
    if len(compact_terms) >= 2:
        candidates.append(
            (
                0,
                SearchSubquery(
                    query=" ".join(compact_terms),
                    source_hints=selected_sources,
                    priority=2,
                    purpose="facet_topic_compact",
                    facet_types=["topic"],
                    provenance=[
                        "topic:rules",
                        *(
                            ["must_have:explicit"]
                            if required_terms
                            else []
                        ),
                    ],
                ),
            )
        )
    priority = {
        "method": 1,
        "dataset": 2,
        "task": 3,
        "paper_type": 4,
        "venue": 5,
        "temporal": 6,
    }
    for facet in facets:
        if facet.facet_type == "topic":
            continue
        rank = priority.get(facet.facet_type, 9)
        if facet.required:
            rank -= 10
        terms = _dedupe([*base, *facet.terms, *required_terms])
        query = " ".join(terms).strip()
        if not query:
            continue
        candidates.append(
            (
                rank,
                SearchSubquery(
                    query=query,
                    source_hints=selected_sources,
                    priority=2,
                    purpose=f"facet_{facet.facet_type}",
                    facet_types=["topic", facet.facet_type],
                    provenance=[
                        "topic:rules",
                        f"{facet.facet_type}:{facet.source}",
                        *(
                            ["must_have:explicit"]
                            if required_terms
                            else []
                        ),
                    ],
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].purpose,
            _query_key(item[1].query),
        )
    )
    return [subquery for _, subquery in candidates]


def _annotate_current_subquery(
    subquery: SearchSubquery,
    facets: list[QueryFacet],
) -> SearchSubquery:
    facet_types = list(PURPOSE_FACETS.get(subquery.purpose, ()))
    if subquery.purpose == "constraint_expansion":
        facet_types = [facet.facet_type for facet in facets]
    available = {facet.facet_type: facet for facet in facets}
    facet_types = [item for item in facet_types if item in available]
    provenance = [
        f"{facet_type}:{available[facet_type].source}"
        for facet_type in facet_types
    ]
    return subquery.model_copy(
        update={"facet_types": facet_types, "provenance": provenance}
    )


def _planning_result(
    *,
    policy: str,
    facets: list[QueryFacet],
    selected: list[SearchSubquery],
    skipped_facets: list[str],
    duplicate_count: int,
) -> QueryPlanningResult:
    selected_types = {
        facet_type
        for subquery in selected
        for facet_type in subquery.facet_types
    }
    return QueryPlanningResult(
        policy=policy,  # type: ignore[arg-type]
        planner_version=QUERY_PLANNER_VERSION,
        facets=facets,
        selected_subqueries=selected,
        skipped_facets=_dedupe(skipped_facets),
        identified_facet_count=len(facets),
        selected_facet_count=sum(
            facet.facet_type in selected_types for facet in facets
        ),
        explicit_facet_count=sum(facet.source == "explicit" for facet in facets),
        selected_subquery_count=len(selected),
        duplicate_subquery_count=duplicate_count,
        skipped_by_budget_count=sum(
            item.startswith("budget:") for item in skipped_facets
        ),
        topic_coverage=_facet_coverage(facets, selected_types, "topic"),
        method_coverage=_facet_coverage(facets, selected_types, "method"),
        dataset_coverage=_facet_coverage(facets, selected_types, "dataset"),
        task_coverage=_facet_coverage(facets, selected_types, "task"),
        paper_type_coverage=_facet_coverage(
            facets,
            selected_types,
            "paper_type",
        ),
    )


def _facet_coverage(
    facets: list[QueryFacet],
    selected_types: set[QueryFacetType],
    facet_type: QueryFacetType,
) -> float:
    identified = any(item.facet_type == facet_type for item in facets)
    if not identified:
        return 1.0
    return 1.0 if facet_type in selected_types else 0.0


def _similar_subquery(
    selected: list[SearchSubquery],
    candidate: SearchSubquery,
) -> SearchSubquery | None:
    candidate_key = _query_key(candidate.query)
    candidate_primary = _primary_facet(candidate)
    for current in selected:
        if current.combination_mode != candidate.combination_mode:
            continue
        if _query_key(current.query) == candidate_key:
            return current
        if (
            candidate_primary is not None
            and candidate_primary == _primary_facet(current)
            and _term_jaccard(current.query, candidate.query) >= 0.8
        ):
            return current
    return None


def _merge_subquery_provenance(
    preferred: SearchSubquery,
    duplicate: SearchSubquery,
) -> SearchSubquery:
    return preferred.model_copy(
        update={
            "facet_types": _dedupe([*preferred.facet_types, *duplicate.facet_types]),
            "provenance": _dedupe([*preferred.provenance, *duplicate.provenance]),
        }
    )


def _primary_facet(subquery: SearchSubquery) -> str | None:
    return next(
        (item for item in subquery.facet_types if item != "topic"),
        None,
    )


def _explicit_must_have(query_analysis: QueryAnalysis) -> list[str]:
    constraints = query_analysis.constraints
    if "must_include_terms" not in constraints.explicit_fields:
        return []
    return list(constraints.must_include_terms)


def _facet_union_candidate(
    query_analysis: QueryAnalysis,
    facets: list[QueryFacet],
    *,
    selected_sources: list[str],
    priority: int,
) -> SearchSubquery | None:
    explicit_must = _explicit_must_have(query_analysis)
    selected_facet: QueryFacet | None = None
    selected_terms: list[str] = []
    for facet_type, minimum_confidence, explicit_only in (
        ("dataset", 1.0, True),
        ("method", 0.8, False),
        ("task", 0.75, False),
        ("topic", 0.8, False),
    ):
        facet = next(
            (item for item in facets if item.facet_type == facet_type),
            None,
        )
        if (
            facet is None
            or facet.confidence < minimum_confidence
            or (explicit_only and facet.source != "explicit")
        ):
            continue
        terms = _reliable_facet_terms(
            query_analysis,
            facet,
            excluded=explicit_must,
        )
        if facet_type == "topic":
            terms = _high_information_topic_terms(terms)
            if len(terms) < 2:
                continue
        else:
            terms = terms[:FACET_UNION_MAX_TERMS]
        if not terms:
            continue
        selected_facet = facet
        selected_terms = terms
        break

    if selected_facet is None:
        return None
    domain_terms = _explicit_domain_terms(
        query_analysis,
        excluded=[*explicit_must, *selected_terms],
    )
    query_terms = _dedupe(
        [*explicit_must, *selected_terms, *domain_terms]
    )
    if not query_terms:
        return None
    facet_type = selected_facet.facet_type
    return SearchSubquery(
        query=_render_logical_terms(query_terms),
        combination_mode="all",
        source_hints=selected_sources,
        priority=priority,
        purpose=f"facet_union_{facet_type}",
        facet_types=[facet_type],
        provenance=_dedupe(
            [
                f"{facet_type}:{selected_facet.source}",
                *("must_have:explicit_hard" for _ in explicit_must[:1]),
                *("domain:explicit" for _ in domain_terms[:1]),
                "execution_tier:supplemental_after_current_rules",
            ]
        ),
    )


def _explicit_domain_terms(
    query_analysis: QueryAnalysis,
    *,
    excluded: list[str],
) -> list[str]:
    constraints = query_analysis.constraints
    if "domains" not in constraints.explicit_fields:
        return []
    excluded_keys = _casefold_set(excluded)
    return [
        item
        for item in _dedupe(constraints.domains)
        if item.casefold() not in excluded_keys
    ][:1]


def _high_information_topic_terms(terms: list[str]) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for index, term in enumerate(terms):
        tokens = _query_terms(term)
        information = sum(len(token) for token in tokens)
        if not tokens or information < 5:
            continue
        candidates.append((index, information, term))
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))[
        :FACET_UNION_MAX_TERMS
    ]
    chosen_indexes = {item[0] for item in ranked}
    return [
        term
        for index, _, term in candidates
        if index in chosen_indexes
    ]


def _disjunctive_facet_terms(
    query_analysis: QueryAnalysis,
    facets: list[QueryFacet],
    explicit_must: list[str],
) -> tuple[list[str], list[QueryFacetType], list[str]]:
    """按 facet 稳定轮转，避免 topic 独占 4～8 个析取名额。"""

    by_type: list[tuple[QueryFacetType, QueryFacet, list[str]]] = []
    for facet_type in ("topic", "method", "dataset", "task"):
        facet = next((item for item in facets if item.facet_type == facet_type), None)
        values = _reliable_facet_terms(
            query_analysis,
            facet,
            excluded=explicit_must,
        )
        if facet is not None and values:
            by_type.append((facet_type, facet, values))

    selected: list[str] = []
    selected_types: list[QueryFacetType] = []
    provenance: list[str] = []
    for offset in range(DISJUNCTIVE_FACETS_MAX_ANY_TERMS):
        added = False
        for facet_type, facet, values in by_type:
            if offset >= len(values):
                continue
            selected = _dedupe([*selected, values[offset]])
            if facet_type not in selected_types:
                selected_types.append(facet_type)
                provenance.append(f"{facet_type}:{facet.source}")
            added = True
            if len(selected) >= DISJUNCTIVE_FACETS_MAX_ANY_TERMS:
                break
        if len(selected) >= DISJUNCTIVE_FACETS_MAX_ANY_TERMS or not added:
            break
    return selected, selected_types, provenance


def _reliable_facet_terms(
    query_analysis: QueryAnalysis,
    facet: QueryFacet | None,
    *,
    excluded: list[str],
) -> list[str]:
    if facet is None or facet.confidence < 0.75:
        return []
    excluded_keys = _casefold_set(excluded)
    original = query_analysis.original_query.casefold()
    values: list[str] = []
    for term in facet.terms:
        normalized = " ".join(str(term).split()).strip()
        key = normalized.casefold()
        tokens = _query_terms(normalized)
        if (
            not normalized
            or len(normalized) > 80
            or key in excluded_keys
            or key in GENERIC_PAPER_TERMS
            or all(
                token.casefold() in _UNRELIABLE_INFERRED_FACET_TERMS
                for token in tokens
            )
            or (
                facet.source != "explicit"
                and not _contains_term(original, normalized)
            )
        ):
            continue
        values.append(normalized)
    return _dedupe(values)


def _render_logical_terms(terms: list[str]) -> str:
    rendered: list[str] = []
    for term in terms:
        normalized = " ".join(str(term).replace('"', " ").split())
        if not normalized:
            continue
        rendered.append(f'"{normalized}"' if " " in normalized else normalized)
    return " ".join(rendered)


def _controlled_core_terms(
    query_analysis: QueryAnalysis,
    facets: list[QueryFacet],
    explicit_must: list[str],
) -> list[str]:
    topic = next((item for item in facets if item.facet_type == "topic"), None)
    candidates = list(topic.terms if topic is not None else [])
    explicit_keys = _casefold_set(explicit_must)
    core = [
        term
        for term in candidates
        if term.casefold() not in explicit_keys
        and term.casefold() not in CONTROLLED_RELAXATION_STOPWORDS
        and term.casefold() not in GENERIC_PAPER_TERMS
    ]
    return _dedupe(core)[:CONTROLLED_RELAXATION_MAX_CORE_TERMS]


def _most_reliable_relaxation_facet(
    facets: list[QueryFacet],
    core_query: str,
    original_query: str,
) -> QueryFacet | None:
    type_priority = {"dataset": 0, "task": 1, "method": 2}
    candidates: list[tuple[int, int, int, QueryFacet]] = []
    core_key = core_query.casefold()
    for index, facet in enumerate(facets):
        if facet.facet_type not in type_priority:
            continue
        useful_terms = [
            term
            for term in facet.terms
            if any(
                token.casefold() not in _UNRELIABLE_INFERRED_FACET_TERMS
                for token in _query_terms(term)
            )
            and (
                facet.source == "explicit"
                or _contains_term(original_query.casefold(), term)
            )
            and not _contains_term(core_key, term)
        ]
        if not useful_terms:
            continue
        candidates.append(
            (
                0 if facet.source == "explicit" else 1,
                type_priority[facet.facet_type],
                index,
                facet.model_copy(update={"terms": useful_terms}),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3]


def _contains_term(query_key: str, term: str) -> bool:
    term_key = " ".join(term.casefold().split())
    if not term_key:
        return False
    if re.fullmatch(r"[a-z0-9+.# -]+", term_key):
        return bool(
            re.search(
                rf"(?<![a-z0-9+.#-]){re.escape(term_key)}(?![a-z0-9+.#-])",
                query_key,
            )
        )
    return term_key in query_key


def _task_terms(query: str) -> list[str]:
    lowered = query.casefold()
    return _dedupe([term for term in TASK_TERMS if term.casefold() in lowered])


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", query):
        cleaned = token.strip(".,;:()[]{}").strip()
        key = cleaned.casefold()
        if not cleaned or key in PLANNER_STOPWORDS:
            continue
        if len(key) <= 2 and key not in {"ai", "cv", "ml"}:
            continue
        terms.append(_canonical_term(cleaned))
    for term in TASK_TERMS:
        if any("\u4e00" <= char <= "\u9fff" for char in term) and term in query:
            terms.append(term)
    return _dedupe(terms)


def _canonical_term(term: str) -> str:
    return (
        term.upper()
        if term.casefold() in {"ai", "cv", "llm", "ml", "nlp", "rag"}
        else term
    )


def _time_term(time_range: object) -> str:
    start = getattr(time_range, "start_year", None)
    end = getattr(time_range, "end_year", None)
    if start is not None and end is not None:
        return f"{start}-{end}"
    if start is not None:
        return f"since {start}"
    if end is not None:
        return f"through {end}"
    return str(getattr(time_range, "label", "temporal") or "temporal")


def _term_jaccard(left: str, right: str) -> float:
    left_terms = _casefold_set(_query_terms(left))
    right_terms = _casefold_set(_query_terms(right))
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 1.0


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def _casefold_set(values: Iterable[str]) -> set[str]:
    return {value.casefold() for value in values if value.strip()}


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        result.append(item)
        seen.add(key)
    return result
