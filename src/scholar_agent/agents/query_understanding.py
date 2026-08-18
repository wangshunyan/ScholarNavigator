"""Rule-based query understanding for the first internal search pipeline."""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Protocol

from scholar_agent.agents.query_planning import (
    plan_controlled_relaxation,
    plan_concept_projection,
    plan_current_plus_disjunctive,
    plan_disjunctive_facets,
    plan_facet_balanced,
    plan_facet_union,
    summarize_current_rules_planning,
)
from scholar_agent.agents.llm_query_planning import (
    LLMPlanningRuntime,
    plan_llm_semantic,
)
from scholar_agent.agents.llm_constrained_rewrite import (
    plan_llm_constrained_rewrite,
)
from scholar_agent.core.search_schemas import (
    QueryAnalysis,
    QueryConstraint,
    QueryIntent,
    QueryPlanningPolicy,
    QueryUnderstandingOptions,
    ResearchDomain,
    SearchPlan,
    SearchSubquery,
    TimeRange,
    normalize_paper_types,
)
from scholar_agent.llm.provider import chat_json as provider_chat_json
from scholar_agent.llm.provider import is_llm_enabled
from scholar_agent.prompts.loader import PromptLoadError, render_messages


RECENT_WINDOW_YEARS = 3
LLM_QUERY_UNDERSTANDING_TIMEOUT_ENV = (
    "SCHOLAR_AGENT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS"
)
DEFAULT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS = 20.0

VENUES = (
    "ACL",
    "EMNLP",
    "NAACL",
    "SIGIR",
    "WWW",
    "KDD",
    "NeurIPS",
    "ICLR",
    "ICML",
    "CVPR",
    "ICCV",
    "ECCV",
    "AAAI",
    "IJCAI",
)

INTENT_PATTERNS: tuple[tuple[QueryIntent, tuple[str, ...]], ...] = (
    ("survey", ("综述", "survey", "review", "related work", "literature review")),
    (
        "recent_progress",
        (
            "最新",
            "近年",
            "近几年",
            "近三年",
            "recent",
            "latest",
            "sota",
            "state-of-the-art",
            "cutting-edge",
        ),
    ),
    (
        "method_comparison",
        ("对比", "比较", "compare", "comparison", "versus", " vs ", " vs."),
    ),
    (
        "benchmark_or_dataset",
        ("benchmark", "dataset", "数据集", "评测", "基准"),
    ),
    ("application", ("应用", "落地", "application", "deployment")),
    (
        "paper_finding",
        (
            "找论文",
            "代表性论文",
            "find papers",
            "which studies",
            "representative papers",
        ),
    ),
)

BIOMEDICAL_TERMS = (
    "protein",
    "gene",
    "genomic",
    "clinical",
    "pubmed",
    "biomedical",
    "medicine",
    "medical",
    "医学",
    "生物",
    "蛋白",
    "基因",
    "临床",
)

MACHINE_LEARNING_TERMS = (
    "llm",
    "large language model",
    "rag",
    "reranking",
    "retrieval",
    "agent",
    "transformer",
    "nlp",
    "cv",
    "computer vision",
    "deep learning",
    "machine learning",
    "embedding",
    "大模型",
    "机器学习",
    "深度学习",
    "检索增强",
    "重排序",
)

COMPUTER_SCIENCE_TERMS = (
    "algorithm",
    "database",
    "software",
    "systems",
    "information retrieval",
    "computer science",
    "算法",
    "数据库",
    "软件",
    "信息检索",
)

METHOD_TERMS = (
    "llm",
    "rag",
    "reranking",
    "retrieval",
    "agent",
    "transformer",
    "embedding",
    "deep learning",
    "machine learning",
    "大模型",
    "检索",
    "重排序",
)

DATASET_TERMS = ("dataset", "benchmark", "数据集", "评测", "基准")

STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "after",
    "application",
    "applications",
    "based",
    "compare",
    "comparison",
    "find",
    "for",
    "from",
    "help",
    "in",
    "latest",
    "of",
    "on",
    "papers",
    "please",
    "recent",
    "review",
    "search",
    "since",
    "studies",
    "survey",
    "the",
    "to",
    "using",
    "which",
    "with",
}

CHINESE_KEYWORD_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("大模型", ("LLM",)),
    ("重排序", ("reranking",)),
    ("检索增强", ("RAG",)),
    ("检索", ("retrieval",)),
    ("搜索", ("search",)),
    ("推荐", ("recommendation",)),
    ("综述", ("survey",)),
    ("数据集", ("dataset",)),
    ("评测", ("benchmark",)),
    ("学术", ("academic",)),
    ("科研", ("scientific literature",)),
    ("论文", ("papers",)),
    ("机器学习", ("machine learning",)),
    ("深度学习", ("deep learning",)),
    ("多模态", ("multimodal",)),
    ("医学", ("biomedical",)),
    ("生物", ("biomedical",)),
    ("蛋白", ("protein",)),
    ("基因", ("gene",)),
    ("临床", ("clinical",)),
)


class LLMJsonClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ...


class QueryUnderstandingAgent:
    """Query analysis with optional LLM JSON enhancement and rule fallback."""

    def __init__(
        self,
        llm_client: "LLMJsonClient | None" = None,
        llm_planning_runtime: LLMPlanningRuntime | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._llm_planning_runtime = llm_planning_runtime

    def analyze(
        self,
        query: str,
        options: QueryUnderstandingOptions | None = None,
    ) -> SearchPlan:
        options = options or QueryUnderstandingOptions()
        # 两种独立 LLM 规划都消费规则解析结果，不能先调用通用 Query
        # Understanding 再进行第二次 LLM 调用。
        if options.use_llm and options.query_planning_policy not in {
            "llm_semantic",
            "llm_constrained_rewrite",
        }:
            return self._analyze_with_optional_llm(query, options)
        return self._analyze_rules(query, options)

    def _analyze_rules(
        self,
        query: str,
        options: QueryUnderstandingOptions,
    ) -> SearchPlan:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        current_year = options.current_year or date.today().year
        language = _detect_language(normalized_query)
        intent = _detect_intent(normalized_query)
        domain = _detect_domain(normalized_query)
        time_range = _parse_time_range(normalized_query, current_year)
        venues = _extract_venues(normalized_query)
        methods = _extract_method_terms(normalized_query)
        datasets = _extract_dataset_terms(normalized_query)
        keyword_terms = _extract_keyword_terms(normalized_query, venues)
        selected_sources, warnings = _select_sources(normalized_query, domain)
        limit_per_source, max_subqueries = _profile_settings(
            options.run_profile, options.top_k
        )

        constraints = merge_query_constraints(
            QueryConstraint(
                time_range=time_range,
                venues=venues,
                methods=methods,
                datasets=datasets,
                domains=[domain],
                must_include_terms=keyword_terms,
                exclude_terms=[],
                paper_types=[],
            ),
            options.explicit_constraints,
        )
        reasoning = [
            f"language={language}",
            f"intent={intent}",
            f"domain={domain}",
            f"sources={','.join(selected_sources)}",
        ]
        if time_range is not None:
            reasoning.append("time_range_detected")
        if venues:
            reasoning.append("venue_constraints_detected")

        query_analysis = QueryAnalysis(
            original_query=normalized_query,
            language=language,
            intent=intent,
            domain=domain,
            constraints=constraints,
            needs_expansion=_needs_expansion(intent, keyword_terms, max_subqueries),
            reasoning=reasoning,
        )
        if options.query_planning_policy == "controlled_relaxation":
            subqueries, query_planning = plan_controlled_relaxation(
                query_analysis,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )
        elif options.query_planning_policy == "disjunctive_facets":
            subqueries, query_planning = plan_disjunctive_facets(
                query_analysis,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )
        elif options.query_planning_policy == "facet_balanced":
            subqueries, query_planning = plan_facet_balanced(
                query_analysis,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )
        else:
            subqueries = _build_subqueries(
                original_query=normalized_query,
                keyword_terms=keyword_terms,
                intent=intent,
                domain=domain,
                constraints=constraints,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )
            subqueries, query_planning = summarize_current_rules_planning(
                query_analysis,
                subqueries,
            )
            if options.query_planning_policy == "prf_v1":
                query_planning = query_planning.model_copy(
                    update={"policy": "prf_v1"}
                )
            if options.query_planning_policy == "concept_projection":
                subqueries, query_planning = plan_concept_projection(
                    query_analysis,
                    current_subqueries=subqueries,
                    selected_sources=selected_sources,
                    max_subqueries=max_subqueries,
                )
            if options.query_planning_policy == "current_plus_disjunctive":
                subqueries, query_planning = plan_current_plus_disjunctive(
                    query_analysis,
                    current_subqueries=subqueries,
                    selected_sources=selected_sources,
                    max_subqueries=max_subqueries,
                )
            if options.query_planning_policy == "facet_union":
                subqueries, query_planning = plan_facet_union(
                    query_analysis,
                    current_subqueries=subqueries,
                    selected_sources=selected_sources,
                    max_subqueries=max_subqueries,
                )
            if options.query_planning_policy == "llm_semantic":
                outcome = plan_llm_semantic(
                    query_analysis,
                    current_subqueries=subqueries,
                    current_result=query_planning,
                    selected_sources=selected_sources,
                    max_subqueries=max_subqueries,
                    run_profile=options.run_profile,
                    explicit_constraints=options.explicit_constraints,
                    llm_client=self._llm_client,
                    runtime=self._llm_planning_runtime,
                )
                subqueries = outcome.subqueries
                query_planning = outcome.result
                warnings = _dedupe([*warnings, *outcome.warnings])
            if options.query_planning_policy == "llm_constrained_rewrite":
                outcome = plan_llm_constrained_rewrite(
                    query_analysis,
                    current_subqueries=subqueries,
                    current_result=query_planning,
                    run_profile=options.run_profile,
                    explicit_constraints=options.explicit_constraints,
                    llm_client=self._llm_client,
                    runtime=self._llm_planning_runtime,
                )
                subqueries = outcome.subqueries
                query_planning = outcome.result
                warnings = _dedupe([*warnings, *outcome.warnings])

        return SearchPlan(
            query_analysis=query_analysis,
            subqueries=subqueries,
            selected_sources=selected_sources,
            limit_per_source=limit_per_source,
            top_k=options.top_k,
            run_profile=options.run_profile,
            enable_refchain=options.enable_refchain,
            enable_semantic_seed_expansion=options.enable_semantic_seed_expansion,
            enable_query_evolution=options.enable_query_evolution,
            query_planning_policy=options.query_planning_policy,
            query_planning=query_planning,
            warnings=warnings,
        )

    def _analyze_with_optional_llm(
        self,
        query: str,
        options: QueryUnderstandingOptions,
    ) -> SearchPlan:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        rule_plan = self._analyze_rules(normalized_query, options)
        if self._llm_client is None and not is_llm_enabled():
            rule_plan.warnings = _dedupe(
                [*rule_plan.warnings, "llm_query_understanding_disabled"]
            )
            return rule_plan

        try:
            raw_plan = _chat_json(
                self._llm_client,
                _build_llm_messages(normalized_query),
                timeout=_llm_query_understanding_timeout_from_env(),
            )
            return _search_plan_from_llm_json(
                raw_plan,
                normalized_query,
                options,
                rule_plan,
            )
        except PromptLoadError:
            rule_plan.warnings = _dedupe(
                [
                    *rule_plan.warnings,
                    "llm_query_understanding_prompt_load_failed",
                ]
            )
            return rule_plan
        except Exception as exc:  # noqa: BLE001 - LLM is optional; keep rules path alive
            rule_plan.warnings = _dedupe(
                [
                    *rule_plan.warnings,
                    f"llm_query_understanding_failed:{_diagnostic_message(exc)}",
                ]
            )
            return rule_plan


def analyze_query(
    query: str,
    *,
    top_k: int = 20,
    run_profile: str = "balanced",
    enable_refchain: bool = False,
    enable_semantic_seed_expansion: bool = False,
    enable_query_evolution: bool = False,
    query_planning_policy: QueryPlanningPolicy = "current_rules",
    current_year: int | None = None,
    use_llm: bool | None = None,
    llm_client: "LLMJsonClient | None" = None,
    llm_planning_runtime: LLMPlanningRuntime | None = None,
    explicit_constraints: QueryConstraint | None = None,
) -> SearchPlan:
    """Analyze a user query into a SearchPlan."""

    options = QueryUnderstandingOptions(
        top_k=top_k,
        run_profile=run_profile,
        enable_refchain=enable_refchain,
        enable_semantic_seed_expansion=enable_semantic_seed_expansion,
        enable_query_evolution=enable_query_evolution,
        query_planning_policy=query_planning_policy,
        current_year=current_year,
        use_llm=use_llm,
        explicit_constraints=explicit_constraints,
    )
    return QueryUnderstandingAgent(
        llm_client=llm_client,
        llm_planning_runtime=llm_planning_runtime,
    ).analyze(query, options)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _detect_language(query: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
    has_latin = bool(re.search(r"[A-Za-z]", query))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def _detect_intent(query: str) -> QueryIntent:
    lowered = f" {query.casefold()} "
    for intent, patterns in INTENT_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return intent
    return "general"


def _detect_domain(query: str) -> ResearchDomain:
    lowered = query.casefold()
    if _contains_any(lowered, BIOMEDICAL_TERMS):
        return "biomedical"
    if _contains_any(lowered, MACHINE_LEARNING_TERMS):
        return "machine_learning"
    if _contains_any(lowered, COMPUTER_SCIENCE_TERMS):
        return "computer_science"
    return "general_science"


def _parse_time_range(query: str, current_year: int) -> TimeRange | None:
    lowered = query.casefold()

    range_match = re.search(
        r"((?:19|20)\d{2})\s*(?:-|–|—|~|至|到)\s*((?:19|20)\d{2})",
        lowered,
    )
    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        return TimeRange(start_year=start_year, end_year=end_year, label="explicit_range")

    from_to_match = re.search(
        r"from\s+((?:19|20)\d{2})\s+(?:to|through|until)\s+((?:19|20)\d{2})",
        lowered,
    )
    if from_to_match:
        start_year = int(from_to_match.group(1))
        end_year = int(from_to_match.group(2))
        return TimeRange(start_year=start_year, end_year=end_year, label="from_to")

    since_match = re.search(r"(?:since|from)\s+((?:19|20)\d{2})", lowered)
    if since_match:
        return TimeRange(
            start_year=int(since_match.group(1)),
            end_year=None,
            label="since",
        )

    after_match = re.search(r"after\s+((?:19|20)\d{2})", lowered)
    if after_match:
        return TimeRange(
            start_year=int(after_match.group(1)) + 1,
            end_year=None,
            label="after",
        )

    last_years_match = re.search(r"last\s+(\d{1,2})\s+years?", lowered)
    if last_years_match:
        years = max(1, int(last_years_match.group(1)))
        return TimeRange(
            start_year=current_year - years,
            end_year=current_year,
            label="recent",
        )

    if "近三年" in lowered:
        return TimeRange(
            start_year=current_year - 3,
            end_year=current_year,
            label="recent",
        )

    recent_markers = (
        "recent",
        "latest",
        "sota",
        "state-of-the-art",
        "cutting-edge",
        "最新",
        "近年",
        "近几年",
    )
    if any(marker in lowered for marker in recent_markers):
        return TimeRange(
            start_year=current_year - RECENT_WINDOW_YEARS,
            end_year=current_year,
            label="recent",
        )

    return None


def _extract_venues(query: str) -> list[str]:
    venues: list[str] = []
    for venue in VENUES:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(venue)}(?![A-Za-z0-9])", query, re.I):
            venues.append(venue)
    return venues


def _extract_terms(query: str, candidates: tuple[str, ...]) -> list[str]:
    lowered = query.casefold()
    return _dedupe([term for term in candidates if term in lowered])


def _extract_method_terms(query: str) -> list[str]:
    terms = _extract_terms(query, METHOD_TERMS)
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9+.#-]*(?:\s+[A-Za-z][A-Za-z0-9+.#-]*){0,2})"
        r"\s+(?:methods?|approaches?)\b",
        query,
        re.I,
    ):
        phrase = " ".join(
            token
            for token in match.group(1).split()
            if token.casefold() not in STOPWORDS
        )
        if phrase:
            terms.append(phrase)
    return _dedupe(terms)


def _extract_dataset_terms(query: str) -> list[str]:
    terms = _extract_terms(query, DATASET_TERMS)
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9_.-]*)\s+(?:dataset|benchmark|corpus)\b",
        query,
        re.I,
    ):
        name = match.group(1)
        if name.casefold() not in {"a", "the", "new", "public"}:
            terms.append(name)
    return _dedupe(terms)


def _extract_keyword_terms(query: str, venues: list[str]) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", query):
        normalized = token.strip(".,;:()[]{}").casefold()
        if not normalized or normalized in STOPWORDS:
            continue
        if len(normalized) <= 2 and token.upper() not in {"AI", "ML", "CV"}:
            continue
        terms.append(_canonical_token(token))

    lowered = query.casefold()
    for chinese, mapped_terms in CHINESE_KEYWORD_MAP:
        if chinese in lowered:
            terms.extend(mapped_terms)

    terms.extend(venues)
    return _dedupe(terms)


def _canonical_token(token: str) -> str:
    upper_tokens = {"llm", "rag", "nlp", "cv", "ai"}
    if token.casefold() in upper_tokens:
        return token.upper()
    return token.strip()


def _select_sources(query: str, domain: ResearchDomain) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    lowered = query.casefold()
    if domain == "biomedical" or "pubmed" in lowered:
        return ["pubmed", "openalex"], warnings
    if domain in {"computer_science", "machine_learning"}:
        return ["arxiv", "openalex"], warnings
    return ["openalex", "arxiv"], warnings


def _profile_settings(run_profile: str, top_k: int) -> tuple[int, int]:
    if run_profile == "fast":
        return max(5, min(10, top_k)), 2
    if run_profile == "high_recall":
        return max(30, min(50, top_k * 2)), 5
    if run_profile == "evaluation":
        return max(10, min(20, top_k)), 3
    return max(10, min(20, top_k)), 3


def _needs_expansion(
    intent: QueryIntent,
    keyword_terms: list[str],
    max_subqueries: int,
) -> bool:
    return max_subqueries > 1 and (intent != "general" or len(keyword_terms) >= 3)


def _build_subqueries(
    *,
    original_query: str,
    keyword_terms: list[str],
    intent: QueryIntent,
    domain: ResearchDomain,
    constraints: QueryConstraint,
    selected_sources: list[str],
    max_subqueries: int,
) -> list[SearchSubquery]:
    base_query = " ".join(keyword_terms).strip() or original_query
    candidates: list[tuple[str, str]] = [
        (original_query, "original_query"),
    ]

    explicit_constraint_query = (
        _constraint_subquery(base_query, constraints)
        if constraints.explicit_fields
        else None
    )
    if explicit_constraint_query:
        candidates.append(explicit_constraint_query)

    intent_query = _intent_subquery(intent, base_query, constraints.time_range)
    if intent_query:
        candidates.append(intent_query)

    candidates.extend(_structured_dimension_subqueries(base_query, constraints))

    if base_query.casefold() != original_query.casefold():
        candidates.append((base_query, "normalized_keywords"))

    domain_query = _domain_subquery(domain, base_query)
    if domain_query:
        candidates.append(domain_query)

    constraint_query = _constraint_subquery(base_query, constraints)
    if constraint_query and not explicit_constraint_query:
        candidates.append(constraint_query)

    subqueries: list[SearchSubquery] = []
    seen: set[str] = set()
    for raw_query, purpose in candidates:
        normalized = _normalize_query(raw_query)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        subqueries.append(
            SearchSubquery(
                query=normalized,
                source_hints=selected_sources,
                priority=min(len(subqueries) + 1, 5),
                purpose=purpose,
            )
        )
        seen.add(key)
        if len(subqueries) >= max_subqueries:
            break

    return subqueries[:5]


def _structured_dimension_subqueries(
    base_query: str,
    constraints: QueryConstraint,
) -> list[tuple[str, str]]:
    """Build bounded expansions only from parsed query dimensions."""

    candidates: list[tuple[str, str]] = []
    if constraints.methods:
        candidates.append(
            (
                _append_query_fragments(
                    base_query,
                    [*constraints.methods, "method"],
                ),
                "method_dimension",
            )
        )
    if constraints.datasets:
        candidates.append(
            (
                _append_query_fragments(
                    base_query,
                    [*constraints.datasets, "dataset evaluation"],
                ),
                "dataset_dimension",
            )
        )
    if constraints.venues:
        candidates.append(
            (
                _append_query_fragments(base_query, [*constraints.venues, "venue"]),
                "venue_dimension",
            )
        )
    if constraints.paper_types:
        candidates.append(
            (
                _append_query_fragments(base_query, list(constraints.paper_types)),
                "paper_type_dimension",
            )
        )
    return candidates


def _append_query_fragments(base_query: str, fragments: list[str]) -> str:
    query = base_query.strip()
    query_key = query.casefold()
    additions: list[str] = []
    for fragment in _dedupe(fragments):
        if _contains_phrase(query_key, fragment.casefold()):
            continue
        additions.append(fragment)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _intent_subquery(
    intent: QueryIntent,
    base_query: str,
    time_range: TimeRange | None,
) -> tuple[str, str] | None:
    if intent == "survey":
        return (
            _append_query_fragments(base_query, ["survey", "review"]),
            "survey_expansion",
        )
    if intent == "recent_progress":
        time_suffix = _format_time_suffix(time_range)
        return f"recent advances {base_query}{time_suffix}", "recent_progress_expansion"
    if intent == "method_comparison":
        return (
            _append_query_fragments(base_query, ["comparison", "versus"]),
            "method_comparison_expansion",
        )
    if intent == "benchmark_or_dataset":
        return (
            _append_query_fragments(
                base_query,
                ["benchmark", "dataset", "evaluation"],
            ),
            "benchmark_dataset_expansion",
        )
    if intent == "application":
        return (
            _append_query_fragments(base_query, ["application", "deployment"]),
            "application_expansion",
        )
    if intent == "paper_finding":
        return (
            _append_query_fragments(base_query, ["representative papers"]),
            "paper_finding_expansion",
        )
    return None


def _domain_subquery(
    domain: ResearchDomain,
    base_query: str,
) -> tuple[str, str] | None:
    if domain == "machine_learning":
        return f"{base_query} machine learning deep learning", "domain_ml_expansion"
    if domain == "computer_science":
        return f"{base_query} computer science information retrieval", "domain_cs_expansion"
    if domain == "biomedical":
        return f"{base_query} biomedical clinical", "domain_biomedical_expansion"
    return None


def _constraint_subquery(
    base_query: str,
    constraints: QueryConstraint,
) -> tuple[str, str] | None:
    fragments: list[str] = []
    fragments.extend(constraints.methods)
    fragments.extend(constraints.datasets)
    fragments.extend(
        domain.replace("_", " ")
        for domain in constraints.domains
        if domain != "general_science"
    )
    if "must_include_terms" in constraints.explicit_fields:
        fragments.extend(constraints.must_include_terms)
    fragments.extend(constraints.paper_types)
    if constraints.venues:
        fragments.extend(constraints.venues)
    time_suffix = _format_time_suffix(constraints.time_range).strip()
    if time_suffix:
        fragments.append(time_suffix)
    if not fragments:
        return None
    fragments = _dedupe(fragments)
    return _append_query_fragments(base_query, fragments), "constraint_expansion"


def _format_time_suffix(time_range: TimeRange | None) -> str:
    if time_range is None:
        return ""
    if time_range.start_year is not None and time_range.end_year is not None:
        return f" {time_range.start_year}-{time_range.end_year}"
    if time_range.start_year is not None:
        return f" since {time_range.start_year}"
    if time_range.end_year is not None:
        return f" through {time_range.end_year}"
    return ""


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, term) for term in terms)


def _contains_phrase(text: str, term: str) -> bool:
    if re.fullmatch(r"[a-z0-9+.# -]+", term):
        return (
            re.search(
                rf"(?<![a-z0-9+.#-]){re.escape(term)}(?![a-z0-9+.#-])",
                text,
            )
            is not None
        )
    return term in text


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


def merge_query_constraints(
    base: QueryConstraint,
    explicit: QueryConstraint | None,
) -> QueryConstraint:
    """Overlay non-empty explicit constraints on inferred constraints."""

    if explicit is None:
        return base

    list_fields = (
        "venues",
        "methods",
        "datasets",
        "domains",
        "must_include_terms",
        "exclude_terms",
        "paper_types",
    )
    values: dict[str, Any] = {
        "time_range": explicit.time_range or base.time_range,
    }
    explicit_fields = list(base.explicit_fields)
    if explicit.time_range is not None:
        explicit_fields.append("time_range")
    for field_name in list_fields:
        explicit_values = list(getattr(explicit, field_name))
        values[field_name] = (
            explicit_values if explicit_values else list(getattr(base, field_name))
        )
        if explicit_values:
            explicit_fields.append(field_name)
    values["explicit_fields"] = _dedupe(explicit_fields)
    return QueryConstraint(**values)


def _ensure_explicit_constraint_subquery(
    subqueries: list[SearchSubquery],
    *,
    original_query: str,
    constraints: QueryConstraint,
    selected_sources: list[str],
    max_subqueries: int,
) -> list[SearchSubquery]:
    if not constraints.explicit_fields:
        return subqueries
    candidate = _constraint_subquery(original_query, constraints)
    if candidate is None:
        return subqueries
    query, purpose = candidate
    candidate_key = _normalize_query(query).casefold()
    if any(item.query.casefold() == candidate_key for item in subqueries):
        return subqueries

    merged = [
        *subqueries[:1],
        SearchSubquery(
            query=query,
            source_hints=selected_sources,
            priority=2,
            purpose=purpose,
        ),
        *subqueries[1:],
    ][: min(max_subqueries, 5)]
    return [
        item.model_copy(update={"priority": min(index + 1, 5)})
        for index, item in enumerate(merged)
    ]


def _build_llm_messages(query: str) -> list[dict[str, str]]:
    return render_messages("query_understanding", {"query": query})


def _chat_json(
    llm_client: LLMJsonClient | None,
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> dict[str, Any]:
    if llm_client is not None:
        return llm_client.chat_json(messages, temperature=0, timeout=timeout)
    return provider_chat_json(messages, temperature=0, timeout=timeout)


def _llm_query_understanding_timeout_from_env() -> float:
    raw_value = os.getenv(LLM_QUERY_UNDERSTANDING_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS
    try:
        timeout = float(raw_value)
    except ValueError:
        return DEFAULT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS
    return (
        timeout
        if timeout > 0
        else DEFAULT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS
    )


def _search_plan_from_llm_json(
    raw_plan: dict[str, Any],
    query: str,
    options: QueryUnderstandingOptions,
    rule_plan: SearchPlan,
) -> SearchPlan:
    warnings = list(rule_plan.warnings)
    warnings.extend(_coerce_string_list(raw_plan.get("warnings")))
    warnings.append("llm_query_understanding_used")

    language = _coerce_literal(
        raw_plan.get("language"),
        allowed={"zh", "en", "mixed", "unknown"},
        fallback=rule_plan.query_analysis.language,
        warning_prefix="llm_invalid_language",
        warnings=warnings,
    )
    intent = _coerce_literal(
        raw_plan.get("intent") or raw_plan.get("intent_type"),
        allowed={
            "survey",
            "recent_progress",
            "method_comparison",
            "benchmark_or_dataset",
            "application",
            "paper_finding",
            "general",
        },
        fallback=rule_plan.query_analysis.intent,
        warning_prefix="llm_invalid_intent",
        warnings=warnings,
    )
    domain = _coerce_literal(
        raw_plan.get("domain"),
        allowed={
            "computer_science",
            "machine_learning",
            "biomedical",
            "general_science",
        },
        fallback=rule_plan.query_analysis.domain,
        warning_prefix="llm_invalid_domain",
        warnings=warnings,
    )

    raw_constraints = raw_plan.get("constraints")
    constraints = _constraints_from_llm_json(
        raw_constraints if isinstance(raw_constraints, dict) else {},
        domain=domain,
        rule_constraints=rule_plan.query_analysis.constraints,
        warnings=warnings,
    )
    constraints = merge_query_constraints(constraints, options.explicit_constraints)
    selected_sources = _filter_llm_sources(
        raw_plan.get("selected_sources"),
        fallback=rule_plan.selected_sources,
        warnings=warnings,
        warning_prefix="llm_selected_source",
    )
    limit_per_source, max_subqueries = _profile_settings(
        options.run_profile,
        options.top_k,
    )

    query_analysis = QueryAnalysis(
        original_query=query,
        language=language,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        domain=domain,  # type: ignore[arg-type]
        constraints=constraints,
        needs_expansion=rule_plan.query_analysis.needs_expansion,
        reasoning=_dedupe(
            [
                *rule_plan.query_analysis.reasoning,
                "llm_query_understanding",
            ]
        ),
    )
    if options.query_planning_policy == "controlled_relaxation":
        subqueries, query_planning = plan_controlled_relaxation(
            query_analysis,
            selected_sources=selected_sources,
            max_subqueries=max_subqueries,
        )
    elif options.query_planning_policy == "disjunctive_facets":
        subqueries, query_planning = plan_disjunctive_facets(
            query_analysis,
            selected_sources=selected_sources,
            max_subqueries=max_subqueries,
        )
    elif options.query_planning_policy == "facet_balanced":
        subqueries, query_planning = plan_facet_balanced(
            query_analysis,
            selected_sources=selected_sources,
            max_subqueries=max_subqueries,
        )
    else:
        subqueries = _subqueries_from_llm_json(
            raw_plan.get("subqueries"),
            selected_sources=selected_sources,
            fallback=rule_plan.subqueries,
            max_subqueries=max_subqueries,
            warnings=warnings,
        )
        subqueries = _ensure_explicit_constraint_subquery(
            subqueries,
            original_query=query,
            constraints=constraints,
            selected_sources=selected_sources,
            max_subqueries=max_subqueries,
        )
        subqueries, query_planning = summarize_current_rules_planning(
            query_analysis,
            subqueries,
        )
        if options.query_planning_policy == "current_plus_disjunctive":
            subqueries, query_planning = plan_current_plus_disjunctive(
                query_analysis,
                current_subqueries=subqueries,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )
        if options.query_planning_policy == "facet_union":
            subqueries, query_planning = plan_facet_union(
                query_analysis,
                current_subqueries=subqueries,
                selected_sources=selected_sources,
                max_subqueries=max_subqueries,
            )

    return SearchPlan(
        query_analysis=query_analysis,
        subqueries=subqueries,
        selected_sources=selected_sources,
        limit_per_source=limit_per_source,
        top_k=options.top_k,
        run_profile=options.run_profile,
        enable_refchain=options.enable_refchain,
        enable_semantic_seed_expansion=options.enable_semantic_seed_expansion,
        enable_query_evolution=options.enable_query_evolution,
        query_planning_policy=options.query_planning_policy,
        query_planning=query_planning,
        warnings=_dedupe(warnings),
    )


def _constraints_from_llm_json(
    raw_constraints: dict[str, Any],
    *,
    domain: str,
    rule_constraints: QueryConstraint,
    warnings: list[str],
) -> QueryConstraint:
    return QueryConstraint(
        time_range=_time_range_from_llm_json(
            raw_constraints.get("time_range"),
            fallback=rule_constraints.time_range,
            warnings=warnings,
        ),
        venues=_dedupe(
            _coerce_string_list(raw_constraints.get("venues"))
            or list(rule_constraints.venues)
        ),
        methods=_dedupe(
            _coerce_string_list(raw_constraints.get("methods"))
            or list(rule_constraints.methods)
        ),
        datasets=_dedupe(
            _coerce_string_list(raw_constraints.get("datasets"))
            or list(rule_constraints.datasets)
        ),
        domains=_dedupe([domain]),
        must_include_terms=_dedupe(
            _coerce_string_list(raw_constraints.get("must_include_terms"))
            or _coerce_string_list(raw_constraints.get("must_have_terms"))
            or list(rule_constraints.must_include_terms)
        ),
        exclude_terms=_dedupe(
            _coerce_string_list(raw_constraints.get("exclude_terms"))
            or _coerce_string_list(raw_constraints.get("excluded_terms"))
            or list(rule_constraints.exclude_terms)
        ),
        paper_types=_paper_types_from_llm_json(
            raw_constraints.get("paper_types"),
            fallback=list(rule_constraints.paper_types),
            warnings=warnings,
        ),
    )


def _paper_types_from_llm_json(
    value: Any,
    *,
    fallback: list[str],
    warnings: list[str],
) -> list[str]:
    if value is None:
        return fallback
    try:
        normalized = normalize_paper_types(value)
    except (TypeError, ValueError):
        warnings.append("llm_invalid_paper_types")
        return fallback
    return normalized or fallback


def _time_range_from_llm_json(
    raw_time_range: Any,
    *,
    fallback: TimeRange | None,
    warnings: list[str],
) -> TimeRange | None:
    if raw_time_range is None:
        return fallback
    if not isinstance(raw_time_range, dict):
        warnings.append("llm_invalid_time_range")
        return fallback

    start_year = _coerce_year(raw_time_range.get("start_year"))
    end_year = _coerce_year(raw_time_range.get("end_year"))
    label = raw_time_range.get("label")
    try:
        return TimeRange(
            start_year=start_year,
            end_year=end_year,
            label=str(label).strip() if label else None,
        )
    except ValueError:
        warnings.append("llm_invalid_time_range")
        return fallback


def _subqueries_from_llm_json(
    raw_subqueries: Any,
    *,
    selected_sources: list[str],
    fallback: list[SearchSubquery],
    max_subqueries: int,
    warnings: list[str],
) -> list[SearchSubquery]:
    if not isinstance(raw_subqueries, list):
        warnings.append("llm_invalid_subqueries")
        return fallback

    subqueries: list[SearchSubquery] = []
    seen: set[str] = set()
    for index, raw_subquery in enumerate(raw_subqueries):
        if isinstance(raw_subquery, str):
            raw_query = raw_subquery
            raw_source_hints: Any = selected_sources
            raw_priority: Any = index + 1
            raw_purpose: Any = "llm_generated"
        elif isinstance(raw_subquery, dict):
            raw_query = raw_subquery.get("query")
            raw_source_hints = raw_subquery.get("source_hints") or selected_sources
            raw_priority = raw_subquery.get("priority", index + 1)
            raw_purpose = raw_subquery.get("purpose") or "llm_generated"
        else:
            warnings.append("llm_invalid_subquery")
            continue

        normalized_query = _normalize_query(str(raw_query or ""))
        key = normalized_query.casefold()
        if not normalized_query or key in seen:
            continue
        source_hints = _filter_llm_sources(
            raw_source_hints,
            fallback=selected_sources,
            warnings=warnings,
            warning_prefix="llm_subquery_source",
        )
        priority = _coerce_priority(raw_priority, fallback=min(index + 1, 5))
        subqueries.append(
            SearchSubquery(
                query=normalized_query,
                source_hints=source_hints,
                priority=priority,
                purpose=_normalize_query(str(raw_purpose)) or "llm_generated",
            )
        )
        seen.add(key)
        if len(subqueries) >= min(max_subqueries, 5):
            break

    if not subqueries:
        warnings.append("llm_empty_subqueries")
        return fallback
    return subqueries


def _filter_llm_sources(
    raw_sources: Any,
    *,
    fallback: list[str],
    warnings: list[str],
    warning_prefix: str,
) -> list[str]:
    if raw_sources is None:
        return list(fallback)
    if isinstance(raw_sources, str):
        candidates = [raw_sources]
    else:
        try:
            candidates = list(raw_sources)
        except TypeError:
            warnings.append(f"{warning_prefix}_invalid")
            return list(fallback)

    allowed: list[str] = []
    seen: set[str] = set()
    for source in candidates:
        key = str(source).strip().lower().replace("-", "_").replace(" ", "_")
        if key == "semanticscholar":
            key = "semantic_scholar"
        if not key or key in seen:
            continue
        seen.add(key)
        if key == "semantic_scholar" and not os.getenv(
            "SEMANTIC_SCHOLAR_API_KEY"
        ):
            warnings.append(f"{warning_prefix}_unavailable_without_key:{key}")
            continue
        if key in {"openalex", "arxiv", "semantic_scholar", "pubmed"}:
            allowed.append(key)
        else:
            warnings.append(f"{warning_prefix}_unsupported:{key}")

    if not allowed:
        warnings.append(f"{warning_prefix}_fallback_default")
        return list(fallback)
    return allowed


def _coerce_literal(
    value: Any,
    *,
    allowed: set[str],
    fallback: str,
    warning_prefix: str,
    warnings: list[str],
) -> str:
    normalized = str(value).strip().lower() if value is not None else ""
    normalized = re.sub(r"[\s\-/]+", "_", normalized).strip("_")
    aliases = {
        "method_compare": "method_comparison",
        "benchmark": "benchmark_or_dataset",
        "dataset": "benchmark_or_dataset",
        "recent_methods": "recent_progress",
        "find_recent_methods": "recent_progress",
        "find_papers": "paper_finding",
        "paper_search": "paper_finding",
        "information_retrieval": "computer_science",
        "computer_science_information_retrieval": "computer_science",
        "cs": "computer_science",
        "ml": "machine_learning",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in allowed:
        return normalized
    if value is not None:
        warnings.append(f"{warning_prefix}:{_diagnostic_message(value)}")
    return fallback


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return []


def _coerce_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if 1800 <= year <= 2200:
        return year
    return None


def _coerce_priority(value: Any, *, fallback: int) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        priority = fallback
    return min(max(priority, 1), 5)


def _diagnostic_message(value: Any) -> str:
    message = str(value).replace("\n", " ").strip()
    return message[:160] or "unknown"
