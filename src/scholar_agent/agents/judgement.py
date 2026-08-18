"""Rule-based relevance judgement for retrieved paper metadata."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.result_lineage import opaque_query_identity
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementCategory,
    JudgementFeatureVector,
    LexicalNormalizationFacet,
    LexicalNormalizationMatch,
    LexicalNormalizationPolicy,
    JudgementPolicy,
    JudgementResult,
    JudgementRuleConfig,
    QueryAnalysis,
    QueryConstraint,
    ResearchDomain,
)
from scholar_agent.agents.lexical_normalization import (
    LEXICAL_NORMALIZATION_VERSION,
    find_lexical_normalization_match,
)
from scholar_agent.agents.judgement_config import (
    CURRENT_RULES_CONFIG,
    judgement_config_hash,
    resolve_judgement_config,
)
from scholar_agent.agents.query_planning import identify_query_facets
from scholar_agent.llm.provider import chat_json as provider_chat_json
from scholar_agent.llm.provider import is_llm_enabled
from scholar_agent.prompts.loader import (
    PromptLoadError,
    render_untrusted_metadata_messages,
)
from scholar_agent.core.untrusted_metadata import (
    UntrustedMetadataObserver,
    build_llm_paper_payload,
    safe_diagnostic_message,
)

# Keep the established test/extension seam while routing this metadata-bearing
# prompt through the stricter renderer.
render_messages = render_untrusted_metadata_messages


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
    "of",
    "on",
    "paper",
    "papers",
    "recent",
    "research",
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

DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "machine_learning": (
        "llm",
        "large language model",
        "rag",
        "retrieval",
        "reranking",
        "transformer",
        "nlp",
        "machine learning",
        "deep learning",
        "neural",
        "embedding",
    ),
    "computer_science": (
        "algorithm",
        "database",
        "information retrieval",
        "retrieval",
        "software",
        "systems",
        "indexing",
    ),
    "biomedical": (
        "biomedical",
        "clinical",
        "gene",
        "genomic",
        "medicine",
        "patient",
        "protein",
        "therapy",
    ),
    "general_science": (
        "evidence",
        "literature",
        "method",
        "research",
        "science",
    ),
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
JUDGEMENT_CATEGORY_ORDER = {
    "highly_relevant": 0,
    "partially_relevant": 1,
    "weakly_relevant": 2,
    "irrelevant": 3,
    "insufficient_evidence": 4,
}

PRODUCTION_CONSTRAINT_PREDICATE_VERSION = "constraint-predicates-current-rules-v1"
PRODUCTION_CONSTRAINT_ORDER = (
    "exclude_terms",
    "must_include_terms",
    "methods",
    "datasets",
    "domains",
    "paper_types",
    "venues",
    "time_range",
)

LLM_JUDGEMENT_BATCH_SIZE_ENV = "SCHOLAR_AGENT_LLM_JUDGEMENT_BATCH_SIZE"
LLM_JUDGEMENT_MAX_PAPERS_ENV = "SCHOLAR_AGENT_LLM_JUDGEMENT_MAX_PAPERS"
LLM_JUDGEMENT_TIMEOUT_SECONDS_ENV = "SCHOLAR_AGENT_LLM_JUDGEMENT_TIMEOUT_SECONDS"
DEFAULT_LLM_JUDGEMENT_BATCH_SIZE = 8
DEFAULT_LLM_JUDGEMENT_MAX_PAPERS = 8
DEFAULT_LLM_JUDGEMENT_TIMEOUT_SECONDS = 25.0
MIN_LLM_JUDGEMENT_BATCH_SIZE = 1
MAX_LLM_JUDGEMENT_BATCH_SIZE = 20
JUDGEMENT_CATEGORIES: set[str] = {
    "highly_relevant",
    "partially_relevant",
    "weakly_relevant",
    "irrelevant",
    "insufficient_evidence",
}
EVIDENCE_SOURCES: set[str] = {"title", "abstract", "venue", "metadata"}


@dataclass(frozen=True)
class _Signal:
    score: float
    matched_terms: list[str]
    evidence: list[EvidenceItem]
    reasons: list[str]
    penalty: float = 0.0
    title_score: float = 0.0
    abstract_score: float = 0.0
    lexical_matches: tuple[LexicalNormalizationMatch, ...] = ()


class LLMJsonClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ...


class _StrictLLMEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Any = None
    text: Any = None
    confidence: Any = None


class _StrictLLMJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_index: Any = None
    score: Any = None
    category: Any = None
    reasoning: Any = None
    evidence: list[_StrictLLMEvidence] | _StrictLLMEvidence | None = None
    matched_terms: Any = None
    warnings: Any = None


class _StrictLLMJudgementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[_StrictLLMJudgement]
    warnings: Any = None


class JudgementAgent:
    """Metadata-only relevance judgement with optional LLM JSON enhancement."""

    def __init__(
        self,
        llm_client: LLMJsonClient | None = None,
        *,
        policy: JudgementPolicy = "current_rules",
        config: JudgementRuleConfig | None = None,
        metadata_observer: UntrustedMetadataObserver | None = None,
    ) -> None:
        self._llm_client = llm_client
        self.policy = policy
        self.config = resolve_judgement_config(policy, config)
        self.llm_call_count = 0
        self._metadata_observer = metadata_observer

    def judge(
        self,
        query_analysis: QueryAnalysis,
        papers: list[Paper],
        *,
        threshold_high: float | None = None,
        threshold_partial: float | None = None,
        threshold_weak: float | None = None,
        use_llm: bool | None = None,
        before_llm_batch: Callable[[], None] | None = None,
    ) -> list[JudgementResult]:
        config = _config_with_threshold_overrides(
            self.config,
            threshold_high=threshold_high,
            threshold_partial=threshold_partial,
            threshold_weak=threshold_weak,
        )
        threshold_high = config.highly_relevant_threshold
        threshold_partial = config.partially_relevant_threshold
        threshold_weak = config.weakly_relevant_threshold
        if use_llm:
            results = self._judge_with_optional_llm(
                query_analysis,
                papers,
                threshold_high=threshold_high,
                threshold_partial=threshold_partial,
                threshold_weak=threshold_weak,
                config=config,
                before_llm_batch=before_llm_batch,
            )
        else:
            results = _judge_papers_rules(
                query_analysis,
                papers,
                threshold_high=threshold_high,
                threshold_partial=threshold_partial,
                threshold_weak=threshold_weak,
                config=config,
            )
        return [
            _enforce_constraint_outcomes(
                query_analysis,
                result,
                threshold_high=threshold_high,
                threshold_partial=threshold_partial,
                threshold_weak=threshold_weak,
                config=config,
            )
            for result in results
        ]

    def _judge_with_optional_llm(
        self,
        query_analysis: QueryAnalysis,
        papers: list[Paper],
        *,
        threshold_high: float,
        threshold_partial: float,
        threshold_weak: float,
        config: JudgementRuleConfig,
        before_llm_batch: Callable[[], None] | None,
    ) -> list[JudgementResult]:
        rule_results = _judge_papers_rules(
            query_analysis,
            papers,
            threshold_high=threshold_high,
            threshold_partial=threshold_partial,
            threshold_weak=threshold_weak,
            config=config,
        )
        if not papers:
            return []
        if self._llm_client is None and not is_llm_enabled():
            return _with_warning(rule_results, "llm_judgement_disabled")

        batch_size = llm_judgement_batch_size_from_env()
        llm_limit = min(len(papers), llm_judgement_max_papers_from_env())
        llm_timeout = llm_judgement_timeout_seconds_from_env()
        results: list[JudgementResult] = list(rule_results)
        for skipped_index in range(llm_limit, len(papers)):
            results[skipped_index] = _with_warning(
                [results[skipped_index]],
                f"llm_judgement_skipped_by_limit:{skipped_index}",
            )[0]

        for start in range(0, llm_limit, batch_size):
            if before_llm_batch is not None:
                before_llm_batch()
            end = min(start + batch_size, llm_limit)
            batch_papers = papers[start:end]
            batch_rule_results = rule_results[start:end]
            try:
                messages = _build_llm_judgement_messages(
                    query_analysis,
                    batch_papers,
                    start_index=start,
                    metadata_observer=self._metadata_observer,
                )
            except PromptLoadError:
                results[start:end] = _with_warning(
                    batch_rule_results,
                    "llm_judgement_prompt_load_failed",
                )
                continue
            try:
                self.llm_call_count += 1
                raw_response = _chat_json(
                    self._llm_client,
                    messages,
                    timeout=llm_timeout,
                )
                batch_results = _judgement_results_from_llm_json(
                    raw_response,
                    query_analysis=query_analysis,
                    papers=batch_papers,
                    rule_results=batch_rule_results,
                    start_index=start,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one failed LLM batch
                batch_results = _with_warning(
                    batch_rule_results,
                    f"llm_judgement_failed:{_diagnostic_message(exc)}",
                )
            results[start:end] = batch_results
        return results


def judge_papers(
    query_analysis: QueryAnalysis,
    papers: list[Paper],
    *,
    threshold_high: float | None = None,
    threshold_partial: float | None = None,
    threshold_weak: float | None = None,
    use_llm: bool | None = None,
    llm_client: LLMJsonClient | None = None,
    before_llm_batch: Callable[[], None] | None = None,
    policy: JudgementPolicy = "current_rules",
    config: JudgementRuleConfig | None = None,
    metadata_observer: UntrustedMetadataObserver | None = None,
) -> list[JudgementResult]:
    """Judge paper relevance using metadata rules or optional LLM JSON."""

    return JudgementAgent(
        llm_client=llm_client,
        policy=policy,
        config=config,
        metadata_observer=metadata_observer,
    ).judge(
        query_analysis,
        papers,
        threshold_high=threshold_high,
        threshold_partial=threshold_partial,
        threshold_weak=threshold_weak,
        use_llm=use_llm,
        before_llm_batch=before_llm_batch,
    )


def llm_judgement_batch_size_from_env() -> int:
    import os

    raw_value = os.getenv(LLM_JUDGEMENT_BATCH_SIZE_ENV)
    if raw_value is None:
        return DEFAULT_LLM_JUDGEMENT_BATCH_SIZE
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_LLM_JUDGEMENT_BATCH_SIZE
    return max(MIN_LLM_JUDGEMENT_BATCH_SIZE, min(MAX_LLM_JUDGEMENT_BATCH_SIZE, value))


def llm_judgement_max_papers_from_env() -> int:
    import os

    raw_value = os.getenv(LLM_JUDGEMENT_MAX_PAPERS_ENV)
    if raw_value is None:
        return DEFAULT_LLM_JUDGEMENT_MAX_PAPERS
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_LLM_JUDGEMENT_MAX_PAPERS
    return value if value >= 1 else DEFAULT_LLM_JUDGEMENT_MAX_PAPERS


def llm_judgement_timeout_seconds_from_env() -> float:
    import os

    raw_value = os.getenv(LLM_JUDGEMENT_TIMEOUT_SECONDS_ENV)
    if raw_value is None:
        return DEFAULT_LLM_JUDGEMENT_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_LLM_JUDGEMENT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_LLM_JUDGEMENT_TIMEOUT_SECONDS


def _judge_papers_rules(
    query_analysis: QueryAnalysis,
    papers: list[Paper],
    *,
    threshold_high: float,
    threshold_partial: float,
    threshold_weak: float,
    config: JudgementRuleConfig,
) -> list[JudgementResult]:
    return [
        _judge_one_paper(
            query_analysis,
            paper,
            threshold_high=threshold_high,
            threshold_partial=threshold_partial,
            threshold_weak=threshold_weak,
            config=config,
        )
        for paper in papers
    ]


def _judge_one_paper(
    query_analysis: QueryAnalysis,
    paper: Paper,
    *,
    threshold_high: float,
    threshold_partial: float,
    threshold_weak: float,
    config: JudgementRuleConfig,
) -> JudgementResult:
    warnings = _metadata_warnings(query_analysis.constraints, paper)
    if not paper.title.strip() and not paper.abstract.strip():
        return JudgementResult(
            paper=paper,
            score=0.0,
            category="insufficient_evidence",
            reasoning=(
                "Both title and abstract are empty; metadata is insufficient "
                "for relevance judgement."
            ),
            evidence=[],
            matched_terms=[],
            warnings=warnings,
            feature_vector=_empty_feature_vector(
                query_analysis,
                paper,
                config,
                category_reason="missing_title_and_abstract",
            ),
        )

    constraints = query_analysis.constraints
    keyword_terms = _query_terms(query_analysis)
    topic_lexical_terms = _dedupe_terms(
        [
            *_tokenize(query_analysis.original_query),
            *constraints.must_include_terms,
            *constraints.methods,
        ]
    )
    keyword_signal = _term_signal(
        terms=keyword_terms,
        paper=paper,
        title_weight=config.title_topic_weight,
        abstract_weight=config.abstract_topic_weight,
        max_score=config.topic_max_score,
        reason_label="query terms",
        lexical_policy=config.lexical_normalization_policy,
        lexical_facet="topic",
        lexical_allowed_terms=topic_lexical_terms,
    )
    must_signal = _term_signal(
        terms=constraints.must_include_terms,
        paper=paper,
        title_weight=config.title_must_have_weight,
        abstract_weight=config.abstract_must_have_weight,
        max_score=config.must_have_max_score,
        reason_label="required terms",
        lexical_policy=config.lexical_normalization_policy,
        lexical_facet="must_have",
    )
    method_signal = _term_signal(
        terms=constraints.methods,
        paper=paper,
        title_weight=config.title_method_weight,
        abstract_weight=config.abstract_method_weight,
        max_score=config.method_max_score,
        reason_label="method terms",
        lexical_policy=config.lexical_normalization_policy,
        lexical_facet="method",
    )
    dataset_signal = _term_signal(
        terms=constraints.datasets,
        paper=paper,
        title_weight=config.title_dataset_weight,
        abstract_weight=config.abstract_dataset_weight,
        max_score=config.dataset_max_score,
        reason_label="dataset terms",
    )
    paper_type_signal = _paper_type_signal(constraints, paper, config)
    domain_signal = _term_signal(
        terms=list(DOMAIN_TERMS.get(query_analysis.domain, ())),
        paper=paper,
        title_weight=config.title_domain_weight,
        abstract_weight=config.abstract_domain_weight,
        max_score=config.domain_max_score,
        reason_label="domain terms",
        lexical_policy=config.lexical_normalization_policy,
        lexical_facet="domain",
    )
    venue_signal = _venue_signal(constraints, paper, config)
    time_signal = _time_signal(constraints, paper, config)

    metadata_completeness = _metadata_completeness(paper)
    score_components = {
        "topic_match": keyword_signal.score,
        "must_have_match": must_signal.score,
        "method_match": method_signal.score,
        "dataset_match": dataset_signal.score,
        "paper_type_match": paper_type_signal.score,
        "domain_match": domain_signal.score,
        "venue_match": venue_signal.score,
        "temporal_match": time_signal.score,
        "venue_mismatch_penalty": -venue_signal.penalty,
        "temporal_mismatch_penalty": -time_signal.penalty,
        "paper_type_mismatch_penalty": -paper_type_signal.penalty,
        "missing_abstract_penalty": (
            -config.missing_abstract_penalty if not paper.abstract.strip() else 0.0
        ),
        "missing_metadata_penalty": -config.missing_metadata_penalty
        * (1.0 - metadata_completeness),
    }
    score = sum(score_components.values())
    evidence = _dedupe_evidence(
        keyword_signal.evidence
        + must_signal.evidence
        + method_signal.evidence
        + dataset_signal.evidence
        + paper_type_signal.evidence
        + domain_signal.evidence
        + venue_signal.evidence
        + time_signal.evidence
    )
    matched_terms = _dedupe_terms(
        keyword_signal.matched_terms
        + must_signal.matched_terms
        + method_signal.matched_terms
        + dataset_signal.matched_terms
        + paper_type_signal.matched_terms
        + domain_signal.matched_terms
    )
    reasons = (
        keyword_signal.reasons
        + must_signal.reasons
        + method_signal.reasons
        + dataset_signal.reasons
        + paper_type_signal.reasons
        + domain_signal.reasons
        + venue_signal.reasons
        + time_signal.reasons
    )
    score, coverage_reasons = _constraint_coverage_adjustment(
        query_analysis,
        paper,
        score,
        config,
    )
    score_components["constraint_coverage_adjustment"] = score - sum(
        score_components.values()
    )
    reasons.extend(coverage_reasons)
    clamped_score = _clamp(score)
    score_components["clamp_adjustment"] = clamped_score - score
    score = round(clamped_score, 4)

    category = _category(
        score,
        threshold_high=threshold_high,
        threshold_partial=threshold_partial,
        threshold_weak=threshold_weak,
    )
    category_reason = f"score_threshold:{category}"
    if len(evidence) < config.minimum_evidence_count:
        category = "insufficient_evidence"
        category_reason = "minimum_evidence_count_not_met"
        reasons.append(category_reason)
    topic_matches, _ = _matching_constraint_terms(
        keyword_terms,
        paper,
        lexical_policy=config.lexical_normalization_policy,
        lexical_allowed_terms=topic_lexical_terms,
    )
    method_matches, _ = _matching_constraint_terms(
        constraints.methods,
        paper,
        lexical_policy=config.lexical_normalization_policy,
    )
    dataset_matches, _ = _matching_constraint_terms(constraints.datasets, paper)
    must_matches, _ = _matching_constraint_terms(
        constraints.must_include_terms,
        paper,
        lexical_policy=config.lexical_normalization_policy,
    )
    paper_type_matches, _ = _matched_paper_types(constraints.paper_types, paper)
    task_terms = _facet_terms(query_analysis, "task")
    task_matches, _ = _matching_constraint_terms(task_terms, paper)
    title_terms, abstract_terms = _matching_terms_by_field(
        _dedupe_terms(
            [
                *keyword_terms,
                *constraints.must_include_terms,
                *constraints.methods,
                *constraints.datasets,
                *task_terms,
            ]
        ),
        paper,
        lexical_policy=config.lexical_normalization_policy,
        lexical_allowed_terms=_dedupe_terms(
            [
                *topic_lexical_terms,
                *constraints.must_include_terms,
                *constraints.methods,
            ]
        ),
    )
    constraint_results = _constraint_results(query_analysis, paper, config)
    feature_vector = JudgementFeatureVector(
        config_version=config.config_version,
        config_hash=judgement_config_hash(config),
        matched_topic_terms=topic_matches,
        matched_method_terms=method_matches,
        matched_dataset_terms=dataset_matches,
        matched_task_terms=task_matches,
        matched_must_have_terms=must_matches,
        matched_paper_types=paper_type_matches,
        lexical_normalization_matches=[
            match
            for signal in (
                keyword_signal,
                must_signal,
                method_signal,
                domain_signal,
            )
            for match in signal.lexical_matches
        ],
        title_matched_terms=title_terms,
        abstract_matched_terms=abstract_terms,
        title_match_score=round(
            sum(
                signal.title_score
                for signal in (
                    keyword_signal,
                    must_signal,
                    method_signal,
                    dataset_signal,
                    domain_signal,
                )
            ),
            6,
        ),
        abstract_match_score=round(
            sum(
                signal.abstract_score
                for signal in (
                    keyword_signal,
                    must_signal,
                    method_signal,
                    dataset_signal,
                    domain_signal,
                )
            ),
            6,
        ),
        venue_match=constraint_results.get("venue"),
        temporal_match=constraint_results.get("time"),
        metadata_completeness=metadata_completeness,
        constraint_results=constraint_results,
        hard_constraint_failures=_hard_constraint_failures(constraint_results),
        score_components={
            key: round(value, 6) for key, value in score_components.items()
        },
        evidence_count=len(evidence),
        final_score=score,
        highly_relevant_threshold=threshold_high,
        partially_relevant_threshold=threshold_partial,
        weakly_relevant_threshold=threshold_weak,
        category_reason=category_reason,
    )
    return JudgementResult(
        paper=paper,
        score=score,
        category=category,
        reasoning=_reasoning(reasons, evidence, warnings),
        evidence=evidence,
        matched_terms=matched_terms,
        warnings=warnings,
        feature_vector=feature_vector,
    )


def _constraint_coverage_adjustment(
    query_analysis: QueryAnalysis,
    paper: Paper,
    score: float,
    config: JudgementRuleConfig,
) -> tuple[float, list[str]]:
    constraints = query_analysis.constraints
    topic_terms = _tokenize(query_analysis.original_query)
    topic_matches, _ = _matching_constraint_terms(
        topic_terms,
        paper,
        lexical_policy=config.lexical_normalization_policy,
    )
    method_matches, _ = _matching_constraint_terms(
        constraints.methods,
        paper,
        lexical_policy=config.lexical_normalization_policy,
    )
    dataset_matches, _ = _matching_constraint_terms(constraints.datasets, paper)
    must_matches, _ = _matching_constraint_terms(
        constraints.must_include_terms,
        paper,
        lexical_policy=config.lexical_normalization_policy,
    )
    paper_type_matches, _ = _matched_paper_types(constraints.paper_types, paper)
    excluded_matches, _ = _matching_constraint_terms(constraints.exclude_terms, paper)

    coverages: list[tuple[str, float]] = []
    if topic_terms:
        coverages.append(("topic", _coverage_ratio(topic_terms, topic_matches)))
    if constraints.methods:
        coverages.append(
            ("method", _coverage_ratio(constraints.methods, method_matches))
        )
    if constraints.datasets:
        coverages.append(
            ("dataset", _coverage_ratio(constraints.datasets, dataset_matches))
        )

    topic_keys = {term.casefold() for term in topic_terms}
    must_keys = {term.casefold() for term in constraints.must_include_terms}
    if constraints.must_include_terms and (
        "must_include_terms" in constraints.explicit_fields or must_keys != topic_keys
    ):
        coverages.append(
            (
                "must_have",
                _coverage_ratio(constraints.must_include_terms, must_matches),
            )
        )
    if constraints.paper_types:
        coverages.append(
            (
                "paper_type",
                _coverage_ratio(constraints.paper_types, paper_type_matches),
            )
        )
    if constraints.venues:
        venue_key = _normalize_venue(paper.venue or "")
        venue_match = bool(venue_key) and any(
            _normalize_venue(expected) in venue_key
            for expected in constraints.venues
        )
        coverages.append(("venue", 1.0 if venue_match else 0.0))
    if constraints.time_range is not None:
        time_match = paper.year is not None and not _outside_time_range(
            constraints,
            paper,
        )
        coverages.append(("time", 1.0 if time_match else 0.0))

    reasons = [
        "constraint_coverage:"
        + ",".join(f"{name}={coverage:.2f}" for name, coverage in coverages)
        + f",excluded_term_match={'true' if excluded_matches else 'false'}"
    ]
    if not coverages:
        return score, reasons

    strong_dimensions = sum(coverage >= 0.9 for _, coverage in coverages)
    mean_coverage = sum(coverage for _, coverage in coverages) / len(coverages)
    broad_topic_only = len(topic_terms) >= 3 and len(topic_matches) <= 1

    if len(coverages) >= 2 and strong_dimensions >= 2 and mean_coverage >= 0.6:
        score += min(
            config.multi_dimension_bonus_cap,
            strong_dimensions * config.multi_dimension_bonus,
        )
        reasons.append("multi_dimension_constraint_coverage")
    elif len(coverages) >= 2 and strong_dimensions <= 1:
        score = min(
            score - config.insufficient_coverage_penalty,
            0.5 + 0.25 * mean_coverage,
        )
        reasons.append("insufficient_multi_dimension_coverage")

    if broad_topic_only and strong_dimensions <= 1:
        score = min(score, config.broad_topic_score_cap)
        reasons.append("broad_topic_match_only")
    return score, reasons


def _coverage_ratio(expected: list[str], matched: list[str]) -> float:
    expected_keys = {item.casefold() for item in expected if item.strip()}
    if not expected_keys:
        return 0.0
    matched_keys = {item.casefold() for item in matched if item.strip()}
    return len(expected_keys & matched_keys) / len(expected_keys)


def _metadata_warnings(constraints: QueryConstraint, paper: Paper) -> list[str]:
    warnings: list[str] = []
    if not paper.title.strip():
        warnings.append("missing_title")
    if not paper.abstract.strip():
        warnings.append("missing_abstract")
    if constraints.time_range is not None and paper.year is None:
        warnings.append("missing_year_for_time_range")
    return warnings


def _query_terms(query_analysis: QueryAnalysis) -> list[str]:
    terms = _tokenize(query_analysis.original_query)
    constraints = query_analysis.constraints
    terms.extend(constraints.must_include_terms)
    terms.extend(constraints.methods)
    terms.extend(constraints.datasets)
    return _dedupe_terms(terms)


def _term_signal(
    *,
    terms: list[str],
    paper: Paper,
    title_weight: float,
    abstract_weight: float,
    max_score: float,
    reason_label: str,
    lexical_policy: LexicalNormalizationPolicy = "off",
    lexical_facet: LexicalNormalizationFacet | None = None,
    lexical_allowed_terms: list[str] | None = None,
) -> _Signal:
    title = paper.title or ""
    abstract = paper.abstract or ""
    title_text = title.casefold()
    abstract_text = abstract.casefold()
    score = 0.0
    title_score = 0.0
    abstract_score = 0.0
    matched_terms: list[str] = []
    evidence: list[EvidenceItem] = []
    lexical_matches: list[LexicalNormalizationMatch] = []
    allowed_lexical_keys = {
        term.casefold()
        for term in (
            lexical_allowed_terms if lexical_allowed_terms is not None else terms
        )
    }

    for term in _dedupe_terms(terms):
        normalized_term = term.casefold()
        if not normalized_term:
            continue
        matched_field: Literal["title", "abstract"] | None = None
        normalization_evidence = None
        if _contains_term(title_text, normalized_term):
            matched_field = "title"
        elif _contains_term(abstract_text, normalized_term):
            matched_field = "abstract"
        elif (
            lexical_policy == "lexical_normalization_v1"
            and lexical_facet is not None
            and term.casefold() in allowed_lexical_keys
        ):
            normalization_evidence = find_lexical_normalization_match(
                term,
                title=title,
                abstract=abstract,
            )
            if normalization_evidence is not None:
                matched_field = normalization_evidence.field
        if matched_field is None:
            continue
        weight = title_weight if matched_field == "title" else abstract_weight
        capped_before = min(score, max_score)
        score += weight
        capped_after = min(score, max_score)
        score_impact = capped_after - capped_before
        if matched_field == "title":
            title_score += title_weight
            evidence.append(
                EvidenceItem(source="title", text=_short_text(title), confidence=0.9)
            )
        else:
            abstract_score += abstract_weight
            snippet = (
                _short_text(abstract)
                if normalization_evidence is not None
                else _abstract_snippet(abstract, normalized_term)
            )
            evidence.append(
                EvidenceItem(source="abstract", text=snippet, confidence=0.72)
            )
        matched_terms.append(term)
        if normalization_evidence is not None:
            lexical_matches.append(
                LexicalNormalizationMatch(
                    policy_version=LEXICAL_NORMALIZATION_VERSION,
                    facet=lexical_facet,
                    original_term=normalization_evidence.original_term,
                    normalized_form=normalization_evidence.normalized_form,
                    field=normalization_evidence.field,
                    score_impact=round(score_impact, 6),
                )
            )

    capped_score = min(score, max_score)
    reasons = []
    if matched_terms:
        reasons.append(
            f"matched {reason_label}: {', '.join(_dedupe_terms(matched_terms)[:8])}"
        )
    scale = min(1.0, max_score / score) if score > 0 else 1.0
    return _Signal(
        score=capped_score,
        matched_terms=_dedupe_terms(matched_terms),
        evidence=evidence,
        reasons=reasons,
        title_score=title_score * scale,
        abstract_score=abstract_score * scale,
        lexical_matches=tuple(lexical_matches),
    )


def _paper_type_signal(
    constraints: QueryConstraint,
    paper: Paper,
    config: JudgementRuleConfig,
) -> _Signal:
    if not constraints.paper_types:
        return _Signal(score=0.0, matched_terms=[], evidence=[], reasons=[])
    matched_types, evidence = _matched_paper_types(constraints.paper_types, paper)
    if not matched_types:
        return _Signal(
            score=0.0,
            matched_terms=[],
            evidence=[],
            reasons=[
                "paper type does not match requested types: "
                + ", ".join(constraints.paper_types)
            ],
            penalty=config.paper_type_mismatch_penalty,
        )
    return _Signal(
        score=min(
            config.paper_type_match_weight * len(matched_types),
            config.paper_type_max_score,
        ),
        matched_terms=matched_types,
        evidence=evidence,
        reasons=["matched paper types: " + ", ".join(matched_types)],
    )


def _venue_signal(
    constraints: QueryConstraint,
    paper: Paper,
    config: JudgementRuleConfig,
) -> _Signal:
    if not constraints.venues:
        return _Signal(score=0.0, matched_terms=[], evidence=[], reasons=[])

    venue = (paper.venue or "").strip()
    if not venue:
        return _Signal(
            score=0.0,
            matched_terms=[],
            evidence=[],
            reasons=["venue constraint present but paper venue is missing"],
            penalty=config.venue_mismatch_penalty,
        )

    venue_key = _normalize_venue(venue)
    for expected in constraints.venues:
        if _normalize_venue(expected) in venue_key:
            return _Signal(
                score=config.venue_match_weight,
                matched_terms=[],
                evidence=[EvidenceItem(source="venue", text=venue, confidence=0.92)],
                reasons=[f"venue matches constraint: {expected}"],
            )
    return _Signal(
        score=0.0,
        matched_terms=[],
        evidence=[],
        reasons=["paper venue does not match requested venues"],
        penalty=config.venue_mismatch_penalty,
    )


def _time_signal(
    constraints: QueryConstraint,
    paper: Paper,
    config: JudgementRuleConfig,
) -> _Signal:
    time_range = constraints.time_range
    if time_range is None:
        return _Signal(score=0.0, matched_terms=[], evidence=[], reasons=[])
    if paper.year is None:
        return _Signal(
            score=0.0,
            matched_terms=[],
            evidence=[],
            reasons=["time range present but paper year is missing"],
        )

    start_year = time_range.start_year
    end_year = time_range.end_year
    if start_year is not None and paper.year < start_year:
        distance = start_year - paper.year
        penalty = (
            config.temporal_early_penalty
            if distance >= 2
            else config.temporal_near_penalty
        )
        return _Signal(
            score=0.0,
            matched_terms=[],
            evidence=[EvidenceItem(source="metadata", text=f"year={paper.year}", confidence=0.85)],
            reasons=[f"paper year {paper.year} is earlier than requested start year {start_year}"],
            penalty=penalty,
        )
    if end_year is not None and paper.year > end_year:
        return _Signal(
            score=0.0,
            matched_terms=[],
            evidence=[EvidenceItem(source="metadata", text=f"year={paper.year}", confidence=0.85)],
            reasons=[f"paper year {paper.year} is later than requested end year {end_year}"],
            penalty=config.temporal_late_penalty,
        )
    return _Signal(
        score=config.temporal_match_weight,
        matched_terms=[],
        evidence=[EvidenceItem(source="metadata", text=f"year={paper.year}", confidence=0.88)],
        reasons=[f"paper year {paper.year} satisfies time constraint"],
    )


def _enforce_constraint_outcomes(
    query_analysis: QueryAnalysis,
    result: JudgementResult,
    *,
    threshold_high: float,
    threshold_partial: float,
    threshold_weak: float,
    config: JudgementRuleConfig,
) -> JudgementResult:
    constraints = query_analysis.constraints
    score = result.score
    category = result.category
    warnings = list(result.warnings)
    evidence = list(result.evidence)
    matched_terms = list(result.matched_terms)
    reasoning_additions: list[str] = []
    rule_score = score

    excluded_matches, excluded_evidence = _matching_constraint_terms(
        constraints.exclude_terms,
        result.paper,
    )
    if excluded_matches:
        warning = "excluded_terms_matched:" + ",".join(excluded_matches)
        warnings.append(warning)
        reasoning_additions.append(
            "excluded terms matched: " + ", ".join(excluded_matches)
        )
        return result.model_copy(
            update={
                "score": 0.0,
                "category": "irrelevant",
                "reasoning": _append_reasoning(
                    result.reasoning,
                    reasoning_additions,
                ),
                "evidence": _dedupe_evidence([*evidence, *excluded_evidence]),
                "matched_terms": _dedupe_terms(
                    [*matched_terms, *excluded_matches]
                ),
                "warnings": _dedupe_terms(warnings),
                "feature_vector": _finalize_feature_vector(
                    result.feature_vector,
                    final_score=0.0,
                    category="irrelevant",
                    category_reason="excluded_term_hard_constraint",
                    hard_constraint_adjustment=-rule_score,
                    forced_failures=["excluded_terms"],
                ),
            }
        )

    if "must_include_terms" in constraints.explicit_fields:
        matched_required, required_evidence = _matching_constraint_terms(
            constraints.must_include_terms,
            result.paper,
            lexical_policy=config.lexical_normalization_policy,
        )
        matched_keys = {item.casefold() for item in matched_required}
        missing_required = [
            term
            for term in constraints.must_include_terms
            if term.casefold() not in matched_keys
        ]
        evidence.extend(required_evidence)
        matched_terms.extend(matched_required)
        if missing_required:
            warnings.append(
                "missing_must_have_terms:" + ",".join(missing_required)
            )
            reasoning_additions.append(
                "missing required terms: " + ", ".join(missing_required)
            )
            score = min(score, max(0.0, threshold_high - 0.0001))
            category = _worse_category(category, "partially_relevant")

    if constraints.datasets:
        matched_datasets, dataset_evidence = _matching_constraint_terms(
            constraints.datasets,
            result.paper,
        )
        evidence.extend(dataset_evidence)
        matched_terms.extend(matched_datasets)
        if (
            "datasets" in constraints.explicit_fields
            and not matched_datasets
        ):
            warnings.append(
                "dataset_terms_not_matched:" + ",".join(constraints.datasets)
            )
            reasoning_additions.append(
                "requested datasets not matched: " + ", ".join(constraints.datasets)
            )
            score = max(0.0, score - config.explicit_dataset_penalty)
            category = _worse_category(
                category,
                _category(
                    score,
                    threshold_high=threshold_high,
                    threshold_partial=threshold_partial,
                    threshold_weak=threshold_weak,
                ),
            )

    if constraints.paper_types:
        matched_types, type_evidence = _matched_paper_types(
            constraints.paper_types,
            result.paper,
        )
        evidence.extend(type_evidence)
        matched_terms.extend(matched_types)
        if not matched_types:
            warnings.append(
                "paper_types_not_matched:" + ",".join(constraints.paper_types)
            )
            reasoning_additions.append(
                "requested paper types not matched: "
                + ", ".join(constraints.paper_types)
            )
            score = max(0.0, score - config.paper_type_mismatch_penalty)
            category = _worse_category(
                category,
                _category(
                    score,
                    threshold_high=threshold_high,
                    threshold_partial=threshold_partial,
                    threshold_weak=threshold_weak,
                ),
            )

    if _outside_time_range(constraints, result.paper):
        warnings.append(f"outside_time_range:{result.paper.year}")
        reasoning_additions.append(
            f"paper year {result.paper.year} is outside the requested time range"
        )
        score = min(score, max(0.0, threshold_high - 0.0001))
        category = _worse_category(category, "partially_relevant")

    final_score = round(_clamp(score), 4)
    return result.model_copy(
        update={
            "score": final_score,
            "category": category,
            "reasoning": _append_reasoning(result.reasoning, reasoning_additions),
            "evidence": _dedupe_evidence(evidence),
            "matched_terms": _dedupe_terms(matched_terms),
            "warnings": _dedupe_terms(warnings),
            "feature_vector": _finalize_feature_vector(
                result.feature_vector,
                final_score=final_score,
                category=category,
                category_reason=(
                    "constraint_guard_applied"
                    if reasoning_additions
                    else result.feature_vector.category_reason
                    if result.feature_vector is not None
                    else f"score_threshold:{category}"
                ),
                hard_constraint_adjustment=final_score - rule_score,
                forced_failures=_hard_constraint_failures(
                    _constraint_results(query_analysis, result.paper, config)
                ),
            ),
        }
    )


def _matching_constraint_terms(
    terms: list[str],
    paper: Paper,
    *,
    lexical_policy: LexicalNormalizationPolicy = "off",
    lexical_allowed_terms: list[str] | None = None,
) -> tuple[list[str], list[EvidenceItem]]:
    title = paper.title or ""
    abstract = paper.abstract or ""
    title_key = title.casefold()
    abstract_key = abstract.casefold()
    matched: list[str] = []
    evidence: list[EvidenceItem] = []
    allowed_lexical_keys = {
        term.casefold()
        for term in (
            lexical_allowed_terms if lexical_allowed_terms is not None else terms
        )
    }
    for term in _dedupe_terms(terms):
        normalized = term.casefold()
        if _contains_term(title_key, normalized):
            matched.append(term)
            evidence.append(
                EvidenceItem(source="title", text=_short_text(title), confidence=0.92)
            )
        elif _contains_term(abstract_key, normalized):
            matched.append(term)
            evidence.append(
                EvidenceItem(
                    source="abstract",
                    text=_abstract_snippet(abstract, normalized),
                    confidence=0.8,
                )
            )
        elif (
            lexical_policy == "lexical_normalization_v1"
            and term.casefold() in allowed_lexical_keys
        ):
            normalized_match = find_lexical_normalization_match(
                term,
                title=title,
                abstract=abstract,
            )
            if normalized_match is None:
                continue
            matched.append(term)
            evidence.append(
                EvidenceItem(
                    source=normalized_match.field,
                    text=(
                        _short_text(title)
                        if normalized_match.field == "title"
                        else _short_text(abstract)
                    ),
                    confidence=(
                        0.92 if normalized_match.field == "title" else 0.8
                    ),
                )
            )
    return _dedupe_terms(matched), _dedupe_evidence(evidence)


def _matched_paper_types(
    paper_types: list[str],
    paper: Paper,
) -> tuple[list[str], list[EvidenceItem]]:
    matched: list[str] = []
    evidence: list[EvidenceItem] = []
    for paper_type in paper_types:
        aliases = list(PAPER_TYPE_TERMS.get(paper_type, (paper_type,)))
        alias_matches, alias_evidence = _matching_constraint_terms(aliases, paper)
        if not alias_matches:
            continue
        matched.append(paper_type)
        evidence.extend(alias_evidence)
    return _dedupe_terms(matched), _dedupe_evidence(evidence)


def _outside_time_range(constraints: QueryConstraint, paper: Paper) -> bool:
    time_range = constraints.time_range
    if time_range is None or paper.year is None:
        return False
    if time_range.start_year is not None and paper.year < time_range.start_year:
        return True
    return time_range.end_year is not None and paper.year > time_range.end_year


def _worse_category(current: str, candidate: str) -> str:
    if JUDGEMENT_CATEGORY_ORDER.get(candidate, 5) > JUDGEMENT_CATEGORY_ORDER.get(
        current,
        5,
    ):
        return candidate
    return current


def _append_reasoning(reasoning: str, additions: list[str]) -> str:
    if not additions:
        return reasoning
    base = reasoning.rstrip().rstrip(".")
    suffix = "; ".join(_dedupe_terms(additions))
    return f"{base}; {suffix}."


def _normalize_venue(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.casefold())


def _category(
    score: float,
    *,
    threshold_high: float,
    threshold_partial: float,
    threshold_weak: float,
) -> str:
    if score >= threshold_high:
        return "highly_relevant"
    if score >= threshold_partial:
        return "partially_relevant"
    if score >= threshold_weak:
        return "weakly_relevant"
    return "irrelevant"


PRODUCTION_JUDGEMENT_COMPONENT_ORDER: tuple[str, ...] = (
    "topic_match",
    "must_have_match",
    "method_match",
    "dataset_match",
    "paper_type_match",
    "domain_match",
    "venue_match",
    "temporal_match",
    "venue_mismatch_penalty",
    "temporal_mismatch_penalty",
    "paper_type_mismatch_penalty",
    "missing_abstract_penalty",
    "missing_metadata_penalty",
    "constraint_coverage_adjustment",
    "clamp_adjustment",
    "hard_constraint_adjustment",
)


def production_judgement_decision_catalog(
    config: JudgementRuleConfig | None = None,
) -> dict[str, Any]:
    """Describe current rule scoring without introducing a second scorer.

    This is an observational contract used by offline audits.  The component
    values themselves remain produced by :func:`judge_papers` and its feature
    vector; callers must not use this catalog to recompute production output.
    """

    resolved = config or CURRENT_RULES_CONFIG
    return {
        "version": "current-rules-judgement-decision-v1",
        "config_version": resolved.config_version,
        "config_hash": judgement_config_hash(resolved),
        "component_order": list(PRODUCTION_JUDGEMENT_COMPONENT_ORDER),
        "component_decimals": 6,
        "score_decimals": 4,
        "finite_only": True,
        "threshold_comparison": "greater_than_or_equal",
        "thresholds": {
            "highly_relevant": resolved.highly_relevant_threshold,
            "partially_relevant": resolved.partially_relevant_threshold,
            "weakly_relevant": resolved.weakly_relevant_threshold,
        },
        "special_category_reasons": [
            "missing_title_and_abstract",
            "minimum_evidence_count_not_met",
            "excluded_term_hard_constraint",
            "constraint_guard_applied",
        ],
    }


def production_category_from_score(
    score: float,
    config: JudgementRuleConfig | None = None,
) -> str:
    """Expose the exact production threshold predicate for boundary audits."""

    resolved = config or CURRENT_RULES_CONFIG
    return _category(
        score,
        threshold_high=resolved.highly_relevant_threshold,
        threshold_partial=resolved.partially_relevant_threshold,
        threshold_weak=resolved.weakly_relevant_threshold,
    )


def trace_judgement_decision(result: JudgementResult) -> dict[str, Any]:
    """Return the already-produced score trace without changing judgement."""

    feature = result.feature_vector
    if feature is None:
        return {
            "status": "feature_vector_missing",
            "score": result.score,
            "category": result.category,
            "components": {},
            "category_reason": "feature_vector_missing",
        }
    return {
        "status": "recorded",
        "score": result.score,
        "category": result.category,
        "components": {
            key: feature.score_components[key]
            for key in PRODUCTION_JUDGEMENT_COMPONENT_ORDER
            if key in feature.score_components
        },
        "category_reason": feature.category_reason,
        "thresholds": {
            "highly_relevant": feature.highly_relevant_threshold,
            "partially_relevant": feature.partially_relevant_threshold,
            "weakly_relevant": feature.weakly_relevant_threshold,
        },
        "hard_constraint_failures": list(feature.hard_constraint_failures),
        "constraint_results": dict(sorted(feature.constraint_results.items())),
    }


def _validate_thresholds(
    threshold_high: float,
    threshold_partial: float,
    threshold_weak: float,
) -> None:
    if not 0 <= threshold_weak <= threshold_partial <= threshold_high <= 1:
        raise ValueError(
            "thresholds must satisfy "
            "0 <= threshold_weak <= threshold_partial <= threshold_high <= 1"
        )


def _config_with_threshold_overrides(
    config: JudgementRuleConfig,
    *,
    threshold_high: float | None,
    threshold_partial: float | None,
    threshold_weak: float | None,
) -> JudgementRuleConfig:
    updates: dict[str, float] = {}
    if threshold_high is not None:
        updates["highly_relevant_threshold"] = threshold_high
    if threshold_partial is not None:
        updates["partially_relevant_threshold"] = threshold_partial
    if threshold_weak is not None:
        updates["weakly_relevant_threshold"] = threshold_weak
    resolved = config.model_copy(update=updates) if updates else config
    _validate_thresholds(
        resolved.highly_relevant_threshold,
        resolved.partially_relevant_threshold,
        resolved.weakly_relevant_threshold,
    )
    return resolved


def _facet_terms(query_analysis: QueryAnalysis, facet_type: str) -> list[str]:
    return _dedupe_terms(
        [
            term
            for facet in identify_query_facets(query_analysis)
            if facet.facet_type == facet_type
            for term in facet.terms
        ]
    )


def _matching_terms_by_field(
    terms: list[str],
    paper: Paper,
    *,
    lexical_policy: LexicalNormalizationPolicy = "off",
    lexical_allowed_terms: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    title = (paper.title or "").casefold()
    abstract = (paper.abstract or "").casefold()
    title_terms: list[str] = []
    abstract_terms: list[str] = []
    allowed_lexical_keys = {
        term.casefold()
        for term in (
            lexical_allowed_terms if lexical_allowed_terms is not None else terms
        )
    }
    for term in _dedupe_terms(terms):
        normalized = term.casefold()
        if _contains_term(title, normalized):
            title_terms.append(term)
        elif _contains_term(abstract, normalized):
            abstract_terms.append(term)
        elif (
            lexical_policy == "lexical_normalization_v1"
            and term.casefold() in allowed_lexical_keys
        ):
            normalized_match = find_lexical_normalization_match(
                term,
                title=paper.title,
                abstract=paper.abstract,
            )
            if normalized_match is None:
                continue
            if normalized_match.field == "title":
                title_terms.append(term)
            else:
                abstract_terms.append(term)
    return _dedupe_terms(title_terms), _dedupe_terms(abstract_terms)


def _metadata_completeness(paper: Paper) -> float:
    identifiers = paper.identifiers.model_dump(mode="json")
    present = (
        bool(paper.title.strip()),
        bool(paper.abstract.strip()),
        bool(paper.authors),
        paper.year is not None,
        any(bool(value) for value in identifiers.values()),
    )
    return round(sum(present) / len(present), 4)


def _constraint_results(
    query_analysis: QueryAnalysis,
    paper: Paper,
    config: JudgementRuleConfig,
) -> dict[str, bool | None]:
    constraints = query_analysis.constraints
    results: dict[str, bool | None] = {}
    if constraints.exclude_terms:
        matches, _ = _matching_constraint_terms(constraints.exclude_terms, paper)
        results["excluded_terms"] = not matches
    if (
        constraints.must_include_terms
        and "must_include_terms" in constraints.explicit_fields
    ):
        matches, _ = _matching_constraint_terms(
            constraints.must_include_terms,
            paper,
            lexical_policy=config.lexical_normalization_policy,
        )
        matched = {item.casefold() for item in matches}
        results["must_have"] = all(
            term.casefold() in matched for term in constraints.must_include_terms
        )
    if constraints.datasets and "datasets" in constraints.explicit_fields:
        matches, _ = _matching_constraint_terms(constraints.datasets, paper)
        results["dataset"] = bool(matches)
    if constraints.paper_types:
        matches, _ = _matched_paper_types(constraints.paper_types, paper)
        results["paper_type"] = bool(matches)
    if constraints.venues:
        venue_key = _normalize_venue(paper.venue or "")
        results["venue"] = bool(venue_key) and any(
            _normalize_venue(expected) in venue_key
            for expected in constraints.venues
        )
    if constraints.time_range is not None:
        results["time"] = paper.year is not None and not _outside_time_range(
            constraints,
            paper,
        )
    return results


def production_constraint_catalog() -> dict[str, Any]:
    """Return the candidate-level constraint contract used by current rules.

    This catalog is observational metadata for offline audits.  The predicates
    remain the same helpers used by :func:`judge_papers`; no filtering path is
    introduced here.
    """

    return {
        "predicate_version": PRODUCTION_CONSTRAINT_PREDICATE_VERSION,
        "field_order": list(PRODUCTION_CONSTRAINT_ORDER),
        "fields": {
            "exclude_terms": {
                "evidence_fields": ["title", "abstract"],
                "enforcement": "hard_guard",
                "combination": "no_expected_term_matches",
                "missing_value": "unknown_when_all_evidence_text_is_empty",
            },
            "must_include_terms": {
                "evidence_fields": ["title", "abstract"],
                "enforcement": "scoring_and_explicit_guard",
                "combination": "all_when_explicit_otherwise_scoring_any",
                "missing_value": "unknown_when_all_evidence_text_is_empty",
            },
            "methods": {
                "evidence_fields": ["title", "abstract"],
                "enforcement": "scoring_only",
                "combination": "any_expected_term_matches",
                "missing_value": "unknown_when_all_evidence_text_is_empty",
            },
            "datasets": {
                "evidence_fields": ["title", "abstract"],
                "enforcement": "scoring_and_explicit_soft_guard",
                "combination": "any_expected_term_matches",
                "missing_value": "unknown_when_all_evidence_text_is_empty",
            },
            "domains": {
                "evidence_fields": [],
                "enforcement": "planning_only_not_consumed_by_candidate_predicate",
                "combination": "not_applicable",
                "missing_value": "not_applicable",
            },
            "paper_types": {
                "evidence_fields": ["title", "abstract"],
                "enforcement": "scoring_and_soft_guard",
                "combination": "any_production_alias_matches",
                "missing_value": "unknown_when_all_evidence_text_is_empty",
            },
            "venues": {
                "evidence_fields": ["venue"],
                "enforcement": "scoring_only",
                "combination": "any_normalized_venue_matches",
                "missing_value": "unknown_when_venue_is_null_or_empty",
            },
            "time_range": {
                "evidence_fields": ["year"],
                "enforcement": "scoring_and_out_of_range_guard",
                "combination": "all_present_inclusive_bounds_match",
                "missing_value": "unknown_when_year_is_null",
            },
        },
    }


def trace_constraint_decisions(
    query_analysis: QueryAnalysis,
    paper: Paper,
    *,
    config: JudgementRuleConfig | None = None,
) -> list[dict[str, Any]]:
    """Trace all production constraint dimensions without changing judgement.

    Expected and matched terms are represented only by counts and stable
    digests.  Raw query or paper metadata is intentionally excluded so the
    trace can be persisted as an opaque offline diagnostic.
    """

    resolved = config or CURRENT_RULES_CONFIG
    constraints = query_analysis.constraints
    explicit = set(constraints.explicit_fields)
    text_state = _constraint_text_field_state(paper)
    output: list[dict[str, Any]] = []

    def term_decision(
        name: str,
        values: list[str],
        *,
        mode: Literal["any", "all", "none"],
        lexical_policy: LexicalNormalizationPolicy = "off",
    ) -> None:
        base = _constraint_decision_base(
            name,
            values,
            explicit=name in explicit,
            field_lineage=text_state,
        )
        if not values:
            output.append(
                {**base, "status": "not_applicable", "reason_code": "no_values"}
            )
            return
        matches, evidence = _matching_constraint_terms(
            values,
            paper,
            lexical_policy=lexical_policy,
        )
        matched_keys = {item.casefold() for item in matches}
        value_keys = {item.casefold() for item in values if item.strip()}
        if mode == "none":
            predicate = not matched_keys
        elif mode == "all":
            predicate = bool(value_keys) and value_keys.issubset(matched_keys)
        else:
            predicate = bool(matched_keys)
        matched_fields = sorted({item.source for item in evidence})
        update = {
            "matched_value_count": len(matched_keys),
            "matched_value_sha256": _constraint_value_digest(matches),
            "matched_evidence_fields": matched_fields,
            "production_predicate_result": predicate,
        }
        if text_state[0]["state"] == "empty" and text_state[1]["state"] == "empty":
            output.append(
                {
                    **base,
                    **update,
                    "status": "unknown",
                    "reason_code": "title_and_abstract_empty",
                }
            )
            return
        reason = {
            ("exclude_terms", True): "no_excluded_term_matched",
            ("exclude_terms", False): "excluded_term_matched",
            ("must_include_terms", True): "required_term_predicate_passed",
            ("must_include_terms", False): "required_term_predicate_failed",
            ("methods", True): "method_term_matched",
            ("methods", False): "method_term_not_matched",
            ("datasets", True): "dataset_term_matched",
            ("datasets", False): "dataset_term_not_matched",
        }[(name, predicate)]
        output.append(
            {
                **base,
                **update,
                "status": "passed" if predicate else "failed",
                "reason_code": reason,
            }
        )

    term_decision("exclude_terms", constraints.exclude_terms, mode="none")
    term_decision(
        "must_include_terms",
        constraints.must_include_terms,
        mode="all" if "must_include_terms" in explicit else "any",
        lexical_policy=resolved.lexical_normalization_policy,
    )
    term_decision(
        "methods",
        constraints.methods,
        mode="any",
        lexical_policy=resolved.lexical_normalization_policy,
    )
    term_decision("datasets", constraints.datasets, mode="any")

    domain_base = _constraint_decision_base(
        "domains",
        constraints.domains,
        explicit="domains" in explicit,
        field_lineage=[],
    )
    output.append(
        {
            **domain_base,
            "status": "not_applicable",
            "reason_code": (
                "not_consumed_by_candidate_predicate"
                if constraints.domains
                else "no_values"
            ),
        }
    )

    type_base = _constraint_decision_base(
        "paper_types",
        list(constraints.paper_types),
        explicit="paper_types" in explicit,
        field_lineage=text_state,
    )
    if not constraints.paper_types:
        output.append(
            {**type_base, "status": "not_applicable", "reason_code": "no_values"}
        )
    else:
        matches, evidence = _matched_paper_types(constraints.paper_types, paper)
        predicate = bool(matches)
        type_update = {
            "matched_value_count": len(matches),
            "matched_value_sha256": _constraint_value_digest(matches),
            "matched_evidence_fields": sorted({item.source for item in evidence}),
            "production_predicate_result": predicate,
        }
        if text_state[0]["state"] == "empty" and text_state[1]["state"] == "empty":
            output.append(
                {
                    **type_base,
                    **type_update,
                    "status": "unknown",
                    "reason_code": "title_and_abstract_empty",
                }
            )
        else:
            output.append(
                {
                    **type_base,
                    **type_update,
                    "status": "passed" if predicate else "failed",
                    "reason_code": (
                        "paper_type_matched" if predicate else "paper_type_not_matched"
                    ),
                }
            )

    venue_base = _constraint_decision_base(
        "venues",
        constraints.venues,
        explicit="venues" in explicit,
        field_lineage=[_field_state("venue", paper.venue)],
    )
    if not constraints.venues:
        output.append(
            {**venue_base, "status": "not_applicable", "reason_code": "no_values"}
        )
    else:
        venue_key = _normalize_venue(paper.venue or "")
        predicate = bool(venue_key) and any(
            _normalize_venue(value) in venue_key for value in constraints.venues
        )
        venue_update = {
            "matched_value_count": int(predicate),
            "matched_value_sha256": (
                _constraint_value_digest(
                    [
                        value
                        for value in constraints.venues
                        if _normalize_venue(value) in venue_key
                    ]
                )
                if predicate
                else _constraint_value_digest([])
            ),
            "matched_evidence_fields": ["venue"] if predicate else [],
            "production_predicate_result": predicate,
        }
        if paper.venue is None or not paper.venue.strip():
            output.append(
                {
                    **venue_base,
                    **venue_update,
                    "status": "unknown",
                    "reason_code": (
                        "venue_null" if paper.venue is None else "venue_empty"
                    ),
                }
            )
        else:
            output.append(
                {
                    **venue_base,
                    **venue_update,
                    "status": "passed" if predicate else "failed",
                    "reason_code": "venue_matched" if predicate else "venue_not_matched",
                }
            )

    time_range = constraints.time_range
    time_values = (
        []
        if time_range is None
        else [
            f"start:{time_range.start_year}",
            f"end:{time_range.end_year}",
        ]
    )
    time_base = _constraint_decision_base(
        "time_range",
        time_values,
        explicit="time_range" in explicit,
        field_lineage=[_field_state("year", paper.year)],
    )
    if time_range is None:
        output.append(
            {**time_base, "status": "not_applicable", "reason_code": "no_values"}
        )
    else:
        predicate = paper.year is not None and not _outside_time_range(
            constraints, paper
        )
        time_update = {
            "matched_value_count": int(predicate),
            "matched_value_sha256": _constraint_value_digest(
                time_values if predicate else []
            ),
            "matched_evidence_fields": ["year"] if paper.year is not None else [],
            "production_predicate_result": predicate,
        }
        if paper.year is None:
            output.append(
                {
                    **time_base,
                    **time_update,
                    "status": "unknown",
                    "reason_code": "year_null",
                }
            )
        else:
            output.append(
                {
                    **time_base,
                    **time_update,
                    "status": "passed" if predicate else "failed",
                    "reason_code": (
                        "within_time_range" if predicate else "outside_time_range"
                    ),
                }
            )

    if [item["constraint"] for item in output] != list(PRODUCTION_CONSTRAINT_ORDER):
        raise RuntimeError("production constraint trace order drift")
    return output


def _constraint_decision_base(
    name: str,
    values: list[str],
    *,
    explicit: bool,
    field_lineage: list[dict[str, str]],
) -> dict[str, Any]:
    catalog = production_constraint_catalog()["fields"][name]
    return {
        "constraint": name,
        "predicate_version": PRODUCTION_CONSTRAINT_PREDICATE_VERSION,
        "enforcement": catalog["enforcement"],
        "explicit": explicit,
        "expected_value_count": len(_dedupe_terms(values)),
        "expected_value_sha256": _constraint_value_digest(values),
        "field_lineage": field_lineage,
        "matched_value_count": 0,
        "matched_value_sha256": _constraint_value_digest([]),
        "matched_evidence_fields": [],
        "production_predicate_result": None,
    }


def _constraint_text_field_state(paper: Paper) -> list[dict[str, str]]:
    return [
        _field_state("title", paper.title),
        _field_state("abstract", paper.abstract),
    ]


def _field_state(field: str, value: Any) -> dict[str, str]:
    if value is None:
        state = "null"
    elif isinstance(value, str) and not value.strip():
        state = "empty"
    else:
        state = "present"
    return {"field": field, "state": state}


def _constraint_value_digest(values: list[str]) -> str:
    normalized = sorted(
        {str(value).strip().casefold() for value in values if str(value).strip()}
    )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hard_constraint_failures(
    results: dict[str, bool | None],
) -> list[str]:
    hard_fields = {"excluded_terms", "must_have", "time"}
    return [
        name
        for name, passed in results.items()
        if name in hard_fields and passed is False
    ]


def _empty_feature_vector(
    query_analysis: QueryAnalysis,
    paper: Paper,
    config: JudgementRuleConfig,
    *,
    category_reason: str,
) -> JudgementFeatureVector:
    results = _constraint_results(query_analysis, paper, config)
    return JudgementFeatureVector(
        config_version=config.config_version,
        config_hash=judgement_config_hash(config),
        metadata_completeness=_metadata_completeness(paper),
        constraint_results=results,
        hard_constraint_failures=_hard_constraint_failures(results),
        score_components={},
        evidence_count=0,
        final_score=0.0,
        highly_relevant_threshold=config.highly_relevant_threshold,
        partially_relevant_threshold=config.partially_relevant_threshold,
        weakly_relevant_threshold=config.weakly_relevant_threshold,
        category_reason=category_reason,
    )


def _finalize_feature_vector(
    feature: JudgementFeatureVector | None,
    *,
    final_score: float,
    category: str,
    category_reason: str,
    hard_constraint_adjustment: float,
    forced_failures: list[str],
) -> JudgementFeatureVector | None:
    if feature is None:
        return None
    del hard_constraint_adjustment  # 分量由最终分数反算，避免累计舍入误差。
    components = dict(feature.score_components)
    components.pop("hard_constraint_adjustment", None)
    components["hard_constraint_adjustment"] = final_score - sum(
        components.values()
    )
    return feature.model_copy(
        update={
            "score_components": {
                key: round(value, 6) for key, value in components.items()
            },
            "hard_constraint_failures": _dedupe_terms(
                [*feature.hard_constraint_failures, *forced_failures]
            ),
            "final_score": final_score,
            "category_reason": category_reason or f"score_threshold:{category}",
        }
    )


def _reasoning(
    reasons: list[str],
    evidence: list[EvidenceItem],
    warnings: list[str],
) -> str:
    if not evidence:
        base = "No title, abstract, venue, or metadata evidence matched the query."
    else:
        base = "; ".join(_dedupe_terms(reasons)) or "Metadata evidence matched the query."
    if warnings:
        return f"{base} Warnings: {', '.join(warnings)}."
    return base


def _tokenize(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", text):
        normalized = token.strip(".,;:()[]{}").casefold()
        if not normalized or normalized in STOPWORDS:
            continue
        if len(normalized) <= 2 and normalized not in {"ai", "ml", "cv"}:
            continue
        terms.append(_canonical_term(token))

    if "大模型" in text:
        terms.append("LLM")
    if "检索增强" in text:
        terms.append("RAG")
    if "检索" in text:
        terms.append("retrieval")
    if "重排序" in text:
        terms.append("reranking")
    if "数据集" in text:
        terms.append("dataset")
    if "评测" in text:
        terms.append("benchmark")
    return _dedupe_terms(terms)


def _canonical_term(term: str) -> str:
    upper_terms = {"ai", "cv", "llm", "ml", "nlp", "rag"}
    clean = term.strip()
    if clean.casefold() in upper_terms:
        return clean.upper()
    return clean


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9+.#-]+", term):
        return re.search(rf"(?<![a-z0-9+.#-]){re.escape(term)}(?![a-z0-9+.#-])", text) is not None
    return term in text


def _abstract_snippet(abstract: str, term: str) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+", abstract.strip())
    for sentence in sentences:
        if term in sentence.casefold():
            return _short_text(sentence)
    return _short_text(abstract)


def _short_text(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip()


def _dedupe_terms(values: list[str]) -> list[str]:
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


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    deduped: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.source, item.text.casefold())
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped[:12]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _chat_json(
    llm_client: LLMJsonClient | None,
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> dict[str, Any]:
    if llm_client is not None:
        return llm_client.chat_json(messages, temperature=0, timeout=timeout)
    return provider_chat_json(messages, temperature=0, timeout=timeout)


def _build_llm_judgement_messages(
    query_analysis: QueryAnalysis,
    papers: list[Paper],
    *,
    start_index: int,
    metadata_observer: UntrustedMetadataObserver | None = None,
) -> list[dict[str, str]]:
    query_identity = opaque_query_identity(query_analysis.original_query)
    payload = {
        "query_analysis": {
            "original_query": query_analysis.original_query,
            "language": query_analysis.language,
            "intent": query_analysis.intent,
            "domain": query_analysis.domain,
            "constraints": query_analysis.constraints.model_dump(mode="json"),
        },
        "papers": [
            {
                "paper_index": start_index + index,
                **build_llm_paper_payload(
                    paper,
                    query_identity=query_identity,
                    observer=metadata_observer,
                ),
            }
            for index, paper in enumerate(papers)
        ],
    }
    return render_messages("relevance_judgement", payload)


def _judgement_results_from_llm_json(
    raw_response: dict[str, Any],
    *,
    query_analysis: QueryAnalysis,
    papers: list[Paper],
    rule_results: list[JudgementResult],
    start_index: int,
) -> list[JudgementResult]:
    try:
        strict_response = _StrictLLMJudgementResponse.model_validate(raw_response)
    except ValidationError:
        raise ValueError("llm_judgement_schema_rejected") from None
    validated = strict_response.model_dump(mode="python")
    raw_judgements = validated["judgements"]

    global_warnings = _coerce_string_list(validated.get("warnings"))
    by_global_index: dict[int, dict[str, Any]] = {}
    invalid_warnings_by_index: dict[int, list[str]] = {}
    for raw_item in raw_judgements:
        if not isinstance(raw_item, dict):
            _append_invalid_warning(
                invalid_warnings_by_index,
                start_index,
                "llm_judgement_invalid_item",
            )
            continue
        parsed_index = _parse_llm_paper_index(
            raw_item.get("paper_index"),
            start_index=start_index,
            batch_size=len(papers),
        )
        if parsed_index is None:
            _append_invalid_warning(
                invalid_warnings_by_index,
                start_index,
                "llm_judgement_missing_paper_index",
            )
            continue
        if parsed_index in by_global_index:
            _append_invalid_warning(
                invalid_warnings_by_index,
                parsed_index,
                f"llm_judgement_duplicate_paper_index:{parsed_index}",
            )
            continue
        by_global_index[parsed_index] = raw_item

    results: list[JudgementResult] = []
    for local_index, (paper, rule_result) in enumerate(zip(papers, rule_results)):
        global_index = start_index + local_index
        item_warnings = [
            "llm_judgement_used",
            *global_warnings,
            *invalid_warnings_by_index.get(global_index, []),
        ]
        raw_item = by_global_index.get(global_index)
        if raw_item is None:
            results.append(
                _with_warning(
                    [rule_result],
                    f"llm_judgement_missing_paper_index:{global_index}",
                    extra_warnings=item_warnings,
                )[0]
            )
            continue

        parsed = _parse_one_llm_judgement(
            raw_item,
            paper=paper,
            rule_result=rule_result,
            global_index=global_index,
            base_warnings=item_warnings,
        )
        results.append(parsed)
    return results


def _parse_one_llm_judgement(
    raw_item: dict[str, Any],
    *,
    paper: Paper,
    rule_result: JudgementResult,
    global_index: int,
    base_warnings: list[str],
) -> JudgementResult:
    warnings = list(base_warnings)
    category = str(raw_item.get("category", "")).strip()
    if category not in JUDGEMENT_CATEGORIES:
        return _with_warning(
            [rule_result],
            f"llm_judgement_invalid_category:{global_index}",
            extra_warnings=warnings,
        )[0]

    score = _parse_score(raw_item.get("score"), warnings, global_index)
    if score is None:
        return _with_warning(
            [rule_result],
            f"llm_judgement_invalid_score:{global_index}",
            extra_warnings=warnings,
        )[0]

    evidence = _normalize_llm_evidence(
        raw_item.get("evidence"),
        paper=paper,
        global_index=global_index,
        warnings=warnings,
    )
    reasoning = _short_text(str(raw_item.get("reasoning") or "").strip(), limit=320)
    if not reasoning:
        reasoning = "LLM judged relevance using candidate metadata only."
    matched_terms = _dedupe_terms(_coerce_string_list(raw_item.get("matched_terms")))
    warnings.extend(_coerce_string_list(raw_item.get("warnings")))
    warnings.extend(rule_result.warnings)

    return JudgementResult(
        paper=paper,
        score=round(score, 4),
        category=category,  # type: ignore[arg-type]
        reasoning=reasoning,
        evidence=_dedupe_evidence(evidence),
        matched_terms=matched_terms,
        warnings=_dedupe_terms(warnings),
    )


def _parse_score(
    raw_score: Any,
    warnings: list[str],
    global_index: int,
) -> float | None:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    clamped = _clamp(score)
    if clamped != score:
        warnings.append(f"llm_judgement_score_clamped:{global_index}")
    return clamped


def _normalize_llm_evidence(
    raw_evidence: Any,
    *,
    paper: Paper,
    global_index: int,
    warnings: list[str],
) -> list[EvidenceItem]:
    if raw_evidence is None:
        warnings.append(f"llm_judgement_missing_evidence:{global_index}")
        return []
    if isinstance(raw_evidence, dict):
        raw_evidence = [raw_evidence]
    if not isinstance(raw_evidence, list):
        warnings.append(f"llm_judgement_invalid_evidence:{global_index}")
        return []

    evidence: list[EvidenceItem] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            warnings.append(f"llm_judgement_invalid_evidence_item:{global_index}")
            continue
        source = str(item.get("source") or "").strip().lower()
        if source not in EVIDENCE_SOURCES:
            warnings.append(
                f"llm_judgement_bad_evidence_source:{global_index}:{source or 'unknown'}"
            )
            continue
        text = _short_text(str(item.get("text") or ""), limit=200)
        grounded_text = _ground_evidence_text(
            source,
            text,
            paper,
            global_index=global_index,
            warnings=warnings,
        )
        if not grounded_text:
            continue
        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
            warnings.append(f"llm_judgement_invalid_evidence_confidence:{global_index}")
        evidence.append(
            EvidenceItem(
                source=source,  # type: ignore[arg-type]
                text=grounded_text,
                confidence=_clamp(confidence),
            )
        )
    return evidence


def _ground_evidence_text(
    source: str,
    text: str,
    paper: Paper,
    *,
    global_index: int,
    warnings: list[str],
) -> str:
    if source == "title":
        return _ground_against_field(
            source,
            text,
            paper.title,
            global_index=global_index,
            warnings=warnings,
        )
    if source == "abstract":
        return _ground_against_field(
            source,
            text,
            paper.abstract,
            global_index=global_index,
            warnings=warnings,
        )
    if source == "venue":
        return _ground_against_field(
            source,
            text,
            paper.venue or "",
            global_index=global_index,
            warnings=warnings,
        )
    metadata_text = _short_text(
        "; ".join(
            item
            for item in (
                f"year={paper.year}" if paper.year is not None else "",
                f"sources={','.join(paper.sources)}" if paper.sources else "",
                (
                    f"citation_count={paper.citation_count}"
                    if paper.citation_count is not None
                    else ""
                ),
            )
            if item
        )
        or text,
        limit=160,
    )
    if text and text.casefold() not in metadata_text.casefold():
        warnings.append(f"llm_judgement_evidence_regrounded:{global_index}:metadata")
    return metadata_text


def _ground_against_field(
    source: str,
    text: str,
    field_value: str,
    *,
    global_index: int,
    warnings: list[str],
) -> str:
    field_text = _short_text(field_value, limit=200)
    if not field_text:
        warnings.append(f"llm_judgement_evidence_missing_metadata:{global_index}:{source}")
        return ""
    if not text:
        return field_text
    if text.casefold() in field_text.casefold():
        return text
    warnings.append(f"llm_judgement_evidence_regrounded:{global_index}:{source}")
    return field_text


def _parse_llm_paper_index(
    raw_index: Any,
    *,
    start_index: int,
    batch_size: int,
) -> int | None:
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if start_index <= index < start_index + batch_size:
        return index
    if 0 <= index < batch_size:
        return start_index + index
    return None


def _with_warning(
    results: list[JudgementResult],
    warning: str,
    *,
    extra_warnings: list[str] | None = None,
) -> list[JudgementResult]:
    warnings_to_add = [*(extra_warnings or []), warning]
    return [
        result.model_copy(
            update={
                "warnings": _dedupe_terms([*result.warnings, *warnings_to_add]),
                "reasoning": _append_warning_to_reasoning(result.reasoning, warning),
            }
        )
        for result in results
    ]


def _append_warning_to_reasoning(reasoning: str, warning: str) -> str:
    if warning in reasoning:
        return reasoning
    return f"{reasoning} Warning: {warning}."


def _append_invalid_warning(
    warnings_by_index: dict[int, list[str]],
    global_index: int,
    warning: str,
) -> None:
    warnings_by_index.setdefault(global_index, []).append(warning)


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return []


def _diagnostic_message(value: Any) -> str:
    return safe_diagnostic_message(value) or "unknown"
