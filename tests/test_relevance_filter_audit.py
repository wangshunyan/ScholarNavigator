from __future__ import annotations

from pathlib import Path

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import JudgementFeatureVector, JudgementResult
from scholar_agent.evaluation.relevance_filter_audit import (
    classify_filtered_gold_root,
    lexical_gap_terms,
    remove_single_score_rule,
    write_relevance_filter_audit,
)


def _judgement(
    *,
    score: float,
    category: str,
    component: float = 0.0,
) -> JudgementResult:
    return JudgementResult(
        paper=Paper(title="Unicode β study", abstract="A complete abstract."),
        score=score,
        category=category,
        reasoning="frozen",
        feature_vector=JudgementFeatureVector(
            config_version="test",
            config_hash="0" * 64,
            metadata_completeness=1.0,
            score_components={"constraint_coverage_adjustment": component},
            final_score=score,
            highly_relevant_threshold=0.72,
            partially_relevant_threshold=0.45,
            weakly_relevant_threshold=0.25,
            category_reason=f"score_threshold:{category}",
        ),
    )


def test_filtered_root_covers_query_and_field_missing() -> None:
    common = {
        "lexical_gap_terms": [],
        "negative_components": {},
        "hard_constraint_failures": [],
        "score": 0.2,
        "partial_threshold": 0.45,
        "rank": 1,
    }
    assert classify_filtered_gold_root(
        **common,
        query_term_count=0,
        structured_term_count=0,
        title_present=True,
        abstract_present=True,
    ) == "query_parsing_missing"
    assert classify_filtered_gold_root(
        **common,
        query_term_count=2,
        structured_term_count=1,
        title_present=True,
        abstract_present=False,
    ) == "field_text_missing"


def test_lexical_gap_detects_morphology_and_unicode_punctuation() -> None:
    gaps = lexical_gap_terms(
        ["studies", "T-cells", "β"],
        [],
        title="A study of T cells — β response",
        abstract="",
    )
    assert gaps == ["studies", "T-cells", "β"]
    assert classify_filtered_gold_root(
        query_term_count=3,
        structured_term_count=0,
        title_present=True,
        abstract_present=True,
        lexical_gap_terms=gaps,
        negative_components={},
        hard_constraint_failures=[],
        score=0.3,
        partial_threshold=0.45,
        rank=2,
    ) == "morphology_or_abbreviation_mismatch"


def test_lexical_gap_detects_expanded_abbreviation() -> None:
    assert lexical_gap_terms(
        ["G-CSF"],
        [],
        title="Granulocyte colony stimulating factor response",
        abstract="",
    ) == ["G-CSF"]


def test_filtered_root_distinguishes_constraint_threshold_and_category_priority(
) -> None:
    common = {
        "query_term_count": 3,
        "structured_term_count": 2,
        "title_present": True,
        "abstract_present": True,
        "lexical_gap_terms": [],
        "hard_constraint_failures": [],
        "score": 0.44,
        "partial_threshold": 0.45,
    }
    assert classify_filtered_gold_root(
        **common,
        negative_components={"constraint_coverage_adjustment": -0.06},
        rank=2,
    ) == "constraint_penalty"
    assert classify_filtered_gold_root(
        **common,
        negative_components={},
        rank=20,
    ) == "fixed_threshold"
    assert classify_filtered_gold_root(
        **common,
        negative_components={},
        rank=21,
    ) == "category_priority"


def test_single_rule_counterfactual_honors_frozen_threshold_boundary() -> None:
    original = _judgement(
        score=0.44,
        category="weakly_relevant",
        component=-0.01,
    )
    changed = remove_single_score_rule(
        [original], "constraint_coverage_adjustment"
    )[0]
    assert original.score == 0.44
    assert original.category == "weakly_relevant"
    assert changed.score == 0.45
    assert changed.category == "partially_relevant"
    assert changed.feature_vector is not None
    assert changed.feature_vector.score_components[
        "constraint_coverage_adjustment"
    ] == 0.0


def test_audit_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"dataset": "x", "case_id": "1"}]
    candidates = [{"dataset": "x", "candidate_id": "doi:1"}]
    filtered = [{"dataset": "x", "primary_root_cause": "fixed_threshold"}]
    aggregate = {"schema_version": "1", "datasets": {"x": {"count": 1}}}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_relevance_filter_audit(first, cases, candidates, filtered, aggregate)
    write_relevance_filter_audit(second, cases, candidates, filtered, aggregate)
    for name in (
        "case_audit.jsonl",
        "candidate_audit.jsonl",
        "filtered_gold_chains.jsonl",
        "aggregate.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
