from __future__ import annotations

import pytest

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import (
    CURRENT_RULES_CONFIG,
    LEXICAL_NORMALIZATION_V1_CONFIG,
)
from scholar_agent.agents.lexical_normalization import (
    find_lexical_normalization_match,
    normalize_lexical_tokens,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint


@pytest.mark.parametrize(
    ("term", "text", "normalized"),
    [
        ("Ｔ-cell", "T cell response", "t cell"),
        ("T-cells", "T cells response", "t cell"),
        ("U.S.", "US cohort", "us"),
        ("Crohn's", "Crohn disease", "crohn"),
        ("methods", "method comparison", "method"),
    ],
)
def test_v1_normalizes_only_fixed_lexical_forms(
    term: str,
    text: str,
    normalized: str,
) -> None:
    match = find_lexical_normalization_match(term, title=text, abstract="")
    assert match is not None
    assert match.normalized_form == normalized
    assert match.field == "title"


def test_short_words_and_ambiguous_plural_forms_do_not_merge() -> None:
    assert find_lexical_normalization_match(
        "AI", title="A I method", abstract=""
    ) is None
    assert find_lexical_normalization_match(
        "gas", title="gases in blood", abstract=""
    ) is None
    assert find_lexical_normalization_match(
        "axis", title="axes of rotation", abstract=""
    ) is None
    assert normalize_lexical_tokens("analysis") != normalize_lexical_tokens(
        "analyses"
    )


def test_default_off_preserves_existing_judgement_and_v1_records_impact() -> None:
    analysis = QueryAnalysis(
        original_query="T-cells methods",
        constraints=QueryConstraint(must_include_terms=["T-cells"]),
    )
    paper = Paper(title="T cells method", abstract="")
    baseline = judge_papers(
        analysis,
        [paper],
        use_llm=False,
        config=CURRENT_RULES_CONFIG,
    )[0]
    experiment = judge_papers(
        analysis,
        [paper],
        use_llm=False,
        config=LEXICAL_NORMALIZATION_V1_CONFIG,
    )[0]
    assert baseline.feature_vector is not None
    assert baseline.feature_vector.lexical_normalization_matches == []
    assert experiment.feature_vector is not None
    matches = experiment.feature_vector.lexical_normalization_matches
    assert [(item.facet, item.original_term, item.normalized_form, item.field) for item in matches] == [
        ("topic", "T-cells", "t cell", "title"),
        ("topic", "methods", "method", "title"),
        ("must_have", "T-cells", "t cell", "title"),
    ]
    assert [item.score_impact for item in matches] == [0.12, 0.12, 0.09]
    assert experiment.score > baseline.score


def test_excluded_and_dataset_terms_keep_exact_matching_semantics() -> None:
    paper = Paper(title="T cells response", abstract="immune evidence")
    excluded = judge_papers(
        QueryAnalysis(
            original_query="immune evidence",
            constraints=QueryConstraint(exclude_terms=["T-cells"]),
        ),
        [paper],
        use_llm=False,
        config=LEXICAL_NORMALIZATION_V1_CONFIG,
    )[0]
    dataset = judge_papers(
        QueryAnalysis(
            original_query="unrelated evidence",
            constraints=QueryConstraint(datasets=["T-cells"]),
        ),
        [paper],
        use_llm=False,
        config=LEXICAL_NORMALIZATION_V1_CONFIG,
    )[0]
    assert "excluded_terms" not in (
        excluded.feature_vector.hard_constraint_failures
        if excluded.feature_vector is not None
        else []
    )
    assert "excluded_terms_matched:T-cells" not in excluded.warnings
    assert excluded.feature_vector is not None
    assert excluded.feature_vector.category_reason != "excluded_term_hard_constraint"
    assert dataset.feature_vector is not None
    assert dataset.feature_vector.score_components["dataset_match"] == 0.0
    assert dataset.feature_vector.lexical_normalization_matches == []


def test_normalization_is_deterministic_and_does_not_use_fuzzy_similarity() -> None:
    first = find_lexical_normalization_match(
        "neuralation", title="neurulation", abstract=""
    )
    second = find_lexical_normalization_match(
        "neuralation", title="neurulation", abstract=""
    )
    assert first is None
    assert second is None
