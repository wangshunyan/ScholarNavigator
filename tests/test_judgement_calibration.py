from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.agents import judgement as judgement_module
from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import (
    CALIBRATED_RULES_V1_CONFIG,
    CURRENT_RULES_CONFIG,
    judgement_config_hash,
    resolve_judgement_config,
)
from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    JudgementRuleConfig,
    QueryAnalysis,
    QueryConstraint,
    TimeRange,
)
from scholar_agent.evaluation.judgement_calibration import (
    CalibrationEvaluation,
    FrozenJudgementCase,
    evaluate_frozen_cases,
    judgement_parameter_grid,
    parameter_grid_hash,
    select_best_evaluation,
    validation_acceptance,
)


def _analysis(*, constraints: QueryConstraint | None = None) -> QueryAnalysis:
    return QueryAnalysis(
        original_query="graph neural network molecular property prediction QM9",
        language="en",
        intent="paper_finding",
        domain="machine_learning",
        constraints=constraints
        or QueryConstraint(
            methods=["graph neural network"],
            datasets=["QM9"],
            domains=["machine_learning"],
        ),
    )


def _paper(
    title: str,
    *,
    doi: str,
    abstract: str = "",
    year: int | None = 2024,
    venue: str | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Researcher"],
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(doi=doi),
        sources=["arxiv"],
        citation_count=5,
    )


def _case() -> FrozenJudgementCase:
    gold = _paper(
        "Graph Neural Network Molecular Property Prediction",
        doi="10.1000/gold",
        abstract="A graph neural network predicts molecular properties on QM9 dataset.",
    )
    broad = _paper(
        "Neural Network Applications",
        doi="10.1000/broad",
        abstract="A broad overview of neural systems.",
    )
    return FrozenJudgementCase(
        case_id="case-1",
        query="graph neural network molecular property prediction QM9",
        query_analysis=_analysis(),
        papers=[gold, broad],
        gold_papers=[EvalGoldPaper(doi="10.1000/gold")],
        replay_cost={
            "retrieval_snapshot_hits": 2,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0,
        },
    )


def test_explicit_current_config_preserves_default_judgement() -> None:
    paper = _case().papers[0]

    implicit = judge_papers(_analysis(), [paper], use_llm=False)[0]
    explicit = judge_papers(
        _analysis(),
        [paper],
        policy="current_rules",
        config=CURRENT_RULES_CONFIG,
        use_llm=False,
    )[0]

    assert explicit == implicit
    assert explicit.feature_vector is not None
    assert explicit.feature_vector.config_hash == judgement_config_hash(
        CURRENT_RULES_CONFIG
    )


def test_feature_components_add_to_final_score() -> None:
    result = judge_papers(_analysis(), [_case().papers[0]], use_llm=False)[0]
    feature = result.feature_vector

    assert feature is not None
    assert sum(feature.score_components.values()) == pytest.approx(
        feature.final_score,
        abs=1e-5,
    )
    assert feature.title_matched_terms
    assert feature.abstract_matched_terms
    assert feature.matched_method_terms == ["graph neural network"]
    assert feature.matched_dataset_terms == ["QM9"]


def test_soft_calibration_cannot_bypass_hard_constraint_guards() -> None:
    permissive = CURRENT_RULES_CONFIG.model_copy(
        update={
            "config_version": "permissive-test",
            "title_topic_weight": 1.0,
            "abstract_topic_weight": 1.0,
            "highly_relevant_threshold": 0.50,
            "partially_relevant_threshold": 0.30,
            "weakly_relevant_threshold": 0.10,
        }
    )
    paper = _paper(
        "Graph neural network molecular prediction QM9 forbidden",
        doi="10.1000/guard",
        abstract="graph neural network molecular prediction",
        year=2010,
    )
    excluded_analysis = _analysis(
        constraints=QueryConstraint(exclude_terms=["forbidden"])
    )
    must_analysis = _analysis(
        constraints=QueryConstraint(
            must_include_terms=["causal"],
            explicit_fields=["must_include_terms"],
        )
    )
    time_analysis = _analysis(
        constraints=QueryConstraint(
            time_range=TimeRange(start_year=2020, end_year=2026),
            explicit_fields=["time_range"],
        )
    )

    excluded = judge_papers(
        excluded_analysis, [paper], config=permissive, use_llm=False
    )[0]
    missing_must = judge_papers(
        must_analysis, [paper], config=permissive, use_llm=False
    )[0]
    outside_time = judge_papers(
        time_analysis, [paper], config=permissive, use_llm=False
    )[0]

    assert excluded.category == "irrelevant"
    assert excluded.score == 0.0
    assert "excluded_terms" in excluded.feature_vector.hard_constraint_failures
    assert missing_must.category != "highly_relevant"
    assert any(item.startswith("missing_must_have_terms:") for item in missing_must.warnings)
    assert outside_time.category != "highly_relevant"
    assert any(item.startswith("outside_time_range:") for item in outside_time.warnings)


def test_threshold_order_is_validated() -> None:
    payload = CURRENT_RULES_CONFIG.model_dump()
    payload.update(
        {
            "weakly_relevant_threshold": 0.6,
            "partially_relevant_threshold": 0.4,
        }
    )

    with pytest.raises(ValidationError, match="weak <= partial <= high"):
        JudgementRuleConfig.model_validate(payload)


def test_calibration_grid_is_fixed_bounded_and_stable() -> None:
    first = judgement_parameter_grid()
    second = judgement_parameter_grid()

    assert len(first) == 128
    assert 50 <= len(first) <= 300
    assert parameter_grid_hash(first) == parameter_grid_hash(second)
    assert [judgement_config_hash(item) for item in first] == [
        judgement_config_hash(item) for item in second
    ]


def test_frozen_evaluation_is_deterministic_and_retrieval_invariant() -> None:
    baseline_first = evaluate_frozen_cases(
        [_case()], CURRENT_RULES_CONFIG, policy="current_rules"
    )
    baseline_second = evaluate_frozen_cases(
        [_case()], CURRENT_RULES_CONFIG, policy="current_rules"
    )
    calibrated = evaluate_frozen_cases(
        [_case()], CALIBRATED_RULES_V1_CONFIG, policy="calibrated_rules_v1"
    )

    assert baseline_first == baseline_second
    assert baseline_first.metrics["candidate_recall"] == calibrated.metrics[
        "candidate_recall"
    ]
    assert baseline_first.metrics["replay_execution_cost"] == calibrated.metrics[
        "replay_execution_cost"
    ]
    assert baseline_first.metrics["replay_execution_cost"][
        "replay_execution_request_count"
    ] == 0


def test_selection_tie_break_prefers_closest_then_stable_hash() -> None:
    common_metrics = {
        "f1_at_20": 0.1,
        "recall_at_20": 0.2,
        "precision_at_20": 0.01,
        "mrr": 0.3,
        "gold_judgement_false_negative_rate": 0.0,
    }
    current = CalibrationEvaluation(
        policy="current_rules",
        config=CURRENT_RULES_CONFIG,
        config_hash=judgement_config_hash(CURRENT_RULES_CONFIG),
        metrics=common_metrics,
    )
    distant_config = CURRENT_RULES_CONFIG.model_copy(
        update={"config_version": "distant", "title_topic_weight": 0.01}
    )
    distant = CalibrationEvaluation(
        policy="calibrated_rules_v1",
        config=distant_config,
        config_hash=judgement_config_hash(distant_config),
        metrics=common_metrics,
    )

    assert select_best_evaluation([distant, current]) == current


def test_validation_acceptance_requires_non_regression_and_controls_volume() -> None:
    baseline = evaluate_frozen_cases(
        [_case()], CURRENT_RULES_CONFIG, policy="current_rules"
    )
    same = validation_acceptance(baseline, baseline)
    regressed = baseline.model_copy(
        update={"metrics": {**baseline.metrics, "f1_at_20": -0.01}}
    )

    assert same["accepted"] is True
    assert validation_acceptance(baseline, regressed)["accepted"] is False


def test_gold_labels_are_only_used_after_rule_judgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    original = judgement_module.judge_papers

    def recording_judge(query_analysis, papers, **kwargs):  # noqa: ANN001, ANN202
        seen["query_analysis"] = query_analysis
        seen["papers"] = papers
        seen["kwargs"] = kwargs
        return original(query_analysis, papers, **kwargs)

    monkeypatch.setattr(
        "scholar_agent.evaluation.judgement_calibration.judge_papers",
        recording_judge,
    )

    evaluation = evaluate_frozen_cases(
        [_case()],
        CURRENT_RULES_CONFIG,
        policy="current_rules",
        include_diagnostics=True,
    )

    assert all(isinstance(item, Paper) for item in seen["papers"])
    assert "gold" not in seen["kwargs"]
    assert any(row["post_run_gold_match"] for row in evaluation.candidate_diagnostics)
    assert all("abstract" not in row for row in evaluation.candidate_diagnostics)


def test_policy_resolves_to_frozen_development_configuration() -> None:
    resolved = resolve_judgement_config("calibrated_rules_v1")

    assert resolved == CALIBRATED_RULES_V1_CONFIG
    assert resolved.config_version == "calibrated-rules-v1"
    assert resolved.highly_relevant_threshold == 0.68
    assert resolved.title_topic_weight == 0.10
