"""冻结候选上的确定性 Judgement 校准与候选级诊断。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import (
    CURRENT_RULES_CONFIG,
    judgement_config_hash,
)
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    JudgementPolicy,
    JudgementRuleConfig,
    QueryAnalysis,
)
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    evaluate_ranking,
    matched_paper_ids,
    recall_at_k,
)
from scholar_agent.evaluation.selection import select_ranked_results


CALIBRATION_GRID_VERSION = "judgement-grid-v1"
SELECTION_OBJECTIVE = [
    "max_f1_at_20",
    "max_recall_at_20",
    "min_gold_judgement_false_negative_rate",
    "max_precision_at_20",
    "max_mrr",
    "min_default_parameter_distance",
    "min_config_hash",
]


class FrozenJudgementCase(BaseModel):
    case_id: str
    query: str
    query_analysis: QueryAnalysis
    papers: list[Paper]
    gold_papers: list[EvalGoldPaper]
    replay_cost: dict[str, Any] = Field(default_factory=dict)


class CalibrationEvaluation(BaseModel):
    policy: JudgementPolicy
    config: JudgementRuleConfig
    config_hash: str
    metrics: dict[str, Any]
    candidate_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


def judgement_parameter_grid() -> list[JudgementRuleConfig]:
    """返回运行前固定、包含当前默认配置的 128 个通用组合。"""

    configs: list[JudgementRuleConfig] = []
    for (
        high,
        partial,
        weak,
        title_topic,
        abstract_topic,
        missing_abstract,
        minimum_evidence,
    ) in product(
        (0.68, 0.72),
        (0.40, 0.45),
        (0.20, 0.25),
        (0.10, 0.12),
        (0.06, 0.075),
        (0.0, 0.03),
        (0, 1),
    ):
        configs.append(
            CURRENT_RULES_CONFIG.model_copy(
                update={
                    "config_version": CALIBRATION_GRID_VERSION,
                    "highly_relevant_threshold": high,
                    "partially_relevant_threshold": partial,
                    "weakly_relevant_threshold": weak,
                    "title_topic_weight": title_topic,
                    "abstract_topic_weight": abstract_topic,
                    "missing_abstract_penalty": missing_abstract,
                    "minimum_evidence_count": minimum_evidence,
                }
            )
        )
    return sorted(configs, key=judgement_config_hash)


def parameter_grid_hash(configs: list[JudgementRuleConfig]) -> str:
    payload = [
        config.model_dump(mode="json")
        for config in sorted(configs, key=judgement_config_hash)
    ]
    return _stable_hash(payload)


def evaluate_frozen_cases(
    cases: list[FrozenJudgementCase],
    config: JudgementRuleConfig,
    *,
    policy: JudgementPolicy,
    include_diagnostics: bool = False,
) -> CalibrationEvaluation:
    per_case: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    aggregate_categories: Counter[str] = Counter()
    all_scores: list[float] = []
    gold_scores: list[float] = []
    non_gold_scores: list[float] = []
    total_returned = 0
    total_non_gold_returned = 0
    total_gold_retrieved = 0
    total_gold_retained = 0
    total_gold_false_negative = 0
    total_gold_categories: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    hard_constraint_failures: Counter[str] = Counter()

    for case in cases:
        judgements = judge_papers(
            case.query_analysis,
            case.papers,
            policy=policy,
            config=config,
            use_llm=False,
        )
        all_ranked = rerank_papers(
            case.query_analysis,
            judgements,
            top_k=max(1, len(judgements)),
        )
        top_ranked = all_ranked[:20]
        returned = select_ranked_results(
            {"ranked_papers": top_ranked},
            policy="highly_and_partial",
        )
        metric_set = evaluate_ranking(returned, case.gold_papers, (5, 10, 20))
        candidate_recall = recall_at_k(
            case.papers,
            case.gold_papers,
            max(1, len(case.papers)),
        )
        gold_state = _gold_state(
            case,
            judgements=judgements,
            all_ranked=all_ranked,
            returned=returned,
        )
        returned_gold_ids = set(matched_paper_ids(returned, case.gold_papers))
        total_returned += len(returned)
        total_non_gold_returned += sum(
            not matched_paper_ids([item], case.gold_papers, k=1)
            for item in returned
        )
        total_gold_retrieved += len(gold_state["retrieved_ids"])
        total_gold_retained += len(gold_state["retained_ids"])
        false_negative_ids = gold_state["retrieved_ids"] - gold_state["retained_ids"]
        total_gold_false_negative += len(false_negative_ids)
        total_gold_categories.update(gold_state["category_by_gold"].values())
        boundary_counts.update(gold_state["boundary_by_gold"].values())
        aggregate_categories.update(item.category for item in judgements)
        for item in judgements:
            if item.feature_vector is not None:
                hard_constraint_failures.update(
                    item.feature_vector.hard_constraint_failures
                )
        all_scores.extend(item.score for item in judgements)
        for item in judgements:
            if matched_paper_ids([item], case.gold_papers, k=1):
                gold_scores.append(item.score)
            else:
                non_gold_scores.append(item.score)
        per_case.append(
            {
                "case_id": case.case_id,
                "candidate_recall": candidate_recall,
                "f1_at_5": metric_set.f1_at_k.get(5, 0.0),
                "f1_at_10": metric_set.f1_at_k.get(10, 0.0),
                "f1_at_20": metric_set.f1_at_k.get(20, 0.0),
                "precision_at_5": metric_set.precision_at_k.get(5, 0.0),
                "precision_at_10": metric_set.precision_at_k.get(10, 0.0),
                "precision_at_20": metric_set.precision_at_k.get(20, 0.0),
                "recall_at_5": metric_set.recall_at_k.get(5, 0.0),
                "recall_at_10": metric_set.recall_at_k.get(10, 0.0),
                "recall_at_20": metric_set.recall_at_k.get(20, 0.0),
                "mrr": metric_set.mrr,
                "ndcg_at_20": metric_set.ndcg_at_k.get(20, 0.0),
                "candidate_count": len(case.papers),
                "returned_count": len(returned),
                "gold_retrieved_count": len(gold_state["retrieved_ids"]),
                "gold_retained_count": len(gold_state["retained_ids"]),
                "gold_false_negative_count": len(false_negative_ids),
                "returned_gold_count": len(returned_gold_ids),
            }
        )
        if include_diagnostics:
            diagnostics.extend(
                _candidate_diagnostics(
                    case,
                    judgements=judgements,
                    all_ranked=all_ranked,
                    returned=returned,
                    false_negative_ids=false_negative_ids,
                )
            )

    case_count = len(cases)
    metrics = {
        "case_count": case_count,
        "candidate_recall": _average(row["candidate_recall"] for row in per_case),
        **{
            name: _average(row[name] for row in per_case)
            for name in (
                "f1_at_5",
                "f1_at_10",
                "f1_at_20",
                "precision_at_5",
                "precision_at_10",
                "precision_at_20",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "mrr",
                "ndcg_at_20",
            )
        },
        "gold_retrieved_count": total_gold_retrieved,
        "gold_judged_highly_count": total_gold_categories["highly_relevant"],
        "gold_judged_partially_count": total_gold_categories[
            "partially_relevant"
        ],
        "gold_judged_weak_count": total_gold_categories["weakly_relevant"],
        "gold_judged_irrelevant_count": total_gold_categories["irrelevant"],
        "gold_insufficient_evidence_count": total_gold_categories[
            "insufficient_evidence"
        ],
        "gold_retained_count": total_gold_retained,
        "gold_judgement_false_negative_count": total_gold_false_negative,
        "gold_judgement_false_negative_rate": (
            total_gold_false_negative / total_gold_retrieved
            if total_gold_retrieved
            else None
        ),
        "average_returned_paper_count": (
            total_returned / case_count if case_count else 0.0
        ),
        "benchmark_non_gold_returned_count": total_non_gold_returned,
        "category_distribution": dict(sorted(aggregate_categories.items())),
        "average_judgement_score": _average(all_scores),
        "average_gold_judgement_score": _average_optional(gold_scores),
        "average_benchmark_non_gold_judgement_score": _average_optional(
            non_gold_scores
        ),
        "judgement_reranking_boundaries": dict(sorted(boundary_counts.items())),
        "hard_constraint_failure_distribution": dict(
            sorted(hard_constraint_failures.items())
        ),
        "reranking_followup_needed": bool(
            boundary_counts["judgement_retained_ranked_outside_top20"]
        ),
        "replay_execution_cost": _aggregate_replay_cost(cases),
        "per_case": per_case,
    }
    return CalibrationEvaluation(
        policy=policy,
        config=config,
        config_hash=judgement_config_hash(config),
        metrics=metrics,
        candidate_diagnostics=diagnostics,
    )


def selection_sort_key(evaluation: CalibrationEvaluation) -> tuple[Any, ...]:
    metrics = evaluation.metrics
    false_negative_rate = metrics.get("gold_judgement_false_negative_rate")
    return (
        -float(metrics.get("f1_at_20") or 0.0),
        -float(metrics.get("recall_at_20") or 0.0),
        float(false_negative_rate if false_negative_rate is not None else 1.0),
        -float(metrics.get("precision_at_20") or 0.0),
        -float(metrics.get("mrr") or 0.0),
        config_distance(evaluation.config),
        evaluation.config_hash,
    )


def select_best_evaluation(
    evaluations: list[CalibrationEvaluation],
) -> CalibrationEvaluation:
    if not evaluations:
        raise ValueError("calibration grid produced no evaluations")
    return min(evaluations, key=selection_sort_key)


def config_distance(config: JudgementRuleConfig) -> float:
    tuned_fields = (
        "highly_relevant_threshold",
        "partially_relevant_threshold",
        "weakly_relevant_threshold",
        "title_topic_weight",
        "abstract_topic_weight",
        "missing_abstract_penalty",
        "minimum_evidence_count",
    )
    return round(
        sum(
            abs(
                float(getattr(config, field))
                - float(getattr(CURRENT_RULES_CONFIG, field))
            )
            for field in tuned_fields
        ),
        8,
    )


def threshold_sensitivity(
    evaluations: list[CalibrationEvaluation],
) -> dict[str, Any]:
    fields = {
        "partially_relevant_threshold": (
            "f1_at_20",
            "recall_at_20",
            "precision_at_20",
        ),
        "weakly_relevant_threshold": ("average_returned_paper_count",),
        "missing_abstract_penalty": ("gold_retained_count",),
        "minimum_evidence_count": ("gold_insufficient_evidence_count",),
    }
    output: dict[str, Any] = {"scope": "small_sample_diagnostic_only"}
    for field, metric_names in fields.items():
        grouped: dict[str, list[CalibrationEvaluation]] = defaultdict(list)
        for evaluation in evaluations:
            grouped[str(getattr(evaluation.config, field))].append(evaluation)
        output[field] = [
            {
                "value": value,
                "combination_count": len(items),
                **{
                    metric: _average(
                        float(item.metrics.get(metric) or 0.0) for item in items
                    )
                    for metric in metric_names
                },
            }
            for value, items in sorted(grouped.items(), key=lambda item: item[0])
        ]
    return output


def validation_acceptance(
    baseline: CalibrationEvaluation,
    calibrated: CalibrationEvaluation,
) -> dict[str, Any]:
    base = baseline.metrics
    candidate = calibrated.metrics
    base_returned = float(base.get("average_returned_paper_count") or 0.0)
    candidate_returned = float(
        candidate.get("average_returned_paper_count") or 0.0
    )
    checks = {
        "f1_at_20_non_regression": candidate["f1_at_20"] >= base["f1_at_20"],
        "recall_at_20_non_regression": (
            candidate["recall_at_20"] >= base["recall_at_20"]
        ),
        "gold_false_negative_non_regression": (
            _none_as_one(candidate.get("gold_judgement_false_negative_rate"))
            <= _none_as_one(base.get("gold_judgement_false_negative_rate"))
        ),
        "precision_at_20_non_regression": (
            candidate["precision_at_20"] >= base["precision_at_20"]
        ),
        "candidate_recall_identical": (
            candidate["candidate_recall"] == base["candidate_recall"]
        ),
        "returned_count_controlled": candidate_returned
        <= max(base_returned * 1.5, base_returned + 2.0),
        "hard_constraints_unchanged": (
            candidate.get("hard_constraint_failure_distribution", {})
            == base.get("hard_constraint_failure_distribution", {})
        ),
    }
    return {
        "accepted": all(checks.values()),
        "status": "calibration_candidate" if all(checks.values()) else "not_accepted",
        "checks": checks,
        "small_sample_diagnostic_only": True,
    }


def _gold_state(
    case: FrozenJudgementCase,
    *,
    judgements: list[Any],
    all_ranked: list[Any],
    returned: list[Any],
) -> dict[str, Any]:
    retrieved_ids = set(matched_paper_ids(case.papers, case.gold_papers))
    retained = [
        item
        for item in judgements
        if item.category in {"highly_relevant", "partially_relevant"}
    ]
    retained_ids = set(matched_paper_ids(retained, case.gold_papers))
    category_by_gold: dict[str, str] = {}
    for item in judgements:
        for match_id in matched_paper_ids([item], case.gold_papers, k=1):
            category_by_gold.setdefault(match_id, item.category)
    top20_ids = set(matched_paper_ids(all_ranked[:20], case.gold_papers))
    returned_ids = set(matched_paper_ids(returned, case.gold_papers))
    boundary_by_gold: dict[str, str] = {}
    for match_id in retrieved_ids:
        if match_id in returned_ids:
            boundary_by_gold[match_id] = "formal_returned"
        elif match_id in top20_ids:
            boundary_by_gold[match_id] = "entered_top20"
        elif match_id in retained_ids:
            boundary_by_gold[match_id] = (
                "judgement_retained_ranked_outside_top20"
            )
        else:
            boundary_by_gold[match_id] = "judgement_not_retained"
    return {
        "retrieved_ids": retrieved_ids,
        "retained_ids": retained_ids,
        "category_by_gold": category_by_gold,
        "boundary_by_gold": boundary_by_gold,
    }


def _candidate_diagnostics(
    case: FrozenJudgementCase,
    *,
    judgements: list[Any],
    all_ranked: list[Any],
    returned: list[Any],
    false_negative_ids: set[str],
) -> list[dict[str, Any]]:
    rank_by_key = {
        _paper_key(item.paper, index): item.rank
        for index, item in enumerate(all_ranked)
    }
    returned_keys = {
        _paper_key(item.paper, index) for index, item in enumerate(returned)
    }
    rows: list[dict[str, Any]] = []
    for index, judgement in enumerate(judgements):
        paper = judgement.paper
        feature = judgement.feature_vector
        match_ids = matched_paper_ids([judgement], case.gold_papers, k=1)
        is_gold = bool(match_ids)
        key = _paper_key(paper, index)
        final_rank = rank_by_key.get(key)
        returned_status = key in returned_keys
        gold_failure_reason = None
        if is_gold and any(match_id in false_negative_ids for match_id in match_ids):
            gold_failure_reason = _gold_failure_reason(judgement)
        rows.append(
            {
                "case_id": case.case_id,
                "paper_stable_identifier": canonical_paper_id(paper),
                "identifiers": paper.identifiers.model_dump(mode="json"),
                "title": paper.title,
                "year": paper.year,
                "source": list(paper.sources),
                "judgement_category": judgement.category,
                "judgement_score": judgement.score,
                "rule_evidence": [
                    {"source": item.source, "confidence": item.confidence}
                    for item in judgement.evidence
                ],
                "matched_topic_terms": (
                    feature.matched_topic_terms if feature else []
                ),
                "matched_method_terms": (
                    feature.matched_method_terms if feature else []
                ),
                "matched_dataset_terms": (
                    feature.matched_dataset_terms if feature else []
                ),
                "matched_task_terms": (
                    feature.matched_task_terms if feature else []
                ),
                "must_have_coverage": _coverage(
                    case.query_analysis.constraints.must_include_terms,
                    feature.matched_must_have_terms if feature else [],
                ),
                "excluded_term_hits": _warning_values(
                    judgement.warnings,
                    "excluded_terms_matched:",
                ),
                "title_match_score": feature.title_match_score if feature else 0.0,
                "abstract_match_score": (
                    feature.abstract_match_score if feature else 0.0
                ),
                "venue_match": feature.venue_match if feature else None,
                "temporal_match": feature.temporal_match if feature else None,
                "metadata_completeness": (
                    feature.metadata_completeness if feature else 0.0
                ),
                "hard_constraint_result": {
                    "passed": not feature.hard_constraint_failures if feature else True,
                    "failures": feature.hard_constraint_failures if feature else [],
                },
                "score_components": feature.score_components if feature else {},
                "category_reason": feature.category_reason if feature else "",
                "final_returned_status": returned_status,
                "final_rank": final_rank,
                "post_run_gold_match": is_gold,
                "post_run_gold_match_keys": match_ids,
                "gold_failure_reason": gold_failure_reason,
                "benchmark_candidate_label": (
                    None
                    if is_gold
                    else "benchmark_non_gold_returned"
                    if returned_status
                    else "benchmark_non_gold_candidate"
                ),
                "reranking_followup_needed": bool(
                    is_gold
                    and judgement.category
                    in {"highly_relevant", "partially_relevant"}
                    and (final_rank or 0) > 20
                ),
            }
        )
    return rows


def _gold_failure_reason(judgement: Any) -> str:
    feature = judgement.feature_vector
    warnings = set(judgement.warnings)
    if any(item.startswith("missing_must_have_terms:") for item in warnings):
        return "gold_failed_must_have"
    if any(item.startswith("outside_time_range:") for item in warnings):
        return "gold_failed_temporal_constraint"
    if feature and feature.venue_match is False:
        return "gold_failed_venue_constraint"
    if "missing_abstract" in warnings:
        return "gold_missing_abstract"
    if feature and feature.metadata_completeness < 0.6:
        return "gold_missing_metadata"
    if judgement.category == "insufficient_evidence":
        return "gold_insufficient_evidence"
    if feature and judgement.score < feature.partially_relevant_threshold:
        return "gold_score_below_partial_threshold"
    if judgement.category == "weakly_relevant":
        return "gold_judged_weak"
    if judgement.category == "irrelevant":
        return "gold_judged_irrelevant"
    return "gold_category_not_returned"


def _aggregate_replay_cost(cases: list[FrozenJudgementCase]) -> dict[str, Any]:
    fields = (
        "retrieval_snapshot_hits",
        "reference_snapshot_hits",
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
    )
    return {
        field: sum(float(case.replay_cost.get(field) or 0.0) for case in cases)
        for field in fields
    }


def _paper_key(paper: Paper, fallback: int) -> str:
    return canonical_paper_id(paper) or f"candidate:{fallback}:{paper.title.casefold()}"


def _coverage(expected: list[str], matched: list[str]) -> float | None:
    expected_keys = {item.casefold() for item in expected if item.strip()}
    if not expected_keys:
        return None
    matched_keys = {item.casefold() for item in matched if item.strip()}
    return len(expected_keys & matched_keys) / len(expected_keys)


def _warning_values(warnings: list[str], prefix: str) -> list[str]:
    values: list[str] = []
    for warning in warnings:
        if warning.startswith(prefix):
            values.extend(item for item in warning[len(prefix) :].split(",") if item)
    return values


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _average_optional(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _none_as_one(value: Any) -> float:
    return 1.0 if value is None else float(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
