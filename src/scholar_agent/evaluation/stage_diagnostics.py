"""Benchmark 阶段命中、丢失原因、来源贡献和瓶颈分析。"""

from __future__ import annotations

import statistics
from collections import Counter
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.core.dedup import normalize_title
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.pipeline_diagnostics import (
    DiagnosticCandidate,
    StageCandidateSnapshot,
)
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    evaluable_gold_count,
    gold_crosswalk_status,
    matched_paper_ids,
    paper_identifier_set,
    recall_at_k,
)
from scholar_agent.evaluation.selection import ResultPolicy, select_ranked_results
from scholar_agent.services.search_service import SearchServiceOutput


DIAGNOSTIC_K_VALUES = (5, 10, 20, 50)
RETURN_CATEGORIES = {"highly_relevant", "partially_relevant"}
FALSE_NEGATIVE_CATEGORIES = {
    "weakly_relevant",
    "irrelevant",
    "insufficient_evidence",
}
DIAGNOSTIC_QUERY_BOILERPLATE = {
    "a",
    "an",
    "about",
    "can",
    "could",
    "find",
    "for",
    "from",
    "give",
    "in",
    "latest",
    "list",
    "looking",
    "me",
    "of",
    "on",
    "paper",
    "papers",
    "please",
    "show",
    "some",
    "tell",
    "the",
    "to",
    "want",
    "with",
    "you",
}
RETRIEVAL_STAGES = (
    "initial_retrieval",
    "semantic_seed_expansion_retrieval",
    "query_evolution_retrieval",
    "refchain_retrieval",
)
STAGE_ORDER = (
    "initial_retrieval",
    "initial_deduplicated",
    "initial_judged",
    "initial_reranked",
    "semantic_seed_expansion_retrieval",
    "post_semantic_seed_expansion_deduplicated",
    "post_semantic_seed_expansion_judged",
    "post_semantic_seed_expansion_reranked",
    "query_evolution_retrieval",
    "post_evolution_deduplicated",
    "post_evolution_judged",
    "post_evolution_reranked",
    "refchain_retrieval",
    "post_refchain_deduplicated",
    "post_refchain_judged",
    "post_refchain_reranked",
    "final_ranked",
    "final_returned",
)


class GoldStageDiagnostic(BaseModel):
    case_id: str
    query: str
    gold_id: str | None
    gold_title: str | None
    found: bool
    first_found_stage: str | None = None
    sources: list[str] = Field(default_factory=list)
    initial_rank: int | None = None
    final_rank: int | None = None
    final_category: str | None = None
    drop_reason: str


def analyze_search_stages(
    eval_query: EvalQuery,
    output: SearchServiceOutput,
    *,
    result_policy: ResultPolicy,
) -> dict[str, Any]:
    snapshots = list(output.stage_snapshots)
    snapshots.append(_final_returned_snapshot(output, result_policy))
    by_stage = {snapshot.stage: snapshot for snapshot in snapshots}
    gold_diagnostics = [
        _analyze_gold(eval_query, gold, by_stage, output)
        for gold in eval_query.gold_papers
        if gold.relevance_grade > 0
    ]
    stage_metrics = _case_stage_metrics(eval_query, by_stage)
    judgement = _judgement_metrics(gold_diagnostics)
    reranking = _reranking_metrics(gold_diagnostics)
    source_contribution = _source_contribution(
        eval_query,
        snapshots,
        output,
    )
    retrieval_diagnostics = _retrieval_query_diagnostics(output)
    query_strategy_contribution = _query_strategy_gold_contribution(
        eval_query,
        snapshots,
        output,
    )
    initial_query_planning = _initial_query_planning_diagnostics(
        eval_query,
        by_stage,
        output,
    )
    query_evolution = _query_evolution_diagnostics(
        eval_query,
        by_stage,
        output,
    )
    refchain = _refchain_diagnostics(eval_query, by_stage, output)
    semantic_seed_expansion = _semantic_seed_expansion_diagnostics(
        eval_query,
        by_stage,
        output,
    )
    stage_costs = _stage_cost_diagnostics(
        output,
        by_stage,
        query_evolution=query_evolution,
        refchain=refchain,
        semantic_seed_expansion=semantic_seed_expansion,
    )
    query_evolution["conclusions"] = classify_module_outcome(
        query_evolution,
        stage_costs["query_evolution"],
        case_count=1,
    )
    refchain["conclusions"] = classify_module_outcome(
        refchain,
        stage_costs["refchain"],
        case_count=1,
    )
    semantic_seed_expansion["conclusions"] = classify_module_outcome(
        semantic_seed_expansion,
        stage_costs["semantic_seed_expansion"],
        case_count=1,
    )
    return {
        "snapshots": [item.model_dump(mode="json") for item in snapshots],
        "gold_diagnostics": [item.model_dump(mode="json") for item in gold_diagnostics],
        "evaluable_gold_count": evaluable_gold_count(eval_query.gold_papers),
        "stage_metrics": stage_metrics,
        "judgement": judgement,
        "reranking": reranking,
        "source_contribution": source_contribution,
        "retrieval_diagnostics": retrieval_diagnostics,
        "query_strategy_contribution": query_strategy_contribution,
        "initial_query_planning": initial_query_planning,
        "query_evolution": query_evolution,
        "refchain": refchain,
        "semantic_seed_expansion": semantic_seed_expansion,
        "stage_costs": stage_costs,
        "budget_stopped": bool(output.budget_status.stop_reasons),
        "budget_stop_reasons": list(output.budget_status.stop_reasons),
    }


def aggregate_stage_diagnostics(
    case_diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    gold_rows = [
        row
        for case in case_diagnostics
        for row in case.get("gold_diagnostics", [])
    ]
    candidate_names = sorted(
        {
            name
            for case in case_diagnostics
            for name in case.get("stage_metrics", {}).get("candidate_recall", {})
        }
    )
    ranked_names = sorted(
        {
            name
            for case in case_diagnostics
            for name in case.get("stage_metrics", {}).get("recall_at_k", {})
        }
    )
    candidate_recall = {
        name: _average_optional(
            [
                case.get("stage_metrics", {})
                .get("candidate_recall", {})
                .get(name)
                for case in case_diagnostics
            ]
        )
        for name in candidate_names
    }
    recall_at_k = {
        name: {
            str(k): _average_optional(
                [
                    case.get("stage_metrics", {})
                    .get("recall_at_k", {})
                    .get(name, {})
                    .get(str(k))
                    for case in case_diagnostics
                ]
            )
            for k in DIAGNOSTIC_K_VALUES
        }
        for name in ranked_names
    }
    judgement = _aggregate_judgement(case_diagnostics)
    reranking = _aggregate_reranking(case_diagnostics)
    evaluable_gold_total = sum(
        int(case.get("evaluable_gold_count") or 0) for case in case_diagnostics
    )
    sources = _aggregate_sources(case_diagnostics, evaluable_gold_total)
    retrieval_diagnostics = _aggregate_retrieval_diagnostics(case_diagnostics)
    query_strategy_contribution = _aggregate_strategy_contribution(
        case_diagnostics
    )
    initial_query_planning = _aggregate_initial_query_planning(
        case_diagnostics
    )
    query_evolution = _aggregate_module_diagnostics(
        case_diagnostics,
        "query_evolution",
    )
    refchain = _aggregate_module_diagnostics(case_diagnostics, "refchain")
    semantic_seed_expansion = _aggregate_module_diagnostics(
        case_diagnostics,
        "semantic_seed_expansion",
    )
    stage_costs = _aggregate_stage_costs(case_diagnostics)
    query_evolution["conclusions"] = classify_module_outcome(
        query_evolution,
        stage_costs["query_evolution"],
        case_count=len(case_diagnostics),
    )
    refchain["conclusions"] = classify_module_outcome(
        refchain,
        stage_costs["refchain"],
        case_count=len(case_diagnostics),
    )
    semantic_seed_expansion["conclusions"] = classify_module_outcome(
        semantic_seed_expansion,
        stage_costs["semantic_seed_expansion"],
        case_count=len(case_diagnostics),
    )
    drop_reasons = dict(
        sorted(Counter(str(row.get("drop_reason")) for row in gold_rows).items())
    )
    budget_stop_rate = (
        sum(bool(case.get("budget_stopped")) for case in case_diagnostics)
        / len(case_diagnostics)
        if case_diagnostics
        else 0.0
    )
    stage_metrics = {
        "case_count": len(case_diagnostics),
        "gold_count": len(gold_rows),
        "evaluable_gold_count": evaluable_gold_total,
        "candidate_recall": candidate_recall,
        "recall_at_k": recall_at_k,
        "initial_retrieval_recall": candidate_recall.get("initial_retrieval"),
        "initial_deduplicated_recall": candidate_recall.get(
            "initial_deduplicated"
        ),
        "post_judgement_recall": candidate_recall.get(
            "post_judgement_retained"
        ),
        "post_rerank_recall": recall_at_k.get("initial_reranked", {}),
        "post_evolution_recall": candidate_recall.get(
            "post_evolution_deduplicated"
        ),
        "post_semantic_seed_expansion_recall": candidate_recall.get(
            "post_semantic_seed_expansion_deduplicated"
        ),
        "post_refchain_recall": candidate_recall.get(
            "post_refchain_deduplicated"
        ),
        "final_returned_recall": recall_at_k.get("final_returned", {}),
        "judgement": judgement,
        "reranking": reranking,
        "source_contribution": sources,
        "retrieval_diagnostics": retrieval_diagnostics,
        "query_strategy_contribution": query_strategy_contribution,
        "initial_query_planning": initial_query_planning,
        "query_evolution": query_evolution,
        "refchain": refchain,
        "semantic_seed_expansion": semantic_seed_expansion,
        "stage_costs": stage_costs,
        "budget_stop_rate": budget_stop_rate,
    }
    bottlenecks = classify_bottlenecks(stage_metrics)
    stage_metrics["bottleneck_labels"] = bottlenecks
    stage_metrics["sample_warning"] = "small_sample_diagnostic_only"
    error_analysis = {
        "drop_reasons": drop_reasons,
        "bottleneck_labels": bottlenecks,
        "sample_warning": "small_sample_diagnostic_only",
    }
    return stage_metrics, error_analysis, gold_rows


def classify_bottlenecks(stage_metrics: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    case_count = int(stage_metrics.get("case_count") or 0)
    if case_count < 30:
        labels.append("insufficient_sample")
    initial_recall = stage_metrics.get("initial_retrieval_recall")
    if initial_recall is not None and float(initial_recall) < 0.5:
        labels.append("retrieval_recall_bottleneck")
    judgement = stage_metrics.get("judgement") or {}
    if float(judgement.get("gold_false_negative_rate") or 0.0) >= 0.25:
        labels.append("judgement_false_negative_bottleneck")
    reranking = stage_metrics.get("reranking") or {}
    if (
        int(reranking.get("eligible_gold_count") or 0) > 0
        and float(reranking.get("outside_top_20_rate") or 0.0) >= 0.25
    ):
        labels.append("reranking_bottleneck")
    sources = stage_metrics.get("source_contribution") or {}
    if float(sources.get("source_error_rate") or 0.0) >= 0.2:
        labels.append("source_reliability_bottleneck")
    if float(stage_metrics.get("budget_stop_rate") or 0.0) >= 0.2:
        labels.append("budget_bottleneck")
    return labels


def _final_returned_snapshot(
    output: SearchServiceOutput,
    result_policy: ResultPolicy,
) -> StageCandidateSnapshot:
    selected = select_ranked_results(output, policy=result_policy)
    final_snapshot = _snapshot_by_name(output.stage_snapshots, "final_ranked")
    final_candidates = final_snapshot.candidates if final_snapshot else []
    candidates: list[DiagnosticCandidate] = []
    for ranked in selected:
        matched = _matching_candidate(final_candidates, ranked)
        if matched is not None:
            candidates.append(matched.model_copy(deep=True))
            continue
        candidates.append(
            DiagnosticCandidate(
                identifiers=ranked.paper.identifiers,
                title=ranked.paper.title,
                year=ranked.paper.year,
                sources=list(ranked.paper.sources),
                rank=ranked.rank,
                category=ranked.category,
                final_score=ranked.final_score,
                matched_terms=list(ranked.matched_terms),
                warnings=list(ranked.warnings),
                rrf_score=ranked.rrf_score,
                rrf_contributions=[
                    item.model_dump(mode="json")
                    for item in ranked.rrf_contributions
                ],
                original_rank=ranked.original_rank,
                rrf_top_20_change=ranked.rrf_top_20_change,
                rrf_rank_change_reason=ranked.rrf_rank_change_reason,
            )
        )
    return StageCandidateSnapshot(stage="final_returned", candidates=candidates)


def _analyze_gold(
    eval_query: EvalQuery,
    gold: EvalGoldPaper,
    snapshots: dict[str, StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> GoldStageDiagnostic:
    stage_matches = {
        stage: _matches(snapshot.candidates, gold)
        for stage, snapshot in snapshots.items()
        if snapshot.status == "completed"
    }
    retrieval_matches = {
        stage: stage_matches.get(stage, []) for stage in RETRIEVAL_STAGES
    }
    found = any(retrieval_matches.values())
    first_found = next(
        (stage for stage in STAGE_ORDER if stage_matches.get(stage)),
        None,
    )
    sources = _stable_strings(
        source
        for matches in retrieval_matches.values()
        for candidate in matches
        for source in candidate.sources
    )
    initial_rank = _first_rank(stage_matches.get("initial_reranked", []))
    final_matches = stage_matches.get("final_ranked", [])
    final_rank = _first_rank(final_matches)
    judgement_match = _latest_match(
        stage_matches,
        (
            "post_refchain_judged",
            "post_evolution_judged",
            "post_semantic_seed_expansion_judged",
            "initial_judged",
        ),
    )
    final_category = (
        judgement_match.category
        if judgement_match is not None
        else (final_matches[0].category if final_matches else None)
    )
    drop_reason = _drop_reason(
        gold,
        found=found,
        stage_matches=stage_matches,
        final_category=final_category,
        final_rank=final_rank,
        output=output,
        snapshots=snapshots,
    )
    return GoldStageDiagnostic(
        case_id=eval_query.query_id,
        query=eval_query.query,
        gold_id=canonical_paper_id(gold),
        gold_title=gold.title,
        found=found,
        first_found_stage=first_found,
        sources=sources,
        initial_rank=initial_rank,
        final_rank=final_rank,
        final_category=final_category,
        drop_reason=drop_reason,
    )


def _drop_reason(
    gold: EvalGoldPaper,
    *,
    found: bool,
    stage_matches: dict[str, list[DiagnosticCandidate]],
    final_category: str | None,
    final_rank: int | None,
    output: SearchServiceOutput,
    snapshots: dict[str, StageCandidateSnapshot],
) -> str:
    if stage_matches.get("final_returned"):
        return "returned"
    if not found:
        crosswalk_status = gold_crosswalk_status(gold)
        if crosswalk_status in {"unavailable", "failed"}:
            return f"identity_crosswalk_{crosswalk_status}"
        initial = snapshots.get("initial_retrieval")
        if initial and initial.skipped_reason == "budget_stopped_before_retrieval":
            return "budget_stopped_before_retrieval"
        if _all_sources_failed(output):
            return "source_failed"
        if _title_seen_without_identifier_match(gold, snapshots):
            return "identifier_not_matched"
        return "not_retrieved"
    first_retrieval = next(
        (stage for stage in RETRIEVAL_STAGES if stage_matches.get(stage)),
        None,
    )
    expected_dedup = {
        "initial_retrieval": "initial_deduplicated",
        "semantic_seed_expansion_retrieval": (
            "post_semantic_seed_expansion_deduplicated"
        ),
        "query_evolution_retrieval": "post_evolution_deduplicated",
        "refchain_retrieval": "post_refchain_deduplicated",
    }.get(first_retrieval or "")
    if expected_dedup and not stage_matches.get(expected_dedup):
        return "removed_or_merged_by_dedup"
    if final_category == "irrelevant":
        return "judged_irrelevant"
    if final_category == "weakly_relevant":
        return "judged_weakly_relevant"
    if final_category == "insufficient_evidence":
        return "insufficient_evidence"
    if final_category in RETURN_CATEGORIES:
        if final_rank is None or final_rank > output.search_plan.top_k:
            return "outside_final_top_k"
    return "not_in_return_categories"


def _case_stage_metrics(
    eval_query: EvalQuery,
    snapshots: dict[str, StageCandidateSnapshot],
) -> dict[str, Any]:
    has_evaluable_gold = evaluable_gold_count(eval_query.gold_papers) > 0
    candidate_stage_names = (
        "initial_retrieval",
        "initial_deduplicated",
        "semantic_seed_expansion_retrieval",
        "post_semantic_seed_expansion_deduplicated",
        "query_evolution_retrieval",
        "post_evolution_deduplicated",
        "refchain_retrieval",
        "post_refchain_deduplicated",
    )
    candidate_recall = {
        stage: _candidate_recall(snapshot, eval_query.gold_papers)
        for stage in candidate_stage_names
        if (snapshot := snapshots.get(stage)) is not None
    }
    latest_judged = _latest_snapshot(
        snapshots,
        (
            "post_refchain_judged",
            "post_evolution_judged",
            "post_semantic_seed_expansion_judged",
            "initial_judged",
        ),
    )
    candidate_recall["post_judgement_retained"] = (
        _candidate_recall(
            latest_judged.model_copy(
                update={
                    "candidates": [
                        item
                        for item in latest_judged.candidates
                        if item.category in RETURN_CATEGORIES
                    ]
                }
            ),
            eval_query.gold_papers,
        )
        if latest_judged is not None
        else None
    )
    ranked_stages = (
        "initial_reranked",
        "post_semantic_seed_expansion_reranked",
        "post_evolution_reranked",
        "post_refchain_reranked",
        "final_ranked",
        "final_returned",
    )
    recall_maps: dict[str, dict[str, float | None]] = {}
    for stage in ranked_stages:
        snapshot = snapshots.get(stage)
        recall_maps[stage] = {
            str(k): (
                recall_at_k(snapshot.candidates, eval_query.gold_papers, k)
                if snapshot is not None
                and snapshot.status == "completed"
                and has_evaluable_gold
                else None
            )
            for k in DIAGNOSTIC_K_VALUES
        }
    return {
        "candidate_recall": candidate_recall,
        "recall_at_k": recall_maps,
    }


def _judgement_metrics(
    gold_diagnostics: list[GoldStageDiagnostic],
) -> dict[str, Any]:
    retrieved = [item for item in gold_diagnostics if item.found]
    counts = Counter(item.final_category for item in retrieved if item.final_category)
    false_negative_count = sum(
        item.final_category in FALSE_NEGATIVE_CATEGORIES
        and item.drop_reason != "returned"
        for item in retrieved
    )
    retained = sum(item.final_category in RETURN_CATEGORIES for item in retrieved)
    denominator = len(retrieved)
    return {
        "retrieved_gold_count": denominator,
        "gold_judged_highly_relevant": counts["highly_relevant"],
        "gold_judged_partially_relevant": counts["partially_relevant"],
        "gold_judged_weakly_relevant": counts["weakly_relevant"],
        "gold_judged_irrelevant": counts["irrelevant"],
        "gold_judged_insufficient_evidence": counts["insufficient_evidence"],
        "gold_retention_after_judgement": (
            retained / denominator if denominator else 0.0
        ),
        "gold_false_negative_count": false_negative_count,
        "gold_false_negative_rate": (
            false_negative_count / denominator if denominator else 0.0
        ),
    }


def _reranking_metrics(
    gold_diagnostics: list[GoldStageDiagnostic],
) -> dict[str, Any]:
    eligible = [
        item
        for item in gold_diagnostics
        if item.found and item.final_category in RETURN_CATEGORIES
    ]
    ranks = [item.final_rank for item in eligible if item.final_rank is not None]
    outside = sum(item.final_rank is None or item.final_rank > 20 for item in eligible)
    return {
        "eligible_gold_count": len(eligible),
        "gold_in_top_5": sum(rank <= 5 for rank in ranks),
        "gold_in_top_10": sum(rank <= 10 for rank in ranks),
        "gold_in_top_20": sum(rank <= 20 for rank in ranks),
        "gold_outside_top_20": outside,
        "outside_top_20_rate": outside / len(eligible) if eligible else 0.0,
        "average_gold_rank": statistics.mean(ranks) if ranks else None,
        "median_gold_rank": statistics.median(ranks) if ranks else None,
        "gold_ranks": ranks,
    }


def _source_contribution(
    eval_query: EvalQuery,
    snapshots: list[StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    candidates = [
        candidate
        for snapshot in snapshots
        if snapshot.stage in RETRIEVAL_STAGES and snapshot.status == "completed"
        for candidate in snapshot.candidates
    ]
    sources = sorted(
        {
            _normalize_source(source)
            for candidate in candidates
            for source in candidate.sources
            if _normalize_source(source)
        }.union(output.search_plan.selected_sources)
    )
    candidate_ids_by_source: dict[str, set[str]] = {source: set() for source in sources}
    gold_ids_by_source: dict[str, set[str]] = {source: set() for source in sources}
    for candidate in candidates:
        candidate_id = canonical_paper_id(candidate)
        candidate_sources = {
            _normalize_source(source) for source in candidate.sources
        } - {""}
        for source in candidate_sources:
            candidate_ids_by_source.setdefault(source, set())
            gold_ids_by_source.setdefault(source, set())
            if candidate_id:
                candidate_ids_by_source[source].add(candidate_id)
            for index, gold in enumerate(eval_query.gold_papers):
                if _candidate_matches(candidate, gold):
                    gold_ids_by_source[source].add(f"{eval_query.query_id}:{index}")

    stats = {source: _empty_source_stats() for source in sources}
    for source_stat in output.source_stats:
        source = "openalex" if source_stat.source == "refchain" else source_stat.source
        if source not in stats:
            stats[source] = _empty_source_stats()
        diagnostics = source_stat.diagnostics
        stats[source]["request_count"] += diagnostics.request_count
        stats[source]["success_count"] += int(
            source_stat.error_message is None
            and source_stat.source_skipped_reason is None
        )
        stats[source]["error_count"] += diagnostics.error_count
        stats[source]["returned_candidate_count"] += source_stat.returned_count
        stats[source]["latency_seconds"] += source_stat.latency_seconds

    total_gold = evaluable_gold_count(eval_query.gold_papers)
    for source, item in stats.items():
        source_candidates = candidate_ids_by_source.get(source, set())
        source_gold = gold_ids_by_source.get(source, set())
        other_gold = set().union(
            *(
                values
                for other, values in gold_ids_by_source.items()
                if other != source
            )
        ) if len(gold_ids_by_source) > 1 else set()
        item["unique_candidate_count"] = len(source_candidates)
        item["gold_hit_count"] = len(source_gold)
        item["unique_gold_hit_count"] = len(source_gold - other_gold)
        item["gold_recall_contribution"] = (
            len(source_gold - other_gold) / total_gold if total_gold else 0.0
        )

    overlap: dict[str, dict[str, int]] = {}
    for left, right in combinations(sorted(stats), 2):
        overlap[f"{left} ∩ {right}"] = {
            "candidate_count": len(
                candidate_ids_by_source.get(left, set())
                & candidate_ids_by_source.get(right, set())
            ),
            "gold_hit_count": len(
                gold_ids_by_source.get(left, set())
                & gold_ids_by_source.get(right, set())
            ),
        }
    request_count = sum(item["request_count"] for item in stats.values())
    error_count = sum(item["error_count"] for item in stats.values())
    return {
        "sources": stats,
        "overlap": overlap,
        "source_error_rate": error_count / request_count if request_count else 0.0,
    }


def _initial_query_planning_diagnostics(
    eval_query: EvalQuery,
    snapshots: dict[str, StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    snapshot = snapshots.get("initial_retrieval")
    subqueries = list(output.search_plan.subqueries)
    calls = list(snapshot.retrieval_calls if snapshot is not None else [])
    candidates = list(snapshot.candidates if snapshot is not None else [])
    candidate_sets = {
        subquery.query: {
            identifier
            for candidate in candidates
            if _candidate_has_initial_query(candidate, subquery.query)
            if (identifier := canonical_paper_id(candidate))
        }
        for subquery in subqueries
    }
    gold_sets = {
        subquery.query: {
            f"{eval_query.query_id}:{index}"
            for index, gold in enumerate(eval_query.gold_papers)
            if any(
                _candidate_matches(candidate, gold)
                for candidate in candidates
                if _candidate_has_initial_query(candidate, subquery.query)
            )
        }
        for subquery in subqueries
    }
    query_rows: list[dict[str, Any]] = []
    ineffective: Counter[str] = Counter()
    facet_contribution: dict[str, dict[str, int]] = {}
    original_query = output.search_plan.query_analysis.original_query
    for subquery in subqueries:
        matching_calls = [
            call for call in calls if call.origin_subquery == subquery.query
        ]
        raw_count = sum(call.returned_count for call in matching_calls)
        unique_ids = candidate_sets[subquery.query]
        other_ids = set().union(
            *(
                values
                for query, values in candidate_sets.items()
                if query != subquery.query
            )
        ) if len(candidate_sets) > 1 else set()
        exclusive_ids = unique_ids - other_ids
        gold_ids = gold_sets[subquery.query]
        other_gold = set().union(
            *(
                values
                for query, values in gold_sets.items()
                if query != subquery.query
            )
        ) if len(gold_sets) > 1 else set()
        duplicate_ratio = (
            max(0, raw_count - len(unique_ids)) / raw_count
            if raw_count
            else None
        )
        executed = any(call.logical_call_executed for call in matching_calls)
        recorded_requests = sum(
            call.recorded_request_count for call in matching_calls
        )
        recorded_latency = sum(
            call.recorded_latency_seconds for call in matching_calls
        )
        reasons = _initial_query_ineffective_reasons(
            subquery,
            subqueries,
            output,
            raw_count=raw_count,
            unique_count=len(unique_ids),
            duplicate_ratio=duplicate_ratio,
            gold_count=len(gold_ids),
            calls=matching_calls,
        )
        ineffective.update(reasons)
        for facet_type in subquery.facet_types:
            bucket = facet_contribution.setdefault(
                facet_type,
                {"unique_candidate_count": 0, "unique_gold_count": 0},
            )
            bucket["unique_candidate_count"] += len(exclusive_ids)
            bucket["unique_gold_count"] += len(gold_ids - other_gold)
        query_rows.append(
            {
                "query": subquery.query,
                "combination_mode": subquery.combination_mode,
                "purpose": subquery.purpose,
                "facet_types": list(subquery.facet_types),
                "provenance": list(subquery.provenance),
                "source_hints": list(subquery.source_hints),
                "adapted_queries": _stable_reasons(
                    call.adapted_query
                    for call in matching_calls
                    if call.adapted_query
                ),
                "status": "executed" if executed else "skipped",
                "skip_reasons": _stable_reasons(
                    call.source_skipped_reason
                    for call in matching_calls
                    if call.source_skipped_reason
                ),
                "raw_candidate_count": raw_count,
                "unique_candidate_count": len(unique_ids),
                "exclusive_candidate_count": len(exclusive_ids),
                "post_run_gold_hit_count": len(gold_ids),
                "post_run_unique_gold_hit_count": len(gold_ids - other_gold),
                "duplicate_candidate_ratio": duplicate_ratio,
                "duplicate_call_count": sum(
                    call.run_dedupe_hit for call in matching_calls
                ),
                "query_character_count": len(subquery.query),
                "query_term_count": len(_planning_terms(subquery.query)),
                "information_retention": _planning_information_retention(
                    original_query,
                    subquery.query,
                ),
                "dimension_coverage": {
                    facet: facet in subquery.facet_types
                    for facet in (
                        "topic",
                        "method",
                        "dataset",
                        "task",
                        "paper_type",
                        "venue",
                        "temporal",
                    )
                },
                "request_count": sum(
                    call.request_count for call in matching_calls
                ),
                "recorded_request_count": recorded_requests,
                "recorded_latency_seconds": recorded_latency,
                "ineffective_reasons": reasons,
            }
        )
    all_gold = set().union(*gold_sets.values()) if gold_sets else set()
    all_ids = set().union(*candidate_sets.values()) if candidate_sets else set()
    planning = output.search_plan.query_planning
    planning_payload = planning.model_dump(mode="json")
    if output.search_plan.query_planning_policy not in {
        "llm_semantic",
        "llm_constrained_rewrite",
    }:
        for field in (
            "provider",
            "model",
            "prompt_name",
            "prompt_version",
            "prompt_hash",
            "snapshot_key",
            "snapshot_status",
            "llm_call_attempted",
            "replayed",
            "fallback_used",
            "fallback_reason",
            "output_valid",
            "original_query_retained",
            "generated_query_count",
            "accepted_query_count",
            "rejected_query_count",
            "rejection_reasons",
            "accepted_queries",
            "terminology_expansions",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_total_tokens",
            "recorded_llm_latency_seconds",
            "constrained_rewrite_input_summary",
            "constrained_rewrite_query",
            "constrained_rewrite_replaced_index",
            "constrained_rewrite_replaced_query",
            "constrained_rewrite_replaced_purpose",
            "constrained_rewrite_skip_reason",
            "constrained_rewrite_validation_rejections",
        ):
            planning_payload.pop(field, None)
    return {
        "policy": output.search_plan.query_planning_policy,
        "planner_version": planning.planner_version,
        "original_query": original_query,
        "query_analysis": output.search_plan.query_analysis.model_dump(mode="json"),
        "planning": planning_payload,
        "subqueries": query_rows,
        "subquery_count": len(subqueries),
        "adapted_query_count": len(
            {
                " ".join((call.adapted_query or "").casefold().split())
                for call in calls
                if call.adapted_query
            }
        ),
        "raw_candidate_count": sum(item["raw_candidate_count"] for item in query_rows),
        "unique_candidate_count": len(all_ids),
        "duplicate_candidate_ratio": (
            max(0, sum(item["raw_candidate_count"] for item in query_rows) - len(all_ids))
            / sum(item["raw_candidate_count"] for item in query_rows)
            if sum(item["raw_candidate_count"] for item in query_rows)
            else None
        ),
        "unique_gold_count": len(all_gold),
        "request_count": sum(item["request_count"] for item in query_rows),
        "recorded_request_count": sum(
            item["recorded_request_count"] for item in query_rows
        ),
        "recorded_latency_seconds": sum(
            item["recorded_latency_seconds"] for item in query_rows
        ),
        "source_error_count": sum(
            (
                call.recorded_error_count
                if call.recorded_request_count
                else call.error_count
            )
            for call in calls
        ),
        "facet_contribution": dict(sorted(facet_contribution.items())),
        "ineffective_reasons": dict(sorted(ineffective.items())),
    }


def _candidate_has_initial_query(candidate: Any, query: str) -> bool:
    return any(
        provenance.origin_stage == "initial_retrieval"
        and provenance.origin_subquery == query
        for provenance in candidate.provenance
    )


def _initial_query_ineffective_reasons(
    subquery: Any,
    subqueries: list[Any],
    output: SearchServiceOutput,
    *,
    raw_count: int,
    unique_count: int,
    duplicate_ratio: float | None,
    gold_count: int,
    calls: list[Any],
) -> list[str]:
    reasons: list[str] = []
    original = output.search_plan.query_analysis.original_query
    normalized = " ".join(subquery.query.casefold().split())
    if subquery.purpose != "original_query" and normalized == " ".join(
        original.casefold().split()
    ):
        reasons.append("original_query_repeated")
    if any(
        other is not subquery
        and _planning_semantic_similarity(other, subquery) >= 0.85
        for other in subqueries
    ):
        reasons.append("duplicate_semantics")
    retention = _planning_information_retention(original, subquery.query)
    if retention < 0.4:
        reasons.append("missing_core_topic")
    if len(subquery.facet_types) > 3:
        reasons.append("too_many_dimensions_combined")
    if subquery.purpose != "original_query":
        for facet, reason in (
            ("method", "missing_method"),
            ("dataset", "missing_dataset"),
            ("task", "missing_task"),
        ):
            if subquery.purpose == f"facet_{facet}" and facet not in subquery.facet_types:
                reasons.append(reason)
    suffixes = {"survey", "review", "benchmark", "evaluation", "comparison"}
    original_terms = _planning_terms(original)
    query_terms = _planning_terms(subquery.query)
    additions = query_terms - original_terms
    if additions and additions <= suffixes and not subquery.facet_types:
        reasons.append("generic_suffix_only")
    if len(query_terms) <= 1:
        reasons.append("over_broad")
    if raw_count == 0 and any(call.logical_call_executed for call in calls):
        reasons.append("over_restrictive")
    if raw_count and unique_count < 5:
        reasons.append("low_unique_candidate_yield")
    if duplicate_ratio is not None and duplicate_ratio >= 0.5:
        reasons.append("high_duplicate_candidate_yield")
    if raw_count and gold_count == 0:
        reasons.append("no_gold_hit")
    if any(
        (
            call.recorded_error_count
            if call.recorded_request_count
            else call.error_count
        )
        for call in calls
    ):
        reasons.append("source_failure")
    if any(
        "budget" in (call.source_skipped_reason or "").casefold()
        or "max_" in (call.source_skipped_reason or "").casefold()
        for call in calls
    ):
        reasons.append("budget_stop")
    if not reasons and not gold_count:
        reasons.append("unknown")
    return _stable_reasons(reasons)


def _planning_semantic_similarity(left: Any, right: Any) -> float:
    left_terms = _planning_terms(left.query)
    right_terms = _planning_terms(right.query)
    union = left_terms | right_terms
    if not union:
        return 1.0
    term_similarity = len(left_terms & right_terms) / len(union)
    purpose_bonus = 0.05 if left.purpose == right.purpose else 0.0
    facet_bonus = 0.05 if set(left.facet_types) == set(right.facet_types) else 0.0
    return min(1.0, term_similarity + purpose_bonus + facet_bonus)


def _planning_information_retention(original: str, query: str) -> float:
    original_terms = _planning_terms(original)
    if not original_terms:
        return 1.0
    return len(original_terms & _planning_terms(query)) / len(original_terms)


def _planning_terms(value: str) -> set[str]:
    return {
        term
        for term in normalize_title(value).split()
        if len(term) > 1 and term not in DIAGNOSTIC_QUERY_BOILERPLATE
    }


def _query_evolution_diagnostics(
    eval_query: EvalQuery,
    snapshots: dict[str, StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    initial = snapshots.get("initial_deduplicated")
    evolved = snapshots.get("query_evolution_retrieval")
    post_evolved = snapshots.get("post_evolution_deduplicated")
    final_returned = snapshots.get("final_returned")
    initial_ids = _candidate_ids(initial)
    evolved_ids = _candidate_ids(evolved)
    new_ids = evolved_ids - initial_ids
    initial_gold = _gold_match_keys(eval_query, initial)
    evolved_gold = _gold_match_keys(eval_query, evolved)
    new_gold = evolved_gold - initial_gold
    returned_ids = _candidate_ids(final_returned)
    returned_gold = _gold_match_keys(eval_query, final_returned)
    records = list(output.query_evolution_records)
    generated = [query for record in records for query in record.generated_queries]
    retrieval_calls = list(evolved.retrieval_calls) if evolved is not None else []
    executed_queries = {
        call.origin_subquery
        for call in retrieval_calls
        if call.logical_call_executed
    }
    stage_skip = evolved.skipped_reason if evolved is not None else None
    query_details = _evolved_query_details(
        generated,
        retrieval_calls,
        evolved,
        snapshots.get("post_evolution_judged"),
        eval_query=eval_query,
        query_analysis=output.search_plan.query_analysis,
        initial_ids=initial_ids,
        stage_skip=stage_skip,
    )
    generated_queries = {item.query for item in generated}
    absent_queries = generated_queries - {
        call.origin_subquery for call in retrieval_calls
    }
    budget_skipped = (
        len(absent_queries) if _is_budget_reason(stage_skip) else 0
    )
    duplicate_queries = len(
        {
            item["query"]
            for item in query_details
            if item["skip_reason"] == "duplicate_query"
        }
    )
    eligible_seeds = max(
        sum(record.eligible_seed_count for record in records),
        _eligible_query_evolution_seed_count(
            snapshots.get("initial_reranked")
        ),
    )
    selected_seeds = sum(record.seed_count for record in records)
    skipped_reasons = _stable_reasons(
        [
            *(
                _normalize_query_evolution_skip(reason)
                for record in records
                for reason in record.skipped_reasons
            ),
            *(item["skip_reason"] for item in query_details),
        ]
    )
    if output.search_plan.enable_query_evolution and eligible_seeds == 0:
        skipped_reasons = _stable_reasons([*skipped_reasons, "no_eligible_seed"])
    if output.search_plan.enable_query_evolution and not generated:
        skipped_reasons = _stable_reasons([*skipped_reasons, "no_new_query"])
    category_stats = _new_candidate_categories(
        new_ids,
        snapshots.get("post_evolution_judged"),
    )
    judgement_filtered, top_k_lost = _module_gold_losses(
        new_gold,
        snapshots.get("post_evolution_judged"),
        final_returned,
        eval_query,
    )
    prior_recall = _candidate_recall_value(initial, eval_query.gold_papers)
    post_recall = _candidate_recall_value(post_evolved or initial, eval_query.gold_papers)
    quality_gates = [record.quality_gate for record in records]
    coverage_gaps = [
        record.coverage_gap.model_dump(mode="json")
        for record in records
        if record.coverage_gap is not None
    ]
    return {
        "enabled": output.search_plan.enable_query_evolution,
        "policy": output.search_plan.query_evolution_policy,
        "triggered": bool(generated),
        "original_query": output.search_plan.query_analysis.original_query,
        "query_intent": output.search_plan.query_analysis.intent,
        "constraints": output.search_plan.query_analysis.constraints.model_dump(
            mode="json"
        ),
        "eligible_seed_count": eligible_seeds,
        "eligible_seed_titles": _stable_reasons(
            title
            for record in records
            for title in record.eligible_seed_titles
        ),
        "selected_seed_count": selected_seeds,
        "selected_seed_titles": _stable_reasons(
            title
            for record in records
            for title in record.seed_paper_titles
        ),
        "generated_query_count": len(generated),
        "executed_query_count": len(executed_queries),
        "duplicate_query_count": max(0, duplicate_queries),
        "budget_skipped_query_count": budget_skipped,
        "source_skipped_query_count": len(
            {
                item["query"]
                for item in query_details
                if item["skip_reason"] in {"source_cooldown", "source_failure"}
            }
        ),
        "evolved_raw_candidate_count": len(evolved.candidates) if evolved else 0,
        "evolved_unique_candidate_count": len(evolved_ids),
        "evolved_new_unique_candidate_count": len(new_ids),
        "evolved_gold_hit_count": _raw_gold_hit_count(eval_query, evolved),
        "evolved_unique_gold_hit_count": len(evolved_gold),
        "evolved_new_unique_gold_count": len(new_gold),
        "evolved_candidates_returned_count": len(new_ids & returned_ids),
        "evolved_gold_returned_count": len(new_gold & returned_gold),
        "gold_found_but_filtered_count": len(new_gold - returned_gold),
        "gold_filtered_by_judgement_count": judgement_filtered,
        "gold_lost_by_top_k_count": top_k_lost,
        "candidate_recall_gain": max(0.0, post_recall - prior_recall),
        "coverage_gaps": coverage_gaps,
        "quality_gate": {
            "raw_candidate_count": sum(
                gate.raw_candidate_count for gate in quality_gates
            ),
            "unique_candidate_count": sum(
                gate.unique_candidate_count for gate in quality_gates
            ),
            "duplicate_candidate_count": sum(
                gate.duplicate_candidate_count for gate in quality_gates
            ),
            "duplicate_with_initial_count": sum(
                gate.duplicate_with_initial_count for gate in quality_gates
            ),
            "accepted_candidate_count": sum(
                gate.accepted_candidate_count for gate in quality_gates
            ),
            "filtered_candidate_count": sum(
                gate.filtered_candidate_count for gate in quality_gates
            ),
            "filtered_reason_counts": dict(
                sorted(
                    Counter(
                        {
                            reason: sum(
                                gate.filtered_reason_counts.get(reason, 0)
                                for gate in quality_gates
                            )
                            for reason in {
                                item
                                for gate in quality_gates
                                for item in gate.filtered_reason_counts
                            }
                        }
                    ).items()
                )
            ),
        },
        "new_candidate_categories": category_stats,
        "queries": query_details,
        "skipped_reasons": skipped_reasons,
    }


def _refchain_diagnostics(
    eval_query: EvalQuery,
    snapshots: dict[str, StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    prior = _latest_snapshot(
        snapshots,
        ("post_evolution_deduplicated", "initial_deduplicated"),
    )
    prior_ranked = _latest_snapshot(
        snapshots,
        ("post_evolution_reranked", "initial_reranked"),
    )
    references = snapshots.get("refchain_retrieval")
    post_refchain = snapshots.get("post_refchain_deduplicated")
    final_returned = snapshots.get("final_returned")
    prior_ids = _candidate_ids(prior)
    reference_ids = _candidate_ids(references)
    new_ids = reference_ids - prior_ids
    prior_gold = _gold_match_keys(eval_query, prior)
    reference_gold = _gold_match_keys(eval_query, references)
    new_gold = reference_gold - prior_gold
    returned_ids = _candidate_ids(final_returned)
    returned_gold = _gold_match_keys(eval_query, final_returned)
    refchain_output = output.refchain_output
    record = refchain_output.record if refchain_output is not None else None
    seeds = list(record.seeds) if record is not None else []
    supported = [seed for seed in seeds if _supported_seed_identifier(seed.paper)]
    seed_details = _refchain_seed_details(record, prior)
    skipped_reasons = _stable_reasons(
        [
            *(
                _normalize_refchain_skip(reason)
                for reason in (
                    record.skipped_reasons if record is not None else []
                )
            ),
            *(item["skip_reason"] for item in seed_details),
        ]
    )
    eligible_seeds = _eligible_refchain_seed_count(prior_ranked)
    if output.search_plan.enable_refchain and eligible_seeds == 0:
        skipped_reasons = _stable_reasons([*skipped_reasons, "no_eligible_seed"])
    category_stats = _new_candidate_categories(
        new_ids,
        snapshots.get("post_refchain_judged"),
    )
    judgement_filtered, top_k_lost = _module_gold_losses(
        new_gold,
        snapshots.get("post_refchain_judged"),
        final_returned,
        eval_query,
    )
    prior_recall = _candidate_recall_value(prior, eval_query.gold_papers)
    post_recall = _candidate_recall_value(
        post_refchain or prior,
        eval_query.gold_papers,
    )
    return {
        "enabled": output.search_plan.enable_refchain,
        "eligible_seed_count": eligible_seeds,
        "selected_seed_count": len(seeds),
        "seed_with_supported_identifier_count": len(supported),
        "seed_without_supported_identifier_count": len(seeds) - len(supported),
        "reference_request_count": (
            refchain_output.diagnostics.request_count if refchain_output else 0
        ),
        "recorded_reference_request_count": (
            refchain_output.recorded_diagnostics.request_count
            if refchain_output
            else 0
        ),
        "recorded_reference_latency_seconds": (
            refchain_output.recorded_latency_seconds if refchain_output else 0.0
        ),
        "raw_reference_count": len(references.candidates) if references else 0,
        "unique_reference_count": len(reference_ids),
        "new_unique_reference_count": len(new_ids),
        "reference_gold_hit_count": _raw_gold_hit_count(eval_query, references),
        "unique_reference_gold_hit_count": len(reference_gold),
        "new_unique_reference_gold_count": len(new_gold),
        "reference_candidates_returned_count": len(new_ids & returned_ids),
        "reference_gold_returned_count": len(new_gold & returned_gold),
        "gold_found_but_filtered_count": len(new_gold - returned_gold),
        "gold_filtered_by_judgement_count": judgement_filtered,
        "gold_lost_by_top_k_count": top_k_lost,
        "candidate_recall_gain": max(0.0, post_recall - prior_recall),
        "new_candidate_categories": category_stats,
        "seeds": seed_details,
        "skipped_reasons": skipped_reasons,
    }


def _semantic_seed_expansion_diagnostics(
    eval_query: EvalQuery,
    snapshots: dict[str, StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    prior = snapshots.get("initial_deduplicated")
    recommendations = snapshots.get("semantic_seed_expansion_retrieval")
    post = snapshots.get("post_semantic_seed_expansion_deduplicated")
    effective_post = (
        post
        if post is not None and post.status == "completed"
        else prior
    )
    final_returned = snapshots.get("final_returned")
    prior_ids = _candidate_ids(prior)
    recommendation_ids = _candidate_ids(recommendations)
    new_ids = recommendation_ids - prior_ids
    prior_gold = _gold_match_keys(eval_query, prior)
    recommendation_gold = _gold_match_keys(eval_query, recommendations)
    new_gold = recommendation_gold - prior_gold
    post_gold = _gold_match_keys(eval_query, effective_post)
    returned_ids = _candidate_ids(final_returned)
    returned_gold = _gold_match_keys(eval_query, final_returned)
    expansion = output.semantic_seed_expansion_output
    record = expansion.record if expansion is not None else None
    category_stats = _new_candidate_categories(
        new_ids,
        snapshots.get("post_semantic_seed_expansion_judged"),
    )
    judgement_filtered, top_k_lost = _module_gold_losses(
        new_gold,
        snapshots.get("post_semantic_seed_expansion_judged"),
        final_returned,
        eval_query,
    )
    prior_recall = (
        _candidate_recall(prior, eval_query.gold_papers)
        if prior is not None and prior.status == "completed"
        else None
    )
    post_recall = (
        _candidate_recall(effective_post, eval_query.gold_papers)
        if effective_post is not None and effective_post.status == "completed"
        else None
    )
    return {
        "enabled": output.search_plan.enable_semantic_seed_expansion,
        "triggered": bool(record is not None and record.seeds),
        "selected_seed_count": len(record.seeds) if record is not None else 0,
        "eligible_seed_count": len(record.seeds) if record is not None else 0,
        "seed_with_supported_identifier_count": (
            len(record.seeds) if record is not None else 0
        ),
        "seed_without_supported_identifier_count": 0,
        "resolution_attempted_case_count": int(
            bool(record is not None and record.resolution_request_identifier_count)
        ),
        "resolution_candidate_count": (
            record.resolution_candidate_count if record is not None else 0
        ),
        "resolution_request_identifier_count": (
            record.resolution_request_identifier_count if record is not None else 0
        ),
        "resolved_candidate_count": (
            record.resolved_candidate_count if record is not None else 0
        ),
        "resolution_conflict_count": (
            record.resolution_conflict_count if record is not None else 0
        ),
        "resolution_missing_count": (
            record.resolution_missing_count if record is not None else 0
        ),
        "direct_seed_count": record.direct_seed_count if record is not None else 0,
        "resolved_seed_count": record.resolved_seed_count if record is not None else 0,
        "resolution_status": (
            record.resolution_status if record is not None else "not_needed"
        ),
        "resolution_snapshot_key": (
            record.resolution_snapshot_key if record is not None else None
        ),
        "resolutions": (
            [item.model_dump(mode="json") for item in record.resolutions]
            if record is not None
            else []
        ),
        "reference_request_count": (
            expansion.diagnostics.request_count if expansion is not None else 0
        ),
        "recorded_reference_request_count": (
            expansion.recorded_diagnostics.request_count
            if expansion is not None
            else 0
        ),
        "recorded_reference_latency_seconds": (
            expansion.recorded_latency_seconds if expansion is not None else 0.0
        ),
        "raw_reference_count": (
            record.raw_recommendation_count if record is not None else 0
        ),
        "unique_reference_count": len(recommendation_ids),
        "new_unique_reference_count": len(new_ids),
        "reference_gold_hit_count": _raw_gold_hit_count(
            eval_query,
            recommendations,
        ),
        "unique_reference_gold_hit_count": len(recommendation_gold),
        "new_unique_reference_gold_count": len(new_gold),
        "initial_gold_lost_after_expansion_count": len(prior_gold - post_gold),
        "reference_candidates_returned_count": len(new_ids & returned_ids),
        "reference_gold_returned_count": len(new_gold & returned_gold),
        "gold_found_but_filtered_count": len(new_gold - returned_gold),
        "gold_filtered_by_judgement_count": judgement_filtered,
        "gold_lost_by_top_k_count": top_k_lost,
        "candidate_recall_before": prior_recall,
        "candidate_recall_after": post_recall,
        "candidate_recall_gain": (
            post_recall - prior_recall
            if post_recall is not None and prior_recall is not None
            else None
        ),
        "new_candidate_categories": category_stats,
        "seeds": (
            [seed.model_dump(mode="json") for seed in record.seeds]
            if record is not None
            else []
        ),
        "snapshot_key": record.snapshot_key if record is not None else None,
        "status": record.status if record is not None else "disabled",
        "skipped_reasons": _stable_reasons(
            [record.skip_reason] if record is not None and record.skip_reason else []
        ),
    }


def _stage_cost_diagnostics(
    output: SearchServiceOutput,
    snapshots: dict[str, StageCandidateSnapshot],
    *,
    query_evolution: dict[str, Any],
    refchain: dict[str, Any],
    semantic_seed_expansion: dict[str, Any],
) -> dict[str, Any]:
    initial_calls = sum(
        call.request_count
        for call in snapshots.get(
            "initial_retrieval",
            StageCandidateSnapshot(stage="initial_retrieval", status="skipped"),
        ).retrieval_calls
    )
    evolution_calls = sum(
        call.request_count
        for call in snapshots.get(
            "query_evolution_retrieval",
            StageCandidateSnapshot(
                stage="query_evolution_retrieval",
                status="skipped",
            ),
        ).retrieval_calls
    )
    initial_recorded_calls = sum(
        call.recorded_request_count
        for call in snapshots.get(
            "initial_retrieval",
            StageCandidateSnapshot(stage="initial_retrieval", status="skipped"),
        ).retrieval_calls
    )
    evolution_snapshot = snapshots.get(
        "query_evolution_retrieval",
        StageCandidateSnapshot(
            stage="query_evolution_retrieval",
            status="skipped",
        ),
    )
    evolution_recorded_calls = sum(
        call.recorded_request_count for call in evolution_snapshot.retrieval_calls
    )
    evolution_recorded_latency = sum(
        call.recorded_latency_seconds for call in evolution_snapshot.retrieval_calls
    )
    reference_calls = (
        output.refchain_output.diagnostics.request_count
        if output.refchain_output
        else 0
    )
    recorded_reference_calls = (
        output.refchain_output.recorded_diagnostics.request_count
        if output.refchain_output
        else 0
    )
    recorded_reference_latency = (
        output.refchain_output.recorded_latency_seconds
        if output.refchain_output
        else 0.0
    )
    semantic_output = output.semantic_seed_expansion_output
    semantic_calls = (
        semantic_output.diagnostics.request_count if semantic_output else 0
    )
    semantic_recorded_calls = (
        semantic_output.recorded_diagnostics.request_count
        if semantic_output
        else 0
    )
    semantic_recorded_latency = (
        semantic_output.recorded_latency_seconds if semantic_output else 0.0
    )
    qe_latency = sum(
        output.stage_latencies.get(name, 0.0)
        for name in (
            "query_evolution",
            "query_evolution_retrieval",
            "query_evolution_judgement",
            "query_evolution_reranking",
        )
    )
    refchain_latency = sum(
        output.stage_latencies.get(name, 0.0)
        for name in (
            "refchain",
            "refchain_judgement",
            "refchain_reranking",
        )
    )
    semantic_latency = output.stage_latencies.get(
        "semantic_seed_expansion",
        0.0,
    )
    return {
        "initial_search_api_calls": initial_calls,
        "query_evolution_api_calls": evolution_calls,
        "refchain_api_calls": reference_calls,
        "semantic_seed_expansion_api_calls": semantic_calls,
        "recorded_initial_search_api_calls": initial_recorded_calls,
        "recorded_query_evolution_api_calls": evolution_recorded_calls,
        "recorded_refchain_api_calls": recorded_reference_calls,
        "recorded_semantic_seed_expansion_api_calls": semantic_recorded_calls,
        "recorded_query_evolution_latency_seconds": evolution_recorded_latency,
        "recorded_refchain_latency_seconds": recorded_reference_latency,
        "recorded_semantic_seed_expansion_latency_seconds": (
            semantic_recorded_latency
        ),
        "retry_count": (
            output.search_diagnostics.retry_count
            + output.reference_diagnostics.retry_count
        ),
        "cache_hit_count": (
            output.search_diagnostics.cache_hit_count
            + output.reference_diagnostics.cache_hit_count
        ),
        "latency_seconds": output.latency_seconds,
        "query_evolution_latency_seconds": qe_latency,
        "refchain_latency_seconds": refchain_latency,
        "semantic_seed_expansion_latency_seconds": semantic_latency,
        "judgement_latency_seconds": output.stage_latencies.get("judgement", 0.0),
        "reranking_latency_seconds": output.stage_latencies.get("reranking", 0.0),
        "query_evolution": _module_marginal_costs(
            evolution_recorded_calls or evolution_calls,
            int(query_evolution["evolved_new_unique_candidate_count"]),
            int(query_evolution["evolved_new_unique_gold_count"]),
            float(query_evolution["candidate_recall_gain"]),
            evolution_recorded_latency or qe_latency,
        ),
        "refchain": _module_marginal_costs(
            recorded_reference_calls or reference_calls,
            int(refchain["new_unique_reference_count"]),
            int(refchain["new_unique_reference_gold_count"]),
            float(refchain["candidate_recall_gain"]),
            recorded_reference_latency or refchain_latency,
        ),
        "semantic_seed_expansion": _module_marginal_costs(
            semantic_recorded_calls or semantic_calls,
            int(semantic_seed_expansion["new_unique_reference_count"]),
            int(semantic_seed_expansion["new_unique_reference_gold_count"]),
            float(semantic_seed_expansion["candidate_recall_gain"] or 0.0),
            semantic_recorded_latency or semantic_latency,
        ),
    }


def _module_marginal_costs(
    api_calls: int,
    new_candidates: int,
    new_gold: int,
    recall_gain: float,
    latency_seconds: float,
) -> dict[str, Any]:
    return {
        "api_calls": api_calls,
        "latency_seconds": latency_seconds,
        "api_per_new_unique_candidate": (
            api_calls / new_candidates if new_candidates else None
        ),
        "api_per_new_unique_gold": api_calls / new_gold if new_gold else None,
        "api_per_0_01_recall_gain": (
            api_calls / (recall_gain / 0.01) if recall_gain > 0 else None
        ),
    }


def classify_module_outcome(
    diagnostics: dict[str, Any],
    costs: dict[str, Any],
    *,
    case_count: int,
) -> list[str]:
    labels: list[str] = []
    if not diagnostics.get("enabled"):
        labels.append("no_action_generated")
    elif int(diagnostics.get("selected_seed_count") or 0) == 0:
        labels.append("no_seed")
    else:
        action_count = int(
            diagnostics.get("executed_query_count")
            or diagnostics.get("reference_request_count")
            or diagnostics.get("raw_reference_count")
            or 0
        )
        new_candidates = int(
            diagnostics.get("evolved_new_unique_candidate_count")
            or diagnostics.get("new_unique_reference_count")
            or 0
        )
        new_gold = int(
            diagnostics.get("evolved_new_unique_gold_count")
            or diagnostics.get("new_unique_reference_gold_count")
            or 0
        )
        recall_gain = float(diagnostics.get("candidate_recall_gain") or 0.0)
        skipped = set(diagnostics.get("skipped_reasons") or [])
        if action_count == 0:
            labels.append("no_action_generated")
        if skipped.intersection({"source_cooldown", "source_failure"}) and not new_candidates:
            labels.append("source_failure_dominated")
        raw_candidates = int(
            diagnostics.get("evolved_raw_candidate_count")
            or diagnostics.get("raw_reference_count")
            or 0
        )
        if raw_candidates and not new_candidates:
            labels.append("mostly_duplicate_candidates")
        elif new_candidates and not new_gold:
            labels.append("new_candidates_but_no_gold")
        if new_gold and int(diagnostics.get("gold_found_but_filtered_count") or 0):
            labels.append("gold_found_but_filtered")
        if recall_gain > 0:
            api_per_gain = costs.get("api_per_0_01_recall_gain")
            if api_per_gain is not None and float(api_per_gain) > 5.0:
                labels.append("positive_recall_gain_high_cost")
            else:
                labels.append("effective")
        elif action_count and not labels:
            labels.append("no_measurable_gain")
    if case_count < 30:
        labels.append("insufficient_sample")
    return _stable_reasons(labels)


def _evolved_query_details(
    generated: list[Any],
    calls: list[Any],
    snapshot: StageCandidateSnapshot | None,
    judged: StageCandidateSnapshot | None,
    *,
    eval_query: EvalQuery,
    query_analysis: Any,
    initial_ids: set[str],
    stage_skip: str | None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for evolved in generated:
        sources = list(evolved.source_hints) or ["unknown"]
        for source in sources:
            matching_calls = [
                call
                for call in calls
                if call.origin_subquery == evolved.query and call.source == source
            ]
            executed = any(call.logical_call_executed for call in matching_calls)
            returned = sum(call.returned_count for call in matching_calls)
            candidate_ids = _query_candidate_ids(snapshot, evolved.query, source)
            judged_candidates = _query_candidates(judged, evolved.query, source)
            categories = Counter(
                candidate.category or "unjudged"
                for candidate in judged_candidates
            )
            skip_reason = _query_execution_skip_reason(
                matching_calls,
                executed=executed,
                returned_count=returned,
                stage_skip=stage_skip,
            )
            scores = [
                float(candidate.judgement_score)
                for candidate in judged_candidates
                if candidate.judgement_score is not None
            ]
            raw_candidates = _query_candidates(snapshot, evolved.query, source)
            raw_gold_hits = sum(
                _candidate_matches(candidate, gold)
                for candidate in raw_candidates
                for gold in eval_query.gold_papers
            )
            unique_gold_hits = {
                index
                for index, gold in enumerate(eval_query.gold_papers)
                if any(
                    _candidate_matches(candidate, gold)
                    for candidate in raw_candidates
                )
            }
            duplicate_ratio = (
                max(0, returned - len(candidate_ids)) / returned
                if returned
                else None
            )
            ineffective_reasons = _evolved_query_ineffective_reasons(
                evolved,
                query_analysis,
                categories,
                returned_count=returned,
                duplicate_ratio=duplicate_ratio,
                unique_gold_hit_count=len(unique_gold_hits),
                skip_reason=skip_reason,
            )
            details.append(
                {
                    "query": evolved.query,
                    "seed_titles": list(evolved.seed_paper_titles),
                    "generation_policy": evolved.generation_policy,
                    "gap_dimensions": list(evolved.gap_dimensions),
                    "source": source,
                    "executed": executed,
                    "skip_reason": skip_reason,
                    "request_count": sum(call.request_count for call in matching_calls),
                    "recorded_request_count": sum(
                        call.recorded_request_count for call in matching_calls
                    ),
                    "recorded_retry_count": sum(
                        call.recorded_retry_count for call in matching_calls
                    ),
                    "recorded_error_count": sum(
                        call.recorded_error_count for call in matching_calls
                    ),
                    "recorded_latency_seconds": sum(
                        call.recorded_latency_seconds for call in matching_calls
                    ),
                    "returned_count": returned,
                    "unique_candidate_count": len(candidate_ids),
                    "duplicate_ratio": duplicate_ratio,
                    "new_unique_candidate_count": len(candidate_ids - initial_ids),
                    "judgement_categories": dict(sorted(categories.items())),
                    "average_judgement_score": (
                        sum(scores) / len(scores) if scores else None
                    ),
                    "highly_relevant_count": categories["highly_relevant"],
                    "partially_relevant_count": categories[
                        "partially_relevant"
                    ],
                    "weak_or_irrelevant_count": sum(
                        categories[category]
                        for category in FALSE_NEGATIVE_CATEGORIES
                    ),
                    "post_run_gold_hit_count": raw_gold_hits,
                    "post_run_unique_gold_hit_count": len(unique_gold_hits),
                    "ineffective_reasons": ineffective_reasons,
                }
            )
    return details


def _evolved_query_ineffective_reasons(
    evolved: Any,
    query_analysis: Any,
    categories: Counter[str],
    *,
    returned_count: int,
    duplicate_ratio: float | None,
    unique_gold_hit_count: int,
    skip_reason: str | None,
) -> list[str]:
    reasons: list[str] = []
    skip_mapping = {
        "duplicate_query": "duplicates_existing_query",
        "budget_stop": "budget_stop",
        "source_cooldown": "source_failure",
        "source_failure": "source_failure",
    }
    if skip_reason in skip_mapping:
        reasons.append(skip_mapping[skip_reason])
    original_terms = _diagnostic_query_terms(query_analysis.original_query)
    query_terms = _diagnostic_query_terms(evolved.query)
    if original_terms and len(original_terms & query_terms) / len(original_terms) < 0.4:
        reasons.append("missing_original_core_terms")
    constraints = query_analysis.constraints
    required = (
        list(constraints.must_include_terms)
        if "must_include_terms" in constraints.explicit_fields
        else []
    )
    if any(
        normalize_title(term) not in normalize_title(evolved.query)
        for term in required
    ):
        reasons.append("missing_required_constraint")
    if any(
        _token_overlap(evolved.query, title) >= 0.8
        for title in evolved.seed_paper_titles
        if title.strip()
    ):
        reasons.append("dominated_by_seed_title")
    if evolved.generation_policy == "seed_expansion" and not evolved.gap_dimensions:
        suffixes = {"survey", "review", "benchmark", "evaluation", "comparison"}
        if query_terms & suffixes and len(query_terms - suffixes) <= 2:
            reasons.append("generic_suffix_only")
    invalid = sum(categories[category] for category in FALSE_NEGATIVE_CATEGORIES)
    total = sum(categories.values())
    if total and invalid / total >= 0.5:
        reasons.extend(["over_broad_query", "candidates_mostly_irrelevant"])
    if returned_count == 0 and skip_reason is None:
        reasons.append("over_restrictive_query")
    if duplicate_ratio is not None and duplicate_ratio >= 0.5:
        reasons.append("candidates_mostly_duplicate")
    if returned_count and unique_gold_hit_count == 0:
        reasons.append("no_unique_gold")
    if "low_information_retention" in (skip_reason or ""):
        reasons.append("low_information_retention")
    if not reasons and (skip_reason is not None or unique_gold_hit_count == 0):
        reasons.append("unknown")
    return _stable_reasons(reasons)


def _diagnostic_query_terms(value: str) -> set[str]:
    return {
        term
        for term in normalize_title(value).split()
        if len(term) > 1 and term not in DIAGNOSTIC_QUERY_BOILERPLATE
    }


def _token_overlap(left: str, right: str) -> float:
    left_terms = set(normalize_title(left).split())
    right_terms = set(normalize_title(right).split())
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _refchain_seed_details(record: Any, prior: StageCandidateSnapshot | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    prior_tokens = set().union(
        *(paper_identifier_set(candidate) for candidate in (prior.candidates if prior else []))
    ) if prior and prior.candidates else set()
    seen: set[str] = set()
    edges_by_seed: dict[str, list[str]] = {}
    for edge in record.reference_edges:
        edges_by_seed.setdefault(edge.seed_paper_id, []).append(edge.reference_paper_id)
    details: list[dict[str, Any]] = []
    for diagnostic in record.seed_diagnostics:
        references = edges_by_seed.get(diagnostic.seed_id or "", [])
        unique = {
            reference
            for reference in references
            if reference not in prior_tokens and reference not in seen
        }
        seen.update(references)
        skip_reason = diagnostic.skip_reason
        if diagnostic.references_returned and not unique:
            skip_reason = "all_references_duplicate"
        details.append(
            {
                "seed_id": diagnostic.seed_id,
                "seed_rank": diagnostic.seed_rank,
                "seed_category": diagnostic.seed_category,
                "seed_score": diagnostic.seed_score,
                "identifier_type": diagnostic.identifier_type,
                "reference_key": diagnostic.snapshot_key,
                "request_count": diagnostic.request_count,
                "recorded_request_count": diagnostic.recorded_request_count,
                "recorded_retry_count": diagnostic.recorded_retry_count,
                "recorded_error_count": diagnostic.recorded_error_count,
                "recorded_latency_seconds": diagnostic.recorded_latency_seconds,
                "references_returned": diagnostic.references_returned,
                "new_unique_references": len(unique),
                "skip_reason": skip_reason,
            }
        )
    return details


def _aggregate_module_diagnostics(
    cases: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    modules = [case.get(name) or {} for case in cases]
    count_keys = (
        (
            "eligible_seed_count",
            "selected_seed_count",
            "generated_query_count",
            "executed_query_count",
            "duplicate_query_count",
            "budget_skipped_query_count",
            "source_skipped_query_count",
            "evolved_raw_candidate_count",
            "evolved_unique_candidate_count",
            "evolved_new_unique_candidate_count",
            "evolved_gold_hit_count",
            "evolved_unique_gold_hit_count",
            "evolved_new_unique_gold_count",
            "evolved_candidates_returned_count",
            "evolved_gold_returned_count",
            "gold_found_but_filtered_count",
            "gold_filtered_by_judgement_count",
            "gold_lost_by_top_k_count",
        )
        if name == "query_evolution"
        else (
            "eligible_seed_count",
            "selected_seed_count",
            "seed_with_supported_identifier_count",
            "seed_without_supported_identifier_count",
            "resolution_attempted_case_count",
            "resolution_candidate_count",
            "resolution_request_identifier_count",
            "resolved_candidate_count",
            "resolution_conflict_count",
            "resolution_missing_count",
            "direct_seed_count",
            "resolved_seed_count",
            "reference_request_count",
            "recorded_reference_request_count",
            "raw_reference_count",
            "unique_reference_count",
            "new_unique_reference_count",
            "reference_gold_hit_count",
            "unique_reference_gold_hit_count",
            "new_unique_reference_gold_count",
            "initial_gold_lost_after_expansion_count",
            "reference_candidates_returned_count",
            "reference_gold_returned_count",
            "gold_found_but_filtered_count",
            "gold_filtered_by_judgement_count",
            "gold_lost_by_top_k_count",
        )
    )
    categories = Counter()
    skip_reasons = Counter()
    policies = Counter()
    quality_gate_counts = Counter()
    quality_gate_reasons = Counter()
    resolution_statuses = Counter()
    for module in modules:
        categories.update((module.get("new_candidate_categories") or {}).get("counts") or {})
        skip_reasons.update(module.get("skipped_reasons") or [])
        if module.get("policy"):
            policies[str(module["policy"])] += 1
        if module.get("resolution_status"):
            resolution_statuses[str(module["resolution_status"])] += 1
        quality_gate = module.get("quality_gate") or {}
        for key in (
            "raw_candidate_count",
            "unique_candidate_count",
            "duplicate_candidate_count",
            "duplicate_with_initial_count",
            "accepted_candidate_count",
            "filtered_candidate_count",
        ):
            quality_gate_counts[key] += int(quality_gate.get(key) or 0)
        quality_gate_reasons.update(
            quality_gate.get("filtered_reason_counts") or {}
        )
    category_total = sum(categories.values())
    result: dict[str, Any] = {
        "enabled": any(module.get("enabled") for module in modules),
        "enabled_case_count": sum(bool(module.get("enabled")) for module in modules),
        "triggered_case_count": sum(bool(module.get("triggered")) for module in modules),
        "policies": dict(sorted(policies.items())),
        "resolution_statuses": dict(sorted(resolution_statuses.items())),
        **{
            key: sum(int(module.get(key) or 0) for module in modules)
            for key in count_keys
        },
        "candidate_recall_gain": _average_optional(
            [module.get("candidate_recall_gain") for module in modules]
        ) or 0.0,
        "candidate_recall_before": _average_optional(
            [module.get("candidate_recall_before") for module in modules]
        ),
        "candidate_recall_after": _average_optional(
            [module.get("candidate_recall_after") for module in modules]
        ),
        "new_candidate_categories": {
            "counts": dict(sorted(categories.items())),
            "ratios": {
                category: count / category_total
                for category, count in sorted(categories.items())
            },
        },
        "skipped_reasons": dict(sorted(skip_reasons.items())),
        "quality_gate": {
            **dict(quality_gate_counts),
            "filtered_reason_counts": dict(sorted(quality_gate_reasons.items())),
            "invalid_candidate_share": (
                quality_gate_counts["filtered_candidate_count"]
                / quality_gate_counts["raw_candidate_count"]
                if quality_gate_counts["raw_candidate_count"]
                else None
            ),
        },
    }
    return result


def _aggregate_stage_costs(cases: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [case.get("stage_costs") or {} for case in cases]
    case_count = len(costs)
    result = {
        "initial_search_api_calls": sum(int(item.get("initial_search_api_calls") or 0) for item in costs),
        "query_evolution_api_calls": sum(int(item.get("query_evolution_api_calls") or 0) for item in costs),
        "refchain_api_calls": sum(int(item.get("refchain_api_calls") or 0) for item in costs),
        "semantic_seed_expansion_api_calls": sum(
            int(item.get("semantic_seed_expansion_api_calls") or 0)
            for item in costs
        ),
        "recorded_initial_search_api_calls": sum(
            int(item.get("recorded_initial_search_api_calls") or 0)
            for item in costs
        ),
        "recorded_query_evolution_api_calls": sum(
            int(item.get("recorded_query_evolution_api_calls") or 0)
            for item in costs
        ),
        "recorded_refchain_api_calls": sum(
            int(item.get("recorded_refchain_api_calls") or 0)
            for item in costs
        ),
        "recorded_semantic_seed_expansion_api_calls": sum(
            int(item.get("recorded_semantic_seed_expansion_api_calls") or 0)
            for item in costs
        ),
        "retry_count": sum(int(item.get("retry_count") or 0) for item in costs),
        "cache_hit_count": sum(int(item.get("cache_hit_count") or 0) for item in costs),
        "average_latency_seconds": _average_optional([item.get("latency_seconds") for item in costs]) or 0.0,
        "average_query_evolution_latency_seconds": _average_optional([item.get("query_evolution_latency_seconds") for item in costs]) or 0.0,
        "average_refchain_latency_seconds": _average_optional([item.get("refchain_latency_seconds") for item in costs]) or 0.0,
        "average_semantic_seed_expansion_latency_seconds": _average_optional(
            [item.get("semantic_seed_expansion_latency_seconds") for item in costs]
        ) or 0.0,
        "average_recorded_query_evolution_latency_seconds": _average_optional(
            [item.get("recorded_query_evolution_latency_seconds") for item in costs]
        ) or 0.0,
        "average_recorded_refchain_latency_seconds": _average_optional(
            [item.get("recorded_refchain_latency_seconds") for item in costs]
        ) or 0.0,
        "average_recorded_semantic_seed_expansion_latency_seconds": _average_optional(
            [
                item.get("recorded_semantic_seed_expansion_latency_seconds")
                for item in costs
            ]
        ) or 0.0,
        "average_judgement_latency_seconds": _average_optional([item.get("judgement_latency_seconds") for item in costs]) or 0.0,
        "average_reranking_latency_seconds": _average_optional([item.get("reranking_latency_seconds") for item in costs]) or 0.0,
    }
    qe_new = sum(int((case.get("query_evolution") or {}).get("evolved_new_unique_candidate_count") or 0) for case in cases)
    qe_gold = sum(int((case.get("query_evolution") or {}).get("evolved_new_unique_gold_count") or 0) for case in cases)
    qe_gain = _average_optional([(case.get("query_evolution") or {}).get("candidate_recall_gain") for case in cases]) or 0.0
    rc_new = sum(int((case.get("refchain") or {}).get("new_unique_reference_count") or 0) for case in cases)
    rc_gold = sum(int((case.get("refchain") or {}).get("new_unique_reference_gold_count") or 0) for case in cases)
    rc_gain = _average_optional([(case.get("refchain") or {}).get("candidate_recall_gain") for case in cases]) or 0.0
    semantic_new = sum(
        int(
            (case.get("semantic_seed_expansion") or {}).get(
                "new_unique_reference_count"
            )
            or 0
        )
        for case in cases
    )
    semantic_gold = sum(
        int(
            (case.get("semantic_seed_expansion") or {}).get(
                "new_unique_reference_gold_count"
            )
            or 0
        )
        for case in cases
    )
    semantic_gain = _average_optional(
        [
            (case.get("semantic_seed_expansion") or {}).get(
                "candidate_recall_gain"
            )
            for case in cases
        ]
    ) or 0.0
    result["query_evolution"] = _module_marginal_costs(
        result["recorded_query_evolution_api_calls"]
        or result["query_evolution_api_calls"],
        qe_new,
        qe_gold,
        qe_gain,
        (
            result["average_recorded_query_evolution_latency_seconds"]
            or result["average_query_evolution_latency_seconds"]
        )
        * case_count,
    )
    result["refchain"] = _module_marginal_costs(
        result["recorded_refchain_api_calls"] or result["refchain_api_calls"],
        rc_new,
        rc_gold,
        rc_gain,
        (
            result["average_recorded_refchain_latency_seconds"]
            or result["average_refchain_latency_seconds"]
        )
        * case_count,
    )
    result["semantic_seed_expansion"] = _module_marginal_costs(
        result["recorded_semantic_seed_expansion_api_calls"]
        or result["semantic_seed_expansion_api_calls"],
        semantic_new,
        semantic_gold,
        semantic_gain,
        (
            result["average_recorded_semantic_seed_expansion_latency_seconds"]
            or result["average_semantic_seed_expansion_latency_seconds"]
        )
        * case_count,
    )
    return result


def _candidate_ids(snapshot: StageCandidateSnapshot | None) -> set[str]:
    if snapshot is None or snapshot.status != "completed":
        return set()
    return {
        identifier
        for candidate in snapshot.candidates
        if (identifier := canonical_paper_id(candidate))
    }


def _query_candidate_ids(
    snapshot: StageCandidateSnapshot | None,
    query: str,
    source: str,
) -> set[str]:
    if snapshot is None:
        return set()
    return {
        identifier
        for candidate in snapshot.candidates
        if any(
            provenance.origin_stage == "query_evolution_retrieval"
            and provenance.origin_subquery == query
            and provenance.source == source
            for provenance in candidate.provenance
        )
        if (identifier := canonical_paper_id(candidate))
    }


def _query_candidates(
    snapshot: StageCandidateSnapshot | None,
    query: str,
    source: str,
) -> list[Any]:
    if snapshot is None:
        return []
    return [
        candidate
        for candidate in snapshot.candidates
        if any(
            provenance.origin_stage == "query_evolution_retrieval"
            and provenance.origin_subquery == query
            and provenance.source == source
            for provenance in candidate.provenance
        )
    ]


def _gold_match_keys(
    eval_query: EvalQuery,
    snapshot: StageCandidateSnapshot | None,
) -> set[str]:
    if snapshot is None or snapshot.status != "completed":
        return set()
    return {
        f"{eval_query.query_id}:{index}"
        for index, gold in enumerate(eval_query.gold_papers)
        if any(_candidate_matches(candidate, gold) for candidate in snapshot.candidates)
    }


def _raw_gold_hit_count(
    eval_query: EvalQuery,
    snapshot: StageCandidateSnapshot | None,
) -> int:
    if snapshot is None or snapshot.status != "completed":
        return 0
    return sum(
        _candidate_matches(candidate, gold)
        for candidate in snapshot.candidates
        for gold in eval_query.gold_papers
    )


def _candidate_recall_value(
    snapshot: StageCandidateSnapshot | None,
    gold: list[EvalGoldPaper],
) -> float:
    if snapshot is None or snapshot.status != "completed":
        return 0.0
    value = _candidate_recall(snapshot, gold)
    return float(value or 0.0)


def _new_candidate_categories(
    new_ids: set[str],
    judged: StageCandidateSnapshot | None,
) -> dict[str, Any]:
    categories = Counter(
        candidate.category or "unjudged"
        for candidate in (judged.candidates if judged else [])
        if canonical_paper_id(candidate) in new_ids
    )
    total = sum(categories.values())
    return {
        "counts": dict(sorted(categories.items())),
        "ratios": {
            category: count / total
            for category, count in sorted(categories.items())
        },
    }


def _module_gold_losses(
    new_gold: set[str],
    judged: StageCandidateSnapshot | None,
    final_returned: StageCandidateSnapshot | None,
    eval_query: EvalQuery,
) -> tuple[int, int]:
    retained_after_judgement = _gold_match_keys_for_categories(
        eval_query,
        judged,
        RETURN_CATEGORIES,
    )
    returned = _gold_match_keys(eval_query, final_returned)
    judgement_filtered = new_gold - retained_after_judgement
    top_k_lost = (new_gold & retained_after_judgement) - returned
    return len(judgement_filtered), len(top_k_lost)


def _gold_match_keys_for_categories(
    eval_query: EvalQuery,
    snapshot: StageCandidateSnapshot | None,
    categories: set[str],
) -> set[str]:
    if snapshot is None or snapshot.status != "completed":
        return set()
    return {
        f"{eval_query.query_id}:{index}"
        for index, gold in enumerate(eval_query.gold_papers)
        if any(
            candidate.category in categories and _candidate_matches(candidate, gold)
            for candidate in snapshot.candidates
        )
    }


def _eligible_query_evolution_seed_count(
    snapshot: StageCandidateSnapshot | None,
) -> int:
    if snapshot is None:
        return 0
    return sum(
        candidate.category == "highly_relevant"
        or (
            candidate.category == "partially_relevant"
            and float(candidate.judgement_score or 0.0) >= 0.45
        )
        for candidate in snapshot.candidates
    )


def _eligible_refchain_seed_count(snapshot: StageCandidateSnapshot | None) -> int:
    if snapshot is None:
        return 0
    return sum(
        candidate.category == "highly_relevant"
        or (
            candidate.category == "partially_relevant"
            and float(candidate.final_score or 0.0) >= 0.45
        )
        for candidate in snapshot.candidates
    )


def _supported_seed_identifier(paper: Any) -> bool:
    return bool(paper.identifiers.openalex_id or paper.identifiers.doi)


def _query_execution_skip_reason(
    calls: list[Any],
    *,
    executed: bool,
    returned_count: int,
    stage_skip: str | None,
) -> str | None:
    if not calls:
        return "budget_stop" if _is_budget_reason(stage_skip) else "duplicate_query"
    reasons = [call.source_skipped_reason or "" for call in calls]
    errors = [call.error_count for call in calls]
    if any("cooldown" in reason or "rate_limit" in reason for reason in reasons):
        return "source_cooldown"
    if any(errors) or any("failure" in reason for reason in reasons):
        return "source_failure"
    if not executed:
        return "duplicate_query"
    if returned_count == 0:
        return "empty_result"
    return None


def _normalize_query_evolution_skip(reason: str) -> str:
    normalized = reason.casefold()
    if "no_relevant_seed" in normalized:
        return "no_eligible_seed"
    if "no_new" in normalized:
        return "no_new_query"
    if "budget" in normalized or "max_" in normalized:
        return "budget_stop"
    if "cooldown" in normalized or "429" in normalized:
        return "source_cooldown"
    if "failed" in normalized or "error" in normalized:
        return "source_failure"
    return reason


def _normalize_refchain_skip(reason: str) -> str:
    normalized = reason.casefold()
    if "no_eligible_seed" in normalized:
        return "no_eligible_seed"
    if "missing_supported_identifier" in normalized:
        return "unsupported_identifier"
    if "budget" in normalized or "max_" in normalized:
        return "budget_stop"
    if "cooldown" in normalized or "429" in normalized:
        return "source_cooldown"
    if "failed" in normalized or "error" in normalized:
        return "source_failure"
    return reason


def _is_budget_reason(reason: str | None) -> bool:
    normalized = (reason or "").casefold()
    return "budget" in normalized or "max_" in normalized


def _stable_reasons(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if item and item not in result:
            result.append(item)
    return result


def _retrieval_query_diagnostics(output: SearchServiceOutput) -> dict[str, Any]:
    stats = list(output.source_stats)
    adapted = [item.adapted_query for item in stats if item.adapted_query]
    actual = [item for item in stats if item.diagnostics.request_count > 0]
    empty = [
        item
        for item in actual
        if item.returned_count == 0 and item.error_message is None
    ]
    errors: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    by_source: dict[str, dict[str, Any]] = {}
    for item in stats:
        source = item.source
        bucket = by_source.setdefault(
            source,
            {
                "logical_call_count": 0,
                "request_count": 0,
                "empty_result_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "strategies": {},
            },
        )
        bucket["logical_call_count"] += int(item.logical_call_executed)
        bucket["request_count"] += item.diagnostics.request_count
        bucket["empty_result_count"] += int(
            item.diagnostics.request_count > 0
            and item.returned_count == 0
            and item.error_message is None
        )
        bucket["error_count"] += item.diagnostics.error_count
        bucket["skipped_count"] += int(item.source_skipped_reason is not None)
        if item.adaptation_strategy:
            strategies = bucket["strategies"]
            strategies[item.adaptation_strategy] = (
                int(strategies.get(item.adaptation_strategy, 0)) + 1
            )
        if item.source_skipped_reason:
            skipped[item.source_skipped_reason] += 1
            continue
        message = (item.error_message or "").casefold()
        if "400" in message:
            errors["http_400"] += 1
        elif "429" in message:
            errors["http_429"] += 1
        elif "timed out" in message or "timeout" in message:
            errors["timeout"] += 1
        elif item.error_message:
            errors["other"] += 1
    return {
        "logical_call_count": sum(item.logical_call_executed for item in stats),
        "actual_request_count": sum(
            item.diagnostics.request_count for item in stats
        ),
        "unique_adapted_query_count": len(
            {_normalized_query(item) for item in adapted}
        ),
        "average_adapted_query_length": (
            sum(len(item) for item in adapted) / len(adapted) if adapted else 0.0
        ),
        "average_adapted_keyword_count": (
            sum(len(item.split()) for item in adapted) / len(adapted)
            if adapted
            else 0.0
        ),
        "empty_result_count": len(empty),
        "empty_result_rate": len(empty) / len(actual) if actual else 0.0,
        "errors": dict(sorted(errors.items())),
        "skipped": dict(sorted(skipped.items())),
        "adaptive": _adaptive_retrieval_diagnostics(stats),
        "by_source": by_source,
    }


def _adaptive_retrieval_diagnostics(stats: list[Any]) -> dict[str, Any]:
    decisions = [
        item
        for item in stats
        if item.adaptation_strategy == "compact_core"
        and item.compact_query_executed is not None
    ]
    executed = [item for item in decisions if item.compact_query_executed]
    safe_keys = {
        _paper_identity_key(paper)
        for item in stats
        if item.adaptation_strategy in {"safe_original", "fallback_original"}
        for paper in item.diagnostic_papers
    }
    compact_keys = {
        _paper_identity_key(paper)
        for item in executed
        for paper in item.diagnostic_papers
    }
    return {
        "compact_decision_count": len(decisions),
        "compact_executed_count": len(executed),
        "compact_execution_ratio": (
            len(executed) / len(decisions) if decisions else 0.0
        ),
        "compact_added_unique_candidate_count": len(compact_keys - safe_keys),
        "skip_reasons": dict(
            sorted(
                Counter(
                    item.compact_query_skipped_reason
                    for item in decisions
                    if item.compact_query_skipped_reason
                ).items()
            )
        ),
    }


def _paper_identity_key(paper: Any) -> str:
    identifiers = paper.identifiers
    for value in (
        identifiers.doi,
        identifiers.arxiv_id,
        identifiers.semantic_scholar_id,
        identifiers.openalex_id,
        identifiers.pubmed_id,
    ):
        if value:
            return str(value).casefold()
    return f"{paper.title.casefold()}::{paper.year}"


def _query_strategy_gold_contribution(
    eval_query: EvalQuery,
    snapshots: list[StageCandidateSnapshot],
    output: SearchServiceOutput,
) -> dict[str, Any]:
    purposes = {
        item.query: item.purpose for item in output.search_plan.subqueries
    }
    purpose_hits: dict[str, set[str]] = {}
    adaptation_hits: dict[str, set[str]] = {}
    compact_only_hits: set[str] = set()
    initial = next(
        (item for item in snapshots if item.stage == "initial_retrieval"),
        None,
    )
    if initial is None:
        return {"subquery_purposes": {}, "adaptation_strategies": {}}
    for candidate in initial.candidates:
        matched = {
            f"{eval_query.query_id}:{index}"
            for index, gold in enumerate(eval_query.gold_papers)
            if _candidate_matches(candidate, gold)
        }
        if not matched:
            continue
        strategies = {
            provenance.adaptation_strategy or "unadapted"
            for provenance in candidate.provenance
        }
        if "compact_core" in strategies and not strategies.intersection(
            {"safe_original", "fallback_original"}
        ):
            compact_only_hits.update(matched)
        for provenance in candidate.provenance:
            purpose = purposes.get(provenance.origin_subquery, "unknown")
            purpose_hits.setdefault(purpose, set()).update(matched)
            strategy = provenance.adaptation_strategy or "unadapted"
            adaptation_hits.setdefault(strategy, set()).update(matched)
    return {
        "subquery_purposes": {
            key: len(value) for key, value in sorted(purpose_hits.items())
        },
        "adaptation_strategies": {
            key: len(value) for key, value in sorted(adaptation_hits.items())
        },
        "compact_gold_increment": len(compact_only_hits),
    }


def _aggregate_retrieval_diagnostics(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = [case.get("retrieval_diagnostics", {}) for case in cases]
    logical = sum(int(item.get("logical_call_count") or 0) for item in diagnostics)
    actual = sum(int(item.get("actual_request_count") or 0) for item in diagnostics)
    empty = sum(int(item.get("empty_result_count") or 0) for item in diagnostics)
    errors = Counter()
    skipped = Counter()
    compact_decisions = 0
    compact_executed = 0
    compact_added = 0
    adaptive_skips: Counter[str] = Counter()
    for item in diagnostics:
        errors.update(item.get("errors") or {})
        skipped.update(item.get("skipped") or {})
        adaptive = item.get("adaptive") or {}
        compact_decisions += int(adaptive.get("compact_decision_count") or 0)
        compact_executed += int(adaptive.get("compact_executed_count") or 0)
        compact_added += int(
            adaptive.get("compact_added_unique_candidate_count") or 0
        )
        adaptive_skips.update(adaptive.get("skip_reasons") or {})
    return {
        "logical_call_count": logical,
        "actual_request_count": actual,
        "average_actual_requests_per_case": actual / len(cases) if cases else 0.0,
        "empty_result_count": empty,
        "empty_result_rate": empty / actual if actual else 0.0,
        "errors": dict(sorted(errors.items())),
        "skipped": dict(sorted(skipped.items())),
        "adaptive": {
            "compact_decision_count": compact_decisions,
            "compact_executed_count": compact_executed,
            "compact_execution_ratio": (
                compact_executed / compact_decisions if compact_decisions else 0.0
            ),
            "compact_added_unique_candidate_count": compact_added,
            "compact_average_added_unique_candidates": (
                compact_added / compact_executed if compact_executed else 0.0
            ),
            "skip_reasons": dict(sorted(adaptive_skips.items())),
        },
        "average_adapted_query_length": _average_optional(
            [item.get("average_adapted_query_length") for item in diagnostics]
        ),
        "average_adapted_keyword_count": _average_optional(
            [item.get("average_adapted_keyword_count") for item in diagnostics]
        ),
    }


def _aggregate_initial_query_planning(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    modules = [case.get("initial_query_planning") or {} for case in cases]
    policies: Counter[str] = Counter()
    ineffective: Counter[str] = Counter()
    facet_contribution: dict[str, Counter[str]] = {}
    raw = unique = gold = requests = recorded_requests = errors = 0
    latency = 0.0
    subquery_count = adapted_count = 0
    for module in modules:
        if module.get("policy"):
            policies[str(module["policy"])] += 1
        ineffective.update(module.get("ineffective_reasons") or {})
        raw += int(module.get("raw_candidate_count") or 0)
        unique += int(module.get("unique_candidate_count") or 0)
        gold += int(module.get("unique_gold_count") or 0)
        requests += int(module.get("request_count") or 0)
        recorded_requests += int(module.get("recorded_request_count") or 0)
        errors += int(module.get("source_error_count") or 0)
        latency += float(module.get("recorded_latency_seconds") or 0.0)
        subquery_count += int(module.get("subquery_count") or 0)
        adapted_count += int(module.get("adapted_query_count") or 0)
        for facet_type, contribution in (
            module.get("facet_contribution") or {}
        ).items():
            bucket = facet_contribution.setdefault(facet_type, Counter())
            bucket.update(contribution)
    case_count = len(modules)
    effective_requests = recorded_requests or requests
    planning_rows = [module.get("planning") or {} for module in modules]
    return {
        "case_count": case_count,
        "policies": dict(sorted(policies.items())),
        "subquery_count": subquery_count,
        "average_subquery_count": subquery_count / case_count if case_count else 0.0,
        "adapted_query_count": adapted_count,
        "average_adapted_query_count": adapted_count / case_count if case_count else 0.0,
        "raw_candidate_count": raw,
        "unique_candidate_count": unique,
        "duplicate_candidate_ratio": max(0, raw - unique) / raw if raw else None,
        "unique_gold_count": gold,
        "request_count": requests,
        "recorded_request_count": recorded_requests,
        "effective_request_count": effective_requests,
        "recorded_latency_seconds": latency,
        "source_error_count": errors,
        "source_error_rate": errors / effective_requests if effective_requests else 0.0,
        "identified_facet_count": sum(
            int(item.get("identified_facet_count") or 0)
            for item in planning_rows
        ),
        "selected_facet_count": sum(
            int(item.get("selected_facet_count") or 0)
            for item in planning_rows
        ),
        "explicit_facet_count": sum(
            int(item.get("explicit_facet_count") or 0)
            for item in planning_rows
        ),
        "duplicate_subquery_count": sum(
            int(item.get("duplicate_subquery_count") or 0)
            for item in planning_rows
        ),
        "skipped_by_budget_count": sum(
            int(item.get("skipped_by_budget_count") or 0)
            for item in planning_rows
        ),
        "facet_contribution": {
            facet_type: dict(sorted(values.items()))
            for facet_type, values in sorted(facet_contribution.items())
        },
        "ineffective_reasons": dict(sorted(ineffective.items())),
    }


def _aggregate_strategy_contribution(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    purposes: Counter[str] = Counter()
    adaptations: Counter[str] = Counter()
    compact_gold_increment = 0
    for case in cases:
        contribution = case.get("query_strategy_contribution") or {}
        purposes.update(contribution.get("subquery_purposes") or {})
        adaptations.update(contribution.get("adaptation_strategies") or {})
        compact_gold_increment += int(
            contribution.get("compact_gold_increment") or 0
        )
    return {
        "subquery_purposes": dict(sorted(purposes.items())),
        "adaptation_strategies": dict(sorted(adaptations.items())),
        "compact_gold_increment": compact_gold_increment,
    }


def _normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _aggregate_judgement(cases: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "retrieved_gold_count",
        "gold_judged_highly_relevant",
        "gold_judged_partially_relevant",
        "gold_judged_weakly_relevant",
        "gold_judged_irrelevant",
        "gold_judged_insufficient_evidence",
        "gold_false_negative_count",
    )
    totals = {
        key: sum(int(case.get("judgement", {}).get(key) or 0) for case in cases)
        for key in keys
    }
    retained = (
        totals["gold_judged_highly_relevant"]
        + totals["gold_judged_partially_relevant"]
    )
    denominator = totals["retrieved_gold_count"]
    return {
        **totals,
        "gold_retention_after_judgement": (
            retained / denominator if denominator else 0.0
        ),
        "gold_false_negative_rate": (
            totals["gold_false_negative_count"] / denominator
            if denominator
            else 0.0
        ),
    }


def _aggregate_reranking(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [
        int(rank)
        for case in cases
        for rank in case.get("reranking", {}).get("gold_ranks", [])
    ]
    eligible = sum(
        int(case.get("reranking", {}).get("eligible_gold_count") or 0)
        for case in cases
    )
    outside = sum(
        int(case.get("reranking", {}).get("gold_outside_top_20") or 0)
        for case in cases
    )
    return {
        "eligible_gold_count": eligible,
        "gold_in_top_5": sum(rank <= 5 for rank in ranks),
        "gold_in_top_10": sum(rank <= 10 for rank in ranks),
        "gold_in_top_20": sum(rank <= 20 for rank in ranks),
        "gold_outside_top_20": outside,
        "outside_top_20_rate": outside / eligible if eligible else 0.0,
        "average_gold_rank": statistics.mean(ranks) if ranks else None,
        "median_gold_rank": statistics.median(ranks) if ranks else None,
    }


def _aggregate_sources(
    cases: list[dict[str, Any]],
    gold_count: int,
) -> dict[str, Any]:
    sources: dict[str, dict[str, float | int]] = {}
    overlaps: dict[str, dict[str, int]] = {}
    for case in cases:
        contribution = case.get("source_contribution", {})
        for source, values in contribution.get("sources", {}).items():
            target = sources.setdefault(source, _empty_source_stats())
            for key in target:
                target[key] += values.get(key, 0)
        for pair, values in contribution.get("overlap", {}).items():
            target_overlap = overlaps.setdefault(
                pair,
                {"candidate_count": 0, "gold_hit_count": 0},
            )
            target_overlap["candidate_count"] += int(values["candidate_count"])
            target_overlap["gold_hit_count"] += int(values["gold_hit_count"])
    for values in sources.values():
        values["gold_recall_contribution"] = (
            values["unique_gold_hit_count"] / gold_count if gold_count else 0.0
        )
    requests = sum(int(item["request_count"]) for item in sources.values())
    errors = sum(int(item["error_count"]) for item in sources.values())
    return {
        "sources": sources,
        "overlap": overlaps,
        "source_error_rate": errors / requests if requests else 0.0,
    }


def _candidate_recall(
    snapshot: StageCandidateSnapshot,
    gold_papers: list[EvalGoldPaper],
) -> float | None:
    if snapshot.status != "completed":
        return None
    gold_count = evaluable_gold_count(gold_papers)
    if not gold_count:
        return None
    return len(matched_paper_ids(snapshot.candidates, gold_papers)) / gold_count


def _matches(
    candidates: list[DiagnosticCandidate],
    gold: EvalGoldPaper,
) -> list[DiagnosticCandidate]:
    return [candidate for candidate in candidates if _candidate_matches(candidate, gold)]


def _candidate_matches(candidate: Any, gold: EvalGoldPaper) -> bool:
    return bool(matched_paper_ids([candidate], [gold]))


def _matching_candidate(
    candidates: list[DiagnosticCandidate],
    ranked: Any,
) -> DiagnosticCandidate | None:
    for candidate in candidates:
        if matched_paper_ids([candidate], [ranked.paper]):
            return candidate
    return None


def _snapshot_by_name(
    snapshots: list[StageCandidateSnapshot],
    name: str,
) -> StageCandidateSnapshot | None:
    return next((item for item in snapshots if item.stage == name), None)


def _latest_snapshot(
    snapshots: dict[str, StageCandidateSnapshot],
    names: tuple[str, ...],
) -> StageCandidateSnapshot | None:
    return next(
        (
            snapshots[name]
            for name in names
            if name in snapshots and snapshots[name].status == "completed"
        ),
        None,
    )


def _latest_match(
    stage_matches: dict[str, list[DiagnosticCandidate]],
    names: tuple[str, ...],
) -> DiagnosticCandidate | None:
    return next(
        (
            stage_matches[name][0]
            for name in names
            if stage_matches.get(name)
        ),
        None,
    )


def _first_rank(candidates: list[DiagnosticCandidate]) -> int | None:
    return next((item.rank for item in candidates if item.rank is not None), None)


def _title_seen_without_identifier_match(
    gold: EvalGoldPaper,
    snapshots: dict[str, StageCandidateSnapshot],
) -> bool:
    if not gold.title:
        return False
    normalized_gold = normalize_title(gold.title)
    return any(
        normalize_title(candidate.title) == normalized_gold
        and (
            gold.year is None
            or candidate.year is None
            or abs(gold.year - candidate.year) <= 1
        )
        for stage in RETRIEVAL_STAGES
        for candidate in snapshots.get(
            stage,
            StageCandidateSnapshot(stage=stage, status="skipped"),
        ).candidates
    )


def _all_sources_failed(output: SearchServiceOutput) -> bool:
    selected = set(output.search_plan.selected_sources)
    relevant = [item for item in output.source_stats if item.source in selected]
    if not relevant:
        return False
    successful_sources = {
        item.source for item in relevant if item.error_message is None
    }
    observed_sources = {item.source for item in relevant}
    return selected.issubset(observed_sources) and not successful_sources


def _normalize_source(source: str) -> str:
    normalized = source.strip().casefold().replace(" ", "_")
    aliases = {
        "semantic": "semantic_scholar",
        "semanticscholar": "semantic_scholar",
        "refchain": "openalex",
    }
    return aliases.get(normalized, normalized)


def _empty_source_stats() -> dict[str, float | int]:
    return {
        "request_count": 0,
        "success_count": 0,
        "error_count": 0,
        "returned_candidate_count": 0,
        "unique_candidate_count": 0,
        "gold_hit_count": 0,
        "unique_gold_hit_count": 0,
        "gold_recall_contribution": 0.0,
        "latency_seconds": 0.0,
    }


def _average_optional(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.mean(numeric) if numeric else None


def _stable_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_source(str(value))
        if not normalized or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result
