"""Pure-Replay upper-bound audit across frozen query-strategy products.

The module intentionally has no dependency on ``SearchService`` or a connector.
It consumes Benchmark result rows and immutable retrieval snapshots, introduces
gold only after query lists have been reconstructed, and never writes snapshots.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import build_identity_profile, identity_evidence_from_profiles
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.datasets import load_dataset
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    _rank_pool,
    identity_cluster_labels,
)
from scholar_agent.evaluation.metrics import (
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
)
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import SnapshotError


AUDIT_SCHEMA_VERSION = "1"
STRATEGY_ORDER = (
    "current_rules",
    "concept_projection",
    "llm_constrained_rewrite",
    "query_evolution",
    "llm_semantic",
)
StrategyName = Literal[
    "current_rules",
    "concept_projection",
    "llm_constrained_rewrite",
    "query_evolution",
    "llm_semantic",
]


class StrategyArtifact(BaseModel):
    """One frozen Benchmark Replay plus the retrieval store it consumed."""

    strategy: StrategyName
    run_dir: Path
    snapshot_dir: Path


class UnionAuditDataset(BaseModel):
    """All complete strategy products available for one fixed dataset slice."""

    name: str
    strategies: list[StrategyArtifact]
    excluded_strategies: dict[str, str] = Field(default_factory=dict)


class CoverOption(NamedTuple):
    option_id: str
    gold_ids: frozenset[str]
    request_count: int
    latency_seconds: float


class _Artifact(NamedTuple):
    spec: StrategyArtifact
    config: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    store: SnapshotStore


class _Observation(NamedTuple):
    strategy: str
    order: int
    stage: str
    source: str
    origin_query: str
    adapted_query: str
    snapshot_key: str
    status: str
    papers: list[Paper]
    costs: dict[str, float | int]
    error_type: str | None


def run_cross_strategy_union_audit(
    datasets: Sequence[UnionAuditDataset],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a deterministic, read-only audit for all supplied dataset slices."""

    case_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for dataset in sorted(datasets, key=lambda item: item.name):
        rows, inputs = _audit_dataset(dataset)
        case_rows.extend(rows)
        input_rows.extend(inputs)
    case_rows.sort(key=lambda row: (str(row["dataset"]), int(row["case_order"])))
    return case_rows, _aggregate(case_rows, datasets, input_rows)


def write_cross_strategy_union_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in case_rows
    )
    _atomic_write_text(root / "case_union_audit.jsonl", payload)
    _atomic_write_json(root / "aggregate.json", aggregate)


def exact_minimum_cover(
    universe: Iterable[str], options: Sequence[CoverOption]
) -> dict[str, Any]:
    """Return a deterministic exact set cover, optimized for requests then cost.

    Query-list coverage is usually tiny (one or two gold papers per case), so a
    dynamic program over covered gold sets is exact and avoids a greedy oracle.
    """

    target = frozenset(str(value) for value in universe)
    useful = sorted(
        (
            CoverOption(
                option.option_id,
                frozenset(option.gold_ids) & target,
                int(option.request_count),
                float(option.latency_seconds),
            )
            for option in options
            if frozenset(option.gold_ids) & target
        ),
        key=lambda item: item.option_id,
    )
    # covered -> (query count, requests, latency, selected ids)
    states: dict[frozenset[str], tuple[int, int, float, tuple[str, ...]]] = {
        frozenset(): (0, 0, 0.0, ())
    }
    for option in useful:
        updates = dict(states)
        for covered, score in states.items():
            combined = covered | option.gold_ids
            candidate = (
                score[0] + 1,
                score[1] + option.request_count,
                score[2] + option.latency_seconds,
                (*score[3], option.option_id),
            )
            existing = updates.get(combined)
            if existing is None or _cover_score(candidate) < _cover_score(existing):
                updates[combined] = candidate
        states = updates
    reachable = max(states, key=lambda value: (len(value), tuple(sorted(value))))
    best = states.get(target) if target in states else states[reachable]
    assert best is not None
    return {
        "universe_gold_ids": sorted(target),
        "covered_gold_ids": sorted(target if target in states else reachable),
        "uncovered_gold_ids": sorted(target - (target if target in states else reachable)),
        "complete": target in states,
        "selected_query_count": best[0],
        "request_count": best[1],
        "latency_seconds": best[2],
        "selected_query_ids": list(best[3]),
    }


def gold_priority_oracle(
    candidates: Sequence[Paper], gold: Sequence[EvalGoldPaper], *, k: int = 20
) -> dict[str, Any]:
    """Compute a candidate-pool oracle without changing the candidate set."""

    winners: list[Paper] = []
    used_candidates: set[int] = set()
    matched_gold_indices: set[int] = set()
    for gold_index, gold_paper in enumerate(gold):
        if not matched_paper_ids(candidates, [gold_paper]):
            continue
        candidate_index = next(
            (
                index
                for index, paper in enumerate(candidates)
                if index not in used_candidates
                and matched_paper_ids([paper], [gold_paper])
            ),
            None,
        )
        if candidate_index is None:
            continue
        winners.append(candidates[candidate_index])
        used_candidates.add(candidate_index)
        matched_gold_indices.add(gold_index)
    ordered = [
        *winners,
        *(paper for index, paper in enumerate(candidates) if index not in used_candidates),
    ]
    formal = ordered[:k]
    metrics = evaluate_ranking(formal, gold, k_values=[k])
    denominator = evaluable_gold_count(gold)
    return {
        "evaluable_gold_count": denominator,
        "candidate_gold_count": len(matched_gold_indices),
        "candidate_gold_ids": matched_paper_ids(candidates, gold),
        "top20_gold_ids": matched_paper_ids(formal, gold, k=k),
        "candidate_recall": (
            len(matched_gold_indices) / denominator if denominator else None
        ),
        "recall_at_20": metrics.recall_at_k[k] if denominator else None,
        "f1_at_20": metrics.f1_at_k[k] if denominator else None,
        "oracle_only_not_achieved_score": True,
    }


def rank_candidate_union(
    analysis: QueryAnalysis,
    strategy_pools: Sequence[Sequence[Paper]],
    gold: Sequence[EvalGoldPaper],
) -> tuple[list[Paper], dict[str, Any]]:
    """Identity-merge strategy pools and apply the unchanged current ranker."""

    candidates = deduplicate_papers(
        [paper for pool in strategy_pools for paper in pool]
    )
    metrics, _, _ = _rank_pool(analysis, candidates, gold)
    return candidates, metrics


def _audit_dataset(
    dataset: UnionAuditDataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts = [_load_artifact(spec) for spec in dataset.strategies]
    artifacts.sort(key=lambda item: STRATEGY_ORDER.index(item.spec.strategy))
    if not artifacts or artifacts[0].spec.strategy != "current_rules":
        raise ValueError(f"{dataset.name}:current_rules artifact required")
    baseline = artifacts[0]
    case_ids = [str(value) for value in baseline.config["case_ids"]]
    eval_cases = {
        item.query_id: item for item in load_dataset(str(baseline.config["dataset"]))
    }
    for artifact in artifacts:
        _validate_artifact(dataset.name, artifact, baseline, case_ids)

    rows = [
        _audit_case(
            dataset.name,
            case_order,
            eval_cases[case_id],
            artifacts,
        )
        for case_order, case_id in enumerate(case_ids)
    ]
    inputs = [
        {
            "dataset": dataset.name,
            "strategy": artifact.spec.strategy,
            "run_config_sha256": _sha256(artifact.spec.run_dir / "config.json"),
            "run_results_sha256": _sha256(artifact.spec.run_dir / "results.jsonl"),
            "snapshot_manifest_sha256": _sha256(
                artifact.spec.snapshot_dir / "manifest.json"
            ),
            "case_count": len(artifact.rows),
        }
        for artifact in artifacts
    ]
    return rows, inputs


def _audit_case(
    dataset_name: str,
    case_order: int,
    eval_query: EvalQuery,
    artifacts: Sequence[_Artifact],
) -> dict[str, Any]:
    artifact_rows: dict[str, dict[str, Any]] = {}
    observations: dict[str, list[_Observation]] = {}
    feature_states: dict[str, dict[str, Any]] = {}
    product_pools: dict[str, list[Paper]] = {}
    for artifact in artifacts:
        strategy = artifact.spec.strategy
        row = artifact.rows[eval_query.query_id]
        feature_states[strategy] = _feature_state(strategy, row)
        values, planned_counts = _read_observations(artifact, row)
        observations[strategy] = values
        try:
            product_pools[strategy] = _product_candidate_pool(row, values)
            product_error = None
        except ValueError as exc:
            product_pools[strategy] = []
            product_error = str(exc)
        terminal_counts = Counter(value.status for value in values)
        artifact_rows[strategy] = {
            "feature_state": feature_states[strategy],
            "planned_call_status_counts": planned_counts,
            "executed_query_list_count": len(values),
            "query_list_status_counts": dict(sorted(terminal_counts.items())),
            "product_candidate_count": len(product_pools[strategy]),
            "product_reconstruction_error": product_error,
            "query_lists": [_observation_row(value) for value in values],
        }

    current_pool = product_pools["current_rules"]
    effective_pools: dict[str, list[Paper]] = {"current_rules": current_pool}
    contributing_strategies = ["current_rules"]
    for artifact in artifacts[1:]:
        strategy = artifact.spec.strategy
        state = feature_states[strategy]["state"]
        if state == "action_executed":
            effective_pools[strategy] = product_pools[strategy]
            contributing_strategies.append(strategy)
        else:
            # A no-op or fallback is evaluated as baseline, never as strategy gain.
            effective_pools[strategy] = current_pool

    strategy_names = [artifact.spec.strategy for artifact in artifacts]
    grouped_labels = identity_cluster_labels(
        [effective_pools[strategy] for strategy in strategy_names]
    )
    strategy_candidate_sets = {
        strategy: set(grouped_labels[index])
        for index, strategy in enumerate(strategy_names)
    }
    strategy_gold_sets = {
        strategy: set(matched_paper_ids(effective_pools[strategy], eval_query.gold_papers))
        for strategy in strategy_names
    }

    analysis = QueryAnalysis.model_validate(
        artifacts[0].rows[eval_query.query_id]["stage_diagnostics"][
            "initial_query_planning"
        ]["query_analysis"]
    )
    current_metrics, _, _ = _rank_pool(analysis, current_pool, eval_query.gold_papers)
    union_candidates, union_metrics = rank_candidate_union(
        analysis,
        [product_pools[strategy] for strategy in contributing_strategies],
        eval_query.gold_papers,
    )
    oracle = gold_priority_oracle(union_candidates, eval_query.gold_papers)

    query_groups = _query_groups(observations, feature_states)
    duplicate_inconsistent = [
        item for item in query_groups if item["owner_count"] > 1 and not item["consistent"]
    ]
    strict_reasons = _strict_reasons(
        strategy_names,
        feature_states,
        observations,
        artifact_rows,
        duplicate_inconsistent,
    )
    strategy_rows = _strategy_rows(
        strategy_names,
        effective_pools,
        strategy_candidate_sets,
        strategy_gold_sets,
        observations,
        feature_states,
        eval_query.gold_papers,
    )
    minimal_cover = _minimum_query_cover(query_groups, eval_query.gold_papers, union_metrics)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": dataset_name,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "query": eval_query.query,
        "evaluable_gold_count": evaluable_gold_count(eval_query.gold_papers),
        "included_strategies": strategy_names,
        "contributing_strategies": contributing_strategies,
        "strategy_products": artifact_rows,
        "strategy_contribution": strategy_rows,
        "query_groups": query_groups,
        "duplicate_query_inconsistent_count": len(duplicate_inconsistent),
        "strict_comparable": not strict_reasons,
        "strict_incomparable_reasons": strict_reasons,
        "current_rules": current_metrics,
        "union_current_ranker": union_metrics,
        "union_gold_priority_oracle": oracle,
        "minimal_query_set_oracle": minimal_cover,
    }


def _strategy_rows(
    strategies: Sequence[str],
    pools: dict[str, list[Paper]],
    candidate_sets: dict[str, set[str]],
    gold_sets: dict[str, set[str]],
    observations: dict[str, list[_Observation]],
    feature_states: dict[str, dict[str, Any]],
    gold: Sequence[EvalGoldPaper],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    query_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for strategy in strategies:
        if strategy != "current_rules" and feature_states[strategy]["state"] != "action_executed":
            continue
        for item in observations[strategy]:
            query_owners[(item.source, item.adapted_query)].add(strategy)
    for strategy in strategies:
        others = [value for value in strategies if value != strategy]
        other_candidates = set().union(*(candidate_sets[value] for value in others))
        other_gold = set().union(*(gold_sets[value] for value in others))
        attributable = deduplicate_papers(
            [
                paper
                for item in observations[strategy]
                if item.status == "success"
                and query_owners[(item.source, item.adapted_query)] == {strategy}
                for paper in item.papers
            ]
        )
        result[strategy] = {
            "feature_state": feature_states[strategy],
            "candidate_count": len(pools[strategy]),
            "candidate_ids": sorted(candidate_sets[strategy]),
            "candidate_gold_ids": sorted(gold_sets[strategy]),
            "observed_independent_candidate_count": len(
                candidate_sets[strategy] - other_candidates
            ),
            "observed_independent_gold_ids": sorted(gold_sets[strategy] - other_gold),
            "unique_query_attributable_candidate_count": len(attributable),
            "unique_query_attributable_gold_ids": sorted(
                matched_paper_ids(attributable, gold)
            ),
        }
    return result


def _query_groups(
    observations: dict[str, list[_Observation]],
    feature_states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[_Observation]] = defaultdict(list)
    for strategy, values in observations.items():
        if strategy != "current_rules" and feature_states[strategy]["state"] != "action_executed":
            continue
        for value in values:
            grouped[(value.source, value.adapted_query)].append(value)
    rows: list[dict[str, Any]] = []
    for (source, query), values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda value: (
                STRATEGY_ORDER.index(value.strategy),
                value.order,
                value.snapshot_key,
            ),
        )
        success = [value for value in ordered if value.status == "success"]
        chosen = success[0] if success else ordered[0]
        consistent = all(value.status == chosen.status for value in ordered)
        if consistent and success:
            consistent = all(
                _equivalent_paper_sequences(chosen.papers, value.papers)
                for value in success[1:]
            )
        rows.append(
            {
                "query_id": _query_id(source, query),
                "source": source,
                "query": query,
                "owner_count": len({value.strategy for value in ordered}),
                "owners": sorted(
                    {value.strategy for value in ordered}, key=STRATEGY_ORDER.index
                ),
                "consistent": consistent,
                "chosen_snapshot_key": chosen.snapshot_key,
                "chosen_status": chosen.status,
                "chosen_candidate_count": len(chosen.papers),
                "chosen_costs": chosen.costs,
                "observations": [
                    {
                        "strategy": value.strategy,
                        "snapshot_key": value.snapshot_key,
                        "status": value.status,
                        "candidate_count": len(value.papers),
                    }
                    for value in ordered
                ],
                "_chosen_papers": chosen.papers,
            }
        )
    return rows


def _minimum_query_cover(
    query_groups: Sequence[dict[str, Any]],
    gold: Sequence[EvalGoldPaper],
    union_metrics: dict[str, Any],
) -> dict[str, Any]:
    options: list[CoverOption] = []
    for group in query_groups:
        if group["chosen_status"] != "success":
            continue
        gold_ids = frozenset(matched_paper_ids(group["_chosen_papers"], gold))
        costs = group["chosen_costs"]
        options.append(
            CoverOption(
                str(group["query_id"]),
                gold_ids,
                int(costs.get("request_count") or 0),
                float(costs.get("latency_seconds") or 0.0),
            )
        )
    result = exact_minimum_cover(union_metrics["candidate_gold_ids"], options)
    for group in query_groups:
        group.pop("_chosen_papers", None)
    result["candidate_union_gold_count"] = len(union_metrics["candidate_gold_ids"])
    result["query_option_count"] = len(options)
    result["all_query_option_request_count"] = sum(
        option.request_count for option in options
    )
    result["all_query_option_latency_seconds"] = sum(
        option.latency_seconds for option in options
    )
    return result


def _feature_state(strategy: str, row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row["stage_diagnostics"]
    planning = diagnostics["initial_query_planning"]["planning"]
    if strategy == "current_rules":
        return {"state": "baseline", "reason": None}
    if strategy == "concept_projection":
        if planning.get("concept_projection_query") and not planning.get(
            "concept_projection_skip_reason"
        ):
            return {"state": "action_executed", "reason": None}
        return {
            "state": "no_action",
            "reason": str(planning.get("concept_projection_skip_reason") or "no_projection"),
        }
    if strategy == "llm_constrained_rewrite":
        if int(planning.get("accepted_query_count") or 0) == 1 and not planning.get(
            "fallback_used"
        ):
            return {"state": "action_executed", "reason": None}
        return {
            "state": "fallback_or_rejected",
            "reason": str(planning.get("fallback_reason") or "rewrite_not_accepted"),
        }
    if strategy == "llm_semantic":
        if int(planning.get("accepted_query_count") or 0) > 0 and not planning.get(
            "fallback_used"
        ):
            return {"state": "action_executed", "reason": None}
        return {
            "state": "fallback_or_rejected",
            "reason": str(planning.get("fallback_reason") or "semantic_not_accepted"),
        }
    evolution = diagnostics["query_evolution"]
    if bool(evolution.get("triggered")) and int(
        evolution.get("executed_query_count") or 0
    ) > 0:
        return {"state": "action_executed", "reason": None}
    return {
        "state": "no_action",
        "reason": ",".join(str(value) for value in evolution.get("skipped_reasons") or [])
        or "query_evolution_not_triggered",
    }


def _read_observations(
    artifact: _Artifact, row: dict[str, Any]
) -> tuple[list[_Observation], dict[str, int]]:
    result: list[_Observation] = []
    planned = Counter()
    seen_keys: set[str] = set()
    order = 0
    for stage in row["stage_diagnostics"]["snapshots"]:
        for call in stage.get("retrieval_calls") or []:
            if not call.get("logical_call_executed"):
                planned[str(call.get("terminal_status") or "not_started")] += 1
                continue
            key = str(call.get("snapshot_key") or "")
            if not key:
                planned["missing_snapshot_key"] += 1
                continue
            if key in seen_keys:
                planned["duplicate_snapshot_key_reuse"] += 1
                continue
            seen_keys.add(key)
            order += 1
            source = str(call.get("source") or "")
            adapted = str(call.get("adapted_query") or "")
            try:
                entry = artifact.store.read_retrieval(key)
                if entry.source != source or entry.adapted_query != adapted:
                    raise ValueError("snapshot_request_mismatch")
                status = str(entry.status)
                papers = list(entry.papers) if status == "success" else []
                diagnostics = entry.diagnostics.model_dump(mode="json")
                costs = _costs(diagnostics, entry.recorded_latency_seconds)
                error_type = _error_type(entry.error_message)
            except SnapshotError as exc:
                status = "missing_or_invalid_snapshot"
                papers = []
                costs = _empty_costs()
                error_type = type(exc).__name__
            planned[status] += 1
            result.append(
                _Observation(
                    artifact.spec.strategy,
                    order,
                    str(stage.get("stage") or "unknown"),
                    source,
                    str(call.get("origin_subquery") or ""),
                    adapted,
                    key,
                    status,
                    papers,
                    costs,
                    error_type,
                )
            )
    return result, dict(sorted(planned.items()))


def _product_candidate_pool(
    row: dict[str, Any], observations: Sequence[_Observation]
) -> list[Paper]:
    if row.get("status") not in {"success", "succeeded"}:
        raise ValueError("benchmark_case_not_success")
    stages = row["stage_diagnostics"]["snapshots"]
    target = next(
        (
            stage
            for name in (
                "post_refchain_deduplicated",
                "post_evolution_deduplicated",
                "initial_deduplicated",
            )
            for stage in stages
            if stage.get("stage") == name and stage.get("status") == "completed"
        ),
        None,
    )
    if target is None:
        raise ValueError("candidate_stage_unavailable")
    raw = deduplicate_papers(
        [
            paper
            for observation in observations
            if observation.status == "success"
            for paper in observation.papers
        ]
    )
    return _align_to_diagnostics(raw, list(target.get("candidates") or []))


def _align_to_diagnostics(
    papers: Sequence[Paper], diagnostics: Sequence[dict[str, Any]]
) -> list[Paper]:
    profiles = [build_identity_profile(paper) for paper in papers]
    used: set[int] = set()
    aligned: list[Paper] = []
    for diagnostic in diagnostics:
        target = build_identity_profile(diagnostic)
        index = next(
            (
                offset
                for offset, profile in enumerate(profiles)
                if offset not in used
                and identity_evidence_from_profiles(profile, target).equivalent
            ),
            None,
        )
        if index is None:
            raise ValueError("candidate_stage_alignment_failed")
        used.add(index)
        aligned.append(papers[index])
    return aligned


def _strict_reasons(
    strategies: Sequence[str],
    feature_states: dict[str, dict[str, Any]],
    observations: dict[str, list[_Observation]],
    artifact_rows: dict[str, dict[str, Any]],
    duplicate_inconsistent: Sequence[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for strategy in strategies:
        if feature_states[strategy]["state"] == "fallback_or_rejected":
            reasons.append(f"{strategy}:fallback_or_rejected")
        if artifact_rows[strategy]["product_reconstruction_error"]:
            reasons.append(f"{strategy}:product_reconstruction_failed")
        for status in sorted(
            {value.status for value in observations[strategy] if value.status != "success"}
        ):
            reasons.append(f"{strategy}:terminal_{status}")
    if duplicate_inconsistent:
        reasons.append("duplicate_query_response_inconsistent")
    return sorted(set(reasons))


def _aggregate(
    case_rows: Sequence[dict[str, Any]],
    datasets: Sequence[UnionAuditDataset],
    inputs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dataset_specs = {item.name: item for item in datasets}
    for name in sorted(dataset_specs):
        rows = [row for row in case_rows if row["dataset"] == name]
        strict = [row for row in rows if row["strict_comparable"]]
        strategies = [item.strategy for item in dataset_specs[name].strategies]
        result[name] = {
            "case_count": len(rows),
            "included_strategies": strategies,
            "excluded_strategies": dict(
                sorted(dataset_specs[name].excluded_strategies.items())
            ),
            "strict_comparable_case_count": len(strict),
            "strict_incomparable_reason_counts": _reason_counts(
                row["strict_incomparable_reasons"] for row in rows
            ),
            "feature_state_counts": {
                strategy: dict(
                    sorted(
                        Counter(
                            row["strategy_products"][strategy]["feature_state"]["state"]
                            for row in rows
                        ).items()
                    )
                )
                for strategy in strategies
            },
            "query_list_status_counts": {
                strategy: _sum_counters(
                    row["strategy_products"][strategy]["query_list_status_counts"]
                    for row in rows
                )
                for strategy in strategies
            },
            "duplicate_query_group_count": sum(
                sum(group["owner_count"] > 1 for group in row["query_groups"])
                for row in rows
            ),
            "duplicate_query_inconsistent_count": sum(
                row["duplicate_query_inconsistent_count"] for row in rows
            ),
            "single_strategy": {
                strategy: _aggregate_strategy(rows, strategy) for strategy in strategies
            },
            "strategy_overlap": _aggregate_overlap(rows, strategies),
            "all_observed": {
                "current_rules": _aggregate_metrics(
                    row["current_rules"] for row in rows
                ),
                "strategy_union_current_ranker": _aggregate_metrics(
                    row["union_current_ranker"] for row in rows
                ),
                "strategy_union_gold_priority_oracle": _aggregate_oracle(
                    row["union_gold_priority_oracle"] for row in rows
                ),
            },
            "strict_comparable_subset": {
                "case_count": len(strict),
                "current_rules": _aggregate_metrics(
                    row["current_rules"] for row in strict
                ),
                "strategy_union_current_ranker": _aggregate_metrics(
                    row["union_current_ranker"] for row in strict
                ),
                "strategy_union_gold_priority_oracle": _aggregate_oracle(
                    row["union_gold_priority_oracle"] for row in strict
                ),
            },
            "minimal_query_set_oracle": _aggregate_minimal_cover(rows),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "offline_frozen_snapshot_replay",
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "inputs": list(inputs),
        "datasets": result,
    }


def _aggregate_strategy(rows: Sequence[dict[str, Any]], strategy: str) -> dict[str, Any]:
    values = [row["strategy_contribution"][strategy] for row in rows]
    return {
        "candidate_count": sum(int(value["candidate_count"]) for value in values),
        "candidate_gold_count": sum(len(value["candidate_gold_ids"]) for value in values),
        "observed_independent_candidate_count": sum(
            int(value["observed_independent_candidate_count"]) for value in values
        ),
        "observed_independent_gold_count": sum(
            len(value["observed_independent_gold_ids"]) for value in values
        ),
        "unique_query_attributable_candidate_count": sum(
            int(value["unique_query_attributable_candidate_count"]) for value in values
        ),
        "unique_query_attributable_gold_count": sum(
            len(value["unique_query_attributable_gold_ids"]) for value in values
        ),
    }


def _aggregate_overlap(
    rows: Sequence[dict[str, Any]], strategies: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left_index, left in enumerate(strategies):
        for right in strategies[left_index + 1 :]:
            candidate_overlap = 0
            gold_overlap = 0
            for row in rows:
                left_value = row["strategy_contribution"][left]
                right_value = row["strategy_contribution"][right]
                candidate_overlap += len(
                    set(left_value["candidate_ids"])
                    & set(right_value["candidate_ids"])
                )
                gold_overlap += len(
                    set(left_value["candidate_gold_ids"])
                    & set(right_value["candidate_gold_ids"])
                )
            result[f"{left}__{right}"] = {
                "candidate_overlap_count": candidate_overlap,
                "gold_overlap_count": gold_overlap,
            }
    return result


def _aggregate_metrics(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(values)
    evaluable = [row for row in rows if row.get("candidate_recall") is not None]
    return {
        "case_count": len(rows),
        "evaluable_case_count": len(evaluable),
        "candidate_count": sum(int(row["candidate_count"]) for row in evaluable),
        "candidate_gold_count": sum(len(row["candidate_gold_ids"]) for row in evaluable),
        "top20_gold_count": sum(len(row["returned_gold_ids"]) for row in evaluable),
        "candidate_recall": _mean(row["candidate_recall"] for row in evaluable),
        "recall_at_20": _mean(row["recall_at_20"] for row in evaluable),
        "f1_at_20": _mean(row["f1_at_20"] for row in evaluable),
    }


def _aggregate_oracle(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(values)
    evaluable = [row for row in rows if row["evaluable_gold_count"] > 0]
    return {
        "case_count": len(rows),
        "evaluable_case_count": len(evaluable),
        "candidate_gold_count": sum(
            int(row["candidate_gold_count"]) for row in evaluable
        ),
        "top20_gold_count": sum(
            len(row["top20_gold_ids"]) for row in evaluable
        ),
        "candidate_recall": _mean(row["candidate_recall"] for row in evaluable),
        "recall_at_20": _mean(row["recall_at_20"] for row in evaluable),
        "f1_at_20": _mean(row["f1_at_20"] for row in evaluable),
        "oracle_only_not_achieved_score": True,
    }


def _aggregate_minimal_cover(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [row["minimal_query_set_oracle"] for row in rows]
    return {
        "case_count": len(values),
        "complete_case_count": sum(bool(value["complete"]) for value in values),
        "candidate_union_gold_count": sum(
            int(value["candidate_union_gold_count"]) for value in values
        ),
        "covered_gold_count": sum(len(value["covered_gold_ids"]) for value in values),
        "uncovered_gold_count": sum(len(value["uncovered_gold_ids"]) for value in values),
        "selected_query_count": sum(int(value["selected_query_count"]) for value in values),
        "request_count": sum(int(value["request_count"]) for value in values),
        "latency_seconds": sum(float(value["latency_seconds"]) for value in values),
        "query_option_count": sum(int(value["query_option_count"]) for value in values),
        "all_query_option_request_count": sum(
            int(value["all_query_option_request_count"]) for value in values
        ),
        "all_query_option_latency_seconds": sum(
            float(value["all_query_option_latency_seconds"]) for value in values
        ),
        "audit_oracle_not_for_production": True,
    }


def _load_artifact(spec: StrategyArtifact) -> _Artifact:
    run_dir = spec.run_dir.expanduser().resolve()
    snapshot_dir = spec.snapshot_dir.expanduser().resolve()
    normalized = spec.model_copy(
        update={"run_dir": run_dir, "snapshot_dir": snapshot_dir}
    )
    return _Artifact(
        normalized,
        _read_json(run_dir / "config.json"),
        _read_rows(run_dir / "results.jsonl"),
        SnapshotStore(snapshot_dir),
    )


def _validate_artifact(
    dataset_name: str,
    artifact: _Artifact,
    baseline: _Artifact,
    case_ids: Sequence[str],
) -> None:
    config = artifact.config
    if config.get("retrieval_mode") != "replay":
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:not replay")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:top_k mismatch")
    if str(config.get("dataset")) != str(baseline.config.get("dataset")):
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:dataset mismatch")
    if [str(value) for value in config.get("case_ids") or []] != list(case_ids):
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:case order mismatch")
    if set(artifact.rows) != set(case_ids):
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:result cases incomplete")
    if not artifact.spec.snapshot_dir.joinpath("manifest.json").is_file():
        raise ValueError(f"{dataset_name}:{artifact.spec.strategy}:manifest missing")


def _observation_row(value: _Observation) -> dict[str, Any]:
    return {
        "stage": value.stage,
        "source": value.source,
        "origin_query": value.origin_query,
        "adapted_query": value.adapted_query,
        "snapshot_key": value.snapshot_key,
        "status": value.status,
        "candidate_count": len(value.papers),
        "costs": value.costs,
        "error_type": value.error_type,
    }


def _costs(diagnostics: dict[str, Any], recorded_latency: float) -> dict[str, float | int]:
    return {
        "request_count": int(diagnostics.get("request_count") or 0),
        "retry_count": int(diagnostics.get("retry_count") or 0),
        "error_count": int(diagnostics.get("error_count") or 0),
        "cache_hit_count": int(diagnostics.get("cache_hit_count") or 0),
        "latency_seconds": float(
            diagnostics.get("latency_seconds") or recorded_latency or 0.0
        ),
        "rate_limit_wait_seconds": float(
            diagnostics.get("rate_limit_wait_seconds") or 0.0
        ),
    }


def _empty_costs() -> dict[str, float | int]:
    return {
        "request_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "cache_hit_count": 0,
        "latency_seconds": 0.0,
        "rate_limit_wait_seconds": 0.0,
    }


def _equivalent_paper_sequences(left: Sequence[Paper], right: Sequence[Paper]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        identity_evidence_from_profiles(
            build_identity_profile(left_paper), build_identity_profile(right_paper)
        ).equivalent
        for left_paper, right_paper in zip(left, right)
    )


def _query_id(source: str, query: str) -> str:
    return hashlib.sha256(
        json.dumps([source, query], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _cover_score(value: tuple[int, int, float, tuple[str, ...]]) -> tuple[Any, ...]:
    return value[0], value[1], round(value[2], 9), value[3]


def _error_type(message: str | None) -> str | None:
    if not message:
        return None
    value = str(message).lower()
    if "429" in value:
        return "http_429_rate_limit"
    if "timeout" in value or "timed out" in value:
        return "timeout"
    if "cancel" in value:
        return "cancelled"
    return "source_failure"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["case_id"])
        if key in result:
            raise ValueError(f"duplicate result case:{key}")
        result[key] = row
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Iterable[float | None]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return sum(rows) / len(rows) if rows else None


def _reason_counts(values: Iterable[Sequence[str]]) -> dict[str, int]:
    return dict(sorted(Counter(item for group in values for item in group).items()))


def _sum_counters(values: Iterable[dict[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for value in values:
        total.update(value)
    return dict(sorted(total.items()))


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
