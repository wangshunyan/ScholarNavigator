"""Gold-free Judgement and rerank decision audit for frozen Record160 Replay.

The module observes values emitted by the production Judgement/rerank path and
reconstructs the frozen ordering from the registered production sort key.  It
does not implement an alternative scorer and never loads evaluator labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.agents.judgement import (
    PRODUCTION_JUDGEMENT_COMPONENT_ORDER,
    production_category_from_score,
    production_judgement_decision_catalog,
    trace_constraint_decisions,
    trace_judgement_decision,
)
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.agents.reranker import (
    production_ranking_decision_catalog,
    trace_ranking_decision,
)
from scholar_agent.core.dedup import deduplicate_papers_with_lineage
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.result_lineage import result_identity
from scholar_agent.core.search_schemas import JudgementResult, QueryAnalysis, RankedPaper
from scholar_agent.evaluation.constraint_decision_audit import (
    _load_component_assignments,
    _opaque_identity,
    _read_json,
    _read_rows,
    _repo_path,
    _sha256,
    _stable_json_sha256,
    _validate_config,
    _validate_file_hash,
    _validate_population,
    summarize_distribution,
)
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.source_fusion_ablation import (
    IdentityRegistry,
    VariantResult,
    rank_variant,
    validate_full_reconstruction,
)
from scholar_agent.evaluation.source_reliability_diagnostics import (
    audit_retrieval_requests,
    classify_query_structure,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "ranking_decision_audit_v1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RETURN_CATEGORIES = frozenset({"highly_relevant", "partially_relevant"})
_SOURCE_ORDER = ("openalex", "arxiv", "semantic_scholar", "pubmed")


class RankingDecisionAuditError(RuntimeError):
    """A score, category, order, or selection invariant was violated."""


class RankingDecisionNotEligible(RankingDecisionAuditError):
    """Frozen evidence cannot support the preregistered ranking audit."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path).expanduser().resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise RankingDecisionAuditError("unsupported_protocol")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise RankingDecisionAuditError("offline_protocol_drift")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_ranking_result",
    ]:
        raise RankingDecisionAuditError("selection_contract_drift")
    frozen = value.get("decision_catalog") or {}
    judgement = production_judgement_decision_catalog(CURRENT_RULES_CONFIG)
    rerank = production_ranking_decision_catalog()
    if frozen.get("judgement_components") != judgement["component_order"]:
        raise RankingDecisionAuditError("judgement_component_catalog_drift")
    if frozen.get("category_thresholds", {}).get("comparison") != judgement[
        "threshold_comparison"
    ]:
        raise RankingDecisionAuditError("category_comparison_drift")
    if {
        key: frozen["category_thresholds"][key]
        for key in ("highly_relevant", "partially_relevant", "weakly_relevant")
    } != judgement["thresholds"]:
        raise RankingDecisionAuditError("category_threshold_drift")
    if frozen.get("rerank_components") != rerank["score_components"]:
        raise RankingDecisionAuditError("rerank_component_catalog_drift")
    if frozen.get("rerank_key") != rerank["sort_key"]:
        raise RankingDecisionAuditError("rerank_key_catalog_drift")
    if frozen.get("result_policy", {}).get("operation_order") != (
        "rerank_all_then_take_first_20_then_apply_category_gate"
    ):
        raise RankingDecisionAuditError("result_policy_drift")
    return value


def run_ranking_decision_audit(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the preregistered audit with network and evaluator access disabled."""

    root = Path(repository_root).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    frozen = protocol["frozen_input"]
    run_dir = _repo_path(root, frozen["run_dir"])
    snapshot_dir = _repo_path(root, frozen["snapshot_dir"])
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    assignments_path = _repo_path(root, frozen["component_assignments"]["path"])
    _validate_file_hash(config_path, frozen["config_sha256"])
    _validate_file_hash(results_path, frozen["record_results_sha256"])
    _validate_file_hash(assignments_path, frozen["component_assignments"]["sha256"])
    before_tree = _tree_sha256(snapshot_dir)
    if before_tree != frozen["snapshot_tree_sha256"]:
        raise RankingDecisionNotEligible("snapshot_tree_hash_drift")
    if sum(path.is_file() for path in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise RankingDecisionNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, protocol)
    rows = _read_rows(results_path)
    if len(rows) != int(protocol["analysis_population"]["record_case_count"]):
        raise RankingDecisionNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[: len(rows)]:
        raise RankingDecisionNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise RankingDecisionNotEligible("missing_component_assignment")

    store = SnapshotStore(snapshot_dir)
    attempts = {"network": 0}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    with _forbid_network(attempts):
        for case_order, row in enumerate(rows):
            case, candidate_rows, case_keys = analyze_case(
                row,
                config=config,
                protocol=protocol,
                store=store,
                component_id=components[str(row["case_id"])],
                case_order=case_order,
            )
            observed_keys.update(case_keys)
            if case["analysis_status"] == "excluded_no_successful_source":
                excluded.append(case)
            else:
                included.append(case)
                decisions.extend(candidate_rows)

    _validate_population(included, excluded, protocol)
    if len(decisions) != int(protocol["analysis_population"]["expected_candidate_count"]):
        raise RankingDecisionNotEligible("candidate_population_drift")
    if len(observed_keys) != int(frozen["observed_snapshot_key_count"]):
        raise RankingDecisionNotEligible("observed_snapshot_key_count_drift")
    if _tree_sha256(snapshot_dir) != before_tree:
        raise RankingDecisionAuditError("snapshot_tree_changed")
    if attempts["network"]:
        raise RankingDecisionAuditError("network_attempt_detected")
    cases = sorted([*included, *excluded], key=lambda item: int(item["case_order"]))
    aggregate = aggregate_analysis(
        included,
        excluded,
        decisions,
        protocol,
        protocol_sha256=_sha256(protocol_file),
        input_hashes={
            "config_sha256": _sha256(config_path),
            "record_results_sha256": _sha256(results_path),
            "snapshot_tree_sha256": before_tree,
            "component_assignments_sha256": _sha256(assignments_path),
        },
        observed_snapshot_key_count=len(observed_keys),
    )
    return cases, decisions, aggregate


def analyze_case(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    store: Any,
    component_id: str,
    case_order: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    stages = {
        str(item.get("stage")): item
        for item in row["stage_diagnostics"]["snapshots"]
    }
    required = set(protocol["reconstruction"]["required_exact_stages"]) | {
        "initial_retrieval"
    }
    if not required.issubset(stages):
        raise RankingDecisionNotEligible("required_frozen_stage_missing")
    sources = [str(value) for value in config["sources"]]
    requests = audit_retrieval_requests(
        stages["initial_retrieval"], config=config, store=store, sources=sources
    )
    successful_source_count = sum(
        int(requests.source_records[source]["snapshot_success_count"]) > 0
        for source in sources
    )
    query_identity = _opaque_identity("query", str(row["case_id"]))
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": (
            "included_main_analysis"
            if successful_source_count
            else "excluded_no_successful_source"
        ),
        "case_order": case_order,
        "query_identity": query_identity,
        "component_identity": _opaque_identity("component", str(component_id)),
        "successful_source_count": successful_source_count,
        "query_structure": classify_query_structure(str(row["original_query"])),
    }
    if not successful_source_count:
        return base, [], requests.observed_keys

    analysis = QueryAnalysis.model_validate(row["query_analysis"])
    raw = [
        paper.model_copy(deep=True)
        for _source, batch in requests.ordered_batches
        for paper in batch
    ]
    deduplicated, _audit, lineage = deduplicate_papers_with_lineage(
        raw, query_identity=query_identity
    )
    candidates = list(deduplicated)
    limit = int(protocol["reconstruction"]["candidate_limit"])
    if len(candidates) > limit:
        candidates = stable_source_coverage_truncate(
            candidates, limit=limit, source_order=sources
        )
    candidates = align_papers_to_diagnostics(
        candidates, stages["initial_deduplicated"]["candidates"]
    )
    full = rank_variant(
        analysis, candidates, top_k=int(protocol["reconstruction"]["top_k"])
    )
    validate_full_reconstruction(full, stages)
    registry = IdentityRegistry()
    candidate_ids = registry.labels(full.candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RankingDecisionAuditError("candidate_identity_duplicate_after_dedup")
    source_sets = {
        source: set(registry.labels(requests.papers_by_source[source]))
        for source in sources
    }
    lineage_by_result = {
        str(item["result_identity"]): item for item in lineage["results"]
    }
    rank_by_id = {registry.label(item.paper): item for item in full.ranked}
    returned_ids = registry.labels([item.paper for item in full.returned])
    returned_set = set(returned_ids)
    traces = [
        trace_ranking_decision(analysis, judgement, index)
        for index, judgement in enumerate(full.judgements)
    ]
    title_keys = sorted({str(item["sort_key"][5]) for item in traces})
    title_ordinals = {value: index for index, value in enumerate(title_keys)}
    top_k = int(protocol["reconstruction"]["top_k"])
    cutline = next((item for item in full.ranked if item.rank == top_k), None)
    cutline_identity = registry.label(cutline.paper) if cutline is not None else None
    records: list[dict[str, Any]] = []
    for candidate_order, (identity, paper, judgement, trace) in enumerate(
        zip(
            candidate_ids,
            full.candidates,
            full.judgements,
            traces,
            strict=True,
        )
    ):
        ranked = rank_by_id[identity]
        judgement_trace = trace_judgement_decision(judgement)
        constraint_trace = trace_constraint_decisions(
            analysis, paper, config=CURRENT_RULES_CONFIG
        )
        failed_dimensions = [
            str(item["constraint"])
            for item in constraint_trace
            if item["status"] == "failed"
        ]
        sort_key = list(trace["sort_key"])
        safe_sort_key = [
            int(sort_key[0]),
            float(sort_key[1]),
            float(sort_key[2]),
            int(sort_key[3]),
            int(sort_key[4]),
            title_ordinals[str(sort_key[5])],
            int(sort_key[6]),
        ]
        safe_tie_key = safe_sort_key[:-1]
        field_lineage = summarize_field_lineage(
            lineage_by_result.get(result_identity(paper)), lineage
        )
        in_rank_window = ranked.rank <= top_k
        retained = judgement.category in _RETURN_CATEGORIES
        selected = identity in returned_set
        record = {
            "schema_version": SCHEMA_VERSION,
            "case_order": case_order,
            "candidate_order": candidate_order,
            "query_identity": query_identity,
            "component_identity": base["component_identity"],
            "candidate_identity": identity,
            "source_provenance": [
                source for source in sources if identity in source_sets[source]
            ],
            "field_lineage": field_lineage,
            "judgement": {
                "components": judgement_trace["components"],
                "missing_components": [
                    name
                    for name in PRODUCTION_JUDGEMENT_COMPONENT_ORDER
                    if name not in judgement_trace["components"]
                ],
                "total_score": judgement.score,
                "category": judgement.category,
                "category_reason": judgement_trace["category_reason"],
                "threshold_margin": threshold_margin(
                    judgement.score, judgement.category
                ),
                "failed_constraint_dimensions": failed_dimensions,
                "hard_constraint_failures": judgement_trace.get(
                    "hard_constraint_failures", []
                ),
            },
            "ranking": {
                "score_breakdown": trace["score_breakdown"],
                "pre_rerank_position": candidate_order + 1,
                "reranked_position": ranked.rank,
                "position_delta": candidate_order + 1 - ranked.rank,
                "sort_key": safe_sort_key,
                "tie_key": safe_tie_key,
                "tie_break": {
                    "title_casefold_ordinal": safe_sort_key[5],
                    "original_index": safe_sort_key[6],
                },
            },
            "top20": {
                "top_k": top_k,
                "cutline_candidate_identity": cutline_identity,
                "rank_margin": top_k - ranked.rank,
                "within_rank_window": in_rank_window,
                "category_gate_passed": retained,
                "final_returned": selected,
                "reason": selection_reason(in_rank_window, retained),
            },
        }
        records.append(record)

    validate_decision_records(records, full, registry, protocol)
    permutation = input_permutation_diagnostic(analysis, full, registry, records)
    return (
        {
            **base,
            "candidate_count": len(records),
            "returned_count": len(full.returned),
            "reconstruction": {
                "initial_deduplicated_exact": True,
                "initial_judged_exact": True,
                "initial_reranked_exact": True,
                "final_returned_exact": True,
                "decision_record_exact": True,
                "digest": _stable_json_sha256(
                    {
                        "candidate_ids": candidate_ids,
                        "ranked_ids": registry.labels(
                            [item.paper for item in full.ranked]
                        ),
                        "returned_ids": returned_ids,
                    }
                ),
            },
            "input_permutation": permutation,
        },
        records,
        requests.observed_keys,
    )


def threshold_margin(score: float, category: str) -> dict[str, Any]:
    thresholds = production_judgement_decision_catalog()["thresholds"]
    high = float(thresholds["highly_relevant"])
    partial = float(thresholds["partially_relevant"])
    weak = float(thresholds["weakly_relevant"])
    if category == "highly_relevant":
        return {
            "lower_threshold": high,
            "lower_margin": _rounded(score - high),
            "upper_threshold": None,
            "upper_margin": None,
        }
    if category == "partially_relevant":
        return {
            "lower_threshold": partial,
            "lower_margin": _rounded(score - partial),
            "upper_threshold": high,
            "upper_margin": _rounded(high - score),
        }
    if category == "weakly_relevant":
        return {
            "lower_threshold": weak,
            "lower_margin": _rounded(score - weak),
            "upper_threshold": partial,
            "upper_margin": _rounded(partial - score),
        }
    if category == "irrelevant":
        return {
            "lower_threshold": None,
            "lower_margin": None,
            "upper_threshold": weak,
            "upper_margin": _rounded(weak - score),
        }
    return {
        "lower_threshold": None,
        "lower_margin": None,
        "upper_threshold": None,
        "upper_margin": None,
    }


def selection_reason(in_rank_window: bool, retained: bool) -> str:
    if in_rank_window and retained:
        return "returned"
    if in_rank_window:
        return "category_gate"
    if retained:
        return "beyond_top20_cutline"
    return "category_gate_and_beyond_top20"


def summarize_field_lineage(
    result: Mapping[str, Any] | None, document: Mapping[str, Any]
) -> dict[str, Any]:
    if result is None:
        raise RankingDecisionAuditError("candidate_field_lineage_missing")
    records = {
        str(item["record_ref"]): item for item in document.get("source_records") or []
    }
    fields: dict[str, Any] = {}
    for decision in result.get("field_decisions") or []:
        refs = [str(value) for value in decision.get("selected_record_refs") or []]
        if any(ref not in records for ref in refs):
            raise RankingDecisionAuditError("field_lineage_record_reference_missing")
        fields[str(decision["field"])] = {
            "status": str(decision["status"]),
            "selection_rule": str(decision["selection_rule"]),
            "selected_source_record_hashes": sorted(
                {str(records[ref]["source_record_sha256"]) for ref in refs}
            ),
            "selected_sources": sorted(
                {
                    str(source)
                    for ref in refs
                    for source in records[ref].get("sources") or []
                }
            ),
            "deterministic_steps": list(decision.get("deterministic_steps") or []),
            "candidate_state_counts": dict(
                sorted(
                    Counter(
                        str(item.get("state") or "missing")
                        for item in decision.get("candidates") or []
                    ).items()
                )
            ),
        }
    return {
        "contract": str(document["contract"]),
        "identity_normalization_version": str(document["identity_normalization_version"]),
        "field_merge_version": str(document["field_merge_version"]),
        "contributing_sources": list(result.get("contributing_sources") or []),
        "fields": dict(sorted(fields.items())),
    }


def validate_decision_records(
    records: Sequence[Mapping[str, Any]],
    full: VariantResult,
    registry: IdentityRegistry,
    protocol: Mapping[str, Any],
) -> None:
    if len(records) != len(full.candidates):
        raise RankingDecisionAuditError("decision_record_count_mismatch")
    identities = [str(item["candidate_identity"]) for item in records]
    if len(identities) != len(set(identities)):
        raise RankingDecisionAuditError("decision_record_duplicate_identity")
    tolerance = float(
        protocol["decision_catalog"]["floating_point"][
            "reconstruction_absolute_tolerance"
        ]
    )
    ranked_by_id = {registry.label(item.paper): item for item in full.ranked}
    for record, judgement in zip(records, full.judgements, strict=True):
        _require_finite(record)
        identity = str(record["candidate_identity"])
        components = record["judgement"]["components"]
        unknown_components = set(components) - set(
            PRODUCTION_JUDGEMENT_COMPONENT_ORDER
        )
        if unknown_components:
            raise RankingDecisionAuditError("unregistered_judgement_component")
        reason = str(record["judgement"]["category_reason"])
        if (
            set(components) != set(PRODUCTION_JUDGEMENT_COMPONENT_ORDER)
            and reason != "missing_title_and_abstract"
        ):
            raise RankingDecisionAuditError("registered_judgement_component_missing")
        component_sum = math.fsum(float(value) for value in components.values())
        if abs(component_sum - float(judgement.score)) > tolerance:
            raise RankingDecisionAuditError("judgement_component_sum_mismatch")
        if float(record["judgement"]["total_score"]) != judgement.score:
            raise RankingDecisionAuditError("judgement_score_record_mismatch")
        if str(record["judgement"]["category"]) != judgement.category:
            raise RankingDecisionAuditError("judgement_category_record_mismatch")
        if reason.startswith("score_threshold:") and production_category_from_score(
            float(judgement.score)
        ) != judgement.category:
            raise RankingDecisionAuditError("category_threshold_mismatch")
        ranked = ranked_by_id[identity]
        if record["ranking"]["score_breakdown"] != ranked.score_breakdown.model_dump(mode="json"):
            raise RankingDecisionAuditError("rerank_breakdown_mismatch")
        if len(record["ranking"]["sort_key"]) != 7:
            raise RankingDecisionAuditError("ranking_key_missing")
    ordered = sorted(records, key=lambda item: tuple(item["ranking"]["sort_key"]))
    reconstructed = [str(item["candidate_identity"]) for item in ordered]
    expected = registry.labels([item.paper for item in full.ranked])
    if reconstructed != expected:
        raise RankingDecisionAuditError("rerank_position_reconstruction_mismatch")
    for position, record in enumerate(ordered, start=1):
        if int(record["ranking"]["reranked_position"]) != position:
            raise RankingDecisionAuditError("reranked_position_mismatch")
        expected_window = position <= int(protocol["reconstruction"]["top_k"])
        expected_gate = record["judgement"]["category"] in _RETURN_CATEGORIES
        if bool(record["top20"]["within_rank_window"]) != expected_window:
            raise RankingDecisionAuditError("top20_rank_window_mismatch")
        if bool(record["top20"]["category_gate_passed"]) != expected_gate:
            raise RankingDecisionAuditError("top20_category_gate_mismatch")
        if bool(record["top20"]["final_returned"]) != (
            expected_window and expected_gate
        ):
            raise RankingDecisionAuditError("top20_final_state_mismatch")
    selected = [
        str(item["candidate_identity"])
        for item in ordered
        if item["top20"]["within_rank_window"]
        and item["top20"]["category_gate_passed"]
    ]
    if selected != registry.labels([item.paper for item in full.returned]):
        raise RankingDecisionAuditError("final_returned_reconstruction_mismatch")
    if len(selected) != len(set(selected)):
        raise RankingDecisionAuditError("final_returned_duplicate_identity")


def input_permutation_diagnostic(
    analysis: QueryAnalysis,
    full: VariantResult,
    registry: IdentityRegistry,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reversed_variant = rank_variant(
        analysis,
        [paper.model_copy(deep=True) for paper in reversed(full.candidates)],
        top_k=20,
    )
    original = registry.labels([item.paper for item in full.ranked])
    permuted = registry.labels([item.paper for item in reversed_variant.ranked])
    original_positions = {value: index for index, value in enumerate(original)}
    changed = [
        value for index, value in enumerate(permuted) if original_positions[value] != index
    ]
    tie_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in records:
        tie_groups[tuple(record["ranking"]["tie_key"])].append(
            str(record["candidate_identity"])
        )
    tied = {value for group in tie_groups.values() if len(group) > 1 for value in group}
    if set(changed) - tied:
        raise RankingDecisionAuditError("input_permutation_changes_non_tied_order")
    return {
        "permutation": "reverse_candidate_input",
        "changed_candidate_count": len(changed),
        "changed_only_within_registered_tie_groups": True,
        "input_order_sensitive_tie_group_count": sum(
            len(group) > 1 and any(value in changed for value in group)
            for group in tie_groups.values()
        ),
    }


def aggregate_analysis(
    cases: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    observed_snapshot_key_count: int,
) -> dict[str, Any]:
    categories = Counter(str(item["judgement"]["category"]) for item in decisions)
    category_order = protocol["decision_catalog"]["category_order"]
    source_rows = {
        source: [item for item in decisions if source in item["source_provenance"]]
        for source in _SOURCE_ORDER
    }
    tie_groups: defaultdict[tuple[str, tuple[Any, ...]], int] = defaultdict(int)
    for item in decisions:
        tie_groups[(str(item["query_identity"]), tuple(item["ranking"]["tie_key"]))] += 1
    movement_values = [float(item["ranking"]["position_delta"]) for item in decisions]
    low_without_failures = [
        item
        for item in decisions
        if item["judgement"]["category"] in {"weakly_relevant", "irrelevant"}
        and not item["judgement"]["failed_constraint_dimensions"]
    ]
    cross_category_overtakes = 0
    by_query: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in decisions:
        by_query[str(item["query_identity"])].append(item)
    for values in by_query.values():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if left["judgement"]["category"] == right["judgement"]["category"]:
                    continue
                if int(left["ranking"]["reranked_position"]) > int(
                    right["ranking"]["reranked_position"]
                ):
                    cross_category_overtakes += 1

    component_names = protocol["decision_catalog"]["judgement_components"]
    components = {
        name: summarize_distribution(
            [
                float(item["judgement"]["components"].get(name, 0.0))
                for item in decisions
            ]
        )
        for name in component_names
    }
    threshold_margins = {
        category: {
            "lower_margin": summarize_distribution(
                [
                    float(item["judgement"]["threshold_margin"]["lower_margin"])
                    for item in decisions
                    if item["judgement"]["category"] == category
                    and item["judgement"]["threshold_margin"]["lower_margin"]
                    is not None
                ]
            ),
            "upper_margin": summarize_distribution(
                [
                    float(item["judgement"]["threshold_margin"]["upper_margin"])
                    for item in decisions
                    if item["judgement"]["category"] == category
                    and item["judgement"]["threshold_margin"]["upper_margin"]
                    is not None
                ]
            ),
        }
        for category in category_order
    }
    source_diagnostics = {
        source: source_summary(values) for source, values in source_rows.items()
    }
    missing_fields: Counter[str] = Counter()
    for item in decisions:
        for field, lineage in item["field_lineage"]["fields"].items():
            for state in ("missing", "null", "empty"):
                missing_fields[f"{field}:{state}"] += int(
                    lineage["candidate_state_counts"].get(state, 0)
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "implementation_base_commit": protocol["implementation_base_commit"],
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256": protocol_sha256,
        "inputs": dict(sorted(input_hashes.items())),
        "closure": {
            "record_case_count": len(cases) + len(excluded),
            "included_main_case_count": len(cases),
            "excluded_no_successful_source_count": len(excluded),
            "candidate_decision_count": len(decisions),
            "observed_snapshot_key_count": observed_snapshot_key_count,
            "reconstruction_exact_case_count": sum(
                bool(case.get("reconstruction")) for case in cases
            ),
            "component_count": len(
                {str(case["component_identity"]) for case in cases}
            ),
        },
        "production_catalog": {
            "judgement": production_judgement_decision_catalog(),
            "rerank": production_ranking_decision_catalog(),
            "top20": protocol["decision_catalog"]["result_policy"],
        },
        "category_counts": {
            category: categories[category] for category in category_order
        },
        "threshold_margin_distributions": threshold_margins,
        "component_contribution_distributions": components,
        "rerank_component_contribution_distributions": (
            rerank_component_contribution_distributions(decisions)
        ),
        "ranking": {
            "position_delta": summarize_distribution(movement_values),
            "promoted_count": sum(value > 0 for value in movement_values),
            "unchanged_count": sum(value == 0 for value in movement_values),
            "demoted_count": sum(value < 0 for value in movement_values),
            "cross_category_overtake_count": cross_category_overtakes,
            "tie_group_count": sum(value > 1 for value in tie_groups.values()),
            "tied_candidate_count": sum(
                value for value in tie_groups.values() if value > 1
            ),
            "maximum_tie_group_size": max(tie_groups.values(), default=0),
            "input_order_sensitive_tie_group_count": sum(
                int(case["input_permutation"]["input_order_sensitive_tie_group_count"])
                for case in cases
            ),
        },
        "top20": {
            "returned_identity_count": sum(
                bool(item["top20"]["final_returned"]) for item in decisions
            ),
            "reason_counts": dict(
                sorted(Counter(str(item["top20"]["reason"]) for item in decisions).items())
            ),
            "rank_margin_distribution": summarize_distribution(
                [float(item["top20"]["rank_margin"]) for item in decisions]
            ),
        },
        "low_category_without_failed_dimensions": {
            "candidate_count": len(low_without_failures),
            "category_counts": dict(
                sorted(
                    Counter(
                        str(item["judgement"]["category"])
                        for item in low_without_failures
                    ).items()
                )
            ),
            "score_distribution": summarize_distribution(
                [float(item["judgement"]["total_score"]) for item in low_without_failures]
            ),
            "component_distributions": {
                name: summarize_distribution(
                    [
                        float(item["judgement"]["components"].get(name, 0.0))
                        for item in low_without_failures
                    ]
                )
                for name in component_names
            },
        },
        "source_diagnostics": source_diagnostics,
        "arxiv_and_openalex_deep_dive": {
            source: source_diagnostics[source] for source in ("arxiv", "openalex")
        },
        "field_missing_state_counts": dict(sorted(missing_fields.items())),
        "semantic_risks": {
            "unregistered_component_count": 0,
            "component_sum_violation_count": 0,
            "category_boundary_violation_count": 0,
            "missing_sort_key_count": 0,
            "nan_or_infinity_count": 0,
            "duplicate_ranked_identity_count": 0,
            "top20_truncation_violation_count": 0,
            "input_order_sensitive_tie_group_count": sum(
                int(case["input_permutation"]["input_order_sensitive_tie_group_count"])
                for case in cases
            ),
        },
        "execution": {
            "gold_or_qrels_loaded": False,
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
            "full1000_inference_performed": False,
        },
        "interpretation": {
            "scope": "production_ranking_decision_explanation_only",
            "relevance_claim_permitted": False,
            "precision_recall_f1_or_official_score": False,
            "warnings": list(protocol["warnings"]),
        },
    }


def source_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_identity_attribution_count": len(values),
        "category_counts": dict(
            sorted(Counter(str(item["judgement"]["category"]) for item in values).items())
        ),
        "category_reason_counts": dict(
            sorted(
                Counter(
                    str(item["judgement"]["category_reason"]) for item in values
                ).items()
            )
        ),
        "judgement_score_distribution": summarize_distribution(
            [float(item["judgement"]["total_score"]) for item in values]
        ),
        "rerank_final_score_distribution": summarize_distribution(
            [
                float(item["ranking"]["score_breakdown"]["final_score"])
                for item in values
            ]
        ),
        "within_top20_rank_window_count": sum(
            bool(item["top20"]["within_rank_window"]) for item in values
        ),
        "category_gate_passed_count": sum(
            bool(item["top20"]["category_gate_passed"]) for item in values
        ),
        "final_returned_attribution_count": sum(
            bool(item["top20"]["final_returned"]) for item in values
        ),
        "top20_reason_counts": dict(
            sorted(Counter(str(item["top20"]["reason"]) for item in values).items())
        ),
        "position_delta_distribution": summarize_distribution(
            [float(item["ranking"]["position_delta"]) for item in values]
        ),
        "failed_constraint_dimension_count": sum(
            len(item["judgement"]["failed_constraint_dimensions"])
            for item in values
        ),
        "failed_constraint_dimension_counts": dict(
            sorted(
                Counter(
                    str(dimension)
                    for item in values
                    for dimension in item["judgement"][
                        "failed_constraint_dimensions"
                    ]
                ).items()
            )
        ),
        "judgement_component_distributions": {
            name: summarize_distribution(
                [
                    float(item["judgement"]["components"].get(name, 0.0))
                    for item in values
                ]
            )
            for name in PRODUCTION_JUDGEMENT_COMPONENT_ORDER
        },
        "rerank_component_contribution_distributions": (
            rerank_component_contribution_distributions(values)
        ),
    }


def rerank_component_contribution_distributions(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    components = ("relevance", "authority", "timeliness", "metadata")
    return {
        name: summarize_distribution(
            [
                float(item["ranking"]["score_breakdown"][f"{name}_score"])
                * float(item["ranking"]["score_breakdown"][f"{name}_weight"])
                * float(item["ranking"]["score_breakdown"]["category_multiplier"])
                for item in values
            ]
        )
        for name in components
    }


def write_analysis(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protocol_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate": root / "aggregate.json",
        "candidate_decisions": root / "candidate_decisions.jsonl",
        "case_diagnostics": root / "case_diagnostics.jsonl",
        "protocol": root / "protocol.json",
    }
    _write_json(paths["aggregate"], aggregate)
    _write_jsonl(
        paths["candidate_decisions"],
        sorted(
            decisions,
            key=lambda item: (
                int(item["case_order"]),
                int(item["candidate_order"]),
            ),
        ),
    )
    _write_jsonl(
        paths["case_diagnostics"],
        sorted(cases, key=lambda item: int(item["case_order"])),
    )
    protocol = _read_json(Path(protocol_path).expanduser().resolve())
    _write_json(paths["protocol"], protocol)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": aggregate["status"],
        "files": {
            name: {"path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in sorted(paths.items())
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_analysis(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("analysis") != CONTRACT_VERSION:
        raise RankingDecisionAuditError("manifest_contract_mismatch")
    for value in manifest.get("files", {}).values():
        path = root / str(value["path"])
        if not path.is_file() or path.stat().st_size != int(value["size"]):
            raise RankingDecisionAuditError("output_missing_or_size_drift")
        if _sha256(path) != str(value["sha256"]):
            raise RankingDecisionAuditError("output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") != "completed":
        raise RankingDecisionAuditError("analysis_not_completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
    }


def _require_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RankingDecisionAuditError(f"non_finite_number:{path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")


def _rounded(value: float) -> float:
    return round(float(value), 8)


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise RankingDecisionAuditError("network_attempt_detected")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
    ):
        yield


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
