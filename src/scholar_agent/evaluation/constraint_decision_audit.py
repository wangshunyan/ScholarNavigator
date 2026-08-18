"""Gold-free explicit-constraint decision audit for frozen Record160 Replay.

The audit reuses production candidate construction, judgement, reranking, and
selection.  It records opaque field-level decisions and read-only one-field
leave-outs; it never loads evaluator labels or interprets selectivity as
relevance quality.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.agents.judgement import (
    PRODUCTION_CONSTRAINT_ORDER,
    production_constraint_catalog,
    trace_constraint_decisions,
)
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
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
CONTRACT_VERSION = "constraint_decision_audit_v1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RETURN_CATEGORIES = frozenset({"highly_relevant", "partially_relevant"})
_FEATURE_KEYS = {
    "exclude_terms": "excluded_terms",
    "must_include_terms": "must_have",
    "datasets": "dataset",
    "paper_types": "paper_type",
    "venues": "venue",
    "time_range": "time",
}


class ConstraintDecisionAuditError(RuntimeError):
    """The constraint trace or frozen reconstruction violated its contract."""


class ConstraintDecisionNotEligible(ConstraintDecisionAuditError):
    """Frozen evidence cannot support the preregistered decision audit."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path).expanduser().resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise ConstraintDecisionAuditError("unsupported_protocol")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise ConstraintDecisionAuditError("offline_protocol_drift")
    expected_order = list(PRODUCTION_CONSTRAINT_ORDER)
    catalog = value.get("constraint_catalog") or {}
    if catalog.get("field_order") != expected_order:
        raise ConstraintDecisionAuditError("constraint_order_drift")
    production = production_constraint_catalog()
    if catalog.get("predicate_version") != production["predicate_version"]:
        raise ConstraintDecisionAuditError("predicate_version_drift")
    for name in expected_order:
        frozen = catalog.get("fields", {}).get(name) or {}
        live = production["fields"][name]
        if frozen.get("evidence_fields") != live["evidence_fields"]:
            raise ConstraintDecisionAuditError(f"constraint_field_source_drift:{name}")
        if frozen.get("enforcement") != live["enforcement"]:
            raise ConstraintDecisionAuditError(f"constraint_enforcement_drift:{name}")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_constraint_result",
    ]:
        raise ConstraintDecisionAuditError("selection_contract_drift")
    return value


def run_constraint_decision_audit(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen audit with network and evaluator access disabled."""

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
        raise ConstraintDecisionNotEligible("snapshot_tree_hash_drift")
    if sum(path.is_file() for path in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise ConstraintDecisionNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, protocol)
    rows = _read_rows(results_path)
    if len(rows) != int(protocol["analysis_population"]["record_case_count"]):
        raise ConstraintDecisionNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[: len(rows)]:
        raise ConstraintDecisionNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise ConstraintDecisionNotEligible("missing_component_assignment")

    attempts = {"network": 0}
    store = SnapshotStore(snapshot_dir)
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
    if len(observed_keys) != int(frozen["observed_snapshot_key_count"]):
        raise ConstraintDecisionNotEligible("observed_snapshot_key_count_drift")
    if _tree_sha256(snapshot_dir) != before_tree:
        raise ConstraintDecisionAuditError("snapshot_tree_changed")
    if attempts["network"]:
        raise ConstraintDecisionAuditError("network_attempt_detected")
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
        raise ConstraintDecisionNotEligible("required_frozen_stage_missing")
    sources = [str(value) for value in config["sources"]]
    requests = audit_retrieval_requests(
        stages["initial_retrieval"], config=config, store=store, sources=sources
    )
    successful_source_count = sum(
        int(requests.source_records[source]["snapshot_success_count"]) > 0
        for source in sources
    )
    query_identity = _opaque_identity("query", str(row["case_id"]))
    component_identity = _opaque_identity("component", str(component_id))
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": (
            "included_main_analysis"
            if successful_source_count
            else "excluded_no_successful_source"
        ),
        "case_order": case_order,
        "query_identity": query_identity,
        "component_identity": component_identity,
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
    candidates = deduplicate_papers(raw)
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
    source_sets = {
        source: set(registry.labels(deduplicate_papers(requests.papers_by_source[source])))
        for source in sources
    }
    judgement_by_id = {
        identity: judgement
        for identity, judgement in zip(candidate_ids, full.judgements, strict=True)
    }
    if len(judgement_by_id) != len(candidate_ids):
        raise ConstraintDecisionAuditError("candidate_identity_not_unique_after_dedup")
    rank_by_id = {
        registry.label(item.paper): item for item in full.ranked
    }
    returned_ids = registry.labels([item.paper for item in full.returned])
    returned_set = set(returned_ids)
    candidate_rows: list[dict[str, Any]] = []
    for candidate_order, (identity, paper) in enumerate(
        zip(candidate_ids, full.candidates, strict=True)
    ):
        judgement = judgement_by_id[identity]
        ranked = rank_by_id[identity]
        trace = trace_constraint_decisions(
            analysis, paper, config=CURRENT_RULES_CONFIG
        )
        validate_trace(trace, judgement.feature_vector)
        failures = [
            item["constraint"] for item in trace if item["status"] == "failed"
        ]
        unknown = [
            item["constraint"] for item in trace if item["status"] == "unknown"
        ]
        retained = judgement.category in _RETURN_CATEGORIES
        in_top20 = ranked.rank <= int(protocol["reconstruction"]["top_k"])
        selected = identity in returned_set
        if selected != (retained and in_top20):
            raise ConstraintDecisionAuditError("final_state_contradiction")
        candidate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_identity": query_identity,
                "component_identity": component_identity,
                "case_order": case_order,
                "candidate_identity": identity,
                "candidate_order": candidate_order,
                "source_provenance": [
                    source for source in sources if identity in source_sets[source]
                ],
                "constraints": trace,
                "failed_constraints": failures,
                "unknown_constraints": unknown,
                "production": {
                    "judgement_score": judgement.score,
                    "judgement_category": judgement.category,
                    "category_reason": (
                        judgement.feature_vector.category_reason
                        if judgement.feature_vector is not None
                        else "feature_vector_missing"
                    ),
                    "hard_constraint_failures": (
                        list(judgement.feature_vector.hard_constraint_failures)
                        if judgement.feature_vector is not None
                        else []
                    ),
                    "retained_by_result_policy": retained,
                    "rank": ranked.rank,
                    "within_top20_rank_window": in_top20,
                    "selected_top20": selected,
                },
            }
        )

    validate_candidate_records(candidate_rows, full, registry)
    reordered = rank_variant(
        reorder_query_constraints(analysis),
        [paper.model_copy(deep=True) for paper in full.candidates],
        top_k=int(protocol["reconstruction"]["top_k"]),
    )
    assert_variant_equivalent(full, reordered, registry)
    reordered_traces = [
        trace_constraint_decisions(
            reorder_query_constraints(analysis), paper, config=CURRENT_RULES_CONFIG
        )
        for paper in full.candidates
    ]
    if reordered_traces != [item["constraints"] for item in candidate_rows]:
        raise ConstraintDecisionAuditError("constraint_trace_order_dependency")

    shadow: dict[str, Any] = {}
    baseline_survivors = {
        registry.label(item.paper)
        for item in full.judgements
        if item.category in _RETURN_CATEGORIES
    }
    baseline_top20 = set(returned_ids)
    active_fields = active_constraint_fields(analysis.constraints)
    for field in PRODUCTION_CONSTRAINT_ORDER:
        variant = rank_variant(
            remove_constraint_field(analysis, field),
            [paper.model_copy(deep=True) for paper in full.candidates],
            top_k=int(protocol["reconstruction"]["top_k"]),
        )
        if registry.labels(variant.candidates) != candidate_ids:
            raise ConstraintDecisionAuditError("shadow_candidate_set_changed")
        survivor_ids = {
            registry.label(item.paper)
            for item in variant.judgements
            if item.category in _RETURN_CATEGORIES
        }
        top20_ids = set(registry.labels([item.paper for item in variant.returned]))
        shadow[field] = {
            "active_in_query": field in active_fields,
            "restored_survivor_identity_count": len(
                survivor_ids - baseline_survivors
            ),
            "lost_survivor_identity_count": len(baseline_survivors - survivor_ids),
            "top20_fill_delta": len(variant.returned) - len(full.returned),
            "top20_identity_added_count": len(top20_ids - baseline_top20),
            "top20_identity_removed_count": len(baseline_top20 - top20_ids),
            "restored_survivor_identity_ids": sorted(
                survivor_ids - baseline_survivors
            ),
            "top20_identity_added_ids": sorted(top20_ids - baseline_top20),
            "top20_identity_removed_ids": sorted(baseline_top20 - top20_ids),
        }

    return (
        {
            **base,
            "candidate_count": len(candidate_rows),
            "constraint_survivor_count": sum(
                item["production"]["retained_by_result_policy"]
                for item in candidate_rows
            ),
            "top20_fill_count": len(full.returned),
            "active_constraint_fields": sorted(active_fields),
            "constraint_order_invariant": True,
            "reconstruction": {
                "initial_deduplicated_exact": True,
                "initial_judged_exact": True,
                "initial_reranked_exact": True,
                "final_returned_exact": True,
                "decision_record_exact": True,
                "digest": _stable_json_sha256(
                    {
                        "candidates": candidate_ids,
                        "survivors": [
                            item["candidate_identity"]
                            for item in candidate_rows
                            if item["production"]["retained_by_result_policy"]
                        ],
                        "top20": returned_ids,
                    }
                ),
            },
            "shadow_leave_one_constraint_out": shadow,
        },
        candidate_rows,
        requests.observed_keys,
    )


def validate_trace(trace: Sequence[Mapping[str, Any]], feature: Any) -> None:
    if [str(item.get("constraint")) for item in trace] != list(
        PRODUCTION_CONSTRAINT_ORDER
    ):
        raise ConstraintDecisionAuditError("constraint_record_missing_or_reordered")
    allowed_states = {"passed", "failed", "not_applicable", "unknown"}
    catalog = production_constraint_catalog()["fields"]
    for item in trace:
        name = str(item["constraint"])
        if item.get("status") not in allowed_states:
            raise ConstraintDecisionAuditError("constraint_state_invalid")
        expected_fields = set(catalog[name]["evidence_fields"])
        referenced = {str(value["field"]) for value in item["field_lineage"]}
        if referenced - expected_fields:
            raise ConstraintDecisionAuditError(f"constraint_field_reference_invalid:{name}")
        if not item.get("reason_code"):
            raise ConstraintDecisionAuditError(f"constraint_reason_missing:{name}")
        if item["status"] == "unknown" and not any(
            value["state"] in {"null", "empty"} for value in item["field_lineage"]
        ):
            raise ConstraintDecisionAuditError(
                f"constraint_unknown_without_missing_field:{name}"
            )
    if feature is None:
        raise ConstraintDecisionAuditError("judgement_feature_vector_missing")
    by_name = {str(item["constraint"]): item for item in trace}
    for name, feature_key in _FEATURE_KEYS.items():
        if feature_key not in feature.constraint_results:
            continue
        if by_name[name]["production_predicate_result"] is not feature.constraint_results[
            feature_key
        ]:
            raise ConstraintDecisionAuditError(
                f"constraint_feature_result_mismatch:{name}"
            )


def validate_candidate_records(
    records: Sequence[Mapping[str, Any]],
    full: VariantResult,
    registry: IdentityRegistry,
) -> None:
    if len(records) != len(full.candidates):
        raise ConstraintDecisionAuditError("decision_record_count_mismatch")
    identities = [str(item["candidate_identity"]) for item in records]
    if len(set(identities)) != len(identities):
        raise ConstraintDecisionAuditError("decision_record_duplicate_identity")
    if identities != registry.labels(full.candidates):
        raise ConstraintDecisionAuditError("decision_record_candidate_order_mismatch")
    reconstructed_survivors = [
        item["candidate_identity"]
        for item in records
        if item["production"]["retained_by_result_policy"]
    ]
    expected_survivors = [
        registry.label(item.paper)
        for item in full.judgements
        if item.category in _RETURN_CATEGORIES
    ]
    if reconstructed_survivors != expected_survivors:
        raise ConstraintDecisionAuditError("constraint_survivor_reconstruction_mismatch")
    reconstructed_top20 = [
        item["candidate_identity"]
        for item in sorted(records, key=lambda value: int(value["production"]["rank"]))
        if item["production"]["within_top20_rank_window"]
        and item["production"]["retained_by_result_policy"]
    ]
    expected_top20 = registry.labels([item.paper for item in full.returned])
    if reconstructed_top20 != expected_top20:
        raise ConstraintDecisionAuditError("top20_reconstruction_mismatch")


def reorder_query_constraints(analysis: QueryAnalysis) -> QueryAnalysis:
    constraints = analysis.constraints
    update: dict[str, Any] = {
        name: list(reversed(getattr(constraints, name)))
        for name in (
            "venues",
            "methods",
            "datasets",
            "domains",
            "must_include_terms",
            "exclude_terms",
            "paper_types",
        )
    }
    update["explicit_fields"] = list(reversed(constraints.explicit_fields))
    return analysis.model_copy(
        update={"constraints": constraints.model_copy(update=update)}, deep=True
    )


def remove_constraint_field(analysis: QueryAnalysis, field: str) -> QueryAnalysis:
    if field not in PRODUCTION_CONSTRAINT_ORDER:
        raise ValueError(f"unsupported constraint field:{field}")
    constraints = analysis.constraints
    value: Any = None if field == "time_range" else []
    update = {
        field: value,
        "explicit_fields": [
            item for item in constraints.explicit_fields if item != field
        ],
    }
    return analysis.model_copy(
        update={"constraints": constraints.model_copy(update=update)}, deep=True
    )


def active_constraint_fields(constraints: QueryConstraint) -> set[str]:
    active = {
        field
        for field in PRODUCTION_CONSTRAINT_ORDER
        if field != "time_range" and bool(getattr(constraints, field))
    }
    if constraints.time_range is not None:
        active.add("time_range")
    return active


def assert_variant_equivalent(
    left: VariantResult, right: VariantResult, registry: IdentityRegistry
) -> None:
    left_judgements = [
        (registry.label(item.paper), item.score, item.category)
        for item in left.judgements
    ]
    right_judgements = [
        (registry.label(item.paper), item.score, item.category)
        for item in right.judgements
    ]
    left_ranked = [
        (registry.label(item.paper), item.rank, item.final_score, item.category)
        for item in left.ranked
    ]
    right_ranked = [
        (registry.label(item.paper), item.rank, item.final_score, item.category)
        for item in right.ranked
    ]
    if left_judgements != right_judgements:
        raise ConstraintDecisionAuditError("constraint_value_order_changes_judgement")
    if left_ranked != right_ranked:
        raise ConstraintDecisionAuditError("constraint_value_order_changes_ranking")
    if registry.labels([item.paper for item in left.returned]) != registry.labels(
        [item.paper for item in right.returned]
    ):
        raise ConstraintDecisionAuditError("constraint_value_order_changes_selection")


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
    constraints: dict[str, Any] = {}
    for field in PRODUCTION_CONSTRAINT_ORDER:
        field_rows = [
            next(
                item
                for item in decision["constraints"]
                if item["constraint"] == field
            )
            for decision in decisions
        ]
        states = Counter(str(item["status"]) for item in field_rows)
        unique_failure = sum(
            item["failed_constraints"] == [field] for item in decisions
        )
        shadow_rows = [
            case["shadow_leave_one_constraint_out"][field] for case in cases
        ]
        per_case_failed = [
            sum(
                field in item["failed_constraints"]
                for item in decisions
                if item["query_identity"] == case["query_identity"]
            )
            for case in cases
        ]
        constraints[field] = {
            "active_candidate_count": sum(
                row["status"] != "not_applicable" for row in field_rows
            ),
            "explicit_candidate_count": sum(bool(row["explicit"]) for row in field_rows),
            "status_counts": {
                state: states[state]
                for state in ("passed", "failed", "not_applicable", "unknown")
            },
            "unique_failure_candidate_count": unique_failure,
            "failed_and_filtered_candidate_count": sum(
                row["status"] == "failed"
                and not decision["production"]["retained_by_result_policy"]
                for row, decision in zip(field_rows, decisions, strict=True)
            ),
            "failed_but_retained_candidate_count": sum(
                row["status"] == "failed"
                and decision["production"]["retained_by_result_policy"]
                for row, decision in zip(field_rows, decisions, strict=True)
            ),
            "unknown_with_false_production_predicate_count": sum(
                row["status"] == "unknown"
                and row["production_predicate_result"] is False
                for row in field_rows
            ),
            "candidate_failure_cluster_summary": cluster_summary(
                cases,
                per_case_failed,
                protocol,
                stream=f"{field}:candidate_failure_count",
            ),
            "shadow_leave_out": {
                "active_query_count": sum(row["active_in_query"] for row in shadow_rows),
                "restored_survivor_identity_count": sum(
                    int(row["restored_survivor_identity_count"])
                    for row in shadow_rows
                ),
                "lost_survivor_identity_count": sum(
                    int(row["lost_survivor_identity_count"]) for row in shadow_rows
                ),
                "top20_fill_delta_total": sum(
                    int(row["top20_fill_delta"]) for row in shadow_rows
                ),
                "top20_identity_added_count": sum(
                    int(row["top20_identity_added_count"]) for row in shadow_rows
                ),
                "top20_identity_removed_count": sum(
                    int(row["top20_identity_removed_count"]) for row in shadow_rows
                ),
                "restored_survivor_cluster_summary": cluster_summary(
                    cases,
                    [float(row["restored_survivor_identity_count"]) for row in shadow_rows],
                    protocol,
                    stream=f"{field}:restored_survivors",
                ),
                "top20_fill_delta_cluster_summary": cluster_summary(
                    cases,
                    [float(row["top20_fill_delta"]) for row in shadow_rows],
                    protocol,
                    stream=f"{field}:top20_fill_delta",
                ),
            },
        }

    combinations = Counter(
        "+".join(item["failed_constraints"])
        if item["failed_constraints"]
        else "none"
        for item in decisions
    )
    source_strata: dict[str, Any] = {}
    for source in ("openalex", "arxiv", "semantic_scholar", "pubmed"):
        selected = [item for item in decisions if source in item["source_provenance"]]
        categories = Counter(
            str(item["production"]["judgement_category"]) for item in selected
        )
        category_reasons = Counter(
            str(item["production"]["category_reason"]) for item in selected
        )
        source_strata[source] = {
            "candidate_identity_count": len(selected),
            "filtered_candidate_count": sum(
                not item["production"]["retained_by_result_policy"]
                for item in selected
            ),
            "filtered_with_failed_dimension_count": sum(
                not item["production"]["retained_by_result_policy"]
                and bool(item["failed_constraints"])
                for item in selected
            ),
            "filtered_without_failed_dimension_count": sum(
                not item["production"]["retained_by_result_policy"]
                and not item["failed_constraints"]
                for item in selected
            ),
            "constraint_survivor_count": sum(
                item["production"]["retained_by_result_policy"] for item in selected
            ),
            "top20_selected_count": sum(
                item["production"]["selected_top20"] for item in selected
            ),
            "judgement_category_counts": dict(sorted(categories.items())),
            "category_reason_counts": dict(sorted(category_reasons.items())),
            "judgement_score_distribution": summarize_distribution(
                [float(item["production"]["judgement_score"]) for item in selected]
            ),
            "constraint_status_counts": {
                field: dict(
                    sorted(
                        Counter(
                            next(
                                value["status"]
                                for value in item["constraints"]
                                if value["constraint"] == field
                            )
                            for item in selected
                        ).items()
                    )
                )
                for field in PRODUCTION_CONSTRAINT_ORDER
            },
            "failure_combination_counts": dict(
                sorted(
                    Counter(
                        "+".join(item["failed_constraints"])
                        if item["failed_constraints"]
                        else "none"
                        for item in selected
                    ).items()
                )
            ),
        }
    query_structure = query_structure_strata(cases, decisions)
    missing_fields = missing_field_strata(decisions)
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
            "reconstruction_exact_case_count": sum(
                bool(case.get("reconstruction")) for case in cases
            ),
            "decision_record_candidate_count": len(decisions),
            "decision_record_constraint_count": len(decisions)
            * len(PRODUCTION_CONSTRAINT_ORDER),
            "observed_snapshot_key_count": observed_snapshot_key_count,
            "constraint_order_invariant_case_count": sum(
                bool(case["constraint_order_invariant"]) for case in cases
            ),
            "component_count": len(
                {str(case["component_identity"]) for case in cases}
            ),
        },
        "production_constraint_catalog": production_constraint_catalog(),
        "constraints": constraints,
        "failure_combination_counts": dict(sorted(combinations.items())),
        "production_selection": {
            "constraint_survivor_identity_count": sum(
                item["production"]["retained_by_result_policy"]
                for item in decisions
            ),
            "filtered_identity_count": sum(
                not item["production"]["retained_by_result_policy"]
                for item in decisions
            ),
            "top20_selected_identity_attribution_count": sum(
                item["production"]["selected_top20"] for item in decisions
            ),
            "filtered_with_failed_dimension_count": sum(
                not item["production"]["retained_by_result_policy"]
                and bool(item["failed_constraints"])
                for item in decisions
            ),
            "filtered_without_failed_dimension_count": sum(
                not item["production"]["retained_by_result_policy"]
                and not item["failed_constraints"]
                for item in decisions
            ),
        },
        "source_strata": source_strata,
        "query_structure_strata": query_structure,
        "missing_field_strata": missing_fields,
        "semantic_audit": {
            "field_reference_violation_count": 0,
            "reason_omission_count": 0,
            "final_state_contradiction_count": 0,
            "constraint_order_dependency_count": 0,
            "domains_with_values_not_consumed_by_candidate_predicate_count": sum(
                next(
                    value["constraint"] == "domains"
                    and value["expected_value_count"] > 0
                    and value["status"] == "not_applicable"
                    for value in item["constraints"]
                    if value["constraint"] == "domains"
                )
                for item in decisions
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
            "scope": "constraint_selectivity_and_interaction_only",
            "relevance_claim_permitted": False,
            "precision_recall_or_official_score": False,
            "shadow_is_deployable_recommendation": False,
            "warnings": list(protocol["warnings"]),
        },
    }


def query_structure_strata(
    cases: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    dimensions = {
        "length_bucket": ["0_80", "81_160", "161_320", "321_plus"],
        "has_quote": [False, True],
        "has_boolean_operator": [False, True],
        "has_year": [False, True],
        "unicode_class": ["ascii_only", "non_ascii_letter", "non_ascii_nonletter"],
    }
    by_query: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in decisions:
        by_query[str(item["query_identity"])].append(item)
    result: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        strata: dict[str, Any] = {}
        for value in values:
            selected_cases = [
                case for case in cases if case["query_structure"][dimension] == value
            ]
            selected_decisions = [
                item
                for case in selected_cases
                for item in by_query[str(case["query_identity"])]
            ]
            strata[str(value).lower()] = {
                "query_count": len(selected_cases),
                "candidate_count": len(selected_decisions),
                "filtered_candidate_count": sum(
                    not item["production"]["retained_by_result_policy"]
                    for item in selected_decisions
                ),
                "constraint_failure_counts": {
                    field: sum(
                        field in item["failed_constraints"]
                        for item in selected_decisions
                    )
                    for field in PRODUCTION_CONSTRAINT_ORDER
                },
                "constraint_unknown_counts": {
                    field: sum(
                        field in item["unknown_constraints"]
                        for item in selected_decisions
                    )
                    for field in PRODUCTION_CONSTRAINT_ORDER
                },
            }
        result[dimension] = strata
    return result


def missing_field_strata(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for decision in decisions:
        for constraint in decision["constraints"]:
            for field in constraint["field_lineage"]:
                counts[
                    (
                        str(constraint["constraint"]),
                        str(field["field"]),
                        str(field["state"]),
                    )
                ] += 1
    return {
        constraint: {
            field: {
                state: counts[(constraint, field, state)]
                for state in ("null", "empty", "present")
            }
            for field in production_constraint_catalog()["fields"][constraint][
                "evidence_fields"
            ]
        }
        for constraint in PRODUCTION_CONSTRAINT_ORDER
    }


def cluster_summary(
    cases: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    protocol: Mapping[str, Any],
    *,
    stream: str,
) -> dict[str, Any]:
    if len(cases) != len(values):
        raise ValueError("cluster summary inputs are not aligned")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for case, value in zip(cases, values, strict=True):
        grouped[str(case["component_identity"])].append(float(value))
    component_values = [statistics.fmean(grouped[key]) for key in sorted(grouped)]
    if not component_values:
        return {
            "query_count": 0,
            "component_count": 0,
            "mean": None,
            "median": None,
            "confidence_interval_95": [None, None],
        }
    settings = protocol["statistical_method"]
    seed = int(settings["seed"]) ^ int.from_bytes(
        hashlib.sha256(stream.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed)
    iterations = int(settings["iterations"])
    bootstrap = [
        statistics.fmean(
            component_values[rng.randrange(len(component_values))]
            for _ in component_values
        )
        for _ in range(iterations)
    ]
    return {
        "query_count": len(cases),
        "component_count": len(component_values),
        "mean": _rounded(statistics.fmean(component_values)),
        "median": _rounded(statistics.median(component_values)),
        "confidence_interval_95": [
            _rounded(_percentile(bootstrap, 0.025)),
            _rounded(_percentile(bootstrap, 0.975)),
        ],
    }


def summarize_distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "q1": None,
            "q3": None,
            "maximum": None,
        }
    return {
        "count": len(ordered),
        "mean": _rounded(statistics.fmean(ordered)),
        "median": _rounded(statistics.median(ordered)),
        "minimum": _rounded(ordered[0]),
        "q1": _rounded(_percentile(ordered, 0.25)),
        "q3": _rounded(_percentile(ordered, 0.75)),
        "maximum": _rounded(ordered[-1]),
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
    case_path = root / "case_diagnostics.jsonl"
    decision_path = root / "candidate_decisions.jsonl"
    aggregate_path = root / "aggregate.json"
    protocol_copy = root / "protocol.json"
    _write_jsonl(case_path, sorted(cases, key=lambda item: int(item["case_order"])))
    _write_jsonl(
        decision_path,
        sorted(
            decisions,
            key=lambda item: (int(item["case_order"]), int(item["candidate_order"])),
        ),
    )
    _write_json(aggregate_path, aggregate)
    _write_json(protocol_copy, _read_json(Path(protocol_path).expanduser().resolve()))
    files = {
        "aggregate": aggregate_path,
        "candidate_decisions": decision_path,
        "case_diagnostics": case_path,
        "protocol": protocol_copy,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": aggregate["status"],
        "files": {
            name: {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(files.items())
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_analysis(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("analysis") != CONTRACT_VERSION:
        raise ConstraintDecisionAuditError("manifest_contract_mismatch")
    for item in manifest.get("files", {}).values():
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise ConstraintDecisionAuditError("output_missing_or_size_drift")
        if _sha256(path) != str(item["sha256"]):
            raise ConstraintDecisionAuditError("output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") != "completed":
        raise ConstraintDecisionAuditError("analysis_not_completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw.get("case_id") or "")
        if not case_id or case_id in seen:
            raise ConstraintDecisionNotEligible("invalid_or_duplicate_record_identity")
        seen.add(case_id)
        diagnostics = raw.get("stage_diagnostics") or {}
        planning = diagnostics.get("initial_query_planning") or {}
        rows.append(
            {
                "case_id": case_id,
                "original_query": str(planning.get("original_query") or ""),
                "query_analysis": planning.get("query_analysis"),
                "stage_diagnostics": {"snapshots": diagnostics.get("snapshots")},
            }
        )
    return rows


def _load_component_assignments(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        query_id = str(raw.get("query_id") or "")
        component_id = str(raw.get("component_id") or "")
        if not query_id or not component_id or query_id in result:
            raise ConstraintDecisionNotEligible("invalid_component_assignment")
        result[query_id] = component_id
    return result


def _validate_config(config: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = {
        "dataset": "auto_scholar_query",
        "query_planning_policy": "current_rules",
        "ranking_policy": "current_rules",
        "judgement_policy": "current_rules",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "sources": ["openalex", "arxiv", "semantic_scholar", "pubmed"],
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ConstraintDecisionNotEligible(f"frozen_config_drift:{field}")
    if int(config.get("budgets", {}).get("max_candidate_papers") or 0) != int(
        protocol["reconstruction"]["candidate_limit"]
    ):
        raise ConstraintDecisionNotEligible("candidate_budget_drift")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise ConstraintDecisionNotEligible("experimental_strategy_enabled")
    if (config.get("judgement_config") or {}).get("lexical_normalization_policy") != "off":
        raise ConstraintDecisionNotEligible("lexical_normalization_not_default_off")


def _validate_population(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    if len(included) != int(protocol["analysis_population"]["main_case_count"]):
        raise ConstraintDecisionNotEligible("main_population_drift")
    if len(excluded) != int(protocol["analysis_population"]["excluded_case_count"]):
        raise ConstraintDecisionNotEligible("excluded_population_drift")
    orders = sorted(int(item["case_order"]) for item in [*included, *excluded])
    if orders != list(range(len(included) + len(excluded))):
        raise ConstraintDecisionAuditError("population_omission_or_duplicate")


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise ConstraintDecisionAuditError("network_attempt_detected")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
    ):
        yield


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_file_hash(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected):
        raise ConstraintDecisionNotEligible(f"frozen_file_hash_drift:{path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConstraintDecisionAuditError("json_root_not_object")
    return value


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _opaque_identity(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}\0{value}".encode("utf-8")).hexdigest()


def _rounded(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
