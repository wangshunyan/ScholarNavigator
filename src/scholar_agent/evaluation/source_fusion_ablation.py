"""Deterministic, gold-free source-fusion ablation over frozen Record160 Replay.

The audit consumes only frozen query plans, source retrieval entries, production
stage snapshots, and precomputed opaque query-component assignments.  It never
loads a Benchmark dataset or evaluator labels.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import SnapshotError


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "source_fusion_ablation_v1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class SourceFusionAblationError(RuntimeError):
    """The frozen contract or an analysis invariant was violated."""


class SourceFusionNotEligible(SourceFusionAblationError):
    """The frozen source data cannot support an exact source ablation."""


@dataclass(frozen=True)
class VariantResult:
    candidates: list[Paper]
    judgements: list[Any]
    ranked: list[Any]
    returned: list[Any]


@dataclass(frozen=True)
class SourceInputs:
    raw_by_source: dict[str, list[Paper]]
    ordered_batches: list[tuple[str, list[Paper]]]
    terminal_counts: dict[str, Counter[str]]
    unassigned_terminal_counts: Counter[str]
    observed_key_count: int


class IdentityRegistry:
    """Assign opaque labels using the production identity-equivalence relation."""

    def __init__(self) -> None:
        self._profiles: list[IdentityProfile] = []
        self._labels: list[str] = []

    def labels(self, values: Sequence[Any]) -> list[str]:
        return [self.label(value) for value in values]

    def label(self, value: Any) -> str:
        profile = build_identity_profile(value)
        match = next(
            (
                index
                for index, existing in enumerate(self._profiles)
                if identity_evidence_from_profiles(existing, profile).equivalent
            ),
            None,
        )
        if match is None:
            self._profiles.append(profile)
            self._labels.append(_opaque_profile_identity(profile))
            match = len(self._labels) - 1
        return self._labels[match]


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).expanduser().resolve()
    value = _read_json(protocol_path)
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise SourceFusionAblationError("unsupported_protocol")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }:
        raise SourceFusionAblationError("offline_protocol_drift")
    if value.get("sources") != [
        "openalex",
        "arxiv",
        "semantic_scholar",
        "pubmed",
    ]:
        raise SourceFusionAblationError("source_order_drift")
    if int(value.get("ranking", {}).get("top_k") or 0) != 20:
        raise SourceFusionAblationError("top_k_drift")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_ablation_result",
    ]:
        raise SourceFusionAblationError("selection_contract_drift")
    return value


def run_source_fusion_ablation(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the preregistered analysis without loading gold or opening network."""

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
        raise SourceFusionNotEligible("snapshot_tree_hash_drift")
    if sum(item.is_file() for item in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise SourceFusionNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, protocol)
    rows = _read_record_rows(results_path)
    expected_record_count = int(protocol["analysis_population"]["record_case_count"])
    if len(rows) != expected_record_count:
        raise SourceFusionNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[:expected_record_count]:
        raise SourceFusionNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise SourceFusionNotEligible("missing_frozen_component_assignment")

    store = SnapshotStore(snapshot_dir)
    attempts = {"network": 0}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    observed_key_count = 0
    with _forbid_network(attempts):
        for case_order, row in enumerate(rows):
            try:
                case = analyze_case(
                    row,
                    config=config,
                    protocol=protocol,
                    store=store,
                    component_id=components[str(row["case_id"])],
                    case_order=case_order,
                )
            except SnapshotError as exc:
                raise SourceFusionNotEligible(
                    f"source_level_snapshot_unavailable:{type(exc).__name__}"
                ) from None
            observed_key_count += int(case["observed_snapshot_key_count"])
            if case["analysis_status"] == "excluded_no_successful_source":
                excluded.append(case)
            else:
                included.append(case)

    validate_population_closure(included, excluded, protocol)
    after_tree = _tree_sha256(snapshot_dir)
    if after_tree != before_tree:
        raise SourceFusionAblationError("snapshot_tree_changed")
    if attempts["network"]:
        raise SourceFusionAblationError("network_attempt_detected")
    included.sort(key=lambda item: int(item["case_order"]))
    excluded.sort(key=lambda item: int(item["case_order"]))
    aggregate = aggregate_analysis(
        included,
        excluded,
        protocol,
        protocol_sha256=_sha256(protocol_file),
        input_hashes={
            "config_sha256": _sha256(config_path),
            "record_results_sha256": _sha256(results_path),
            "snapshot_tree_sha256": before_tree,
            "component_assignments_sha256": _sha256(assignments_path),
        },
        observed_key_count=observed_key_count,
    )
    return [*included, *excluded], aggregate


def analyze_case(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    store: Any,
    component_id: str,
    case_order: int,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    query_identity = _opaque_query_identity(case_id)
    stages = {
        str(item.get("stage")): item
        for item in row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
        "final_returned",
    }
    if not required.issubset(stages):
        raise SourceFusionNotEligible("required_frozen_stage_missing")
    source_inputs = reconstruct_source_inputs(
        stages["initial_retrieval"], config=config, store=store
    )
    source_states = {
        source: _source_state(source_inputs.terminal_counts[source])
        for source in protocol["sources"]
    }
    successful_source_count = sum(
        source_inputs.terminal_counts[source]["success"] > 0
        for source in protocol["sources"]
    )
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": (
            "included_main_analysis"
            if successful_source_count
            else "excluded_no_successful_source"
        ),
        "case_order": case_order,
        "query_identity": query_identity,
        "component_identity": str(component_id),
        "successful_source_count": successful_source_count,
        "source_states": source_states,
        "unassigned_terminal_counts": dict(
            sorted(source_inputs.unassigned_terminal_counts.items())
        ),
        "observed_snapshot_key_count": source_inputs.observed_key_count,
    }
    if not successful_source_count:
        return base

    analysis = QueryAnalysis.model_validate(
        row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    sources = [str(value) for value in protocol["sources"]]
    limit = int(config["budgets"]["max_candidate_papers"])
    pools = {
        "full": build_candidate_pool(
            source_inputs.ordered_batches,
            included_sources=sources,
            limit=limit,
            source_order=sources,
        )
    }
    frozen_full = align_papers_to_diagnostics(
        pools["full"], stages["initial_deduplicated"]["candidates"]
    )
    full = rank_variant(analysis, frozen_full, top_k=int(protocol["ranking"]["top_k"]))
    validate_full_reconstruction(full, stages)

    variants: dict[str, VariantResult] = {"full": full}
    for source in sources:
        variants[f"loo:{source}"] = rank_variant(
            analysis,
            build_candidate_pool(
                source_inputs.ordered_batches,
                included_sources=[item for item in sources if item != source],
                limit=limit,
                source_order=sources,
            ),
            top_k=int(protocol["ranking"]["top_k"]),
        )
        variants[f"single:{source}"] = rank_variant(
            analysis,
            build_candidate_pool(
                source_inputs.ordered_batches,
                included_sources=[source],
                limit=limit,
                source_order=sources,
            ),
            top_k=int(protocol["ranking"]["top_k"]),
        )

    registry = IdentityRegistry()
    candidate_labels = {
        name: registry.labels(result.candidates) for name, result in variants.items()
    }
    returned_labels = {
        name: registry.labels([item.paper for item in result.returned])
        for name, result in variants.items()
    }
    source_papers = {
        source: deduplicate_papers(list(source_inputs.raw_by_source[source]))
        for source in sources
    }
    source_labels = {
        source: registry.labels(source_papers[source]) for source in sources
    }
    source_sets = {source: set(source_labels[source]) for source in sources}
    full_returned = returned_labels["full"]
    result_sources: dict[str, Any] = {}
    for source in sources:
        other = set().union(
            *(source_sets[item] for item in sources if item != source)
        )
        loo_returned = returned_labels[f"loo:{source}"]
        single_returned = returned_labels[f"single:{source}"]
        result_sources[source] = {
            "terminal_counts": dict(sorted(source_inputs.terminal_counts[source].items())),
            "empty_response": not source_labels[source],
            "source_unique_identity_count": len(source_sets[source]),
            "source_exclusive_identity_count": len(source_sets[source] - other),
            "source_redundant_identity_count": len(source_sets[source] & other),
            "pair_overlap_counts": {
                item: len(source_sets[source] & source_sets[item])
                for item in sources
                if item != source
            },
            "single_source": variant_summary(
                candidate_labels[f"single:{source}"], single_returned
            ),
            "leave_one_out": {
                **variant_summary(candidate_labels[f"loo:{source}"], loo_returned),
                **compare_ranked_lists(
                    full_returned,
                    loo_returned,
                    persistence=float(protocol["rbo"]["persistence"]),
                    depth=int(protocol["rbo"]["depth"]),
                ),
            },
        }
    return {
        **base,
        "reconstruction": {
            "initial_deduplicated_exact": True,
            "initial_judged_exact": True,
            "initial_reranked_exact": True,
            "final_returned_exact": True,
            "digest": _stable_json_sha256(
                {
                    "candidate_ids": candidate_labels["full"],
                    "returned_ids": full_returned,
                    "source_states": source_states,
                }
            ),
        },
        "full": variant_summary(candidate_labels["full"], full_returned),
        "sources": result_sources,
    }


def reconstruct_source_inputs(
    initial: Mapping[str, Any], *, config: Mapping[str, Any], store: Any
) -> SourceInputs:
    sources = [str(value) for value in config["sources"]]
    raw_by_source = {source: [] for source in sources}
    ordered_batches: list[tuple[str, list[Paper]]] = []
    terminal_counts = {source: Counter() for source in sources}
    unassigned_terminal_counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    observed = 0
    for call in initial.get("retrieval_calls") or []:
        source = str(call.get("source") or "")
        if source not in raw_by_source:
            if (
                source == "subquery"
                and bool(call.get("logical_call_executed"))
                and not call.get("snapshot_key")
                and int(call.get("returned_count") or 0) == 0
                and str(call.get("terminal_status") or "") == "timeout"
            ):
                unassigned_terminal_counts["subquery_timeout"] += 1
                continue
            raise SourceFusionNotEligible("unknown_source_in_frozen_call")
        if not bool(call.get("logical_call_executed")):
            terminal_counts[source][str(call.get("terminal_status") or "not_started")] += 1
            continue
        key = str(call.get("snapshot_key") or "")
        if not key:
            terminal_counts[source][
                str(
                    call.get("terminal_status")
                    or (
                        "not_started"
                        if call.get("source_skipped_reason")
                        else "unknown_terminal"
                    )
                )
            ] += 1
            continue
        if key in seen_keys:
            raise SourceFusionNotEligible("duplicate_executed_snapshot_key")
        seen_keys.add(key)
        entry = store.read_retrieval(key)
        recorded_terminal = call.get("terminal_status")
        if (
            entry.source != source
            or entry.adapted_query != str(call.get("adapted_query") or "")
            or (
                recorded_terminal is not None
                and entry.status != str(recorded_terminal)
            )
            or entry.limit != int(config["top_k"])
            or entry.adapter_policy != str(config["query_adapter_policy"])
        ):
            raise SourceFusionNotEligible("frozen_source_request_signature_mismatch")
        observed += 1
        terminal_counts[source][entry.status] += 1
        if entry.status == "success":
            batch = [item.model_copy(deep=True) for item in entry.papers]
            raw_by_source[source].extend(batch)
            ordered_batches.append((source, batch))
    return SourceInputs(
        raw_by_source,
        ordered_batches,
        terminal_counts,
        unassigned_terminal_counts,
        observed,
    )


def build_candidate_pool(
    ordered_batches: Sequence[tuple[str, Sequence[Paper]]],
    *,
    included_sources: Sequence[str],
    limit: int,
    source_order: Sequence[str],
) -> list[Paper]:
    included = set(included_sources)
    raw = [
        paper.model_copy(deep=True)
        for source, batch in ordered_batches
        if source in included
        for paper in batch
    ]
    candidates = deduplicate_papers(raw)
    if len(candidates) > limit:
        candidates = stable_source_coverage_truncate(
            candidates, limit=limit, source_order=source_order
        )
    return candidates


def rank_variant(
    analysis: QueryAnalysis, candidates: list[Paper], *, top_k: int
) -> VariantResult:
    judgements = judge_papers(
        analysis,
        candidates,
        use_llm=False,
        policy="current_rules",
        config=CURRENT_RULES_CONFIG,
    )
    ranked = rerank_papers(analysis, judgements, top_k=len(judgements))
    returned = select_ranked_results(
        {"ranked_papers": ranked[:top_k]}, policy="highly_and_partial"
    )
    return VariantResult(candidates, judgements, ranked, returned)


def validate_full_reconstruction(
    full: VariantResult, stages: Mapping[str, Mapping[str, Any]]
) -> None:
    try:
        _assert_equivalent_sequence(
            full.candidates, stages["initial_deduplicated"]["candidates"]
        )
        frozen_judged = list(stages["initial_judged"].get("candidates") or [])
        if len(full.judgements) != len(frozen_judged):
            raise ValueError("judgement_length")
        for live, frozen in zip(full.judgements, frozen_judged, strict=True):
            if (
                not _equivalent(live.paper, frozen)
                or live.category != str(frozen.get("category") or "")
                or live.score != float(frozen["judgement_score"])
            ):
                raise ValueError("judgement_value")
        frozen_ranked = list(stages["initial_reranked"].get("candidates") or [])
        if len(full.ranked) != len(frozen_ranked):
            raise ValueError("ranking_length")
        for live, frozen in zip(full.ranked, frozen_ranked, strict=True):
            if (
                not _equivalent(live.paper, frozen)
                or live.rank != int(frozen["rank"])
                or live.category != str(frozen["category"])
                or live.final_score != float(frozen["final_score"])
            ):
                raise ValueError("ranking_value")
        _assert_equivalent_sequence(
            [item.paper for item in full.returned],
            stages["final_returned"].get("candidates") or [],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceFusionNotEligible(
            f"frozen_current_rules_reconstruction_mismatch:{exc}"
        ) from None


def variant_summary(candidate_ids: Sequence[str], returned_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "unified_identity_count": len(set(candidate_ids)),
        "top20_fill_count": len(returned_ids),
        "candidate_identity_digest": _stable_json_sha256(list(candidate_ids)),
        "top20_identity_digest": _stable_json_sha256(list(returned_ids)),
        "top20_identity_ids": list(returned_ids),
    }


def compare_ranked_lists(
    full: Sequence[str], variant: Sequence[str], *, persistence: float, depth: int
) -> dict[str, Any]:
    full_top = list(full[:depth])
    variant_top = list(variant[:depth])
    full_set = set(full_top)
    variant_set = set(variant_top)
    shared = full_set & variant_set
    union = full_set | variant_set
    full_rank = {identity: index + 1 for index, identity in enumerate(full_top)}
    variant_rank = {identity: index + 1 for index, identity in enumerate(variant_top)}
    displacements = [
        abs(full_rank[identity] - variant_rank[identity]) for identity in shared
    ]
    return {
        "full_top20_identity_loss_count": len(full_set - variant_set),
        "top20_jaccard": len(shared) / len(union) if union else 1.0,
        "rank_biased_overlap": finite_extrapolated_rbo(
            full_top, variant_top, persistence=persistence, depth=depth
        ),
        "shared_identity_count": len(shared),
        "shared_identity_mean_absolute_rank_displacement": (
            statistics.fmean(displacements) if displacements else None
        ),
        "shared_identity_median_absolute_rank_displacement": (
            statistics.median(displacements) if displacements else None
        ),
    }


def finite_extrapolated_rbo(
    left: Sequence[str], right: Sequence[str], *, persistence: float, depth: int
) -> float:
    if not 0 < persistence < 1 or depth <= 0:
        raise ValueError("invalid RBO parameters")
    if not left and not right:
        return 1.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    weighted = 0.0
    overlap_at_depth = 0.0
    for rank in range(1, depth + 1):
        if rank <= len(left):
            left_seen.add(left[rank - 1])
        if rank <= len(right):
            right_seen.add(right[rank - 1])
        prefix_size = max(len(left_seen), len(right_seen), 1)
        agreement = len(left_seen & right_seen) / prefix_size
        weighted += (1 - persistence) * (persistence ** (rank - 1)) * agreement
        overlap_at_depth = agreement
    return weighted + (persistence**depth) * overlap_at_depth


def aggregate_analysis(
    cases: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    observed_key_count: int,
) -> dict[str, Any]:
    sources = [str(value) for value in protocol["sources"]]
    full_metrics = {}
    for metric in ("top20_fill_count", "unified_identity_count"):
        values = [float(case["full"][metric]) for case in cases]
        full_metrics[metric] = {
            "query_distribution": summarize_distribution(values),
            "cluster_summary": cluster_summary(
                cases, values, protocol, "full", f"full:{metric}"
            ),
        }
    source_rows: dict[str, dict[str, Any]] = {}
    inference_by_metric: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in sources:
        contribution = {
            metric: summarize_distribution(
                [float(case["sources"][source][metric]) for case in cases]
            )
            for metric in (
                "source_unique_identity_count",
                "source_exclusive_identity_count",
                "source_redundant_identity_count",
            )
        }
        contribution["source_empty_response_count"] = sum(
            bool(case["sources"][source]["empty_response"]) for case in cases
        )
        contribution["source_empty_response_rate"] = (
            contribution["source_empty_response_count"] / len(cases) if cases else None
        )
        contribution["pair_overlap_counts"] = {
            other: summarize_distribution(
                [
                    float(case["sources"][source]["pair_overlap_counts"][other])
                    for case in cases
                ]
            )
            for other in sources
            if other != source
        }
        single_metrics = {}
        loo_metrics = {}
        for metric in ("top20_fill_count", "unified_identity_count"):
            single_values = [
                float(case["sources"][source]["single_source"][metric])
                for case in cases
            ]
            loo_values = [
                float(case["sources"][source]["leave_one_out"][metric])
                for case in cases
            ]
            full_values = [float(case["full"][metric]) for case in cases]
            single_metrics[metric] = {
                "absolute": cluster_summary(cases, single_values, protocol, source, metric),
                "difference_vs_full": cluster_summary(
                    cases,
                    [left - right for left, right in zip(single_values, full_values)],
                    protocol,
                    source,
                    f"single_difference:{metric}",
                ),
            }
            loo_stats = cluster_summary(
                cases,
                [left - right for left, right in zip(loo_values, full_values)],
                protocol,
                source,
                metric,
            )
            loo_metrics[metric] = {
                "absolute": cluster_summary(
                    cases, loo_values, protocol, source, f"loo_absolute:{metric}"
                ),
                "difference_vs_full": loo_stats,
            }
            inference_by_metric[metric][source] = loo_stats
        comparison_effects = {
            "full_top20_identity_loss_count": lambda value: -float(value),
            "top20_jaccard": lambda value: float(value) - 1.0,
            "rank_biased_overlap": lambda value: float(value) - 1.0,
            "shared_identity_mean_absolute_rank_displacement": lambda value: -float(value),
        }
        for metric, transform in comparison_effects.items():
            values: list[float | None] = []
            for case in cases:
                raw = case["sources"][source]["leave_one_out"][metric]
                values.append(None if raw is None else transform(raw))
            stats = cluster_summary(cases, values, protocol, source, metric)
            loo_metrics[metric] = {
                "effect_vs_identical_full": stats,
                "raw_distribution": summarize_distribution(
                    [
                        float(case["sources"][source]["leave_one_out"][metric])
                        for case in cases
                        if case["sources"][source]["leave_one_out"][metric] is not None
                    ]
                ),
                "missing_reason_counts": {
                    "no_shared_top20_identity": sum(
                        case["sources"][source]["leave_one_out"][metric] is None
                        for case in cases
                    )
                },
            }
            inference_by_metric[metric][source] = stats
        source_rows[source] = {
            "contribution": contribution,
            "single_source": single_metrics,
            "leave_one_out": loo_metrics,
            "source_state_counts": dict(
                sorted(Counter(case["source_states"][source] for case in cases).items())
            ),
        }
    corrections = {
        metric: holm_bonferroni(
            {
                source: float(inference_by_metric[metric][source]["sign_flip_p_value"])
                for source in sources
            },
            alpha=float(protocol["comparison_family"]["alpha"]),
        )
        for metric in protocol["multiple_comparison_metrics"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "implementation_base_commit": protocol["implementation_base_commit"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": _sha256(Path(__file__)),
        "inputs": dict(sorted(input_hashes.items())),
        "closure": {
            "record_case_count": len(cases) + len(excluded),
            "included_main_case_count": len(cases),
            "excluded_no_successful_source_count": len(excluded),
            "reconstruction_exact_case_count": sum(
                all(bool(value) for key, value in case["reconstruction"].items() if key.endswith("_exact"))
                for case in cases
            ),
            "component_count": len({str(case["component_identity"]) for case in cases}),
            "observed_snapshot_key_count": observed_key_count,
        },
        "full": full_metrics,
        "sources": source_rows,
        "holm_bonferroni": corrections,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_or_qrels_loaded": False,
            "quality_metric_count": 0,
            "full1000_inference_performed": False,
        },
        "interpretation": {
            "scope": "coverage_identity_and_rank_stability_only",
            "relevance_claim_permitted": False,
            "precision_recall_or_official_score": False,
            "warnings": list(protocol["warnings"]),
        },
    }


def cluster_summary(
    cases: Sequence[Mapping[str, Any]],
    values: Sequence[float | None],
    protocol: Mapping[str, Any],
    source: str,
    metric: str,
) -> dict[str, Any]:
    if len(cases) != len(values):
        raise ValueError("cluster statistic inputs are not aligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    missing = 0
    for case, value in zip(cases, values, strict=True):
        if value is None:
            missing += 1
            continue
        grouped[str(case["component_identity"])].append(float(value))
    component_values = [statistics.fmean(grouped[key]) for key in sorted(grouped)]
    if not component_values:
        return {
            "query_count": len(cases),
            "observed_query_count": 0,
            "missing_query_count": missing,
            "component_count": 0,
            "mean": None,
            "median": None,
            "confidence_interval_95": [None, None],
            "sign_flip_p_value": 1.0,
        }
    seed = _derived_seed(int(protocol["bootstrap"]["seed"]), source, metric)
    bootstrap = _bootstrap_component_means(
        component_values,
        seed=seed,
        iterations=int(protocol["bootstrap"]["iterations"]),
    )
    permutation_seed = _derived_seed(
        int(protocol["comparison_family"]["permutation_seed"]), source, metric
    )
    p_value = _cluster_sign_flip(
        component_values,
        seed=permutation_seed,
        iterations=int(protocol["comparison_family"]["permutation_iterations"]),
    )
    return {
        "query_count": len(cases),
        "observed_query_count": len(cases) - missing,
        "missing_query_count": missing,
        "component_count": len(component_values),
        "mean": _rounded(statistics.fmean(component_values)),
        "median": _rounded(statistics.median(component_values)),
        "confidence_interval_95": [
            _rounded(_percentile(bootstrap, 0.025)),
            _rounded(_percentile(bootstrap, 0.975)),
        ],
        "sign_flip_p_value": _rounded(p_value),
    }


def holm_bonferroni(
    p_values: Mapping[str, float], *, alpha: float
) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted_running = 0.0
    rejected_prefix = True
    result: dict[str, dict[str, Any]] = {}
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        adjusted_running = max(adjusted_running, min(1.0, (total - index) * p_value))
        threshold = alpha / (total - index)
        rejected_prefix = rejected_prefix and p_value <= threshold
        result[name] = {
            "raw_p_value": _rounded(p_value),
            "adjusted_p_value": _rounded(adjusted_running),
            "holm_threshold": _rounded(threshold),
            "reject_at_family_alpha": rejected_prefix,
        }
    return {name: result[name] for name in sorted(result)}


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
            "total": 0,
        }
    return {
        "count": len(ordered),
        "mean": _rounded(statistics.fmean(ordered)),
        "median": _rounded(statistics.median(ordered)),
        "minimum": _rounded(ordered[0]),
        "q1": _rounded(_percentile(ordered, 0.25)),
        "q3": _rounded(_percentile(ordered, 0.75)),
        "maximum": _rounded(ordered[-1]),
        "total": _rounded(sum(ordered)),
    }


def validate_population_closure(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    expected_main = int(protocol["analysis_population"]["main_case_count"])
    expected_excluded = int(protocol["analysis_population"]["excluded_case_count"])
    if len(included) != expected_main or len(excluded) != expected_excluded:
        raise SourceFusionAblationError("frozen_population_or_exclusion_drift")
    all_orders = [int(item["case_order"]) for item in [*included, *excluded]]
    if sorted(all_orders) != list(range(expected_main + expected_excluded)):
        raise SourceFusionAblationError("population_has_omission_or_duplicate")
    if any(item["analysis_status"] != "included_main_analysis" for item in included):
        raise SourceFusionAblationError("post_hoc_inclusion_detected")
    if any(
        item["analysis_status"] != "excluded_no_successful_source" for item in excluded
    ):
        raise SourceFusionAblationError("unregistered_exclusion_detected")


def write_analysis(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protocol_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_path = root / "case_diagnostics.jsonl"
    aggregate_path = root / "aggregate.json"
    protocol_copy = root / "protocol.json"
    _write_jsonl(case_path, sorted(cases, key=lambda item: int(item["case_order"])))
    _write_json(aggregate_path, aggregate)
    protocol = _read_json(Path(protocol_path).expanduser().resolve())
    _write_json(protocol_copy, protocol)
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
            for name, path in (
                ("aggregate", aggregate_path),
                ("case_diagnostics", case_path),
                ("protocol", protocol_copy),
            )
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_analysis(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("analysis") != CONTRACT_VERSION:
        raise SourceFusionAblationError("analysis_manifest_contract_mismatch")
    for item in manifest.get("files", {}).values():
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise SourceFusionAblationError("analysis_output_missing_or_size_drift")
        if _sha256(path) != str(item["sha256"]):
            raise SourceFusionAblationError("analysis_output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") != "completed":
        raise SourceFusionAblationError("analysis_not_completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
    }


def _read_record_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw.get("case_id") or "")
        if not case_id or case_id in seen:
            raise SourceFusionNotEligible("invalid_or_duplicate_record_identity")
        seen.add(case_id)
        diagnostics = raw.get("stage_diagnostics") or {}
        rows.append(
            {
                "case_id": case_id,
                "status": raw.get("status"),
                "stage_diagnostics": {
                    "initial_query_planning": diagnostics.get(
                        "initial_query_planning"
                    ),
                    "snapshots": diagnostics.get("snapshots"),
                },
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
            raise SourceFusionNotEligible("invalid_frozen_component_assignment")
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
        "sources": list(protocol["sources"]),
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise SourceFusionNotEligible(f"frozen_config_drift:{field}")
    if int(config.get("budgets", {}).get("max_candidate_papers") or 0) != int(
        protocol["ranking"]["candidate_limit"]
    ):
        raise SourceFusionNotEligible("candidate_budget_drift")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise SourceFusionNotEligible("experimental_strategy_enabled")
    if (config.get("judgement_config") or {}).get("lexical_normalization_policy") != "off":
        raise SourceFusionNotEligible("lexical_normalization_not_default_off")


def _source_state(counts: Counter[str]) -> str:
    successful = counts["success"] > 0
    failed = counts["failed"] > 0 or counts["timeout"] > 0
    if successful and failed:
        return "partial_failure"
    if successful:
        return "success"
    if failed:
        return "failed"
    return "not_started"


def _assert_equivalent_sequence(left: Sequence[Any], right: Sequence[Any]) -> None:
    if len(left) != len(right):
        raise ValueError("identity_sequence_length")
    if any(not _equivalent(a, b) for a, b in zip(left, right, strict=True)):
        raise ValueError("identity_sequence_value")


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _opaque_profile_identity(profile: IdentityProfile) -> str:
    payload = {
        "authors": sorted(profile.authors),
        "field_values": list(profile.field_values),
        "identifiers": sorted(profile.identifiers),
        "title": profile.title,
        "year": profile.year,
    }
    return "paper:" + _stable_json_sha256(payload)[:24]


def _opaque_query_identity(case_id: str) -> str:
    return "query:" + hashlib.sha256(
        ("source-fusion-query-v1\0" + case_id).encode("utf-8")
    ).hexdigest()[:24]


def _bootstrap_component_means(
    values: Sequence[float], *, seed: int, iterations: int
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    return sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(iterations)
    )


def _cluster_sign_flip(
    values: Sequence[float], *, seed: int, iterations: int
) -> float:
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        sampled = statistics.fmean(
            value if rng.getrandbits(1) else -value for value in values
        )
        if abs(sampled) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower]) * (1 - weight) + float(values[upper]) * weight


def _derived_seed(base: int, source: str, metric: str) -> int:
    digest = hashlib.sha256(f"{source}\0{metric}".encode("utf-8")).digest()
    return base ^ int.from_bytes(digest[:8], "big")


def _rounded(value: float) -> float:
    return round(float(value), 12)


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts["network"] += 1
        raise SourceFusionAblationError("network_attempt_detected")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket, "getaddrinfo", blocked),
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
    ):
        yield


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_file_hash(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected):
        raise SourceFusionNotEligible(f"frozen_input_hash_drift:{path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceFusionAblationError("expected_json_object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
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
