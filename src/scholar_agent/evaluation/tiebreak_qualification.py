"""Gold-free qualification of the optional deterministic tie-break v2.

The audit reconstructs the frozen Record160 candidate pool through production
canonicalization, identity deduplication, Judgement and reranking.  Candidate
permutations are applied only after the immutable deduplicated pool exists, so
the gate measures tie-break behavior without changing authoritative merges.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.agents.reranker import (
    DEFAULT_TIEBREAK_POLICY,
    DETERMINISTIC_TIEBREAK_V2,
    DeterministicTieBreakUnavailable,
    deterministic_tiebreak_v2_catalog,
    production_ranking_decision_catalog,
    rerank_papers,
    trace_deterministic_tiebreak_v2,
)
from scholar_agent.core.dedup import deduplicate_papers_with_lineage
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
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
)
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.source_fusion_ablation import (
    IdentityRegistry,
    VariantResult,
    rank_variant,
    validate_full_reconstruction,
)
from scholar_agent.evaluation.source_reliability_diagnostics import (
    audit_retrieval_requests,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "deterministic_tiebreak_qualification_v1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_QUALIFIED = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RETURN_CATEGORIES = frozenset({"highly_relevant", "partially_relevant"})


class TieBreakQualificationError(RuntimeError):
    """A protocol, reconstruction, or permutation invariant was violated."""


class TieBreakNotEligible(TieBreakQualificationError):
    """Frozen evidence cannot support the preregistered qualification."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path).expanduser().resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise TieBreakQualificationError("unsupported_protocol")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise TieBreakQualificationError("offline_protocol_drift")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_tie_result",
    ]:
        raise TieBreakQualificationError("selection_contract_drift")
    current = production_ranking_decision_catalog()["sort_key"]
    if value.get("current_policy", {}).get("primary_key_fields") != current[:-1]:
        raise TieBreakQualificationError("primary_key_catalog_drift")
    if value.get("current_policy", {}).get("tie_break_field") != current[-1]:
        raise TieBreakQualificationError("current_tie_break_catalog_drift")
    if value.get("current_policy", {}).get("tie_definition") != (
        "all primary key values compare exactly equal after the existing production "
        "score rounding and field normalization; no tolerance, bucket, or approximate "
        "comparison is permitted"
    ):
        raise TieBreakQualificationError("exact_tie_definition_drift")
    v2 = deterministic_tiebreak_v2_catalog()
    frozen_v2 = value.get("deterministic_tiebreak_v2") or {}
    if frozen_v2.get("name") != v2["version"]:
        raise TieBreakQualificationError("v2_version_drift")
    if frozen_v2.get("default_enabled") is not False:
        raise TieBreakQualificationError("v2_default_state_drift")
    if DEFAULT_TIEBREAK_POLICY != "original_index_v1":
        raise TieBreakQualificationError("production_default_tie_break_changed")
    if value.get("permutations", {}).get("fixed_profiles") != [
        "canonical_input",
        "reverse_input",
        "reverse_source_blocks",
        "rotate_source_blocks",
        "deterministic_random",
        "shard_completion_2_0_1",
    ]:
        raise TieBreakQualificationError("permutation_contract_drift")
    return value


def run_tiebreak_qualification(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen qualification with network and evaluator access blocked."""

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
        raise TieBreakNotEligible("snapshot_tree_hash_drift")
    if sum(path.is_file() for path in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise TieBreakNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, protocol)
    rows = _read_rows(results_path)
    if len(rows) != int(protocol["analysis_population"]["record_case_count"]):
        raise TieBreakNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[: len(rows)]:
        raise TieBreakNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise TieBreakNotEligible("missing_component_assignment")

    store = SnapshotStore(snapshot_dir)
    attempts = {"network": 0}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    with _forbid_network(attempts):
        for case_order, row in enumerate(rows):
            case, candidates, ties, keys = analyze_case(
                row,
                config=config,
                protocol=protocol,
                store=store,
                component_id=components[str(row["case_id"])],
                case_order=case_order,
            )
            observed_keys.update(keys)
            if case["analysis_status"] == "excluded_no_successful_source":
                excluded.append(case)
            else:
                included.append(case)
                candidate_rows.extend(candidates)
                tie_rows.extend(ties)

    _validate_population(included, excluded, protocol)
    if len(candidate_rows) != int(
        protocol["analysis_population"]["expected_candidate_count"]
    ):
        raise TieBreakNotEligible("candidate_population_drift")
    if len(observed_keys) != int(frozen["observed_snapshot_key_count"]):
        raise TieBreakNotEligible("observed_snapshot_key_count_drift")
    if _tree_sha256(snapshot_dir) != before_tree:
        raise TieBreakQualificationError("snapshot_tree_changed")
    if attempts["network"]:
        raise TieBreakQualificationError("network_attempt_detected")

    cases = sorted([*included, *excluded], key=lambda item: int(item["case_order"]))
    aggregate = aggregate_analysis(
        included,
        excluded,
        candidate_rows,
        tie_rows,
        protocol,
        protocol_sha256=_sha256(protocol_file),
        input_hashes={
            "component_assignments_sha256": _sha256(assignments_path),
            "config_sha256": _sha256(config_path),
            "record_results_sha256": _sha256(results_path),
            "snapshot_tree_sha256": before_tree,
        },
        observed_snapshot_key_count=len(observed_keys),
    )
    return cases, candidate_rows, tie_rows, aggregate


def analyze_case(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    store: Any,
    component_id: str,
    case_order: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    stages = {
        str(item.get("stage")): item
        for item in row["stage_diagnostics"]["snapshots"]
    }
    required = set(protocol["reconstruction"]["required_exact_stages"]) | {
        "initial_retrieval"
    }
    if not required.issubset(stages):
        raise TieBreakNotEligible("required_frozen_stage_missing")
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
    }
    if not successful_source_count:
        return base, [], [], requests.observed_keys

    analysis = QueryAnalysis.model_validate(row["query_analysis"])
    raw = [
        paper.model_copy(deep=True)
        for _source, batch in requests.ordered_batches
        for paper in batch
    ]
    deduplicated, _dedup_audit, lineage = deduplicate_papers_with_lineage(
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
    current = rank_variant(
        analysis, candidates, top_k=int(protocol["reconstruction"]["top_k"])
    )
    validate_full_reconstruction(current, stages)
    registry = IdentityRegistry()
    candidate_ids = registry.labels(current.candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise TieBreakQualificationError("candidate_identity_duplicate_after_dedup")

    lineage_by_identity: dict[str, Mapping[str, Any]] = {}
    for item in lineage["results"]:
        identity = registry.label(Paper.model_validate(item["final_result"]))
        if identity in lineage_by_identity:
            raise TieBreakNotEligible("lineage_identity_collision")
        lineage_by_identity[identity] = item
    if set(lineage_by_identity) != set(candidate_ids):
        raise TieBreakNotEligible("lineage_candidate_set_mismatch")
    refs_by_identity = {
        identity: tuple(str(value) for value in item["contributing_record_refs"])
        for identity, item in lineage_by_identity.items()
    }
    lineage_hash_by_identity = {
        identity: _stable_json_sha256(item)
        for identity, item in lineage_by_identity.items()
    }
    event_digest = frozen_event_digest(stages)
    profiles = build_permutations(
        candidates,
        candidate_ids,
        query_identity=query_identity,
        protocol=protocol,
    )
    result = qualify_candidate_pool(
        analysis,
        candidates,
        current=current,
        registry=registry,
        refs_by_identity=refs_by_identity,
        lineage_hash_by_identity=lineage_hash_by_identity,
        event_digest=event_digest,
        profiles=profiles,
        sources=sources,
        top_k=int(protocol["reconstruction"]["top_k"]),
        query_identity=query_identity,
        case_order=case_order,
    )
    return (
        {
            **base,
            **result["case"],
            "reconstruction": {
                "initial_deduplicated_exact": True,
                "initial_judged_exact": True,
                "initial_reranked_exact": True,
                "final_returned_exact": True,
                "field_lineage_complete": True,
                "frozen_event_digest": event_digest,
            },
        },
        result["candidates"],
        result["ties"],
        requests.observed_keys,
    )


def qualify_candidate_pool(
    analysis: QueryAnalysis,
    candidates: Sequence[Paper],
    *,
    current: VariantResult,
    registry: IdentityRegistry,
    refs_by_identity: Mapping[str, Sequence[str]],
    lineage_hash_by_identity: Mapping[str, str],
    event_digest: str,
    profiles: Mapping[str, Sequence[int]],
    sources: Sequence[str],
    top_k: int,
    query_identity: str,
    case_order: int,
) -> dict[str, Any]:
    """Qualify v2 on one immutable candidate pool and fixed permutations."""

    canonical_ids = registry.labels(current.candidates)
    current_ranked_ids = registry.labels([item.paper for item in current.ranked])
    current_returned_ids = registry.labels([item.paper for item in current.returned])
    judgement_by_id = {
        identity: judgement
        for identity, judgement in zip(canonical_ids, current.judgements, strict=True)
    }
    traces_by_id = {
        identity: trace_deterministic_tiebreak_v2(
            analysis,
            judgement,
            index,
            source_record_refs=refs_by_identity.get(identity, ()),
        )
        for index, (identity, judgement) in enumerate(
            zip(canonical_ids, current.judgements, strict=True)
        )
    }
    groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for identity in canonical_ids:
        groups[tuple(traces_by_id[identity]["primary_sort_key"])].append(identity)
    tie_groups = [values for values in groups.values() if len(values) > 1]
    tie_by_identity = {
        identity: values for values in tie_groups for identity in values
    }

    permutation_projections: dict[str, dict[str, Any]] = {}
    for name, indexes in profiles.items():
        if sorted(indexes) != list(range(len(candidates))):
            raise TieBreakQualificationError("invalid_candidate_permutation")
        permuted = [candidates[index].model_copy(deep=True) for index in indexes]
        variant = rank_v2(
            analysis,
            permuted,
            registry=registry,
            refs_by_identity=refs_by_identity,
            top_k=top_k,
        )
        projection = semantic_projection(
            variant,
            registry=registry,
            lineage_hash_by_identity=lineage_hash_by_identity,
            event_digest=event_digest,
        )
        permutation_projections[name] = projection
    canonical_v2 = permutation_projections["canonical_input"]
    canonical_digest = _stable_json_sha256(canonical_v2)
    unstable_profiles = [
        name
        for name, value in permutation_projections.items()
        if _stable_json_sha256(value) != canonical_digest
    ]
    if unstable_profiles:
        raise TieBreakQualificationError(
            "v2_permutation_instability:" + ",".join(sorted(unstable_profiles))
        )

    v2_ranked_ids = list(canonical_v2["ranked_ids"])
    v2_returned_ids = list(canonical_v2["returned_ids"])
    primary_by_id = {
        identity: tuple(traces_by_id[identity]["primary_sort_key"])
        for identity in canonical_ids
    }
    non_tie_order_changed = non_tie_order_drift(
        current_ranked_ids, v2_ranked_ids, primary_by_id
    )
    top20_membership_changed = set(current_ranked_ids[:top_k]) != set(
        v2_ranked_ids[:top_k]
    )
    returned_membership_changed = set(current_returned_ids) != set(v2_returned_ids)
    authoritative_identity_changed = set(current_ranked_ids) != set(v2_ranked_ids)
    score_or_category_changed = any(
        canonical_v2["candidate_semantics"][identity]
        != {
            "category": judgement_by_id[identity].category,
            "final_score": next(
                item.final_score
                for item in current.ranked
                if registry.label(item.paper) == identity
            ),
            "judgement_score": judgement_by_id[identity].score,
            "lineage_sha256": lineage_hash_by_identity[identity],
        }
        for identity in canonical_ids
    )
    not_qualified_reasons = sorted(
        reason
        for reason, changed in {
            "authoritative_identity_changed": authoritative_identity_changed,
            "non_tie_order_changed": non_tie_order_changed,
            "returned_membership_changed": returned_membership_changed,
            "score_or_category_changed": score_or_category_changed,
            "top20_membership_changed": top20_membership_changed,
        }.items()
        if changed
    )
    current_positions = {
        identity: index + 1 for index, identity in enumerate(current_ranked_ids)
    }
    v2_positions = {identity: index + 1 for index, identity in enumerate(v2_ranked_ids)}
    title_values = sorted(
        {str(traces_by_id[identity]["primary_sort_key"][5]) for identity in canonical_ids}
    )
    title_ordinals = {value: index for index, value in enumerate(title_values)}
    candidate_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    for identity in canonical_ids:
        trace = traces_by_id[identity]
        raw_primary = list(trace["primary_sort_key"])
        safe_primary = [*raw_primary[:5], title_ordinals[str(raw_primary[5])]]
        group = tie_by_identity.get(identity)
        tie_identity = (
            opaque_tie_identity(query_identity, safe_primary) if group else None
        )
        current_rank = current_positions[identity]
        v2_rank = v2_positions[identity]
        candidate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_order": case_order,
                "query_identity": query_identity,
                "candidate_identity": identity,
                "source_provenance": [
                    source
                    for source in sources
                    if source in set(judgement_by_id[identity].paper.sources)
                ],
                "primary_sort_key": safe_primary,
                "stable_key": str(trace["stable_key"]),
                "stable_key_kind": str(trace["stable_key_kind"]),
                "exact_tie": group is not None,
                "tie_identity": tie_identity,
                "current_rank": current_rank,
                "v2_rank": v2_rank,
                "rank_changed": current_rank != v2_rank,
                "current_top20": current_rank <= top_k,
                "v2_top20": v2_rank <= top_k,
                "current_returned": identity in set(current_returned_ids),
                "v2_returned": identity in set(v2_returned_ids),
                "category": judgement_by_id[identity].category,
                "judgement_score": judgement_by_id[identity].score,
                "lineage_sha256": lineage_hash_by_identity[identity],
            }
        )
    for group in tie_groups:
        primary = list(traces_by_id[group[0]]["primary_sort_key"])
        safe_primary = [*primary[:5], title_ordinals[str(primary[5])]]
        current_members = [value for value in current_ranked_ids if value in set(group)]
        v2_members = [value for value in v2_ranked_ids if value in set(group)]
        ranks = [current_positions[value] for value in group]
        tie_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_order": case_order,
                "query_identity": query_identity,
                "tie_identity": opaque_tie_identity(query_identity, safe_primary),
                "group_size": len(group),
                "source_union": sorted(
                    {
                        source
                        for identity in group
                        for source in judgement_by_id[identity].paper.sources
                    }
                ),
                "primary_sort_key": safe_primary,
                "crosses_top20_cutline": min(ranks) <= top_k < max(ranks),
                "current_member_order": current_members,
                "v2_member_order": v2_members,
                "internal_order_changed": current_members != v2_members,
                "top20_membership_changed": {
                    value for value in current_members if current_positions[value] <= top_k
                }
                != {value for value in v2_members if v2_positions[value] <= top_k},
                "permutation_stable": True,
            }
        )
    return {
        "case": {
            "candidate_count": len(candidates),
            "tie_group_count": len(tie_groups),
            "tied_candidate_count": sum(len(value) for value in tie_groups),
            "cutline_tie_group_count": sum(
                bool(item["crosses_top20_cutline"]) for item in tie_rows
            ),
            "current_v2_changed_candidate_count": sum(
                current_positions[value] != v2_positions[value]
                for value in canonical_ids
            ),
            "top20_membership_changed": top20_membership_changed,
            "returned_membership_changed": returned_membership_changed,
            "non_tie_order_changed": non_tie_order_changed,
            "authoritative_identity_changed": authoritative_identity_changed,
            "score_or_category_changed": score_or_category_changed,
            "not_qualified_reasons": not_qualified_reasons,
            "permutation_profile_count": len(profiles),
            "permutation_semantic_sha256": canonical_digest,
            "all_permutations_byte_equivalent": True,
            "event_semantics_equal": True,
            "field_lineage_semantics_equal": True,
        },
        "candidates": candidate_rows,
        "ties": tie_rows,
    }


def rank_v2(
    analysis: QueryAnalysis,
    candidates: Sequence[Paper],
    *,
    registry: IdentityRegistry,
    refs_by_identity: Mapping[str, Sequence[str]],
    top_k: int,
) -> VariantResult:
    judgements = judge_papers(
        analysis,
        [paper.model_copy(deep=True) for paper in candidates],
        use_llm=False,
        policy="current_rules",
        config=CURRENT_RULES_CONFIG,
    )
    refs = {
        index: refs_by_identity.get(registry.label(judgement.paper), ())
        for index, judgement in enumerate(judgements)
    }
    try:
        ranked = rerank_papers(
            analysis,
            judgements,
            top_k=len(judgements),
            tie_break_policy=DETERMINISTIC_TIEBREAK_V2,
            source_record_refs=refs,
        )
    except DeterministicTieBreakUnavailable as exc:
        raise TieBreakNotEligible(str(exc)) from exc
    returned = select_ranked_results(
        {"ranked_papers": ranked[:top_k]}, policy="highly_and_partial"
    )
    return VariantResult(list(candidates), judgements, ranked, returned)


def semantic_projection(
    variant: VariantResult,
    *,
    registry: IdentityRegistry,
    lineage_hash_by_identity: Mapping[str, str],
    event_digest: str,
) -> dict[str, Any]:
    candidate_ids = registry.labels(variant.candidates)
    final_score_by_id = {
        registry.label(item.paper): item.final_score for item in variant.ranked
    }
    return {
        "ranked_ids": registry.labels([item.paper for item in variant.ranked]),
        "returned_ids": registry.labels([item.paper for item in variant.returned]),
        "candidate_semantics": {
            identity: {
                "category": judgement.category,
                "final_score": final_score_by_id[identity],
                "judgement_score": judgement.score,
                "lineage_sha256": lineage_hash_by_identity[identity],
            }
            for identity, judgement in zip(
                candidate_ids, variant.judgements, strict=True
            )
        },
        "event_semantics_sha256": event_digest,
    }


def build_permutations(
    candidates: Sequence[Paper],
    candidate_ids: Sequence[str],
    *,
    query_identity: str,
    protocol: Mapping[str, Any],
) -> dict[str, list[int]]:
    size = len(candidates)
    canonical = list(range(size))
    source_order = [str(value) for value in protocol["permutations"]["source_order"]]
    source_groups: dict[str, list[int]] = {source: [] for source in source_order}
    source_groups["unattributed"] = []
    for index, paper in enumerate(candidates):
        source = next(
            (value for value in source_order if value in set(paper.sources)),
            "unattributed",
        )
        source_groups[source].append(index)
    reverse_blocks = [
        index
        for source in reversed([*source_order, "unattributed"])
        for index in source_groups[source]
    ]
    rotated_sources = [*source_order[1:], source_order[0], "unattributed"]
    rotate_blocks = [
        index for source in rotated_sources for index in source_groups[source]
    ]
    random_order = list(canonical)
    seed = int(protocol["permutations"]["deterministic_random_seed"])
    seed += int(hashlib.sha256(query_identity.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(random_order)
    shard_count = int(protocol["permutations"]["shard_count"])
    shard_groups = {index: [] for index in range(shard_count)}
    for index, identity in enumerate(candidate_ids):
        shard = int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16) % shard_count
        shard_groups[shard].append(index)
    shard_completion = [
        index for shard in (2, 0, 1) for index in shard_groups[shard]
    ]
    return {
        "canonical_input": canonical,
        "reverse_input": list(reversed(canonical)),
        "reverse_source_blocks": reverse_blocks,
        "rotate_source_blocks": rotate_blocks,
        "deterministic_random": random_order,
        "shard_completion_2_0_1": shard_completion,
    }


def non_tie_order_drift(
    current: Sequence[str],
    candidate: Sequence[str],
    primary_by_id: Mapping[str, tuple[Any, ...]],
) -> bool:
    current_positions = {value: index for index, value in enumerate(current)}
    candidate_positions = {value: index for index, value in enumerate(candidate)}
    if set(current_positions) != set(candidate_positions):
        return True
    for left_index, left in enumerate(current):
        for right in current[left_index + 1 :]:
            if primary_by_id[left] == primary_by_id[right]:
                continue
            if candidate_positions[left] > candidate_positions[right]:
                return True
    return False


def frozen_event_digest(stages: Mapping[str, Mapping[str, Any]]) -> str:
    projection = [
        {
            "stage": str(item.get("stage")),
            "status": str(item.get("status")),
            "skipped_reason": item.get("skipped_reason"),
            "candidate_count": len(item.get("candidates") or []),
            "retrieval_call_count": len(item.get("retrieval_calls") or []),
        }
        for item in stages.values()
    ]
    return _stable_json_sha256(projection)


def opaque_tie_identity(query_identity: str, primary_key: Sequence[Any]) -> str:
    return "tie:" + _stable_json_sha256(
        {"query_identity": query_identity, "primary_key": list(primary_key)}
    )[:24]


def aggregate_analysis(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    ties: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    observed_snapshot_key_count: int,
) -> dict[str, Any]:
    reasons = sorted(
        {
            str(reason)
            for item in included
            for reason in item.get("not_qualified_reasons") or []
        }
    )
    status = "not_qualified" if reasons else "qualified_for_review"
    exit_code = EXIT_NOT_QUALIFIED if reasons else EXIT_QUALIFIED
    tie_source_counts = Counter(
        source for item in ties for source in item.get("source_union") or []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": status,
        "exit_code": exit_code,
        "implementation_base_commit": protocol["implementation_base_commit"],
        "implementation_hashes": {
            "qualification_sha256": _sha256(Path(__file__).resolve()),
            "reranker_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "agents" / "reranker.py"
            ),
        },
        "protocol_sha256": protocol_sha256,
        "inputs": dict(sorted(input_hashes.items())),
        "closure": {
            "record_case_count": len(included) + len(excluded),
            "included_main_case_count": len(included),
            "excluded_no_successful_source_count": len(excluded),
            "candidate_count": len(candidates),
            "observed_snapshot_key_count": observed_snapshot_key_count,
            "component_count": len(
                {str(item["component_identity"]) for item in included}
            ),
        },
        "policy": {
            "production_default": DEFAULT_TIEBREAK_POLICY,
            "v2": deterministic_tiebreak_v2_catalog(),
            "v2_enabled_by_default": False,
            "exact_tie_float_tolerance": None,
        },
        "ties": {
            "group_count": len(ties),
            "candidate_count": sum(int(item["group_size"]) for item in ties),
            "maximum_group_size": max(
                (int(item["group_size"]) for item in ties), default=0
            ),
            "cutline_group_count": sum(
                bool(item["crosses_top20_cutline"]) for item in ties
            ),
            "internal_order_changed_group_count": sum(
                bool(item["internal_order_changed"]) for item in ties
            ),
            "top20_membership_changed_group_count": sum(
                bool(item["top20_membership_changed"]) for item in ties
            ),
            "source_attribution_counts": dict(sorted(tie_source_counts.items())),
            "groups": sorted(
                [
                    {
                        key: item[key]
                        for key in (
                            "query_identity",
                            "tie_identity",
                            "group_size",
                            "source_union",
                            "crosses_top20_cutline",
                            "current_member_order",
                            "v2_member_order",
                            "internal_order_changed",
                            "top20_membership_changed",
                            "permutation_stable",
                        )
                    }
                    for item in ties
                ],
                key=lambda item: (item["query_identity"], item["tie_identity"]),
            ),
        },
        "current_v2_comparison": {
            "changed_candidate_count": sum(
                bool(item["rank_changed"]) for item in candidates
            ),
            "changed_query_count": sum(
                int(item["current_v2_changed_candidate_count"]) > 0
                for item in included
            ),
            "top20_membership_changed_query_count": sum(
                bool(item["top20_membership_changed"]) for item in included
            ),
            "returned_membership_changed_query_count": sum(
                bool(item["returned_membership_changed"]) for item in included
            ),
            "non_tie_order_changed_query_count": sum(
                bool(item["non_tie_order_changed"]) for item in included
            ),
            "authoritative_identity_changed_query_count": sum(
                bool(item["authoritative_identity_changed"]) for item in included
            ),
            "score_or_category_changed_query_count": sum(
                bool(item["score_or_category_changed"]) for item in included
            ),
            "event_semantics_changed_query_count": sum(
                not bool(item["event_semantics_equal"]) for item in included
            ),
            "field_lineage_changed_query_count": sum(
                not bool(item["field_lineage_semantics_equal"]) for item in included
            ),
        },
        "permutation_stability": {
            "profile_count_per_query": len(
                protocol["permutations"]["fixed_profiles"]
            ),
            "profiles": list(protocol["permutations"]["fixed_profiles"]),
            "stable_query_count": sum(
                bool(item["all_permutations_byte_equivalent"])
                for item in included
            ),
            "unstable_query_count": sum(
                not bool(item["all_permutations_byte_equivalent"])
                for item in included
            ),
        },
        "qualification": {
            "status": status,
            "not_qualified_reasons": reasons,
            "automatic_enable_permitted": False,
            "review_required": status == "qualified_for_review",
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
            "scope": "deterministic_tie_break_reproducibility_only",
            "relevance_claim_permitted": False,
            "precision_recall_f1_or_official_score": False,
            "warnings": list(protocol["warnings"]),
        },
    }


def write_analysis(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    ties: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protocol_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate": root / "aggregate.json",
        "candidate_diagnostics": root / "candidate_diagnostics.jsonl",
        "case_diagnostics": root / "case_diagnostics.jsonl",
        "tie_groups": root / "tie_groups.jsonl",
        "protocol": root / "protocol.json",
    }
    _write_json(paths["aggregate"], aggregate)
    _write_jsonl(
        paths["candidate_diagnostics"],
        sorted(
            candidates,
            key=lambda item: (int(item["case_order"]), str(item["candidate_identity"])),
        ),
    )
    _write_jsonl(
        paths["case_diagnostics"],
        sorted(cases, key=lambda item: int(item["case_order"])),
    )
    _write_jsonl(
        paths["tie_groups"],
        sorted(
            ties,
            key=lambda item: (int(item["case_order"]), str(item["tie_identity"])),
        ),
    )
    _write_json(paths["protocol"], _read_json(Path(protocol_path).resolve()))
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
            for name, path in sorted(paths.items())
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_analysis(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("analysis") != CONTRACT_VERSION:
        raise TieBreakQualificationError("manifest_contract_mismatch")
    for value in manifest.get("files", {}).values():
        path = root / str(value["path"])
        if not path.is_file() or path.stat().st_size != int(value["size"]):
            raise TieBreakQualificationError("output_missing_or_size_drift")
        if _sha256(path) != str(value["sha256"]):
            raise TieBreakQualificationError("output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") not in {"qualified_for_review", "not_qualified"}:
        raise TieBreakQualificationError("analysis_terminal_status_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": aggregate["status"],
        "exit_code": int(aggregate["exit_code"]),
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
    }


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise TieBreakQualificationError("network_attempt_detected")

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
