"""Gold-free source reliability funnel over the frozen Record160 Replay.

The module consumes only query-visible planning fields, frozen retrieval
Snapshots, and production pipeline diagnostics.  It never loads a dataset,
evaluator labels, or quality metrics.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import socket
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.identity import build_identity_profile
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.schemas import RetrievalSnapshotEntry
from scholar_agent.evaluation.snapshots.store import (
    SnapshotIntegrityError,
    SnapshotMissingError,
)
from scholar_agent.evaluation.source_fusion_ablation import (
    IdentityRegistry,
    rank_variant,
    summarize_distribution,
    validate_full_reconstruction,
)


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "source_reliability_diagnostics_v1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_CATEGORIES = {"highly_relevant", "partially_relevant"}
_BOOLEAN_RE = re.compile(r"(?<!\w)(?:and|or|not)(?!\w)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_QUOTE_CHARS = frozenset("'\"‘’“”")
_HTTP_STATUS_RE = re.compile(r"(?:^|\D)[45]\d{2}(?:\D|$)")


class SourceReliabilityError(RuntimeError):
    """The frozen contract or an accounting invariant was violated."""


class SourceReliabilityNotEligible(SourceReliabilityError):
    """The frozen data cannot support the preregistered funnel."""


@dataclass
class RequestAudit:
    papers_by_source: dict[str, list[Paper]]
    ordered_batches: list[tuple[str, list[Paper]]]
    source_records: dict[str, dict[str, Any]]
    unassigned_terminal_counts: Counter[str]
    observed_keys: set[str]


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path).expanduser().resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise SourceReliabilityError("unsupported_protocol")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }:
        raise SourceReliabilityError("offline_protocol_drift")
    if value.get("sources") != [
        "openalex",
        "arxiv",
        "semantic_scholar",
        "pubmed",
    ]:
        raise SourceReliabilityError("source_order_drift")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_failure_or_yield",
    ]:
        raise SourceReliabilityError("selection_contract_drift")
    if value.get("funnel", {}).get("stages") != [
        "logical_request",
        "authoritative_snapshot",
        "raw_provider_record",
        "parsed_record",
        "canonical_identity",
        "source_unique_identity",
        "global_unique_identity",
        "budget_retained_identity",
        "constraint_survivor",
        "top20_identity",
    ]:
        raise SourceReliabilityError("funnel_contract_drift")
    return value


def run_source_reliability_diagnostics(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the preregistered diagnostic without loading gold or opening network."""

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
        raise SourceReliabilityNotEligible("snapshot_tree_hash_drift")
    if sum(path.is_file() for path in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise SourceReliabilityNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, protocol)
    rows = _read_reliability_rows(results_path)
    if len(rows) != int(protocol["analysis_population"]["record_case_count"]):
        raise SourceReliabilityNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[: len(rows)]:
        raise SourceReliabilityNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise SourceReliabilityNotEligible("missing_frozen_component_assignment")

    store = SnapshotStore(snapshot_dir)
    attempts = {"network": 0}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    with _forbid_network(attempts):
        for case_order, row in enumerate(rows):
            case, case_keys = analyze_case(
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

    _validate_population(included, excluded, protocol)
    expected_keys = int(frozen["observed_snapshot_key_count"])
    if len(observed_keys) != expected_keys:
        raise SourceReliabilityNotEligible("observed_snapshot_key_count_drift")
    after_tree = _tree_sha256(snapshot_dir)
    if after_tree != before_tree:
        raise SourceReliabilityError("snapshot_tree_changed")
    if attempts["network"]:
        raise SourceReliabilityError("network_attempt_detected")

    cases = sorted([*included, *excluded], key=lambda item: int(item["case_order"]))
    aggregate = aggregate_diagnostics(
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
        observed_snapshot_key_count=len(observed_keys),
    )
    return cases, aggregate


def analyze_case(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    store: Any,
    component_id: str,
    case_order: int,
) -> tuple[dict[str, Any], set[str]]:
    stages = {
        str(item.get("stage")): item for item in row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
        "final_returned",
    }
    if not required.issubset(stages):
        raise SourceReliabilityNotEligible("required_frozen_stage_missing")
    sources = [str(value) for value in protocol["sources"]]
    request_audit = audit_retrieval_requests(
        stages["initial_retrieval"],
        config=config,
        store=store,
        sources=sources,
    )
    query_states = {
        source: source_query_state(request_audit.source_records[source])
        for source in sources
    }
    successful_source_count = sum(
        int(request_audit.source_records[source]["snapshot_success_count"]) > 0
        for source in sources
    )
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": (
            "included_main_analysis"
            if successful_source_count
            else "excluded_no_successful_source"
        ),
        "case_order": case_order,
        "query_identity": _opaque_query_identity(str(row["case_id"])),
        "component_identity": _opaque_component_identity(str(component_id)),
        "successful_source_count": successful_source_count,
        "query_structure": classify_query_structure(str(row["original_query"])),
        "unassigned_terminal_counts": dict(
            sorted(request_audit.unassigned_terminal_counts.items())
        ),
        "source_diagnostics": {},
    }
    if not successful_source_count:
        base["source_diagnostics"] = {
            source: finalize_source_request_record(
                request_audit.source_records[source], query_state=query_states[source]
            )
            for source in sources
        }
        return base, request_audit.observed_keys

    analysis = QueryAnalysis.model_validate(row["query_analysis"])
    ordered_raw = [
        paper.model_copy(deep=True)
        for _source, batch in request_audit.ordered_batches
        for paper in batch
    ]
    global_before_budget = deduplicate_papers(ordered_raw)
    candidate_limit = int(config["budgets"]["max_candidate_papers"])
    budget_candidates = list(global_before_budget)
    if len(budget_candidates) > candidate_limit:
        budget_candidates = stable_source_coverage_truncate(
            budget_candidates,
            limit=candidate_limit,
            source_order=sources,
        )
    frozen_candidates = align_papers_to_diagnostics(
        budget_candidates, stages["initial_deduplicated"]["candidates"]
    )
    full = rank_variant(analysis, frozen_candidates, top_k=20)
    validate_full_reconstruction(full, stages)

    registry = IdentityRegistry()
    source_unique_papers = {
        source: deduplicate_papers(request_audit.papers_by_source[source])
        for source in sources
    }
    source_sets = {
        source: set(registry.labels(source_unique_papers[source])) for source in sources
    }
    global_set = set(registry.labels(global_before_budget))
    budget_set = set(registry.labels(full.candidates))
    constraint_set = set(
        registry.labels(
            [
                item.paper
                for item in full.judgements
                if item.category in _ALLOWED_CATEGORIES
            ]
        )
    )
    top20_set = set(registry.labels([item.paper for item in full.returned]))
    reconstruction_digest = _stable_json_sha256(
        {
            "candidates": registry.labels(full.candidates),
            "returned": registry.labels([item.paper for item in full.returned]),
            "query_states": query_states,
        }
    )

    source_diagnostics: dict[str, Any] = {}
    for source in sources:
        record = request_audit.source_records[source]
        source_set = source_sets[source]
        other_set = set().union(
            *(source_sets[item] for item in sources if item != source)
        )
        funnel = derive_source_funnel(
            parsed_record_count=int(record["parsed_record_count"]),
            canonical_record_count=int(record["canonical_record_count"]),
            source_identity_set=source_set,
            global_identity_set=global_set,
            budget_identity_set=budget_set,
            constraint_identity_set=constraint_set,
            top20_identity_set=top20_set,
            raw_provider_record_count=None,
        )
        source_diagnostics[source] = {
            **finalize_source_request_record(record, query_state=query_states[source]),
            "funnel": funnel,
            "cross_source_redundant_identity_count": len(source_set & other_set),
            "primary_outcome": classify_primary_outcome(
                funnel,
                record,
                query_state=query_states[source],
            ),
        }
    return (
        {
            **base,
            "reconstruction": {
                "initial_deduplicated_exact": True,
                "initial_judged_exact": True,
                "initial_reranked_exact": True,
                "final_returned_exact": True,
                "digest": reconstruction_digest,
            },
            "source_diagnostics": source_diagnostics,
        },
        request_audit.observed_keys,
    )


def audit_retrieval_requests(
    initial: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    store: Any,
    sources: Sequence[str],
) -> RequestAudit:
    papers_by_source = {source: [] for source in sources}
    ordered_batches: list[tuple[str, list[Paper]]] = []
    source_records = {source: new_source_request_record() for source in sources}
    unassigned: Counter[str] = Counter()
    seen: dict[str, tuple[str, str, int]] = {}
    observed_keys: set[str] = set()
    for call in initial.get("retrieval_calls") or []:
        source = str(call.get("source") or "")
        if source not in source_records:
            if (
                source == "subquery"
                and bool(call.get("logical_call_executed"))
                and not call.get("snapshot_key")
                and int(call.get("returned_count") or 0) == 0
                and str(call.get("terminal_status") or "") == "timeout"
            ):
                unassigned["subquery_timeout"] += 1
                continue
            raise SourceReliabilityNotEligible("unknown_source_in_frozen_call")
        record = source_records[source]
        record["logical_request_count"] += 1
        executed = bool(call.get("logical_call_executed"))
        record["logical_executed_count"] += int(executed)
        key = str(call.get("snapshot_key") or "")
        if not executed or not key:
            request_state = classify_no_snapshot_call(call)
            record["request_state_counts"][request_state] += 1
            if request_state == "unknown_failure":
                record["unknown_evidence_count"] += 1
            continue

        record["snapshot_reference_count"] += 1
        signature = (
            source,
            str(call.get("adapted_query") or ""),
            int(call.get("returned_count") or 0),
        )
        if key in seen:
            record["duplicate_snapshot_reference_count"] += 1
            if seen[key] != signature:
                raise SourceReliabilityError("duplicate_snapshot_reference_signature_drift")
            continue
        seen[key] = signature
        observed_keys.add(key)
        record["unique_snapshot_count"] += 1
        try:
            entry = store.read_retrieval(key)
        except SnapshotMissingError:
            record["request_state_counts"]["snapshot_missing"] += 1
            record["snapshot_missing_count"] += 1
            record["unknown_evidence_count"] += 1
            continue
        except SnapshotIntegrityError as exc:
            raise SourceReliabilityError("snapshot_integrity_violation") from exc
        _validate_snapshot_signature(entry, call, config, source)
        unknown_fields = inspect_unknown_snapshot_fields(store, key)
        if unknown_fields:
            raise SourceReliabilityError("unknown_snapshot_schema_fields")
        record["authoritative_snapshot_count"] += 1
        record["recorded_request_count"] += int(entry.diagnostics.request_count)
        record["recorded_retry_count"] += int(entry.diagnostics.retry_count)
        record["recorded_error_count"] += int(entry.diagnostics.error_count)
        record["recorded_cache_hit_count"] += int(entry.diagnostics.cache_hit_count)
        record["recorded_rate_limit_wait_seconds"] += float(
            entry.diagnostics.rate_limit_wait_seconds
        )
        record["recorded_latency_seconds"] += float(entry.recorded_latency_seconds)
        if entry.status == "failed":
            if entry.papers or int(call.get("returned_count") or 0):
                raise SourceReliabilityError("failed_snapshot_exposes_records")
            category = classify_failure(entry.error_message, entry.warnings)
            record["request_state_counts"][category] += 1
            record["failure_category_counts"][category] += 1
            record["snapshot_failed_count"] += 1
            if category == "unknown_failure":
                record["unknown_evidence_count"] += 1
            continue

        batch = [paper.model_copy(deep=True) for paper in entry.papers]
        if int(call.get("returned_count") or 0) != len(batch):
            raise SourceReliabilityError("parsed_record_count_not_conserved")
        record["request_state_counts"]["success"] += 1
        record["snapshot_success_count"] += 1
        record["raw_provider_record_unknown_request_count"] += 1
        record["successful_request_yields"].append(len(batch))
        record["legal_empty_success_count"] += int(not batch)
        record["success_nonempty_count"] += int(bool(batch))
        record["parsed_record_count"] += len(batch)
        canonical_count = 0
        for paper in batch:
            try:
                build_identity_profile(paper)
            except (AttributeError, TypeError, ValueError):
                continue
            canonical_count += 1
        record["canonical_record_count"] += canonical_count
        papers_by_source[source].extend(batch)
        ordered_batches.append((source, batch))
    for source, record in source_records.items():
        classified = sum(record["request_state_counts"].values()) + int(
            record["duplicate_snapshot_reference_count"]
        )
        if classified != int(record["logical_request_count"]):
            raise SourceReliabilityError(f"logical_request_accounting_drift:{source}")
        if int(record["canonical_record_count"]) > int(record["parsed_record_count"]):
            raise SourceReliabilityError("canonical_record_count_exceeds_parsed")
    return RequestAudit(
        papers_by_source,
        ordered_batches,
        source_records,
        unassigned,
        observed_keys,
    )


def new_source_request_record() -> dict[str, Any]:
    return {
        "logical_request_count": 0,
        "logical_executed_count": 0,
        "snapshot_reference_count": 0,
        "unique_snapshot_count": 0,
        "duplicate_snapshot_reference_count": 0,
        "snapshot_missing_count": 0,
        "authoritative_snapshot_count": 0,
        "snapshot_success_count": 0,
        "snapshot_failed_count": 0,
        "legal_empty_success_count": 0,
        "success_nonempty_count": 0,
        "raw_provider_record_count": None,
        "raw_provider_record_unknown_request_count": 0,
        "parsed_record_count": 0,
        "canonical_record_count": 0,
        "request_state_counts": Counter(),
        "failure_category_counts": Counter(),
        "successful_request_yields": [],
        "unknown_evidence_count": 0,
        "recorded_request_count": 0,
        "recorded_retry_count": 0,
        "recorded_error_count": 0,
        "recorded_cache_hit_count": 0,
        "recorded_rate_limit_wait_seconds": 0.0,
        "recorded_latency_seconds": 0.0,
    }


def finalize_source_request_record(
    record: Mapping[str, Any], *, query_state: str
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in record.items()
        if key not in {"request_state_counts", "failure_category_counts"}
    }
    result["request_state_counts"] = dict(
        sorted(Counter(record["request_state_counts"]).items())
    )
    result["failure_category_counts"] = dict(
        sorted(Counter(record["failure_category_counts"]).items())
    )
    result["query_state"] = query_state
    result["recorded_rate_limit_wait_seconds"] = _rounded(
        float(result["recorded_rate_limit_wait_seconds"])
    )
    result["recorded_latency_seconds"] = _rounded(
        float(result["recorded_latency_seconds"])
    )
    return result


def classify_no_snapshot_call(call: Mapping[str, Any]) -> str:
    terminal = str(call.get("terminal_status") or "").strip().casefold()
    skipped = str(call.get("source_skipped_reason") or "").strip().casefold()
    if not bool(call.get("logical_call_executed")) or skipped or terminal == "not_started":
        return "not_started"
    if terminal == "success":
        raise SourceReliabilityError("success_without_snapshot")
    if terminal:
        return classify_failure(terminal, [])
    return "unknown_failure"


def classify_failure(error_message: str | None, warnings: Sequence[str]) -> str:
    text = " ".join([str(error_message or ""), *[str(value) for value in warnings]])
    normalized = text.casefold()
    if any(
        token in normalized
        for token in (
            "schema",
            "parse",
            "decode",
            "malformed",
            "validation",
            "json",
            "xml",
            "payload_shape",
            "payload-shape",
        )
    ):
        return "parse_failure"
    if (
        "http" in normalized
        or "status_" in normalized
        or _HTTP_STATUS_RE.search(normalized)
    ):
        return "http_failure"
    if any(
        token in normalized
        for token in (
            "timeout",
            "timed out",
            "dns",
            "tls",
            "socket",
            "connection",
            "network",
        )
    ):
        return "transport_failure"
    if any(
        token in normalized
        for token in (
            "provider",
            "source_failure",
            "source failure",
            "source_outage",
            "source outage",
            "failed",
        )
    ):
        return "provider_failure"
    return "unknown_failure"


def source_query_state(record: Mapping[str, Any]) -> str:
    counts = Counter(record["request_state_counts"])
    success = counts["success"] > 0
    failed = sum(
        counts[name]
        for name in (
            "snapshot_missing",
            "transport_failure",
            "http_failure",
            "provider_failure",
            "parse_failure",
            "unknown_failure",
        )
    ) > 0
    if success and failed:
        return "partial_failure"
    if success:
        return "success"
    if failed:
        return "failed"
    return "not_started"


def derive_source_funnel(
    *,
    parsed_record_count: int,
    canonical_record_count: int,
    source_identity_set: set[str],
    global_identity_set: set[str],
    budget_identity_set: set[str],
    constraint_identity_set: set[str],
    top20_identity_set: set[str],
    raw_provider_record_count: int | None,
) -> dict[str, Any]:
    source_unique = len(source_identity_set)
    global_unique = len(source_identity_set & global_identity_set)
    budget_retained = len(source_identity_set & budget_identity_set)
    constraint_survivors = len(source_identity_set & constraint_identity_set)
    top20 = len(source_identity_set & top20_identity_set)
    if not (
        canonical_record_count <= parsed_record_count
        and source_unique <= canonical_record_count
        and global_unique <= source_unique
        and budget_retained <= global_unique
        and constraint_survivors <= budget_retained
        and top20 <= constraint_survivors
    ):
        raise SourceReliabilityError("funnel_count_not_monotonic")
    values: dict[str, int | None] = {
        "raw_provider_record_count": raw_provider_record_count,
        "parsed_record_count": parsed_record_count,
        "canonical_record_count": canonical_record_count,
        "source_unique_identity_count": source_unique,
        "global_unique_identity_count": global_unique,
        "budget_retained_identity_count": budget_retained,
        "constraint_survivor_identity_count": constraint_survivors,
        "top20_identity_count": top20,
    }
    pairs = {
        "parse_loss": (raw_provider_record_count, parsed_record_count),
        "canonicalization_loss": (parsed_record_count, canonical_record_count),
        "identity_dedup_loss": (canonical_record_count, source_unique),
        "global_identity_loss": (source_unique, global_unique),
        "global_budget_loss": (global_unique, budget_retained),
        "constraint_loss": (budget_retained, constraint_survivors),
        "top20_selection_loss": (constraint_survivors, top20),
    }
    losses: dict[str, dict[str, Any]] = {}
    for name, (before, after) in pairs.items():
        if before is None or after is None:
            losses[name] = {"count": None, "rate": None, "status": "unknown"}
            continue
        loss = before - after
        losses[name] = {
            "count": loss,
            "rate": _ratio(loss, before),
            "status": "known",
        }
    return {"stages": values, "losses": losses}


def classify_primary_outcome(
    funnel: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    query_state: str,
) -> str:
    stages = funnel["stages"]
    if int(stages["top20_identity_count"] or 0) > 0:
        return "contributed_top20"
    if int(stages["constraint_survivor_identity_count"] or 0) > 0:
        return "valid_not_top20"
    if int(stages["budget_retained_identity_count"] or 0) > 0:
        return "constraint_loss"
    if int(stages["global_unique_identity_count"] or 0) > 0:
        return "global_budget_loss"
    if int(stages["source_unique_identity_count"] or 0) > 0:
        return "dedup_loss"
    if int(stages["parsed_record_count"] or 0) > int(
        stages["canonical_record_count"] or 0
    ):
        return "canonicalization_or_identity_loss"
    if int(record["legal_empty_success_count"]) > 0 and int(
        stages["parsed_record_count"] or 0
    ) == 0:
        return "legal_empty_response"
    if query_state == "not_started":
        return "not_started"
    if query_state in {"failed", "partial_failure"}:
        return "request_failure"
    return "unknown"


def classify_query_structure(query: str) -> dict[str, Any]:
    length = len(query)
    if length <= 80:
        length_bucket = "0_80"
    elif length <= 160:
        length_bucket = "81_160"
    elif length <= 320:
        length_bucket = "161_320"
    else:
        length_bucket = "321_plus"
    non_ascii = [character for character in query if ord(character) > 127]
    if not non_ascii:
        unicode_class = "ascii_only"
    elif any(unicodedata.category(character).startswith("L") for character in non_ascii):
        unicode_class = "non_ascii_letter"
    else:
        unicode_class = "non_ascii_nonletter"
    return {
        "length_bucket": length_bucket,
        "has_quote": any(character in _QUOTE_CHARS for character in query),
        "has_boolean_operator": bool(_BOOLEAN_RE.search(query)),
        "has_year": bool(_YEAR_RE.search(query)),
        "unicode_class": unicode_class,
    }


def aggregate_diagnostics(
    cases: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    observed_snapshot_key_count: int,
) -> dict[str, Any]:
    sources = [str(value) for value in protocol["sources"]]
    source_reports: dict[str, Any] = {}
    for source in sources:
        records = [case["source_diagnostics"][source] for case in cases]
        request_states = Counter()
        failure_categories = Counter()
        query_states = Counter()
        primary_outcomes = Counter()
        funnel_totals: Counter[str] = Counter()
        loss_totals: dict[str, dict[str, Any]] = {}
        successful_yields: list[float] = []
        operational: Counter[str] = Counter()
        operational_float: defaultdict[str, float] = defaultdict(float)
        for record in records:
            request_states.update(record["request_state_counts"])
            failure_categories.update(record["failure_category_counts"])
            query_states[str(record["query_state"])] += 1
            primary_outcomes[str(record["primary_outcome"])] += 1
            successful_yields.extend(
                float(value) for value in record["successful_request_yields"]
            )
            for name, value in record["funnel"]["stages"].items():
                if value is not None:
                    funnel_totals[name] += int(value)
            for name in (
                "logical_request_count",
                "logical_executed_count",
                "snapshot_reference_count",
                "unique_snapshot_count",
                "duplicate_snapshot_reference_count",
                "snapshot_missing_count",
                "authoritative_snapshot_count",
                "snapshot_success_count",
                "snapshot_failed_count",
                "legal_empty_success_count",
                "success_nonempty_count",
                "raw_provider_record_unknown_request_count",
                "unknown_evidence_count",
                "recorded_request_count",
                "recorded_retry_count",
                "recorded_error_count",
                "recorded_cache_hit_count",
            ):
                operational[name] += int(record[name])
            operational_float["recorded_rate_limit_wait_seconds"] += float(
                record["recorded_rate_limit_wait_seconds"]
            )
            operational_float["recorded_latency_seconds"] += float(
                record["recorded_latency_seconds"]
            )
        for loss_name, before_name, after_name in (
            ("parse_loss", "raw_provider_record_count", "parsed_record_count"),
            ("canonicalization_loss", "parsed_record_count", "canonical_record_count"),
            (
                "identity_dedup_loss",
                "canonical_record_count",
                "source_unique_identity_count",
            ),
            (
                "global_identity_loss",
                "source_unique_identity_count",
                "global_unique_identity_count",
            ),
            (
                "global_budget_loss",
                "global_unique_identity_count",
                "budget_retained_identity_count",
            ),
            (
                "constraint_loss",
                "budget_retained_identity_count",
                "constraint_survivor_identity_count",
            ),
            (
                "top20_selection_loss",
                "constraint_survivor_identity_count",
                "top20_identity_count",
            ),
        ):
            if before_name == "raw_provider_record_count":
                loss_totals[loss_name] = {
                    "count": None,
                    "rate": None,
                    "status": "unknown",
                }
                continue
            before = funnel_totals[before_name]
            after = funnel_totals[after_name]
            loss_totals[loss_name] = {
                "count": before - after,
                "rate": _ratio(before - after, before),
                "status": "known",
            }
        source_reports[source] = {
            "query_state_counts": dict(sorted(query_states.items())),
            "queries_with_successful_snapshot_count": sum(
                int(record["snapshot_success_count"]) > 0 for record in records
            ),
            "partial_failure_count": query_states["partial_failure"],
            "partial_failure_rate": _ratio(query_states["partial_failure"], len(cases)),
            "request_state_counts": dict(sorted(request_states.items())),
            "failure_category_counts": dict(sorted(failure_categories.items())),
            "primary_outcome_counts": dict(sorted(primary_outcomes.items())),
            "funnel_totals": {
                **dict(sorted(funnel_totals.items())),
                "raw_provider_record_count": None,
                "cross_source_redundant_identity_attribution_count": sum(
                    int(record["cross_source_redundant_identity_count"])
                    for record in records
                ),
            },
            "stage_losses": loss_totals,
            "legal_empty_success_request_rate": _ratio(
                operational["legal_empty_success_count"],
                operational["snapshot_success_count"],
            ),
            "queries_with_legal_empty_success_count": sum(
                int(record["legal_empty_success_count"]) > 0 for record in records
            ),
            "queries_with_only_legal_empty_success_count": sum(
                int(record["snapshot_success_count"]) > 0
                and int(record["funnel"]["stages"]["parsed_record_count"]) == 0
                for record in records
            ),
            "raw_nonempty_zero_valid_query_count": None,
            "parsed_nonempty_zero_canonical_query_count": sum(
                int(record["funnel"]["stages"]["parsed_record_count"]) > 0
                and int(record["funnel"]["stages"]["canonical_record_count"]) == 0
                for record in records
            ),
            "successful_request_output": {
                "request_distribution": summarize_distribution(successful_yields),
                "cluster_mean_confidence_interval": cluster_success_yield_summary(
                    cases, source, protocol
                ),
            },
            "query_structure_strata": query_structure_strata(cases, source),
            "operational_counts": {
                **dict(sorted(operational.items())),
                "recorded_rate_limit_wait_seconds": _rounded(
                    operational_float["recorded_rate_limit_wait_seconds"]
                ),
                "recorded_latency_seconds": _rounded(
                    operational_float["recorded_latency_seconds"]
                ),
            },
        }
    implementation_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "implementation_base_commit": protocol["implementation_base_commit"],
        "implementation_sha256": _sha256(implementation_path),
        "protocol_sha256": protocol_sha256,
        "inputs": dict(sorted(input_hashes.items())),
        "closure": {
            "record_case_count": len(cases) + len(excluded),
            "included_main_case_count": len(cases),
            "excluded_no_successful_source_count": len(excluded),
            "reconstruction_exact_case_count": sum(
                bool(case.get("reconstruction")) for case in cases
            ),
            "observed_snapshot_key_count": observed_snapshot_key_count,
            "component_count": len(
                {str(case["component_identity"]) for case in cases}
            ),
        },
        "sources": source_reports,
        "unassigned_terminal_counts": dict(
            sorted(
                sum(
                    (
                        Counter(case["unassigned_terminal_counts"])
                        for case in [*cases, *excluded]
                    ),
                    Counter(),
                ).items()
            )
        ),
        "unknown_evidence": {
            "raw_provider_record_count_available": False,
            "reason": "retrieval Snapshots persist adapter-parsed Paper records but not pre-parser provider record counts",
            "unknown_failure_evidence_count": sum(
                int(case["source_diagnostics"][source]["unknown_evidence_count"])
                for case in cases
                for source in sources
            ),
        },
        "silent_loss": {
            "accounting_violation_count": 0,
            "duplicate_snapshot_reference_count": sum(
                int(
                    case["source_diagnostics"][source][
                        "duplicate_snapshot_reference_count"
                    ]
                )
                for case in cases
                for source in sources
            ),
            "unknown_schema_field_count": 0,
            "failed_disguised_as_empty_success_count": 0,
            "status_record_contradiction_count": 0,
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
            "scope": "source_availability_and_pipeline_accounting_only",
            "relevance_claim_permitted": False,
            "precision_recall_or_official_score": False,
            "warnings": list(protocol["warnings"]),
        },
    }


def cluster_success_yield_summary(
    cases: Sequence[Mapping[str, Any]],
    source: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    observed = 0
    for case in cases:
        record = case["source_diagnostics"][source]
        successes = int(record["snapshot_success_count"])
        if not successes:
            continue
        observed += 1
        grouped[str(case["component_identity"])].append(
            float(record["funnel"]["stages"]["parsed_record_count"]) / successes
        )
    if not grouped:
        return {
            "query_count": len(cases),
            "observed_query_count": 0,
            "missing_query_count": len(cases),
            "component_count": 0,
            "mean": None,
            "median": None,
            "confidence_interval_95": [None, None],
        }
    component_values = [
        statistics.fmean(values) for _component, values in sorted(grouped.items())
    ]
    bootstrap = _bootstrap_component_means(
        component_values,
        seed=_derived_seed(int(protocol["bootstrap"]["seed"]), source),
        iterations=int(protocol["bootstrap"]["iterations"]),
    )
    return {
        "query_count": len(cases),
        "observed_query_count": observed,
        "missing_query_count": len(cases) - observed,
        "component_count": len(component_values),
        "mean": _rounded(statistics.fmean(component_values)),
        "median": _rounded(statistics.median(component_values)),
        "confidence_interval_95": [
            _rounded(_percentile(bootstrap, 0.025)),
            _rounded(_percentile(bootstrap, 0.975)),
        ],
    }


def query_structure_strata(
    cases: Sequence[Mapping[str, Any]], source: str
) -> dict[str, Any]:
    dimensions = {
        "length_bucket": ["0_80", "81_160", "161_320", "321_plus"],
        "has_quote": [False, True],
        "has_boolean_operator": [False, True],
        "has_year": [False, True],
        "unicode_class": [
            "ascii_only",
            "non_ascii_letter",
            "non_ascii_nonletter",
        ],
    }
    result: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        strata: dict[str, Any] = {}
        for value in values:
            selected = [
                case for case in cases if case["query_structure"][dimension] == value
            ]
            states = [case["source_diagnostics"][source] for case in selected]
            failed = sum(
                item["query_state"] in {"failed", "partial_failure"}
                for item in states
            )
            legal_empty = sum(int(item["legal_empty_success_count"]) > 0 for item in states)
            unknown = sum(int(item["unknown_evidence_count"]) > 0 for item in states)
            strata[str(value).lower()] = {
                "query_count": len(selected),
                "queries_with_failure_count": failed,
                "queries_with_failure_rate": _ratio(failed, len(selected)),
                "queries_with_legal_empty_count": legal_empty,
                "queries_with_legal_empty_rate": _ratio(legal_empty, len(selected)),
                "queries_with_unknown_evidence_count": unknown,
            }
        result[dimension] = strata
    return result


def inspect_unknown_snapshot_fields(store: Any, key: str) -> list[str]:
    retrieval_dir = getattr(store, "retrieval_dir", None)
    if retrieval_dir is None:
        return []
    path = Path(retrieval_dir) / f"{key}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceReliabilityError("snapshot_envelope_unreadable") from exc
    if not isinstance(raw, dict):
        raise SourceReliabilityError("snapshot_envelope_not_object")
    return sorted(set(raw) - set(RetrievalSnapshotEntry.model_fields))


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
    _write_json(protocol_copy, _read_json(Path(protocol_path).expanduser().resolve()))
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
        raise SourceReliabilityError("analysis_manifest_contract_mismatch")
    for item in manifest.get("files", {}).values():
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise SourceReliabilityError("analysis_output_missing_or_size_drift")
        if _sha256(path) != str(item["sha256"]):
            raise SourceReliabilityError("analysis_output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") != "completed":
        raise SourceReliabilityError("analysis_not_completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
    }


def _read_reliability_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw.get("case_id") or "")
        if not case_id or case_id in seen:
            raise SourceReliabilityNotEligible("invalid_or_duplicate_record_identity")
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
            raise SourceReliabilityNotEligible("invalid_frozen_component_assignment")
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
            raise SourceReliabilityNotEligible(f"frozen_config_drift:{field}")
    if int(config.get("budgets", {}).get("max_candidate_papers") or 0) != int(
        protocol["reconstruction"]["candidate_limit"]
    ):
        raise SourceReliabilityNotEligible("candidate_budget_drift")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise SourceReliabilityNotEligible("experimental_strategy_enabled")
    if (config.get("judgement_config") or {}).get("lexical_normalization_policy") != "off":
        raise SourceReliabilityNotEligible("lexical_normalization_not_default_off")


def _validate_snapshot_signature(
    entry: Any,
    call: Mapping[str, Any],
    config: Mapping[str, Any],
    source: str,
) -> None:
    recorded_terminal = call.get("terminal_status")
    if (
        entry.source != source
        or entry.adapted_query != str(call.get("adapted_query") or "")
        or entry.limit != int(config["top_k"])
        or entry.adapter_policy != str(config["query_adapter_policy"])
        or (
            recorded_terminal is not None
            and entry.status != str(recorded_terminal)
        )
    ):
        raise SourceReliabilityError("frozen_source_request_signature_mismatch")


def _validate_population(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    expected_main = int(protocol["analysis_population"]["main_case_count"])
    expected_excluded = int(protocol["analysis_population"]["excluded_case_count"])
    if len(included) != expected_main or len(excluded) != expected_excluded:
        raise SourceReliabilityError("frozen_population_or_exclusion_drift")
    orders = [int(item["case_order"]) for item in [*included, *excluded]]
    if sorted(orders) != list(range(expected_main + expected_excluded)):
        raise SourceReliabilityError("population_has_omission_or_duplicate")
    if any(
        item.get("analysis_status") != "included_main_analysis" for item in included
    ):
        raise SourceReliabilityError("post_hoc_inclusion_detected")
    if any(
        item.get("analysis_status") != "excluded_no_successful_source"
        for item in excluded
    ):
        raise SourceReliabilityError("unregistered_exclusion_detected")


def _bootstrap_component_means(
    values: Sequence[float], *, seed: int, iterations: int
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    return sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(iterations)
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower]) * (1 - weight) + float(values[upper]) * weight


def _derived_seed(base: int, source: str) -> int:
    digest = hashlib.sha256(
        f"{source}\0successful_request_yield".encode("utf-8")
    ).digest()
    return base ^ int.from_bytes(digest[:8], "big")


def _ratio(numerator: int, denominator: int) -> float | None:
    return _rounded(numerator / denominator) if denominator else None


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _opaque_query_identity(case_id: str) -> str:
    return "query:" + hashlib.sha256(
        ("source-reliability-query-v1\0" + case_id).encode("utf-8")
    ).hexdigest()[:24]


def _opaque_component_identity(component_id: str) -> str:
    return "component:" + hashlib.sha256(
        ("source-reliability-component-v1\0" + component_id).encode("utf-8")
    ).hexdigest()[:24]


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts["network"] += 1
        raise SourceReliabilityError("network_attempt_detected")

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
        raise SourceReliabilityNotEligible(f"frozen_input_hash_drift:{path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceReliabilityError("expected_json_object")
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
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
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


__all__ = [
    "CONTRACT_VERSION",
    "EXIT_COMPLETED",
    "EXIT_NOT_ELIGIBLE",
    "EXIT_USAGE_ERROR",
    "EXIT_VIOLATION",
    "SourceReliabilityError",
    "SourceReliabilityNotEligible",
    "audit_retrieval_requests",
    "classify_failure",
    "classify_primary_outcome",
    "classify_query_structure",
    "cluster_success_yield_summary",
    "derive_source_funnel",
    "load_protocol",
    "run_source_reliability_diagnostics",
    "verify_analysis",
    "write_analysis",
]
