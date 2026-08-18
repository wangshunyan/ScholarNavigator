"""Offline cross-execution determinism gate for SearchService Replay fixtures.

The gate executes existing local ``RetrievalOutput`` fixtures through the real
``SearchService`` pipeline. It does not evaluate gold, calculate quality
metrics, access runtime configuration, or write Snapshot state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal, Protocol
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.fixture_loader import (
    build_fixture_retriever,
    load_retrieval_outputs,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.evaluation.snapshots.store import normalize_snapshot_query
from scholar_agent.core.search_schemas import SearchBudget
from scholar_agent.services.search_service import (
    SearchCancelled,
    SearchService,
    SearchServiceOutput,
)


CONTRACT_VERSION = "execution_determinism_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "execution_determinism_gate"
EXIT_PASSED = 0
EXIT_INVARIANT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
DEFAULT_SNAPSHOT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "benchmark_snapshots"


class ExecutionDeterminismError(RuntimeError):
    """Malformed protocol or fixture that cannot be audited safely."""


class FixtureNotEligible(ExecutionDeterminismError):
    """The fixture lacks enough information for the requested invariants."""


class QueryFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: str = Field(min_length=1)


class ExecutionRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    process_id: int
    invocation_index: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0.0)


class ExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    payload: dict[str, Any]


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["succeeded", "cancelled", "failed"]
    result: dict[str, Any] | None
    events: list[ExecutionEvent]
    error_type: str | None = None
    runtime: ExecutionRuntime


class CanonicalizationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(self.path.split("."))


class GateCheckpoint(BaseModel):
    """Query-result checkpoint mirroring canonical Benchmark resume semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_query_identities: list[str]
    completed_records: list[ExecutionRecord]
    completed_identities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionBackend(Protocol):
    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        """Execute one query through the audited pipeline."""


class SearchServiceFixtureBackend:
    """Real SearchService backed only by the repository's local fixtures."""

    def __init__(
        self,
        retrieval_outputs: Mapping[str, Any],
        *,
        service_max_workers: int,
        run_config: Mapping[str, Any],
        result_lineage_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._service = SearchService(
            retriever=build_fixture_retriever(retrieval_outputs),
            max_workers=service_max_workers,
        )
        self._run_config = dict(run_config)
        self._invocation_index = 0
        self._result_lineage_sink = result_lineage_sink

    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        self._invocation_index += 1
        events: list[ExecutionEvent] = []
        started = time.perf_counter()
        result: dict[str, Any] | None = None
        status: Literal["succeeded", "cancelled", "failed"] = "succeeded"
        error_type: str | None = None
        try:
            lineage_arguments = (
                {
                    "result_lineage_callback": lambda document: (
                        self._result_lineage_sink(query.identity, document)
                    )
                }
                if self._result_lineage_sink is not None
                else {}
            )
            output: SearchServiceOutput = self._service.run_search(
                query.query,
                event_callback=lambda name, payload: events.append(
                    ExecutionEvent(event=name, payload=copy.deepcopy(payload))
                ),
                should_cancel=should_cancel,
                **lineage_arguments,
                **self._run_config,
            )
            result = output.model_dump(mode="json")
        except SearchCancelled:
            status = "cancelled"
            error_type = "SearchCancelled"
        except Exception as exc:  # noqa: BLE001 - terminal state is compared
            status = "failed"
            error_type = type(exc).__name__
        return ExecutionRecord(
            query_identity=query.identity,
            status=status,
            result=result,
            events=events,
            error_type=error_type,
            runtime=ExecutionRuntime(
                run_id=f"{execution_label}-{self._invocation_index}",
                process_id=os.getpid(),
                invocation_index=self._invocation_index,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
            ),
        )


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionDeterminismError("protocol_unreadable") from exc
    if not isinstance(protocol, dict):
        raise ExecutionDeterminismError("protocol_root_must_be_object")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("contract") != CONTRACT_VERSION
    ):
        raise ExecutionDeterminismError("protocol_version_incompatible")
    if protocol.get("score_scope") != "determinism_only_not_quality_or_official_score":
        raise ExecutionDeterminismError("protocol_score_scope_invalid")
    fixture = protocol.get("fixture")
    if not isinstance(fixture, dict):
        raise ExecutionDeterminismError("fixture_contract_missing")
    if fixture.get("kind") != "local_retrieval_output_replay_fixture":
        raise ExecutionDeterminismError("fixture_kind_invalid")
    fixture_path = _repo_path(repository_root, str(fixture.get("retrieval_outputs_path") or ""))
    if not fixture_path.is_file():
        raise FixtureNotEligible("fixture_missing")
    if fixture_path.stat().st_size != fixture.get("size_bytes"):
        raise FixtureNotEligible("fixture_size_drift")
    if sha256_file(fixture_path) != fixture.get("sha256"):
        raise FixtureNotEligible("fixture_hash_drift")
    selection = fixture.get("query_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("gold_blind") is not True
        or selection.get("policy")
        != "first_n_unique_retrieval_queries_in_file_order"
    ):
        raise ExecutionDeterminismError("fixture_selection_policy_invalid")
    execution = protocol.get("execution")
    if not isinstance(execution, dict):
        raise ExecutionDeterminismError("execution_contract_missing")
    arguments = execution.get("search_service_arguments")
    if not isinstance(arguments, dict):
        raise ExecutionDeterminismError("search_service_arguments_missing")
    for field in (
        "enable_llm_judgement",
        "enable_llm_query_understanding",
        "enable_query_evolution",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_synthesis",
    ):
        if arguments.get(field) is not False:
            raise ExecutionDeterminismError(f"offline_feature_must_be_disabled:{field}")
    rules = protocol.get("canonicalization", {}).get("excluded_fields")
    if not isinstance(rules, list) or not rules:
        raise ExecutionDeterminismError("canonicalization_rules_missing")
    parsed_rules = [CanonicalizationRule.model_validate(item) for item in rules]
    paths = [item.path for item in parsed_rules]
    if len(paths) != len(set(paths)):
        raise ExecutionDeterminismError("duplicate_canonicalization_rule")
    return protocol


def load_query_fixtures(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> tuple[list[QueryFixture], dict[str, Any]]:
    fixture = protocol["fixture"]
    path = _repo_path(repository_root, str(fixture["retrieval_outputs_path"]))
    outputs = load_retrieval_outputs(path)
    count = int(fixture["query_selection"]["count"])
    if count < 2 or len(outputs) < count:
        raise FixtureNotEligible("insufficient_fixture_queries")
    selected = list(outputs)[:count]
    queries = [
        QueryFixture(
            identity=stable_hash({"query": normalize_snapshot_query(query)}),
            query=normalize_snapshot_query(query),
        )
        for query in selected
    ]
    if len({item.identity for item in queries}) != len(queries):
        raise FixtureNotEligible("duplicate_fixture_query_identity")
    expected_order = str(fixture["query_selection"]["order_sha256"])
    observed_order = stable_hash([item.identity for item in queries])
    if observed_order != expected_order:
        raise FixtureNotEligible("fixture_query_order_drift")
    return queries, outputs


def canonicalize_execution_record(
    record: ExecutionRecord,
    rules: Sequence[CanonicalizationRule],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replace only explicitly matched transient fields; preserve all ordering."""

    payload = record.model_dump(mode="json")
    return canonicalize_explicit_fields(payload, rules)


def canonicalize_explicit_fields(
    payload: Any,
    rules: Sequence[CanonicalizationRule],
) -> tuple[Any, dict[str, int]]:
    """Canonicalize only registered field paths without sorting list values."""

    counts = {rule.path: 0 for rule in rules}

    def visit(value: Any, path: tuple[str, ...]) -> Any:
        matching = [rule for rule in rules if _path_matches(path, rule.segments)]
        if matching:
            rule = matching[0]
            counts[rule.path] += 1
            return {"excluded_transient_field": rule.path}
        if isinstance(value, dict):
            return {key: visit(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, (*path, str(index))) for index, item in enumerate(value)]
        return value

    return visit(copy.deepcopy(payload), ()), counts


def build_checkpoint(
    expected_queries: Sequence[QueryFixture],
    records: Sequence[ExecutionRecord],
    *,
    config_sha256: str,
) -> GateCheckpoint:
    expected = [item.identity for item in expected_queries]
    completed = [item.query_identity for item in records]
    _validate_completed_identities(expected, completed, allow_partial=True)
    return GateCheckpoint(
        config_sha256=config_sha256,
        expected_query_identities=expected,
        completed_records=[item.model_copy(deep=True) for item in records],
        completed_identities_sha256=stable_hash(completed),
    )


def merge_checkpoint_resume(
    checkpoint: GateCheckpoint,
    resumed_records: Sequence[ExecutionRecord],
    *,
    config_sha256: str,
) -> list[ExecutionRecord]:
    if checkpoint.config_sha256 != config_sha256:
        raise ExecutionDeterminismError("resume_configuration_drift")
    checkpoint_ids = [item.query_identity for item in checkpoint.completed_records]
    if stable_hash(checkpoint_ids) != checkpoint.completed_identities_sha256:
        raise ExecutionDeterminismError("checkpoint_identity_hash_drift")
    resumed_ids = [item.query_identity for item in resumed_records]
    combined_ids = [*checkpoint_ids, *resumed_ids]
    _validate_completed_identities(
        checkpoint.expected_query_identities, combined_ids, allow_partial=False
    )
    indexed = {
        item.query_identity: item.model_copy(deep=True)
        for item in [*checkpoint.completed_records, *resumed_records]
    }
    return [indexed[identity] for identity in checkpoint.expected_query_identities]


def run_execution_determinism(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    backend_factory: Callable[[], ExecutionBackend] | None = None,
    fault: str | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    queries, retrieval_outputs = load_query_fixtures(
        protocol, repository_root=repository_root
    )
    rules = [
        CanonicalizationRule.model_validate(item)
        for item in protocol["canonicalization"]["excluded_fields"]
    ]
    execution = protocol["execution"]
    run_config = dict(execution["search_service_arguments"])
    if isinstance(run_config.get("budget"), Mapping):
        run_config["budget"] = SearchBudget.model_validate(run_config["budget"])
    service_workers = int(execution["service_max_workers"])
    concurrent_workers = int(execution["controlled_concurrent_workers"])
    split_count = int(execution["checkpoint_split_count"])
    if concurrent_workers < 2 or not 0 < split_count < len(queries):
        raise FixtureNotEligible("execution_fixture_bounds_invalid")
    factory = backend_factory or (
        lambda: SearchServiceFixtureBackend(
            retrieval_outputs,
            service_max_workers=service_workers,
            run_config=run_config,
        )
    )
    config_sha256 = stable_hash(
        {
            "search_service_arguments": execution["search_service_arguments"],
            "service_max_workers": service_workers,
        }
    )
    snapshot_before = tree_signature(snapshot_root)
    attempts = {"network": 0}
    with forbid_network(attempts):
        baseline = _execute_serial(factory(), queries, "baseline")

        repeated_backend = factory()
        repeat_first = _execute_serial(repeated_backend, queries, "repeat_first")
        repeat_second = _execute_serial(repeated_backend, queries, "repeat_second")

        singles = [
            factory().execute(
                query,
                execution_label="single",
                should_cancel=lambda: False,
            )
            for query in queries
        ]
        reordered = _execute_serial(factory(), list(reversed(queries)), "reordered")
        concurrent = _execute_concurrent(
            factory(), queries, workers=concurrent_workers, label="concurrent"
        )

        checkpoint_backend = factory()
        partial = _execute_serial(
            checkpoint_backend, queries[:split_count], "checkpoint"
        )
        checkpoint = build_checkpoint(
            queries, partial, config_sha256=config_sha256
        )
        resumed = _execute_serial(
            factory(), queries[split_count:], "resume"
        )
        checkpoint_resume = merge_checkpoint_resume(
            checkpoint, resumed, config_sha256=config_sha256
        )

        cancellation_backend = factory()
        cancelled = cancellation_backend.execute(
            queries[0],
            execution_label="cancelled",
            should_cancel=lambda: True,
        )
        after_cancel = cancellation_backend.execute(
            queries[1],
            execution_label="after_cancel",
            should_cancel=lambda: False,
        )

    snapshot_after = tree_signature(snapshot_root)
    scenarios = {
        "repeat_same_configuration": (repeat_first, repeat_second),
        "single_vs_batch": (baseline, singles),
        "batch_reorder": (baseline, reordered),
        "serial_vs_controlled_concurrent": (baseline, concurrent),
        "checkpoint_resume": (baseline, checkpoint_resume),
        "cancellation_isolation": ([baseline[1]], [after_cancel]),
    }
    if fault == "semantic_result_change":
        target = scenarios["serial_vs_controlled_concurrent"][1][0]
        if target.result is None:
            raise FixtureNotEligible("fault_target_has_no_result")
        target.result["deduplicated_count"] = int(
            target.result.get("deduplicated_count") or 0
        ) + 1
    elif fault is not None:
        raise ExecutionDeterminismError("unsupported_fault")

    invariant_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for invariant, (left, right) in scenarios.items():
        row, row_violations = _compare_record_sets(
            invariant, left, right, rules=rules
        )
        invariant_rows.append(row)
        violations.extend(row_violations)
    if cancelled.status != "cancelled":
        violations.append(
            {
                "invariant": "cancellation_isolation",
                "query_identity": queries[0].identity,
                "first_difference_path": "$.cancelled.status",
                "left_sha256": stable_hash("cancelled"),
                "right_sha256": stable_hash(cancelled.status),
            }
        )
    if attempts["network"]:
        violations.append(
            {
                "invariant": "offline_execution",
                "query_identity": None,
                "first_difference_path": "$.execution.network_request_count",
                "left_sha256": stable_hash(0),
                "right_sha256": stable_hash(attempts["network"]),
            }
        )
    snapshot_write_count = int(snapshot_before != snapshot_after)
    if snapshot_write_count:
        violations.append(
            {
                "invariant": "snapshot_read_only",
                "query_identity": None,
                "first_difference_path": "$.execution.snapshot_tree_sha256",
                "left_sha256": stable_hash(snapshot_before),
                "right_sha256": stable_hash(snapshot_after),
            }
        )
    violations = sorted(
        violations,
        key=lambda item: (
            str(item["invariant"]),
            str(item.get("query_identity") or ""),
            str(item["first_difference_path"]),
        ),
    )
    status = "passed" if not violations else "invariant_violation"
    exclusion_counts = {item.path: 0 for item in rules}
    for record in baseline:
        _, counts = canonicalize_execution_record(record, rules)
        for path, count in counts.items():
            exclusion_counts[path] += count
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if not violations else EXIT_INVARIANT_VIOLATION,
        "score_scope": "determinism_only_not_quality_or_official_score",
        "fixture": {
            "kind": "local_retrieval_output_replay_fixture",
            "query_count": len(queries),
            "query_identity_order_sha256": stable_hash(
                [item.identity for item in queries]
            ),
            "input_sha256": protocol["fixture"]["sha256"],
            "selection_policy": protocol["fixture"]["query_selection"]["policy"],
            "gold_accessed": False,
        },
        "canonicalization": {
            "policy": "explicit_field_paths_only_unknown_fields_preserved",
            "list_order_policy": "preserved",
            "dictionary_output_policy": "sorted_keys_only_for_serialization",
            "excluded_fields": [item.model_dump(mode="json") for item in rules],
            "baseline_excluded_field_match_counts": exclusion_counts,
        },
        "comparison_scope": [
            "retrieval_outputs",
            "ranked_and_all_ranked_papers_in_order",
            "deduplicated_identity_bearing_papers",
            "stage_snapshots_and_terminal_statuses",
            "semantic_event_names_payloads_and_order",
        ],
        "invariants": sorted(invariant_rows, key=lambda item: item["invariant"]),
        "violation_count": len(violations),
        "violations": violations,
        "cancellation_probe": {
            "cancelled_status": cancelled.status,
            "subsequent_status": after_cancel.status,
        },
        "frozen_baselines": _legacy_eligibility(protocol, repository_root),
        "execution": {
            "network_request_count": attempts["network"],
            "llm_request_count": 0,
            "snapshot_write_count": snapshot_write_count,
            "quality_metric_count": 0,
            "fault_injection": fault,
        },
    }


def replay_canonical_fixture(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    fault: Literal["semantic_result_change"] | None = None,
    collect_result_lineage: bool = False,
    install_network_guard: bool = True,
) -> dict[str, Any]:
    """Replay one ordered fixture batch through SearchService and canonicalize it.

    This is the reusable production Replay seam used by portable-capsule checks.
    It deliberately returns no quality metric and only excludes the field paths
    registered by ``execution_determinism_v1``.
    """

    queries, retrieval_outputs = load_query_fixtures(
        protocol, repository_root=repository_root
    )
    rules = [
        CanonicalizationRule.model_validate(item)
        for item in protocol["canonicalization"]["excluded_fields"]
    ]
    execution = protocol["execution"]
    run_config = dict(execution["search_service_arguments"])
    if isinstance(run_config.get("budget"), Mapping):
        run_config["budget"] = SearchBudget.model_validate(run_config["budget"])
    lineage_documents: list[tuple[str, dict[str, Any]]] = []
    backend = SearchServiceFixtureBackend(
        retrieval_outputs,
        service_max_workers=int(execution["service_max_workers"]),
        run_config=run_config,
        result_lineage_sink=(
            lambda identity, document: lineage_documents.append((identity, document))
        )
        if collect_result_lineage
        else None,
    )
    snapshot_before = tree_signature(snapshot_root)
    attempts = {"network": 0}
    network_guard = forbid_network(attempts) if install_network_guard else nullcontext()
    with network_guard:
        raw_records = _execute_serial(backend, queries, "capsule_replay")
    snapshot_after = tree_signature(snapshot_root)
    records = []
    for raw in raw_records:
        canonical, _counts = canonicalize_execution_record(raw, rules)
        records.append(
            {
                "case_id": raw.query_identity,
                "status": raw.status,
                "canonical_record": canonical,
                "canonical_sha256": stable_hash(canonical),
            }
        )
    if fault == "semantic_result_change":
        if not records or not isinstance(records[0]["canonical_record"], dict):
            raise FixtureNotEligible("fault_target_has_no_result")
        canonical_record = records[0]["canonical_record"]
        result = canonical_record.get("result")
        if not isinstance(result, dict):
            raise FixtureNotEligible("fault_target_has_no_result")
        result["deduplicated_count"] = int(result.get("deduplicated_count") or 0) + 1
        records[0]["canonical_sha256"] = stable_hash(canonical_record)
    elif fault is not None:
        raise ExecutionDeterminismError("unsupported_fault")
    snapshot_write_count = int(snapshot_before != snapshot_after)
    if attempts["network"] or snapshot_write_count:
        raise ExecutionDeterminismError("offline_replay_side_effect_detected")
    payload = {
        "schema_version": "1",
        "contract": "canonical_search_service_replay_v1",
        "score_scope": "replay_semantics_only_not_quality_or_official_score",
        "query_count": len(records),
        "query_identity_order_sha256": stable_hash(
            [item["case_id"] for item in records]
        ),
        "records": records,
        "records_sha256": stable_hash(records),
        "canonicalization": {
            "policy": "explicit_field_paths_only_unknown_fields_preserved",
            "excluded_fields": [item.model_dump(mode="json") for item in rules],
        },
        "execution": {
            "network_request_count": attempts["network"],
            "llm_request_count": 0,
            "snapshot_write_count": snapshot_write_count,
            "quality_metric_count": 0,
        },
    }
    if collect_result_lineage:
        lineage_summaries = [
            {
                "query_identity": identity,
                "lineage_sha256": stable_hash(document),
                "source_record_count": len(document.get("source_records") or []),
                "result_count": len(document.get("results") or []),
            }
            for identity, document in lineage_documents
        ]
        payload["result_lineage"] = {
            "contract": "result_lineage_v1",
            "query_count": len(lineage_summaries),
            "summaries": lineage_summaries,
            "summaries_sha256": stable_hash(lineage_summaries),
        }
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Report whether legacy frozen baselines can enter this gate."""

    profiles = _legacy_eligibility(protocol, repository_root)
    eligible_count = sum(item["status"] == "eligible" for item in profiles)
    status = "passed" if eligible_count == len(profiles) else "not_eligible"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if status == "passed" else EXIT_NOT_ELIGIBLE,
        "score_scope": "determinism_only_not_quality_or_official_score",
        "profiles": profiles,
        "eligible_count": eligible_count,
        "profile_count": len(profiles),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def _execute_serial(
    backend: ExecutionBackend,
    queries: Sequence[QueryFixture],
    label: str,
) -> list[ExecutionRecord]:
    return [
        backend.execute(
            query,
            execution_label=label,
            should_cancel=lambda: False,
        )
        for query in queries
    ]


def _execute_concurrent(
    backend: ExecutionBackend,
    queries: Sequence[QueryFixture],
    *,
    workers: int,
    label: str,
) -> list[ExecutionRecord]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                backend.execute,
                query,
                execution_label=label,
                should_cancel=lambda: False,
            )
            for query in queries
        ]
        return [future.result() for future in futures]


def _compare_record_sets(
    invariant: str,
    left_records: Sequence[ExecutionRecord],
    right_records: Sequence[ExecutionRecord],
    *,
    rules: Sequence[CanonicalizationRule],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = _canonical_index(left_records, rules)
    right = _canonical_index(right_records, rules)
    identities = sorted(set(left) | set(right))
    violations: list[dict[str, Any]] = []
    for identity in identities:
        left_value = left.get(identity)
        right_value = right.get(identity)
        differences = compare_profiles(left_value, right_value, max_diffs=1)
        if differences:
            violations.append(
                {
                    "invariant": invariant,
                    "query_identity": identity,
                    "first_difference_path": differences[0]["path"],
                    "left_sha256": stable_hash(left_value),
                    "right_sha256": stable_hash(right_value),
                }
            )
    left_hash = stable_hash({key: left[key] for key in sorted(left)})
    right_hash = stable_hash({key: right[key] for key in sorted(right)})
    return (
        {
            "invariant": invariant,
            "status": "passed" if not violations else "invariant_violation",
            "compared_query_count": len(identities),
            "left_sha256": left_hash,
            "right_sha256": right_hash,
        },
        violations,
    )


def _canonical_index(
    records: Sequence[ExecutionRecord], rules: Sequence[CanonicalizationRule]
) -> dict[str, dict[str, Any]]:
    identities = [item.query_identity for item in records]
    if len(identities) != len(set(identities)):
        raise ExecutionDeterminismError("duplicate_execution_query_identity")
    return {
        item.query_identity: canonicalize_execution_record(item, rules)[0]
        for item in records
    }


def _validate_completed_identities(
    expected: Sequence[str], completed: Sequence[str], *, allow_partial: bool
) -> None:
    if len(completed) != len(set(completed)):
        raise ExecutionDeterminismError("resume_duplicate_query_identity")
    unknown = sorted(set(completed) - set(expected))
    if unknown:
        raise ExecutionDeterminismError("resume_unknown_query_identity")
    if list(completed) != list(expected[: len(completed)]):
        raise ExecutionDeterminismError("resume_query_order_or_omission")
    if not allow_partial and len(completed) != len(expected):
        raise ExecutionDeterminismError("resume_query_omission")


def _legacy_eligibility(
    protocol: Mapping[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    spec = protocol.get("frozen_baseline_eligibility") or {}
    path = _repo_path(repository_root, str(spec.get("legacy_audit_path") or ""))
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        return [
            {
                "profile_id": "frozen_baselines",
                "status": "not_eligible",
                "reason": "legacy_audit_missing_or_hash_drift",
            }
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in sorted(payload.get("profiles", []), key=lambda row: row["profile_id"]):
        rows.append(
            {
                "profile_id": item["profile_id"],
                "status": "not_eligible",
                "reason": item.get("status") or "legacy_metadata_incomplete",
                "expected_query_count": item.get("expected_query_count"),
                "observed_record_count": item.get("observed_record_count"),
                "missing_metadata_fields": list(
                    item.get("missing_run_manifest_v1_fields") or []
                ),
            }
        )
    return rows


def _path_matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return len(path) == len(pattern) and all(
        expected == "*" or expected == actual
        for actual, expected in zip(path, pattern)
    )


def _repo_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ExecutionDeterminismError("path_must_be_repository_relative")
    root = repository_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ExecutionDeterminismError("path_resolves_outside_repository") from exc
    return resolved


def tree_signature(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@contextmanager
def forbid_network(attempts: dict[str, int]):
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise ExecutionDeterminismError("network_access_forbidden")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield
