"""Authoritative, deterministic resource ledgers for offline run auditing.

``resource_ledger_v1`` is an optional observation of the production connector
events and :class:`SearchBudgetRuntime` counters.  It deliberately does not
estimate missing provider usage or prices.  The gate only trusts ledgers that
are committed with a run generation; compatibility reports and logs are never
accepted as accounting inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import stable_json_bytes
from scholar_agent.evaluation.snapshot_resume import stable_hash


LEDGER_CONTRACT = "resource_ledger_v1"
GATE_CONTRACT = "resource_accounting_integrity_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "resource_accounting_integrity_gate"
EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4

BUDGET_DIMENSIONS = (
    "search_rounds",
    "candidate_papers",
    "llm_calls",
    "total_tokens",
)

QuantityState = Literal["known", "not_available", "not_applicable"]
OpaqueIdentity = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TerminalStatus = Literal[
    "success",
    "partial",
    "failed",
    "timeout",
    "rate_limited",
    "cancelled",
    "skipped",
    "budget_rejected",
]


class ResourceAccountingError(RuntimeError):
    """The input cannot be audited without guessing resource consumption."""


class ResourceLedgerNotEligible(ResourceAccountingError):
    """The run has no authoritative committed resource ledger."""


class Quantity(BaseModel):
    """A measured quantity whose absence is never represented as numeric zero."""

    model_config = ConfigDict(extra="forbid")

    state: QuantityState
    value: int | float | None = None
    unit: str
    reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "Quantity":
        if self.state == "known":
            if self.value is None or self.value < 0:
                raise ValueError("known quantity requires a non-negative value")
            if self.reason is not None:
                raise ValueError("known quantity cannot carry an absence reason")
        elif self.value is not None or not self.reason:
            raise ValueError("unknown quantity requires reason and no numeric value")
        return self


def known(value: int | float, unit: str) -> Quantity:
    return Quantity(state="known", value=value, unit=unit)


def unavailable(unit: str, reason: str = "not_exposed_by_authoritative_signal") -> Quantity:
    return Quantity(state="not_available", unit=unit, reason=reason)


def not_applicable(unit: str, reason: str = "operation_not_applicable") -> Quantity:
    return Quantity(state="not_applicable", unit=unit, reason=reason)


class BudgetVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_rounds: int = Field(default=0, ge=0)
    candidate_papers: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def plus(self, other: "BudgetVector") -> "BudgetVector":
        return BudgetVector(
            **{
                field: int(getattr(self, field)) + int(getattr(other, field))
                for field in BUDGET_DIMENSIONS
            }
        )

    def minus(self, other: "BudgetVector") -> "BudgetVector":
        values = {
            field: int(getattr(self, field)) - int(getattr(other, field))
            for field in BUDGET_DIMENSIONS
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("budget vector subtraction would become negative")
        return BudgetVector(**values)


class ResourceOperation(BaseModel):
    """One adapter, LLM, budget, cache, cancellation, or terminal operation."""

    model_config = ConfigDict(extra="forbid")

    operation_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_identity: OpaqueIdentity
    query_identity: OpaqueIdentity
    source_identity: OpaqueIdentity | None = None
    attempt_identity: OpaqueIdentity
    operation_type: Literal[
        "budget_reservation",
        "adapter_call",
        "llm_call",
        "budget_settlement",
        "budget_release",
        "cancellation",
    ]
    request_sequence: int = Field(ge=0)
    parent_operation_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    checkpoint_generation: int = Field(ge=0)
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_event_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    budget_reserved: BudgetVector = Field(default_factory=BudgetVector)
    budget_consumed: BudgetVector = Field(default_factory=BudgetVector)
    budget_released: BudgetVector = Field(default_factory=BudgetVector)
    budget_rejected: BudgetVector = Field(default_factory=BudgetVector)
    api_request_count: Quantity = Field(
        default_factory=lambda: not_applicable("requests")
    )
    pagination_count: Quantity = Field(
        default_factory=lambda: not_applicable("pages")
    )
    retry_count: int = Field(default=0, ge=0)
    returned_record_count: int = Field(default=0, ge=0)
    llm_call_count: int = Field(default=0, ge=0, le=1)
    prompt_tokens: Quantity = Field(
        default_factory=lambda: not_applicable("tokens")
    )
    completion_tokens: Quantity = Field(
        default_factory=lambda: not_applicable("tokens")
    )
    total_tokens: Quantity = Field(
        default_factory=lambda: not_applicable("tokens")
    )
    provider_cost: Quantity = Field(
        default_factory=lambda: not_applicable("provider_currency")
    )
    cache_status: Literal["hit", "miss", "not_applicable"] = "not_applicable"
    cache_hit_count: int = Field(default=0, ge=0)
    terminal_status: TerminalStatus
    adapter_started: bool = False
    started_after_cancellation: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ResourceTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_request_count: Quantity
    pagination_count: Quantity
    retry_count: int = Field(ge=0)
    returned_record_count: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    prompt_tokens: Quantity
    completion_tokens: Quantity
    total_tokens: Quantity
    provider_cost: Quantity
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    cancellation_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


class BudgetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limits: BudgetVector
    reserved: BudgetVector
    enforced_consumed: BudgetVector
    released: BudgetVector
    rejected: BudgetVector
    remaining: BudgetVector
    actual_total_tokens: Quantity
    latency_limit_seconds: float = Field(ge=0)
    elapsed_seconds: Quantity


class QueryResourceLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["resource_ledger_v1"] = LEDGER_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    run_identity: OpaqueIdentity
    query_identity: OpaqueIdentity
    attempt_identity: OpaqueIdentity
    checkpoint_generation: int = Field(ge=0)
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_terminal_status: Literal[
        "succeeded", "failed", "cancelled", "not_started"
    ]
    operations: list[ResourceOperation]
    totals: ResourceTotals
    budget: BudgetSummary
    semantic_event_identities: list[str] = Field(default_factory=list)
    authoritative: bool = True


class ResourceLedgerV1(BaseModel):
    """Run-level authority assembled only from selected committed query ledgers."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["resource_ledger_v1"] = LEDGER_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    run_identity: OpaqueIdentity
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_query_identities: list[OpaqueIdentity]
    queries: list[QueryResourceLedger]
    totals: ResourceTotals
    budget: BudgetSummary
    selected_attempts: dict[OpaqueIdentity, OpaqueIdentity]
    superseded_attempts: list[OpaqueIdentity] = Field(default_factory=list)
    authority: Literal["committed_generation_only"] = "committed_generation_only"
    score_scope: Literal[
        "resource_accounting_only_not_quality_or_official_score"
    ] = "resource_accounting_only_not_quality_or_official_score"


class ResourceLedgerObserver:
    """Observe existing SearchService signals without changing their semantics."""

    def __init__(self, budget: Any) -> None:
        self._limits = BudgetVector(
            search_rounds=int(budget.max_search_rounds),
            candidate_papers=int(budget.max_candidate_papers),
            llm_calls=int(budget.max_llm_calls),
            total_tokens=int(budget.max_total_tokens),
        )
        self._latency_limit_seconds = float(budget.max_latency_seconds)
        self._events: list[tuple[str, dict[str, Any]]] = []
        self._llm_calls: list[dict[str, Any]] = []
        self._completed_rounds = 0
        self._candidate_count = 0
        self._elapsed_seconds: float | None = None
        self._stop_reasons: list[str] = []
        self._cancelled = False

    def observe_semantic_event(self, name: str, payload: Mapping[str, Any]) -> None:
        if name not in {"connector_started", "connector_completed"}:
            return
        allowed = {
            "query_index",
            "source",
            "connector",
            "adapted_query",
            "adaptation_strategy",
            "request_count",
            "retry_count",
            "error_count",
            "returned_count",
            "cache_hit",
            "cache_hit_count",
            "logical_call_executed",
            "run_dedupe_hit",
            "source_skipped_reason",
            "error_message",
        }
        observed = {
            key: copy.deepcopy(payload[key]) for key in allowed if key in payload
        }
        if name == "connector_started" and self._cancelled:
            observed["resource_started_after_cancellation"] = True
        self._events.append((name, observed))

    def observe_llm_call(self, payload: Mapping[str, Any]) -> None:
        observed = copy.deepcopy(dict(payload))
        if self._cancelled:
            observed["resource_started_after_cancellation"] = True
        self._llm_calls.append(observed)

    def observe_source_stats(self, values: Sequence[Mapping[str, Any]]) -> None:
        """Capture non-retrieval source modules that expose the same diagnostics."""

        for value in values:
            source = str(value.get("source") or "")
            if source not in {"refchain", "semantic_seed_expansion"}:
                continue
            payload = copy.deepcopy(dict(value))
            payload.setdefault("connector", source)
            payload.setdefault("adapted_query", source)
            self._events.append(("connector_started", dict(payload)))
            self._events.append(("connector_completed", payload))

    def observe_budget_event(self, name: str, payload: Mapping[str, Any]) -> None:
        if name == "search_round_consumed":
            self._completed_rounds = int(payload.get("completed_search_rounds") or 0)
        elif name == "candidate_budget_observed":
            self._candidate_count = int(payload.get("candidate_count") or 0)
        elif name == "budget_finalized":
            self._completed_rounds = int(payload.get("completed_search_rounds") or 0)
            self._candidate_count = int(payload.get("candidate_count") or 0)
            elapsed = payload.get("elapsed_seconds")
            self._elapsed_seconds = float(elapsed) if elapsed is not None else None
            self._stop_reasons = sorted(
                {str(item) for item in payload.get("stop_reasons") or []}
            )

    def observe_cancellation(self, stage: str) -> None:
        self._cancelled = True
        self._events.append(("resource_cancelled", {"stage": str(stage)}))

    def build_query_ledger(
        self,
        *,
        run_identity: str,
        query_identity: str,
        attempt_identity: str,
        checkpoint_generation: int,
        manifest_identity: str,
        terminal_status: Literal["succeeded", "failed", "cancelled", "not_started"],
    ) -> QueryResourceLedger:
        base = {
            "run_identity": run_identity,
            "query_identity": query_identity,
            "attempt_identity": attempt_identity,
            "checkpoint_generation": checkpoint_generation,
            "manifest_identity": manifest_identity,
        }
        operations: list[ResourceOperation] = []
        root_identity = _operation_identity(base, "budget_reservation", "root")
        operations.append(
            ResourceOperation(
                operation_identity=root_identity,
                operation_type="budget_reservation",
                request_sequence=0,
                budget_reserved=self._limits,
                terminal_status="success",
                **base,
            )
        )
        connector_operations, event_identities = self._connector_operations(
            base, root_identity
        )
        operations.extend(connector_operations)
        sequence = 1 + len(connector_operations)
        token_enforced = 0
        for index, observation in enumerate(self._llm_calls):
            usage_available = bool(observation.get("usage_available"))
            prompt = int(observation.get("prompt_tokens") or 0)
            completion = int(observation.get("completion_tokens") or 0)
            total = int(observation.get("total_tokens") or 0)
            token_enforced += total
            event_identity = stable_hash(_stable_observation(observation))
            event_identities.append(event_identity)
            operations.append(
                ResourceOperation(
                    operation_identity=_operation_identity(
                        base, "llm_call", str(index)
                    ),
                    operation_type="llm_call",
                    request_sequence=sequence,
                    parent_operation_identity=root_identity,
                    budget_consumed=BudgetVector(llm_calls=1, total_tokens=total),
                    llm_call_count=1,
                    prompt_tokens=(
                        known(prompt, "tokens")
                        if usage_available
                        else unavailable("tokens", "provider_usage_not_available")
                    ),
                    completion_tokens=(
                        known(completion, "tokens")
                        if usage_available
                        else unavailable("tokens", "provider_usage_not_available")
                    ),
                    total_tokens=(
                        known(total, "tokens")
                        if usage_available
                        else unavailable("tokens", "provider_usage_not_available")
                    ),
                    provider_cost=unavailable(
                        "provider_currency", "provider_cost_not_exposed"
                    ),
                    terminal_status=_llm_terminal(observation),
                    started_after_cancellation=bool(
                        observation.get("resource_started_after_cancellation")
                    ),
                    semantic_event_identity=event_identity,
                    details={"provider_usage_available": usage_available},
                    **base,
                )
            )
            sequence += 1
        settlement_consumed = BudgetVector(
            search_rounds=self._completed_rounds,
            candidate_papers=self._candidate_count,
        )
        operations.append(
            ResourceOperation(
                operation_identity=_operation_identity(base, "budget_settlement", "final"),
                operation_type="budget_settlement",
                request_sequence=sequence,
                parent_operation_identity=root_identity,
                budget_consumed=settlement_consumed,
                terminal_status="success" if terminal_status == "succeeded" else "partial",
                **base,
            )
        )
        sequence += 1
        consumed = settlement_consumed.plus(
            BudgetVector(llm_calls=len(self._llm_calls), total_tokens=token_enforced)
        )
        release_values = {
            field: max(0, int(getattr(self._limits, field)) - int(getattr(consumed, field)))
            for field in BUDGET_DIMENSIONS
        }
        rejected = _rejected_budget(self._stop_reasons)
        operations.append(
            ResourceOperation(
                operation_identity=_operation_identity(base, "budget_release", "final"),
                operation_type="budget_release",
                request_sequence=sequence,
                parent_operation_identity=root_identity,
                budget_released=BudgetVector(**release_values),
                budget_rejected=rejected,
                terminal_status="success",
                **base,
            )
        )
        if self._cancelled or terminal_status == "cancelled":
            sequence += 1
            operations.append(
                ResourceOperation(
                    operation_identity=_operation_identity(base, "cancellation", "final"),
                    operation_type="cancellation",
                    request_sequence=sequence,
                    parent_operation_identity=root_identity,
                    terminal_status="cancelled",
                    **base,
                )
            )
        operations = sorted(operations, key=lambda item: item.request_sequence)
        totals = _operation_totals(operations)
        budget_summary = _budget_summary(
            operations,
            limits=self._limits,
            latency_limit_seconds=self._latency_limit_seconds,
            elapsed_seconds=(
                known(self._elapsed_seconds, "seconds")
                if self._elapsed_seconds is not None
                else unavailable("seconds", "run_terminated_before_final_status")
            ),
        )
        return QueryResourceLedger(
            **base,
            query_terminal_status=terminal_status,
            operations=operations,
            totals=totals,
            budget=budget_summary,
            semantic_event_identities=sorted(set(event_identities)),
        )

    def _connector_operations(
        self, base: Mapping[str, Any], root_identity: str
    ) -> tuple[list[ResourceOperation], list[str]]:
        started: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        completed: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for name, payload in self._events:
            if name not in {"connector_started", "connector_completed"}:
                continue
            key = _connector_key(payload)
            (started if name == "connector_started" else completed)[key].append(payload)
        rows: list[tuple[str, int, dict[str, Any] | None, dict[str, Any] | None]] = []
        for key in sorted(set(started) | set(completed)):
            left = sorted(started[key], key=lambda item: stable_hash(_stable_observation(item)))
            right = sorted(completed[key], key=lambda item: stable_hash(_stable_observation(item)))
            for ordinal in range(max(len(left), len(right))):
                rows.append(
                    (
                        key,
                        ordinal,
                        left[ordinal] if ordinal < len(left) else None,
                        right[ordinal] if ordinal < len(right) else None,
                    )
                )
        operations: list[ResourceOperation] = []
        event_identities: list[str] = []
        for sequence, (key, ordinal, before, after) in enumerate(rows, start=1):
            payload = dict(after or before or {})
            source = str(payload.get("source") or payload.get("connector") or "unknown")
            event_identity = stable_hash(
                {
                    "started": _stable_observation(before or {}),
                    "completed": _stable_observation(after or {}),
                }
            )
            event_identities.append(event_identity)
            request_count = (
                known(int(payload.get("request_count") or 0), "requests")
                if after is not None
                else unavailable("requests", "adapter_started_without_terminal_diagnostics")
            )
            cache_hit = bool(payload.get("cache_hit"))
            cache_hit_count = max(
                int(payload.get("cache_hit_count") or 0),
                int(cache_hit),
            )
            call_skipped = bool(
                payload.get("source_skipped_reason")
                or payload.get("logical_call_executed") is False
            )
            operations.append(
                ResourceOperation(
                    operation_identity=_operation_identity(
                        base, "adapter_call", f"{key}:{ordinal}"
                    ),
                    operation_type="adapter_call",
                    request_sequence=sequence,
                    parent_operation_identity=root_identity,
                    source_identity=opaque_resource_identity("source", source),
                    api_request_count=request_count,
                    pagination_count=unavailable(
                        "pages", "connector_diagnostics_do_not_expose_page_count"
                    ),
                    retry_count=int(payload.get("retry_count") or 0),
                    returned_record_count=int(payload.get("returned_count") or 0),
                    cache_status=(
                        "hit" if cache_hit else "not_applicable" if call_skipped else "miss"
                    ),
                    cache_hit_count=cache_hit_count,
                    terminal_status=_connector_terminal(payload, completed=after is not None),
                    adapter_started=before is not None,
                    started_after_cancellation=bool(
                        (before or {}).get("resource_started_after_cancellation")
                    ),
                    semantic_event_identity=event_identity,
                    details={
                        "query_index": payload.get("query_index"),
                        "source_kind": source,
                        "logical_call_executed": bool(
                            payload.get("logical_call_executed", True)
                        ),
                        "run_dedupe_hit": bool(payload.get("run_dedupe_hit")),
                        "recorded_diagnostics_authoritative_for_this_run": False,
                    },
                    **base,
                )
            )
        return operations, event_identities


def load_gate_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceAccountingError("resource_accounting_protocol_unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != GATE_CONTRACT
        or payload.get("ledger_contract") != LEDGER_CONTRACT
        or payload.get("identity_policy")
        != "opaque_sha256_run_query_source_attempt"
        or payload.get("missing_usage_semantics")
        != "unknown_or_not_available_never_numeric_zero"
        or payload.get("sensitive_error_policy")
        != "classify_without_echoing_raw_error_or_request_context"
        or payload.get("score_scope")
        != "resource_accounting_only_not_quality_or_official_score"
    ):
        raise ResourceAccountingError("resource_accounting_protocol_incompatible")
    return payload


def authority_manifest_identity(
    run_identity: str,
    *,
    expected_query_identities: Sequence[str],
    configuration: Mapping[str, Any],
) -> str:
    """Stable identity for the generation-chain run contract (not a file hash)."""

    return stable_hash(
        {
            "manifest_kind": "benchmark_run_commit_v1",
            "run_identity": run_identity,
            "expected_query_identities": list(expected_query_identities),
            "configuration": dict(configuration),
        }
    )


def opaque_resource_identity(kind: str, value: str) -> str:
    """Return a non-reversible stable identity for accounting authorities."""

    return stable_hash({"resource_identity_kind": kind, "value": value})


def build_run_ledger(
    query_ledgers: Sequence[QueryResourceLedger | Mapping[str, Any]],
    *,
    run_identity: str,
    manifest_identity: str,
    expected_query_identities: Sequence[str],
    selected_attempts: Mapping[str, str] | None = None,
    superseded_attempts: Sequence[str] = (),
) -> ResourceLedgerV1:
    parsed = [
        item if isinstance(item, QueryResourceLedger) else QueryResourceLedger.model_validate(item)
        for item in query_ledgers
    ]
    selection = dict(selected_attempts or {item.query_identity: item.attempt_identity for item in parsed})
    selected = [
        item for item in parsed if selection.get(item.query_identity) == item.attempt_identity
    ]
    order = {identity: index for index, identity in enumerate(expected_query_identities)}
    selected.sort(key=lambda item: order.get(item.query_identity, len(order)))
    operations = [operation for item in selected for operation in item.operations]
    return ResourceLedgerV1(
        run_identity=run_identity,
        manifest_identity=manifest_identity,
        expected_query_identities=list(expected_query_identities),
        queries=selected,
        totals=_operation_totals(operations),
        budget=_aggregate_budget([item.budget for item in selected]),
        selected_attempts=selection,
        superseded_attempts=sorted(set(str(item) for item in superseded_attempts)),
    )


def validate_resource_ledger(
    ledger: ResourceLedgerV1 | Mapping[str, Any],
    *,
    authoritative_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts = {"network": 0, "llm": 0, "snapshot_write": 0}
    violations: list[dict[str, Any]] = []
    try:
        parsed = ledger if isinstance(ledger, ResourceLedgerV1) else ResourceLedgerV1.model_validate(ledger)
    except ValidationError as exc:
        return _report(
            "accounting_or_budget_violation",
            [_violation("ledger_schema", "$", "resource_ledger_v1", exc.errors()[0].get("type"))],
            attempts,
        )
    expected = parsed.expected_query_identities
    observed = [item.query_identity for item in parsed.queries]
    if observed != expected:
        violations.append(_violation("query_accounting_coverage", "$.queries", expected, observed))
    if len(observed) != len(set(observed)):
        violations.append(_violation("query_accounting_duplicate", "$.queries", "unique", observed))
    if set(parsed.selected_attempts) != set(expected):
        violations.append(
            _violation(
                "selected_attempt_coverage",
                "$.selected_attempts",
                sorted(expected),
                sorted(parsed.selected_attempts),
            )
        )
    all_operations: list[ResourceOperation] = []
    for query_index, query in enumerate(parsed.queries):
        path = f"$.queries[{query_index}]"
        if (
            query.run_identity != parsed.run_identity
            or query.manifest_identity != parsed.manifest_identity
            or parsed.selected_attempts.get(query.query_identity) != query.attempt_identity
        ):
            violations.append(
                _violation("query_authority_binding", path, "run/manifest/selected attempt", "mismatch")
            )
        identities: set[str] = set()
        cancellation_sequence: int | None = None
        for operation_index, operation in enumerate(query.operations):
            op_path = f"{path}.operations[{operation_index}]"
            if operation.operation_identity in identities:
                violations.append(_violation("duplicate_operation_identity", op_path, "unique", operation.operation_identity))
            identities.add(operation.operation_identity)
            if (
                operation.run_identity != query.run_identity
                or operation.query_identity != query.query_identity
                or operation.attempt_identity != query.attempt_identity
                or operation.manifest_identity != query.manifest_identity
                or operation.checkpoint_generation != query.checkpoint_generation
            ):
                violations.append(_violation("operation_authority_binding", op_path, "query authority", "mismatch"))
            if operation.operation_type == "adapter_call":
                requests = _known_int(operation.api_request_count)
                if requests is not None and operation.retry_count > max(0, requests - 1):
                    violations.append(_violation("retry_request_conservation", op_path, f"retry <= {max(0, requests - 1)}", operation.retry_count))
                if operation.cache_status == "hit" and requests not in {None, 0}:
                    violations.append(_violation("cache_hit_external_consumption", op_path, 0, requests))
                if requests and not operation.adapter_started:
                    violations.append(_violation("external_call_without_adapter_operation", op_path, True, False))
            if operation.operation_type == "llm_call" and operation.llm_call_count != 1:
                violations.append(_violation("llm_call_entry_cardinality", op_path, 1, operation.llm_call_count))
            if operation.operation_type == "cancellation":
                cancellation_sequence = operation.request_sequence
            elif cancellation_sequence is not None and operation.request_sequence > cancellation_sequence:
                if _operation_has_consumption(operation):
                    violations.append(_violation("consumption_after_cancellation", op_path, "zero", "non_zero"))
            if operation.started_after_cancellation and _operation_has_consumption(operation):
                violations.append(_violation("unauthorized_post_cancel_consumption", op_path, False, True))
        recomputed_totals = _operation_totals(query.operations)
        if recomputed_totals != query.totals:
            violations.append(_violation("query_totals_conservation", f"{path}.totals", recomputed_totals.model_dump(mode="json"), query.totals.model_dump(mode="json")))
        expected_budget = _budget_summary(
            query.operations,
            limits=query.budget.limits,
            latency_limit_seconds=query.budget.latency_limit_seconds,
            elapsed_seconds=query.budget.elapsed_seconds,
        )
        if expected_budget != query.budget:
            violations.append(_violation("query_budget_conservation", f"{path}.budget", expected_budget.model_dump(mode="json"), query.budget.model_dump(mode="json")))
        violations.extend(
            _budget_balance_violations(expected_budget, path + ".recomputed")
        )
        violations.extend(_budget_balance_violations(query.budget, path))
        all_operations.extend(query.operations)
    expected_totals = _operation_totals(all_operations)
    if expected_totals != parsed.totals:
        violations.append(_violation("run_totals_equal_query_sum", "$.totals", expected_totals.model_dump(mode="json"), parsed.totals.model_dump(mode="json")))
    expected_budget = _aggregate_budget([item.budget for item in parsed.queries])
    if expected_budget != parsed.budget:
        violations.append(_violation("run_budget_equal_query_sum", "$.budget", expected_budget.model_dump(mode="json"), parsed.budget.model_dump(mode="json")))
    if authoritative_records is not None:
        query_by_identity = {item.query_identity: item for item in parsed.queries}
        for record_index, record in enumerate(authoritative_records):
            embedded = record.get("resource_ledger")
            if not isinstance(embedded, Mapping):
                violations.append(
                    _violation(
                        "committed_record_ledger_missing",
                        f"$.records[{record_index}]",
                        "resource_ledger_v1",
                        None,
                    )
                )
                continue
            identity = str(embedded.get("query_identity") or "")
            query = query_by_identity.get(identity)
            if query is None:
                violations.append(
                    _violation(
                        "committed_record_ledger_not_selected",
                        f"$.records[{record_index}]",
                        "selected query ledger",
                        identity,
                    )
                )
                continue
            cost = record.get("cost_report")
            if not isinstance(cost, Mapping):
                continue
            requests = _known_int(query.totals.api_request_count)
            if requests is not None:
                expected_calls = requests + query.totals.llm_call_count
                if int(cost.get("api_call_count") or 0) != expected_calls:
                    violations.append(
                        _violation(
                            "adapter_llm_call_ledger_completeness",
                            f"$.records[{record_index}].cost_report.api_call_count",
                            int(cost.get("api_call_count") or 0),
                            expected_calls,
                        )
                    )
            for field, actual in (
                ("retry_count", query.totals.retry_count),
                ("cache_hit_count", query.totals.cache_hit_count),
                ("llm_call_count", query.totals.llm_call_count),
            ):
                expected_value = int(cost.get(field) or 0)
                if expected_value != actual:
                    violations.append(
                        _violation(
                            f"committed_{field}_ledger_completeness",
                            f"$.records[{record_index}].cost_report.{field}",
                            expected_value,
                            actual,
                        )
                    )
    return _report(
        "passed" if not violations else "accounting_or_budget_violation",
        _with_authority_context(violations, parsed),
        attempts,
    )


def audit_frozen_eligibility(legacy_audit_path: Path) -> dict[str, Any]:
    payload = json.loads(legacy_audit_path.read_text(encoding="utf-8"))
    rows = []
    for item in sorted(payload.get("profiles") or [], key=lambda value: str(value.get("profile_id"))):
        profile = str(item.get("profile_id") or "unknown")
        if "record160" not in profile and "full1000" not in profile:
            continue
        rows.append(
            {
                "profile_id": profile,
                "status": "not_eligible",
                "reason": "missing_authoritative_per_operation_ledger_and_complete_commit_generation",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_CONTRACT,
        "gate": GATE_NAME,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "profiles": rows,
        "observation": _zero_observation(),
        "score_scope": "resource_accounting_only_not_quality_or_official_score",
    }


def audit_evidence_registry(registry_path: Path) -> dict[str, Any]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    claims = []
    for item in sorted(payload.get("strategies") or [], key=lambda value: str(value.get("strategy_id"))):
        if item.get("efficiency_cost") is None:
            continue
        claims.append(
            {
                "strategy_id": str(item.get("strategy_id")),
                "status": "not_eligible",
                "reason": "historical_cost_claim_has_no_authoritative_resource_ledger",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_CONTRACT,
        "gate": GATE_NAME,
        "status": "not_eligible" if claims else "passed",
        "exit_code": EXIT_NOT_ELIGIBLE if claims else EXIT_PASSED,
        "cost_claims": claims,
        "registry_mutated": False,
        "observation": _zero_observation(),
        "score_scope": "resource_accounting_only_not_quality_or_official_score",
    }


def audit_shard_aggregate(
    aggregate_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Validate only ledgers referenced by the aggregate's selected attempts."""

    from scholar_agent.evaluation.run_provenance import (  # noqa: PLC0415
        resolve_repo_path,
    )
    from scholar_agent.evaluation.sharded_execution import (  # noqa: PLC0415
        ShardAggregateV1,
    )

    try:
        aggregate = ShardAggregateV1.model_validate_json(
            aggregate_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        return _report(
            "accounting_or_budget_violation",
            [_violation("shard_aggregate_unreadable", "$", "shard_aggregate_v1", type(exc).__name__)],
            _zero_observation(),
        )
    if not aggregate.resource_ledgers:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": GATE_CONTRACT,
            "gate": GATE_NAME,
            "status": "not_eligible",
            "exit_code": EXIT_NOT_ELIGIBLE,
            "reason": "shard_aggregate_missing_authoritative_ledgers",
            "observation": _zero_observation(),
            "score_scope": "resource_accounting_only_not_quality_or_official_score",
        }
    selected = {
        (item.shard_index, item.attempt_id) for item in aggregate.selected_shards
    }
    violations: list[dict[str, Any]] = []
    observed_queries: set[str] = set()
    for index, reference in enumerate(aggregate.resource_ledgers):
        ref_key = (reference.shard_index, reference.attempt_id)
        if ref_key not in selected:
            violations.append(
                _violation(
                    "superseded_or_unselected_shard_ledger",
                    f"$.resource_ledgers[{index}]",
                    sorted(selected),
                    list(ref_key),
                )
            )
            continue
        path = resolve_repo_path(repository_root, reference.path)
        try:
            if _sha256_file(path) != reference.sha256:
                raise ResourceAccountingError("ledger_hash_mismatch")
            ledger = ResourceLedgerV1.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError, ResourceAccountingError) as exc:
            violations.append(
                _violation(
                    "selected_shard_ledger_invalid",
                    f"$.resource_ledgers[{index}]",
                    reference.sha256,
                    type(exc).__name__,
                )
            )
            continue
        if ledger.manifest_identity != reference.manifest_identity:
            violations.append(
                _violation(
                    "selected_shard_ledger_manifest_mismatch",
                    f"$.resource_ledgers[{index}]",
                    reference.manifest_identity,
                    ledger.manifest_identity,
                )
            )
        nested = validate_resource_ledger(ledger)
        if nested["status"] != "passed":
            violations.append(
                _violation(
                    "selected_shard_ledger_accounting_invalid",
                    f"$.resource_ledgers[{index}]",
                    "passed",
                    nested["status"],
                )
            )
        for identity in ledger.expected_query_identities:
            if identity in observed_queries:
                violations.append(
                    _violation(
                        "shard_query_consumption_double_counted",
                        f"$.resource_ledgers[{index}]",
                        "unique query",
                        identity,
                    )
                )
            observed_queries.add(identity)
    if len(observed_queries) != len(aggregate.records):
        violations.append(
            _violation(
                "shard_ledger_query_coverage",
                "$.resource_ledgers",
                len(aggregate.records),
                len(observed_queries),
            )
        )
    return _report(
        "passed" if not violations else "accounting_or_budget_violation",
        violations,
        _zero_observation(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_fixture_report(
    *, controlled_fault: str | None = None, shard_resume: bool = False
) -> dict[str, Any]:
    ledger = _fixture_run_ledger(shard_resume=shard_resume)
    payload = ledger.model_dump(mode="json")
    if controlled_fault == "double_charge":
        payload["queries"][0]["operations"][-2]["budget_consumed"]["candidate_papers"] += 1
    elif controlled_fault == "missing_call":
        payload["queries"][0]["operations"] = [
            item for item in payload["queries"][0]["operations"] if item["operation_type"] != "adapter_call"
        ]
    elif controlled_fault == "fake_cache_consumption":
        query = payload["queries"][3]
        operation = next(item for item in query["operations"] if item["operation_type"] == "adapter_call")
        operation["api_request_count"] = {"state": "known", "value": 1, "unit": "requests", "reason": None}
    elif controlled_fault == "negative_remaining":
        payload["queries"][0]["budget"]["remaining"]["candidate_papers"] = -1
    elif controlled_fault == "over_budget":
        payload["queries"][0]["operations"][-2]["budget_consumed"]["search_rounds"] = 99
    elif controlled_fault == "post_cancel":
        query = payload["queries"][5]
        operation = next(item for item in query["operations"] if item["operation_type"] == "adapter_call")
        operation["started_after_cancellation"] = True
        operation["api_request_count"] = {
            "state": "known",
            "value": 1,
            "unit": "requests",
            "reason": None,
        }
    elif controlled_fault == "stale_attempt":
        payload["selected_attempts"][payload["expected_query_identities"][0]] = (
            opaque_resource_identity("attempt", "superseded-attempt")
        )
    elif controlled_fault is not None:
        raise ResourceAccountingError(f"unsupported_controlled_fault:{controlled_fault}")
    with _forbid_network():
        report = validate_resource_ledger(payload)
    report["fixture"] = {
        "query_count": len(ledger.queries),
        "operation_count": sum(len(item.operations) for item in ledger.queries),
        "retry_and_pagination_covered": True,
        "cancel_covered": True,
        "resume_and_shard_selection_covered": shard_resume,
        "unknown_token_and_cost_preserved": True,
    }
    return report


def _fixture_run_ledger(*, shard_resume: bool) -> ResourceLedgerV1:
    run_id = opaque_resource_identity("run", "offline-resource-fixture")
    query_ids = [
        opaque_resource_identity("query", f"fixture-query-{index:02d}")
        for index in range(7)
    ]
    manifest = stable_hash({"fixture": LEDGER_CONTRACT, "run": run_id})
    attempts = {
        identity: opaque_resource_identity("attempt", f"{identity}:final")
        for identity in query_ids
    }
    queries: list[QueryResourceLedger] = []
    scenarios = [
        (1, 0, False, "failed"),
        (3, 1, False, "partial"),
        (2, 1, False, "rate_limited"),
        (0, 0, True, "success"),
        (1, 0, False, "timeout"),
        (0, 0, False, "cancelled"),
        (1, 0, False, "success"),
    ]
    for index, (requests, retries, cache_hit, terminal) in enumerate(scenarios):
        observer = ResourceLedgerObserver(_FixtureBudget())
        observer.observe_semantic_event(
            "connector_started",
            {"query_index": 0, "source": "fixture", "adapted_query": "opaque"},
        )
        observer.observe_semantic_event(
            "connector_completed",
            {
                "query_index": 0,
                "source": "fixture",
                "adapted_query": "opaque",
                "request_count": requests,
                "retry_count": retries,
                "returned_count": index + 1 if terminal in {"success", "partial"} else 0,
                "cache_hit": cache_hit,
                "error_message": None if terminal == "success" else terminal,
            },
        )
        observer.observe_budget_event("search_round_consumed", {"completed_search_rounds": 1})
        observer.observe_budget_event("candidate_budget_observed", {"candidate_count": min(index + 1, 4)})
        if index == 6:
            observer.observe_llm_call(
                {
                    "usage_available": False,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "terminal_status": "success",
                }
            )
        if terminal == "cancelled":
            observer.observe_cancellation("fixture")
        observer.observe_budget_event(
            "budget_finalized",
            {
                "completed_search_rounds": 1,
                "candidate_count": min(index + 1, 4),
                "elapsed_seconds": 1.0,
                "stop_reasons": [],
            },
        )
        query_ledger = observer.build_query_ledger(
                run_identity=run_id,
                query_identity=query_ids[index],
                attempt_identity=attempts[query_ids[index]],
                checkpoint_generation=index + 1,
                manifest_identity=manifest,
                terminal_status=(
                    "cancelled"
                    if terminal == "cancelled"
                    else "failed"
                    if terminal not in {"success", "partial"}
                    else "succeeded"
                ),
            )
        if index == 1:
            operations = [item.model_copy(deep=True) for item in query_ledger.operations]
            adapter = next(
                item for item in operations if item.operation_type == "adapter_call"
            )
            adapter.pagination_count = known(2, "pages")
            query_ledger = query_ledger.model_copy(
                update={
                    "operations": operations,
                    "totals": _operation_totals(operations),
                }
            )
        queries.append(query_ledger)
    superseded = (
        [opaque_resource_identity("attempt", "fixture-superseded")]
        if shard_resume
        else []
    )
    return build_run_ledger(
        queries,
        run_identity=run_id,
        manifest_identity=manifest,
        expected_query_identities=query_ids,
        selected_attempts=attempts,
        superseded_attempts=superseded,
    )


class _FixtureBudget:
    max_search_rounds = 3
    max_candidate_papers = 20
    max_llm_calls = 3
    max_total_tokens = 1000
    max_latency_seconds = 30.0


def _operation_identity(base: Mapping[str, Any], kind: str, discriminator: str) -> str:
    return stable_hash(
        {
            "run": base["run_identity"],
            "query": base["query_identity"],
            "attempt": base["attempt_identity"],
            "generation": base["checkpoint_generation"],
            "kind": kind,
            "discriminator": discriminator,
        }
    )


def _connector_key(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "query_index": payload.get("query_index"),
            "source": payload.get("source") or payload.get("connector"),
            "adapted_query": payload.get("adapted_query"),
            "adaptation_strategy": payload.get("adaptation_strategy"),
        }
    )


def _stable_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        if key not in {"latency_seconds", "error_message"}
    }


def _connector_terminal(payload: Mapping[str, Any], *, completed: bool) -> TerminalStatus:
    if not completed:
        return "partial"
    if payload.get("source_skipped_reason") or payload.get("logical_call_executed") is False:
        return "skipped"
    message = str(payload.get("error_message") or "").casefold()
    if not message:
        return "success"
    if "timeout" in message or "deadline" in message:
        return "timeout"
    if "429" in message or "rate" in message:
        return "rate_limited"
    if "cancel" in message:
        return "cancelled"
    if int(payload.get("returned_count") or 0) > 0:
        return "partial"
    return "failed"


def _llm_terminal(payload: Mapping[str, Any]) -> TerminalStatus:
    value = str(payload.get("terminal_status") or "failed")
    return value if value in {"success", "failed", "timeout", "cancelled"} else "failed"  # type: ignore[return-value]


def _rejected_budget(stop_reasons: Sequence[str]) -> BudgetVector:
    values = {field: 0 for field in BUDGET_DIMENSIONS}
    mapping = {
        "max_search_rounds": "search_rounds",
        "max_candidate_papers": "candidate_papers",
        "max_llm_calls": "llm_calls",
        "max_total_tokens": "total_tokens",
    }
    for reason in stop_reasons:
        for marker, dimension in mapping.items():
            if marker in reason:
                values[dimension] = 1
    return BudgetVector(**values)


def _known_int(quantity: Quantity) -> int | None:
    if quantity.state != "known" or quantity.value is None:
        return None
    return int(quantity.value)


def _aggregate_quantities(values: Sequence[Quantity], unit: str) -> Quantity:
    applicable = [item for item in values if item.state != "not_applicable"]
    if not applicable:
        return not_applicable(unit)
    unavailable_values = [item for item in applicable if item.state == "not_available"]
    if unavailable_values:
        return unavailable(unit, "one_or_more_authoritative_values_unavailable")
    return known(sum(float(item.value or 0) for item in applicable), unit)


def _operation_totals(operations: Sequence[ResourceOperation]) -> ResourceTotals:
    llm = [item for item in operations if item.llm_call_count]
    adapters = [item for item in operations if item.operation_type == "adapter_call"]
    return ResourceTotals(
        api_request_count=_aggregate_quantities([item.api_request_count for item in adapters], "requests"),
        pagination_count=_aggregate_quantities([item.pagination_count for item in adapters], "pages"),
        retry_count=sum(item.retry_count for item in operations),
        returned_record_count=sum(item.returned_record_count for item in operations),
        llm_call_count=sum(item.llm_call_count for item in operations),
        prompt_tokens=_aggregate_quantities([item.prompt_tokens for item in llm], "tokens"),
        completion_tokens=_aggregate_quantities([item.completion_tokens for item in llm], "tokens"),
        total_tokens=_aggregate_quantities([item.total_tokens for item in llm], "tokens"),
        provider_cost=_aggregate_quantities([item.provider_cost for item in llm], "provider_currency"),
        cache_hit_count=sum(
            max(item.cache_hit_count, int(item.cache_status == "hit"))
            for item in operations
        ),
        cache_miss_count=sum(item.cache_status == "miss" for item in operations),
        cancellation_count=sum(item.terminal_status == "cancelled" for item in operations),
        timeout_count=sum(item.terminal_status == "timeout" for item in operations),
        partial_count=sum(item.terminal_status == "partial" for item in operations),
        error_count=sum(item.terminal_status in {"failed", "timeout", "rate_limited"} for item in operations),
    )


def _sum_vectors(values: Sequence[BudgetVector]) -> BudgetVector:
    result = BudgetVector()
    for value in values:
        result = result.plus(value)
    return result


def _budget_summary(
    operations: Sequence[ResourceOperation],
    *,
    limits: BudgetVector,
    latency_limit_seconds: float,
    elapsed_seconds: Quantity,
) -> BudgetSummary:
    reserved = _sum_vectors([item.budget_reserved for item in operations])
    consumed = _sum_vectors([item.budget_consumed for item in operations])
    released = _sum_vectors([item.budget_released for item in operations])
    rejected = _sum_vectors([item.budget_rejected for item in operations])
    actual_tokens = _aggregate_quantities(
        [item.total_tokens for item in operations if item.llm_call_count], "tokens"
    )
    return BudgetSummary(
        limits=limits,
        reserved=reserved,
        enforced_consumed=consumed,
        released=released,
        rejected=rejected,
        remaining=released,
        actual_total_tokens=actual_tokens,
        latency_limit_seconds=latency_limit_seconds,
        elapsed_seconds=elapsed_seconds,
    )


def _aggregate_budget(values: Sequence[BudgetSummary]) -> BudgetSummary:
    elapsed = _aggregate_quantities([item.elapsed_seconds for item in values], "seconds")
    actual = _aggregate_quantities([item.actual_total_tokens for item in values], "tokens")
    return BudgetSummary(
        limits=_sum_vectors([item.limits for item in values]),
        reserved=_sum_vectors([item.reserved for item in values]),
        enforced_consumed=_sum_vectors([item.enforced_consumed for item in values]),
        released=_sum_vectors([item.released for item in values]),
        rejected=_sum_vectors([item.rejected for item in values]),
        remaining=_sum_vectors([item.remaining for item in values]),
        actual_total_tokens=actual,
        latency_limit_seconds=sum(item.latency_limit_seconds for item in values),
        elapsed_seconds=elapsed,
    )


def _budget_balance_violations(summary: BudgetSummary, path: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for field in BUDGET_DIMENSIONS:
        limit = int(getattr(summary.limits, field))
        reserved = int(getattr(summary.reserved, field))
        consumed = int(getattr(summary.enforced_consumed, field))
        released = int(getattr(summary.released, field))
        remaining = int(getattr(summary.remaining, field))
        if reserved != limit:
            violations.append(_violation("budget_reservation_equals_limit", f"{path}.budget.{field}", limit, reserved))
        if consumed > limit:
            violations.append(_violation("budget_limit_not_exceeded", f"{path}.budget.{field}", f"<= {limit}", consumed))
        if reserved != consumed + released:
            violations.append(_violation("budget_reserve_consume_release_conservation", f"{path}.budget.{field}", reserved, consumed + released))
        if remaining != released:
            violations.append(_violation("budget_remaining_equals_release", f"{path}.budget.{field}", released, remaining))
    return violations


def _operation_has_consumption(operation: ResourceOperation) -> bool:
    requests = _known_int(operation.api_request_count)
    return bool(
        (requests or 0)
        or operation.llm_call_count
        or any(int(getattr(operation.budget_consumed, field)) for field in BUDGET_DIMENSIONS)
    )


def _violation(invariant: str, path: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "invariant": invariant,
        "ledger_path": path,
        "expected": expected,
        "actual": actual,
        "first_difference_path": path,
    }


def _with_authority_context(
    violations: Sequence[Mapping[str, Any]], ledger: ResourceLedgerV1
) -> list[dict[str, Any]]:
    contextual: list[dict[str, Any]] = []
    for item in violations:
        value = dict(item)
        value["run_identity"] = ledger.run_identity
        path = str(value.get("ledger_path") or "")
        if path.startswith("$.queries["):
            try:
                query_index = int(path.split("[", 1)[1].split("]", 1)[0])
                query = ledger.queries[query_index]
            except (IndexError, ValueError):
                query = None
            if query is not None:
                value["query_identity"] = query.query_identity
                value["attempt_identity"] = query.attempt_identity
                operation_marker = ".operations["
                if operation_marker in path:
                    try:
                        operation_index = int(
                            path.split(operation_marker, 1)[1].split("]", 1)[0]
                        )
                        operation = query.operations[operation_index]
                    except (IndexError, ValueError):
                        operation = None
                    if operation is not None and operation.source_identity is not None:
                        value["source_identity"] = operation.source_identity
        contextual.append(value)
    return contextual


def _report(status: str, violations: Sequence[Mapping[str, Any]], attempts: Mapping[str, int]) -> dict[str, Any]:
    ordered = sorted(
        (dict(item) for item in violations),
        key=lambda item: (str(item.get("invariant")), str(item.get("ledger_path"))),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_CONTRACT,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if not ordered else EXIT_VIOLATION,
        "violations": ordered,
        "violation_count": len(ordered),
        "observation": {
            "network_request_count": int(attempts.get("network", 0)),
            "llm_request_count": int(attempts.get("llm", 0)),
            "snapshot_write_count": int(attempts.get("snapshot_write", 0)),
            "quality_metric_count": 0,
        },
        "score_scope": "resource_accounting_only_not_quality_or_official_score",
    }


def _zero_observation() -> dict[str, int]:
    return {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
    }


@contextmanager
def _forbid_network() -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("offline_resource_gate_attempted_network")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield


def stable_report_bytes(value: Mapping[str, Any]) -> bytes:
    return stable_json_bytes(dict(value), indent=None)
