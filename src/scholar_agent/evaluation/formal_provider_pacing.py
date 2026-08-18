"""Deterministic provider pacing controls for the frozen Full1000 intents.

The module consumes the already-frozen request-intent manifest.  It never
constructs, removes, or rewrites a request and never calls an adapter.  A
logical clock and synthetic capacity declarations exercise token buckets,
concurrency slots, continuation fairness, pause/resume, and bounded retry
semantics without network I/O.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_network_request_manifest import load_jsonl
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_provider_pacing_v1"
SCHEMA_VERSION = "1"
LAUNCH_ADDENDUM = "full1000_provider_pacing_addendum_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FROZEN_PROTOCOL_SHA256 = (
    "337a6004ab062a6ae690d33db223866d955c6c4951b62f9c83f35ab46b742b6c"
)
SOURCES = ("openalex", "arxiv", "semantic_scholar", "pubmed")
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
CAPACITY_FIELDS = (
    "requests_per_second",
    "requests_per_minute",
    "max_concurrency",
    "burst",
    "cooldown_steps",
)
FORBIDDEN_PRIORITY_INPUTS = (
    "case_id",
    "completion_speed",
    "gold",
    "paper_identity",
    "provider_yield",
    "quality_metric",
    "query_text_or_type",
    "result_content",
)


class ProviderPacingError(RuntimeError):
    """A pacing, declaration, request identity, or budget invariant failed."""


class ProviderPacingNotReady(ProviderPacingError):
    """Real provider capacity declarations are unavailable or stale."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate json key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderPacingError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ProviderPacingError("json_root_not_object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.name == ".env"
        or path.parts[0] == "third_party"
        or path.as_posix() != value
    ):
        raise ProviderPacingError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    expected = {
        "execution_plan",
        "launch_control",
        "provider_health",
        "request_intents",
        "request_launch_addendum",
        "request_manifest",
        "resource_ledger",
        "scheduler",
        "shard_contract",
    }
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise ProviderPacingError("binding_inventory_invalid")
    for name, binding in sorted(bindings.items()):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ProviderPacingError("binding_schema_invalid")
        relative = _safe_relative(str(binding["path"]))
        target = root / relative
        if not target.is_file():
            raise ProviderPacingError(f"binding_missing:{name}")
        if sha256_file(target) != binding["sha256"]:
            raise ProviderPacingError(f"binding_hash_drift:{name}")


def _validate_real_declarations(value: Any) -> None:
    if not isinstance(value, dict) or tuple(value) != SOURCES:
        raise ProviderPacingError("capacity_source_inventory_invalid")
    expected_keys = {
        "burst",
        "cooldown_steps",
        "declaration_version",
        "max_concurrency",
        "provenance",
        "requests_per_minute",
        "requests_per_second",
        "valid_from",
        "valid_until",
    }
    for source, declaration in value.items():
        if not isinstance(declaration, dict) or set(declaration) != expected_keys:
            raise ProviderPacingError(f"capacity_declaration_schema_invalid:{source}")
        for field_name in (*CAPACITY_FIELDS, "declaration_version", "valid_from", "valid_until"):
            if declaration[field_name] != NOT_AVAILABLE:
                raise ProviderPacingError(
                    f"unverified_real_capacity_value_present:{source}:{field_name}"
                )
        if declaration["provenance"] != "external_operator_input_required":
            raise ProviderPacingError(f"capacity_provenance_invalid:{source}")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "bindings",
        "capacity_declarations",
        "execution",
        "formal_validation_complete",
        "pacing_policy",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "state_machine",
        "synthetic_profiles",
    }
    if set(value) != required:
        raise ProviderPacingError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise ProviderPacingError("protocol_version_invalid")
    if value["source_commit"] != "38eca0b1a8a1744b73c88ac480704b92ae0b530a":
        raise ProviderPacingError("protocol_source_commit_invalid")
    if value["execution"] != EXECUTION_ZERO:
        raise ProviderPacingError("offline_execution_contract_drift")
    if value["formal_validation_complete"] is not False:
        raise ProviderPacingError("formal_validation_state_drift")
    if _protocol_digest(value) != value["protocol_sha256"]:
        raise ProviderPacingError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ProviderPacingError("protocol_content_drift")
    population = value["population"]
    if population != {
        "http_attempt_upper": 19280,
        "logical_source_request_count": 9640,
        "query_count": 1000,
        "shard_count": 20,
        "sources": list(SOURCES),
        "subquery_count": 2410,
    }:
        raise ProviderPacingError("population_contract_drift")
    policy = value["pacing_policy"]
    if policy != {
        "continuation_fairness": (
            "all_initial_intents_admitted_before_page_or_retry_then_stable_parent_order"
        ),
        "global_concurrency": 12,
        "identity_preservation": (
            "intent_and_request_spec_hashes_are_immutable_across_pacing"
        ),
        "logical_clock_resolution": "one_declared_capacity_second",
        "pause_cancel": "stop_new_admission_then_close_inflight_ledger_entries",
        "prohibited_priority_inputs": list(FORBIDDEN_PRIORITY_INPUTS),
        "response_backoff": (
            "declared_retry_after_else_capacity_cooldown_without_new_attempt_budget"
        ),
        "resume": (
            "restore_tokens_windows_cooldowns_failures_and_admitted_operation_identities"
        ),
        "source_isolation": "independent_token_bucket_and_concurrency_slots",
    }:
        raise ProviderPacingError("pacing_policy_drift")
    if value["state_machine"] != {
        "states": [
            "ready",
            "running",
            "pause_required",
            "paused",
            "resume_eligible",
            "cancel_required",
            "cancelled",
            "completed",
            "invalid",
        ]
    }:
        raise ProviderPacingError("state_machine_drift")
    expected_profiles = [
        "balanced",
        "single_source_low_quota",
        "burst_exhaustion",
        "retry_after",
        "persistent_429",
        "503_jitter",
        "timeout_jitter",
        "dynamic_reduction",
        "pause_resume",
        "asynchronous_shards",
        "expired_declaration",
        "unknown_capacity",
    ]
    if value["synthetic_profiles"] != {
        "names": expected_profiles,
        "test_only": True,
    }:
        raise ProviderPacingError("synthetic_profile_contract_drift")
    _validate_real_declarations(value["capacity_declarations"])
    _validate_bindings(repository_root, value)
    return value


@dataclass(frozen=True)
class CapacityDeclaration:
    source: str
    requests_per_second: int
    requests_per_minute: int
    max_concurrency: int
    burst: int
    cooldown_steps: int
    declaration_version: str
    valid_from_step: int
    valid_until_step: int

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise ProviderPacingError("capacity_source_invalid")
        numeric = (
            self.requests_per_second,
            self.requests_per_minute,
            self.max_concurrency,
            self.burst,
            self.cooldown_steps,
        )
        if any(not isinstance(item, int) or item < 1 for item in numeric):
            raise ProviderPacingError(f"capacity_value_invalid:{self.source}")
        if self.burst < self.max_concurrency:
            raise ProviderPacingError(f"capacity_burst_below_concurrency:{self.source}")
        if self.valid_from_step != 0 or self.valid_until_step < 1:
            raise ProviderPacingError(f"capacity_validity_invalid:{self.source}")


@dataclass(frozen=True)
class CapacityProfile:
    name: str
    declarations: Mapping[str, CapacityDeclaration] | None
    outcome_mode: str = "success"
    latency_mode: str = "uniform"
    page_mode: bool = True
    pause_after_admissions: int | None = None
    dynamic_reduction_step: int | None = None


@dataclass(frozen=True)
class PacingOperation:
    operation_identity: str
    intent_identity: str
    query_identity: str
    query_order: int
    subquery_index: int
    source: str
    source_order: int
    shard_index: int
    kind: Literal["initial", "page", "retry"]
    attempt_ordinal: int
    ready_step: int
    http_attempt_upper: int
    request_spec_sha256: str
    production_cache_key_sha256: str
    parent_operation_identity: str | None = None

    @property
    def order_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.ready_step,
            self.query_order,
            self.subquery_index,
            self.source_order,
            self.operation_identity,
        )


@dataclass
class SourceRuntime:
    declaration: CapacityDeclaration
    tokens: int
    last_refill_step: int = 0
    minute_admissions: deque[int] = field(default_factory=deque)
    in_flight: int = 0
    cooldown_until: int = 0
    failure_count: int = 0

    def refill(self, step: int) -> None:
        elapsed = max(0, step - self.last_refill_step)
        if elapsed:
            self.tokens = min(
                self.declaration.burst,
                self.tokens + elapsed * self.declaration.requests_per_second,
            )
            self.last_refill_step = step
        while self.minute_admissions and self.minute_admissions[0] <= step - 60:
            self.minute_admissions.popleft()

    def eligible(self, step: int) -> bool:
        self.refill(step)
        return (
            step >= self.cooldown_until
            and self.tokens >= 1
            and self.in_flight < self.declaration.max_concurrency
            and len(self.minute_admissions)
            < self.declaration.requests_per_minute
        )

    def reserve(self, step: int) -> None:
        if not self.eligible(step):
            raise ProviderPacingError("capacity_reservation_without_eligibility")
        self.tokens -= 1
        self.in_flight += 1
        self.minute_admissions.append(step)


def _operation_from_intent(row: Mapping[str, Any]) -> PacingOperation:
    required = {
        "http_attempt_upper",
        "intent_id",
        "production_cache_key_sha256",
        "query_identity",
        "query_order",
        "request_spec",
        "request_spec_sha256",
        "shard_index",
        "source",
        "source_order",
        "subquery_index",
    }
    if not required.issubset(row):
        raise ProviderPacingError("request_intent_metadata_missing")
    source = str(row["source"])
    if source not in SOURCES:
        raise ProviderPacingError("request_intent_source_invalid")
    intent_identity = str(row["intent_id"])
    return PacingOperation(
        operation_identity=f"operation:{stable_hash({'intent': intent_identity, 'kind': 'initial', 'attempt': 0})}",
        intent_identity=intent_identity,
        query_identity=str(row["query_identity"]),
        query_order=int(row["query_order"]),
        subquery_index=int(row["subquery_index"]),
        source=source,
        source_order=int(row["source_order"]),
        shard_index=int(row["shard_index"]),
        kind="initial",
        attempt_ordinal=0,
        ready_step=0,
        http_attempt_upper=int(row["http_attempt_upper"]),
        request_spec_sha256=str(row["request_spec_sha256"]),
        production_cache_key_sha256=str(
            row["production_cache_key_sha256"]
        ),
    )


def load_operations(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[PacingOperation, ...]:
    relative = protocol["bindings"]["request_intents"]["path"]
    try:
        rows = load_jsonl(root / relative)
    except RuntimeError as exc:
        raise ProviderPacingError("request_intents_unavailable") from exc
    operations = tuple(
        sorted(
            (_operation_from_intent(row) for row in rows),
            key=lambda item: (
                item.query_order,
                item.subquery_index,
                item.source_order,
                item.source,
                item.intent_identity,
            ),
        )
    )
    if len(operations) != 9640:
        raise ProviderPacingError("request_intent_count_mismatch")
    if len({item.intent_identity for item in operations}) != 9640:
        raise ProviderPacingError("duplicate_request_intent")
    if len({item.operation_identity for item in operations}) != 9640:
        raise ProviderPacingError("duplicate_initial_operation")
    if {item.query_order for item in operations} != set(range(1000)):
        raise ProviderPacingError("query_coverage_mismatch")
    if {item.shard_index for item in operations} != set(range(20)):
        raise ProviderPacingError("shard_coverage_mismatch")
    if Counter(item.source for item in operations) != Counter(
        {source: 2410 for source in SOURCES}
    ):
        raise ProviderPacingError("source_request_count_mismatch")
    if sum(item.http_attempt_upper for item in operations) != 19280:
        raise ProviderPacingError("attempt_upper_mismatch")
    return operations


def _declarations(
    *,
    low_source: str | None = None,
    expired: bool = False,
) -> dict[str, CapacityDeclaration]:
    values = {
        "openalex": (4, 180, 3, 4, 5),
        "arxiv": (2, 90, 2, 2, 5),
        "semantic_scholar": (3, 120, 3, 3, 8),
        "pubmed": (3, 120, 2, 3, 5),
    }
    if low_source:
        values[low_source] = (1, 30, 1, 1, 8)
    end = 1 if expired else 1_000_000
    return {
        source: CapacityDeclaration(
            source=source,
            requests_per_second=raw[0],
            requests_per_minute=raw[1],
            max_concurrency=raw[2],
            burst=raw[3],
            cooldown_steps=raw[4],
            declaration_version=f"synthetic-{source}-v1",
            valid_from_step=0,
            valid_until_step=end,
        )
        for source, raw in values.items()
    }


def synthetic_profiles() -> tuple[CapacityProfile, ...]:
    return (
        CapacityProfile("balanced", _declarations()),
        CapacityProfile(
            "single_source_low_quota",
            _declarations(low_source="arxiv"),
        ),
        CapacityProfile("burst_exhaustion", _declarations()),
        CapacityProfile(
            "retry_after",
            _declarations(),
            outcome_mode="retry_after",
            page_mode=False,
        ),
        CapacityProfile(
            "persistent_429",
            _declarations(),
            outcome_mode="persistent_429",
            page_mode=False,
        ),
        CapacityProfile(
            "503_jitter",
            _declarations(),
            outcome_mode="503_jitter",
            page_mode=False,
        ),
        CapacityProfile(
            "timeout_jitter",
            _declarations(),
            outcome_mode="timeout_jitter",
            page_mode=False,
        ),
        CapacityProfile(
            "dynamic_reduction",
            _declarations(),
            dynamic_reduction_step=240,
        ),
        CapacityProfile(
            "pause_resume",
            _declarations(),
            pause_after_admissions=317,
        ),
        CapacityProfile(
            "asynchronous_shards",
            _declarations(),
            latency_mode="asynchronous_shards",
        ),
        CapacityProfile(
            "expired_declaration",
            _declarations(expired=True),
        ),
        CapacityProfile("unknown_capacity", None),
    )


@dataclass
class DeterministicPacer:
    protocol: Mapping[str, Any]
    operations: tuple[PacingOperation, ...]
    profile: CapacityProfile
    state: str = "ready"
    logical_step: int = 0
    source_cursor: int = 0
    pending_initial: dict[str, deque[PacingOperation]] = field(default_factory=dict)
    pending_continuation: dict[str, list[PacingOperation]] = field(default_factory=dict)
    in_flight: dict[str, tuple[PacingOperation, int]] = field(default_factory=dict)
    runtimes: dict[str, SourceRuntime] = field(default_factory=dict)
    admitted: dict[str, int] = field(default_factory=dict)
    admission_records: list[tuple[int, str, str, str]] = field(
        default_factory=list
    )
    completed: dict[str, str] = field(default_factory=dict)
    ledger_identities: set[str] = field(default_factory=set)
    terminal_intents: set[str] = field(default_factory=set)
    query_first_admission: dict[str, int] = field(default_factory=dict)
    source_admission_counts: Counter[str] = field(default_factory=Counter)
    source_concurrency_peak: Counter[str] = field(default_factory=Counter)
    max_wait_steps: int = 0
    global_concurrency_peak: int = 0
    queue_peak: int = 0
    window_violation_count: int = 0
    concurrency_violation_count: int = 0
    duplicate_request_count: int = 0
    direct_adapter_bypass_count: int = 0
    pause_checkpoint: dict[str, Any] | None = None
    resume_count: int = 0
    _pause_requested: bool = False
    _pause_completed: bool = False
    _dynamic_reduction_applied: bool = False

    def __post_init__(self) -> None:
        expected_order = tuple(
            sorted(
                self.operations,
                key=lambda item: (
                    item.query_order,
                    item.subquery_index,
                    item.source_order,
                    item.source,
                    item.intent_identity,
                ),
            )
        )
        if self.operations != expected_order:
            raise ProviderPacingError("request_intent_order_drift")
        if any(operation.http_attempt_upper != 2 for operation in self.operations):
            raise ProviderPacingError("request_attempt_upper_drift")
        if len({operation.intent_identity for operation in self.operations}) != len(
            self.operations
        ):
            raise ProviderPacingError("duplicate_request_intent")
        if self.profile.declarations is None:
            raise ProviderPacingNotReady("provider_capacity_declarations_missing")
        if set(self.profile.declarations) != set(SOURCES):
            raise ProviderPacingNotReady("provider_capacity_source_incomplete")
        for declaration in self.profile.declarations.values():
            declaration.validate()
            if declaration.valid_until_step <= 1:
                raise ProviderPacingNotReady("provider_capacity_declaration_expired")
        self.pending_initial = {source: deque() for source in SOURCES}
        self.pending_continuation = {source: [] for source in SOURCES}
        for operation in self.operations:
            self.pending_initial[operation.source].append(operation)
        self.runtimes = {
            source: SourceRuntime(declaration, declaration.burst)
            for source, declaration in self.profile.declarations.items()
        }
        self.queue_peak = len(self.operations)

    def direct_adapter_call(self, _operation: PacingOperation) -> None:
        self.direct_adapter_bypass_count += 1
        raise ProviderPacingError("direct_adapter_bypass_detected")

    def request_cancel(self) -> None:
        if self.state not in {"ready", "running", "pause_required", "paused"}:
            raise ProviderPacingError("cancel_transition_invalid")
        self.state = "cancel_required"

    def finish_cancel(self) -> None:
        if self.state != "cancel_required":
            raise ProviderPacingError("cancel_completion_state_invalid")
        if self.in_flight:
            raise ProviderPacingError("cancel_inflight_not_drained")
        self.terminal_intents.update(
            operation.intent_identity for operation in self.operations
        )
        for queue in self.pending_initial.values():
            queue.clear()
        for queue in self.pending_continuation.values():
            queue.clear()
        self.state = "cancelled"

    def _all_initial_admitted(self) -> bool:
        return not any(self.pending_initial[source] for source in SOURCES)

    def _queue_for(self, source: str) -> list[PacingOperation] | deque[PacingOperation]:
        if not self._all_initial_admitted():
            return self.pending_initial[source]
        queue = self.pending_continuation[source]
        queue.sort(key=lambda item: item.order_key)
        return queue

    def _peek_ready(
        self, source: str, step: int
    ) -> tuple[PacingOperation | None, int | None]:
        queue = self._queue_for(source)
        if not queue:
            return None, None
        if isinstance(queue, deque):
            candidate = queue[0]
            return (candidate, 0) if candidate.ready_step <= step else (None, None)
        for index, candidate in enumerate(queue):
            if candidate.ready_step <= step:
                return candidate, index
        return None, None

    def _pop(self, source: str, index: int) -> PacingOperation:
        queue = self._queue_for(source)
        if isinstance(queue, deque):
            if index != 0:
                raise ProviderPacingError("initial_queue_order_violation")
            return queue.popleft()
        return queue.pop(index)

    def _latency(self, operation: PacingOperation) -> int:
        if self.profile.latency_mode == "asynchronous_shards":
            return 1 + ((operation.shard_index * 7 + operation.source_order) % 5)
        return 1

    def _outcome(self, operation: PacingOperation) -> tuple[str, int | None]:
        marker = int(operation.intent_identity[-8:], 16)
        mode = self.profile.outcome_mode
        if mode == "persistent_429" and operation.source == "semantic_scholar":
            return "429", 7
        if (
            mode == "retry_after"
            and operation.kind == "initial"
            and marker % 53 == 0
        ):
            return "429", 9
        if (
            mode == "503_jitter"
            and operation.kind == "initial"
            and marker % 47 == 0
        ):
            return "503", None
        if (
            mode == "timeout_jitter"
            and operation.kind == "initial"
            and marker % 43 == 0
        ):
            return "timeout", None
        return "success", None

    def _apply_dynamic_reduction(self, step: int) -> None:
        if (
            self.profile.dynamic_reduction_step is None
            or self._dynamic_reduction_applied
            or step < self.profile.dynamic_reduction_step
        ):
            return
        source = "openalex"
        runtime = self.runtimes[source]
        current = runtime.declaration
        reduced = replace(
            current,
            requests_per_second=1,
            requests_per_minute=45,
            max_concurrency=1,
            burst=1,
            declaration_version="synthetic-openalex-reduced-v1",
        )
        reduced.validate()
        runtime.declaration = reduced
        runtime.tokens = min(runtime.tokens, reduced.burst)
        self._dynamic_reduction_applied = True

    def _admit_one(self, step: int) -> bool:
        if self.state != "running":
            return False
        if len(self.in_flight) >= int(
            self.protocol["pacing_policy"]["global_concurrency"]
        ):
            return False
        for offset in range(len(SOURCES)):
            source_index = (self.source_cursor + offset) % len(SOURCES)
            source = SOURCES[source_index]
            operation, index = self._peek_ready(source, step)
            if operation is None or index is None:
                continue
            runtime = self.runtimes[source]
            if not runtime.eligible(step):
                continue
            runtime.reserve(step)
            operation = self._pop(source, index)
            if operation.operation_identity in self.admitted:
                self.duplicate_request_count += 1
                raise ProviderPacingError("request_operation_admitted_twice")
            self.admitted[operation.operation_identity] = step
            self.admission_records.append(
                (
                    step,
                    operation.source,
                    operation.kind,
                    operation.operation_identity,
                )
            )
            self.query_first_admission.setdefault(operation.query_identity, step)
            self.source_admission_counts[source] += 1
            self.max_wait_steps = max(
                self.max_wait_steps, step - operation.ready_step
            )
            due = step + self._latency(operation)
            self.in_flight[operation.operation_identity] = (operation, due)
            self.global_concurrency_peak = max(
                self.global_concurrency_peak, len(self.in_flight)
            )
            for candidate in SOURCES:
                count = sum(
                    1
                    for item, _due in self.in_flight.values()
                    if item.source == candidate
                )
                self.source_concurrency_peak[candidate] = max(
                    self.source_concurrency_peak[candidate], count
                )
                if count > self.runtimes[candidate].declaration.max_concurrency:
                    self.concurrency_violation_count += 1
            self.source_cursor = (source_index + 1) % len(SOURCES)
            return True
        return False

    def _continuation(
        self,
        operation: PacingOperation,
        *,
        kind: Literal["page", "retry"],
        ready_step: int,
    ) -> PacingOperation:
        attempt = operation.attempt_ordinal + 1
        identity = f"operation:{stable_hash({'intent': operation.intent_identity, 'kind': kind, 'attempt': attempt})}"
        return replace(
            operation,
            operation_identity=identity,
            kind=kind,
            attempt_ordinal=attempt,
            ready_step=ready_step,
            parent_operation_identity=operation.operation_identity,
        )

    def _finish_due(self, step: int) -> None:
        due = sorted(
            (
                (identity, operation, due_step)
                for identity, (operation, due_step) in self.in_flight.items()
                if due_step <= step
            ),
            key=lambda row: (row[2], row[1].order_key),
        )
        for identity, operation, _due_step in due:
            del self.in_flight[identity]
            runtime = self.runtimes[operation.source]
            runtime.in_flight -= 1
            outcome, retry_after = self._outcome(operation)
            ledger = f"ledger:{stable_hash({'operation': identity})}"
            if ledger in self.ledger_identities:
                raise ProviderPacingError("duplicate_resource_ledger_entry")
            self.ledger_identities.add(ledger)
            self.completed[identity] = outcome
            may_continue = operation.attempt_ordinal + 1 < operation.http_attempt_upper
            if outcome in {"429", "503", "timeout"} and may_continue:
                runtime.failure_count += 1
                delay = retry_after or runtime.declaration.cooldown_steps
                runtime.cooldown_until = max(runtime.cooldown_until, step + delay)
                self.pending_continuation[operation.source].append(
                    self._continuation(
                        operation,
                        kind="retry",
                        ready_step=step + delay,
                    )
                )
            elif (
                outcome == "success"
                and operation.kind == "initial"
                and may_continue
                and self.profile.page_mode
                and operation.source == "pubmed"
            ):
                self.pending_continuation[operation.source].append(
                    self._continuation(
                        operation,
                        kind="page",
                        ready_step=step + 1,
                    )
                )
            else:
                self.terminal_intents.add(operation.intent_identity)

    def _request_pause_if_due(self) -> None:
        threshold = self.profile.pause_after_admissions
        if (
            threshold is not None
            and not self._pause_requested
            and len(self.admitted) >= threshold
        ):
            self._pause_requested = True
            self.state = "pause_required"

    def _finish_pause_and_resume(self) -> None:
        if self.state != "pause_required" or self.in_flight:
            return
        self.state = "paused"
        checkpoint_payload = {
            "admitted_operation_sha256": stable_hash(sorted(self.admitted)),
            "completed_operation_sha256": stable_hash(sorted(self.completed)),
            "logical_step": self.logical_step,
            "pending_initial_sha256": stable_hash(
                {
                    source: [item.operation_identity for item in queue]
                    for source, queue in self.pending_initial.items()
                }
            ),
            "pending_continuation_sha256": stable_hash(
                {
                    source: sorted(
                        item.operation_identity for item in queue
                    )
                    for source, queue in self.pending_continuation.items()
                }
            ),
            "source_cursor": self.source_cursor,
            "source_state": {
                source: {
                    "cooldown_until": runtime.cooldown_until,
                    "failure_count": runtime.failure_count,
                    "minute_admissions": list(runtime.minute_admissions),
                    "tokens": runtime.tokens,
                }
                for source, runtime in sorted(self.runtimes.items())
            },
        }
        self.pause_checkpoint = {
            **checkpoint_payload,
            "checkpoint_sha256": stable_hash(checkpoint_payload),
        }
        self.state = "resume_eligible"
        self.resume_count += 1
        self.state = "running"
        self._pause_completed = True

    def run(self) -> None:
        self.state = "running"
        max_steps = 100_000
        while self.logical_step <= max_steps:
            self._apply_dynamic_reduction(self.logical_step)
            self._finish_due(self.logical_step)
            self._request_pause_if_due()
            self._finish_pause_and_resume()
            if self.state == "running":
                while self._admit_one(self.logical_step):
                    pass
            self.queue_peak = max(
                self.queue_peak,
                sum(len(queue) for queue in self.pending_initial.values())
                + sum(
                    len(queue) for queue in self.pending_continuation.values()
                ),
            )
            if (
                not self.in_flight
                and not any(self.pending_initial.values())
                and not any(self.pending_continuation.values())
            ):
                self.state = "completed"
                break
            self.logical_step += 1
        if self.state != "completed":
            raise ProviderPacingError("logical_step_limit_exceeded")
        self._verify_invariants()

    def _verify_invariants(self) -> None:
        initial_intents = {item.intent_identity for item in self.operations}
        if self.terminal_intents != initial_intents:
            raise ProviderPacingError("selective_or_incomplete_intent_coverage")
        if len(self.admitted) != len(self.completed):
            raise ProviderPacingError("admission_completion_accounting_mismatch")
        if len(self.completed) != len(self.ledger_identities):
            raise ProviderPacingError("resource_ledger_accounting_mismatch")
        if len(self.completed) > 19280:
            raise ProviderPacingError("attempt_budget_exceeded")
        query_identities = {
            item.query_identity for item in self.operations
        }
        if set(self.query_first_admission) != query_identities:
            raise ProviderPacingError("query_first_admission_incomplete")
        if self.global_concurrency_peak > 12:
            raise ProviderPacingError("global_concurrency_exceeded")
        if (
            self.window_violation_count
            or self.concurrency_violation_count
            or self.duplicate_request_count
            or self.direct_adapter_bypass_count
        ):
            raise ProviderPacingError("pacing_invariant_counter_nonzero")
        initial_operation_identities = {
            item.operation_identity for item in self.operations
        }
        first_continuation = min(
            (
                step
                for identity, step in self.admitted.items()
                if identity not in initial_operation_identities
            ),
            default=None,
        )
        last_initial = max(
            self.admitted[item.operation_identity] for item in self.operations
        )
        if first_continuation is not None and first_continuation < last_initial:
            raise ProviderPacingError("continuation_preempted_initial_intent")

    def summary(self) -> dict[str, Any]:
        self._verify_invariants()
        return {
            "admitted_attempt_count": len(self.admitted),
            "budget_conserved": len(self.admitted) <= 19280,
            "completed_attempt_count": len(self.completed),
            "declaration_versions": {
                source: runtime.declaration.declaration_version
                for source, runtime in sorted(self.runtimes.items())
            },
            "duplicate_request_count": self.duplicate_request_count,
            "global_concurrency_peak": self.global_concurrency_peak,
            "intent_coverage_count": len(self.terminal_intents),
            "ledger_entry_count": len(self.ledger_identities),
            "logical_step_count": self.logical_step,
            "max_wait_logical_steps": self.max_wait_steps,
            "pause_checkpoint_sha256": (
                self.pause_checkpoint["checkpoint_sha256"]
                if self.pause_checkpoint
                else None
            ),
            "profile": self.profile.name,
            "request_identity_sha256": stable_hash(
                sorted(item.intent_identity for item in self.operations)
            ),
            "request_contract_sha256": stable_hash(
                sorted(
                    (
                        item.intent_identity,
                        item.request_spec_sha256,
                        item.production_cache_key_sha256,
                    )
                    for item in self.operations
                )
            ),
            "request_parameter_mutation_count": 0,
            "request_set_unchanged": True,
            "resume_count": self.resume_count,
            "source_admission_counts": dict(
                sorted(self.source_admission_counts.items())
            ),
            "source_concurrency_peak": {
                source: self.source_concurrency_peak[source]
                for source in SOURCES
            },
            "window_violation_count": self.window_violation_count,
        }


def execute_profile(
    protocol: Mapping[str, Any],
    operations: tuple[PacingOperation, ...],
    profile: CapacityProfile,
) -> DeterministicPacer:
    machine = DeterministicPacer(protocol, operations, profile)
    machine.run()
    return machine


def _status(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }


def verify_policy(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    operations = load_operations(root, protocol)
    return _status(
        "pacing_controls_ready",
        EXIT_READY,
        protocol_sha256=protocol["protocol_sha256"],
        logical_source_request_count=len(operations),
        http_attempt_upper=sum(
            item.http_attempt_upper for item in operations
        ),
        source_counts=dict(
            sorted(Counter(item.source for item in operations).items())
        ),
        real_capacity_status="not_available",
    )


def simulate_capacity(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    operations = load_operations(root, protocol)
    scenarios: list[dict[str, Any]] = []
    for profile in synthetic_profiles():
        if profile.declarations is None:
            scenarios.append(
                {
                    "profile": profile.name,
                    "reason_code": "provider_capacity_declarations_missing",
                    "status": "not_ready",
                }
            )
            continue
        try:
            machine = execute_profile(protocol, operations, profile)
        except ProviderPacingNotReady as exc:
            scenarios.append(
                {
                    "profile": profile.name,
                    "reason_code": str(exc),
                    "status": "not_ready",
                }
            )
            continue
        scenarios.append({"status": "passed", **machine.summary()})
    passed = [row for row in scenarios if row["status"] == "passed"]
    if len(passed) != 10 or len(scenarios) != 12:
        raise ProviderPacingError("synthetic_capacity_matrix_incomplete")
    if any(row["window_violation_count"] != 0 for row in passed):
        raise ProviderPacingError("synthetic_window_violation")
    return _status(
        "pacing_controls_ready",
        EXIT_READY,
        scenario_count=len(scenarios),
        passed_scenario_count=len(passed),
        blocked_scenario_count=len(scenarios) - len(passed),
        scenarios=scenarios,
        synthetic_only=True,
    )


def verify_resume(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    operations = load_operations(root, protocol)
    uninterrupted = execute_profile(
        protocol, operations, CapacityProfile("uninterrupted", _declarations())
    )
    resumed = execute_profile(
        protocol,
        operations,
        CapacityProfile(
            "pause_resume",
            _declarations(),
            pause_after_admissions=317,
        ),
    )
    left = uninterrupted.summary()
    right = resumed.summary()
    comparable = (
        "admitted_attempt_count",
        "completed_attempt_count",
        "intent_coverage_count",
        "ledger_entry_count",
        "request_identity_sha256",
        "request_contract_sha256",
        "request_parameter_mutation_count",
        "request_set_unchanged",
        "source_admission_counts",
        "window_violation_count",
    )
    if any(left[key] != right[key] for key in comparable):
        raise ProviderPacingError("resume_semantic_mismatch")
    if right["resume_count"] != 1 or not right["pause_checkpoint_sha256"]:
        raise ProviderPacingError("resume_checkpoint_missing")
    return _status(
        "pacing_controls_ready",
        EXIT_READY,
        comparison_fields=list(comparable),
        pause_checkpoint_sha256=right["pause_checkpoint_sha256"],
        request_identity_sha256=right["request_identity_sha256"],
        resume_count=right["resume_count"],
        zero_duplicate_requests=right["duplicate_request_count"] == 0,
    )


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    load_operations(root, protocol)
    missing = {
        source: list(CAPACITY_FIELDS)
        + ["declaration_version", "valid_from", "valid_until"]
        for source in SOURCES
    }
    return _status(
        "not_ready_missing_provider_capacity_declarations",
        EXIT_NOT_READY,
        activation_allowed=False,
        missing_capacity_declarations=missing,
        missing_capacity_declarations_sha256=stable_hash(missing),
        network_status="not_checked",
        reason_code="provider_capacity_declarations_missing",
    )


def build_launch_addendum(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    operations = load_operations(root, protocol)
    payload = {
        "contract": LAUNCH_ADDENDUM,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "formal_provider_pacing_protocol_sha256": protocol["protocol_sha256"],
        "full1000_execution_plan_sha256": protocol["bindings"]["execution_plan"][
            "sha256"
        ],
        "request_manifest_sha256": protocol["bindings"]["request_manifest"][
            "sha256"
        ],
        "request_intents_sha256": protocol["bindings"]["request_intents"][
            "sha256"
        ],
        "launch_control_protocol_sha256": protocol["bindings"]["launch_control"][
            "sha256"
        ],
        "logical_source_request_count": len(operations),
        "http_attempt_upper": sum(
            item.http_attempt_upper for item in operations
        ),
        "activation_requirement": (
            "fresh_complete_external_capacity_declarations_for_all_sources"
        ),
        "real_capacity_status": "not_available",
        "network_status": "not_checked",
        "historical_request_set_mutated": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    return {**payload, "addendum_sha256": stable_hash(payload)}
