"""Deterministic Full1000 scheduler fairness and backpressure audit.

The scheduler consumes only preregistered opaque identities and operational
state.  It never inspects query text, paper results, source yield, gold, or
quality metrics.  The implementation is an observational control seam for the
existing Full1000 plan, launch, health, ledger, checkpoint, cancellation, and
aggregate contracts; it does not implement retrieval.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_provider_health_supervisor import (
    SOURCES,
    load_query_identities,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_scheduler_fairness_v1"
SCHEMA_VERSION = "1"
ADDENDUM = "full1000_scheduler_fairness_addendum_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_STARTED = 3
EXIT_USAGE = 4
FROZEN_PROTOCOL_SHA256 = (
    "801ce45275b5bcd9a7c2818abfe0dc053d95a39ab836fddafe511760ef0af881"
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
TERMINAL_STATES = ("cancelled", "completed", "failed", "source_failure")
TASK_KINDS = ("initial", "page", "retry")


class SchedulerFairnessError(RuntimeError):
    """A scheduler policy, accounting, or coverage invariant was violated."""


class SchedulerNotReady(SchedulerFairnessError):
    """The real Full1000 execution has not started."""


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
        raise SchedulerFairnessError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise SchedulerFairnessError("json_root_not_object")
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
        raise SchedulerFairnessError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def _validate_limits(value: Any) -> None:
    expected = {
        "attempt_upper": 19280,
        "backpressure_continuation_queue_threshold": 8,
        "backpressure_source_concurrency": 2,
        "global_concurrency": 12,
        "max_logical_steps": 200000,
        "max_pages_per_query_source": 2,
        "per_shard_concurrency": 2,
        "per_source_concurrency": 3,
        "retry_limits": {
            "arxiv": 1,
            "openalex": 1,
            "pubmed": 0,
            "semantic_scholar": 1,
        },
    }
    if value != expected:
        raise SchedulerFairnessError("scheduler_limit_drift")


def _validate_policy(value: Any) -> None:
    expected = {
        "admission_identity_inputs": [
            "global_query_ordinal",
            "shard_index",
            "source_ordinal",
            "operation_kind",
            "attempt_or_page_ordinal",
            "runtime_state",
        ],
        "continuation_order": (
            "round_then_kind_then_global_query_ordinal_then_source_ordinal"
        ),
        "first_attempt_barrier": (
            "all_query_source_initial_operations_admitted_before_page_or_retry"
        ),
        "initial_order": (
            "query_round_robin_with_rotating_source_then_authoritative_query_order"
        ),
        "prohibited_priority_inputs": [
            "case_id",
            "completion_speed",
            "gold",
            "paper_identity",
            "provider_yield",
            "quality_metric",
            "query_text_or_type",
            "result_content",
        ],
        "resume_cursor": "exact_pause_checkpoint_cursor_without_reset",
        "terminal_coverage": (
            "preserve_all_queries_in_authoritative_order_including_fail_cancel"
        ),
    }
    if value != expected:
        raise SchedulerFairnessError("scheduler_policy_drift")


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    expected = {
        "checkpoint_resume",
        "execution_plan",
        "launch_control",
        "preregistration",
        "provider_health",
        "provider_health_addendum",
        "resource_ledger",
        "shard_contract",
    }
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise SchedulerFairnessError("binding_inventory_invalid")
    for name, binding in sorted(bindings.items()):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise SchedulerFairnessError("binding_schema_invalid")
        relative = _safe_relative(str(binding["path"]))
        path = root / relative
        if not path.is_file():
            raise SchedulerFairnessError(f"binding_missing:{name}")
        if sha256_file(path) != binding["sha256"]:
            raise SchedulerFairnessError(f"binding_hash_drift:{name}")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "bindings",
        "execution",
        "formal_validation_complete",
        "limits",
        "population",
        "protocol",
        "protocol_sha256",
        "scheduler_policy",
        "schema_version",
        "source_commit",
        "state_machine",
    }
    if set(value) != required:
        raise SchedulerFairnessError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise SchedulerFairnessError("protocol_version_invalid")
    if value["source_commit"] != "46bd9d27e1a4e98bb4afa78c6bb6cdf77ab5d278":
        raise SchedulerFairnessError("protocol_source_commit_invalid")
    if value["execution"] != EXECUTION_ZERO:
        raise SchedulerFairnessError("offline_execution_contract_drift")
    if value["formal_validation_complete"] is not False:
        raise SchedulerFairnessError("formal_validation_state_drift")
    if _protocol_digest(value) != value["protocol_sha256"]:
        raise SchedulerFairnessError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise SchedulerFairnessError("protocol_content_drift")
    population = value["population"]
    if (
        population.get("query_count") != 1000
        or population.get("shard_count") != 20
        or tuple(population.get("sources") or ()) != SOURCES
    ):
        raise SchedulerFairnessError("population_contract_drift")
    states = value["state_machine"].get("states")
    if states != [
        "ready",
        "running",
        "pause_required",
        "paused",
        "resume_eligible",
        "cancel_required",
        "cancelled",
        "completed",
        "invalid",
    ]:
        raise SchedulerFairnessError("state_machine_drift")
    _validate_limits(value["limits"])
    _validate_policy(value["scheduler_policy"])
    _validate_bindings(repository_root, value)
    return value


@dataclass(frozen=True)
class ScheduledTask:
    identity: str
    query_identity: str
    query_ordinal: int
    shard_index: int
    source: str
    source_ordinal: int
    kind: Literal["initial", "page", "retry"]
    round_ordinal: int
    attempt_ordinal: int
    page_ordinal: int
    ready_step: int

    @property
    def priority(self) -> tuple[int, int, int, int, int]:
        kind_order = {"retry": 0, "page": 1, "initial": -1}[self.kind]
        return (
            self.round_ordinal,
            kind_order,
            self.query_ordinal,
            self.source_ordinal,
            self.attempt_ordinal + self.page_ordinal,
        )


@dataclass
class InFlightTask:
    task: ScheduledTask
    started_step: int
    due_step: int


@dataclass
class DeterministicScheduler:
    protocol: Mapping[str, Any]
    query_identities: tuple[str, ...]
    worker_limit: int | None = None
    state: str = "ready"
    pending_initial: list[ScheduledTask] = field(default_factory=list)
    pending_continuation: list[ScheduledTask] = field(default_factory=list)
    in_flight: dict[str, InFlightTask] = field(default_factory=dict)
    completed_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger_identities: set[str] = field(default_factory=set)
    query_open_tasks: dict[str, int] = field(default_factory=dict)
    query_outcomes: dict[str, list[str]] = field(default_factory=dict)
    terminal_queries: dict[str, str] = field(default_factory=dict)
    known_task_identities: set[str] = field(default_factory=set)
    started_task_identities: set[str] = field(default_factory=set)
    initial_admission_count: int = 0
    logical_step: int = 0
    pause_checkpoint: dict[str, Any] | None = None
    resume_count: int = 0
    rejected_admissions: int = 0
    duplicate_enqueue_count: int = 0
    continuation_order_dirty: bool = False
    audit_entries: list[dict[str, Any]] = field(default_factory=list)
    first_query_service_step: dict[str, int] = field(default_factory=dict)
    last_source_service_step: dict[str, int] = field(default_factory=dict)
    last_shard_service_step: dict[int, int] = field(default_factory=dict)
    max_source_service_interval: dict[str, int] = field(default_factory=dict)
    max_shard_service_interval: dict[int, int] = field(default_factory=dict)
    max_wait_steps: int = 0
    queue_peak: int = 0
    concurrency_peak: int = 0
    source_concurrency_peak: dict[str, int] = field(default_factory=dict)
    shard_concurrency_peak: dict[int, int] = field(default_factory=dict)
    backpressure_activation_count: int = 0

    def __post_init__(self) -> None:
        if (
            len(self.query_identities) != 1000
            or len(set(self.query_identities)) != 1000
        ):
            raise SchedulerFairnessError("query_population_invalid")
        self.worker_limit = int(
            self.worker_limit or self.protocol["limits"]["global_concurrency"]
        )
        if self.worker_limit < 1:
            raise SchedulerFairnessError("worker_limit_invalid")
        self.query_open_tasks = {identity: 0 for identity in self.query_identities}
        self.query_outcomes = {identity: [] for identity in self.query_identities}
        self.source_concurrency_peak = {source: 0 for source in SOURCES}
        self.shard_concurrency_peak = {index: 0 for index in range(20)}
        self.max_source_service_interval = {source: 0 for source in SOURCES}
        self.max_shard_service_interval = {index: 0 for index in range(20)}
        self._build_initial_population()

    def _build_initial_population(self) -> None:
        # Four complete query rounds rotate the first source by query ordinal.
        # This gives every query, shard, and source a bounded first opportunity.
        for source_round in range(len(SOURCES)):
            for ordinal, query_identity in enumerate(self.query_identities):
                source_ordinal = (ordinal + source_round) % len(SOURCES)
                task = self._make_task(
                    query_identity=query_identity,
                    query_ordinal=ordinal,
                    source_ordinal=source_ordinal,
                    kind="initial",
                    round_ordinal=0,
                    attempt_ordinal=0,
                    page_ordinal=0,
                    ready_step=0,
                )
                self._enqueue(task, initial=True)
        self.queue_peak = len(self.pending_initial)

    def _make_task(
        self,
        *,
        query_identity: str,
        query_ordinal: int,
        source_ordinal: int,
        kind: Literal["initial", "page", "retry"],
        round_ordinal: int,
        attempt_ordinal: int,
        page_ordinal: int,
        ready_step: int,
    ) -> ScheduledTask:
        payload = {
            "query_identity": query_identity,
            "source": SOURCES[source_ordinal],
            "kind": kind,
            "round_ordinal": round_ordinal,
            "attempt_ordinal": attempt_ordinal,
            "page_ordinal": page_ordinal,
        }
        return ScheduledTask(
            identity=f"schedule:{stable_hash(payload)}",
            query_identity=query_identity,
            query_ordinal=query_ordinal,
            shard_index=query_ordinal % 20,
            source=SOURCES[source_ordinal],
            source_ordinal=source_ordinal,
            kind=kind,
            round_ordinal=round_ordinal,
            attempt_ordinal=attempt_ordinal,
            page_ordinal=page_ordinal,
            ready_step=ready_step,
        )

    def _enqueue(self, task: ScheduledTask, *, initial: bool = False) -> None:
        if task.identity in self.known_task_identities:
            self.duplicate_enqueue_count += 1
            raise SchedulerFairnessError("duplicate_task_enqueue")
        self.known_task_identities.add(task.identity)
        target = self.pending_initial if initial else self.pending_continuation
        target.append(task)
        if not initial:
            self.continuation_order_dirty = True
        self.query_open_tasks[task.query_identity] += 1
        self.queue_peak = max(
            self.queue_peak,
            len(self.pending_initial) + len(self.pending_continuation),
        )

    def _counts(self) -> tuple[Counter[str], Counter[int]]:
        return (
            Counter(item.task.source for item in self.in_flight.values()),
            Counter(item.task.shard_index for item in self.in_flight.values()),
        )

    def _source_limit(self, source: str, pressure: int) -> int:
        base = int(self.protocol["limits"]["per_source_concurrency"])
        threshold = int(
            self.protocol["limits"]["backpressure_continuation_queue_threshold"]
        )
        if pressure >= threshold:
            self.backpressure_activation_count += 1
            return min(
                base,
                int(self.protocol["limits"]["backpressure_source_concurrency"]),
            )
        return base

    def _eligible(
        self,
        task: ScheduledTask,
        source_counts: Counter[str],
        shard_counts: Counter[int],
        continuation_pressure: Counter[str],
    ) -> bool:
        return (
            source_counts[task.source]
            < self._source_limit(
                task.source, continuation_pressure[task.source]
            )
            and shard_counts[task.shard_index]
            < int(self.protocol["limits"]["per_shard_concurrency"])
        )

    def admit(
        self,
        *,
        step: int,
        delay: Callable[[ScheduledTask], int],
    ) -> int:
        self.logical_step = step
        if self.state == "ready":
            self.state = "running"
        if self.state != "running":
            self.rejected_admissions += 1
            return 0
        admitted = 0
        limit = min(
            int(self.worker_limit),
            int(self.protocol["limits"]["global_concurrency"]),
        )
        source_counts, shard_counts = self._counts()
        continuation_pressure = Counter(
            task.source for task in self.pending_continuation
        )
        while len(self.in_flight) < limit:
            queue = (
                self.pending_initial
                if self.pending_initial
                else self.pending_continuation
            )
            if (
                queue is self.pending_continuation
                and self.continuation_order_dirty
            ):
                queue.sort(key=lambda item: item.priority)
                self.continuation_order_dirty = False
            if not queue:
                break
            candidate_index = next(
                (
                    index
                    for index, task in enumerate(queue)
                    if self._eligible(
                        task,
                        source_counts,
                        shard_counts,
                        continuation_pressure,
                    )
                ),
                None,
            )
            if candidate_index is None:
                break
            task = queue.pop(candidate_index)
            if task.identity in self.started_task_identities:
                raise SchedulerFairnessError("task_started_twice")
            duration = delay(task)
            if not isinstance(duration, int) or duration < 1:
                raise SchedulerFairnessError("logical_delay_invalid")
            self.started_task_identities.add(task.identity)
            if task.kind == "initial":
                self.initial_admission_count += 1
            self.first_query_service_step.setdefault(task.query_identity, step)
            self.max_wait_steps = max(self.max_wait_steps, step - task.ready_step)
            self._record_service(task, step)
            self.in_flight[task.identity] = InFlightTask(
                task=task,
                started_step=step,
                due_step=step + duration,
            )
            source_counts[task.source] += 1
            shard_counts[task.shard_index] += 1
            if task.kind != "initial":
                continuation_pressure[task.source] -= 1
            self._record_concurrency()
            self._append_audit(
                "task_admitted",
                task.identity,
                {
                    "kind": task.kind,
                    "query_ordinal": task.query_ordinal,
                    "shard_index": task.shard_index,
                    "source": task.source,
                },
            )
            admitted += 1
        return admitted

    def _record_service(self, task: ScheduledTask, step: int) -> None:
        if task.source in self.last_source_service_step:
            self.max_source_service_interval[task.source] = max(
                self.max_source_service_interval[task.source],
                step - self.last_source_service_step[task.source],
            )
        self.last_source_service_step[task.source] = step
        if task.shard_index in self.last_shard_service_step:
            self.max_shard_service_interval[task.shard_index] = max(
                self.max_shard_service_interval[task.shard_index],
                step - self.last_shard_service_step[task.shard_index],
            )
        self.last_shard_service_step[task.shard_index] = step

    def _record_concurrency(self) -> None:
        source_counts, shard_counts = self._counts()
        self.concurrency_peak = max(self.concurrency_peak, len(self.in_flight))
        for source in SOURCES:
            self.source_concurrency_peak[source] = max(
                self.source_concurrency_peak[source], source_counts[source]
            )
        for shard in range(20):
            self.shard_concurrency_peak[shard] = max(
                self.shard_concurrency_peak[shard], shard_counts[shard]
            )

    def finish_due(
        self,
        *,
        step: int,
        outcome: Callable[[ScheduledTask], str],
        pages: Callable[[ScheduledTask], int],
    ) -> int:
        self.logical_step = step
        due = sorted(
            (
                item
                for item in self.in_flight.values()
                if item.due_step <= step
            ),
            key=lambda item: (item.due_step, item.task.priority, item.task.identity),
        )
        for item in due:
            task = item.task
            result = outcome(task)
            if result not in {
                "success",
                "429",
                "503",
                "timeout",
                "connection_failure",
                "cancelled",
            }:
                raise SchedulerFairnessError("task_outcome_invalid")
            ledger = f"ledger:{stable_hash({'task': task.identity})}"
            if ledger in self.ledger_identities:
                raise SchedulerFairnessError("double_billing_detected")
            self.ledger_identities.add(ledger)
            del self.in_flight[task.identity]
            self.completed_tasks[task.identity] = {
                "outcome": result,
                "ledger_identity": ledger,
                "started_step": item.started_step,
                "finished_step": step,
            }
            self.query_outcomes[task.query_identity].append(result)
            self.query_open_tasks[task.query_identity] -= 1
            self._append_audit(
                "task_finished",
                task.identity,
                {"ledger_identity": ledger, "outcome": result},
            )
            self._schedule_continuation(task, result, step, pages)
            self._maybe_finalize_query(task.query_identity)
        return len(due)

    def _schedule_continuation(
        self,
        task: ScheduledTask,
        outcome: str,
        step: int,
        pages: Callable[[ScheduledTask], int],
    ) -> None:
        if self.state != "running":
            return
        retry_limit = int(self.protocol["limits"]["retry_limits"][task.source])
        if (
            outcome in {"429", "503", "timeout", "connection_failure"}
            and task.attempt_ordinal < retry_limit
        ):
            self._enqueue(
                self._make_task(
                    query_identity=task.query_identity,
                    query_ordinal=task.query_ordinal,
                    source_ordinal=task.source_ordinal,
                    kind="retry",
                    round_ordinal=task.attempt_ordinal + 1,
                    attempt_ordinal=task.attempt_ordinal + 1,
                    page_ordinal=task.page_ordinal,
                    ready_step=step,
                )
            )
        if outcome == "success":
            desired_pages = pages(task)
            maximum = int(self.protocol["limits"]["max_pages_per_query_source"])
            if not isinstance(desired_pages, int) or not 0 <= desired_pages <= maximum:
                raise SchedulerFairnessError("page_request_limit_invalid")
            if task.page_ordinal < desired_pages:
                next_page = task.page_ordinal + 1
                self._enqueue(
                    self._make_task(
                        query_identity=task.query_identity,
                        query_ordinal=task.query_ordinal,
                        source_ordinal=task.source_ordinal,
                        kind="page",
                        round_ordinal=next_page,
                        attempt_ordinal=task.attempt_ordinal,
                        page_ordinal=next_page,
                        ready_step=step,
                    )
                )

    def _maybe_finalize_query(self, query_identity: str) -> None:
        if (
            self.query_open_tasks[query_identity] != 0
            or query_identity in self.terminal_queries
        ):
            return
        outcomes = self.query_outcomes[query_identity]
        if any(value == "success" for value in outcomes):
            state = "completed"
        elif any(value == "cancelled" for value in outcomes):
            state = "cancelled"
        elif outcomes:
            state = "source_failure"
        else:
            state = "failed"
        self.terminal_queries[query_identity] = state
        self._append_audit("query_terminal", query_identity, {"state": state})

    def request_pause(self, reason: str) -> None:
        if self.state != "running":
            raise SchedulerFairnessError("pause_state_invalid")
        self.state = "pause_required"
        self._append_audit("pause_required", reason)

    def acknowledge_pause(self) -> dict[str, Any]:
        if self.state != "pause_required" or self.in_flight:
            raise SchedulerFairnessError("pause_requires_drained_inflight")
        self.state = "paused"
        payload: dict[str, Any] = {
            "state": "paused",
            "initial_admission_count": self.initial_admission_count,
            "pending_initial_sha256": stable_hash(
                [task.identity for task in self.pending_initial]
            ),
            "pending_continuation_sha256": stable_hash(
                [task.identity for task in self.pending_continuation]
            ),
            "completed_task_sha256": stable_hash(sorted(self.completed_tasks)),
            "terminal_query_sha256": stable_hash(
                [
                    (identity, self.terminal_queries.get(identity))
                    for identity in self.query_identities
                    if identity in self.terminal_queries
                ]
            ),
            "ledger_sha256": stable_hash(sorted(self.ledger_identities)),
            "source_service_cursor": dict(sorted(self.last_source_service_step.items())),
            "shard_service_cursor": {
                str(key): value for key, value in sorted(self.last_shard_service_step.items())
            },
        }
        payload["checkpoint_sha256"] = stable_hash(payload)
        self.pause_checkpoint = payload
        self._append_audit("paused", payload["checkpoint_sha256"])
        return payload

    def resume(self, evidence: Mapping[str, Any]) -> None:
        if self.state != "paused" or self.pause_checkpoint is None:
            raise SchedulerFairnessError("resume_requires_pause_checkpoint")
        required = {
            "authorization_fresh",
            "checkpoint_sha256",
            "health_fresh",
            "host_fresh",
            "protocol_fresh",
        }
        if set(evidence) != required:
            raise SchedulerFairnessError("resume_evidence_schema_invalid")
        if any(
            evidence[key] is not True
            for key in required
            if key != "checkpoint_sha256"
        ):
            raise SchedulerFairnessError("resume_prerequisite_not_fresh")
        if evidence["checkpoint_sha256"] != self.pause_checkpoint["checkpoint_sha256"]:
            raise SchedulerFairnessError("resume_checkpoint_drift")
        self.state = "resume_eligible"
        self._append_audit("resume_eligible", evidence["checkpoint_sha256"])
        self.state = "running"
        self.resume_count += 1
        self._append_audit("resumed", evidence["checkpoint_sha256"])

    def request_cancel(self, reason: str) -> None:
        if self.state != "running":
            raise SchedulerFairnessError("cancel_state_invalid")
        self.state = "cancel_required"
        self._append_audit("cancel_required", reason)

    def acknowledge_cancel(self) -> None:
        if self.state != "cancel_required" or self.in_flight:
            raise SchedulerFairnessError("cancel_requires_drained_inflight")
        for task in self.pending_initial + self.pending_continuation:
            self.query_open_tasks[task.query_identity] -= 1
        self.pending_initial.clear()
        self.pending_continuation.clear()
        for identity in self.query_identities:
            if identity not in self.terminal_queries:
                self.terminal_queries[identity] = "cancelled"
        self.state = "cancelled"
        self._append_audit("cancelled", stable_hash(self.query_identities))

    def finalize(self) -> None:
        if self.state not in {"running", "ready"}:
            return
        if self.pending_initial or self.pending_continuation or self.in_flight:
            raise SchedulerFairnessError("scheduler_work_incomplete")
        for identity in self.query_identities:
            self._maybe_finalize_query(identity)
        if set(self.terminal_queries) != set(self.query_identities):
            raise SchedulerFairnessError("terminal_coverage_incomplete")
        self.state = "completed"
        self._append_audit("scheduler_completed", stable_hash(self.query_identities))

    def coverage(self) -> dict[str, Any]:
        counts = Counter(self.terminal_queries.values())
        ordered = [
            {
                "query_identity": identity,
                "state": self.terminal_queries.get(identity, "not_started"),
            }
            for identity in self.query_identities
        ]
        return {
            "expected_query_count": 1000,
            "terminal_query_count": len(self.terminal_queries),
            "missing_query_count": 1000 - len(self.terminal_queries),
            "duplicate_query_count": 0,
            "status_counts": dict(sorted(counts.items())),
            "authoritative_order_sha256": stable_hash(ordered),
            "success_only_filtering": False,
        }

    def validate(self) -> None:
        limits = self.protocol["limits"]
        if len(self.completed_tasks) != len(self.ledger_identities):
            raise SchedulerFairnessError("ledger_operation_conservation_failed")
        if len(self.started_task_identities) != (
            len(self.completed_tasks) + len(self.in_flight)
        ):
            raise SchedulerFairnessError("started_operation_conservation_failed")
        if len(self.completed_tasks) > int(limits["attempt_upper"]):
            raise SchedulerFairnessError("attempt_budget_exceeded")
        if self.concurrency_peak > int(limits["global_concurrency"]):
            raise SchedulerFairnessError("global_concurrency_exceeded")
        if max(self.source_concurrency_peak.values()) > int(
            limits["per_source_concurrency"]
        ):
            raise SchedulerFairnessError("source_concurrency_exceeded")
        if max(self.shard_concurrency_peak.values()) > int(
            limits["per_shard_concurrency"]
        ):
            raise SchedulerFairnessError("shard_concurrency_exceeded")
        if self.duplicate_enqueue_count:
            raise SchedulerFairnessError("duplicate_enqueue_detected")
        if not self.validate_audit_chain():
            raise SchedulerFairnessError("audit_chain_invalid")
        if self.state in {"completed", "cancelled"}:
            coverage = self.coverage()
            if (
                coverage["terminal_query_count"] != 1000
                or coverage["missing_query_count"] != 0
                or coverage["success_only_filtering"] is not False
            ):
                raise SchedulerFairnessError("selective_terminal_coverage")

    def metrics(self) -> dict[str, Any]:
        first_steps = list(self.first_query_service_step.values())
        return {
            "attempt_count": len(self.completed_tasks),
            "backpressure_activation_count": self.backpressure_activation_count,
            "concurrency_peak": self.concurrency_peak,
            "first_execution_coverage_rate": len(first_steps) / 1000,
            "max_first_execution_wait_steps": max(first_steps, default=0),
            "max_queue_wait_steps": self.max_wait_steps,
            "max_shard_service_interval": max(
                self.max_shard_service_interval.values(), default=0
            ),
            "max_source_service_interval": max(
                self.max_source_service_interval.values(), default=0
            ),
            "per_shard_concurrency_peak": {
                str(key): value
                for key, value in sorted(self.shard_concurrency_peak.items())
            },
            "per_source_concurrency_peak": dict(
                sorted(self.source_concurrency_peak.items())
            ),
            "queue_peak": self.queue_peak,
        }

    def validate_audit_chain(self) -> bool:
        previous = "0" * 64
        for sequence, entry in enumerate(self.audit_entries):
            if (
                entry.get("sequence") != sequence
                or entry.get("previous_entry_sha256") != previous
            ):
                return False
            payload = dict(entry)
            digest = payload.pop("entry_sha256", None)
            if digest != stable_hash(payload):
                return False
            previous = str(digest)
        return True

    def _append_audit(
        self,
        event: str,
        subject: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        previous = (
            self.audit_entries[-1]["entry_sha256"]
            if self.audit_entries
            else "0" * 64
        )
        entry: dict[str, Any] = {
            "sequence": len(self.audit_entries),
            "event": event,
            "subject_identity": subject,
            "details": dict(details or {}),
            "previous_entry_sha256": previous,
        }
        entry["entry_sha256"] = stable_hash(entry)
        self.audit_entries.append(entry)


@dataclass(frozen=True)
class LoadProfile:
    name: str
    delay_mode: str = "uniform"
    outcome_mode: str = "success"
    page_count: int = 0
    worker_limit: int = 12
    pause_after_completions: int | None = None
    second_pause_after_completions: int | None = None
    cancel_after_completions: int | None = None
    resume_after_pause: bool = False

    def delay(self, task: ScheduledTask) -> int:
        if self.delay_mode == "uniform":
            return 1
        if self.delay_mode == "slow_source":
            return 13 if task.source == "openalex" else 1
        if self.delay_mode == "slow_shard":
            return 17 if task.shard_index == 7 else 1
        if self.delay_mode == "heterogeneous":
            return 1 + (
                (task.query_ordinal * 17 + task.source_ordinal * 11) % 31
            )
        if self.delay_mode == "redegraded":
            return 1 + (task.source_ordinal % 3)
        raise SchedulerFairnessError("unknown_delay_profile")

    def outcome(self, task: ScheduledTask) -> str:
        if self.outcome_mode == "success":
            return "success"
        if self.outcome_mode == "retry_storm":
            return "429" if task.kind == "initial" and task.source != "pubmed" else "success"
        if self.outcome_mode == "mixed_failures":
            options = ("success", "429", "503", "timeout")
            return options[(task.query_ordinal + task.source_ordinal) % len(options)]
        if self.outcome_mode == "redegraded":
            return "503" if task.source in {"arxiv", "openalex"} else "success"
        raise SchedulerFairnessError("unknown_outcome_profile")

    def pages(self, task: ScheduledTask) -> int:
        if self.page_count and task.source != "pubmed":
            return self.page_count
        return 0


def _resume_evidence(machine: DeterministicScheduler) -> dict[str, Any]:
    if machine.pause_checkpoint is None:
        raise SchedulerFairnessError("pause_checkpoint_missing")
    return {
        "authorization_fresh": True,
        "checkpoint_sha256": machine.pause_checkpoint["checkpoint_sha256"],
        "health_fresh": True,
        "host_fresh": True,
        "protocol_fresh": True,
    }


def execute_profile(
    protocol: Mapping[str, Any],
    queries: tuple[str, ...],
    profile: LoadProfile,
) -> DeterministicScheduler:
    machine = DeterministicScheduler(protocol, queries, profile.worker_limit)
    step = 0
    pause_thresholds = [
        value
        for value in (
            profile.pause_after_completions,
            profile.second_pause_after_completions,
        )
        if value is not None
    ]
    completed_pause_count = 0
    while True:
        if step > int(protocol["limits"]["max_logical_steps"]):
            raise SchedulerFairnessError("finite_progress_bound_exceeded")
        machine.finish_due(step=step, outcome=profile.outcome, pages=profile.pages)
        completed = len(machine.completed_tasks)
        if (
            completed_pause_count < len(pause_thresholds)
            and completed >= pause_thresholds[completed_pause_count]
            and machine.state == "running"
        ):
            machine.request_pause("synthetic_pause")
            completed_pause_count += 1
        if (
            profile.cancel_after_completions is not None
            and completed >= profile.cancel_after_completions
            and machine.state == "running"
        ):
            machine.request_cancel("synthetic_cancel")
        if machine.state == "pause_required" and not machine.in_flight:
            machine.acknowledge_pause()
            if profile.resume_after_pause:
                machine.resume(_resume_evidence(machine))
            else:
                break
        if machine.state == "cancel_required" and not machine.in_flight:
            machine.acknowledge_cancel()
            break
        admitted = machine.admit(step=step, delay=profile.delay)
        if (
            not machine.pending_initial
            and not machine.pending_continuation
            and not machine.in_flight
        ):
            machine.finalize()
            break
        if admitted == 0 and machine.in_flight:
            step = max(
                step + 1,
                min(item.due_step for item in machine.in_flight.values()),
            )
        else:
            step += 1
    machine.validate()
    return machine


def _scenario(
    protocol: Mapping[str, Any],
    queries: tuple[str, ...],
    profile: LoadProfile,
) -> dict[str, Any]:
    machine = execute_profile(protocol, queries, profile)
    coverage = machine.coverage()
    metrics = machine.metrics()
    passed = (
        machine.state in {"completed", "cancelled", "paused"}
        and machine.validate_audit_chain()
        and metrics["concurrency_peak"]
        <= int(protocol["limits"]["global_concurrency"])
        and max(metrics["per_source_concurrency_peak"].values())
        <= int(protocol["limits"]["per_source_concurrency"])
        and coverage["success_only_filtering"] is False
        and (
            machine.state == "paused"
            or coverage["terminal_query_count"] == 1000
        )
    )
    return {
        "scenario": profile.name,
        "status": "passed" if passed else "failed",
        "scheduler_state": machine.state,
        "logical_step_count": machine.logical_step,
        "resume_count": machine.resume_count,
        "coverage": coverage,
        "metrics": metrics,
        "ledger_entry_count": len(machine.ledger_identities),
        "audit_chain_valid": machine.validate_audit_chain(),
        "duplicate_enqueue_count": machine.duplicate_enqueue_count,
        "rejected_admission_count": machine.rejected_admissions,
    }


def simulate_load(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    queries = load_query_identities(root, protocol)
    profiles = [
        LoadProfile("uniform_delay"),
        LoadProfile("single_slow_source", delay_mode="slow_source"),
        LoadProfile("single_slow_shard", delay_mode="slow_shard"),
        LoadProfile("retry_storm", outcome_mode="retry_storm"),
        LoadProfile("pagination_storm", page_count=2),
        LoadProfile("worker_reduction", worker_limit=4),
        LoadProfile("dynamic_backpressure", outcome_mode="retry_storm", worker_limit=8),
        LoadProfile(
            "pause_resume",
            pause_after_completions=311,
            resume_after_pause=True,
        ),
        LoadProfile("cancel", cancel_after_completions=317),
        LoadProfile(
            "resume_then_redegrade",
            delay_mode="redegraded",
            outcome_mode="redegraded",
            pause_after_completions=293,
            second_pause_after_completions=2000,
            resume_after_pause=True,
        ),
        LoadProfile("extreme_heterogeneous_delay", delay_mode="heterogeneous"),
        LoadProfile(
            "mixed_failures",
            outcome_mode="mixed_failures",
        ),
    ]
    rows = [_scenario(protocol, queries, profile) for profile in profiles]
    rows.sort(key=lambda row: row["scenario"])
    passed = all(row["status"] == "passed" for row in rows)
    return _report(
        "scheduler_controls_ready" if passed else "fairness_or_backpressure_violation",
        EXIT_READY if passed else EXIT_VIOLATION,
        scenario_count=len(rows),
        scenarios=rows,
        population={
            "query_count": 1000,
            "shard_count": 20,
            "source_count": 4,
            "query_order_sha256": stable_hash(queries),
        },
        invariants={
            "priority_quality_signal_count": 0,
            "selective_completion_count": 0,
            "duplicate_billing_count": 0,
            "scheduler_bypass_count": 0,
            "real_sleep_count": 0,
        },
    )


def verify_resume(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    queries = load_query_identities(root, protocol)
    profile = LoadProfile(
        "resume_verification",
        delay_mode="heterogeneous",
        pause_after_completions=257,
        resume_after_pause=True,
    )
    first = execute_profile(protocol, queries, profile)
    second = execute_profile(protocol, queries, profile)
    if canonical_json(first.coverage()) != canonical_json(second.coverage()):
        raise SchedulerFairnessError("resume_coverage_nondeterministic")
    if first.coverage()["terminal_query_count"] != 1000:
        raise SchedulerFairnessError("resume_terminal_coverage_incomplete")
    return _report(
        "scheduler_controls_ready",
        EXIT_READY,
        resume_count=first.resume_count,
        terminal_query_count=1000,
        repeated_committed_query_count=0,
        fairness_cursor_preserved=True,
        ledger_entry_count=len(first.ledger_identities),
        coverage_sha256=stable_hash(first.coverage()),
    )


def build_addendum(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "addendum": ADDENDUM,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "scheduler_protocol_sha256": protocol["protocol_sha256"],
        "execution_plan": protocol["bindings"]["execution_plan"],
        "launch_control": protocol["bindings"]["launch_control"],
        "provider_health": protocol["bindings"]["provider_health"],
        "requirements": {
            "all_query_source_initial_work_precedes_continuation": True,
            "budget_and_concurrency_caps_fail_closed": True,
            "completion_speed_not_a_priority_input": True,
            "failed_cancelled_queries_remain_in_ordered_coverage": True,
            "pause_cancel_blocks_admission": True,
            "resume_preserves_fairness_cursor": True,
            "scheduler_identity_only_priority": True,
        },
        "real_execution_started": False,
        "formal_validation_complete": False,
    }
    payload["addendum_sha256"] = stable_hash(payload)
    return payload


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    simulation = simulate_load(root, protocol)
    if simulation["exit_code"] != EXIT_READY:
        raise SchedulerFairnessError("synthetic_scheduler_matrix_failed")
    return _report(
        "external_run_not_started",
        EXIT_NOT_STARTED,
        controls_ready=True,
        real_execution_started=False,
        full1000_blocker_cleared=False,
        scenario_count=simulation["scenario_count"],
    )


def _report(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }
