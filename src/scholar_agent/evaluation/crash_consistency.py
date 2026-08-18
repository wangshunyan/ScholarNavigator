"""Crash-consistent persistence for Benchmark run state and its offline gate.

The production Benchmark runner uses :class:`BenchmarkRunCommitStore` as the
authoritative checkpoint.  Each generation is assembled in a same-directory
pending directory, every file is flushed, the directory is atomically renamed,
and a durable ``COMMITTED`` marker is installed last.  Top-level Benchmark
files remain compatibility mirrors; resume never trusts them once a committed
generation exists.

The gate in this module drives that same store through deterministic fault
points.  It does not use gold, runtime configuration, network access, sleeps,
or process termination.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import socket
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, get_args
from unittest.mock import patch

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_WINDOWS_LOCK_CONFLICT_ERRNOS = frozenset(
    value
    for value in (
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", None),
        getattr(errno, "EWOULDBLOCK", None),
    )
    if value is not None
)


CONTRACT_VERSION = "crash_consistency_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "crash_consistency_gate"
STORE_DIRECTORY = ".run_commits"
EXIT_PASSED = 0
EXIT_INVARIANT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4

RunState = Literal["running", "completed", "failed"]
FaultPoint = Literal[
    "before_stage_create",
    "after_stage_create",
    "mid_write",
    "flush_failure",
    "fsync_failure",
    "checkpoint_committed_manifest_missing",
    "manifest_written_completion_missing",
    "before_replace",
    "after_replace",
    "disk_full",
    "permission_denied",
    "after_commit_marker",
]
FAULT_POINTS = tuple(get_args(FaultPoint))

_GENERATION_PATTERN = re.compile(r"^generation-(\d{8})$")
_PENDING_PATTERN = re.compile(r"^\.generation-(\d{8})\.pending-[A-Za-z0-9_-]+$")
_ALLOWED_PUBLIC_ARTIFACTS = frozenset(
    {
        "config.json",
        "dataset_report.json",
        "results.jsonl",
        "failures.jsonl",
        "metrics.json",
        "stage_metrics.json",
        "error_analysis.json",
        "gold_diagnostics.jsonl",
        "result_lineage.jsonl",
        "resource_ledger.json",
        "provider_ingest_provenance.json",
        "provider_ingest_raw.tar",
        "summary.md",
    }
)
_INTERNAL_FILES = frozenset(
    {
        "delta.json",
        "checkpoint.json",
        "events.jsonl",
        "run_manifest.json",
        "generation_manifest.json",
        "RUN_COMPLETED",
        "COMMITTED",
    }
)
_SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")


class CrashConsistencyError(RuntimeError):
    """The run state or protocol cannot be validated safely."""


class CrashNotEligible(CrashConsistencyError):
    """The requested frozen artifacts lack atomic-generation evidence."""


class ConcurrentWriterError(CrashConsistencyError):
    """Another writer owns the run-directory commit lock."""


class InjectedCrash(CrashConsistencyError):
    """Deterministic interruption used by the offline gate."""


@dataclass(frozen=True)
class FaultInjector:
    """A deterministic filesystem/writer seam; it triggers exactly once."""

    point: FaultPoint | None = None

    def hit(self, point: FaultPoint) -> None:
        if self.point != point:
            return
        if point == "disk_full":
            raise OSError(errno.ENOSPC, "injected_disk_full")
        if point == "permission_denied":
            raise PermissionError(errno.EACCES, "injected_permission_denied")
        if point == "flush_failure":
            raise OSError(errno.EIO, "injected_flush_failure")
        if point == "fsync_failure":
            raise OSError(errno.EIO, "injected_fsync_failure")
        raise InjectedCrash(f"injected:{point}")


@dataclass(frozen=True)
class CommittedRunState:
    run_id: str
    generation: int
    status: RunState
    expected_query_ids: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    dataset_report: dict[str, Any]
    reports: dict[str, bytes]
    event_count: int
    generation_path: Path

    @property
    def rows_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["case_id"]): dict(row) for row in self.records}


def stable_json_bytes(value: Any, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_error(value: object) -> str:
    """Return a stable error class/message without secrets or machine paths."""

    text = _SECRET_PATTERN.sub(r"\1\2[redacted]", str(value))
    text = re.sub(r"(?i)\.env(?:\.[A-Za-z0-9_-]+)?", "[environment-file]", text)
    text = _ABSOLUTE_PATH_PATTERN.sub("[absolute-path]", text)
    return text[:300]


def durable_atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    injector: FaultInjector | None = None,
    temporary_suffix: str = "writer",
) -> None:
    """Durably replace one file without exposing a partially written value."""

    fault = injector or FaultInjector()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{temporary_suffix}.tmp")
    try:
        fault.hit("disk_full")
        fault.hit("permission_denied")
        with temporary.open("xb") as handle:
            if fault.point == "mid_write":
                midpoint = max(1, len(content) // 2)
                handle.write(content[:midpoint])
                handle.flush()
                raise InjectedCrash("injected:mid_write")
            handle.write(content)
            fault.hit("flush_failure")
            handle.flush()
            fault.hit("fsync_failure")
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def durable_atomic_write_text(path: Path, text: str) -> None:
    durable_atomic_write_bytes(path, text.encode("utf-8"))


class BenchmarkRunCommitStore:
    """Append-only, generation-based authority for Benchmark persistence."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.root = run_directory / STORE_DIRECTORY
        self.generations = self.root / "generations"
        self.lock_path = self.root / "writer.lock"
        self._cached_state: CommittedRunState | None = None

    @property
    def has_commits(self) -> bool:
        return any(self._generation_directories())

    def initialize(
        self,
        *,
        run_id: str,
        expected_query_ids: Sequence[str],
        config: Mapping[str, Any],
        dataset_report: Mapping[str, Any],
        comparison_binding: Mapping[str, Any] | None = None,
        shard_binding: Mapping[str, Any] | None = None,
        injector: FaultInjector | None = None,
    ) -> CommittedRunState:
        if self.has_commits:
            raise CrashConsistencyError("run_store_already_initialized")
        expected = _validated_expected_ids(expected_query_ids)
        normalized_config = dict(config)
        if comparison_binding is not None:
            if "comparison" in normalized_config:
                raise CrashConsistencyError("comparison_binding_duplicated")
            normalized_config["comparison"] = _validated_comparison_binding(
                comparison_binding
            )
        if shard_binding is not None:
            if "shard" in normalized_config:
                raise CrashConsistencyError("shard_binding_duplicated")
            normalized_config["shard"] = _validated_shard_binding(shard_binding)
        delta = {
            "kind": "initialize",
            "config": normalized_config,
            "dataset_report": dict(dataset_report),
        }
        return self._commit(
            run_id=run_id,
            expected_query_ids=expected,
            delta=delta,
            status="running",
            reports={},
            injector=injector,
        )

    def commit_record(
        self,
        record: Mapping[str, Any],
        *,
        injector: FaultInjector | None = None,
    ) -> CommittedRunState:
        state = self._current_state()
        case_id = str(record.get("case_id") or "").strip()
        if case_id not in state.expected_query_ids:
            raise CrashConsistencyError("record_query_identity_unknown")
        delta = {"kind": "record", "record": dict(record)}
        return self._commit(
            run_id=state.run_id,
            expected_query_ids=state.expected_query_ids,
            delta=delta,
            status="running",
            reports={},
            injector=injector,
        )

    def commit_completion(
        self,
        reports: Mapping[str, bytes],
        *,
        injector: FaultInjector | None = None,
    ) -> CommittedRunState:
        state = self._current_state()
        _validate_report_names(reports)
        if len(state.records) != len(state.expected_query_ids):
            raise CrashConsistencyError("completion_record_count_incomplete")
        normalized_reports = {name: bytes(value) for name, value in reports.items()}
        if state.status == "completed" and state.reports == normalized_reports:
            return state
        return self._commit(
            run_id=state.run_id,
            expected_query_ids=state.expected_query_ids,
            delta={"kind": "complete"},
            status="completed",
            reports=normalized_reports,
            injector=injector,
        )

    def load_latest(self) -> CommittedRunState:
        candidates = sorted(self._generation_directories(), reverse=True)
        for _generation, path in candidates:
            try:
                state = self._load_chain(path)
                self._cached_state = state
                return state
            except (CrashConsistencyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise CrashConsistencyError("no_valid_committed_generation")

    def _current_state(self) -> CommittedRunState:
        return self._cached_state or self.load_latest()

    def cleanup_uncommitted_temporaries(self) -> list[str]:
        """Remove only pending directories proven outside the committed namespace."""

        removed: list[str] = []
        if not self.generations.is_dir():
            return removed
        with self.writer_lock():
            for path in sorted(self.generations.iterdir(), key=lambda item: item.name):
                if path.is_dir() and _PENDING_PATTERN.fullmatch(path.name):
                    shutil.rmtree(path)
                    removed.append(path.name)
            _fsync_directory(self.generations)
        return removed

    def materialize_compatibility_view(
        self, state: CommittedRunState | None = None
    ) -> None:
        """Update legacy top-level files from one already committed generation."""

        value = state or self.load_latest()
        public = self.public_artifacts(value)
        for name, content in sorted(public.items()):
            durable_atomic_write_bytes(
                self.run_directory / name,
                content,
                temporary_suffix="committed-view",
            )

    def public_artifacts(self, state: CommittedRunState) -> dict[str, bytes]:
        rows = [dict(row) for row in state.records]
        results = b"".join(stable_json_bytes(row, indent=None) for row in rows)
        failures = [
            {
                "case_id": row["case_id"],
                "query": row.get("query", ""),
                "status": row.get("status", "failed"),
                "error_type": row.get("error_type") or "Unknown",
                "error_message": row.get("error") or "",
            }
            for row in rows
            if row.get("status") != "succeeded"
        ]
        failure_bytes = b"".join(
            stable_json_bytes(row, indent=None) for row in failures
        )
        return {
            "config.json": stable_json_bytes(state.config),
            "dataset_report.json": stable_json_bytes(state.dataset_report),
            "results.jsonl": results,
            "failures.jsonl": failure_bytes,
            **state.reports,
        }

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            _lock_file_descriptor(descriptor)
            acquired = True
            _fsync_directory(self.root)
            yield
        finally:
            try:
                if acquired:
                    _unlock_file_descriptor(descriptor)
            finally:
                os.close(descriptor)

    def _commit(
        self,
        *,
        run_id: str,
        expected_query_ids: Sequence[str],
        delta: Mapping[str, Any],
        status: RunState,
        reports: Mapping[str, bytes],
        injector: FaultInjector | None,
    ) -> CommittedRunState:
        fault = injector or FaultInjector()
        self.generations.mkdir(parents=True, exist_ok=True)
        with self.writer_lock():
            previous: CommittedRunState | None
            if self._cached_state is not None:
                previous = self._cached_state
            else:
                try:
                    previous = self.load_latest()
                except CrashConsistencyError:
                    previous = None
            generation = self._next_generation()
            if previous is not None:
                if previous.run_id != run_id:
                    raise CrashConsistencyError("run_identity_drift")
                if tuple(expected_query_ids) != previous.expected_query_ids:
                    raise CrashConsistencyError("expected_query_identity_drift")
            state_rows = previous.rows_by_id if previous is not None else {}
            if delta.get("kind") == "record":
                row = delta.get("record")
                if not isinstance(row, dict):
                    raise CrashConsistencyError("record_delta_invalid")
                state_rows[str(row["case_id"])] = dict(row)
            ordered_records = tuple(
                state_rows[item]
                for item in expected_query_ids
                if item in state_rows
            )
            prior_event_count = previous.event_count if previous is not None else 0
            events = _events_for_delta(
                delta,
                generation=generation,
                sequence_start=prior_event_count,
                record_count=len(ordered_records),
            )
            event_count = prior_event_count + len(events)
            config = (
                dict(delta["config"])
                if delta.get("kind") == "initialize"
                else dict(previous.config if previous is not None else {})
            )
            dataset_report = (
                dict(delta["dataset_report"])
                if delta.get("kind") == "initialize"
                else dict(previous.dataset_report if previous is not None else {})
            )
            combined_reports = (
                {name: bytes(content) for name, content in reports.items()}
                if status == "completed"
                else {}
            )
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "generation": generation,
                "parent_generation": previous.generation if previous else None,
                "cursor": len(ordered_records),
                "record_count": len(ordered_records),
                "record_identities_sha256": _ordered_hash(
                    [str(row["case_id"]) for row in ordered_records]
                ),
                "event_count": event_count,
                "status": status,
            }
            run_manifest = {
                "schema_version": SCHEMA_VERSION,
                "manifest_kind": "benchmark_run_commit_v1",
                "run_id": run_id,
                "generation": generation,
                "parent_generation": previous.generation if previous else None,
                "status": status,
                "expected_count": len(expected_query_ids),
                "completed_count": len(ordered_records),
                "expected_query_identities_sha256": _ordered_hash(expected_query_ids),
                "record_identities_sha256": checkpoint["record_identities_sha256"],
                "event_count": event_count,
                "config_sha256": sha256_bytes(stable_json_bytes(config)),
                "reports": sorted(combined_reports),
                "score_scope": "persistence_only_not_quality_or_official_score",
            }
            fault.hit("before_stage_create")
            stage = self.generations / (
                f".generation-{generation:08d}.pending-writer"
            )
            final = self.generations / f"generation-{generation:08d}"
            if stage.exists() or final.exists():
                raise CrashConsistencyError("generation_path_collision")
            stage.mkdir()
            fault.hit("after_stage_create")
            try:
                write_fault = fault if fault.point in {
                    "mid_write",
                    "flush_failure",
                    "fsync_failure",
                    "disk_full",
                    "permission_denied",
                } else FaultInjector()
                self._write_generation_file(
                    stage / "delta.json", stable_json_bytes(dict(delta)), write_fault
                )
                self._write_generation_file(
                    stage / "checkpoint.json", stable_json_bytes(checkpoint), FaultInjector()
                )
                fault.hit("checkpoint_committed_manifest_missing")
                self._write_generation_file(
                    stage / "events.jsonl",
                    b"".join(stable_json_bytes(item, indent=None) for item in events),
                    FaultInjector(),
                )
                self._write_generation_file(
                    stage / "run_manifest.json",
                    stable_json_bytes(run_manifest),
                    FaultInjector(),
                )
                if delta.get("kind") == "initialize":
                    self._write_generation_file(
                        stage / "config.json", stable_json_bytes(config), FaultInjector()
                    )
                    self._write_generation_file(
                        stage / "dataset_report.json",
                        stable_json_bytes(dataset_report),
                        FaultInjector(),
                    )
                for name, content in sorted(combined_reports.items()):
                    self._write_generation_file(stage / name, content, FaultInjector())
                identities = [
                    _file_identity(path, stage)
                    for path in sorted(stage.iterdir(), key=lambda item: item.name)
                    if path.is_file()
                ]
                generation_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "contract": CONTRACT_VERSION,
                    "generation": generation,
                    "parent_generation": previous.generation if previous else None,
                    "status": status,
                    "record_count": len(ordered_records),
                    "event_count": event_count,
                    "checkpoint_cursor": len(ordered_records),
                    "files": identities,
                }
                self._write_generation_file(
                    stage / "generation_manifest.json",
                    stable_json_bytes(generation_manifest),
                    FaultInjector(),
                )
                fault.hit("manifest_written_completion_missing")
                if status == "completed":
                    self._write_generation_file(
                        stage / "RUN_COMPLETED",
                        stable_json_bytes(
                            {
                                "generation": generation,
                                "record_count": len(ordered_records),
                                "generation_manifest_sha256": sha256_file(
                                    stage / "generation_manifest.json"
                                ),
                            },
                            indent=None,
                        ),
                        FaultInjector(),
                    )
                _fsync_directory(stage)
                fault.hit("before_replace")
                os.replace(stage, final)
                _fsync_directory(self.generations)
                fault.hit("after_replace")
                committed = {
                    "generation": generation,
                    "generation_manifest_sha256": sha256_file(
                        final / "generation_manifest.json"
                    ),
                }
                durable_atomic_write_bytes(
                    final / "COMMITTED",
                    stable_json_bytes(committed, indent=None),
                    temporary_suffix="commit-marker",
                )
                _fsync_directory(final)
                fault.hit("after_commit_marker")
            except BaseException:
                # A renamed directory without COMMITTED is deliberately retained:
                # readers ignore it and the gate can audit that boundary.
                if stage.exists() and fault.point not in {
                    "after_stage_create",
                    "mid_write",
                    "flush_failure",
                    "fsync_failure",
                    "disk_full",
                    "permission_denied",
                    "checkpoint_committed_manifest_missing",
                    "manifest_written_completion_missing",
                    "before_replace",
                }:
                    shutil.rmtree(stage)
                raise
        state = CommittedRunState(
            run_id=run_id,
            generation=generation,
            status=status,
            expected_query_ids=tuple(expected_query_ids),
            records=ordered_records,
            config=config,
            dataset_report=dataset_report,
            reports=combined_reports,
            event_count=event_count,
            generation_path=final,
        )
        self._cached_state = state
        return state

    @staticmethod
    def _write_generation_file(
        path: Path, content: bytes, injector: FaultInjector
    ) -> None:
        durable_atomic_write_bytes(
            path,
            content,
            injector=injector,
            temporary_suffix="generation",
        )

    def _generation_directories(self) -> list[tuple[int, Path]]:
        if not self.generations.is_dir():
            return []
        result: list[tuple[int, Path]] = []
        for path in self.generations.iterdir():
            match = _GENERATION_PATTERN.fullmatch(path.name)
            if match and path.is_dir() and (path / "COMMITTED").is_file():
                result.append((int(match.group(1)), path))
        return result

    def _load_chain(self, latest_path: Path) -> CommittedRunState:
        latest_manifest = self._validate_generation(latest_path)
        generation = int(latest_manifest["generation"])
        chain: list[tuple[Path, dict[str, Any]]] = []
        observed: set[int] = set()
        current_path = latest_path
        while True:
            manifest = self._validate_generation(current_path)
            current_generation = int(manifest["generation"])
            if current_generation in observed:
                raise CrashConsistencyError("generation_lineage_cycle")
            observed.add(current_generation)
            chain.append((current_path, manifest))
            parent = manifest.get("parent_generation")
            if parent is None:
                break
            if int(parent) >= current_generation:
                raise CrashConsistencyError("generation_parent_invalid")
            current_path = self.generations / f"generation-{int(parent):08d}"
            if not (current_path / "COMMITTED").is_file():
                raise CrashConsistencyError("generation_parent_missing")
        chain.reverse()
        run_id = ""
        expected_ids: tuple[str, ...] = ()
        config: dict[str, Any] = {}
        dataset_report: dict[str, Any] = {}
        rows: dict[str, dict[str, Any]] = {}
        reports: dict[str, bytes] = {}
        cumulative_events = 0
        status: RunState = "running"
        previous_generation: int | None = None
        for path, manifest in chain:
            current_generation = int(manifest["generation"])
            delta = _load_json_object(path / "delta.json")
            run_manifest = _load_json_object(path / "run_manifest.json")
            checkpoint = _load_json_object(path / "checkpoint.json")
            if manifest.get("parent_generation") != previous_generation:
                raise CrashConsistencyError("generation_parent_mixed")
            for name, value in (
                ("checkpoint", checkpoint),
                ("run_manifest", run_manifest),
            ):
                if value.get("generation") != current_generation:
                    raise CrashConsistencyError(f"{name}_generation_mismatch")
                if value.get("parent_generation") != previous_generation:
                    raise CrashConsistencyError(f"{name}_parent_mismatch")
            if not run_id:
                if delta.get("kind") != "initialize":
                    raise CrashConsistencyError("first_generation_not_initialize")
                run_id = str(run_manifest.get("run_id") or "")
                config = _load_json_object(path / "config.json")
                dataset_report = _load_json_object(path / "dataset_report.json")
                case_ids = config.get("case_ids")
                if not isinstance(case_ids, list):
                    raise CrashConsistencyError("expected_query_identities_missing")
                expected_ids = _validated_expected_ids([str(item) for item in case_ids])
            elif run_manifest.get("run_id") != run_id:
                raise CrashConsistencyError("run_identity_mixed_generation")
            if run_manifest.get("expected_query_identities_sha256") != _ordered_hash(
                expected_ids
            ):
                raise CrashConsistencyError("expected_query_identity_hash_mismatch")
            if delta.get("kind") == "record":
                record = delta.get("record")
                if not isinstance(record, dict):
                    raise CrashConsistencyError("record_delta_invalid")
                case_id = str(record.get("case_id") or "")
                if case_id not in expected_ids:
                    raise CrashConsistencyError("record_query_identity_unknown")
                rows[case_id] = record
            events = _load_jsonl(path / "events.jsonl")
            for offset, event in enumerate(events, start=1):
                if event.get("sequence") != cumulative_events + offset:
                    raise CrashConsistencyError("event_sequence_mismatch")
                if event.get("generation") != current_generation:
                    raise CrashConsistencyError("event_generation_mismatch")
            cumulative_events += len(events)
            ordered = [rows[item] for item in expected_ids if item in rows]
            if int(checkpoint.get("record_count", -1)) != len(ordered):
                raise CrashConsistencyError("checkpoint_record_count_mismatch")
            if int(checkpoint.get("cursor", -1)) != len(ordered):
                raise CrashConsistencyError("checkpoint_cursor_mismatch")
            if checkpoint.get("record_identities_sha256") != _ordered_hash(
                [str(row["case_id"]) for row in ordered]
            ):
                raise CrashConsistencyError("checkpoint_record_identity_mismatch")
            if int(checkpoint.get("event_count", -1)) != cumulative_events:
                raise CrashConsistencyError("checkpoint_event_count_mismatch")
            if int(run_manifest.get("completed_count", -1)) != len(ordered):
                raise CrashConsistencyError("run_manifest_record_count_mismatch")
            if int(run_manifest.get("event_count", -1)) != cumulative_events:
                raise CrashConsistencyError("run_manifest_event_count_mismatch")
            if run_manifest.get("config_sha256") != sha256_bytes(
                stable_json_bytes(config)
            ):
                raise CrashConsistencyError("run_manifest_config_hash_mismatch")
            status = str(run_manifest.get("status"))  # type: ignore[assignment]
            if status not in {"running", "completed", "failed"}:
                raise CrashConsistencyError("run_status_invalid")
            if checkpoint.get("status") != status or manifest.get("status") != status:
                raise CrashConsistencyError("generation_status_mixed")
            if int(manifest.get("checkpoint_cursor", -1)) != len(ordered):
                raise CrashConsistencyError("generation_checkpoint_cursor_mismatch")
            if status == "completed":
                if len(ordered) != len(expected_ids):
                    raise CrashConsistencyError("completed_record_count_incomplete")
                if not (path / "RUN_COMPLETED").is_file():
                    raise CrashConsistencyError("completed_marker_missing")
                completed_marker = _load_json_object(path / "RUN_COMPLETED")
                if (
                    completed_marker.get("generation") != current_generation
                    or completed_marker.get("record_count") != len(ordered)
                    or completed_marker.get("generation_manifest_sha256")
                    != sha256_file(path / "generation_manifest.json")
                    or not events
                    or events[-1].get("event") != "run_completed"
                ):
                    raise CrashConsistencyError("completed_marker_mismatch")
                reports = {
                    name: (path / name).read_bytes()
                    for name in run_manifest.get("reports") or []
                }
            elif (path / "RUN_COMPLETED").exists():
                raise CrashConsistencyError("premature_completed_marker")
            else:
                reports = {}
            if int(manifest.get("record_count", -1)) != len(ordered):
                raise CrashConsistencyError("generation_record_count_mismatch")
            if int(manifest.get("event_count", -1)) != cumulative_events:
                raise CrashConsistencyError("generation_event_count_mismatch")
            previous_generation = current_generation

        ordered_records = tuple(rows[item] for item in expected_ids if item in rows)
        return CommittedRunState(
            run_id=run_id,
            generation=generation,
            status=status,
            expected_query_ids=expected_ids,
            records=ordered_records,
            config=config,
            dataset_report=dataset_report,
            reports=reports,
            event_count=cumulative_events,
            generation_path=latest_path,
        )

    def _validate_generation(self, path: Path) -> dict[str, Any]:
        match = _GENERATION_PATTERN.fullmatch(path.name)
        if not match:
            raise CrashConsistencyError("generation_path_invalid")
        committed = _load_json_object(path / "COMMITTED")
        manifest_path = path / "generation_manifest.json"
        if sha256_file(manifest_path) != committed.get("generation_manifest_sha256"):
            raise CrashConsistencyError("commit_marker_manifest_hash_mismatch")
        manifest = _load_json_object(manifest_path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != CONTRACT_VERSION
            or int(manifest.get("generation", -1)) != int(match.group(1))
            or committed.get("generation") != int(match.group(1))
        ):
            raise CrashConsistencyError("generation_manifest_identity_mismatch")
        listed = manifest.get("files")
        if not isinstance(listed, list):
            raise CrashConsistencyError("generation_file_inventory_missing")
        expected_names: set[str] = {"generation_manifest.json", "COMMITTED"}
        for item in listed:
            if not isinstance(item, dict):
                raise CrashConsistencyError("generation_file_identity_invalid")
            name = str(item.get("path") or "")
            _validate_relative_name(name)
            file_path = path / name
            expected_names.add(name)
            if not file_path.is_file():
                raise CrashConsistencyError("generation_file_missing")
            if file_path.stat().st_size != item.get("size_bytes"):
                raise CrashConsistencyError("generation_file_size_mismatch")
            if sha256_file(file_path) != item.get("sha256"):
                raise CrashConsistencyError("generation_file_hash_mismatch")
        if (path / "RUN_COMPLETED").is_file():
            expected_names.add("RUN_COMPLETED")
        actual_names = {item.name for item in path.iterdir() if item.is_file()}
        if actual_names != expected_names:
            raise CrashConsistencyError("generation_file_inventory_mismatch")
        return manifest

    def _next_generation(self) -> int:
        """Never reuse a number left by a damaged or interrupted generation."""

        highest = 0
        if self.generations.is_dir():
            for path in self.generations.iterdir():
                committed = _GENERATION_PATTERN.fullmatch(path.name)
                pending = _PENDING_PATTERN.fullmatch(path.name)
                match = committed or pending
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrashConsistencyError("protocol_unreadable") from exc
    if not isinstance(protocol, dict):
        raise CrashConsistencyError("protocol_root_invalid")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("contract") != CONTRACT_VERSION
    ):
        raise CrashConsistencyError("protocol_version_incompatible")
    if protocol.get("score_scope") != "persistence_only_not_quality_or_official_score":
        raise CrashConsistencyError("protocol_score_scope_invalid")
    scenarios = protocol.get("fault_scenarios")
    if not isinstance(scenarios, list) or tuple(scenarios) != FAULT_POINTS:
        raise CrashConsistencyError("fault_matrix_incomplete")
    frozen = protocol.get("frozen_baseline_eligibility")
    if not isinstance(frozen, dict):
        raise CrashConsistencyError("frozen_eligibility_missing")
    relative = _repo_path(repository_root, str(frozen.get("legacy_audit_path") or ""))
    if not relative.is_file() or sha256_file(relative) != frozen.get("sha256"):
        raise CrashNotEligible("frozen_legacy_audit_identity_drift")
    return protocol


def run_crash_consistency(
    protocol: Mapping[str, Any],
    *,
    work_root: Path,
    controlled_fault: Literal["non_atomic_writer"] | None = None,
) -> dict[str, Any]:
    """Execute the deterministic crash matrix through the production store."""

    attempts = {"network": 0}
    violations: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    with _forbid_network(attempts):
        matrix_root = work_root / "matrix"
        matrix_root.mkdir(parents=True, exist_ok=True)
        for index, point in enumerate(protocol["fault_scenarios"], start=1):
            scenario = _run_fault_scenario(
                matrix_root / f"scenario-{index:02d}", str(point)
            )
            scenario_rows.append(scenario)
            violations.extend(scenario["violations"])
        normal = _run_normal_scenario(matrix_root / "normal")
        violations.extend(normal["violations"])
        corrupt = _run_corrupt_latest_scenario(matrix_root / "corrupt-latest")
        violations.extend(corrupt["violations"])
        stale = _run_stale_temp_scenario(matrix_root / "stale-temp")
        violations.extend(stale["violations"])
        concurrent = _run_concurrent_writer_scenario(matrix_root / "concurrent")
        violations.extend(concurrent["violations"])
        if controlled_fault == "non_atomic_writer":
            non_atomic = _run_non_atomic_fault(matrix_root / "non-atomic")
            violations.extend(non_atomic["violations"])
        else:
            non_atomic = {"status": "not_injected", "violations": []}

    violations = sorted(
        violations,
        key=lambda item: (
            str(item.get("fault_point")),
            str(item.get("invariant")),
            str(item.get("first_violation_path")),
        ),
    )
    status = "passed" if not violations else "invariant_violation"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if not violations else EXIT_INVARIANT_VIOLATION,
        "score_scope": "persistence_only_not_quality_or_official_score",
        "fault_scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
        "normal_multi_generation": normal,
        "corrupt_latest_recovery": corrupt,
        "temporary_cleanup": stale,
        "concurrent_writer": concurrent,
        "controlled_non_atomic_writer": non_atomic,
        "violation_count": len(violations),
        "violations": violations,
        "execution": {
            "network_request_count": attempts["network"],
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "real_process_kill_count": 0,
            "real_disk_fill_count": 0,
            "sleep_race_count": 0,
            "controlled_fault": controlled_fault,
        },
    }


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    frozen = protocol["frozen_baseline_eligibility"]
    audit_path = _repo_path(repository_root, str(frozen["legacy_audit_path"]))
    audit = _load_json_object(audit_path)
    rows = []
    for profile in audit.get("profiles") or []:
        missing = sorted(
            set(profile.get("missing_run_manifest_v1_fields") or [])
            | {
                "crash_consistency.commit_generation",
                "crash_consistency.commit_marker",
                "crash_consistency.directory_fsync_evidence",
                "crash_consistency.atomic_replace_evidence",
            }
        )
        rows.append(
            {
                "profile_id": profile.get("profile_id"),
                "status": "not_eligible",
                "reason": "atomic_generation_evidence_unavailable",
                "missing_fields": missing,
                "files_modified": 0,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "score_scope": "persistence_only_not_quality_or_official_score",
        "profile_count": len(rows),
        "profiles": rows,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "frozen_artifact_write_count": 0,
        },
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, stable_json_bytes(dict(value)))


def _run_fault_scenario(root: Path, point: str) -> dict[str, Any]:
    store, previous = _seed_store(root)
    completion_fault = point == "manifest_written_completion_missing"
    if completion_fault:
        previous = store.commit_record(_record("query-2", "succeeded"))
    before_hash = _tree_hash(previous.generation_path)
    committed_after_fault = False
    error_type = None
    try:
        if completion_fault:
            store.commit_completion(
                {
                    "metrics.json": stable_json_bytes(
                        {"scope": "persistence_fixture"}
                    ),
                    "summary.md": b"# Offline persistence fixture\n",
                },
                injector=FaultInjector(point=point),  # type: ignore[arg-type]
            )
        else:
            store.commit_record(
                _record("query-2", "succeeded"),
                injector=FaultInjector(point=point),  # type: ignore[arg-type]
            )
    except (CrashConsistencyError, OSError) as exc:
        error_type = type(exc).__name__
    recovered = store.load_latest()
    committed_after_fault = recovered.generation > previous.generation
    expected_committed = point == "after_commit_marker"
    violations: list[dict[str, Any]] = []
    if committed_after_fault != expected_committed:
        violations.append(
            _violation(
                point,
                recovered.generation,
                "commit_boundary_visibility",
                "$.generation",
                previous.generation + (1 if expected_committed else 0),
                recovered.generation,
            )
        )
    if _tree_hash(previous.generation_path) != before_hash:
        violations.append(
            _violation(
                point,
                previous.generation,
                "previous_generation_immutable",
                "$.previous_generation.tree_sha256",
                before_hash,
                _tree_hash(previous.generation_path),
            )
        )
    expected_records = 2 if expected_committed or completion_fault else 1
    if len(recovered.records) != expected_records:
        violations.append(
            _violation(
                point,
                recovered.generation,
                "resume_last_complete_checkpoint",
                "$.record_count",
                expected_records,
                len(recovered.records),
            )
        )
    return {
        "fault_point": point,
        "status": "passed" if not violations else "invariant_violation",
        "pre_fault_generation": previous.generation,
        "recovered_generation": recovered.generation,
        "record_count": len(recovered.records),
        "fault_terminal": error_type or "committed_then_interrupted",
        "violations": violations,
    }


def _run_normal_scenario(root: Path) -> dict[str, Any]:
    store = BenchmarkRunCommitStore(root)
    state = store.initialize(
        run_id="offline-gate",
        expected_query_ids=["query-1", "query-2"],
        config=_config(["query-1", "query-2"]),
        dataset_report={"count": 2},
    )
    state = store.commit_record(_record("query-1", "succeeded"))
    state = store.commit_record(_record("query-2", "failed"))
    reports = {
        "metrics.json": stable_json_bytes({"scope": "persistence_fixture"}),
        "summary.md": b"# Offline persistence fixture\n",
    }
    state = store.commit_completion(reports)
    recovered = store.load_latest()
    store.materialize_compatibility_view(recovered)
    violations: list[dict[str, Any]] = []
    if state.generation != recovered.generation or recovered.status != "completed":
        violations.append(
            _violation(
                "normal",
                recovered.generation,
                "completed_generation_readable",
                "$.status",
                "completed",
                recovered.status,
            )
        )
    if [row["case_id"] for row in recovered.records] != ["query-1", "query-2"]:
        violations.append(
            _violation(
                "normal",
                recovered.generation,
                "resume_without_duplicate_or_omission",
                "$.record_identities",
                ["query-1", "query-2"],
                [row["case_id"] for row in recovered.records],
            )
        )
    return {
        "status": "passed" if not violations else "invariant_violation",
        "generation_count": recovered.generation,
        "record_count": len(recovered.records),
        "event_count": recovered.event_count,
        "completed": recovered.status == "completed",
        "violations": violations,
    }


def _run_corrupt_latest_scenario(root: Path) -> dict[str, Any]:
    store, previous = _seed_store(root)
    latest = store.commit_record(_record("query-2", "succeeded"))
    (latest.generation_path / "delta.json").write_bytes(b'{"torn":')
    recovered = store.load_latest()
    violations: list[dict[str, Any]] = []
    if recovered.generation != previous.generation:
        violations.append(
            _violation(
                "corrupt_latest_generation",
                latest.generation,
                "fallback_to_last_valid_generation",
                "$.recovered_generation",
                previous.generation,
                recovered.generation,
            )
        )
    return {
        "status": "passed" if not violations else "invariant_violation",
        "damaged_generation": latest.generation,
        "recovered_generation": recovered.generation,
        "violations": violations,
    }


def _run_stale_temp_scenario(root: Path) -> dict[str, Any]:
    store, state = _seed_store(root)
    pending = store.generations / ".generation-00000003.pending-stale"
    pending.mkdir()
    (pending / "partial.json").write_bytes(b"{")
    removed = store.cleanup_uncommitted_temporaries()
    recovered = store.load_latest()
    violations: list[dict[str, Any]] = []
    if removed != [pending.name] or recovered.generation != state.generation:
        violations.append(
            _violation(
                "stale_temporary",
                recovered.generation,
                "cleanup_only_uncommitted_temporary",
                "$.removed",
                [pending.name],
                removed,
            )
        )
    return {
        "status": "passed" if not violations else "invariant_violation",
        "removed": removed,
        "preserved_generation": recovered.generation,
        "violations": violations,
    }


def _run_concurrent_writer_scenario(root: Path) -> dict[str, Any]:
    store, state = _seed_store(root)
    rejected = False
    with store.writer_lock():
        try:
            store.commit_record(_record("query-2", "succeeded"))
        except ConcurrentWriterError:
            rejected = True
    recovered = store.load_latest()
    violations: list[dict[str, Any]] = []
    if not rejected or recovered.generation != state.generation:
        violations.append(
            _violation(
                "concurrent_writer",
                recovered.generation,
                "concurrent_writer_rejected",
                "$.writer_lock",
                "rejected",
                "accepted" if not rejected else "state_changed",
            )
        )
    return {
        "status": "passed" if not violations else "invariant_violation",
        "second_writer": "rejected" if rejected else "accepted",
        "preserved_generation": recovered.generation,
        "violations": violations,
    }


def _run_non_atomic_fault(root: Path) -> dict[str, Any]:
    store, state = _seed_store(root)
    expected = _tree_hash(state.generation_path)
    # Deterministically model an in-place non-atomic update destroying the
    # previously committed value.  The gate must make this visible as a breach.
    (state.generation_path / "delta.json").write_bytes(b'{"record":')
    observed = _tree_hash(state.generation_path)
    violation = _violation(
        "controlled_non_atomic_writer",
        state.generation,
        "previous_generation_immutable",
        "$.generation.tree_sha256",
        expected,
        observed,
    )
    return {
        "status": "invariant_violation",
        "generation": state.generation,
        "violations": [violation],
    }


def _seed_store(root: Path) -> tuple[BenchmarkRunCommitStore, CommittedRunState]:
    store = BenchmarkRunCommitStore(root)
    store.initialize(
        run_id="offline-gate",
        expected_query_ids=["query-1", "query-2"],
        config=_config(["query-1", "query-2"]),
        dataset_report={"count": 2},
    )
    state = store.commit_record(_record("query-1", "succeeded"))
    return store, state


def _record(case_id: str, status: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": f"offline {case_id}",
        "status": status,
        "error_type": None if status == "succeeded" else "InjectedFixtureFailure",
        "error": None if status == "succeeded" else "offline fixture failure",
    }


def _config(case_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "dataset": "offline_fixture",
        "case_count": len(case_ids),
        "case_ids": list(case_ids),
        "resume_signature": "a" * 64,
    }


def _events_for_delta(
    delta: Mapping[str, Any],
    *,
    generation: int,
    sequence_start: int,
    record_count: int,
) -> list[dict[str, Any]]:
    kind = delta.get("kind")
    if kind == "initialize":
        return [
            {
                "sequence": sequence_start + 1,
                "generation": generation,
                "event": "run_initialized",
                "record_count": record_count,
            }
        ]
    if kind == "record":
        record = delta["record"]
        return [
            {
                "sequence": sequence_start + 1,
                "generation": generation,
                "event": "query_state_committed",
                "query_identity": record["case_id"],
                "query_status": record.get("status"),
                "record_count": record_count,
            }
        ]
    if kind == "complete":
        return [
            {
                "sequence": sequence_start + 1,
                "generation": generation,
                "event": "run_completed",
                "record_count": record_count,
            }
        ]
    raise CrashConsistencyError("delta_kind_invalid")


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validated_expected_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise CrashConsistencyError("expected_query_identities_invalid")
    if len(set(normalized)) != len(normalized):
        raise CrashConsistencyError("expected_query_identities_duplicate")
    return normalized


def _validated_comparison_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "contract",
        "plan_sha256",
        "role",
        "common_execution_contract_sha256",
    }
    if set(value) != expected_keys:
        raise CrashConsistencyError("comparison_binding_fields_invalid")
    normalized = {str(key): item for key, item in value.items()}
    if normalized["contract"] != "comparison_plan_v1":
        raise CrashConsistencyError("comparison_binding_contract_invalid")
    if normalized["role"] not in {"baseline", "candidate"}:
        raise CrashConsistencyError("comparison_binding_role_invalid")
    for field in ("plan_sha256", "common_execution_contract_sha256"):
        if not isinstance(normalized[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", normalized[field]
        ):
            raise CrashConsistencyError("comparison_binding_digest_invalid")
    return normalized


def _validated_shard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "contract",
        "plan_sha256",
        "shard_index",
        "shard_count",
        "expected_query_identities_sha256",
        "common_execution_contract_sha256",
        "attempt_id",
        "supersedes_attempt_id",
    }
    if set(value) != expected_keys:
        raise CrashConsistencyError("shard_binding_fields_invalid")
    normalized = {str(key): item for key, item in value.items()}
    if normalized["contract"] != "shard_plan_v1":
        raise CrashConsistencyError("shard_binding_contract_invalid")
    shard_index = normalized["shard_index"]
    shard_count = normalized["shard_count"]
    if (
        not isinstance(shard_index, int)
        or isinstance(shard_index, bool)
        or not isinstance(shard_count, int)
        or isinstance(shard_count, bool)
        or shard_count < 1
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise CrashConsistencyError("shard_binding_index_invalid")
    for field in (
        "plan_sha256",
        "expected_query_identities_sha256",
        "common_execution_contract_sha256",
    ):
        if not isinstance(normalized[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", normalized[field]
        ):
            raise CrashConsistencyError("shard_binding_digest_invalid")
    attempt = normalized["attempt_id"]
    supersedes = normalized["supersedes_attempt_id"]
    attempt_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
    if not isinstance(attempt, str) or not re.fullmatch(attempt_pattern, attempt):
        raise CrashConsistencyError("shard_binding_attempt_invalid")
    if supersedes is not None and (
        not isinstance(supersedes, str)
        or not re.fullmatch(attempt_pattern, supersedes)
        or supersedes == attempt
    ):
        raise CrashConsistencyError("shard_binding_supersedes_invalid")
    return normalized


def _validate_report_names(reports: Mapping[str, bytes]) -> None:
    invalid = set(reports) - (_ALLOWED_PUBLIC_ARTIFACTS - {
        "config.json", "dataset_report.json", "results.jsonl", "failures.jsonl"
    })
    if invalid:
        raise CrashConsistencyError("report_artifact_name_invalid")


def _validate_relative_name(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise CrashConsistencyError("generation_file_path_invalid")
    if value not in _INTERNAL_FILES and value not in _ALLOWED_PUBLIC_ARTIFACTS:
        raise CrashConsistencyError("generation_file_name_unregistered")


def _ordered_hash(values: Sequence[str]) -> str:
    return sha256_bytes(stable_json_bytes(list(values), indent=None))


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrashConsistencyError("json_object_required")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if not isinstance(item, dict):
            raise CrashConsistencyError("jsonl_object_required")
        values.append(item)
    return values


def _tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sha256_bytes(stable_json_bytes(rows, indent=None))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lock_file_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.fstat(descriptor).st_size == 0:
            try:
                os.write(descriptor, b"\0")
            except OSError as exc:
                if exc.errno in _WINDOWS_LOCK_CONFLICT_ERRNOS:
                    raise ConcurrentWriterError(
                        "concurrent_writer_rejected"
                    ) from exc
                raise
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in _WINDOWS_LOCK_CONFLICT_ERRNOS:
                raise ConcurrentWriterError("concurrent_writer_rejected") from exc
            raise
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ConcurrentWriterError("concurrent_writer_rejected") from exc


def _unlock_file_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _violation(
    fault_point: str,
    generation: int,
    invariant: str,
    path: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "fault_point": fault_point,
        "commit_generation": generation,
        "invariant": invariant,
        "first_violation_path": path,
        "expected_sha256": sha256_bytes(stable_json_bytes(expected, indent=None)),
        "observed_sha256": sha256_bytes(stable_json_bytes(observed, indent=None)),
    }


def _repo_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise CrashConsistencyError("repository_path_invalid")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CrashConsistencyError("repository_path_outside_root") from exc
    return resolved


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    original_connect = socket.socket.connect

    def blocked_connect(*args: Any, **kwargs: Any) -> Any:
        attempts["network"] += 1
        raise AssertionError("network_forbidden")

    with patch.object(socket.socket, "connect", blocked_connect):
        yield
    socket.socket.connect = original_connect
