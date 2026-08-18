"""LLM 语义查询规划的独立 Record/Replay 快照。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.agents.llm_query_planning import (
    LLMPlanningExecution,
    LLMPlanningRequest,
)
from scholar_agent.evaluation.snapshots.schemas import SnapshotPlanEntry
from scholar_agent.evaluation.snapshots.store import (
    SnapshotConflictError,
    SnapshotIntegrityError,
    SnapshotMissingError,
    utc_now,
)
from scholar_agent.llm.provider import get_llm_request_options


LLMPlanningMode = Literal["live", "record", "replay", "record-missing"]
LLM_PLANNING_SNAPSHOT_SCHEMA_VERSION = "1"


class LLMPlanningSnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LLM_PLANNING_SNAPSHOT_SCHEMA_VERSION
    key: str = Field(min_length=64, max_length=64)
    provider: str
    model: str | None = None
    base_url_host: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str = Field(min_length=64, max_length=64)
    normalized_input_hash: str = Field(min_length=64, max_length=64)
    request_options: dict[str, Any]
    status: Literal["success", "failed"]
    raw_response: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    llm_call_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)


class LLMPlanningCostReport(BaseModel):
    mode: str
    snapshot_hits: int = Field(default=0, ge=0)
    snapshot_writes: int = Field(default=0, ge=0)
    missing_keys: list[str] = Field(default_factory=list)
    observed_keys: list[str] = Field(default_factory=list)
    live_call_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    replay_execution_request_count: int = Field(default=0, ge=0)
    replay_execution_retry_count: int = Field(default=0, ge=0)
    replay_execution_network_wait_seconds: float = Field(default=0.0, ge=0.0)


def llm_planning_snapshot_key(request: LLMPlanningRequest) -> tuple[str, str]:
    normalized_input = _canonical_json(request.input_payload)
    normalized_input_hash = hashlib.sha256(normalized_input.encode("utf-8")).hexdigest()
    payload = {
        "schema_version": request.schema_version,
        "query_planning_policy": request.query_planning_policy,
        "provider": request.provider,
        "model": request.model,
        "base_url_host": request.base_url_host,
        "prompt_name": request.prompt_name,
        "prompt_version": request.prompt_version,
        "prompt_hash": request.prompt_hash,
        "normalized_input_hash": normalized_input_hash,
        "request_options": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "max_supplemental_queries": request.max_supplemental_queries,
        },
        "run_profile": request.run_profile,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(), normalized_input_hash


class LLMPlanningSnapshotStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.directory = self.root / "llm_planning"

    def read(self, key: str) -> LLMPlanningSnapshotEntry:
        path = self._path(key)
        if not path.is_file():
            raise SnapshotMissingError(f"llm_planning_snapshot_missing:{key}")
        try:
            entry = LLMPlanningSnapshotEntry.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise SnapshotIntegrityError(f"llm_planning_snapshot_invalid:{key}") from exc
        if entry.key != key:
            raise SnapshotIntegrityError(f"llm_planning_snapshot_key_mismatch:{key}")
        if entry.schema_version != LLM_PLANNING_SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotIntegrityError(f"llm_planning_snapshot_schema:{key}")
        if entry.content_hash != _content_hash(entry):
            raise SnapshotIntegrityError(f"llm_planning_snapshot_hash_mismatch:{key}")
        return entry

    def write(
        self,
        entry: LLMPlanningSnapshotEntry,
        *,
        overwrite: bool = False,
    ) -> bool:
        if entry.content_hash != _content_hash(entry):
            raise SnapshotIntegrityError("llm_planning_snapshot_content_hash_invalid")
        path = self._path(entry.key)
        if path.is_file():
            existing = self.read(entry.key)
            if existing.content_hash == entry.content_hash:
                return False
            if not overwrite:
                raise SnapshotConflictError(
                    f"llm_planning_snapshot_conflict:{entry.key}"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    entry.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def identity(self) -> tuple[str, str | None, str | None] | None:
        identities: set[tuple[str, str | None, str | None]] = set()
        paths = sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []
        for path in paths:
            try:
                entry = self.read(path.stem)
            except (SnapshotIntegrityError, SnapshotMissingError):
                continue
            identities.add((entry.provider, entry.model, entry.base_url_host))
        return next(iter(identities)) if len(identities) == 1 else None

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise SnapshotIntegrityError("llm_planning_snapshot_key_invalid")
        return self.directory / f"{key}.json"


class LLMPlanningSnapshotRuntime:
    """LLM 规划快照模式与检索快照模式相互独立。"""

    def __init__(
        self,
        store: LLMPlanningSnapshotStore,
        *,
        mode: LLMPlanningMode,
        group_name: str = "baseline",
    ) -> None:
        self.store = store
        self.mode = mode
        self.group_name = group_name
        self._case_id = ""
        self._case = LLMPlanningCostReport(mode=mode)
        self._requests: dict[str, LLMPlanningRequest] = {}
        self._request_cases: dict[str, str] = {}
        self._case_keys: dict[str, list[str]] = {}
        self._missing: list[str] = []
        self._last_key: str | None = None
        self._last_status: str | None = None
        self._last_call_attempted = False

    def begin_case(self, case_id: str) -> None:
        self._case_id = case_id
        self._case = LLMPlanningCostReport(mode=self.mode)
        self._last_key = None
        self._last_status = None
        self._last_call_attempted = False

    def identity(self) -> tuple[str, str | None, str | None] | None:
        return self.store.identity()

    def finish_case(self) -> LLMPlanningCostReport:
        return self._case.model_copy(deep=True)

    def missing_keys(self) -> list[str]:
        return list(self._missing)

    def dependency_keys(self, case_id: str) -> list[str]:
        return list(self._case_keys.get(case_id, []))

    def failure_diagnostics(self) -> dict[str, Any]:
        return {
            "snapshot_key": self._last_key,
            "snapshot_status": self._last_status,
            "llm_call_attempted": self._last_call_attempted,
        }

    def plan_entries(self) -> list[SnapshotPlanEntry]:
        entries: list[SnapshotPlanEntry] = []
        for key in self._missing:
            request = self._requests[key]
            entries.append(
                SnapshotPlanEntry(
                    key=key,
                    entry_type="llm_planning",
                    source="llm",
                    limit=0,
                    connector_version=request.schema_version,
                    required_by_group=self.group_name,
                    case_id=self._request_cases.get(key, self._case_id),
                    stage="llm_query_planning",
                    generated_by="llm_query_planning",
                    query_planning_policy=request.query_planning_policy,
                    query_planner_version=request.prompt_version,
                    priority=1,
                    already_present=False,
                    llm_request=request.model_dump(mode="json"),
                )
            )
        return entries

    def execute(
        self,
        request: LLMPlanningRequest,
        messages: list[dict[str, str]],
        client: Any | None,
        *,
        timeout: float,
    ) -> LLMPlanningExecution:
        key, input_hash = llm_planning_snapshot_key(request)
        self._last_key = key
        self._requests[key] = request.model_copy(deep=True)
        self._request_cases[key] = self._case_id
        _stable_append(self._case_keys.setdefault(self._case_id, []), key)
        _stable_append(self._case.observed_keys, key)
        existing = self._read_optional(key)
        if self.mode == "replay":
            if existing is None:
                self._last_status = "missing"
                _stable_append(self._case.missing_keys, key)
                _stable_append(self._missing, key)
                raise SnapshotMissingError(f"llm_planning_snapshot_missing:{key}")
            return self._replay(existing)
        if self.mode == "record-missing" and existing is not None:
            if existing.status == "success":
                return self._replay(existing)
            # failed 条目允许恢复，且仅补齐该缺失/失败键。
        if self.mode == "live":
            self._last_status = "live"
            self._last_call_attempted = client is not None
            return _live_execution(client, messages, timeout=timeout)
        if client is None:
            self._last_status = "unconfigured"
            raise RuntimeError("llm_unconfigured")

        self._last_status = "record"
        self._last_call_attempted = True
        try:
            execution = _live_execution(client, messages, timeout=timeout)
        except Exception as exc:
            self._last_status = "failed"
            entry = _entry(
                request,
                key=key,
                input_hash=input_hash,
                status="failed",
                raw_response=None,
                error_message=_sanitize(str(exc)),
                execution=None,
            )
            wrote = self.store.write(entry, overwrite=existing is not None)
            self._case.snapshot_writes += int(wrote)
            self._case.live_call_count += 1
            raise
        entry = _entry(
            request,
            key=key,
            input_hash=input_hash,
            status="success",
            raw_response=execution.raw_response,
            error_message=None,
            execution=execution,
        )
        wrote = self.store.write(
            entry,
            overwrite=self.mode == "record" or existing is not None,
        )
        self._case.snapshot_writes += int(wrote)
        self._case.live_call_count += execution.llm_call_attempted
        self._case.prompt_tokens += execution.prompt_tokens
        self._case.completion_tokens += execution.completion_tokens
        self._case.total_tokens += execution.total_tokens
        self._case.recorded_latency_seconds += execution.recorded_latency_seconds
        return execution.model_copy(
            update={
                "snapshot_key": key,
                "snapshot_status": "record",
            }
        )

    def _read_optional(self, key: str) -> LLMPlanningSnapshotEntry | None:
        try:
            return self.store.read(key)
        except SnapshotMissingError:
            return None

    def _replay(self, entry: LLMPlanningSnapshotEntry) -> LLMPlanningExecution:
        if entry.status != "success" or entry.raw_response is None:
            self._last_status = "failed"
            raise RuntimeError("llm_planning_snapshot_failed")
        self._last_status = "replay"
        self._last_call_attempted = False
        self._case.snapshot_hits += 1
        self._case.prompt_tokens += entry.prompt_tokens
        self._case.completion_tokens += entry.completion_tokens
        self._case.total_tokens += entry.total_tokens
        self._case.recorded_latency_seconds += entry.recorded_latency_seconds
        return LLMPlanningExecution(
            raw_response=entry.raw_response,
            snapshot_key=entry.key,
            snapshot_status="replay",
            llm_call_attempted=False,
            replayed=True,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            total_tokens=entry.total_tokens,
            recorded_latency_seconds=entry.recorded_latency_seconds,
        )


def collect_llm_plan_entry(
    entry: SnapshotPlanEntry,
    store: LLMPlanningSnapshotStore,
    client: Any,
) -> LLMPlanningExecution:
    if entry.entry_type != "llm_planning" or entry.llm_request is None:
        raise ValueError(f"llm_planning_plan_entry_invalid:{entry.key}")
    request = LLMPlanningRequest.model_validate(entry.llm_request)
    from scholar_agent.prompts.loader import render_messages

    runtime = LLMPlanningSnapshotRuntime(
        store,
        mode="record-missing",
        group_name=entry.required_by_group,
    )
    runtime.begin_case(entry.case_id)
    execution = runtime.execute(
        request,
        render_messages(request.prompt_name, request.input_payload),
        client,
        timeout=float(get_llm_request_options()["timeout_seconds"]),
    )
    if execution.snapshot_key != entry.key:
        raise SnapshotIntegrityError("llm_planning_plan_key_mismatch")
    return execution


def _live_execution(
    client: Any | None,
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> LLMPlanningExecution:
    if client is None:
        raise RuntimeError("llm_unconfigured")
    before = _token_usage(client)
    started = time.perf_counter()
    raw_response = client.chat_json(messages, temperature=0, timeout=timeout)
    elapsed = time.perf_counter() - started
    if not isinstance(raw_response, dict):
        raise ValueError("llm_planning_invalid_schema")
    after = _token_usage(client)
    return LLMPlanningExecution(
        raw_response=raw_response,
        llm_call_attempted=True,
        snapshot_status="live",
        prompt_tokens=max(0, after[0] - before[0]),
        completion_tokens=max(0, after[1] - before[1]),
        total_tokens=max(0, after[2] - before[2]),
        recorded_latency_seconds=elapsed,
    )


def _entry(
    request: LLMPlanningRequest,
    *,
    key: str,
    input_hash: str,
    status: Literal["success", "failed"],
    raw_response: dict[str, Any] | None,
    error_message: str | None,
    execution: LLMPlanningExecution | None,
) -> LLMPlanningSnapshotEntry:
    entry = LLMPlanningSnapshotEntry(
        key=key,
        provider=request.provider,
        model=request.model,
        base_url_host=request.base_url_host,
        prompt_name=request.prompt_name,
        prompt_version=request.prompt_version,
        prompt_hash=request.prompt_hash,
        normalized_input_hash=input_hash,
        request_options={
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "max_supplemental_queries": request.max_supplemental_queries,
            "schema_version": request.schema_version,
        },
        status=status,
        raw_response=raw_response,
        error_message=error_message,
        warnings=[] if status == "success" else ["llm_planning_request_failed"],
        llm_call_count=1,
        prompt_tokens=execution.prompt_tokens if execution is not None else 0,
        completion_tokens=execution.completion_tokens if execution is not None else 0,
        total_tokens=execution.total_tokens if execution is not None else 0,
        recorded_latency_seconds=(
            execution.recorded_latency_seconds if execution is not None else 0.0
        ),
        recorded_at=utc_now(),
        content_hash="0" * 64,
    )
    return entry.model_copy(update={"content_hash": _content_hash(entry)})


def _content_hash(entry: LLMPlanningSnapshotEntry) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("content_hash", None)
    payload.pop("recorded_at", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sanitize(value: str) -> str:
    return re.sub(
        r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)[^\s&,;]+",
        r"\1\2[REDACTED]",
        value,
    )[:1000]


def _token_usage(client: Any | None) -> tuple[int, int, int]:
    usage = getattr(client, "token_usage", None)
    if usage is None:
        return 0, 0, 0

    def value(name: str) -> int:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    return value("prompt_tokens"), value("completion_tokens"), value("total_tokens")


def _stable_append(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
