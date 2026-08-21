"""Record/replay snapshots for constrained post-retrieval LLM feedback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.evaluation.snapshots.store import (
    SnapshotConflictError,
    SnapshotIntegrityError,
    SnapshotMissingError,
    utc_now,
)


LLMFeedbackMode = Literal["live", "record", "replay", "record-missing"]
LLM_FEEDBACK_SNAPSHOT_SCHEMA_VERSION = "1"


class LLMFeedbackRequest(BaseModel):
    """A de-identified, deterministic description of one feedback request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LLM_FEEDBACK_SNAPSHOT_SCHEMA_VERSION
    provider: str
    model: str | None = None
    base_url_host: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str = Field(min_length=64, max_length=64)
    request_identity: dict[str, Any]
    temperature: float = 0.0
    max_tokens: int = Field(ge=1)
    max_supplemental_queries: int = Field(ge=0, le=1)


class LLMFeedbackExecution(BaseModel):
    raw_response: dict[str, Any]
    snapshot_key: str | None = None
    snapshot_status: str | None = None
    llm_call_attempted: bool = False
    replayed: bool = False
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    http_attempts: int = Field(default=0, ge=0)
    http_429_count: int = Field(default=0, ge=0)
    retry_after_seconds: list[float] = Field(default_factory=list)
    retry_wait_seconds: float = Field(default=0.0, ge=0.0)
    provider_failure_class: str | None = None
    provider_cache_hit: bool = False


class LLMFeedbackSnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LLM_FEEDBACK_SNAPSHOT_SCHEMA_VERSION
    key: str = Field(min_length=64, max_length=64)
    provider: str
    model: str | None = None
    base_url_host: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str = Field(min_length=64, max_length=64)
    request_identity: dict[str, Any]
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
    http_attempts: int = Field(default=0, ge=0)
    http_429_count: int = Field(default=0, ge=0)
    retry_after_seconds: list[float] = Field(default_factory=list)
    retry_wait_seconds: float = Field(default=0.0, ge=0.0)
    provider_failure_class: str | None = None
    provider_cache_hit: bool = False
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)


class LLMFeedbackCostReport(BaseModel):
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


def llm_feedback_snapshot_key(request: LLMFeedbackRequest) -> str:
    payload = {
        "schema_version": request.schema_version,
        "provider": request.provider,
        "model": request.model,
        "base_url_host": request.base_url_host,
        "prompt_name": request.prompt_name,
        "prompt_version": request.prompt_version,
        "prompt_hash": request.prompt_hash,
        "request_identity": request.request_identity,
        "request_options": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "max_supplemental_queries": request.max_supplemental_queries,
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class LLMFeedbackSnapshotStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.directory = self.root / "llm_feedback"

    def read(self, key: str) -> LLMFeedbackSnapshotEntry:
        path = self._path(key)
        if not path.is_file():
            raise SnapshotMissingError(f"llm_feedback_snapshot_missing:{key}")
        try:
            entry = LLMFeedbackSnapshotEntry.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise SnapshotIntegrityError(f"llm_feedback_snapshot_invalid:{key}") from exc
        if entry.key != key:
            raise SnapshotIntegrityError(f"llm_feedback_snapshot_key_mismatch:{key}")
        if entry.schema_version != LLM_FEEDBACK_SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotIntegrityError(f"llm_feedback_snapshot_schema:{key}")
        if entry.content_hash != _content_hash(entry):
            raise SnapshotIntegrityError(f"llm_feedback_snapshot_hash_mismatch:{key}")
        return entry

    def write(self, entry: LLMFeedbackSnapshotEntry, *, overwrite: bool = False) -> bool:
        if entry.content_hash != _content_hash(entry):
            raise SnapshotIntegrityError("llm_feedback_snapshot_content_hash_invalid")
        path = self._path(entry.key)
        if path.is_file():
            existing = self.read(entry.key)
            if existing.content_hash == entry.content_hash:
                return False
            if not overwrite:
                raise SnapshotConflictError(f"llm_feedback_snapshot_conflict:{entry.key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
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
            raise SnapshotIntegrityError("llm_feedback_snapshot_key_invalid")
        return self.directory / f"{key}.json"


class LLMFeedbackSnapshotRuntime:
    """A fail-closed live/record/replay boundary for feedback requests."""

    def __init__(self, store: LLMFeedbackSnapshotStore, *, mode: LLMFeedbackMode) -> None:
        self.store = store
        self.mode = mode
        self._case = LLMFeedbackCostReport(mode=mode)
        self._last_key: str | None = None
        self._last_status: str | None = None
        self._last_call_attempted = False

    def begin_case(self, case_id: str = "") -> None:  # case_id is reserved for audit symmetry.
        del case_id
        self._case = LLMFeedbackCostReport(mode=self.mode)
        self._last_key = None
        self._last_status = None
        self._last_call_attempted = False

    def identity(self) -> tuple[str, str | None, str | None] | None:
        return self.store.identity()

    def finish_case(self) -> LLMFeedbackCostReport:
        return self._case.model_copy(deep=True)

    def failure_diagnostics(self) -> dict[str, Any]:
        return {
            "snapshot_key": self._last_key,
            "snapshot_status": self._last_status,
            "llm_call_attempted": self._last_call_attempted,
        }

    def execute(
        self,
        request: LLMFeedbackRequest,
        messages: list[dict[str, str]],
        client: Any | None,
        *,
        timeout: float,
    ) -> LLMFeedbackExecution:
        key = llm_feedback_snapshot_key(request)
        self._last_key = key
        _append_once(self._case.observed_keys, key)
        existing = self._read_optional(key)
        if self.mode == "replay":
            if existing is None:
                self._last_status = "missing"
                _append_once(self._case.missing_keys, key)
                raise SnapshotMissingError(f"llm_feedback_snapshot_missing:{key}")
            return self._replay(existing)
        if self.mode == "record-missing" and existing is not None and existing.status == "success":
            return self._replay(existing)
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
            wrote = self.store.write(
                _entry(request, key=key, status="failed", raw_response=None, error_message=_sanitize(str(exc)), execution=None),
                overwrite=existing is not None,
            )
            self._case.snapshot_writes += int(wrote)
            self._case.live_call_count += 1
            raise
        wrote = self.store.write(
            _entry(request, key=key, status="success", raw_response=execution.raw_response, error_message=None, execution=execution),
            overwrite=self.mode == "record" or existing is not None,
        )
        self._case.snapshot_writes += int(wrote)
        self._case.live_call_count += int(execution.llm_call_attempted)
        self._case.prompt_tokens += execution.prompt_tokens
        self._case.completion_tokens += execution.completion_tokens
        self._case.total_tokens += execution.total_tokens
        self._case.recorded_latency_seconds += execution.recorded_latency_seconds
        return execution.model_copy(update={"snapshot_key": key, "snapshot_status": "record"})

    def _read_optional(self, key: str) -> LLMFeedbackSnapshotEntry | None:
        try:
            return self.store.read(key)
        except SnapshotMissingError:
            return None

    def _replay(self, entry: LLMFeedbackSnapshotEntry) -> LLMFeedbackExecution:
        if entry.status != "success" or entry.raw_response is None:
            self._last_status = "failed"
            raise RuntimeError("llm_feedback_snapshot_failed")
        self._last_status = "replay"
        self._last_call_attempted = False
        self._case.snapshot_hits += 1
        self._case.prompt_tokens += entry.prompt_tokens
        self._case.completion_tokens += entry.completion_tokens
        self._case.total_tokens += entry.total_tokens
        self._case.recorded_latency_seconds += entry.recorded_latency_seconds
        return LLMFeedbackExecution(
            raw_response=entry.raw_response,
            snapshot_key=entry.key,
            snapshot_status="replay",
            llm_call_attempted=False,
            replayed=True,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            total_tokens=entry.total_tokens,
            recorded_latency_seconds=entry.recorded_latency_seconds,
            http_attempts=entry.http_attempts,
            http_429_count=entry.http_429_count,
            retry_after_seconds=entry.retry_after_seconds,
            retry_wait_seconds=entry.retry_wait_seconds,
            provider_failure_class=entry.provider_failure_class,
            provider_cache_hit=entry.provider_cache_hit,
        )


def _live_execution(client: Any | None, messages: list[dict[str, str]], *, timeout: float) -> LLMFeedbackExecution:
    if client is None:
        raise RuntimeError("llm_unconfigured")
    before = _token_usage(client)
    started = time.perf_counter()
    raw = client.chat_json(messages, temperature=0, timeout=timeout)
    elapsed = time.perf_counter() - started
    if not isinstance(raw, dict):
        raise ValueError("llm_feedback_invalid_schema")
    after = _token_usage(client)
    transport = getattr(client, "last_call_diagnostics", None)
    return LLMFeedbackExecution(
        raw_response=raw,
        llm_call_attempted=True,
        snapshot_status="live",
        prompt_tokens=max(0, after[0] - before[0]),
        completion_tokens=max(0, after[1] - before[1]),
        total_tokens=max(0, after[2] - before[2]),
        recorded_latency_seconds=elapsed,
        http_attempts=max(0, int(getattr(transport, "http_attempts", 0))),
        http_429_count=max(0, int(getattr(transport, "http_429_count", 0))),
        retry_after_seconds=[max(0.0, float(value)) for value in getattr(transport, "retry_after_seconds", ())],
        retry_wait_seconds=max(0.0, float(getattr(transport, "retry_wait_seconds", 0.0))),
        provider_failure_class=getattr(transport, "failure_class", None),
        provider_cache_hit=bool(getattr(transport, "cache_hit", False)),
    )


def _entry(
    request: LLMFeedbackRequest,
    *,
    key: str,
    status: Literal["success", "failed"],
    raw_response: dict[str, Any] | None,
    error_message: str | None,
    execution: LLMFeedbackExecution | None,
) -> LLMFeedbackSnapshotEntry:
    entry = LLMFeedbackSnapshotEntry(
        key=key,
        provider=request.provider,
        model=request.model,
        base_url_host=request.base_url_host,
        prompt_name=request.prompt_name,
        prompt_version=request.prompt_version,
        prompt_hash=request.prompt_hash,
        request_identity=request.request_identity,
        request_options={
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "max_supplemental_queries": request.max_supplemental_queries,
            "schema_version": request.schema_version,
        },
        status=status,
        raw_response=raw_response,
        error_message=error_message,
        warnings=[] if status == "success" else ["llm_feedback_request_failed"],
        llm_call_count=1,
        prompt_tokens=execution.prompt_tokens if execution else 0,
        completion_tokens=execution.completion_tokens if execution else 0,
        total_tokens=execution.total_tokens if execution else 0,
        recorded_latency_seconds=execution.recorded_latency_seconds if execution else 0.0,
        http_attempts=execution.http_attempts if execution else 0,
        http_429_count=execution.http_429_count if execution else 0,
        retry_after_seconds=execution.retry_after_seconds if execution else [],
        retry_wait_seconds=execution.retry_wait_seconds if execution else 0.0,
        provider_failure_class=execution.provider_failure_class if execution else None,
        provider_cache_hit=execution.provider_cache_hit if execution else False,
        recorded_at=utc_now(),
        content_hash="0" * 64,
    )
    return entry.model_copy(update={"content_hash": _content_hash(entry)})


def _content_hash(entry: LLMFeedbackSnapshotEntry) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("content_hash", None)
    payload.pop("recorded_at", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sanitize(value: str) -> str:
    return re.sub(r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)[^\s&,;]+", r"\1\2[REDACTED]", value)[:1000]


def _token_usage(client: Any) -> tuple[int, int, int]:
    usage = getattr(client, "token_usage", None)
    getter = usage.get if isinstance(usage, dict) else lambda key, default=0: getattr(usage, key, default)
    return tuple(max(0, int(getter(key, 0) or 0)) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
