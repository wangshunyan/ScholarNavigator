"""Deterministic off-site backup and restore controls for future Full1000 runs.

The authority is the production ``BenchmarkRunCommitStore`` generation chain
plus the launch authorization/audit, resource ledger, and provider-ingest
artifacts committed with each selected shard attempt. Compatibility mirrors are
never read. This module performs no network, LLM, Snapshot, or quality work.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    STORE_DIRECTORY,
    BenchmarkRunCommitStore,
    CrashConsistencyError,
    durable_atomic_write_bytes,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
)
from scholar_agent.evaluation.full1000_launch_control import (
    LaunchControlError,
    LaunchOperationMachine,
    OperationAuditLog,
    build_authorization,
    build_preparation,
    load_protocol as load_launch_protocol,
    validate_authorization,
    validate_launch_evidence,
)
from scholar_agent.evaluation.provider_ingest_provenance import (
    EncodingMetadata,
    ProviderIngestBundle,
    create_envelope,
    verify_capture_bundle,
    write_capture_bundle,
)
from scholar_agent.evaluation.resource_accounting import (
    ResourceLedgerObserver,
    ResourceLedgerV1,
    build_run_ledger,
    opaque_resource_identity,
    validate_resource_ledger,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_run_disaster_recovery_v1"
SCHEMA_VERSION = "1"
FROZEN_PROTOCOL_SHA256 = (
    "db930449dc5903a7015fba72f557f25945f4b69f56b7ba84fedea03cce94393d"
)
BACKUP_CONTRACT = "formal_run_backup_v1"
STATE_CONTRACT = "formal_run_recovery_state_v1"
AGGREGATE_CONTRACT = "formal_run_recovery_aggregate_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
OBJECTS_DIRECTORY = "objects"
MANIFESTS_DIRECTORY = "manifests"
LATEST_FILE = "LATEST.json"
AUTHORITY_FILES = (
    "authority/authorization.json",
    "authority/operation_audit.json",
    "authority/prepared.json",
    "authority/recovery_state.json",
)
REPORT_FILES = (
    "provider_ingest_provenance.json",
    "provider_ingest_raw.tar",
    "resource_ledger.json",
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
_HEX64 = r"^[0-9a-f]{64}$"
_PROHIBITED_NAMES = frozenset({".env", "writer.lock"})
_PROHIBITED_FRAGMENTS = (".pending-", ".tmp", "__pycache__")


class DisasterRecoveryError(RuntimeError):
    """Backup or restore input violates the frozen authority contract."""


class DisasterRecoveryNotEligible(DisasterRecoveryError):
    """The requested real run has no complete authoritative restore point."""


class BackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def validate_path(self) -> "BackupFile":
        _safe_relative(self.path)
        return self


class RestorePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_cursor: int = Field(ge=0, le=1000)
    completed_shards: list[int]
    selected_attempts: dict[str, str]
    generation_by_shard: dict[str, int]
    aggregate_state: Literal["pending", "completed"]
    operation_audit_sha256: str = Field(pattern=_HEX64)
    resource_request_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_shards(self) -> "RestorePoint":
        if self.completed_shards != sorted(set(self.completed_shards)):
            raise ValueError("completed shards must be sorted and unique")
        if any(value not in range(20) for value in self.completed_shards):
            raise ValueError("completed shard out of range")
        expected = {str(value) for value in self.completed_shards}
        if set(self.generation_by_shard) != expected:
            raise ValueError("generation coverage mismatch")
        if self.aggregate_state == "completed" and self.completed_shards != list(
            range(20)
        ):
            raise ValueError("completed aggregate requires every shard")
        return self


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["formal_run_backup_v1"] = BACKUP_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol_sha256: str = Field(pattern=_HEX64)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    backup_id: str = Field(pattern=_HEX64)
    parent_backup_id: str | None = Field(default=None, pattern=_HEX64)
    run_identity: str = Field(pattern=_HEX64)
    plan_sha256: str = Field(pattern=_HEX64)
    authorization_sha256: str = Field(pattern=_HEX64)
    files: list[BackupFile]
    restore_point: RestorePoint
    manifest_sha256: str = Field(pattern=_HEX64)
    score_scope: Literal[
        "disaster_recovery_only_not_quality_or_official_score"
    ] = "disaster_recovery_only_not_quality_or_official_score"

    @model_validator(mode="after")
    def validate_manifest(self) -> "BackupManifest":
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("backup inventory must be sorted and unique")
        payload = self.model_dump(mode="json")
        manifest_sha256 = payload.pop("manifest_sha256")
        if stable_hash(payload) != manifest_sha256:
            raise ValueError("backup manifest digest mismatch")
        backup_payload = dict(payload)
        backup_payload.pop("backup_id")
        if stable_hash(backup_payload) != self.backup_id:
            raise ValueError("backup identity mismatch")
        return self


class RecoveryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["formal_run_recovery_state_v1"] = STATE_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    run_identity: str = Field(pattern=_HEX64)
    plan_sha256: str = Field(pattern=_HEX64)
    authorization_sha256: str = Field(pattern=_HEX64)
    selected_attempts: dict[str, str]
    attempt_statuses: dict[str, Literal["not_started", "failed", "completed", "revoked"]]
    completed_shards: list[int]
    generation_by_shard: dict[str, int]
    adapter_call_count: int = Field(ge=0)
    aggregate_state: Literal["pending", "completed"]
    aggregate_sha256: str | None = Field(default=None, pattern=_HEX64)
    state_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def validate_state(self) -> "RecoveryState":
        if set(self.selected_attempts) != {str(value) for value in range(20)}:
            raise ValueError("selected attempt coverage mismatch")
        if self.completed_shards != sorted(set(self.completed_shards)):
            raise ValueError("completed shards must be sorted and unique")
        if set(self.generation_by_shard) != {
            str(value) for value in self.completed_shards
        }:
            raise ValueError("generation coverage mismatch")
        for shard in self.completed_shards:
            selected = self.selected_attempts[str(shard)]
            if self.attempt_statuses.get(selected) != "completed":
                raise ValueError("completed shard does not select completed attempt")
        if self.aggregate_state == "completed":
            if self.completed_shards != list(range(20)) or not self.aggregate_sha256:
                raise ValueError("aggregate completion state invalid")
        elif self.aggregate_sha256 is not None:
            raise ValueError("pending aggregate cannot have digest")
        payload = self.model_dump(mode="json")
        digest = payload.pop("state_sha256")
        if stable_hash(payload) != digest:
            raise ValueError("recovery state digest mismatch")
        return self


class _FixtureBudget:
    max_search_rounds = 1
    max_candidate_papers = 1
    max_llm_calls = 0
    max_total_tokens = 0
    max_latency_seconds = 1.0


class _AdapterCounter:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def call(self, query_identity: str) -> None:
        self.calls[query_identity] = self.calls.get(query_identity, 0) + 1
        if self.calls[query_identity] != 1:
            raise DisasterRecoveryError("committed_query_repeated_after_restore")

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(value), temporary_suffix="recovery")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DisasterRecoveryError("duplicate_json_key")
        value[key] = item
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DisasterRecoveryError("non_finite_json_number")
            ),
        )
    except DisasterRecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DisasterRecoveryError("json_input_unreadable") from exc
    if not isinstance(value, dict):
        raise DisasterRecoveryError("json_root_not_object")
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part in _PROHIBITED_NAMES for part in path.parts)
        or any(fragment in part for part in path.parts for fragment in _PROHIBITED_FRAGMENTS)
    ):
        raise DisasterRecoveryError("unsafe_or_prohibited_backup_path")
    return path.as_posix()


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = _read_object(path)
    required = {
        "activation",
        "authority",
        "backup",
        "bindings",
        "formal_validation_complete",
        "population",
        "protocol",
        "protocol_sha256",
        "restore",
        "schema_version",
        "score_scope",
        "source_commit",
    }
    if set(value) != required:
        raise DisasterRecoveryError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise DisasterRecoveryError("protocol_version_invalid")
    if value.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise DisasterRecoveryError("protocol_policy_drift")
    if _protocol_digest(value) != value["protocol_sha256"]:
        raise DisasterRecoveryError("protocol_digest_invalid")
    if not isinstance(value.get("bindings"), dict) or set(value["bindings"]) != {
        "crash_consistency",
        "execution_plan",
        "launch_control",
        "provider_capture_addendum",
        "provider_ingest",
        "resource_accounting",
        "sharded_execution",
    }:
        raise DisasterRecoveryError("protocol_schema_invalid")
    if value["population"] != {
        "query_count": 1000,
        "query_order_sha256": "1d310756a0a5115ea33aec23939a4e9867302c85750448810252f174e5e74563",
        "shard_count": 20,
    }:
        raise DisasterRecoveryError("protocol_population_drift")
    for name, binding in value["bindings"].items():
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)
        ):
            raise DisasterRecoveryError("protocol_schema_invalid")
        relative = _safe_relative(str(binding.get("path") or ""))
        bound = repository_root / relative
        if not bound.is_file():
            raise DisasterRecoveryNotEligible(f"bound_input_missing:{name}")
        if sha256_file(bound) != binding.get("sha256"):
            raise DisasterRecoveryError(f"bound_input_hash_drift:{name}")
    plan = _read_object(
        repository_root / value["bindings"]["execution_plan"]["path"]
    )
    if (
        plan.get("plan_sha256")
        != value["bindings"]["execution_plan"]["embedded_plan_sha256"]
        or plan.get("population", {}).get("count") != 1000
        or plan.get("sharding", {}).get("shard_count") != 20
    ):
        raise DisasterRecoveryError("execution_plan_binding_drift")
    return value


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if completed.returncode != 0:
        raise DisasterRecoveryNotEligible("git_identity_unavailable")
    return completed.stdout.strip()


def _object_path(backup_root: Path, digest: str) -> Path:
    return backup_root / OBJECTS_DIRECTORY / digest[:2] / digest


def _manifest_path(backup_root: Path, backup_id: str) -> Path:
    return backup_root / MANIFESTS_DIRECTORY / f"{backup_id}.json"


def _latest(backup_root: Path) -> str | None:
    path = backup_root / LATEST_FILE
    if not path.is_file():
        return None
    value = _read_object(path)
    backup_id = str(value.get("backup_id") or "")
    if stable_hash({"backup_id": backup_id}) != value.get("latest_sha256"):
        raise DisasterRecoveryError("latest_pointer_digest_invalid")
    return backup_id


def _load_manifest(backup_root: Path, backup_id: str) -> BackupManifest:
    try:
        return BackupManifest.model_validate(_read_object(_manifest_path(backup_root, backup_id)))
    except ValidationError as exc:
        raise DisasterRecoveryError("backup_manifest_invalid") from exc


def _load_recovery_state(run_root: Path) -> RecoveryState:
    try:
        return RecoveryState.model_validate(
            _read_object(run_root / "authority/recovery_state.json")
        )
    except ValidationError as exc:
        raise DisasterRecoveryError("recovery_state_invalid") from exc


def _load_plan(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    return _read_object(root / protocol["bindings"]["execution_plan"]["path"])


def _load_launch_protocol(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    return load_launch_protocol(root / protocol["bindings"]["launch_control"]["path"])


def _validate_authority(
    run_root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[RecoveryState, dict[str, Any], dict[str, Any], OperationAuditLog]:
    try:
        prepared = _read_object(run_root / "authority/prepared.json")
        authorization = _read_object(run_root / "authority/authorization.json")
        audit_value = _read_object(run_root / "authority/operation_audit.json")
        audit = OperationAuditLog.model_validate(audit_value)
    except ValidationError as exc:
        raise DisasterRecoveryError("launch_audit_invalid") from exc
    launch_protocol = _load_launch_protocol(repository_root, protocol)
    try:
        validate_authorization(prepared, authorization, launch_protocol)
        validate_launch_evidence(prepared, authorization, audit_value, launch_protocol)
    except LaunchControlError as exc:
        raise DisasterRecoveryError("launch_authority_invalid") from exc
    state = _load_recovery_state(run_root)
    if (
        state.authorization_sha256 != authorization.get("authorization_sha256")
        or state.plan_sha256
        != protocol["bindings"]["execution_plan"]["embedded_plan_sha256"]
    ):
        raise DisasterRecoveryError("recovery_authority_binding_mismatch")
    completed_from_audit = sorted(
        int(item.shard_index)
        for item in audit.entries
        if item.event == "shard_completed" and item.shard_index is not None
    )
    if completed_from_audit != state.completed_shards:
        raise DisasterRecoveryError("audit_completed_shard_mismatch")
    selected_from_audit = _replay_operation_audit(prepared, authorization, audit)
    if selected_from_audit.selected_attempts != {
        int(key): value for key, value in state.selected_attempts.items()
    }:
        raise DisasterRecoveryError("audit_attempt_selection_mismatch")
    aggregate_events = [
        entry for entry in audit.entries if entry.event == "aggregate_requested"
    ]
    if (state.aggregate_state == "completed") != (len(aggregate_events) == 1):
        raise DisasterRecoveryError("audit_aggregate_state_mismatch")
    return state, prepared, authorization, audit


def _selected_attempt_path(run_root: Path, shard: int, attempt_id: str) -> Path:
    expected_prefix = f"shard-{shard:02d}-attempt-"
    if not attempt_id.startswith(expected_prefix):
        raise DisasterRecoveryError("attempt_identity_shard_mismatch")
    return run_root / "shards" / f"shard-{shard:02d}" / attempt_id


def _validate_completed_attempt(
    run_root: Path,
    shard: int,
    attempt_id: str,
    expected_queries: Sequence[str],
    expected_generation: int,
) -> tuple[Any, ResourceLedgerV1, ProviderIngestBundle]:
    attempt_root = _selected_attempt_path(run_root, shard, attempt_id)
    store = BenchmarkRunCommitStore(attempt_root)
    try:
        committed = store.load_latest()
    except CrashConsistencyError as exc:
        raise DisasterRecoveryError("committed_generation_invalid") from exc
    if (
        committed.status != "completed"
        or committed.generation != expected_generation
        or list(committed.expected_query_ids) != list(expected_queries)
        or [str(row.get("case_id")) for row in committed.records]
        != list(expected_queries)
    ):
        raise DisasterRecoveryError("completed_attempt_scope_mismatch")
    report_root = committed.generation_path
    if set(REPORT_FILES) - set(committed.reports):
        raise DisasterRecoveryError("authoritative_report_missing")
    try:
        ledger = ResourceLedgerV1.model_validate_json(
            (report_root / "resource_ledger.json").read_text(encoding="utf-8")
        )
        provider = ProviderIngestBundle.model_validate_json(
            (report_root / "provider_ingest_provenance.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise DisasterRecoveryError("ledger_or_provider_report_invalid") from exc
    ledger_report = validate_resource_ledger(ledger)
    provider_report = verify_capture_bundle(
        report_root / "provider_ingest_provenance.json",
        report_root / "provider_ingest_raw.tar",
        resource_ledger_path=report_root / "resource_ledger.json",
    )
    if ledger_report["status"] != "passed" or provider_report["status"] != "passed":
        raise DisasterRecoveryError("ledger_or_provider_conservation_failed")
    if any(
        item.checkpoint_generation != committed.generation
        for item in ledger.queries
    ) or provider.checkpoint_generation != committed.generation:
        raise DisasterRecoveryError("generation_authority_mismatch")
    resource_queries = [opaque_resource_identity("query", value) for value in expected_queries]
    if ledger.expected_query_identities != resource_queries:
        raise DisasterRecoveryError("resource_query_coverage_mismatch")
    return committed, ledger, provider


def _validate_aggregate(
    run_root: Path,
    state: RecoveryState,
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    aggregate_path = run_root / "aggregate/aggregate.json"
    if state.aggregate_state == "pending":
        if aggregate_path.exists():
            raise DisasterRecoveryError("premature_aggregate_present")
        return None
    aggregate = _read_object(aggregate_path)
    payload = dict(aggregate)
    digest = payload.pop("aggregate_sha256", None)
    if (
        aggregate.get("contract") != AGGREGATE_CONTRACT
        or digest != stable_hash(payload)
        or digest != state.aggregate_sha256
        or aggregate.get("query_count") != 1000
        or aggregate.get("selected_attempts") != state.selected_attempts
        or aggregate.get("operation_audit_sha256")
        != _read_object(run_root / "authority/operation_audit.json").get(
            "audit_sha256"
        )
        or [item.get("query_identity") for item in aggregate.get("records", [])]
        != list(plan["population"]["identities"])
    ):
        raise DisasterRecoveryError("aggregate_authority_invalid")
    return aggregate


def validate_run_root(
    run_root: Path,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    state, prepared, authorization, audit = _validate_authority(
        run_root, repository_root, protocol
    )
    plan = _load_plan(repository_root, protocol)
    request_count = 0
    generations: dict[str, int] = {}
    for shard in state.completed_shards:
        attempt_id = state.selected_attempts[str(shard)]
        shard_plan = plan["sharding"]["shards"][shard]
        committed, ledger, _provider = _validate_completed_attempt(
            run_root,
            shard,
            attempt_id,
            shard_plan["query_identities"],
            state.generation_by_shard[str(shard)],
        )
        generations[str(shard)] = committed.generation
        requests = ledger.totals.api_request_count
        if requests.state != "known":
            raise DisasterRecoveryError("request_count_not_authoritative")
        request_count += int(requests.value or 0)
    if request_count != state.adapter_call_count:
        raise DisasterRecoveryError("adapter_call_count_mismatch")
    aggregate = _validate_aggregate(run_root, state, plan)
    return {
        "state": state,
        "prepared": prepared,
        "authorization": authorization,
        "audit": audit,
        "aggregate": aggregate,
        "generation_by_shard": generations,
        "request_count": request_count,
    }


def _collect_authoritative_files(
    run_root: Path,
    validation: Mapping[str, Any],
) -> list[BackupFile]:
    state: RecoveryState = validation["state"]
    paths = [run_root / value for value in AUTHORITY_FILES]
    for shard in state.completed_shards:
        attempt = state.selected_attempts[str(shard)]
        attempt_root = _selected_attempt_path(run_root, shard, attempt)
        generations = attempt_root / STORE_DIRECTORY / "generations"
        for generation in sorted(generations.iterdir(), key=lambda value: value.name):
            if not generation.is_dir() or not (generation / "COMMITTED").is_file():
                continue
            for path in sorted(generation.rglob("*")):
                if path.is_file():
                    paths.append(path)
    if state.aggregate_state == "completed":
        paths.append(run_root / "aggregate/aggregate.json")
    files: list[BackupFile] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise DisasterRecoveryError("authoritative_file_unavailable")
        relative = _safe_relative(path.relative_to(run_root).as_posix())
        files.append(
            BackupFile(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    files.sort(key=lambda value: value.path)
    if len({item.path for item in files}) != len(files):
        raise DisasterRecoveryError("authoritative_file_duplicated")
    return files


def _restore_point(validation: Mapping[str, Any]) -> RestorePoint:
    state: RecoveryState = validation["state"]
    audit: OperationAuditLog = validation["audit"]
    return RestorePoint(
        query_cursor=state.adapter_call_count,
        completed_shards=state.completed_shards,
        selected_attempts=state.selected_attempts,
        generation_by_shard=state.generation_by_shard,
        aggregate_state=state.aggregate_state,
        operation_audit_sha256=audit.audit_sha256,
        resource_request_count=validation["request_count"],
    )


def create_backup(
    run_root: Path,
    backup_root: Path,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    parent_backup_id: str | None = None,
    fault: Literal["after_objects", "after_manifest"] | None = None,
) -> dict[str, Any]:
    validation = validate_run_root(
        run_root, repository_root=repository_root, protocol=protocol
    )
    state: RecoveryState = validation["state"]
    latest = _latest(backup_root)
    parent = parent_backup_id if parent_backup_id is not None else latest
    if parent is not None:
        _load_manifest(backup_root, parent)
        if latest is not None and parent != latest:
            raise DisasterRecoveryError("backup_parent_is_not_latest")
    files = _collect_authoritative_files(run_root, validation)
    new_object_count = 0
    for item in files:
        source = run_root / item.path
        target = _object_path(backup_root, item.sha256)
        if target.exists():
            if target.is_symlink() or sha256_file(target) != item.sha256:
                raise DisasterRecoveryError("backup_object_collision")
            continue
        durable_atomic_write_bytes(
            target,
            source.read_bytes(),
            temporary_suffix="backup-object",
        )
        new_object_count += 1
    if fault == "after_objects":
        raise DisasterRecoveryError("injected_backup_interruption")
    payload: dict[str, Any] = {
        "contract": BACKUP_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "backup_id": "",
        "parent_backup_id": parent,
        "run_identity": state.run_identity,
        "plan_sha256": state.plan_sha256,
        "authorization_sha256": state.authorization_sha256,
        "files": [item.model_dump(mode="json") for item in files],
        "restore_point": _restore_point(validation).model_dump(mode="json"),
        "score_scope": "disaster_recovery_only_not_quality_or_official_score",
    }
    backup_payload = dict(payload)
    backup_payload.pop("backup_id")
    payload["backup_id"] = stable_hash(backup_payload)
    payload["manifest_sha256"] = stable_hash(payload)
    manifest = BackupManifest.model_validate(payload)
    manifest_path = _manifest_path(backup_root, manifest.backup_id)
    if manifest_path.exists():
        existing = _load_manifest(backup_root, manifest.backup_id)
        if existing != manifest:
            raise DisasterRecoveryError("backup_identity_collision")
    else:
        durable_atomic_write_bytes(
            manifest_path,
            canonical_json(manifest.model_dump(mode="json")),
            temporary_suffix="backup-manifest",
        )
    if fault == "after_manifest":
        raise DisasterRecoveryError("injected_backup_interruption")
    latest_value = {
        "backup_id": manifest.backup_id,
        "latest_sha256": stable_hash({"backup_id": manifest.backup_id}),
    }
    durable_atomic_write_bytes(
        backup_root / LATEST_FILE,
        canonical_json(latest_value),
        temporary_suffix="backup-latest",
    )
    return {
        "status": "recovery_controls_ready",
        "exit_code": EXIT_READY,
        "backup_id": manifest.backup_id,
        "parent_backup_id": manifest.parent_backup_id,
        "file_count": len(manifest.files),
        "new_object_count": new_object_count,
        "query_cursor": manifest.restore_point.query_cursor,
        "completed_shard_count": len(manifest.restore_point.completed_shards),
        "aggregate_state": manifest.restore_point.aggregate_state,
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }


def _validate_parent_chain(
    backup_root: Path,
    manifest: BackupManifest,
) -> list[str]:
    chain: list[str] = []
    observed: set[str] = set()
    current: BackupManifest | None = manifest
    while current is not None:
        if current.backup_id in observed:
            raise DisasterRecoveryError("backup_parent_cycle")
        observed.add(current.backup_id)
        chain.append(current.backup_id)
        if current.parent_backup_id is None:
            current = None
        else:
            current = _load_manifest(backup_root, current.parent_backup_id)
    return chain


def _materialize_manifest(
    manifest: BackupManifest,
    backup_root: Path,
    target: Path,
) -> None:
    for item in manifest.files:
        object_path = _object_path(backup_root, item.sha256)
        if (
            not object_path.is_file()
            or object_path.is_symlink()
            or object_path.stat().st_size != item.size_bytes
            or sha256_file(object_path) != item.sha256
        ):
            raise DisasterRecoveryError("backup_object_missing_or_tampered")
        destination = target / _safe_relative(item.path)
        durable_atomic_write_bytes(
            destination,
            object_path.read_bytes(),
            temporary_suffix="restore-file",
        )


def verify_backup(
    backup_root: Path,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    backup_id: str | None = None,
    require_latest: bool = True,
) -> dict[str, Any]:
    latest = _latest(backup_root)
    if latest is None:
        raise DisasterRecoveryNotEligible("backup_latest_missing")
    selected = backup_id or latest
    if require_latest and selected != latest:
        raise DisasterRecoveryError("backup_rollback_forbidden")
    manifest = _load_manifest(backup_root, selected)
    if (
        manifest.protocol_sha256 != protocol["protocol_sha256"]
        or manifest.plan_sha256
        != protocol["bindings"]["execution_plan"]["embedded_plan_sha256"]
    ):
        raise DisasterRecoveryError("backup_protocol_or_plan_drift")
    chain = _validate_parent_chain(backup_root, manifest)
    with tempfile.TemporaryDirectory(prefix="formal-run-backup-verify-") as temporary:
        restored = Path(temporary) / "run"
        _materialize_manifest(manifest, backup_root, restored)
        validation = validate_run_root(
            restored, repository_root=repository_root, protocol=protocol
        )
    return {
        "status": "recovery_controls_ready",
        "exit_code": EXIT_READY,
        "backup_id": manifest.backup_id,
        "parent_chain_length": len(chain),
        "file_count": len(manifest.files),
        "query_cursor": manifest.restore_point.query_cursor,
        "completed_shard_count": len(manifest.restore_point.completed_shards),
        "request_count": validation["request_count"],
        "aggregate_state": manifest.restore_point.aggregate_state,
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }


def restore_backup(
    backup_root: Path,
    target: Path,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    backup_id: str | None = None,
) -> dict[str, Any]:
    if target.exists() and (not target.is_dir() or next(target.iterdir(), None) is not None):
        raise DisasterRecoveryError("restore_target_not_empty")
    latest = _latest(backup_root)
    if latest is None:
        raise DisasterRecoveryNotEligible("backup_latest_missing")
    selected = backup_id or latest
    if selected != latest:
        raise DisasterRecoveryError("backup_rollback_forbidden")
    verify_backup(
        backup_root,
        repository_root=repository_root,
        protocol=protocol,
        backup_id=selected,
    )
    manifest = _load_manifest(backup_root, selected)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.restore.lock"
    stage = target.parent / f".{target.name}.restore-{selected}.pending"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if stage.exists():
            raise DisasterRecoveryError("stale_restore_stage_present")
        stage.mkdir()
        _materialize_manifest(manifest, backup_root, stage)
        validation = validate_run_root(
            stage, repository_root=repository_root, protocol=protocol
        )
        if target.exists():
            target.rmdir()
        os.replace(stage, target)
        _fsync_directory(target.parent)
    except FileExistsError as exc:
        raise DisasterRecoveryError("concurrent_restorer_rejected") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)
    authorization = validation["authorization"]
    current_head = _git_head(repository_root)
    execution_allowed = authorization.get("observed_head") == current_head
    return {
        "status": (
            "recovery_controls_ready"
            if execution_allowed
            else "restored_read_only_commit_mismatch"
        ),
        "exit_code": EXIT_READY,
        "backup_id": selected,
        "query_cursor": validation["state"].adapter_call_count,
        "completed_shard_count": len(validation["state"].completed_shards),
        "execution_allowed": execution_allowed,
        "resume_requires_new_authorization": not execution_allowed,
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }


def _replay_operation_audit(
    prepared: Mapping[str, Any],
    authorization: Mapping[str, Any],
    audit: OperationAuditLog,
) -> LaunchOperationMachine:
    machine = LaunchOperationMachine(prepared, authorization)
    for entry in audit.entries:
        if entry.event == "authorized":
            machine.authorize()
        elif entry.event == "started":
            machine.start()
        elif entry.event == "paused":
            machine.pause()
        elif entry.event == "resumed":
            machine.resume()
        elif entry.event == "shard_failed":
            machine.fail_shard(int(entry.shard_index))
        elif entry.event == "attempt_superseded":
            machine.supersede(int(entry.shard_index))
        elif entry.event == "shard_completed":
            machine.complete_shard(int(entry.shard_index), entry.attempt_id)
        elif entry.event == "aggregate_requested":
            machine.aggregate()
        elif entry.event == "cancelled":
            machine.cancel()
        elif entry.event == "revoked":
            machine.revoke()
        else:
            raise DisasterRecoveryError("unsupported_launch_audit_event")
    if machine.audit_log().audit_sha256 != audit.audit_sha256:
        raise DisasterRecoveryError("launch_audit_replay_drift")
    return machine


def _initial_attempt_statuses(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        attempt["attempt_id"]: "not_started"
        for shard in plan["sharding"]["shards"]
        for attempt in shard["attempts"]
    }


def _state_payload(
    *,
    run_identity: str,
    plan_sha256: str,
    authorization_sha256: str,
    machine: LaunchOperationMachine,
    attempt_statuses: Mapping[str, str],
    generation_by_shard: Mapping[str, int],
    adapter_call_count: int,
    aggregate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": STATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "run_identity": run_identity,
        "plan_sha256": plan_sha256,
        "authorization_sha256": authorization_sha256,
        "selected_attempts": {
            str(key): value for key, value in sorted(machine.selected_attempts.items())
        },
        "attempt_statuses": dict(sorted(attempt_statuses.items())),
        "completed_shards": sorted(machine.completed_shards),
        "generation_by_shard": dict(sorted(generation_by_shard.items())),
        "adapter_call_count": adapter_call_count,
        "aggregate_state": "completed" if aggregate is not None else "pending",
        "aggregate_sha256": aggregate.get("aggregate_sha256") if aggregate else None,
    }
    payload["state_sha256"] = stable_hash(payload)
    return payload


def _write_authority(
    run_root: Path,
    *,
    prepared: Mapping[str, Any],
    authorization: Mapping[str, Any],
    machine: LaunchOperationMachine,
    run_identity: str,
    plan_sha256: str,
    attempt_statuses: Mapping[str, str],
    generation_by_shard: Mapping[str, int],
    adapter_call_count: int,
    aggregate: Mapping[str, Any] | None,
) -> None:
    write_json(run_root / "authority/prepared.json", prepared)
    write_json(run_root / "authority/authorization.json", authorization)
    write_json(
        run_root / "authority/operation_audit.json",
        machine.audit_log().model_dump(mode="json"),
    )
    write_json(
        run_root / "authority/recovery_state.json",
        _state_payload(
            run_identity=run_identity,
            plan_sha256=plan_sha256,
            authorization_sha256=str(authorization["authorization_sha256"]),
            machine=machine,
            attempt_statuses=attempt_statuses,
            generation_by_shard=generation_by_shard,
            adapter_call_count=adapter_call_count,
            aggregate=aggregate,
        ),
    )


def _build_query_ledger(
    *,
    run_identity: str,
    query_identity: str,
    attempt_identity: str,
    generation: int,
    manifest_identity: str,
) -> Any:
    observer = ResourceLedgerObserver(_FixtureBudget())
    observer.observe_semantic_event(
        "connector_started",
        {
            "query_index": 0,
            "source": "openalex",
            "adapted_query": "opaque",
        },
    )
    observer.observe_semantic_event(
        "connector_completed",
        {
            "query_index": 0,
            "source": "openalex",
            "adapted_query": "opaque",
            "request_count": 1,
            "retry_count": 0,
            "returned_count": 0,
            "cache_hit": False,
            "error_message": None,
        },
    )
    observer.observe_budget_event(
        "search_round_consumed", {"completed_search_rounds": 1}
    )
    observer.observe_budget_event("candidate_budget_observed", {"candidate_count": 0})
    observer.observe_budget_event(
        "budget_finalized",
        {
            "completed_search_rounds": 1,
            "candidate_count": 0,
            "elapsed_seconds": 0.0,
            "stop_reasons": [],
        },
    )
    return observer.build_query_ledger(
        run_identity=run_identity,
        query_identity=query_identity,
        attempt_identity=attempt_identity,
        checkpoint_generation=generation,
        manifest_identity=manifest_identity,
        terminal_status="succeeded",
    )


def _execute_shard(
    run_root: Path,
    shard_plan: Mapping[str, Any],
    attempt_id: str,
    counter: _AdapterCounter,
) -> int:
    shard = int(shard_plan["shard_index"])
    attempt_root = _selected_attempt_path(run_root, shard, attempt_id)
    queries = list(shard_plan["query_identities"])
    resource_queries = [opaque_resource_identity("query", value) for value in queries]
    run_identity = opaque_resource_identity("run", f"{shard}:{attempt_id}")
    resource_attempt = opaque_resource_identity("attempt", attempt_id)
    final_generation = len(queries) + 2
    manifest_identity = stable_hash(
        {
            "manifest_kind": "benchmark_run_commit_v1",
            "run_identity": run_identity,
            "shard": shard,
            "attempt_id": attempt_id,
            "generation": final_generation,
        }
    )
    store = BenchmarkRunCommitStore(attempt_root)
    store.initialize(
        run_id=run_identity,
        expected_query_ids=queries,
        config={
            "case_ids": queries,
            "fixture_only": True,
            "shard_index": shard,
            "attempt_id": attempt_id,
        },
        dataset_report={"dataset": "synthetic_full1000_recovery_fixture"},
    )
    query_ledgers = []
    captured = []
    for query, resource_query in zip(queries, resource_queries, strict=True):
        counter.call(query)
        query_ledger = _build_query_ledger(
            run_identity=run_identity,
            query_identity=resource_query,
            attempt_identity=resource_attempt,
            generation=final_generation,
            manifest_identity=manifest_identity,
        )
        adapter_operation = next(
            item
            for item in query_ledger.operations
            if item.operation_type == "adapter_call"
        )
        envelope = create_envelope(
            run_identity=run_identity,
            query_identity=resource_query,
            source="openalex",
            attempt_identity=resource_attempt,
            request_sequence=0,
            resource_operation_identity=adapter_operation.operation_identity,
            checkpoint_generation=final_generation,
            manifest_identity=manifest_identity,
            parser_name="openalex_search",
            raw_bytes=b'{"results":[]}',
            http_status=200,
            content_type="application/json",
            encoding=EncodingMetadata(state="known", value="utf-8"),
            compression="identity",
            terminal_state="success",
        )
        captured.append(envelope)
        query_ledgers.append(query_ledger)
        store.commit_record(
            {
                "case_id": query,
                "status": "succeeded",
                "final_returned": [
                    {
                        "identity": stable_hash({"paper": query}),
                        "rank": 1,
                    }
                ],
            }
        )
    ledger = build_run_ledger(
        query_ledgers,
        run_identity=run_identity,
        manifest_identity=manifest_identity,
        expected_query_identities=resource_queries,
    )
    if validate_resource_ledger(ledger)["status"] != "passed":
        raise DisasterRecoveryError("fixture_resource_ledger_invalid")
    with tempfile.TemporaryDirectory(prefix="recovery-shard-reports-") as temporary:
        report_root = Path(temporary)
        ledger_path = report_root / "resource_ledger.json"
        provider_path = report_root / "provider_ingest_provenance.json"
        archive_path = report_root / "provider_ingest_raw.tar"
        write_json(ledger_path, ledger.model_dump(mode="json"))
        write_capture_bundle(
            provider_path,
            archive_path,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            checkpoint_generation=final_generation,
            resource_ledger_path=ledger_path,
            captured=captured,
        )
        committed = store.commit_completion(
            {
                "resource_ledger.json": ledger_path.read_bytes(),
                "provider_ingest_provenance.json": provider_path.read_bytes(),
                "provider_ingest_raw.tar": archive_path.read_bytes(),
            }
        )
    if committed.generation != final_generation:
        raise DisasterRecoveryError("fixture_generation_prediction_drift")
    return committed.generation


def _aggregate(
    run_root: Path,
    plan: Mapping[str, Any],
    machine: LaunchOperationMachine,
) -> dict[str, Any]:
    records_by_query: dict[str, dict[str, Any]] = {}
    ledger_hashes: dict[str, str] = {}
    request_count = 0
    for shard in range(20):
        attempt_id = machine.selected_attempts[shard]
        committed = BenchmarkRunCommitStore(
            _selected_attempt_path(run_root, shard, attempt_id)
        ).load_latest()
        for row in committed.records:
            records_by_query[str(row["case_id"])] = {
                "query_identity": str(row["case_id"]),
                "final_returned": row["final_returned"],
                "terminal_status": str(row["status"]),
            }
        ledger_path = committed.generation_path / "resource_ledger.json"
        ledger = ResourceLedgerV1.model_validate_json(
            ledger_path.read_text(encoding="utf-8")
        )
        ledger_hashes[str(shard)] = sha256_file(ledger_path)
        request_count += int(ledger.totals.api_request_count.value or 0)
    ordered = [
        records_by_query[value] for value in plan["population"]["identities"]
    ]
    payload: dict[str, Any] = {
        "contract": AGGREGATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "query_count": len(ordered),
        "records": ordered,
        "selected_attempts": {
            str(key): value for key, value in sorted(machine.selected_attempts.items())
        },
        "resource_ledger_sha256_by_shard": ledger_hashes,
        "resource_request_count": request_count,
        "operation_audit_sha256": machine.audit_log().audit_sha256,
        "top20_delivery_sha256": stable_hash(
            [item["final_returned"] for item in ordered]
        ),
        "score_scope": "delivery_equivalence_only_not_quality_or_official_score",
    }
    payload["aggregate_sha256"] = stable_hash(payload)
    write_json(run_root / "aggregate/aggregate.json", payload)
    return payload


def _initialize_fake_run(
    run_root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    LaunchOperationMachine,
    dict[str, Any],
    dict[str, str],
    dict[str, int],
    str,
]:
    plan = _load_plan(repository_root, protocol)
    launch_protocol = _load_launch_protocol(repository_root, protocol)
    prepared = build_preparation(
        repository_root,
        launch_protocol,
        authoritative_root=run_root,
        check_freshness=False,
    )
    authorization = build_authorization(prepared, launch_protocol)
    machine = LaunchOperationMachine(prepared, authorization)
    machine.authorize()
    machine.start()
    statuses = _initial_attempt_statuses(plan)
    generations: dict[str, int] = {}
    run_identity = stable_hash(
        {
            "fixture": PROTOCOL,
            "plan_sha256": plan["plan_sha256"],
            "authorization_sha256": authorization["authorization_sha256"],
        }
    )
    _write_authority(
        run_root,
        prepared=prepared,
        authorization=authorization,
        machine=machine,
        run_identity=run_identity,
        plan_sha256=plan["plan_sha256"],
        attempt_statuses=statuses,
        generation_by_shard=generations,
        adapter_call_count=0,
        aggregate=None,
    )
    return (
        prepared,
        authorization,
        machine,
        plan,
        statuses,
        generations,
        run_identity,
    )


def _resume_fake_run(
    run_root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
    counter: _AdapterCounter,
) -> dict[str, Any]:
    validation = validate_run_root(
        run_root, repository_root=repository_root, protocol=protocol
    )
    state: RecoveryState = validation["state"]
    machine = _replay_operation_audit(
        validation["prepared"], validation["authorization"], validation["audit"]
    )
    plan = _load_plan(repository_root, protocol)
    statuses = dict(state.attempt_statuses)
    generations = dict(state.generation_by_shard)
    for shard in range(20):
        if shard in machine.completed_shards:
            continue
        if shard == 15 and machine.selected_attempts[shard].endswith("attempt-0"):
            machine.fail_shard(shard)
            statuses[machine.selected_attempts[shard]] = "failed"
            machine.supersede(shard)
        attempt = machine.selected_attempts[shard]
        generation = _execute_shard(
            run_root, plan["sharding"]["shards"][shard], attempt, counter
        )
        statuses[attempt] = "completed"
        generations[str(shard)] = generation
        machine.complete_shard(shard)
    machine.aggregate()
    aggregate = _aggregate(run_root, plan, machine)
    _write_authority(
        run_root,
        prepared=validation["prepared"],
        authorization=validation["authorization"],
        machine=machine,
        run_identity=state.run_identity,
        plan_sha256=plan["plan_sha256"],
        attempt_statuses=statuses,
        generation_by_shard=generations,
        adapter_call_count=counter.total,
        aggregate=aggregate,
    )
    return {
        "aggregate": aggregate,
        "audit_sha256": machine.audit_log().audit_sha256,
        "request_count": counter.total,
    }


def _run_uninterrupted_control(
    run_root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    (
        prepared,
        authorization,
        machine,
        plan,
        statuses,
        generations,
        run_identity,
    ) = _initialize_fake_run(run_root, repository_root, protocol)
    counter = _AdapterCounter()
    for shard in range(20):
        if shard == 15:
            machine.fail_shard(shard)
            statuses[machine.selected_attempts[shard]] = "failed"
            machine.supersede(shard)
        attempt = machine.selected_attempts[shard]
        generations[str(shard)] = _execute_shard(
            run_root, plan["sharding"]["shards"][shard], attempt, counter
        )
        statuses[attempt] = "completed"
        machine.complete_shard(shard)
    machine.aggregate()
    aggregate = _aggregate(run_root, plan, machine)
    _write_authority(
        run_root,
        prepared=prepared,
        authorization=authorization,
        machine=machine,
        run_identity=run_identity,
        plan_sha256=plan["plan_sha256"],
        attempt_statuses=statuses,
        generation_by_shard=generations,
        adapter_call_count=counter.total,
        aggregate=aggregate,
    )
    return {
        "aggregate": aggregate,
        "audit_sha256": machine.audit_log().audit_sha256,
        "request_count": counter.total,
    }


def simulate_disaster(
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="formal-run-disaster-") as temporary:
        root = Path(temporary)
        source = root / "primary"
        backup_root = root / "offsite"
        restored = root / "restored"
        control = root / "control"
        (
            prepared,
            authorization,
            machine,
            plan,
            statuses,
            generations,
            run_identity,
        ) = _initialize_fake_run(source, repository_root, protocol)
        counter = _AdapterCounter()
        for shard in range(10):
            attempt = machine.selected_attempts[shard]
            generations[str(shard)] = _execute_shard(
                source, plan["sharding"]["shards"][shard], attempt, counter
            )
            statuses[attempt] = "completed"
            machine.complete_shard(shard)
        _write_authority(
            source,
            prepared=prepared,
            authorization=authorization,
            machine=machine,
            run_identity=run_identity,
            plan_sha256=plan["plan_sha256"],
            attempt_statuses=statuses,
            generation_by_shard=generations,
            adapter_call_count=counter.total,
            aggregate=None,
        )
        first = create_backup(
            source,
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
        )
        shutil.rmtree(source)
        restored_report = restore_backup(
            backup_root,
            restored,
            repository_root=repository_root,
            protocol=protocol,
        )
        recovered = _resume_fake_run(
            restored, repository_root, protocol, counter
        )
        second = create_backup(
            restored,
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
            parent_backup_id=first["backup_id"],
        )
        final_backup = verify_backup(
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
        )
        uninterrupted = _run_uninterrupted_control(
            control, repository_root, protocol
        )
        equivalence = {
            "aggregate": recovered["aggregate"] == uninterrupted["aggregate"],
            "audit": recovered["audit_sha256"] == uninterrupted["audit_sha256"],
            "resource_requests": recovered["request_count"]
            == uninterrupted["request_count"]
            == 1000,
            "top20_delivery": recovered["aggregate"]["top20_delivery_sha256"]
            == uninterrupted["aggregate"]["top20_delivery_sha256"],
        }
        if not all(equivalence.values()):
            raise DisasterRecoveryError("recovered_run_not_equivalent")
        scenarios = _deterministic_fault_matrix(
            repository_root,
            protocol,
            restored,
            backup_root,
            first["backup_id"],
            second["backup_id"],
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "recovery_controls_ready",
            "exit_code": EXIT_READY,
            "query_count": 1000,
            "shard_count": 20,
            "partial_backup_query_cursor": first["query_cursor"],
            "restored_query_cursor": restored_report["query_cursor"],
            "final_request_count": recovered["request_count"],
            "duplicate_request_count": 0,
            "parent_chain_length": final_backup["parent_chain_length"],
            "replacement_shard": 15,
            "equivalence": equivalence,
            "scenario_count": len(scenarios),
            "scenarios": [
                {"scenario": name, "blocked": scenarios[name]}
                for name in sorted(scenarios)
            ],
            "fixture_only": True,
            "execution": EXECUTION_ZERO,
            "formal_validation_complete": False,
        }


def _deterministic_fault_matrix(
    repository_root: Path,
    protocol: Mapping[str, Any],
    run_root: Path,
    backup_root: Path,
    old_backup_id: str,
    latest_backup_id: str,
) -> dict[str, bool]:
    scenarios: dict[str, bool] = {}
    try:
        verify_backup(
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
            backup_id=old_backup_id,
        )
    except DisasterRecoveryError:
        scenarios["old_backup_rollback"] = True
    latest_before_interruption = _latest(backup_root)
    try:
        create_backup(
            run_root,
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
            fault="after_objects",
        )
    except DisasterRecoveryError:
        scenarios["backup_interruption"] = (
            _latest(backup_root) == latest_before_interruption
        )
    duplicate_counter = _AdapterCounter()
    duplicate_counter.call("opaque-committed-query")
    try:
        duplicate_counter.call("opaque-committed-query")
    except DisasterRecoveryError:
        scenarios["duplicate_charge_after_resume"] = True
    with tempfile.TemporaryDirectory(prefix="recovery-fault-matrix-") as temporary:
        root = Path(temporary)
        nonempty = root / "nonempty"
        nonempty.mkdir()
        (nonempty / "occupied").write_bytes(b"x")
        try:
            restore_backup(
                backup_root,
                nonempty,
                repository_root=repository_root,
                protocol=protocol,
            )
        except DisasterRecoveryError:
            scenarios["non_empty_target"] = True
        lock_target = root / "locked"
        lock = root / ".locked.restore.lock"
        lock.write_bytes(b"held")
        try:
            restore_backup(
                backup_root,
                lock_target,
                repository_root=repository_root,
                protocol=protocol,
            )
        except DisasterRecoveryError:
            scenarios["concurrent_restorer"] = True
        finally:
            lock.unlink(missing_ok=True)

        missing_root = root / "missing-member"
        shutil.copytree(backup_root, missing_root)
        missing_manifest = _load_manifest(missing_root, latest_backup_id)
        _object_path(missing_root, missing_manifest.files[0].sha256).unlink()
        scenarios["missing_member"] = _verification_is_rejected(
            missing_root, repository_root, protocol
        )

        tamper_root = root / "hash-tamper"
        shutil.copytree(backup_root, tamper_root)
        tamper_manifest = _load_manifest(tamper_root, latest_backup_id)
        tamper_object = _object_path(tamper_root, tamper_manifest.files[0].sha256)
        tamper_object.write_bytes(tamper_object.read_bytes() + b"tamper")
        scenarios["hash_tamper"] = _verification_is_rejected(
            tamper_root, repository_root, protocol
        )

        parent_root = root / "parent-break"
        shutil.copytree(backup_root, parent_root)
        parent_manifest = _load_manifest(parent_root, latest_backup_id)
        if parent_manifest.parent_backup_id is None:
            raise DisasterRecoveryError("fixture_parent_backup_missing")
        _manifest_path(parent_root, parent_manifest.parent_backup_id).unlink()
        scenarios["parent_chain_break_or_cycle"] = _verification_is_rejected(
            parent_root, repository_root, protocol
        )

        mixed_root = root / "mixed-generation"
        shutil.copytree(backup_root, mixed_root)
        _repackage_backup_file(
            mixed_root,
            latest_backup_id,
            "authority/recovery_state.json",
            _mutate_generation_reference,
        )
        scenarios["mixed_generation"] = _verification_is_rejected(
            mixed_root, repository_root, protocol
        )

        audit_root = root / "audit-truncation"
        shutil.copytree(backup_root, audit_root)
        _repackage_backup_file(
            audit_root,
            latest_backup_id,
            "authority/operation_audit.json",
            _truncate_operation_audit,
        )
        scenarios["audit_chain_truncation"] = _verification_is_rejected(
            audit_root, repository_root, protocol
        )

        raw_root = root / "raw-missing"
        shutil.copytree(backup_root, raw_root)
        latest_raw = _load_manifest(raw_root, latest_backup_id)
        raw_candidates = [
            item.path
            for item in latest_raw.files
            if item.path.endswith("/provider_ingest_raw.tar")
        ]
        if not raw_candidates:
            raise DisasterRecoveryError("fixture_raw_response_missing")
        _repackage_without_file(raw_root, latest_backup_id, raw_candidates[0])
        scenarios["raw_response_missing"] = _verification_is_rejected(
            raw_root, repository_root, protocol
        )
    if latest_backup_id != _latest(backup_root):
        raise DisasterRecoveryError("latest_backup_pointer_drift")
    expected = {
        "audit_chain_truncation",
        "backup_interruption",
        "concurrent_restorer",
        "duplicate_charge_after_resume",
        "hash_tamper",
        "missing_member",
        "mixed_generation",
        "non_empty_target",
        "old_backup_rollback",
        "parent_chain_break_or_cycle",
        "raw_response_missing",
    }
    if set(scenarios) != expected or not all(scenarios.values()):
        missing_or_unblocked = sorted(
            expected - {name for name, blocked in scenarios.items() if blocked}
        )
        raise DisasterRecoveryError(
            "fault_matrix_incomplete:" + ",".join(missing_or_unblocked)
        )
    return scenarios


def _verification_is_rejected(
    backup_root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> bool:
    try:
        verify_backup(
            backup_root,
            repository_root=repository_root,
            protocol=protocol,
        )
    except DisasterRecoveryError:
        return True
    return False


def _reseal_manifest(
    backup_root: Path,
    previous_backup_id: str,
    payload: dict[str, Any],
) -> str:
    payload.pop("manifest_sha256", None)
    payload["backup_id"] = ""
    backup_payload = dict(payload)
    backup_payload.pop("backup_id")
    payload["backup_id"] = stable_hash(backup_payload)
    payload["manifest_sha256"] = stable_hash(payload)
    manifest = BackupManifest.model_validate(payload)
    durable_atomic_write_bytes(
        _manifest_path(backup_root, manifest.backup_id),
        canonical_json(manifest.model_dump(mode="json")),
        temporary_suffix="fixture-manifest",
    )
    latest_value = {
        "backup_id": manifest.backup_id,
        "latest_sha256": stable_hash({"backup_id": manifest.backup_id}),
    }
    durable_atomic_write_bytes(
        backup_root / LATEST_FILE,
        canonical_json(latest_value),
        temporary_suffix="fixture-latest",
    )
    if previous_backup_id == manifest.backup_id:
        raise DisasterRecoveryError("fixture_mutation_did_not_change_manifest")
    return manifest.backup_id


def _repackage_backup_file(
    backup_root: Path,
    backup_id: str,
    logical_path: str,
    mutate: Any,
) -> str:
    manifest = _load_manifest(backup_root, backup_id)
    payload = manifest.model_dump(mode="json")
    matched = False
    for item in payload["files"]:
        if item["path"] != logical_path:
            continue
        matched = True
        changed = mutate(
            _object_path(backup_root, item["sha256"]).read_bytes()
        )
        item["sha256"] = sha256_bytes(changed)
        item["size_bytes"] = len(changed)
        durable_atomic_write_bytes(
            _object_path(backup_root, item["sha256"]),
            changed,
            temporary_suffix="fixture-object",
        )
    if not matched:
        raise DisasterRecoveryError("fixture_logical_file_missing")
    return _reseal_manifest(backup_root, backup_id, payload)


def _repackage_without_file(
    backup_root: Path,
    backup_id: str,
    logical_path: str,
) -> str:
    manifest = _load_manifest(backup_root, backup_id)
    payload = manifest.model_dump(mode="json")
    payload["files"] = [
        item for item in payload["files"] if item["path"] != logical_path
    ]
    if len(payload["files"]) == len(manifest.files):
        raise DisasterRecoveryError("fixture_logical_file_missing")
    return _reseal_manifest(backup_root, backup_id, payload)


def _mutate_generation_reference(raw: bytes) -> bytes:
    value = json.loads(raw)
    current = int(value["generation_by_shard"]["0"])
    value["generation_by_shard"]["0"] = max(0, current - 1)
    value.pop("state_sha256")
    value["state_sha256"] = stable_hash(value)
    return canonical_json(value)


def _truncate_operation_audit(raw: bytes) -> bytes:
    value = json.loads(raw)
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DisasterRecoveryError("fixture_audit_entries_missing")
    entries.pop()
    value.pop("audit_sha256", None)
    value["audit_sha256"] = stable_hash(value)
    return canonical_json(value)


def audit_readiness(repository_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    _load_plan(repository_root, protocol)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "external_run_not_started",
        "exit_code": EXIT_BLOCKED,
        "controls_ready": True,
        "real_backup_created": False,
        "real_run_started": False,
        "network_status": "not_checked",
        "credential_status": "not_checked",
        "full1000_completed": False,
        "formal_validation_complete": False,
        "execution": EXECUTION_ZERO,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
