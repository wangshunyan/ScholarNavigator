"""Gold-blind missing-key audit and deterministic Snapshot resume scheduling.

The module operates on a frozen Snapshot plan, query-only planning artifacts,
and top-level Record terminal fields.  It deliberately does not import dataset
adapters or evaluators and never computes effectiveness metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.evaluation.query_planning_regression import (
    _select_top_level_string_fields,
)
from scholar_agent.evaluation.snapshots.schemas import (
    QUERY_ADAPTER_VERSION,
    RetrievalSnapshotEntry,
    SnapshotPlanEntry,
    SnapshotPlanRound,
)
from scholar_agent.evaluation.snapshots.store import (
    SnapshotStore,
    entry_content_hash,
    normalize_snapshot_query,
    retrieval_snapshot_key,
    utc_now,
)


RESUME_SCHEMA_VERSION = "1"
RESUME_POLICY_VERSION = "autoscholar_full1000_resume_v1"
KeyClassification = Literal["success", "failed", "missing", "not_started"]
ResumeExecutor = Callable[["ResumeRequest"], ConnectorSearchResult]


class SnapshotResumeError(RuntimeError):
    """Resume manifest or frozen input violates an auditable invariant."""


class ResumeRuntimeConfig(BaseModel):
    """Retrieval-affecting CLI configuration protected by the resume hash."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    dataset_split: str
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    run_profile: str
    sources: list[str]
    result_policy: str
    top_k: int = Field(ge=1)
    query_adapter_policy: str
    query_planning_policy: str
    ranking_policy: str
    judgement_policy: str
    enable_query_evolution: bool
    query_evolution_policy: str
    enable_refchain: bool
    enable_semantic_seed_expansion: bool
    enable_llm_query_understanding: bool
    enable_llm_judgement: bool
    current_year: int | None
    budgets: dict[str, int | float]

    def sha256(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class ResumeRequest(BaseModel):
    """One immutable retrieval request in deterministic execution order."""

    model_config = ConfigDict(extra="forbid")

    schedule_index: int = Field(ge=0)
    key: str = Field(min_length=64, max_length=64)
    source: str
    case_id: str
    case_index: int = Field(ge=0)
    adapted_query: str
    normalized_query: str
    limit: int = Field(ge=0)
    adapter_policy: str
    query_adapter_version: str = QUERY_ADAPTER_VERSION
    connector_version: str
    stage: str
    origin_subquery: str | None = None
    priority: int = Field(ge=1)
    initial_classification: Literal["failed", "missing", "not_started"]
    initial_snapshot_content_hash: str | None = None
    request_signature: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_failed_fingerprint(self) -> "ResumeRequest":
        if (
            self.initial_classification == "failed"
            and self.initial_snapshot_content_hash is None
        ):
            raise ValueError("failed resume request requires initial snapshot hash")
        if (
            self.initial_classification != "failed"
            and self.initial_snapshot_content_hash is not None
        ):
            raise ValueError("only failed resume requests may fingerprint a snapshot")
        return self


class ResumeManifest(BaseModel):
    """Versioned, static schedule plus frozen request/config provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RESUME_SCHEMA_VERSION
    policy_version: str = RESUME_POLICY_VERSION
    scope: str = "snapshot_resume_only_not_effectiveness_evaluation"
    dataset: str
    snapshot_name: str
    snapshot_dir: str
    required_plan_path: str
    required_key_count: int = Field(ge=0)
    resume_key_count: int = Field(ge=0)
    classification_counts: dict[str, int]
    retry_policy: dict[str, object]
    schedule_policy: dict[str, object]
    source_order: list[str]
    runtime_config: ResumeRuntimeConfig
    runtime_config_sha256: str = Field(min_length=64, max_length=64)
    input_hashes: dict[str, str]
    required_keys_sha256: str = Field(min_length=64, max_length=64)
    requests_sha256: str = Field(min_length=64, max_length=64)
    gold_fields_accessed: bool = False
    effectiveness_metrics_generated: bool = False
    requests: list[ResumeRequest]

    @model_validator(mode="after")
    def validate_closed_manifest(self) -> "ResumeManifest":
        keys = [request.key for request in self.requests]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate resume key")
        if len(keys) != self.resume_key_count:
            raise ValueError("resume key count mismatch")
        if [request.schedule_index for request in self.requests] != list(
            range(len(self.requests))
        ):
            raise ValueError("resume schedule indexes must be contiguous")
        if self.runtime_config.sha256() != self.runtime_config_sha256:
            raise ValueError("runtime config hash mismatch")
        expected_requests_hash = stable_hash(
            [request.model_dump(mode="json") for request in self.requests]
        )
        if expected_requests_hash != self.requests_sha256:
            raise ValueError("resume request hash mismatch")
        if self.gold_fields_accessed or self.effectiveness_metrics_generated:
            raise ValueError("resume manifest must remain gold blind")
        return self


class ResumeProgress(BaseModel):
    """Recomputed progress; no mutable cursor is trusted."""

    manifest_key_count: int = Field(ge=0)
    pending_key_count: int = Field(ge=0)
    completed_success_count: int = Field(ge=0)
    completed_failed_count: int = Field(ge=0)
    skipped_existing_count: int = Field(ge=0)
    pending_keys: list[str]


class ResumeExecutionReport(BaseModel):
    dry_run: bool
    initial_progress: ResumeProgress
    final_progress: ResumeProgress
    attempted_count: int = Field(ge=0)
    written_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    network_request_count: int = Field(ge=0)
    snapshot_write_count: int = Field(ge=0)


def stable_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_config_from_record_config(path: Path) -> ResumeRuntimeConfig:
    """Project only retrieval-affecting, non-secret fields from frozen config."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotResumeError("invalid frozen record config") from exc
    if not isinstance(payload, dict):
        raise SnapshotResumeError("invalid frozen record config")
    llm = payload.get("llm")
    if not isinstance(llm, dict):
        llm = {}
    required = {
        "dataset",
        "dataset_split",
        "offset",
        "limit",
        "run_profile",
        "sources",
        "result_policy",
        "top_k",
        "query_adapter_policy",
        "query_planning_policy",
        "ranking_policy",
        "judgement_policy",
        "enable_query_evolution",
        "query_evolution_policy",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "budgets",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise SnapshotResumeError("record config missing:" + ",".join(missing))
    return ResumeRuntimeConfig(
        dataset=str(payload["dataset"]),
        dataset_split=str(payload["dataset_split"]),
        offset=int(payload["offset"]),
        limit=int(payload["limit"]),
        run_profile=str(payload["run_profile"]),
        sources=[str(source) for source in payload["sources"]],
        result_policy=str(payload["result_policy"]),
        top_k=int(payload["top_k"]),
        query_adapter_policy=str(payload["query_adapter_policy"]),
        query_planning_policy=str(payload["query_planning_policy"]),
        ranking_policy=str(payload["ranking_policy"]),
        judgement_policy=str(payload["judgement_policy"]),
        enable_query_evolution=bool(payload["enable_query_evolution"]),
        query_evolution_policy=str(payload["query_evolution_policy"]),
        enable_refchain=bool(payload["enable_refchain"]),
        enable_semantic_seed_expansion=bool(
            payload["enable_semantic_seed_expansion"]
        ),
        enable_llm_query_understanding=bool(llm.get("query_understanding", False)),
        enable_llm_judgement=bool(llm.get("judgement", False)),
        current_year=(
            int(payload["current_year"])
            if payload.get("current_year") is not None
            else None
        ),
        budgets={
            str(key): float(value) if isinstance(value, float) else int(value)
            for key, value in dict(payload["budgets"]).items()
        },
    )


def request_signature(entry: SnapshotPlanEntry) -> str:
    if entry.entry_type != "retrieval" or entry.adapted_query is None:
        raise SnapshotResumeError(f"unsupported required entry:{entry.key}")
    if entry.adapter_policy is None:
        raise SnapshotResumeError(f"missing adapter policy:{entry.key}")
    payload = {
        "key": entry.key,
        "source": entry.source,
        "adapted_query": normalize_snapshot_query(entry.adapted_query),
        "limit": entry.limit,
        "adapter_policy": entry.adapter_policy,
        "query_adapter_version": QUERY_ADAPTER_VERSION,
        "connector_version": entry.connector_version,
    }
    return stable_hash(payload)


def validate_request_key(entry: SnapshotPlanEntry) -> str:
    if entry.adapted_query is None or entry.adapter_policy is None:
        raise SnapshotResumeError(f"incomplete retrieval request:{entry.key}")
    expected_key, normalized_query = retrieval_snapshot_key(
        source=entry.source,
        adapted_query=entry.adapted_query,
        limit=entry.limit,
        adapter_policy=entry.adapter_policy,
        connector_version=entry.connector_version,
    )
    if expected_key != entry.key:
        raise SnapshotResumeError(f"request signature drift:{entry.key}")
    return normalized_query


def validate_snapshot_request(
    plan_entry: SnapshotPlanEntry,
    snapshot: RetrievalSnapshotEntry,
) -> None:
    expected = {
        "key": plan_entry.key,
        "source": plan_entry.source,
        "normalized_query": normalize_snapshot_query(plan_entry.adapted_query or ""),
        "limit": plan_entry.limit,
        "adapter_policy": plan_entry.adapter_policy,
        "query_adapter_version": QUERY_ADAPTER_VERSION,
        "connector_version": plan_entry.connector_version,
    }
    actual = {
        "key": snapshot.key,
        "source": snapshot.source,
        "normalized_query": snapshot.normalized_query,
        "limit": snapshot.limit,
        "adapter_policy": snapshot.adapter_policy,
        "query_adapter_version": snapshot.query_adapter_version,
        "connector_version": snapshot.connector_version,
    }
    if actual != expected:
        differing = sorted(key for key in expected if expected[key] != actual[key])
        raise SnapshotResumeError(
            f"snapshot request mismatch:{plan_entry.key}:{','.join(differing)}"
        )


def load_record_terminals(path: Path) -> dict[str, str]:
    """Read only top-level case_id/status, structurally skipping every other field."""

    terminals: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = _select_top_level_string_fields(
                raw,
                fields=frozenset({"case_id", "status"}),
                line_number=line_number,
            )
            case_id = fields.get("case_id")
            status = fields.get("status")
            if not case_id or not status:
                raise SnapshotResumeError(f"invalid record terminal line:{line_number}")
            if case_id in terminals:
                raise SnapshotResumeError(f"duplicate record case:{case_id}")
            terminals[case_id] = status
    return terminals


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SnapshotResumeError(f"invalid jsonl:{path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise SnapshotResumeError(f"invalid jsonl row:{path}:{line_number}")
            rows.append(payload)
    return rows


def _load_plan_round(path: Path) -> SnapshotPlanRound:
    try:
        plan = SnapshotPlanRound.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise SnapshotResumeError("invalid frozen plan round") from exc
    keys = [entry.key for entry in plan.entries]
    if len(keys) != len(set(keys)):
        raise SnapshotResumeError("duplicate required key")
    return plan


def _classify_entries(
    plan: SnapshotPlanRound,
    store: SnapshotStore,
    attempted_cases: set[str],
) -> tuple[list[dict[str, Any]], dict[str, RetrievalSnapshotEntry]]:
    snapshots: dict[str, RetrievalSnapshotEntry] = {}
    rows: list[dict[str, Any]] = []
    for entry in plan.entries:
        normalized_query = validate_request_key(entry)
        snapshot_path = store.retrieval_dir / f"{entry.key}.json"
        if snapshot_path.is_file():
            snapshot = store.read_retrieval(entry.key)
            validate_snapshot_request(entry, snapshot)
            snapshots[entry.key] = snapshot
            classification: KeyClassification = snapshot.status
        elif entry.case_id in attempted_cases:
            classification = "missing"
        else:
            classification = "not_started"
        rows.append(
            {
                "key": entry.key,
                "source": entry.source,
                "case_id": entry.case_id,
                "classification": classification,
                "request_signature": request_signature(entry),
                "normalized_query_sha256": hashlib.sha256(
                    normalized_query.encode("utf-8")
                ).hexdigest(),
                "snapshot_content_hash": (
                    snapshots[entry.key].content_hash
                    if entry.key in snapshots
                    else None
                ),
            }
        )
    return rows, snapshots


def _case_features(
    query_rows: Sequence[Mapping[str, Any]],
    planning_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    if len(query_rows) != len(planning_rows):
        raise SnapshotResumeError("query/planning row count mismatch")
    case_order: dict[str, int] = {}
    features: dict[str, dict[str, object]] = {}
    lengths: list[tuple[int, int, str]] = []
    for index, (query_row, plan_row) in enumerate(zip(query_rows, planning_rows)):
        query_id = query_row.get("query_id")
        query = query_row.get("query")
        if query_id != plan_row.get("query_id") or not isinstance(query, str):
            raise SnapshotResumeError(f"query/planning order drift:{index}")
        if not isinstance(query_id, str) or query_id in case_order:
            raise SnapshotResumeError(f"invalid query id:{index}")
        plan = plan_row.get("plan")
        if not isinstance(plan, dict):
            raise SnapshotResumeError(f"missing frozen plan:{query_id}")
        analysis = plan.get("query_analysis")
        constraints = analysis.get("constraints") if isinstance(analysis, dict) else None
        quality = plan_row.get("quality")
        if not isinstance(constraints, dict) or not isinstance(quality, dict):
            raise SnapshotResumeError(f"missing planning features:{query_id}")
        case_order[query_id] = index
        lengths.append((len(query), index, query_id))
        features[query_id] = {
            "case_index": index,
            "order_quartile": f"q{index // 250 + 1}",
            "query_length": len(query),
            "subquery_count": int(quality.get("subquery_count", 0)),
            "has_method_constraint": bool(constraints.get("methods")),
            "has_dataset_constraint": bool(constraints.get("datasets")),
            "has_time_constraint": constraints.get("time_range") is not None,
        }
    for rank, (_, _, query_id) in enumerate(sorted(lengths)):
        features[query_id]["length_quartile"] = f"q{rank // 250 + 1}"
    return features, case_order


def _count_by_strata(
    classified_rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, int]]]:
    dimensions = {
        "source": lambda row, _: str(row["source"]),
        "query_order_quartile": lambda _, feature: str(feature["order_quartile"]),
        "query_length_quartile": lambda _, feature: str(feature["length_quartile"]),
        "subquery_count": lambda _, feature: str(feature["subquery_count"]),
        "method_constraint": lambda _, feature: str(
            feature["has_method_constraint"]
        ).lower(),
        "dataset_constraint": lambda _, feature: str(
            feature["has_dataset_constraint"]
        ).lower(),
        "time_constraint": lambda _, feature: str(
            feature["has_time_constraint"]
        ).lower(),
    }
    aggregate: dict[str, dict[str, Counter[str]]] = {
        dimension: defaultdict(Counter) for dimension in dimensions
    }
    for row in classified_rows:
        feature = features.get(str(row["case_id"]))
        if feature is None:
            raise SnapshotResumeError(f"unknown case in required plan:{row['case_id']}")
        for dimension, selector in dimensions.items():
            bucket = selector(row, feature)
            aggregate[dimension][bucket][str(row["classification"])] += 1
            aggregate[dimension][bucket]["total"] += 1
    return {
        dimension: {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(buckets.items())
        }
        for dimension, buckets in aggregate.items()
    }


def fair_resume_schedule(
    entries: Sequence[SnapshotPlanEntry],
    *,
    source_order: Sequence[str],
    case_order: Mapping[str, int],
) -> list[SnapshotPlanEntry]:
    """Round-robin sources and cases without depending on input entry order."""

    known_sources = set(source_order)
    unknown_sources = sorted({entry.source for entry in entries} - known_sources)
    if unknown_sources:
        raise SnapshotResumeError(f"unknown schedule sources:{','.join(unknown_sources)}")
    per_source: dict[str, dict[str, deque[SnapshotPlanEntry]]] = {
        source: defaultdict(deque) for source in source_order
    }
    for entry in sorted(
        entries,
        key=lambda item: (
            source_order.index(item.source),
            case_order.get(item.case_id, 10**9),
            item.priority,
            item.origin_subquery or "",
            item.key,
        ),
    ):
        if entry.case_id not in case_order:
            raise SnapshotResumeError(f"unknown schedule case:{entry.case_id}")
        per_source[entry.source][entry.case_id].append(entry)

    source_cases: dict[str, deque[str]] = {}
    for source in source_order:
        source_cases[source] = deque(
            sorted(per_source[source], key=lambda case_id: case_order[case_id])
        )

    scheduled: list[SnapshotPlanEntry] = []
    last_case: str | None = None
    last_source: str | None = None
    total_by_source = {
        source: sum(len(bucket) for bucket in per_source[source].values())
        for source in source_order
    }
    emitted_by_source = {source: 0 for source in source_order}
    remaining = len(entries)
    while remaining:
        active = [source for source in source_order if source_cases[source]]
        if not active:
            raise SnapshotResumeError("resume scheduler stalled")
        candidates = sorted(
            active,
            key=lambda source: (
                Fraction(
                    emitted_by_source[source] + 1,
                    total_by_source[source],
                ),
                source_order.index(source),
            ),
        )
        source = candidates[0]
        if source == last_source and len(candidates) > 1:
            source = candidates[1]
        cases = source_cases[source]
        if len(cases) > 1 and cases[0] == last_case:
            cases.rotate(-1)
        case_id = cases.popleft()
        bucket = per_source[source][case_id]
        entry = bucket.popleft()
        scheduled.append(entry)
        last_case = case_id
        last_source = source
        emitted_by_source[source] += 1
        remaining -= 1
        if bucket:
            cases.append(case_id)
    return scheduled


def _request_from_plan(
    entry: SnapshotPlanEntry,
    *,
    schedule_index: int,
    case_index: int,
    classification: Literal["failed", "missing", "not_started"],
    snapshot: RetrievalSnapshotEntry | None,
) -> ResumeRequest:
    normalized_query = validate_request_key(entry)
    return ResumeRequest(
        schedule_index=schedule_index,
        key=entry.key,
        source=entry.source,
        case_id=entry.case_id,
        case_index=case_index,
        adapted_query=entry.adapted_query or "",
        normalized_query=normalized_query,
        limit=entry.limit,
        adapter_policy=entry.adapter_policy or "",
        connector_version=entry.connector_version,
        stage=entry.stage,
        origin_subquery=entry.origin_subquery,
        priority=entry.priority,
        initial_classification=classification,
        initial_snapshot_content_hash=(
            snapshot.content_hash if classification == "failed" and snapshot else None
        ),
        request_signature=request_signature(entry),
    )


def build_resume_audit(
    *,
    plan_round_path: Path,
    snapshot_dir: Path,
    record_results_path: Path,
    record_config_path: Path,
    query_input_path: Path,
    planning_baseline_path: Path,
    runtime_config: ResumeRuntimeConfig,
    source_order: Sequence[str],
) -> tuple[dict[str, Any], ResumeManifest, list[dict[str, Any]]]:
    """Build a closed 4-way audit and retry-once fair resume schedule."""

    plan = _load_plan_round(plan_round_path)
    if plan.group != "baseline":
        raise SnapshotResumeError("unexpected frozen plan group")
    store = SnapshotStore(snapshot_dir)
    snapshot_manifest = store.read_manifest()
    if snapshot_manifest.snapshot_name != plan.snapshot_name:
        raise SnapshotResumeError("snapshot/plan name mismatch")
    if runtime_config.sha256() == "":  # pragma: no cover - defensive only
        raise SnapshotResumeError("empty config hash")

    terminals = load_record_terminals(record_results_path)
    query_rows = _read_jsonl(query_input_path)
    planning_rows = _read_jsonl(planning_baseline_path)
    features, case_order = _case_features(query_rows, planning_rows)
    unknown_terminals = sorted(set(terminals) - set(case_order))
    if unknown_terminals:
        raise SnapshotResumeError(f"unknown record cases:{unknown_terminals[0]}")

    classified_rows, snapshots = _classify_entries(
        plan,
        store,
        set(terminals),
    )
    counts = Counter(str(row["classification"]) for row in classified_rows)
    if sum(counts.values()) != len(plan.entries):
        raise SnapshotResumeError("required key classification did not close")

    classification_by_key = {
        str(row["key"]): str(row["classification"]) for row in classified_rows
    }
    eligible_entries = [
        entry
        for entry in plan.entries
        if classification_by_key[entry.key] in {"failed", "missing", "not_started"}
    ]
    scheduled = fair_resume_schedule(
        eligible_entries,
        source_order=source_order,
        case_order=case_order,
    )
    requests = [
        _request_from_plan(
            entry,
            schedule_index=index,
            case_index=case_order[entry.case_id],
            classification=classification_by_key[entry.key],  # type: ignore[arg-type]
            snapshot=snapshots.get(entry.key),
        )
        for index, entry in enumerate(scheduled)
    ]
    required_keys = sorted(entry.key for entry in plan.entries)
    input_hashes = {
        "frozen_plan_round": sha256_file(plan_round_path),
        "record_config": sha256_file(record_config_path),
        "record_terminals_projection": stable_hash(
            [{"case_id": key, "status": terminals[key]} for key in sorted(terminals)]
        ),
        "query_input": sha256_file(query_input_path),
        "planning_baseline": sha256_file(planning_baseline_path),
        "snapshot_manifest": sha256_file(store.manifest_path),
        "required_snapshot_state": stable_hash(
            {
                key: snapshot.content_hash
                for key, snapshot in sorted(snapshots.items())
            }
        ),
    }
    request_payloads = [request.model_dump(mode="json") for request in requests]
    resume_manifest = ResumeManifest(
        dataset=runtime_config.dataset,
        snapshot_name=plan.snapshot_name,
        snapshot_dir=_repository_relative(snapshot_dir),
        required_plan_path=_repository_relative(plan_round_path),
        required_key_count=len(plan.entries),
        resume_key_count=len(requests),
        classification_counts=dict(sorted(counts.items())),
        retry_policy={
            "missing": "eligible",
            "not_started": "eligible",
            "failed": "all_frozen_failed_keys_eligible_once",
            "success": "never_overwrite",
            "failed_retry_completion": (
                "skip when current failed content hash differs from frozen hash"
            ),
        },
        schedule_policy={
            "version": RESUME_POLICY_VERSION,
            "source_rotation": (
                "weighted fair rotation by emitted/total ratio; fixed source_order "
                "tie-break; avoid adjacent same source while alternatives remain"
            ),
            "case_rotation": "frozen query manifest order within each source",
            "same_case_guard": "rotate once when an alternative case exists",
            "entry_tiebreak": "priority, origin_subquery, snapshot_key",
            "input_order_independent": True,
        },
        source_order=list(source_order),
        runtime_config=runtime_config,
        runtime_config_sha256=runtime_config.sha256(),
        input_hashes=input_hashes,
        required_keys_sha256=stable_hash(required_keys),
        requests_sha256=stable_hash(request_payloads),
        requests=requests,
    )
    resume_manifest = ResumeManifest.model_validate(
        resume_manifest.model_dump(mode="json")
    )

    schedule_indices = {request.key: request.schedule_index for request in requests}
    audit_rows: list[dict[str, Any]] = []
    for row in classified_rows:
        feature = features[str(row["case_id"])]
        audit_rows.append(
            {
                **row,
                "case_index": feature["case_index"],
                "query_order_quartile": feature["order_quartile"],
                "query_length": feature["query_length"],
                "query_length_quartile": feature["length_quartile"],
                "subquery_count": feature["subquery_count"],
                "constraints": {
                    "method": feature["has_method_constraint"],
                    "dataset": feature["has_dataset_constraint"],
                    "time": feature["has_time_constraint"],
                },
                "resume_eligible": row["classification"] != "success",
                "schedule_index": schedule_indices.get(str(row["key"])),
            }
        )
    audit_rows.sort(key=lambda row: str(row["key"]))

    consecutive_sources, consecutive_cases = _max_consecutive(requests)
    summary = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "audit": "autoscholar_full1000_snapshot_resume",
        "scope": "missingness_and_execution_readiness_not_effectiveness",
        "dataset": runtime_config.dataset,
        "required_key_count": len(plan.entries),
        "record_terminal_case_count": len(terminals),
        "classification_counts": dict(sorted(counts.items())),
        "classification_closed": sum(counts.values()) == len(plan.entries),
        "request_signature_drift_count": 0,
        "resume_key_count": len(requests),
        "resume_source_counts": dict(
            sorted(Counter(request.source for request in requests).items())
        ),
        "bias_audit": _count_by_strata(classified_rows, features),
        "schedule": {
            "policy_version": RESUME_POLICY_VERSION,
            "max_consecutive_same_source": consecutive_sources,
            "max_consecutive_same_case": consecutive_cases,
            "requests_sha256": resume_manifest.requests_sha256,
        },
        "execution": {
            "dataset_adapter_invoked": False,
            "evaluator_invoked": False,
            "gold_fields_accessed": False,
            "effectiveness_metrics_generated": False,
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
        "input_hashes": input_hashes,
    }
    return summary, resume_manifest, audit_rows


def _repository_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return resolved.as_posix()


def _max_consecutive(requests: Sequence[ResumeRequest]) -> tuple[int, int]:
    max_source = max_case = current_source = current_case = 0
    previous_source = previous_case = None
    for request in requests:
        current_source = current_source + 1 if request.source == previous_source else 1
        current_case = current_case + 1 if request.case_id == previous_case else 1
        max_source = max(max_source, current_source)
        max_case = max(max_case, current_case)
        previous_source = request.source
        previous_case = request.case_id
    return max_source, max_case


def write_resume_audit(
    output_dir: Path,
    summary: Mapping[str, object],
    manifest: ResumeManifest,
    audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "resume_manifest": output_dir / "resume_manifest.json",
        "key_audit": output_dir / "key_audit.jsonl",
    }
    _atomic_write_json(paths["summary"], dict(summary))
    _atomic_write_json(
        paths["resume_manifest"], manifest.model_dump(mode="json")
    )
    _atomic_write_jsonl(paths["key_audit"], audit_rows)
    return {name: sha256_file(path) for name, path in paths.items()}


def load_resume_manifest(path: Path) -> ResumeManifest:
    try:
        return ResumeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise SnapshotResumeError("invalid resume manifest") from exc


def validate_manifest_required_plan(
    manifest: ResumeManifest,
    *,
    repository_root: Path,
) -> None:
    plan_path = (repository_root / manifest.required_plan_path).resolve()
    expected_hash = manifest.input_hashes.get("frozen_plan_round")
    if expected_hash is None or sha256_file(plan_path) != expected_hash:
        raise SnapshotResumeError("resume required plan hash drift")
    plan = _load_plan_round(plan_path)
    by_key = {entry.key: entry for entry in plan.entries}
    if stable_hash(sorted(by_key)) != manifest.required_keys_sha256:
        raise SnapshotResumeError("resume required key index drift")
    for request in manifest.requests:
        entry = by_key.get(request.key)
        if entry is None:
            raise SnapshotResumeError(f"unknown resume key:{request.key}")
        if request_signature(entry) != request.request_signature:
            raise SnapshotResumeError(f"resume request signature drift:{request.key}")


def validate_runtime_config(
    manifest: ResumeManifest,
    runtime_config: ResumeRuntimeConfig,
) -> None:
    actual_hash = runtime_config.sha256()
    if actual_hash != manifest.runtime_config_sha256:
        expected = manifest.runtime_config.model_dump(mode="json")
        actual = runtime_config.model_dump(mode="json")
        differing = sorted(key for key in expected if expected[key] != actual[key])
        raise SnapshotResumeError(
            "resume config drift:" + ",".join(differing or ["hash"])
        )


def recompute_resume_progress(
    manifest: ResumeManifest,
    store: SnapshotStore,
) -> ResumeProgress:
    pending: list[str] = []
    success = failed = skipped = 0
    for request in manifest.requests:
        path = store.retrieval_dir / f"{request.key}.json"
        if not path.is_file():
            pending.append(request.key)
            continue
        snapshot = store.read_retrieval(request.key)
        _validate_resume_snapshot_request(request, snapshot)
        if (
            request.initial_classification == "failed"
            and snapshot.content_hash == request.initial_snapshot_content_hash
        ):
            pending.append(request.key)
            continue
        skipped += 1
        if snapshot.status == "success":
            success += 1
        else:
            failed += 1
    return ResumeProgress(
        manifest_key_count=len(manifest.requests),
        pending_key_count=len(pending),
        completed_success_count=success,
        completed_failed_count=failed,
        skipped_existing_count=skipped,
        pending_keys=pending,
    )


def execute_resume_manifest(
    manifest: ResumeManifest,
    store: SnapshotStore,
    *,
    executor: ResumeExecutor | None,
    dry_run: bool,
) -> ResumeExecutionReport:
    """Run only pending scheduled keys; dry-run is strictly read-only."""

    initial = recompute_resume_progress(manifest, store)
    if dry_run:
        return ResumeExecutionReport(
            dry_run=True,
            initial_progress=initial,
            final_progress=initial,
            attempted_count=0,
            written_count=0,
            success_count=0,
            failed_count=0,
            network_request_count=0,
            snapshot_write_count=0,
        )
    if executor is None:
        raise SnapshotResumeError("resume executor required")
    pending = set(initial.pending_keys)
    attempted = written = success = failed = network_requests = 0
    for request in manifest.requests:
        if request.key not in pending:
            continue
        attempted += 1
        started = time.perf_counter()
        try:
            result = executor(request)
        except Exception as exc:  # noqa: BLE001 - terminal must be auditable
            result = ConnectorSearchResult(
                error_message=f"resume_executor_exception:{type(exc).__name__}",
                warnings=["resume_executor_exception"],
                diagnostics=ConnectorDiagnostics(error_count=1),
            )
        elapsed = max(result.latency_seconds, time.perf_counter() - started)
        network_requests += result.diagnostics.request_count
        snapshot = RetrievalSnapshotEntry(
            key=request.key,
            source=request.source,
            adapted_query=request.adapted_query,
            normalized_query=request.normalized_query,
            limit=request.limit,
            adapter_policy=request.adapter_policy,
            query_adapter_version=request.query_adapter_version,
            connector_version=request.connector_version,
            status="failed" if result.error_message else "success",
            papers=[paper.model_copy(deep=True) for paper in result.papers],
            error_message=_sanitize_terminal_text(result.error_message),
            warnings=[
                *(_sanitize_terminal_text(item) or "" for item in result.warnings),
                *(
                    ["resume_manifest_retry_attempted"]
                    if request.initial_classification == "failed"
                    and "resume_manifest_retry_attempted" not in result.warnings
                    else []
                ),
            ],
            diagnostics=result.diagnostics.model_copy(deep=True),
            recorded_latency_seconds=elapsed,
            recorded_at=utc_now(),
            content_hash="0" * 64,
        )
        snapshot = snapshot.model_copy(
            update={"content_hash": entry_content_hash(snapshot)}
        )
        did_write = store.write_retrieval(
            snapshot,
            overwrite=request.initial_classification == "failed",
        )
        written += int(did_write)
        success += int(snapshot.status == "success")
        failed += int(snapshot.status == "failed")
    final = recompute_resume_progress(manifest, store)
    return ResumeExecutionReport(
        dry_run=False,
        initial_progress=initial,
        final_progress=final,
        attempted_count=attempted,
        written_count=written,
        success_count=success,
        failed_count=failed,
        network_request_count=network_requests,
        snapshot_write_count=written,
    )


def _sanitize_terminal_text(value: str | None) -> str | None:
    if value is None:
        return None
    sanitized = re.sub(
        r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)[^\s&,;]+",
        r"\1\2[REDACTED]",
        str(value),
    )
    return sanitized[:2000]


def _validate_resume_snapshot_request(
    request: ResumeRequest,
    snapshot: RetrievalSnapshotEntry,
) -> None:
    actual = {
        "key": snapshot.key,
        "source": snapshot.source,
        "adapted_query": snapshot.normalized_query,
        "limit": snapshot.limit,
        "adapter_policy": snapshot.adapter_policy,
        "query_adapter_version": snapshot.query_adapter_version,
        "connector_version": snapshot.connector_version,
    }
    expected = {
        "key": request.key,
        "source": request.source,
        "adapted_query": request.normalized_query,
        "limit": request.limit,
        "adapter_policy": request.adapter_policy,
        "query_adapter_version": request.query_adapter_version,
        "connector_version": request.connector_version,
    }
    if actual != expected:
        raise SnapshotResumeError(f"resume snapshot request drift:{request.key}")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
