from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.evaluation.snapshot_resume import (
    _classify_entries,
    ResumeManifest,
    ResumeRequest,
    ResumeRuntimeConfig,
    SnapshotResumeError,
    execute_resume_manifest,
    fair_resume_schedule,
    recompute_resume_progress,
    request_signature,
    stable_hash,
    validate_manifest_required_plan,
    validate_runtime_config,
)
from scholar_agent.evaluation.snapshots.schemas import (
    RetrievalSnapshotEntry,
    SnapshotPlanEntry,
    SnapshotPlanRound,
)
from scholar_agent.evaluation.snapshots.store import (
    SnapshotStore,
    entry_content_hash,
    retrieval_snapshot_key,
)


def _config() -> ResumeRuntimeConfig:
    return ResumeRuntimeConfig(
        dataset="auto_scholar_query",
        dataset_split="test",
        offset=0,
        limit=1000,
        run_profile="balanced",
        sources=["openalex", "arxiv", "semantic_scholar", "pubmed"],
        result_policy="highly_and_partial",
        top_k=20,
        query_adapter_policy="adaptive",
        query_planning_policy="current_rules",
        ranking_policy="current_rules",
        judgement_policy="current_rules",
        enable_query_evolution=False,
        query_evolution_policy="off",
        enable_refchain=False,
        enable_semantic_seed_expansion=False,
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        current_year=None,
        budgets={
            "max_search_rounds": 2,
            "max_candidate_papers": 200,
            "max_llm_calls": 20,
            "max_total_tokens": 50000,
            "max_latency_seconds": 90.0,
        },
    )


def _plan_entry(
    source: str,
    case_id: str,
    query: str,
    *,
    priority: int = 2,
) -> SnapshotPlanEntry:
    key, _ = retrieval_snapshot_key(
        source=source,
        adapted_query=query,
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
    )
    return SnapshotPlanEntry(
        key=key,
        entry_type="retrieval",
        source=source,
        adapted_query=query,
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
        required_by_group="baseline",
        case_id=case_id,
        stage="initial_retrieval",
        origin_subquery=query,
        generated_by="initial_retrieval",
        query_planning_policy="current_rules",
        query_planner_version="1.9.0",
        priority=priority,
    )


def _request(
    entry: SnapshotPlanEntry,
    index: int,
    *,
    classification: str = "missing",
    initial_hash: str | None = None,
) -> ResumeRequest:
    _, normalized = retrieval_snapshot_key(
        source=entry.source,
        adapted_query=entry.adapted_query or "",
        limit=entry.limit,
        adapter_policy=entry.adapter_policy or "",
        connector_version=entry.connector_version,
    )
    return ResumeRequest(
        schedule_index=index,
        key=entry.key,
        source=entry.source,
        case_id=entry.case_id,
        case_index=index,
        adapted_query=entry.adapted_query or "",
        normalized_query=normalized,
        limit=entry.limit,
        adapter_policy=entry.adapter_policy or "",
        connector_version=entry.connector_version,
        stage=entry.stage,
        origin_subquery=entry.origin_subquery,
        priority=entry.priority,
        initial_classification=classification,
        initial_snapshot_content_hash=initial_hash,
        request_signature=request_signature(entry),
    )


def _manifest(
    requests: list[ResumeRequest],
    *,
    plan_path: str = "plan.json",
    plan_hash: str = "a" * 64,
    required_keys: list[str] | None = None,
) -> ResumeManifest:
    request_payloads = [request.model_dump(mode="json") for request in requests]
    config = _config()
    return ResumeManifest(
        dataset=config.dataset,
        snapshot_name="snapshot",
        snapshot_dir="snapshots",
        required_plan_path=plan_path,
        required_key_count=len(required_keys or requests),
        resume_key_count=len(requests),
        classification_counts={"missing": len(requests)},
        retry_policy={"failed": "once"},
        schedule_policy={"source_rotation": "fixed"},
        source_order=config.sources,
        runtime_config=config,
        runtime_config_sha256=config.sha256(),
        input_hashes={"frozen_plan_round": plan_hash},
        required_keys_sha256=stable_hash(sorted(required_keys or [r.key for r in requests])),
        requests_sha256=stable_hash(request_payloads),
        requests=requests,
    )


def _snapshot(
    request: ResumeRequest,
    *,
    status: str,
    error_message: str | None = None,
    warnings: list[str] | None = None,
) -> RetrievalSnapshotEntry:
    entry = RetrievalSnapshotEntry(
        key=request.key,
        source=request.source,
        adapted_query=request.adapted_query,
        normalized_query=request.normalized_query,
        limit=request.limit,
        adapter_policy=request.adapter_policy,
        connector_version=request.connector_version,
        status=status,
        error_message=error_message,
        warnings=warnings or [],
        diagnostics=ConnectorDiagnostics(
            request_count=1,
            error_count=int(status == "failed"),
        ),
        recorded_at="2026-07-21T00:00:00+00:00",
        content_hash="0" * 64,
    )
    return entry.model_copy(update={"content_hash": entry_content_hash(entry)})


def test_fair_schedule_rotates_sources_cases_and_is_input_order_independent() -> None:
    entries = [
        _plan_entry("openalex", "case-0", "q0-a"),
        _plan_entry("openalex", "case-0", "q0-b"),
        _plan_entry("openalex", "case-1", "q1-a"),
        _plan_entry("arxiv", "case-0", "q0-c", priority=1),
        _plan_entry("arxiv", "case-1", "q1-b", priority=1),
        _plan_entry("arxiv", "case-2", "q2", priority=1),
    ]
    kwargs = {
        "source_order": ["openalex", "arxiv"],
        "case_order": {"case-0": 0, "case-1": 1, "case-2": 2},
    }
    forward = fair_resume_schedule(entries, **kwargs)
    reverse = fair_resume_schedule(list(reversed(entries)), **kwargs)

    assert [entry.key for entry in forward] == [entry.key for entry in reverse]
    assert [entry.source for entry in forward[:4]] == [
        "openalex",
        "arxiv",
        "openalex",
        "arxiv",
    ]
    assert all(
        left.case_id != right.case_id
        for left, right in zip(forward, forward[1:])
    )


def test_manifest_rejects_duplicate_keys() -> None:
    entry = _plan_entry("openalex", "case-0", "query")
    request = _request(entry, 0)
    with pytest.raises(ValidationError, match="duplicate resume key"):
        _manifest([request, request.model_copy(update={"schedule_index": 1})])


def test_required_keys_close_success_failed_missing_and_not_started(
    tmp_path: Path,
) -> None:
    entries = [
        _plan_entry("openalex", "case-0", "success"),
        _plan_entry("arxiv", "case-0", "failed", priority=1),
        _plan_entry("pubmed", "case-0", "missing"),
        _plan_entry("semantic_scholar", "case-1", "not-started"),
    ]
    requests = [_request(entry, index) for index, entry in enumerate(entries)]
    store = SnapshotStore(tmp_path / "snapshots")
    store.write_retrieval(_snapshot(requests[0], status="success"))
    store.write_retrieval(
        _snapshot(requests[1], status="failed", error_message="timeout")
    )
    plan = SnapshotPlanRound(
        snapshot_name="snapshot",
        group="baseline",
        round_index=2,
        entries=entries,
        created_at="2026-07-21T00:00:00+00:00",
    )

    rows, snapshots = _classify_entries(plan, store, {"case-0"})

    assert [row["classification"] for row in rows] == [
        "success",
        "failed",
        "missing",
        "not_started",
    ]
    assert set(snapshots) == {entries[0].key, entries[1].key}


def test_runtime_config_drift_names_changed_field() -> None:
    manifest = _manifest([_request(_plan_entry("openalex", "case", "q"), 0)])
    with pytest.raises(SnapshotResumeError, match="top_k"):
        validate_runtime_config(
            manifest,
            _config().model_copy(update={"top_k": 50}),
        )


def test_required_plan_rejects_unknown_key(tmp_path: Path) -> None:
    entry = _plan_entry("openalex", "case", "query")
    plan = SnapshotPlanRound(
        snapshot_name="snapshot",
        group="baseline",
        round_index=2,
        entries=[entry],
        created_at="2026-07-21T00:00:00+00:00",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    unknown = _plan_entry("arxiv", "case", "other", priority=1)
    manifest = _manifest(
        [_request(unknown, 0)],
        plan_path="plan.json",
        plan_hash=_sha(plan_path),
        required_keys=[entry.key],
    )
    with pytest.raises(SnapshotResumeError, match="unknown resume key"):
        validate_manifest_required_plan(manifest, repository_root=tmp_path)


def test_progress_skips_success_and_retries_frozen_failure_once(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")
    missing_entry = _plan_entry("openalex", "case-0", "missing")
    failed_entry = _plan_entry("arxiv", "case-1", "failed", priority=1)
    failed_seed = _request(
        failed_entry,
        1,
        classification="failed",
        initial_hash="f" * 64,
    )
    initial_failed = _snapshot(
        failed_seed,
        status="failed",
        error_message="timeout",
    )
    failed_request = failed_seed.model_copy(
        update={"initial_snapshot_content_hash": initial_failed.content_hash}
    )
    missing_request = _request(missing_entry, 0)
    manifest = _manifest([missing_request, failed_request])
    store.write_retrieval(initial_failed)

    before = recompute_resume_progress(manifest, store)
    assert before.pending_keys == [missing_request.key, failed_request.key]

    calls: list[str] = []

    def fake_executor(request: ResumeRequest) -> ConnectorSearchResult:
        calls.append(request.key)
        if request.key == failed_request.key:
            return ConnectorSearchResult(
                error_message="timeout",
                warnings=[],
                diagnostics=ConnectorDiagnostics(request_count=1, error_count=1),
            )
        return ConnectorSearchResult(
            diagnostics=ConnectorDiagnostics(request_count=1)
        )

    report = execute_resume_manifest(
        manifest,
        store,
        executor=fake_executor,
        dry_run=False,
    )
    assert calls == [missing_request.key, failed_request.key]
    assert report.final_progress.pending_key_count == 0
    assert report.final_progress.completed_success_count == 1
    assert report.final_progress.completed_failed_count == 1
    retried = store.read_retrieval(failed_request.key)
    assert "resume_manifest_retry_attempted" in retried.warnings

    second = execute_resume_manifest(
        manifest,
        store,
        executor=fake_executor,
        dry_run=False,
    )
    assert second.attempted_count == 0
    assert calls == [missing_request.key, failed_request.key]


def test_dry_run_is_read_only_and_does_not_call_executor(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")
    request = _request(_plan_entry("pubmed", "case", "query"), 0)
    manifest = _manifest([request])
    before = sorted(tmp_path.rglob("*"))

    report = execute_resume_manifest(
        manifest,
        store,
        executor=lambda _: pytest.fail("executor must not run"),
        dry_run=True,
    )

    assert report.network_request_count == 0
    assert report.snapshot_write_count == 0
    assert report.final_progress.pending_key_count == 1
    assert sorted(tmp_path.rglob("*")) == before


def test_existing_success_is_never_overwritten(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")
    request = _request(_plan_entry("openalex", "case", "query"), 0)
    existing = _snapshot(request, status="success")
    store.write_retrieval(existing)
    manifest = _manifest([request])

    report = execute_resume_manifest(
        manifest,
        store,
        executor=lambda _: pytest.fail("completed key must be skipped"),
        dry_run=False,
    )

    assert report.attempted_count == 0
    assert store.read_retrieval(request.key).content_hash == existing.content_hash


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
