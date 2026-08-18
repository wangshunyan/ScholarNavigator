from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import SearchBudget
from scholar_agent.evaluation.crash_consistency import BenchmarkRunCommitStore
from scholar_agent.evaluation.resource_accounting import (
    EXIT_NOT_ELIGIBLE,
    EXIT_PASSED,
    EXIT_VIOLATION,
    QueryResourceLedger,
    ResourceLedgerObserver,
    ResourceLedgerV1,
    audit_evidence_registry,
    audit_frozen_eligibility,
    audit_shard_aggregate,
    build_run_ledger,
    deterministic_fixture_report,
    load_gate_protocol,
    opaque_resource_identity,
    stable_report_bytes,
    validate_resource_ledger,
)
from scholar_agent.evaluation.sharded_execution import ShardAggregateV1
from scholar_agent.evaluation.snapshot_resume import stable_hash
from scholar_agent.evaluation.experiment_pairing import opaque_query_identity
from scholar_agent.services.search_budget import BudgetedLLMClient, SearchBudgetRuntime
from scholar_agent.services.search_service import SearchService


pytestmark = pytest.mark.resource_accounting_integrity_regression


def _invariants(report: dict[str, Any]) -> set[str]:
    return {str(item["invariant"]) for item in report.get("violations", [])}


def _opaque(kind: str, value: str) -> str:
    return opaque_resource_identity(kind, value)


def test_fixture_covers_success_retry_pagination_cache_timeout_cancel_and_unknown() -> None:
    first = deterministic_fixture_report(shard_resume=True)
    second = deterministic_fixture_report(shard_resume=True)

    assert first == second
    assert stable_report_bytes(first) == stable_report_bytes(second)
    assert first["exit_code"] == EXIT_PASSED
    assert first["fixture"] == {
        "query_count": 7,
        "operation_count": first["fixture"]["operation_count"],
        "retry_and_pagination_covered": True,
        "cancel_covered": True,
        "resume_and_shard_selection_covered": True,
        "unknown_token_and_cost_preserved": True,
    }
    protocol = load_gate_protocol(
        Path("benchmark/resource_accounting_integrity_v1_protocol.json")
    )
    assert protocol["ledger_contract"] == "resource_ledger_v1"


@pytest.mark.parametrize(
    ("fault", "invariant"),
    [
        ("double_charge", "query_budget_conservation"),
        ("missing_call", "query_totals_conservation"),
        ("fake_cache_consumption", "cache_hit_external_consumption"),
        ("negative_remaining", "ledger_schema"),
        ("over_budget", "budget_limit_not_exceeded"),
        ("post_cancel", "unauthorized_post_cancel_consumption"),
        ("stale_attempt", "query_authority_binding"),
    ],
)
def test_controlled_accounting_faults_are_stable(fault: str, invariant: str) -> None:
    first = deterministic_fixture_report(controlled_fault=fault)
    second = deterministic_fixture_report(controlled_fault=fault)

    assert first == second
    assert first["exit_code"] == EXIT_VIOLATION
    assert invariant in _invariants(first)
    if fault != "negative_remaining":
        contextual = next(
            item for item in first["violations"] if item["invariant"] == invariant
        )
        assert len(contextual["run_identity"]) == 64
        if str(contextual["ledger_path"]).startswith("$.queries["):
            assert len(contextual["query_identity"]) == 64
            assert len(contextual["attempt_identity"]) == 64


def test_provider_tokens_and_cost_remain_unknown_instead_of_zero() -> None:
    ledger = _fixture_ledger()
    llm = next(
        operation
        for query in ledger.queries
        for operation in query.operations
        if operation.operation_type == "llm_call"
    )

    assert llm.prompt_tokens.state == "not_available"
    assert llm.prompt_tokens.value is None
    assert llm.total_tokens.state == "not_available"
    assert llm.provider_cost.state == "not_available"
    assert ledger.totals.total_tokens.state == "not_available"
    assert ledger.totals.provider_cost.state == "not_available"


def test_observer_is_independent_of_connector_event_arrival_order() -> None:
    budget = SearchBudget(max_search_rounds=2, max_candidate_papers=10)
    payloads = [
        (
            "connector_started",
            {"query_index": 1, "source": "pubmed", "adapted_query": "b"},
        ),
        (
            "connector_completed",
            {
                "query_index": 1,
                "source": "pubmed",
                "adapted_query": "b",
                "request_count": 1,
                "returned_count": 2,
            },
        ),
        (
            "connector_started",
            {"query_index": 0, "source": "arxiv", "adapted_query": "a"},
        ),
        (
            "connector_completed",
            {
                "query_index": 0,
                "source": "arxiv",
                "adapted_query": "a",
                "request_count": 2,
                "retry_count": 1,
                "returned_count": 1,
            },
        ),
    ]

    def build(events: list[tuple[str, dict[str, Any]]]) -> QueryResourceLedger:
        observer = ResourceLedgerObserver(budget)
        for name, payload in events:
            observer.observe_semantic_event(name, payload)
        observer.observe_budget_event(
            "budget_finalized",
            {
                "completed_search_rounds": 1,
                "candidate_count": 3,
                "elapsed_seconds": 0.5,
                "stop_reasons": [],
            },
        )
        return observer.build_query_ledger(
            run_identity=_opaque("run", "run"),
            query_identity=_opaque("query", "query"),
            attempt_identity=_opaque("attempt", "attempt-0"),
            checkpoint_generation=1,
            manifest_identity="a" * 64,
            terminal_status="succeeded",
        )

    first = build(payloads)
    second = build(payloads[2:] + payloads[:2])

    assert [item.operation_identity for item in first.operations] == [
        item.operation_identity for item in second.operations
    ]
    assert first.totals == second.totals


def test_observer_preserves_numeric_connector_cache_hit_counts() -> None:
    observer = ResourceLedgerObserver(SearchBudget())
    observer.observe_semantic_event(
        "connector_started",
        {"query_index": 0, "source": "local_bm25", "adapted_query": "fixture"},
    )
    observer.observe_semantic_event(
        "connector_completed",
        {
            "query_index": 0,
            "source": "local_bm25",
            "adapted_query": "fixture",
            "request_count": 0,
            "returned_count": 10,
            "cache_hit": False,
            "cache_hit_count": 3,
        },
    )
    observer.observe_budget_event(
        "budget_finalized",
        {
            "completed_search_rounds": 1,
            "candidate_count": 10,
            "elapsed_seconds": 0.5,
            "stop_reasons": [],
        },
    )
    ledger = observer.build_query_ledger(
        run_identity=_opaque("run", "run"),
        query_identity=_opaque("query", "query"),
        attempt_identity=_opaque("attempt", "attempt-0"),
        checkpoint_generation=1,
        manifest_identity="c" * 64,
        terminal_status="succeeded",
    )
    adapter = next(
        item for item in ledger.operations if item.operation_type == "adapter_call"
    )

    assert adapter.cache_status == "miss"
    assert adapter.cache_hit_count == 3
    assert ledger.totals.cache_hit_count == 3


def test_budget_runtime_observes_existing_llm_accounting_without_estimating_cost() -> None:
    observer = ResourceLedgerObserver(SearchBudget())
    runtime = SearchBudgetRuntime(
        SearchBudget(), resource_accounting_observer=observer
    )
    client = _NoUsageLLM()

    assert BudgetedLLMClient(client, runtime).chat_json([]) == {"ok": True}
    runtime.record_search_round()
    runtime.record_candidate_count(2)
    runtime.finalize_resource_accounting(2)
    ledger = observer.build_query_ledger(
        run_identity=_opaque("run", "run"),
        query_identity=_opaque("query", "query"),
        attempt_identity=_opaque("attempt", "attempt-0"),
        checkpoint_generation=1,
        manifest_identity="b" * 64,
        terminal_status="succeeded",
    )

    assert runtime.used_llm_calls == 1
    assert ledger.totals.llm_call_count == 1
    assert ledger.totals.total_tokens.state == "not_available"
    assert validate_resource_ledger(
        ResourceLedgerV1(
            run_identity=_opaque("run", "run"),
            manifest_identity="b" * 64,
            expected_query_identities=[_opaque("query", "query")],
            queries=[ledger],
            totals=ledger.totals,
            budget=ledger.budget,
            selected_attempts={
                _opaque("query", "query"): _opaque("attempt", "attempt-0")
            },
        )
    )["status"] == "passed"


def test_explicit_zero_provider_usage_is_known_while_cost_remains_unknown() -> None:
    observer = ResourceLedgerObserver(SearchBudget())
    runtime = SearchBudgetRuntime(
        SearchBudget(), resource_accounting_observer=observer
    )
    BudgetedLLMClient(_ZeroUsageLLM(), runtime).chat_json([])
    runtime.finalize_resource_accounting(0)
    ledger = observer.build_query_ledger(
        run_identity=_opaque("run", "known-zero"),
        query_identity=_opaque("query", "known-zero"),
        attempt_identity=_opaque("attempt", "known-zero"),
        checkpoint_generation=1,
        manifest_identity="d" * 64,
        terminal_status="succeeded",
    )
    operation = next(
        item for item in ledger.operations if item.operation_type == "llm_call"
    )

    assert operation.total_tokens.state == "known"
    assert operation.total_tokens.value == 0
    assert operation.provider_cost.state == "not_available"


def test_connector_error_details_are_classified_without_sensitive_echo() -> None:
    observer = ResourceLedgerObserver(SearchBudget())
    observer.observe_semantic_event(
        "connector_started",
        {"source": "openalex", "adapted_query": "opaque"},
    )
    observer.observe_semantic_event(
        "connector_completed",
        {
            "source": "openalex",
            "adapted_query": "opaque",
            "request_count": 1,
            "error_message": "Bearer sentinel-secret from /private/.env",
        },
    )
    ledger = observer.build_query_ledger(
        run_identity=_opaque("run", "redaction"),
        query_identity=_opaque("query", "redaction"),
        attempt_identity=_opaque("attempt", "redaction"),
        checkpoint_generation=1,
        manifest_identity="e" * 64,
        terminal_status="failed",
    )
    serialized = ledger.model_dump_json()

    assert "sentinel-secret" not in serialized
    assert "/private/.env" not in serialized


def test_search_service_observation_does_not_change_results_or_events() -> None:
    retriever = _ObservedRetriever()
    service = SearchService(retriever=retriever, max_workers=1)
    budget = SearchBudget(max_search_rounds=2, max_candidate_papers=20)
    baseline_events: list[tuple[str, dict[str, Any]]] = []
    observed_events: list[tuple[str, dict[str, Any]]] = []

    baseline = service.run_search(
        "offline deterministic query",
        sources_override=["openalex"],
        enable_synthesis=False,
        current_year=2025,
        budget=budget,
        event_callback=lambda name, payload: baseline_events.append((name, payload)),
    )
    observer = ResourceLedgerObserver(budget)
    observed = service.run_search(
        "offline deterministic query",
        sources_override=["openalex"],
        enable_synthesis=False,
        current_year=2025,
        budget=budget,
        event_callback=lambda name, payload: observed_events.append((name, payload)),
        resource_accounting_observer=observer,
    )

    assert [item.paper.identifiers.doi for item in baseline.ranked_papers] == [
        item.paper.identifiers.doi for item in observed.ranked_papers
    ]
    assert baseline.raw_count == observed.raw_count
    assert baseline.deduplicated_count == observed.deduplicated_count
    assert [name for name, _ in baseline_events] == [name for name, _ in observed_events]
    ledger = observer.build_query_ledger(
        run_identity=_opaque("run", "run"),
        query_identity=_opaque("query", "query"),
        attempt_identity=_opaque("attempt", "attempt-0"),
        checkpoint_generation=1,
        manifest_identity="c" * 64,
        terminal_status="succeeded",
    )
    assert ledger.totals.api_request_count.value
    assert ledger.totals.retry_count == 0


def test_committed_generation_accepts_ledger_report_and_legacy_remains_ineligible(
    tmp_path: Path,
) -> None:
    ledger = _fixture_ledger()
    store = BenchmarkRunCommitStore(tmp_path / "run")
    state = store.initialize(
        run_id="run",
        expected_query_ids=[f"case-{index}" for index in range(7)],
        config={"resume_signature": "a" * 64},
        dataset_report={"count": 7},
    )
    for index in range(7):
        state = store.commit_record(
            {"case_id": f"case-{index}", "status": "succeeded"}
        )
    state = store.commit_completion(
        {"resource_ledger.json": json.dumps(ledger.model_dump(mode="json")).encode()}
    )

    assert state.status == "completed"
    assert "resource_ledger.json" in store.public_artifacts(state)
    frozen = audit_frozen_eligibility(
        Path("benchmark/run_provenance_legacy_audit.json")
    )
    assert frozen["exit_code"] == EXIT_NOT_ELIGIBLE
    assert all(item["status"] == "not_eligible" for item in frozen["profiles"])


def test_evidence_registry_cost_claims_are_read_only_not_eligible() -> None:
    report = audit_evidence_registry(
        Path("benchmark/evidence_registry_baseline/registry.json")
    )

    assert report["registry_mutated"] is False
    assert report["exit_code"] in {EXIT_PASSED, EXIT_NOT_ELIGIBLE}
    assert all(item["status"] == "not_eligible" for item in report["cost_claims"])


def test_authoritative_record_detects_unobserved_adapter_call() -> None:
    ledger = _fixture_ledger()
    first = ledger.queries[0]
    record = {
        "resource_ledger": first.model_dump(mode="json"),
        "cost_report": {
            "api_call_count": 99,
            "retry_count": first.totals.retry_count,
            "cache_hit_count": first.totals.cache_hit_count,
            "llm_call_count": first.totals.llm_call_count,
        },
    }

    report = validate_resource_ledger(ledger, authoritative_records=[record])

    assert report["exit_code"] == EXIT_VIOLATION
    assert "adapter_llm_call_ledger_completeness" in _invariants(report)


def test_resume_selects_only_committed_final_attempt() -> None:
    ledger = _fixture_ledger()
    query = ledger.queries[0]
    observer = ResourceLedgerObserver(SearchBudget())
    observer.observe_budget_event(
        "budget_finalized",
        {
            "completed_search_rounds": 0,
            "candidate_count": 0,
            "elapsed_seconds": 0.1,
            "stop_reasons": [],
        },
    )
    old_attempt = _opaque("attempt", "attempt-old")
    old = observer.build_query_ledger(
        run_identity=ledger.run_identity,
        query_identity=query.query_identity,
        attempt_identity=old_attempt,
        checkpoint_generation=0,
        manifest_identity=ledger.manifest_identity,
        terminal_status="failed",
    )
    rebuilt = build_run_ledger(
        [old, *ledger.queries],
        run_identity=ledger.run_identity,
        manifest_identity=ledger.manifest_identity,
        expected_query_identities=ledger.expected_query_identities,
        selected_attempts=ledger.selected_attempts,
        superseded_attempts=[old_attempt],
    )

    assert len(rebuilt.queries) == len(ledger.queries)
    assert rebuilt.queries[0].attempt_identity == ledger.selected_attempts[
        query.query_identity
    ]
    assert rebuilt.superseded_attempts == [old_attempt]
    assert validate_resource_ledger(rebuilt)["status"] == "passed"


def test_shard_aggregate_counts_only_selected_ledger_references(tmp_path: Path) -> None:
    source = _fixture_ledger()
    groups = [source.queries[:3], source.queries[3:]]
    references = []
    selected = []
    records = []
    for shard_index, queries in enumerate(groups):
        expected = [item.query_identity for item in queries]
        ledger = build_run_ledger(
            queries,
            run_identity=source.run_identity,
            manifest_identity=source.manifest_identity,
            expected_query_identities=expected,
        )
        path = tmp_path / f"ledger-{shard_index}.json"
        path.write_text(ledger.model_dump_json(), encoding="utf-8")
        digest = _sha256(path)
        references.append(
            {
                "shard_index": shard_index,
                "attempt_id": "attempt-final",
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": digest,
                "manifest_identity": source.manifest_identity,
            }
        )
        selected.append(
            {
                "shard_index": shard_index,
                "attempt_id": "attempt-final",
                "manifest_path": f"manifest-{shard_index}.json",
                "manifest_sha256": "a" * 64,
                "run_id": f"shard-{shard_index}",
                "commit_generation": 1,
                "generation_manifest_sha256": "b" * 64,
                "record_count": len(queries),
                "event_count": len(queries),
            }
        )
        records.extend(
            {"case_id": f"case-{len(records) + offset}", "status": "succeeded"}
            for offset in range(len(queries))
        )
    identities = [opaque_query_identity(str(row["case_id"])) for row in records]
    payload = {
        "schema_version": "1",
        "contract": "shard_aggregate_v1",
        "score_scope": "partition_and_merge_only_not_quality_or_official_score",
        "plan_path": "plan.json",
        "plan_sha256": "c" * 64,
        "query_count": len(records),
        "query_order_sha256": stable_hash(identities),
        "selected_shards": selected,
        "records": records,
        "commit_events": [],
        "terminal_counts": {"succeeded": len(records)},
        "operational_counts": {},
        "resource_ledgers": references,
        "completed": True,
    }
    aggregate = ShardAggregateV1(
        **payload, aggregate_summary_sha256=stable_hash(payload)
    )
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(aggregate.model_dump_json(), encoding="utf-8")

    report = audit_shard_aggregate(aggregate_path, repository_root=tmp_path)

    assert report["exit_code"] == EXIT_PASSED
    changed = aggregate.model_dump(mode="json")
    changed["resource_ledgers"][0]["attempt_id"] = "attempt-superseded"
    summary = {
        key: value
        for key, value in changed.items()
        if key != "aggregate_summary_sha256"
    }
    changed["aggregate_summary_sha256"] = stable_hash(summary)
    aggregate_path.write_text(json.dumps(changed), encoding="utf-8")
    failed = audit_shard_aggregate(aggregate_path, repository_root=tmp_path)
    assert failed["exit_code"] == EXIT_VIOLATION
    assert "superseded_or_unselected_shard_ledger" in _invariants(failed)


class _NoUsageLLM:
    token_usage = None

    def chat_json(self, *_args: Any, **_kwargs: Any) -> dict[str, bool]:
        return {"ok": True}


class _ZeroUsageLLM:
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def chat_json(self, *_args: Any, **_kwargs: Any) -> dict[str, bool]:
        return {"ok": True}


class _ObservedRetriever:
    emits_connector_events = True

    def __getstate__(self) -> None:
        raise TypeError("fixture stays in process")

    def __call__(
        self,
        query: str,
        *,
        limit_per_source: int,
        sources: list[str],
        connector_event_callback: Any,
        **_kwargs: Any,
    ) -> RetrievalOutput:
        paper = Paper(
            title="Offline fixture paper",
            authors=["Fixture Author"],
            year=2024,
            abstract="offline deterministic query evidence",
            identifiers=PaperIdentifiers(doi="10.1000/offline-fixture"),
            sources=["openalex"],
        )
        connector_event_callback(
            "connector_started",
            {
                "source": "openalex",
                "adapted_query": query,
                "adaptation_strategy": "fixture",
            },
        )
        stats = SourceStats(
            source="openalex",
            query=query,
            adapted_query=query,
            adaptation_strategy="fixture",
            returned_count=1,
            diagnostic_papers=[paper],
            diagnostics=ConnectorDiagnostics(request_count=1),
        )
        connector_event_callback(
            "connector_completed",
            {
                "source": "openalex",
                "adapted_query": query,
                "adaptation_strategy": "fixture",
                "returned_count": 1,
                "request_count": 1,
                "retry_count": 0,
                "cache_hit": False,
            },
        )
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=1,
            deduplicated_count=1,
            papers=[paper],
            source_stats=[stats],
        )


def _fixture_ledger() -> ResourceLedgerV1:
    # The fixture builder is intentionally reached through the public report
    # once, while this helper imports the deterministic model for assertions.
    from scholar_agent.evaluation.resource_accounting import _fixture_run_ledger

    return _fixture_run_ledger(shard_resume=True)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
