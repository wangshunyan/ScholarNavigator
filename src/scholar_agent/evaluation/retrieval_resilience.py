"""Offline four-source retrieval resilience gate.

The gate injects deterministic connector-boundary outcomes through the same
provider hook used by Snapshot Replay, then executes the real SearchService
pipeline. It never reads gold, computes quality metrics, or calls a network.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.agents.retriever import (
    RetrievalOutput,
    SourceStats,
    clear_retrieval_cache,
    clear_source_cooldowns,
    retrieve_papers,
)
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import SearchBudget
from scholar_agent.evaluation.execution_determinism import (
    CanonicalizationRule,
    canonicalize_explicit_fields,
    forbid_network,
    tree_signature,
)
from scholar_agent.evaluation.metrics import canonical_paper_id
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.services.search_service import SearchService, SearchServiceOutput


CONTRACT_VERSION = "retrieval_resilience_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "retrieval_resilience_gate"
EXIT_PASSED = 0
EXIT_INVARIANT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
FOUR_SOURCES = ("openalex", "arxiv", "semantic_scholar", "pubmed")
DEFAULT_SNAPSHOT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "benchmark_snapshots"


FaultOutcome = Literal[
    "healthy",
    "timeout",
    "rate_limit",
    "connection_failure",
    "malformed_json",
    "missing_critical_field",
    "invalid_type",
    "pagination_loop",
    "duplicate_records",
    "identity_conflict",
    "empty_response",
    "not_started",
    "partial_then_failure",
    "unexpected_exception",
]
SourceTerminal = Literal[
    "success",
    "success_empty",
    "partial_completion",
    "failed",
    "not_started",
]


class RetrievalResilienceError(RuntimeError):
    """Malformed protocol or a gate execution that cannot be audited."""


class ResilienceNotEligible(RetrievalResilienceError):
    """The requested frozen or local fixture cannot support source faults."""


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(min_length=1)
    outcomes: dict[str, FaultOutcome] = Field(default_factory=dict)
    expected_terminal: Literal["completed", "partial_failure", "all_sources_failed"]


class ProviderCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limit: int = Field(ge=1)
    outcome: FaultOutcome
    request_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    snapshot_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    sources: dict[str, list[Paper]]


class FaultInjectedReplayRetriever:
    """Snapshot-provider fault injection over the production retriever path."""

    emits_connector_events = True

    def __init__(
        self,
        scenario: ScenarioSpec,
        fixture: FixtureBundle,
        *,
        controlled_fault: str | None = None,
    ) -> None:
        self.scenario = scenario
        self.fixture = fixture
        self.controlled_fault = controlled_fault
        self._lock = RLock()
        self._calls: list[ProviderCall] = []
        self._fault_used = False
        self._allowed_papers: dict[str, Paper] = {}
        self._expected_papers: dict[str, Paper] = {}
        clear_retrieval_cache()
        clear_source_cooldowns()

    @property
    def calls(self) -> list[ProviderCall]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._calls]

    @property
    def allowed_candidate_ids(self) -> set[str]:
        with self._lock:
            return set(self._allowed_papers)

    @property
    def expected_candidate_ids(self) -> set[str]:
        with self._lock:
            return set(self._expected_papers)

    def __call__(self, query: str, **kwargs: object) -> RetrievalOutput:
        return retrieve_papers(
            query,
            **kwargs,
            connector_result_provider=self._provide,
            replay_recorded_terminals=True,
            recorded_terminal_lookup=self._is_recorded_terminal,
        )

    def _is_recorded_terminal(
        self,
        source: str,
        _adapted_query: str,
        _limit: int,
    ) -> bool:
        return self.scenario.outcomes.get(source, "healthy") != "not_started"

    def _provide(
        self,
        source: str,
        adapted_query: str,
        limit: int,
        _policy: str,
        _live_search: Callable[[str, int], ConnectorSearchResult],
    ) -> ConnectorSearchResult:
        outcome = self.scenario.outcomes.get(source, "healthy")
        snapshot_key = stable_hash(
            {
                "contract": CONTRACT_VERSION,
                "scenario": self.scenario.scenario,
                "source": source,
                "query": " ".join(adapted_query.casefold().split()),
                "limit": limit,
            }
        )
        result, raised_error = self._outcome_result(source, outcome, snapshot_key)
        diagnostics = (
            result.diagnostics
            if result is not None
            else ConnectorDiagnostics(error_count=1)
        )
        if self.controlled_fault == "budget_overrun" and not self._fault_used:
            diagnostics = diagnostics.model_copy(
                update={"request_count": 3, "retry_count": 2}
            )
            self._fault_used = True
            if result is not None:
                result = result.model_copy(update={"diagnostics": diagnostics})
        call = ProviderCall(
            source=source,
            query_sha256=stable_hash(
                {"query": " ".join(adapted_query.casefold().split())}
            ),
            limit=limit,
            outcome=outcome,
            request_count=diagnostics.request_count,
            retry_count=diagnostics.retry_count,
            error_count=diagnostics.error_count,
            returned_count=len(result.papers) if result is not None else 0,
            snapshot_key=snapshot_key,
        )
        with self._lock:
            self._calls.append(call)
            if result is not None:
                for paper in result.papers:
                    identity = _paper_identity(paper)
                    self._allowed_papers[identity] = paper.model_copy(deep=True)
                    self._expected_papers[identity] = paper.model_copy(deep=True)
        if raised_error is not None:
            raise raised_error
        assert result is not None
        return result

    def _outcome_result(
        self,
        source: str,
        outcome: FaultOutcome,
        snapshot_key: str,
    ) -> tuple[ConnectorSearchResult | None, Exception | None]:
        base = [item.model_copy(deep=True) for item in self.fixture.sources[source]]
        if outcome == "healthy":
            return _replay_result(source, snapshot_key, base), None
        if outcome == "duplicate_records":
            return _replay_result(source, snapshot_key, [*base, base[0], base[0]]), None
        if outcome == "identity_conflict":
            conflicts = _identity_conflict_papers(source)
            return _replay_result(source, snapshot_key, [*base, *conflicts]), None
        if outcome == "empty_response":
            return _replay_result(source, snapshot_key, []), None
        if outcome == "not_started":
            raise RetrievalResilienceError("not_started_outcome_must_not_execute")
        if outcome == "partial_then_failure":
            return (
                _replay_result(
                    source,
                    snapshot_key,
                    base[-1:],
                    error_message="partial_response_terminated",
                    diagnostics=ConnectorDiagnostics(request_count=2, error_count=1),
                ),
                None,
            )
        if outcome == "timeout":
            return _failed_result(snapshot_key, "connector_timeout", retry=True), None
        if outcome == "rate_limit":
            return (
                _failed_result(
                    snapshot_key,
                    "http_status:429",
                    retry=True,
                    rate_limit_wait_seconds=0.5,
                ),
                None,
            )
        if outcome == "connection_failure":
            return _failed_result(snapshot_key, "connection_failure", retry=True), None
        if outcome == "pagination_loop":
            return _failed_result(snapshot_key, "pagination_loop_detected", retry=True), None
        if outcome == "malformed_json":
            return None, ValueError("adapter_malformed_json")
        if outcome == "missing_critical_field":
            return None, ValueError("adapter_missing_critical_field")
        if outcome == "invalid_type":
            return None, TypeError("adapter_invalid_field_type")
        if outcome == "unexpected_exception":
            return (
                None,
                RuntimeError(
                    "unexpected_adapter_exception Authorization: Bearer "
                    "sensitive-bearer-value "
                    "api_key=private-value .env /Users/example/private/raw.json"
                ),
            )
        raise RetrievalResilienceError(f"unsupported_outcome:{outcome}")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalResilienceError("protocol_unreadable") from exc
    if not isinstance(protocol, dict):
        raise RetrievalResilienceError("protocol_root_must_be_object")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("contract") != CONTRACT_VERSION
    ):
        raise RetrievalResilienceError("protocol_version_incompatible")
    if protocol.get("score_scope") != "resilience_only_not_quality_or_official_score":
        raise RetrievalResilienceError("protocol_score_scope_invalid")
    if tuple(protocol.get("sources") or []) != FOUR_SOURCES:
        raise RetrievalResilienceError("four_source_order_invalid")
    fixture = protocol.get("fixture")
    if not isinstance(fixture, dict):
        raise RetrievalResilienceError("fixture_contract_missing")
    fixture_path = _repo_path(repository_root, str(fixture.get("path") or ""))
    if not fixture_path.is_file():
        raise ResilienceNotEligible("fixture_missing")
    if fixture_path.stat().st_size != fixture.get("size_bytes"):
        raise ResilienceNotEligible("fixture_size_drift")
    if sha256_file(fixture_path) != fixture.get("sha256"):
        raise ResilienceNotEligible("fixture_hash_drift")
    scenarios = [ScenarioSpec.model_validate(item) for item in protocol.get("scenarios", [])]
    if not scenarios or len({item.scenario for item in scenarios}) != len(scenarios):
        raise RetrievalResilienceError("scenario_matrix_missing_or_duplicate")
    unknown_sources = sorted(
        {
            source
            for scenario in scenarios
            for source in scenario.outcomes
            if source not in FOUR_SOURCES
        }
    )
    if unknown_sources:
        raise RetrievalResilienceError("scenario_unknown_source")
    observed_outcomes = {
        outcome for scenario in scenarios for outcome in scenario.outcomes.values()
    }
    required_outcomes = set(FaultOutcome.__args__) - {"healthy"}  # type: ignore[attr-defined]
    if not required_outcomes <= observed_outcomes:
        raise RetrievalResilienceError("scenario_outcome_coverage_incomplete")
    execution = protocol.get("execution")
    if not isinstance(execution, dict):
        raise RetrievalResilienceError("execution_contract_missing")
    arguments = execution.get("search_service_arguments")
    if not isinstance(arguments, dict):
        raise RetrievalResilienceError("search_service_arguments_missing")
    for field in (
        "enable_llm_judgement",
        "enable_llm_query_understanding",
        "enable_query_evolution",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_synthesis",
    ):
        if arguments.get(field) is not False:
            raise RetrievalResilienceError(f"offline_feature_must_be_disabled:{field}")
    rules = protocol.get("canonicalization", {}).get("excluded_fields")
    if not isinstance(rules, list) or not rules:
        raise RetrievalResilienceError("canonicalization_rules_missing")
    parsed_rules = [CanonicalizationRule.model_validate(item) for item in rules]
    paths = [item.path for item in parsed_rules]
    if len(paths) != len(set(paths)):
        raise RetrievalResilienceError("duplicate_canonicalization_rule")
    return protocol


def load_fixture(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> FixtureBundle:
    path = _repo_path(repository_root, str(protocol["fixture"]["path"]))
    fixture = FixtureBundle.model_validate_json(path.read_text(encoding="utf-8"))
    if set(fixture.sources) != set(FOUR_SOURCES):
        raise ResilienceNotEligible("fixture_source_coverage_invalid")
    if any(not fixture.sources[source] for source in FOUR_SOURCES):
        raise ResilienceNotEligible("fixture_source_empty")
    return fixture


def run_retrieval_resilience(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    controlled_fault: str | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    if controlled_fault not in {None, "budget_overrun"}:
        raise RetrievalResilienceError("unsupported_controlled_fault")
    fixture = load_fixture(protocol, repository_root=repository_root)
    scenarios = [ScenarioSpec.model_validate(item) for item in protocol["scenarios"]]
    rules = [
        CanonicalizationRule.model_validate(item)
        for item in protocol["canonicalization"]["excluded_fields"]
    ]
    execution = protocol["execution"]
    arguments = copy.deepcopy(execution["search_service_arguments"])
    budget = SearchBudget.model_validate(arguments.pop("budget"))
    max_calls_per_subquery = int(execution["max_adapter_calls_per_source_per_subquery"])
    max_requests_per_call = int(execution["max_physical_requests_per_adapter_call"])
    snapshot_before = tree_signature(snapshot_root)
    attempts = {"network": 0}
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    with forbid_network(attempts):
        for scenario_index, scenario in enumerate(scenarios):
            row, row_violations = _run_scenario(
                scenario,
                fixture,
                arguments=arguments,
                budget=budget,
                rules=rules,
                max_calls_per_subquery=max_calls_per_subquery,
                max_requests_per_call=max_requests_per_call,
                controlled_fault=(controlled_fault if scenario_index == 0 else None),
            )
            rows.append(row)
            violations.extend(row_violations)
    snapshot_after = tree_signature(snapshot_root)
    snapshot_write_count = int(snapshot_before != snapshot_after)
    if attempts["network"]:
        violations.append(
            _violation(
                "gate",
                None,
                "offline_execution",
                "$.execution.network_request_count",
                0,
                attempts["network"],
            )
        )
    if snapshot_write_count:
        violations.append(
            _violation(
                "gate",
                None,
                "snapshot_read_only",
                "$.execution.snapshot_tree_sha256",
                snapshot_before,
                snapshot_after,
            )
        )
    violations = sorted(
        violations,
        key=lambda item: (
            str(item["scenario"]),
            str(item.get("source") or ""),
            str(item["invariant"]),
            str(item["first_difference_path"]),
        ),
    )
    status = "passed" if not violations else "invariant_violation"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if not violations else EXIT_INVARIANT_VIOLATION,
        "score_scope": "resilience_only_not_quality_or_official_score",
        "sources": list(FOUR_SOURCES),
        "scenario_count": len(rows),
        "scenarios": sorted(rows, key=lambda item: item["scenario"]),
        "violation_count": len(violations),
        "violations": violations,
        "frozen_baselines": _legacy_eligibility(protocol, repository_root),
        "execution": {
            "network_request_count": attempts["network"],
            "llm_request_count": 0,
            "snapshot_write_count": snapshot_write_count,
            "quality_metric_count": 0,
            "controlled_fault": controlled_fault,
        },
    }


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    profiles = _legacy_eligibility(protocol, repository_root)
    eligible_count = sum(item["status"] == "eligible" for item in profiles)
    status = "passed" if eligible_count == len(profiles) else "not_eligible"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": EXIT_PASSED if status == "passed" else EXIT_NOT_ELIGIBLE,
        "score_scope": "resilience_only_not_quality_or_official_score",
        "profiles": profiles,
        "eligible_count": eligible_count,
        "profile_count": len(profiles),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_scenario(
    scenario: ScenarioSpec,
    fixture: FixtureBundle,
    *,
    arguments: Mapping[str, Any],
    budget: SearchBudget,
    rules: Sequence[CanonicalizationRule],
    max_calls_per_subquery: int,
    max_requests_per_call: int,
    controlled_fault: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = FaultInjectedReplayRetriever(
        scenario,
        fixture,
        controlled_fault=controlled_fault,
    )
    service = SearchService(retriever=retriever, max_workers=1)
    events: list[dict[str, Any]] = []
    output: SearchServiceOutput = service.run_search(
        str(arguments["query"]),
        top_k=int(arguments["top_k"]),
        run_profile=str(arguments["run_profile"]),  # type: ignore[arg-type]
        enable_refchain=False,
        enable_semantic_seed_expansion=False,
        enable_query_evolution=False,
        query_evolution_policy="off",
        query_planning_policy="current_rules",
        ranking_policy="current_rules",
        enable_synthesis=False,
        current_year=int(arguments["current_year"]),
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        sources_override=list(FOUR_SOURCES),
        budget=budget,
        collect_diagnostics=True,
        query_adapter_policy=str(arguments["query_adapter_policy"]),  # type: ignore[arg-type]
        event_callback=lambda name, payload: events.append(
            {"event": name, "payload": copy.deepcopy(payload)}
        ),
    )
    source_rows = _source_terminal_rows(output.source_stats)
    terminal = _scenario_terminal(source_rows)
    actual_ids = {
        _paper_identity(item.paper) for item in output.all_ranked_papers
    }
    candidate_order = [
        _paper_identity(item.paper) for item in output.all_ranked_papers
    ]
    canonical, exclusion_counts = canonicalize_explicit_fields(
        {"output": output.model_dump(mode="json"), "events": events},
        rules,
    )
    calls = retriever.calls
    call_counts = Counter(item.source for item in calls)
    request_counts = Counter()
    for item in calls:
        request_counts[item.source] += item.request_count
    violations: list[dict[str, Any]] = []

    missing_expected = sorted(retriever.expected_candidate_ids - actual_ids)
    if missing_expected:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "healthy_results_preserved",
                "$.candidates.missing_expected_identities",
                [],
                missing_expected,
            )
        )
    unexpected = sorted(actual_ids - retriever.allowed_candidate_ids)
    if unexpected:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "fault_payload_isolation",
                "$.candidates.unexpected_identities",
                [],
                unexpected,
            )
        )
    if len(candidate_order) != len(set(candidate_order)):
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "unified_identity_deduplication",
                "$.candidates.identity_order",
                "unique",
                candidate_order,
            )
        )
    expected_source_statuses = {
        source: _expected_source_terminal(scenario.outcomes.get(source, "healthy"))
        for source in FOUR_SOURCES
    }
    observed_source_statuses = {
        item["source"]: item["status"] for item in source_rows
    }
    for source in FOUR_SOURCES:
        if observed_source_statuses[source] != expected_source_statuses[source]:
            violations.append(
                _violation(
                    scenario.scenario,
                    source,
                    "source_terminal_attribution",
                    f"$.sources.{source}.status",
                    expected_source_statuses[source],
                    observed_source_statuses[source],
                )
            )
    if terminal != scenario.expected_terminal:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "scenario_terminal_status",
                "$.terminal_status",
                scenario.expected_terminal,
                terminal,
            )
        )
    if terminal == "all_sources_failed" and actual_ids:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "all_sources_failed_fail_closed",
                "$.candidates.identity_count",
                0,
                len(actual_ids),
            )
        )
    if output.search_plan.selected_sources != list(FOUR_SOURCES):
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "frozen_source_configuration",
                "$.output.search_plan.selected_sources",
                list(FOUR_SOURCES),
                output.search_plan.selected_sources,
            )
        )
    if (
        output.search_plan.query_planning_policy != "current_rules"
        or output.search_plan.ranking_policy != "current_rules"
        or output.judgement_policy != "current_rules"
        or output.query_evolution_records
        or output.refchain_output is not None
        or output.semantic_seed_expansion_output is not None
        or output.llm_call_count
    ):
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "no_hidden_strategy_switch",
                "$.output.policies",
                "current_rules_and_all_experiments_off",
                stable_hash(
                    {
                        "query_planning": output.search_plan.query_planning_policy,
                        "ranking": output.search_plan.ranking_policy,
                        "judgement": output.judgement_policy,
                        "query_evolution_records": len(output.query_evolution_records),
                        "refchain": output.refchain_output is not None,
                        "semantic_seed": output.semantic_seed_expansion_output is not None,
                        "llm_calls": output.llm_call_count,
                    }
                ),
            )
        )
    subquery_count = len(output.search_plan.subqueries)
    for source in FOUR_SOURCES:
        max_calls = subquery_count * max_calls_per_subquery
        if call_counts[source] > max_calls:
            violations.append(
                _violation(
                    scenario.scenario,
                    source,
                    "per_source_adapter_budget",
                    f"$.budget.calls.{source}",
                    max_calls,
                    call_counts[source],
                )
            )
    for index, call in enumerate(calls):
        if call.request_count > max_requests_per_call or call.retry_count > 1:
            violations.append(
                _violation(
                    scenario.scenario,
                    call.source,
                    "bounded_existing_retry_semantics",
                    f"$.provider_calls.{index}.request_count",
                    max_requests_per_call,
                    call.request_count,
                )
            )
        if call.limit > output.search_plan.limit_per_source:
            violations.append(
                _violation(
                    scenario.scenario,
                    call.source,
                    "per_source_candidate_budget",
                    f"$.provider_calls.{index}.limit",
                    output.search_plan.limit_per_source,
                    call.limit,
                )
            )
    if len(output.all_ranked_papers) > budget.max_candidate_papers:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "global_candidate_budget",
                "$.output.all_ranked_papers.length",
                budget.max_candidate_papers,
                len(output.all_ranked_papers),
            )
        )
    event_error = _event_sequence_error(events)
    if event_error is not None:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "event_sequence_legality",
                event_error,
                "balanced_and_ordered",
                "invalid",
            )
        )
    sensitive_path = _first_sensitive_path(
        {"output": output.model_dump(mode="json"), "events": events}
    )
    if sensitive_path is not None:
        violations.append(
            _violation(
                scenario.scenario,
                None,
                "sensitive_error_redaction",
                sensitive_path,
                "redacted",
                "sensitive_value_detected",
            )
        )
    return (
        {
            "scenario": scenario.scenario,
            "status": "passed" if not violations else "invariant_violation",
            "terminal_status": terminal,
            "source_terminals": source_rows,
            "candidate_identity_count": len(actual_ids),
            "candidate_identity_order_sha256": stable_hash(candidate_order),
            "expected_preserved_identity_count": len(
                retriever.expected_candidate_ids
            ),
            "provider_call_count": len(calls),
            "provider_call_source_counts": {
                source: call_counts[source] for source in FOUR_SOURCES
            },
            "physical_request_source_counts": {
                source: request_counts[source] for source in FOUR_SOURCES
            },
            "retry_count": sum(item.retry_count for item in calls),
            "error_count": sum(item.error_count for item in calls),
            "event_count": len(events),
            "semantic_output_sha256": stable_hash(canonical),
            "excluded_field_match_counts": exclusion_counts,
        },
        violations,
    )


def _source_terminal_rows(source_stats: Sequence[SourceStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in FOUR_SOURCES:
        source_rows = [item for item in source_stats if item.source == source]
        relevant = [
            item
            for item in source_rows
            if item.logical_call_executed
        ]
        skipped = [item for item in source_rows if not item.logical_call_executed]
        errors = [item for item in relevant if item.error_message is not None]
        returned = sum(len(item.diagnostic_papers) for item in relevant)
        if errors and returned:
            status: SourceTerminal = "partial_completion"
        elif errors:
            status = "failed"
        elif relevant and returned:
            status = "success"
        elif relevant:
            status = "success_empty"
        else:
            status = "not_started"
        reasons = sorted(
            {
                *(_error_reason(item.error_message) for item in errors),
                *(
                    item.source_skipped_reason
                    for item in skipped
                    if item.source_skipped_reason
                ),
            }
        )
        rows.append(
            {
                "source": source,
                "status": status,
                "logical_call_count": len(relevant),
                "skipped_call_count": len(skipped),
                "returned_record_count": returned,
                "error_call_count": len(errors),
                "reasons": reasons,
            }
        )
    return rows


def _scenario_terminal(source_rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(item["status"]) for item in source_rows]
    if all(status == "failed" for status in statuses):
        return "all_sources_failed"
    if any(status in {"failed", "partial_completion", "not_started"} for status in statuses):
        return "partial_failure"
    return "completed"


def _expected_source_terminal(outcome: FaultOutcome) -> SourceTerminal:
    if outcome in {"healthy", "duplicate_records", "identity_conflict"}:
        return "success"
    if outcome == "empty_response":
        return "success_empty"
    if outcome == "not_started":
        return "not_started"
    if outcome == "partial_then_failure":
        return "partial_completion"
    return "failed"


def _event_sequence_error(events: Sequence[Mapping[str, Any]]) -> str | None:
    names = [str(item.get("event") or "") for item in events]
    required = [
        "query_understanding_started",
        "query_understanding_completed",
        "retrieval_started",
        "retrieval_completed",
        "deduplication_completed",
        "judgement_started",
        "judgement_completed",
        "reranking_started",
        "reranking_completed",
    ]
    positions: list[int] = []
    for name in required:
        if names.count(name) != 1:
            return f"$.events.{name}.count"
        positions.append(names.index(name))
    if positions != sorted(positions):
        return "$.events.stage_order"
    balances: Counter[tuple[Any, ...]] = Counter()
    for index, item in enumerate(events):
        name = str(item.get("event") or "")
        if name not in {"connector_started", "connector_completed"}:
            continue
        payload = item.get("payload") or {}
        assert isinstance(payload, Mapping)
        key = (
            payload.get("query_index"),
            payload.get("source"),
            payload.get("adapted_query"),
        )
        if name == "connector_started":
            balances[key] += 1
        else:
            balances[key] -= 1
            if balances[key] < 0:
                return f"$.events.{index}.connector_completed"
    if any(value != 0 for value in balances.values()):
        return "$.events.connector_balance"
    return None


def _first_sensitive_path(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            found = _first_sensitive_path(value[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _first_sensitive_path(item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if not isinstance(value, str):
        return None
    forbidden = (
        "sensitive-bearer-value",
        "private-value",
        ".env",
        "/Users/",
        "/home/",
        "/tmp/",
        "raw.json",
    )
    if any(token in value for token in forbidden):
        return path
    if re.search(r"(?i)\bbearer\s+(?!\[redacted\])\S+", value):
        return path
    if re.search(r"(?i)\bapi[_-]?key\s*[:=]\s*(?!\[redacted\])\S+", value):
        return path
    return None


def _error_reason(value: str | None) -> str:
    text = (value or "").casefold()
    for reason in (
        "connector_timeout",
        "http_status:429",
        "connection_failure",
        "adapter_malformed_json",
        "adapter_missing_critical_field",
        "adapter_invalid_field_type",
        "pagination_loop_detected",
        "partial_response_terminated",
        "unexpected_adapter_exception",
    ):
        if reason in text:
            return reason
    return "source_failure"


def _replay_result(
    source: str,
    snapshot_key: str,
    papers: list[Paper],
    *,
    error_message: str | None = None,
    diagnostics: ConnectorDiagnostics | None = None,
) -> ConnectorSearchResult:
    effective = diagnostics or ConnectorDiagnostics(
        request_count=2 if source == "pubmed" else 1
    )
    return ConnectorSearchResult(
        papers=[item.model_copy(deep=True) for item in papers],
        error_message=error_message,
        warnings=[error_message] if error_message else [],
        diagnostics=effective,
        snapshot_provenance="snapshot_replay",
        snapshot_key=snapshot_key,
        snapshot_hit=True,
        recorded_diagnostics=effective,
        recorded_latency_seconds=0.0,
    )


def _failed_result(
    snapshot_key: str,
    error_message: str,
    *,
    retry: bool,
    rate_limit_wait_seconds: float = 0.0,
) -> ConnectorSearchResult:
    diagnostics = ConnectorDiagnostics(
        request_count=2 if retry else 1,
        retry_count=1 if retry else 0,
        error_count=2 if retry else 1,
        rate_limit_wait_seconds=rate_limit_wait_seconds,
    )
    return ConnectorSearchResult(
        error_message=error_message,
        warnings=[error_message],
        diagnostics=diagnostics,
        snapshot_provenance="snapshot_replay",
        snapshot_key=snapshot_key,
        snapshot_hit=True,
        recorded_diagnostics=diagnostics,
        recorded_latency_seconds=0.0,
    )


def _identity_conflict_papers(source: str) -> list[Paper]:
    common = {
        "title": "Conflicting Stable Identifiers in Retrieval Records",
        "authors": ["Jordan Lee"],
        "year": 2025,
        "abstract": "A fixture for conservative identity conflict handling.",
        "sources": [source],
    }
    return [
        Paper(
            **common,
            identifiers=PaperIdentifiers(doi="10.5555/conflict-a"),
        ),
        Paper(
            **common,
            identifiers=PaperIdentifiers(doi="10.5555/conflict-b"),
        ),
    ]


def _paper_identity(paper: Paper) -> str:
    identity = canonical_paper_id(paper)
    if identity is None:
        raise ResilienceNotEligible("fixture_paper_identity_unavailable")
    return identity


def _violation(
    scenario: str,
    source: str | None,
    invariant: str,
    path: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "source": source,
        "invariant": invariant,
        "first_difference_path": path,
        "expected_sha256": stable_hash(expected),
        "observed_sha256": stable_hash(observed),
    }


def _legacy_eligibility(
    protocol: Mapping[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    spec = protocol.get("frozen_baseline_eligibility") or {}
    path = _repo_path(repository_root, str(spec.get("legacy_audit_path") or ""))
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        return [
            {
                "profile_id": "frozen_baselines",
                "status": "not_eligible",
                "reason": "legacy_audit_missing_or_hash_drift",
            }
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in sorted(payload.get("profiles", []), key=lambda row: row["profile_id"]):
        rows.append(
            {
                "profile_id": item["profile_id"],
                "status": "not_eligible",
                "reason": "source_level_fault_replay_metadata_unavailable",
                "expected_query_count": item.get("expected_query_count"),
                "observed_record_count": item.get("observed_record_count"),
                "legacy_status": item.get("status"),
            }
        )
    return rows


def _repo_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise RetrievalResilienceError("path_must_be_repository_relative")
    root = repository_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RetrievalResilienceError("path_resolves_outside_repository") from exc
    return resolved
