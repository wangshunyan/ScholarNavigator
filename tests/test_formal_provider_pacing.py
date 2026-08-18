from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_provider_pacing import (
    CAPACITY_FIELDS,
    EXIT_NOT_READY,
    EXIT_READY,
    CapacityProfile,
    DeterministicPacer,
    ProviderPacingError,
    ProviderPacingNotReady,
    _declarations,
    audit_readiness,
    build_launch_addendum,
    canonical_json,
    execute_profile,
    load_operations,
    load_protocol,
    simulate_capacity,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_provider_pacing_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_provider_pacing.py"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture(scope="module")
def operations(protocol: dict[str, object]):
    return load_operations(ROOT, protocol)


@pytest.fixture(scope="module")
def balanced_machine(protocol: dict[str, object], operations):
    return execute_profile(
        protocol,
        operations,
        CapacityProfile("balanced", _declarations()),
    )


@pytest.fixture(scope="module")
def capacity_matrix(protocol: dict[str, object]):
    return simulate_capacity(ROOT, protocol)


def _small(operations):
    return tuple(item for item in operations if item.query_order < 3)


def test_protocol_freezes_population_policy_and_real_unknown_capacity(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == (
        "38eca0b1a8a1744b73c88ac480704b92ae0b530a"
    )
    assert protocol["population"] == {
        "http_attempt_upper": 19280,
        "logical_source_request_count": 9640,
        "query_count": 1000,
        "shard_count": 20,
        "sources": ["openalex", "arxiv", "semantic_scholar", "pubmed"],
        "subquery_count": 2410,
    }
    assert protocol["pacing_policy"]["global_concurrency"] == 12
    assert set(protocol["pacing_policy"]["prohibited_priority_inputs"]) >= {
        "gold",
        "query_text_or_type",
        "result_content",
        "quality_metric",
    }
    for declaration in protocol["capacity_declarations"].values():
        assert all(declaration[field] == "not_available" for field in CAPACITY_FIELDS)
        assert declaration["declaration_version"] == "not_available"
        assert declaration["valid_from"] == declaration["valid_until"] == "not_available"


def test_request_manifest_is_consumed_exactly_without_regeneration(operations) -> None:
    assert len(operations) == 9640
    assert len({item.intent_identity for item in operations}) == 9640
    assert {item.query_order for item in operations} == set(range(1000))
    assert {item.shard_index for item in operations} == set(range(20))
    assert sum(item.http_attempt_upper for item in operations) == 19280
    counts = defaultdict(int)
    for operation in operations:
        counts[operation.source] += 1
    assert dict(counts) == {
        "openalex": 2410,
        "arxiv": 2410,
        "semantic_scholar": 2410,
        "pubmed": 2410,
    }


def test_balanced_profile_has_zero_limit_violations_and_full_coverage(
    balanced_machine: DeterministicPacer,
) -> None:
    summary = balanced_machine.summary()
    assert balanced_machine.state == "completed"
    assert summary["intent_coverage_count"] == 9640
    assert summary["admitted_attempt_count"] == 12050
    assert summary["ledger_entry_count"] == summary["admitted_attempt_count"]
    assert summary["request_set_unchanged"] is True
    assert summary["request_parameter_mutation_count"] == 0
    assert len(summary["request_contract_sha256"]) == 64
    assert summary["window_violation_count"] == 0
    assert summary["duplicate_request_count"] == 0
    assert summary["global_concurrency_peak"] <= 12


def test_every_sliding_window_and_source_concurrency_respects_declaration(
    balanced_machine: DeterministicPacer,
) -> None:
    per_source_steps: dict[str, list[int]] = defaultdict(list)
    for step, source, _kind, _identity in balanced_machine.admission_records:
        per_source_steps[source].append(step)
    for source, steps in per_source_steps.items():
        declaration = balanced_machine.runtimes[source].declaration
        for index, start in enumerate(steps):
            count = sum(1 for step in steps[index:] if step < start + 60)
            assert count <= declaration.requests_per_minute
        assert (
            balanced_machine.source_concurrency_peak[source]
            <= declaration.max_concurrency
        )


def test_initial_requests_have_a_strict_barrier_before_retry(operations, protocol) -> None:
    failing = next(
        item for item in operations if int(item.intent_identity[-8:], 16) % 53 == 0
    )
    selected_ids = {
        *(item.intent_identity for item in operations[:20]),
        failing.intent_identity,
    }
    selected = tuple(
        item for item in operations if item.intent_identity in selected_ids
    )
    machine = execute_profile(
        protocol,
        selected,
        CapacityProfile(
            "retry_after",
            _declarations(),
            outcome_mode="retry_after",
            page_mode=False,
        ),
    )
    initial_steps = [
        step for step, _source, kind, _identity in machine.admission_records if kind == "initial"
    ]
    retry_steps = [
        step for step, _source, kind, _identity in machine.admission_records if kind == "retry"
    ]
    assert retry_steps
    assert min(retry_steps) >= max(initial_steps)
    retry_identity = f"operation:{stable_hash({'intent': failing.intent_identity, 'kind': 'retry', 'attempt': 1})}"
    assert machine.admitted[retry_identity] - machine.admitted[
        failing.operation_identity
    ] >= 10
    assert machine.summary()["duplicate_request_count"] == 0


def test_capacity_matrix_covers_required_profiles_and_budget(
    capacity_matrix: dict[str, object],
) -> None:
    assert capacity_matrix["status"] == "pacing_controls_ready"
    assert capacity_matrix["exit_code"] == EXIT_READY
    assert capacity_matrix["scenario_count"] == 12
    assert capacity_matrix["passed_scenario_count"] == 10
    rows = {row["profile"]: row for row in capacity_matrix["scenarios"]}
    assert rows["expired_declaration"]["status"] == "not_ready"
    assert rows["unknown_capacity"]["status"] == "not_ready"
    for name, row in rows.items():
        if row["status"] != "passed":
            continue
        assert row["admitted_attempt_count"] <= 19280, name
        assert row["intent_coverage_count"] == 9640, name
        assert row["window_violation_count"] == 0, name
        assert row["duplicate_request_count"] == 0, name
        assert row["request_parameter_mutation_count"] == 0, name
        assert row["request_set_unchanged"] is True, name
    assert rows["persistent_429"]["admitted_attempt_count"] == 12050
    assert rows["pause_resume"]["resume_count"] == 1
    assert rows["dynamic_reduction"]["declaration_versions"]["openalex"].endswith(
        "reduced-v1"
    )


def test_pause_resume_preserves_tokens_cursor_and_request_identities(
    protocol, operations
) -> None:
    subset = _small(operations)
    uninterrupted = execute_profile(
        protocol,
        subset,
        CapacityProfile("uninterrupted", _declarations()),
    )
    resumed = execute_profile(
        protocol,
        subset,
        CapacityProfile(
            "pause_resume",
            _declarations(),
            pause_after_admissions=9,
        ),
    )
    left = uninterrupted.summary()
    right = resumed.summary()
    for key in (
        "admitted_attempt_count",
        "completed_attempt_count",
        "intent_coverage_count",
        "ledger_entry_count",
        "request_identity_sha256",
        "request_contract_sha256",
        "request_parameter_mutation_count",
        "source_admission_counts",
    ):
        assert left[key] == right[key]
    assert right["resume_count"] == 1
    assert right["pause_checkpoint_sha256"]


def test_cancel_stops_new_admission_and_drains_inflight(protocol, operations) -> None:
    machine = DeterministicPacer(
        protocol,
        _small(operations),
        CapacityProfile("cancel", _declarations(), page_mode=False),
    )
    machine.state = "running"
    assert machine._admit_one(0) is True
    admitted_before = len(machine.admitted)
    machine.request_cancel()
    assert machine._admit_one(0) is False
    machine._finish_due(10)
    machine.finish_cancel()
    assert machine.state == "cancelled"
    assert len(machine.admitted) == admitted_before
    assert len(machine.completed) == len(machine.ledger_identities) == admitted_before


def test_unknown_and_expired_real_declarations_fail_closed(protocol, operations) -> None:
    with pytest.raises(ProviderPacingNotReady, match="missing"):
        DeterministicPacer(
            protocol,
            _small(operations),
            CapacityProfile("unknown", None),
        )
    with pytest.raises(ProviderPacingNotReady, match="expired"):
        DeterministicPacer(
            protocol,
            _small(operations),
            CapacityProfile("expired", _declarations(expired=True)),
        )
    # Synthetic declarations are test-only and can never satisfy real readiness.
    assert audit_readiness(ROOT, protocol)["exit_code"] == EXIT_NOT_READY


def test_request_reordering_attempt_drift_and_direct_adapter_bypass_fail(
    protocol, operations
) -> None:
    subset = _small(operations)
    with pytest.raises(ProviderPacingError, match="order_drift"):
        DeterministicPacer(
            protocol,
            tuple(reversed(subset)),
            CapacityProfile("reordered", _declarations()),
        )
    changed = (replace(subset[0], http_attempt_upper=3), *subset[1:])
    with pytest.raises(ProviderPacingError, match="attempt_upper_drift"):
        DeterministicPacer(
            protocol,
            changed,
            CapacityProfile("attempt-drift", _declarations()),
        )
    machine = DeterministicPacer(
        protocol,
        subset,
        CapacityProfile("bypass", _declarations()),
    )
    with pytest.raises(ProviderPacingError, match="direct_adapter_bypass"):
        machine.direct_adapter_call(subset[0])


def test_protocol_value_tampering_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text())
    raw["pacing_policy"]["global_concurrency"] = 13
    payload = copy.deepcopy(raw)
    payload.pop("protocol_sha256")
    from scholar_agent.evaluation.snapshot_resume import stable_hash

    raw["protocol_sha256"] = stable_hash(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProviderPacingError, match="content_drift"):
        load_protocol(path, repository_root=ROOT)


def test_launch_addendum_binds_request_plan_and_blocks_activation(
    protocol: dict[str, object],
) -> None:
    addendum = build_launch_addendum(ROOT, protocol)
    assert addendum["logical_source_request_count"] == 9640
    assert addendum["http_attempt_upper"] == 19280
    assert addendum["real_capacity_status"] == "not_available"
    assert addendum["historical_request_set_mutated"] is False
    assert addendum["activation_requirement"] == (
        "fresh_complete_external_capacity_declarations_for_all_sources"
    )


def test_small_profile_reports_are_byte_deterministic(protocol, operations) -> None:
    profile = CapacityProfile("deterministic", _declarations(), page_mode=False)
    left = execute_profile(protocol, _small(operations), profile).summary()
    right = execute_profile(protocol, _small(operations), profile).summary()
    assert canonical_json(left) == canonical_json(right)


@pytest.mark.parametrize(
    ("command", "expected_code", "expected_status"),
    [
        ("verify-policy", 0, "pacing_controls_ready"),
        (
            "audit-readiness",
            3,
            "not_ready_missing_provider_capacity_declarations",
        ),
    ],
)
def test_cli_exit_codes_no_traceback_and_deterministic(
    command: str, expected_code: int, expected_status: str
) -> None:
    args = [sys.executable, str(CLI), command]
    first = subprocess.run(args, cwd=ROOT, capture_output=True, check=False)
    second = subprocess.run(args, cwd=ROOT, capture_output=True, check=False)
    assert first.returncode == second.returncode == expected_code
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    report = json.loads(first.stdout)
    assert report["status"] == expected_status
    assert report["execution"]["network_request_count"] == 0
    assert report["formal_validation_complete"] is False


def test_missing_protocol_is_stable_exit_two_without_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--protocol",
            str(tmp_path / "missing.json"),
            "verify-policy",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert json.loads(result.stdout)["status"] == "pacing_or_budget_violation"


def test_readiness_freshness_and_public_contract_integrations_preserve_blockers() -> None:
    readiness_contract = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text()
    )
    readiness = json.loads(
        (
            ROOT
            / "benchmark/validation_readiness_bundle_v1_release/readiness.json"
        ).read_text()
    )
    freshness = json.loads(
        (
            ROOT
            / "benchmark/validation_evidence_freshness_v1_addenda.json"
        ).read_text()
    )
    public = json.loads(
        (
            ROOT
            / "benchmark/public_contract_compatibility_v1_protocol.json"
        ).read_text()
    )
    claims = {row["claim_id"]: row for row in readiness_contract["claims"]}
    gates = {row["gate_id"]: row for row in readiness_contract["read_only_gates"]}
    assert claims["architecture_formal_provider_pacing_ready"]["status"] == "verified"
    assert gates["formal_provider_pacing"]["expected_exit_code"] == 3
    assert readiness["formal_validation_complete"] is False
    assert readiness["blocker_count"] == 3
    assert any(
        row["component_id"] == "formal_provider_pacing"
        for row in freshness["components"]
    )
    assert "formal_provider_pacing" in public["artifact_contracts"]
    assert public["cli_contracts"]["formal_provider_pacing"]["exit_codes"] == [
        0,
        2,
        3,
        4,
    ]
