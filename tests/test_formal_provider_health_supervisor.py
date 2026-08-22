from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_provider_health_supervisor import (
    EXIT_NOT_OBSERVED,
    EXIT_READY,
    ProviderHealthError,
    ProviderHealthSupervisor,
    _finish_sequence,
    _valid_resume_evidence,
    bind_launch_authorization,
    build_addendum,
    canonical_json,
    load_protocol,
    load_query_identities,
    simulate_run,
    verify_resume_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/formal_provider_health_supervisor_v1_protocol.json"
)
CLI = ROOT / "scripts/check_formal_provider_health.py"


def _cli_env() -> dict[str, str]:
    environment = {"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "src"}
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            if os.environ.get(name):
                environment[name] = os.environ[name]
    return environment


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture()
def queries(protocol: dict[str, object]) -> tuple[str, ...]:
    return load_query_identities(ROOT, protocol)


def test_protocol_binds_full1000_and_preregistered_health_thresholds(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == "d732c7727a732d171ad9b762b28f89b9e9053c4a"
    assert protocol["population"]["query_count"] == 1000
    assert protocol["population"]["http_attempt_upper"] == 19280
    assert protocol["thresholds"]["source"] == {
        "consecutive_failures_degraded": 3,
        "consecutive_failures_pause": 6,
        "rolling_failure_min_observations": 12,
        "rolling_failure_pause_ratio": 0.75,
        "rolling_window": 20,
        "successful_zero_progress_pause": 10,
    }
    assert protocol["unknown_provider_limits"]["provider_rate_limit"] == "not_available"


def test_full1000_identity_and_order_close_exactly(
    queries: tuple[str, ...], protocol: dict[str, object]
) -> None:
    assert len(queries) == len(set(queries)) == 1000
    from scholar_agent.evaluation.snapshot_resume import stable_hash

    assert stable_hash(queries) == protocol["population"]["query_order_sha256"]


def test_transient_jitter_does_not_pause(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(
        supervisor,
        "transient-test",
        "arxiv",
        [("429", 0), ("timeout", 0)] + [("success", 1)] * 10,
    )
    assert supervisor.state == "healthy"
    assert supervisor.reason_codes == []


def test_consecutive_failure_boundaries_are_exact(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(supervisor, "boundary-three", "arxiv", [("503", 0)] * 3)
    assert supervisor.state == "degraded"
    _finish_sequence(
        supervisor,
        "boundary-six",
        "arxiv",
        [("503", 0)] * 3,
        query_offset=10,
    )
    assert supervisor.state == "pause_required"


def test_rolling_ratio_boundary_and_zero_progress_boundary(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    rolling = ProviderHealthSupervisor(protocol, queries)
    outcomes = [
        ("429", 0) if index % 4 != 3 else ("success", 1)
        for index in range(12)
    ]
    _finish_sequence(rolling, "rolling", "openalex", outcomes)
    assert rolling.state == "pause_required"

    zero = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(zero, "zero", "pubmed", [("success", 0)] * 9)
    assert zero.state != "pause_required"
    _finish_sequence(
        zero, "zero-final", "pubmed", [("success", 0)], query_offset=20
    )
    assert zero.state == "pause_required"


def test_pause_blocks_new_work_but_allows_inflight_drain(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    supervisor.start_operation("operation:a", queries[0], "arxiv", "query")
    supervisor.start_operation("operation:b", queries[1], "openalex", "page")
    supervisor.record_control_failure("storage_capacity_failure")
    with pytest.raises(ProviderHealthError, match="forbidden"):
        supervisor.start_operation("operation:c", queries[2], "pubmed", "retry")
    with pytest.raises(ProviderHealthError, match="drained"):
        supervisor.acknowledge_pause()
    supervisor.finish_operation(
        "operation:a",
        outcome="cancelled",
        progress_records=0,
        ledger_entry_identity="ledger:a",
    )
    supervisor.finish_operation(
        "operation:b",
        outcome="cancelled",
        progress_records=0,
        ledger_entry_identity="ledger:b",
    )
    checkpoint = supervisor.acknowledge_pause()
    assert checkpoint["in_flight_count"] == 0
    assert supervisor.state == "paused"


def test_pause_retains_failed_and_cancelled_query_coverage(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    supervisor.commit_query(queries[0], "completed")
    supervisor.commit_query(queries[1], "failed")
    supervisor.commit_query(queries[2], "cancelled")
    supervisor.commit_query(queries[3], "source_failure")
    supervisor.record_control_failure("provenance_write_failure")
    checkpoint = supervisor.acknowledge_pause()
    assert checkpoint["committed_query_count"] == 4
    assert checkpoint["coverage_status_counts"] == {
        "cancelled": 1,
        "completed": 1,
        "failed": 1,
        "source_failure": 1,
    }
    assert supervisor.coverage_summary()["success_only_filtering"] is False


def test_resume_requires_every_fresh_prerequisite_and_same_checkpoint(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    supervisor.record_control_failure("storage_capacity_failure")
    supervisor.acknowledge_pause()
    evidence = _valid_resume_evidence(supervisor)
    for key in (
        "authorization_fresh",
        "capacity_fresh",
        "health_clearance_observed",
        "host_attestation_fresh",
        "protocol_fresh",
    ):
        changed = copy.deepcopy(evidence)
        changed[key] = False
        with pytest.raises(ProviderHealthError, match="prerequisite"):
            supervisor.make_resume_eligible(changed)
    changed = copy.deepcopy(evidence)
    changed["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ProviderHealthError, match="checkpoint"):
        supervisor.make_resume_eligible(changed)


def test_resume_preserves_failure_counters_and_rejects_repeat_request(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    _finish_sequence(supervisor, "resume-test", "arxiv", [("429", 0)] * 6)
    supervisor.commit_query(queries[20], "source_failure")
    supervisor.acknowledge_pause()
    previous_failures = supervisor.sources["arxiv"].total_failures
    supervisor.make_resume_eligible(_valid_resume_evidence(supervisor))
    supervisor.resume()
    assert supervisor.sources["arxiv"].total_failures == previous_failures
    with pytest.raises(ProviderHealthError, match="committed_query_repeat"):
        supervisor.start_operation(
            "operation:repeat", queries[20], "arxiv", "query"
        )


def test_duplicate_completion_and_double_billing_fail_closed(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    supervisor.start_operation("operation:a", queries[0], "arxiv", "query")
    supervisor.finish_operation(
        "operation:a",
        outcome="success",
        progress_records=1,
        ledger_entry_identity="ledger:a",
    )
    with pytest.raises(ProviderHealthError, match="not_in_flight"):
        supervisor.finish_operation(
            "operation:a",
            outcome="success",
            progress_records=1,
            ledger_entry_identity="ledger:b",
        )
    supervisor.start_operation("operation:b", queries[1], "openalex", "query")
    with pytest.raises(ProviderHealthError, match="not_unique"):
        supervisor.finish_operation(
            "operation:b",
            outcome="success",
            progress_records=1,
            ledger_entry_identity="ledger:a",
        )


def test_operation_audit_chain_is_complete_and_tamper_evident(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    supervisor.start_operation("operation:a", queries[0], "arxiv", "query")
    supervisor.finish_operation(
        "operation:a",
        outcome="success",
        progress_records=1,
        ledger_entry_identity="ledger:a",
    )
    supervisor.commit_query(queries[0], "completed")
    assert supervisor.validate_audit_chain() is True
    assert [row["event"] for row in supervisor.audit_entries] == [
        "operation_started",
        "operation_finished",
        "generation_committed",
    ]
    supervisor.audit_entries[1]["details"]["progress_records"] = 2
    assert supervisor.validate_audit_chain() is False


def test_illegal_selective_coverage_and_aggregate_are_not_accepted(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    supervisor = ProviderHealthSupervisor(protocol, queries)
    for query in queries[:10]:
        supervisor.commit_query(query, "completed")
    assert supervisor.aggregate_eligible() is False
    assert supervisor.coverage_summary()["missing_query_count"] == 990


def test_simulated_1000_query_matrix_is_complete_and_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_run(ROOT, protocol)
    second = simulate_run(ROOT, protocol)
    assert first["exit_code"] == EXIT_READY
    assert first["scenario_count"] == 14
    assert canonical_json(first) == canonical_json(second)
    assert all(row["status"] == "passed" for row in first["scenarios"])
    assert all(row["audit_chain_valid"] for row in first["scenarios"])
    resumed = next(
        row
        for row in first["scenarios"]
        if row["scenario"] == "pause_resume_full_coverage"
    )
    assert resumed["coverage"]["committed_query_count"] == 1000
    assert resumed["coverage"]["aggregate_eligible"] is True
    assert resumed["historical_failure_count_preserved"] is True


def test_resume_fixture_is_deterministic(
    protocol: dict[str, object],
) -> None:
    first = verify_resume_fixture(ROOT, protocol)
    second = verify_resume_fixture(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["historical_failures_preserved"] == 6
    assert first["repeated_request_count"] == 0


def test_addendum_keeps_real_activation_and_formal_validation_blocked(
    protocol: dict[str, object],
) -> None:
    addendum = build_addendum(protocol)
    assert addendum["real_provider_health_observed"] is False
    assert addendum["formal_validation_complete"] is False
    assert addendum["requirements"]["failure_counters_not_reset"] is True
    assert addendum["requirements"]["source_switch_on_resume"] is False


def test_launch_binding_requires_qualified_host_and_zero_health_counters(
    protocol: dict[str, object],
) -> None:
    authorization = {"authorization_sha256": "a" * 64}
    host = {"attestation_sha256": "b" * 64, "status": "host_qualified"}
    bound = bind_launch_authorization(authorization, host, protocol)
    assert bound["initial_health_state"] == "healthy"
    assert bound["initial_attempt_count"] == bound["initial_failure_count"] == 0
    assert (
        bound["provider_health_protocol_sha256"] == protocol["protocol_sha256"]
    )
    rejected = dict(host)
    rejected["status"] = "not_ready_unverified_or_insufficient_host"
    with pytest.raises(ProviderHealthError, match="qualified"):
        bind_launch_authorization(authorization, rejected, protocol)


def test_protocol_policy_edit_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    changed = copy.deepcopy(protocol)
    changed["thresholds"]["source"]["consecutive_failures_pause"] = 7
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ProviderHealthError, match="digest"):
        load_protocol(path, repository_root=ROOT)


def test_cli_commands_and_real_readiness_are_stable() -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=ROOT,
            env=_cli_env(),
            capture_output=True,
            check=False,
        )

    first = run("simulate-run")
    second = run("simulate-run")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""

    policy = run("verify-policy")
    assert policy.returncode == EXIT_READY
    resume = run("verify-resume")
    assert resume.returncode == EXIT_READY

    readiness = run("audit-readiness")
    assert readiness.returncode == EXIT_NOT_OBSERVED
    assert json.loads(readiness.stdout)["status"] == (
        "external_provider_health_not_observed"
    )
    assert readiness.stderr == b""


def test_cli_malformed_resume_evidence_has_no_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "resume.json"
    malformed.write_text('{"authorization_fresh":true}\n', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "verify-resume",
            "--evidence",
            str(malformed),
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert b"Traceback" not in completed.stdout


def test_public_readiness_and_freshness_registration() -> None:
    public = json.loads(
        (ROOT / "benchmark/public_contract_compatibility_v1_protocol.json").read_text()
    )
    assert "formal_provider_health_supervisor" in public["cli_contracts"]
    readiness = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text()
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"]
        == "architecture_formal_provider_health_supervisor_ready"
    )
    assert claim["status"] == "verified"
    assert "does not clear" in claim["boundary"]
    assert {row["blocker_id"] for row in readiness["blockers"]} == {
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    }
    addenda = json.loads(
        (
            ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json"
        ).read_text()
    )
    assert addenda["claim_component_bindings"][
        "architecture_formal_provider_health_supervisor_ready"
    ] == ["formal_provider_health_supervisor"]
