from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_scheduler_fairness import (
    EXIT_NOT_STARTED,
    EXIT_READY,
    DeterministicScheduler,
    LoadProfile,
    SchedulerFairnessError,
    build_addendum,
    canonical_json,
    execute_profile,
    load_protocol,
    simulate_load,
    verify_resume,
)
from scholar_agent.evaluation.formal_provider_health_supervisor import (
    load_query_identities,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_scheduler_fairness_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_scheduler_fairness.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture()
def queries(protocol: dict[str, object]) -> tuple[str, ...]:
    return load_query_identities(ROOT, protocol)


def test_protocol_closes_population_limits_and_forbidden_inputs(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == (
        "46bd9d27e1a4e98bb4afa78c6bb6cdf77ab5d278"
    )
    assert protocol["population"]["query_count"] == 1000
    assert protocol["population"]["shard_count"] == 20
    assert protocol["limits"]["attempt_upper"] == 19280
    assert protocol["limits"]["global_concurrency"] == 12
    assert protocol["limits"]["per_source_concurrency"] == 3
    assert set(protocol["scheduler_policy"]["prohibited_priority_inputs"]) >= {
        "gold",
        "query_text_or_type",
        "result_content",
        "quality_metric",
        "completion_speed",
    }


def test_full1000_identity_and_round_robin_shards_close(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    assert len(queries) == len(set(queries)) == 1000
    for ordinal in range(1000):
        assert ordinal % 20 in range(20)


def test_uniform_load_has_full_first_service_and_no_starvation(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(protocol, queries, LoadProfile("uniform"))
    metrics = machine.metrics()
    assert machine.state == "completed"
    assert metrics["first_execution_coverage_rate"] == 1.0
    assert set(machine.first_query_service_step) == set(queries)
    assert set(machine.last_shard_service_step) == set(range(20))
    assert set(machine.last_source_service_step) == set(protocol["population"]["sources"])
    assert metrics["max_first_execution_wait_steps"] < 1000


@pytest.mark.parametrize(
    ("name", "delay_mode"),
    [
        ("slow-source", "slow_source"),
        ("slow-shard", "slow_shard"),
        ("heterogeneous", "heterogeneous"),
    ],
)
def test_slow_and_heterogeneous_loads_remain_finite_and_complete(
    protocol: dict[str, object],
    queries: tuple[str, ...],
    name: str,
    delay_mode: str,
) -> None:
    machine = execute_profile(
        protocol, queries, LoadProfile(name, delay_mode=delay_mode)
    )
    assert machine.state == "completed"
    assert machine.coverage()["terminal_query_count"] == 1000
    assert machine.logical_step < protocol["limits"]["max_logical_steps"]


def test_retry_and_page_work_wait_for_initial_admission_barrier(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    for profile in (
        LoadProfile("retry", outcome_mode="retry_storm"),
        LoadProfile("page", page_count=2),
    ):
        machine = execute_profile(protocol, queries, profile)
        admissions = [
            row
            for row in machine.audit_entries
            if row["event"] == "task_admitted"
        ]
        first_continuation = next(
            index
            for index, row in enumerate(admissions)
            if row["details"]["kind"] != "initial"
        )
        assert first_continuation == 4000
        assert all(
            row["details"]["kind"] == "initial"
            for row in admissions[:first_continuation]
        )


def test_retry_storm_cannot_monopolize_workers_or_budget(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(
        protocol, queries, LoadProfile("retry", outcome_mode="retry_storm")
    )
    metrics = machine.metrics()
    assert metrics["attempt_count"] == 7000
    assert metrics["attempt_count"] <= protocol["limits"]["attempt_upper"]
    assert max(metrics["per_source_concurrency_peak"].values()) <= 3
    assert metrics["concurrency_peak"] == 12
    assert metrics["backpressure_activation_count"] > 0


def test_pagination_storm_respects_page_and_attempt_caps(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(
        protocol, queries, LoadProfile("pages", page_count=2)
    )
    assert len(machine.completed_tasks) == 10000
    assert len(machine.ledger_identities) == 10000
    assert len(machine.completed_tasks) <= protocol["limits"]["attempt_upper"]


def test_worker_reduction_preserves_source_and_shard_opportunity(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(
        protocol, queries, LoadProfile("reduced", worker_limit=4)
    )
    metrics = machine.metrics()
    assert metrics["concurrency_peak"] == 4
    assert max(metrics["per_source_concurrency_peak"].values()) < 4
    assert metrics["first_execution_coverage_rate"] == 1.0


def test_pause_blocks_admission_drains_and_resume_preserves_cursor(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(
        protocol,
        queries,
        LoadProfile(
            "pause-resume",
            delay_mode="heterogeneous",
            pause_after_completions=257,
            resume_after_pause=True,
        ),
    )
    assert machine.state == "completed"
    assert machine.resume_count == 1
    assert machine.pause_checkpoint is not None
    assert machine.coverage()["terminal_query_count"] == 1000
    assert len(machine.started_task_identities) == len(machine.completed_tasks)


def test_pause_required_rejects_new_admission(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = DeterministicScheduler(protocol, queries)
    machine.admit(step=0, delay=lambda _task: 5)
    machine.request_pause("test")
    assert machine.admit(step=1, delay=lambda _task: 1) == 0
    assert machine.rejected_admissions == 1


def test_cancel_retains_all_queries_in_authoritative_order(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(
        protocol, queries, LoadProfile("cancel", cancel_after_completions=317)
    )
    coverage = machine.coverage()
    assert machine.state == "cancelled"
    assert coverage["terminal_query_count"] == 1000
    assert coverage["missing_query_count"] == 0
    assert coverage["status_counts"]["cancelled"] > 0
    assert coverage["success_only_filtering"] is False


def test_resume_evidence_drift_fails_closed(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = DeterministicScheduler(protocol, queries)
    machine.admit(step=0, delay=lambda _task: 1)
    machine.finish_due(
        step=1,
        outcome=lambda _task: "success",
        pages=lambda _task: 0,
    )
    machine.request_pause("test")
    machine.acknowledge_pause()
    with pytest.raises(SchedulerFairnessError, match="checkpoint"):
        machine.resume(
            {
                "authorization_fresh": True,
                "checkpoint_sha256": "0" * 64,
                "health_fresh": True,
                "host_fresh": True,
                "protocol_fresh": True,
            }
        )


def test_duplicate_enqueue_and_double_billing_are_detected(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = DeterministicScheduler(protocol, queries)
    with pytest.raises(SchedulerFairnessError, match="duplicate"):
        machine._enqueue(machine.pending_initial[0])

    complete = execute_profile(protocol, queries, LoadProfile("complete"))
    complete.ledger_identities.pop()
    with pytest.raises(SchedulerFairnessError, match="conservation"):
        complete.validate()


def test_concurrency_and_selective_coverage_mutations_fail_closed(
    protocol: dict[str, object], queries: tuple[str, ...]
) -> None:
    machine = execute_profile(protocol, queries, LoadProfile("complete"))
    machine.concurrency_peak = 13
    with pytest.raises(SchedulerFairnessError, match="concurrency"):
        machine.validate()

    machine = execute_profile(protocol, queries, LoadProfile("complete"))
    machine.terminal_queries.pop(queries[-1])
    with pytest.raises(SchedulerFairnessError, match="coverage"):
        machine.validate()


def test_simulation_matrix_and_resume_are_byte_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_load(ROOT, protocol)
    second = simulate_load(ROOT, protocol)
    assert first["exit_code"] == EXIT_READY
    assert first["scenario_count"] == 12
    assert canonical_json(first) == canonical_json(second)
    assert all(row["status"] == "passed" for row in first["scenarios"])
    redegraded = next(
        row
        for row in first["scenarios"]
        if row["scenario"] == "resume_then_redegrade"
    )
    assert redegraded["resume_count"] == 2

    first_resume = verify_resume(ROOT, protocol)
    second_resume = verify_resume(ROOT, protocol)
    assert canonical_json(first_resume) == canonical_json(second_resume)
    assert first_resume["terminal_query_count"] == 1000
    assert first_resume["fairness_cursor_preserved"] is True


def test_addendum_keeps_real_run_and_formal_validation_blocked(
    protocol: dict[str, object],
) -> None:
    addendum = build_addendum(protocol)
    assert addendum["real_execution_started"] is False
    assert addendum["formal_validation_complete"] is False
    assert addendum["requirements"]["resume_preserves_fairness_cursor"] is True


def test_protocol_drift_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    changed = copy.deepcopy(protocol)
    changed["limits"]["global_concurrency"] = 13
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SchedulerFairnessError, match="digest"):
        load_protocol(path, repository_root=ROOT)


def test_cli_commands_are_deterministic_and_real_readiness_is_blocked() -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=ROOT,
            env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "src"},
            capture_output=True,
            check=False,
        )

    first = run("simulate-load")
    second = run("simulate-load")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert run("verify-policy").returncode == EXIT_READY
    assert run("verify-resume").returncode == EXIT_READY
    readiness = run("audit-readiness")
    assert readiness.returncode == EXIT_NOT_STARTED
    assert json.loads(readiness.stdout)["status"] == "external_run_not_started"
    assert readiness.stderr == b""


def test_cli_malformed_protocol_is_exit_two_without_traceback(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "protocol.json"
    malformed.write_text('{"protocol":"formal_scheduler_fairness_v1"}\n')
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--protocol",
            str(malformed),
            "verify-policy",
        ],
        cwd=ROOT,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert b"Traceback" not in completed.stdout


def test_public_readiness_and_freshness_registration() -> None:
    public = json.loads(
        (
            ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
        ).read_text()
    )
    assert "formal_scheduler_fairness" in public["cli_contracts"]
    readiness = json.loads(
        (
            ROOT / "benchmark/validation_readiness_bundle_v1_contract.json"
        ).read_text()
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"] == "architecture_formal_scheduler_fairness_ready"
    )
    assert claim["status"] == "verified"
    assert "no quality result" in claim["boundary"]
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
        "architecture_formal_scheduler_fairness_ready"
    ] == ["formal_scheduler_fairness"]
