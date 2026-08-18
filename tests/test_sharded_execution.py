from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import run_benchmark
from scholar_agent.core.evaluation_schemas import EvalQuery
from scholar_agent.evaluation.crash_consistency import BenchmarkRunCommitStore
from scholar_agent.evaluation.run_provenance import RunManifestV1
from scholar_agent.evaluation.sharded_execution import (
    EXIT_NOT_READY,
    EXIT_PASSED,
    EXIT_VIOLATION,
    ShardPlanV1,
    build_local_fixture,
    deterministic_assignments,
    deterministic_fixture_report,
    load_attempt_set,
    load_shard_plan,
    select_queries_for_shard,
    validate_aggregate,
    validate_and_merge,
)


pytestmark = pytest.mark.sharded_execution_integrity_regression


def _invariants(report: dict[str, object]) -> set[str]:
    return {
        str(item["invariant"])
        for item in report.get("violations", [])  # type: ignore[union-attr]
    }


@pytest.mark.parametrize(
    ("query_count", "shard_count", "expected_sizes"),
    [
        (5, 1, [5]),
        (5, 2, [3, 2]),
        (5, 3, [2, 2, 1]),
        (7, 3, [3, 2, 2]),
    ],
)
def test_assignment_is_round_robin_complete_and_deterministic(
    query_count: int, shard_count: int, expected_sizes: list[int]
) -> None:
    identities = [f"query:{index:064x}" for index in range(query_count)]
    first = deterministic_assignments(identities, shard_count)
    second = deterministic_assignments(identities, shard_count)
    assert first == second
    assert [len(value) for value in first] == expected_sizes
    assert {item for shard in first for item in shard} == set(identities)


def test_plan_rejects_missing_duplicate_wrong_assignment_and_reordering(
    tmp_path: Path,
) -> None:
    plan_path, _, _, _ = build_local_fixture(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for mutation in ("missing", "duplicate", "wrong", "reorder"):
        changed = json.loads(json.dumps(payload))
        if mutation == "missing":
            changed["shards"][0]["query_identities"].pop()
        elif mutation == "duplicate":
            changed["shards"][0]["query_identities"].append(
                changed["shards"][0]["query_identities"][0]
            )
        elif mutation == "wrong":
            changed["shards"][0]["query_identities"][0] = changed["shards"][1][
                "query_identities"
            ][0]
        else:
            changed["shards"][0]["query_identities"].reverse()
        for shard in changed["shards"]:
            from scholar_agent.evaluation.snapshot_resume import stable_hash

            shard["query_identities_sha256"] = stable_hash(shard["query_identities"])
        with pytest.raises(ValidationError):
            ShardPlanV1.model_validate(changed)


def test_monolithic_and_sharded_results_are_byte_deterministic() -> None:
    first = deterministic_fixture_report()
    second = deterministic_fixture_report()
    assert first == second
    assert first["exit_code"] == EXIT_PASSED
    assert first["query_count"] == 7
    assert first["terminal_counts"] == {
        "cancelled": 2,
        "excluded": 1,
        "failed": 2,
        "succeeded": 2,
    }
    assert first["observation"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
        "outcome_based_attempt_selection": False,
    }


def test_plan_aggregate_and_report_bytes_repeat_across_directories(
    tmp_path: Path,
) -> None:
    outputs: list[tuple[bytes, bytes, bytes]] = []
    for name in ("first", "second"):
        root = tmp_path / name
        plan, attempts, monolithic, _ = build_local_fixture(root)
        aggregate = root / "aggregate.json"
        report = validate_and_merge(
            plan,
            attempts,
            repository_root=root,
            output_path=aggregate,
            monolithic_manifest_path=monolithic,
        )
        outputs.append(
            (
                plan.read_bytes(),
                aggregate.read_bytes(),
                json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            )
        )
    assert outputs[0] == outputs[1]


def test_different_shard_completion_order_does_not_change_merge() -> None:
    first = deterministic_fixture_report(completion_order=(0, 1, 2))
    second = deterministic_fixture_report(completion_order=(2, 0, 1))
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    assert first["terminal_counts"] == second["terminal_counts"]
    assert first["exit_code"] == second["exit_code"] == EXIT_PASSED
    reversed_queries = deterministic_fixture_report(reverse_query_completion=True)
    assert first["aggregate_sha256"] == reversed_queries["aggregate_sha256"]


@pytest.mark.parametrize(
    ("fault", "invariant"),
    [
        ("duplicate_query", "aggregate_duplicate_query"),
        ("missing_query", "aggregate_query_missing"),
        ("common_success_filter", "outcome_based_query_filtering"),
        ("config_drift", "shard_generation_configuration_drift"),
    ],
)
def test_controlled_partition_and_filter_faults_are_stable(
    fault: str, invariant: str
) -> None:
    report = deterministic_fixture_report(controlled_fault=fault)  # type: ignore[arg-type]
    assert report["exit_code"] == EXIT_VIOLATION
    assert invariant in _invariants(report)
    assert report == deterministic_fixture_report(controlled_fault=fault)  # type: ignore[arg-type]


def test_incomplete_shard_is_not_ready_and_preserves_other_terminals() -> None:
    report = deterministic_fixture_report(incomplete_shard=1)
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["status"] == "not_ready"
    assert report["selected_shard_count"] == 2
    assert report["pending"] == [
        {"reason": "shard_attempt_incomplete", "shard": 1, "attempt": "attempt-0"}
    ]


def test_explicit_retry_chain_selects_unique_tip_without_outcome_selection() -> None:
    report = deterministic_fixture_report(retry_shard=1)
    assert report["exit_code"] == EXIT_PASSED
    assert report["observation"]["outcome_based_attempt_selection"] is False


def test_stale_or_double_final_attempt_is_rejected(tmp_path: Path) -> None:
    plan, attempts_path, monolithic, _ = build_local_fixture(tmp_path, retry_shard=1)
    payload = json.loads(attempts_path.read_text(encoding="utf-8"))
    retry = next(
        item
        for item in payload["attempts"]
        if item["shard_index"] == 1 and item["attempt_id"] == "attempt-1"
    )
    retry["supersedes_attempt_id"] = None
    attempts_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report = validate_and_merge(
        plan,
        attempts_path,
        repository_root=tmp_path,
        monolithic_manifest_path=monolithic,
    )
    assert report["exit_code"] == EXIT_VIOLATION
    assert "unique_final_attempt_missing" in _invariants(report)

    payload = json.loads(attempts_path.read_text(encoding="utf-8"))
    retry["supersedes_attempt_id"] = "missing-attempt"
    payload["attempts"] = [
        retry if item["shard_index"] == 1 and item["attempt_id"] == "attempt-1" else item
        for item in payload["attempts"]
    ]
    attempts_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report = validate_and_merge(plan, attempts_path, repository_root=tmp_path)
    assert "superseded_attempt_missing" in _invariants(report)


def test_plan_hash_and_shard_binding_drift_are_rejected(tmp_path: Path) -> None:
    plan, attempts_path, _, _ = build_local_fixture(tmp_path)
    payload = json.loads(attempts_path.read_text(encoding="utf-8"))
    payload["plan_sha256"] = "0" * 64
    attempts_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_and_merge(plan, attempts_path, repository_root=tmp_path)
    assert "attempt_set_plan_hash_mismatch" in _invariants(report)


def test_generation_lineage_damage_and_duplicate_commit_fail(tmp_path: Path) -> None:
    plan, attempts_path, _, _ = build_local_fixture(tmp_path)
    attempts = load_attempt_set(attempts_path)
    reference = attempts.attempts[0]
    manifest = RunManifestV1.model_validate_json(
        (tmp_path / reference.manifest_path).read_text(encoding="utf-8")
    )
    store = BenchmarkRunCommitStore(tmp_path / manifest.output_directory)
    (store.load_latest().generation_path / "COMMITTED").unlink()
    report = validate_and_merge(plan, attempts_path, repository_root=tmp_path)
    assert "manifest_generation_state_mismatch" in _invariants(report)


def test_aggregate_is_new_read_only_file_and_tampering_is_detected(
    tmp_path: Path,
) -> None:
    plan, attempts, monolithic, _ = build_local_fixture(tmp_path)
    output = tmp_path / "aggregate.json"
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    report = validate_and_merge(
        plan,
        attempts,
        repository_root=tmp_path,
        output_path=output,
        monolithic_manifest_path=monolithic,
    )
    assert report["exit_code"] == EXIT_PASSED
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and path != output
    }
    assert before == after
    first = output.read_bytes()
    assert json.loads(first)["records"][0]["status"] == "succeeded"

    payload = json.loads(first)
    payload["records"][0]["status"] = "failed"
    from scholar_agent.evaluation.snapshot_resume import stable_hash

    payload["aggregate_summary_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "aggregate_summary_sha256"}
    )
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tampered = validate_aggregate(output, plan, attempts, repository_root=tmp_path)
    assert tampered["exit_code"] == EXIT_VIOLATION
    assert "aggregate_reference_or_content_tampered" in _invariants(tampered)


def test_runner_shard_selection_is_plan_bound_and_default_is_unchanged(
    tmp_path: Path,
) -> None:
    plan_path, _, _, _ = build_local_fixture(tmp_path, shard_count=3)
    plan = load_shard_plan(plan_path)
    queries = [
        EvalQuery(query_id=identity, query=f"query {index}", gold_papers=[])
        for index, identity in enumerate(plan.queries.identities)
    ]
    default = run_benchmark.BenchmarkRunOptions(
        dataset="autoscholar_query",
        run_id="default-fixture",
    )
    assert run_benchmark._select_shard_population(default, queries) == queries

    options = run_benchmark.BenchmarkRunOptions(
        dataset="autoscholar_query",
        run_id="shard-fixture",
        shard_plan_path=plan_path,
        shard_index=1,
        shard_attempt_id="attempt-0",
    )
    selected = run_benchmark._select_shard_population(options, queries)
    assert [item.query_id for item in selected] == plan.shards[1].query_identities

    reordered = list(reversed(queries))
    with pytest.raises(ValueError, match="population_or_order"):
        run_benchmark._select_shard_population(options, reordered)


def test_run_manifest_and_generation_zero_share_exact_shard_binding(
    tmp_path: Path,
) -> None:
    _, attempts_path, _, _ = build_local_fixture(tmp_path)
    reference = load_attempt_set(attempts_path).attempts[0]
    manifest = RunManifestV1.model_validate_json(
        (tmp_path / reference.manifest_path).read_text(encoding="utf-8")
    )
    state = BenchmarkRunCommitStore(tmp_path / manifest.output_directory).load_latest()
    observed = {
        key: value
        for key, value in manifest.shard.model_dump(mode="json").items()  # type: ignore[union-attr]
        if key != "plan"
    }
    assert state.config["shard"] == observed
