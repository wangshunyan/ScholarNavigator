from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_crash_consistency as crash_cli  # noqa: E402
from scholar_agent.evaluation.crash_consistency import (  # noqa: E402
    EXIT_INVARIANT_VIOLATION,
    EXIT_NOT_ELIGIBLE,
    BenchmarkRunCommitStore,
    ConcurrentWriterError,
    FaultInjector,
    InjectedCrash,
    audit_frozen_baseline_eligibility,
    durable_atomic_write_bytes,
    load_protocol,
    run_crash_consistency,
    sanitize_error,
    sha256_file,
    stable_json_bytes,
    write_json,
)


PROTOCOL = ROOT / "benchmark" / "crash_consistency_v1_protocol.json"


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _config() -> dict[str, Any]:
    return {
        "dataset": "offline",
        "case_count": 2,
        "case_ids": ["q1", "q2"],
        "resume_signature": "a" * 64,
    }


def _record(case_id: str, status: str = "succeeded") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": f"query {case_id}",
        "status": status,
        "error_type": None if status == "succeeded" else "FixtureError",
        "error": None if status == "succeeded" else "offline failure",
    }


def _store(tmp_path: Path) -> tuple[BenchmarkRunCommitStore, int]:
    store = BenchmarkRunCommitStore(tmp_path / "run")
    store.initialize(
        run_id="test-run",
        expected_query_ids=["q1", "q2"],
        config=_config(),
        dataset_report={"count": 2},
    )
    state = store.commit_record(_record("q1"))
    return store, state.generation


@pytest.mark.crash_consistency_regression
def test_repository_crash_matrix_passes_offline(tmp_path: Path) -> None:
    report = run_crash_consistency(_protocol(), work_root=tmp_path)
    assert report["status"] == "passed"
    assert report["exit_code"] == 0
    assert report["fault_scenario_count"] == 12
    assert report["violation_count"] == 0
    assert {row["fault_point"] for row in report["scenarios"]} == set(
        _protocol()["fault_scenarios"]
    )
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "real_process_kill_count": 0,
        "real_disk_fill_count": 0,
        "sleep_race_count": 0,
        "controlled_fault": None,
    }


@pytest.mark.parametrize(
    "point",
    [
        "before_stage_create",
        "after_stage_create",
        "mid_write",
        "flush_failure",
        "fsync_failure",
        "checkpoint_committed_manifest_missing",
        "manifest_written_completion_missing",
        "before_replace",
        "after_replace",
        "disk_full",
        "permission_denied",
    ],
)
def test_each_precommit_fault_preserves_previous_generation(
    tmp_path: Path, point: str
) -> None:
    store, generation = _store(tmp_path)
    prior_marker_hash = sha256_file(
        store.generations / f"generation-{generation:08d}" / "COMMITTED"
    )
    with pytest.raises((InjectedCrash, OSError)):
        store.commit_record(
            _record("q2"), injector=FaultInjector(point=point)  # type: ignore[arg-type]
        )
    recovered = store.load_latest()
    assert recovered.generation == generation
    assert [row["case_id"] for row in recovered.records] == ["q1"]
    assert sha256_file(
        store.generations / f"generation-{generation:08d}" / "COMMITTED"
    ) == prior_marker_hash


def test_interruption_after_commit_marker_recovers_new_generation(
    tmp_path: Path,
) -> None:
    store, generation = _store(tmp_path)
    with pytest.raises(InjectedCrash, match="after_commit_marker"):
        store.commit_record(
            _record("q2"),
            injector=FaultInjector(point="after_commit_marker"),
        )
    recovered = store.load_latest()
    assert recovered.generation == generation + 1
    assert [row["case_id"] for row in recovered.records] == ["q1", "q2"]


def test_normal_multi_generation_resume_and_completion_are_consistent(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    state = store.commit_record(_record("q2", "failed"))
    reports = {
        "metrics.json": stable_json_bytes({"internal": True}),
        "summary.md": b"# report\n",
    }
    completed = store.commit_completion(reports)
    restored = store.load_latest()
    assert restored.generation == completed.generation
    assert restored.status == "completed"
    assert restored.reports == reports
    assert [row["case_id"] for row in restored.records] == ["q1", "q2"]
    assert restored.event_count == 4
    assert (restored.generation_path / "RUN_COMPLETED").is_file()
    store.materialize_compatibility_view(restored)
    assert len(
        (store.run_directory / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2
    assert json.loads(
        (store.run_directory / "metrics.json").read_text(encoding="utf-8")
    ) == {"internal": True}


def test_corrupt_latest_and_missing_completion_marker_fall_back(tmp_path: Path) -> None:
    store, generation = _store(tmp_path)
    latest = store.commit_record(_record("q2"))
    (latest.generation_path / "delta.json").write_bytes(b'{"torn":')
    assert store.load_latest().generation == generation

    repaired = store.commit_record(_record("q2"))
    completed = store.commit_completion(
        {
            "metrics.json": stable_json_bytes({"ok": True}),
            "summary.md": b"ok\n",
        }
    )
    (completed.generation_path / "RUN_COMPLETED").unlink()
    assert store.load_latest().generation == repaired.generation
    assert store.load_latest().status == "running"


def test_cleanup_only_removes_pending_directory(tmp_path: Path) -> None:
    store, generation = _store(tmp_path)
    pending = store.generations / ".generation-00000003.pending-fixture"
    pending.mkdir()
    (pending / "partial").write_bytes(b"x")
    unknown = store.generations / "manual-history"
    unknown.mkdir()
    assert store.cleanup_uncommitted_temporaries() == [pending.name]
    assert not pending.exists()
    assert unknown.is_dir()
    assert store.load_latest().generation == generation


def test_concurrent_writer_is_rejected_without_state_change(tmp_path: Path) -> None:
    store, generation = _store(tmp_path)
    with store.writer_lock():
        with pytest.raises(ConcurrentWriterError, match="concurrent_writer"):
            store.commit_record(_record("q2"))
    assert store.load_latest().generation == generation


def test_writer_lock_is_released_after_interrupted_context(tmp_path: Path) -> None:
    store, generation = _store(tmp_path)
    with pytest.raises(InjectedCrash):
        with store.writer_lock():
            raise InjectedCrash("simulated_writer_interruption")
    recovered = store.commit_record(_record("q2"))
    assert recovered.generation == generation + 1
    assert len(recovered.records) == 2


@pytest.mark.parametrize(
    "point", ["mid_write", "flush_failure", "fsync_failure", "disk_full", "permission_denied"]
)
def test_single_file_atomic_writer_keeps_old_value(
    tmp_path: Path, point: str
) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old")
    with pytest.raises((InjectedCrash, OSError)):
        durable_atomic_write_bytes(
            target,
            b"new-value",
            injector=FaultInjector(point=point),  # type: ignore[arg-type]
            temporary_suffix="fault",
        )
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_sensitive_error_details_are_redacted() -> None:
    value = (
        "Authorization: Bearer secret api_key=private .env "
        "/Users/person/project/private.json"
    )
    sanitized = sanitize_error(value)
    for forbidden in ("secret", "private", ".env", "/Users/", "private.json"):
        assert forbidden not in sanitized
    assert "[redacted]" in sanitized
    assert "[environment-file]" in sanitized
    assert "[absolute-path]" in sanitized


def test_controlled_non_atomic_writer_has_stable_exit_two(tmp_path: Path) -> None:
    report = run_crash_consistency(
        _protocol(), work_root=tmp_path, controlled_fault="non_atomic_writer"
    )
    assert report["exit_code"] == EXIT_INVARIANT_VIOLATION
    assert report["status"] == "invariant_violation"
    assert report["violations"][-1]["invariant"] == (
        "previous_generation_immutable"
    )


def test_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = run_crash_consistency(_protocol(), work_root=tmp_path / "first")
    second = run_crash_consistency(_protocol(), work_root=tmp_path / "second")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_frozen_baseline_is_structurally_not_eligible() -> None:
    report = audit_frozen_baseline_eligibility(
        _protocol(), repository_root=ROOT
    )
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["status"] == "not_eligible"
    assert report["profile_count"] == 2
    assert {item["reason"] for item in report["profiles"]} == {
        "atomic_generation_evidence_unavailable"
    }
    assert all(item["files_modified"] == 0 for item in report["profiles"])


def test_protocol_drift_and_cli_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["schema_version"] = "2"
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    with pytest.raises(Exception, match="protocol_version"):
        load_protocol(path, repository_root=ROOT)

    missing = tmp_path / "private" / "missing-protocol.json"
    assert crash_cli.main(["--protocol", str(missing), "check"]) == 4
    usage = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert str(tmp_path) not in json.dumps(usage)

    assert crash_cli.main([]) == 4
    monkeypatch.setattr(
        crash_cli,
        "run_crash_consistency",
        lambda *_args, **_kwargs: {
            "schema_version": "1",
            "contract": "crash_consistency_v1",
            "gate": "crash_consistency_gate",
            "status": "invariant_violation",
            "exit_code": 2,
        },
    )
    assert crash_cli.main(["check"]) == 2
    assert crash_cli.main(["audit-frozen"]) == 3
