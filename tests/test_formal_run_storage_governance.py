from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_run_storage_governance import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_VIOLATION,
    CapacityObservation,
    CleanupCandidate,
    InjectedCapacityLedger,
    StorageGovernanceError,
    StoragePlan,
    audit_readiness,
    build_launch_addendum,
    build_storage_plan,
    canonical_json,
    cleanup_eligibility,
    enforce_capture_limit,
    load_protocol,
    simulate_pressure,
    verify_capacity,
)
from scholar_agent.evaluation.provider_ingest_provenance import (
    EncodingMetadata,
    ProviderCaptureRecorder,
    opaque_identity,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/formal_run_storage_governance_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_run_storage.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _writer() -> str:
    return stable_hash({"writer": "test"})


def _operation(name: str) -> str:
    return stable_hash({"operation": name})


def test_plan_binds_frozen_full1000_limits(protocol: dict[str, object]) -> None:
    plan = build_storage_plan(ROOT, protocol)
    validated = StoragePlan.model_validate(plan)
    assert validated.query_count == 1000
    assert validated.max_http_attempts == 19280
    assert validated.selected_generation_upper == 1040
    assert validated.shard_count == 20
    assert validated.quotas["provider_response"].max_bytes == 32 * 1024 * 1024
    assert validated.preflight["credit_for_compression_or_sparse_files"] == 0
    assert validated.preflight["credit_for_future_cleanup"] == 0
    addendum = build_launch_addendum(ROOT, protocol, plan)
    assert addendum["legacy_launch_authorization_reusable"] is False
    assert addendum["activation_requirements"]["primary_and_backup_capacity_verified"]


def test_protocol_rejects_bound_input_drift(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    changed = copy.deepcopy(protocol)
    changed["population"]["http_attempt_upper"] = 19279  # type: ignore[index]
    changed.pop("protocol_sha256")
    changed["protocol_sha256"] = stable_hash(changed)
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json(changed))
    with pytest.raises(StorageGovernanceError, match="protocol_content_drift"):
        load_protocol(path, repository_root=ROOT)


def test_reserve_commit_release_conserves_bytes_and_inodes() -> None:
    ledger = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=3,
        quota_bytes=100,
    )
    writer = _writer()
    ledger.acquire_writer(writer)
    first = _operation("first")
    second = _operation("second")
    ledger.reserve(
        operation_identity=first,
        writer_identity=writer,
        artifact_type="generation",
        requested_bytes=60,
        requested_files=1,
    )
    ledger.commit(first)
    ledger.reserve(
        operation_identity=second,
        writer_identity=writer,
        artifact_type="temporary",
        requested_bytes=40,
        requested_files=2,
    )
    ledger.release(second)
    value = ledger.snapshot()
    assert value.committed_bytes == 60
    assert value.committed_files == 1
    assert value.reserved_bytes == 0
    assert value.reserved_files == 0


def test_low_space_and_inode_failure_preserve_previous_commit() -> None:
    ledger = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=2,
        quota_bytes=100,
    )
    writer = _writer()
    ledger.acquire_writer(writer)
    first = _operation("first")
    ledger.reserve(
        operation_identity=first,
        writer_identity=writer,
        artifact_type="generation",
        requested_bytes=50,
        requested_files=1,
    )
    ledger.commit(first)
    before = ledger.snapshot()
    with pytest.raises(StorageGovernanceError, match="storage_bytes_unavailable"):
        ledger.reserve(
            operation_identity=_operation("enospc"),
            writer_identity=writer,
            artifact_type="generation",
            requested_bytes=51,
            requested_files=1,
        )
    with pytest.raises(StorageGovernanceError, match="storage_inodes_unavailable"):
        ledger.reserve(
            operation_identity=_operation("inode"),
            writer_identity=writer,
            artifact_type="generation",
            requested_bytes=1,
            requested_files=2,
        )
    after = ledger.snapshot()
    assert (after.committed_bytes, after.committed_files) == (
        before.committed_bytes,
        before.committed_files,
    )


def test_capacity_drop_rolls_back_reservation() -> None:
    ledger = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=2,
        quota_bytes=100,
    )
    writer = _writer()
    ledger.acquire_writer(writer)
    operation = _operation("drop")
    ledger.reserve(
        operation_identity=operation,
        writer_identity=writer,
        artifact_type="generation",
        requested_bytes=80,
        requested_files=1,
    )
    ledger.shrink(capacity_bytes=79)
    with pytest.raises(StorageGovernanceError, match="capacity_dropped_before_commit"):
        ledger.commit(operation)
    assert ledger.committed_bytes == 0
    assert not ledger.reservations


def test_capture_limit_is_exact_and_fail_closed() -> None:
    assert enforce_capture_limit(b"abcd", enabled=True, max_bytes=4)["accepted"]
    exceeded = enforce_capture_limit(b"abcde", enabled=True, max_bytes=4)
    assert exceeded == {
        "capture_enabled": True,
        "accepted": False,
        "attempt_terminal": True,
        "reason_code": "capture_size_exceeded",
        "captured_bytes": 0,
    }
    disabled = enforce_capture_limit(b"x", enabled=False, max_bytes=4)
    assert disabled["accepted"] is False
    assert disabled["reason_code"] == "capture_not_enabled"


def test_provider_capture_recorder_never_parses_or_truncates_oversize() -> None:
    recorder = ProviderCaptureRecorder(
        run_identity=opaque_identity("run", "storage"),
        query_identity=opaque_identity("query", "storage"),
        attempt_identity=opaque_identity("attempt", "storage"),
        checkpoint_generation=1,
        manifest_identity=opaque_identity("manifest", "storage"),
        capture_limit_bytes=4,
    )
    envelope = recorder.record_attempt(
        source="openalex",
        request_sequence=0,
        resource_operation_identity=opaque_identity("operation", "storage"),
        parser_name="openalex_search",
        raw_bytes=b'{"results":[]}',
        http_status=200,
        content_type="application/json",
        encoding=EncodingMetadata(state="known", value="utf-8"),
        compression="identity",
        terminal_state="success",
    )
    assert envelope.terminal_state == "capture_size_exceeded"
    assert envelope.terminal_reason_code == "capture_size_exceeded"
    assert envelope.raw_response is None
    assert envelope.parsed_record_count == 0


@pytest.mark.parametrize(
    ("candidate", "eligible", "reason"),
    [
        (
            CleanupCandidate(
                relative_path="tmp/pending.json",
                kind="temporary",
                committed=False,
                backup_verified=False,
                generations_older_than_window=0,
            ),
            True,
            "uncommitted_temporary_only",
        ),
        (
            CleanupCandidate(
                relative_path="generations/old",
                kind="generation",
                committed=True,
                backup_verified=True,
                generations_older_than_window=3,
            ),
            True,
            "verified_backup_outside_retention_window",
        ),
        (
            CleanupCandidate(
                relative_path="generations/recent",
                kind="generation",
                committed=True,
                backup_verified=True,
                generations_older_than_window=2,
            ),
            False,
            "inside_retention_window",
        ),
        (
            CleanupCandidate(
                relative_path="raw/response.bin",
                kind="generation",
                committed=True,
                backup_verified=True,
                generations_older_than_window=9,
                is_raw_response=True,
            ),
            False,
            "protected_authoritative_artifact",
        ),
    ],
)
def test_retention_window_and_protected_artifacts(
    candidate: CleanupCandidate, eligible: bool, reason: str
) -> None:
    assert cleanup_eligibility(candidate, retention_window=2) == (eligible, reason)


def test_backup_capacity_and_unknown_quota_fail_closed(
    protocol: dict[str, object]
) -> None:
    plan = build_storage_plan(ROOT, protocol)
    primary_required = plan["preflight"]["requirements"]["primary"]
    backup_required = plan["preflight"]["requirements"]["backup"]
    primary = CapacityObservation(
        available_bytes=primary_required["bytes"],
        available_inodes=primary_required["inodes"],
        filesystem_quota_bytes=primary_required["bytes"],
        target_kind="primary",
    )
    backup = CapacityObservation(
        available_bytes=backup_required["bytes"],
        available_inodes=backup_required["inodes"],
        filesystem_quota_bytes="not_available",
        target_kind="backup",
    )
    report = verify_capacity(plan, primary=primary, backup=backup)
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["missing_capacity_fields"] == ["backup.filesystem_quota_bytes"]

    insufficient = backup.model_copy(
        update={
            "filesystem_quota_bytes": backup_required["bytes"],
            "available_bytes": backup_required["bytes"] - 1,
        }
    )
    report = verify_capacity(plan, primary=primary, backup=insufficient)
    assert report["exit_code"] == EXIT_VIOLATION
    assert report["violations"][0]["invariant"] == "available_bytes_below_required"

    qualified = verify_capacity(
        plan,
        primary=primary,
        backup=backup.model_copy(
            update={
                "filesystem_quota_bytes": backup_required["bytes"],
                "available_bytes": backup_required["bytes"],
            }
        ),
    )
    assert qualified["exit_code"] == EXIT_READY


def test_concurrent_writer_is_rejected() -> None:
    ledger = InjectedCapacityLedger(
        capacity_bytes=100,
        capacity_inodes=2,
        quota_bytes=100,
    )
    ledger.acquire_writer(_writer())
    with pytest.raises(StorageGovernanceError, match="concurrent_writer_rejected"):
        ledger.acquire_writer(stable_hash({"writer": "second"}))


def test_1000_query_pressure_matrix_and_reports_are_deterministic(
    protocol: dict[str, object]
) -> None:
    first = simulate_pressure(ROOT, protocol)
    second = simulate_pressure(ROOT, protocol)
    assert first["exit_code"] == EXIT_READY
    assert first["closure"] == {
        "query_count": 1000,
        "http_attempt_upper": 19280,
        "selected_generation_upper": 1040,
        "shard_count": 20,
    }
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == 9


def test_real_readiness_remains_blocked(protocol: dict[str, object]) -> None:
    report = audit_readiness(ROOT, protocol)
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["full1000_blocker_cleared"] is False
    assert len(report["missing_capacity_fields"]) == 6


def test_readiness_freshness_and_launch_addendum_are_registered() -> None:
    readiness = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text()
    )
    assert {
        item["blocker_id"] for item in readiness["blockers"]
    } == {
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    }
    claim = next(
        item
        for item in readiness["claims"]
        if item["claim_id"] == "architecture_formal_run_storage_controls_ready"
    )
    assert claim["status"] == "verified"
    assert "actual primary/backup" in claim["boundary"]
    assert any(
        item["gate_id"] == "formal_run_storage_governance"
        and item["expected_exit_code"] == EXIT_NOT_READY
        for item in readiness["read_only_gates"]
    )
    addenda = json.loads(
        (ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json").read_text()
    )
    assert (
        addenda["claim_component_bindings"][
            "architecture_formal_run_storage_controls_ready"
        ]
        == ["formal_run_storage_governance"]
    )


def test_cli_exit_codes_and_json_are_stable(tmp_path: Path) -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=ROOT,
            env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": "src"},
            capture_output=True,
            check=False,
        )

    first = run("simulate-pressure")
    second = run("simulate-pressure")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert json.loads(first.stdout)["scenario_count"] == 9

    audit = run("audit-readiness")
    assert audit.returncode == EXIT_NOT_READY
    assert json.loads(audit.stdout)["status"] == "not_ready_capacity_unverified"
    assert audit.stderr == b""

    malformed = tmp_path / "protocol.json"
    malformed.write_text("{}\n", encoding="utf-8")
    rejected = run("--protocol", str(malformed), "build-plan")
    assert rejected.returncode == EXIT_VIOLATION
    assert json.loads(rejected.stdout)["status"] == "quota_or_retention_violation"
    assert rejected.stderr == b""
