from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_shard_streaming_retention import (
    BACKUP_REQUIRED_BYTES,
    CURRENT_PRIMARY_AVAILABLE_BYTES,
    ShardRetentionError,
    ShardRetentionNotReady,
    EvictionLedger,
    acquire_writer,
    aggregate_mixed,
    audit_readiness,
    build_addendum,
    build_fixture_authority,
    canonical_json,
    capacity_requirement,
    create_archive,
    evict_local,
    load_protocol,
    restore_authority,
    simulate_streaming,
    verify_archive,
    verify_capacity,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/formal_shard_streaming_retention_v1_protocol.json"
)
CLI = ROOT / "scripts/check_formal_shard_retention.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def test_protocol_freezes_window_and_default_off(
    protocol: dict[str, object],
) -> None:
    assert protocol["policy"] == {
        "active_shard_window": 4,
        "allowed_active_shard_windows": [1, 2, 4],
        "default_enabled": False,
        "runtime_window_adjustment_allowed": False,
        "window_limits_residency_only": True,
    }
    addendum = build_addendum(protocol)
    assert addendum["active_shard_window"] == 4
    assert addendum["default_enabled"] is False
    with pytest.raises(
        ShardRetentionError, match="runtime_window_adjustment_forbidden"
    ):
        build_addendum(protocol, window=2)


def test_window_capacity_is_exact_and_current_primary_qualifies() -> None:
    assert capacity_requirement(1)["required_primary_bytes"] == 82_946_555_904
    assert capacity_requirement(2)["required_primary_bytes"] == 117_977_382_912
    assert capacity_requirement(4)["required_primary_bytes"] == 188_039_036_928
    assert all(
        CURRENT_PRIMARY_AVAILABLE_BYTES
        >= capacity_requirement(window)["required_primary_bytes"]
        for window in (1, 2, 4)
    )
    with pytest.raises(
        ShardRetentionError, match="active_shard_window_invalid"
    ):
        capacity_requirement(3)


def test_capacity_keeps_unknown_backup_as_blocker(
    protocol: dict[str, object],
) -> None:
    report = verify_capacity(
        build_addendum(protocol),
        primary_available_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
        primary_available_inodes=100_000,
        primary_quota_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
        backup_available_bytes="not_available",
        backup_available_inodes="not_available",
        backup_quota_bytes="not_available",
        backup_failure_domain_independent="not_available",
    )
    assert report["exit_code"] == 3
    assert report["primary_qualified"] is True
    assert report["backup_qualified"] is False
    with pytest.raises(
        ShardRetentionError, match="backup_capacity_or_failure_domain"
    ):
        verify_capacity(
            build_addendum(protocol),
            primary_available_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
            primary_available_inodes=100_000,
            primary_quota_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
            backup_available_bytes=BACKUP_REQUIRED_BYTES - 1,
            backup_available_inodes=300_000,
            backup_quota_bytes=BACKUP_REQUIRED_BYTES - 1,
            backup_failure_domain_independent=True,
        )
    drifted = build_addendum(protocol)
    drifted["active_shard_window"] = 1
    drifted["addendum_sha256"] = "0" * 64
    with pytest.raises(
        ShardRetentionError, match="addendum_identity_invalid"
    ):
        verify_capacity(
            drifted,
            primary_available_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
            primary_available_inodes=100_000,
            primary_quota_bytes=CURRENT_PRIMARY_AVAILABLE_BYTES,
            backup_available_bytes="not_available",
            backup_available_inodes="not_available",
            backup_quota_bytes="not_available",
            backup_failure_domain_independent="not_available",
        )


def test_archive_inventory_hash_and_parent_chain() -> None:
    first = build_fixture_authority(0)
    first_archive = create_archive(
        first,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    verify_archive(first_archive, authority=first)
    second = build_fixture_authority(1)
    second_archive = create_archive(
        second,
        parent_archive_sha256=first_archive["archive_sha256"],
        backup_qualified=True,
        restore_drill_verified=True,
    )
    archives = {
        first_archive["archive_sha256"]: first_archive,
        second_archive["archive_sha256"]: second_archive,
    }
    verify_archive(second_archive, authority=second, known_archives=archives)
    tampered = copy.deepcopy(second_archive)
    tampered["files"][0]["size"] += 1
    with pytest.raises(
        ShardRetentionError, match="archive_integrity_invalid"
    ):
        verify_archive(tampered, authority=second, known_archives=archives)


def test_release_requires_backup_restore_and_no_authoritative_reference() -> None:
    authority = build_fixture_authority(0)
    with pytest.raises(
        ShardRetentionNotReady, match="qualified_backup_unavailable"
    ):
        create_archive(
            authority,
            parent_archive_sha256=None,
            backup_qualified=False,
            restore_drill_verified=True,
        )
    archive = create_archive(
        authority,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    authority.resume_point = True
    with pytest.raises(
        ShardRetentionError, match="local_release_forbidden_reference"
    ):
        evict_local(authority, archive, EvictionLedger())
    authority.resume_point = False
    with pytest.raises(
        ShardRetentionError, match="local_release_forbidden_reference"
    ):
        evict_local(
            authority,
            archive,
            EvictionLedger(),
            transparency_path_referenced=True,
        )


def test_interrupted_release_preserves_authority_and_receipt_chain() -> None:
    authority = build_fixture_authority(0)
    archive = create_archive(
        authority,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    ledger = EvictionLedger()
    with pytest.raises(
        ShardRetentionError,
        match="eviction_interrupted_authority_preserved",
    ):
        evict_local(
            authority,
            archive,
            ledger,
            fault_after_start=True,
        )
    assert authority.local_present is True
    assert [row["state"] for row in ledger.receipts] == ["eviction_started"]
    ledger.receipts[0]["sequence"] = 2
    with pytest.raises(
        ShardRetentionError, match="eviction_receipt_chain_invalid"
    ):
        ledger.verify()


def test_restore_and_mixed_aggregate_do_not_repeat_requests() -> None:
    archives: dict[str, dict[str, object]] = {}
    local: dict[int, object] = {}
    references: list[dict[str, object]] = []
    for shard in range(20):
        authority = build_fixture_authority(shard)
        if shard == 19:
            local[shard] = authority
            references.append(
                {
                    "shard_index": shard,
                    "location": "local",
                    "authority_sha256": authority.authority_sha256,
                }
            )
            continue
        archive = create_archive(
            authority,
            parent_archive_sha256=None,
            backup_qualified=True,
            restore_drill_verified=True,
        )
        archives[archive["archive_sha256"]] = archive
        references.append(
            {
                "shard_index": shard,
                "location": "archive",
                "archive_sha256": archive["archive_sha256"],
                "authority_sha256": archive["authority_sha256"],
            }
        )
    report = aggregate_mixed(references, archives=archives, local=local)
    assert report["query_count"] == 1000
    restored = restore_authority(next(iter(archives.values())))
    assert restored.query_results == build_fixture_authority(0).query_results


def test_non_genesis_archive_restore_requires_and_accepts_parent_chain() -> None:
    first = build_fixture_authority(0)
    first_archive = create_archive(
        first,
        parent_archive_sha256=None,
        backup_qualified=True,
        restore_drill_verified=True,
    )
    second = build_fixture_authority(1)
    second_archive = create_archive(
        second,
        parent_archive_sha256=first_archive["archive_sha256"],
        backup_qualified=True,
        restore_drill_verified=True,
    )
    with pytest.raises(ShardRetentionError, match="archive_parent_missing"):
        restore_authority(second_archive)
    restored = restore_authority(
        second_archive,
        known_archives={
            first_archive["archive_sha256"]: first_archive,
            second_archive["archive_sha256"]: second_archive,
        },
    )
    assert restored.authority_sha256 == second.authority_sha256


def test_duplicate_shard_and_double_writer_fail_closed() -> None:
    authority = build_fixture_authority(0)
    acquire_writer(authority, "writer-a")
    with pytest.raises(
        ShardRetentionError, match="concurrent_shard_writer_rejected"
    ):
        acquire_writer(authority, "writer-b")
    duplicate = [
        {
            "shard_index": 0,
            "location": "local",
            "authority_sha256": authority.authority_sha256,
        }
    ] * 20
    with pytest.raises(
        ShardRetentionError, match="aggregate_duplicate_shard"
    ):
        aggregate_mixed(duplicate, archives={}, local={0: authority})


def test_1000_query_streaming_windows_are_exact_and_equivalent(
    protocol: dict[str, object],
) -> None:
    report = simulate_streaming(protocol)
    assert report["exit_code"] == 0
    for window in (1, 2, 4):
        row = report["window_reports"][str(window)]
        assert row["peak_resident_shards"] == window
        assert row["adapter_call_count"] == 1000
        assert row["duplicate_request_count"] == 0
        assert row["resource_ledger_conserved"] is True
        assert row["aggregate_matches_uninterrupted"] is True


def test_real_readiness_reports_primary_gain_but_backup_blocker(
    protocol: dict[str, object],
) -> None:
    report = audit_readiness(protocol)
    assert report["status"] == "not_ready_missing_qualified_backup"
    assert report["exit_code"] == 3
    assert report["current_primary_qualified"] is True
    assert report["full1000_blocker_cleared"] is False


def test_cli_is_deterministic_and_never_tracebacks() -> None:
    command = [
        sys.executable,
        str(CLI),
        "--repository-root",
        str(ROOT),
        "simulate-streaming",
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert json.loads(first.stdout)["query_count"] == 1000


def test_cli_invalid_addendum_returns_two_without_traceback(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"active_shard_window":4}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--repository-root",
            str(ROOT),
            "verify-capacity",
            "--addendum",
            str(bad),
            "--primary-available-bytes",
            str(CURRENT_PRIMARY_AVAILABLE_BYTES),
            "--primary-available-inodes",
            "100000",
            "--primary-quota-bytes",
            str(CURRENT_PRIMARY_AVAILABLE_BYTES),
            "--backup-available-bytes",
            "not_available",
            "--backup-available-inodes",
            "not_available",
            "--backup-quota-bytes",
            "not_available",
            "--backup-failure-domain-independent",
            "not_available",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout


def test_report_bytes_are_deterministic(
    protocol: dict[str, object],
) -> None:
    assert canonical_json(simulate_streaming(protocol)) == canonical_json(
        simulate_streaming(protocol)
    )


def test_readiness_freshness_and_public_contract_are_integrated() -> None:
    readiness = json.loads(
        (
            ROOT / "benchmark/validation_readiness_bundle_v1_contract.json"
        ).read_text(encoding="utf-8")
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"]
        == "architecture_formal_shard_streaming_retention_ready"
    )
    assert claim["status"] == "verified"
    assert set(claim["evidence_ids"]) == {
        "formal_shard_retention_addendum",
        "formal_shard_retention_protocol",
        "formal_shard_retention_readiness",
        "formal_shard_retention_simulation",
    }
    gate = next(
        row
        for row in readiness["read_only_gates"]
        if row["gate_id"] == "formal_shard_streaming_retention"
    )
    assert gate["expected_exit_code"] == 3

    freshness = json.loads(
        (
            ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json"
        ).read_text(encoding="utf-8")
    )
    assert freshness["claim_component_bindings"][claim["claim_id"]] == [
        "formal_shard_streaming_retention"
    ]
    assert freshness["gate_component_bindings"][gate["gate_id"]] == [
        "formal_shard_streaming_retention"
    ]

    public = json.loads(
        (
            ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
        ).read_text(encoding="utf-8")
    )
    assert public["artifact_contracts"]["formal_shard_streaming_retention"] == (
        "benchmark/formal_shard_streaming_retention_v1_protocol.json"
    )
    assert public["cli_contracts"]["formal_shard_streaming_retention"][
        "exit_codes"
    ] == [0, 2, 3, 4]
