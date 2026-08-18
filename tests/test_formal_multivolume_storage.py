from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_multivolume_storage import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_VIOLATION,
    SINGLE_VOLUME_PRIMARY_BYTES,
    MultiVolumeStorageError,
    MultiVolumeStorageNotReady,
    MultiVolumeTopology,
    _synthetic_profiles,
    acquire_shard_writer,
    authorize_migration,
    build_launch_addendum,
    build_topology,
    canonical_json,
    load_profiles,
    load_protocol,
    simulate_run,
    verify_aggregate,
    verify_capacity,
    verify_resume_mapping,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_multivolume_storage_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_multivolume_storage.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _topology(
    protocol: dict[str, object],
    *,
    primary_bytes: tuple[int, ...] = (400_000_000_000, 400_000_000_000),
    primary_inodes: tuple[int, ...] = (50_000, 50_000),
) -> tuple[dict[str, object], list[object]]:
    profiles = _synthetic_profiles(
        primary_bytes=primary_bytes,
        primary_inodes=primary_inodes,
    )
    return build_topology(ROOT, protocol, profiles), profiles


def _attempts(topology: dict[str, object]) -> list[dict[str, object]]:
    validated = MultiVolumeTopology.model_validate(topology)
    return [
        {
            "shard_index": binding.shard_index,
            "volume_identity": binding.volume_identity,
            "selected": True,
            "volume_online": True,
            "manifest_sha256": stable_hash(
                {"manifest": binding.shard_index}
            ),
            "generation_identity": stable_hash(
                {"generation": binding.shard_index}
            ),
        }
        for binding in validated.shard_bindings
    ]


def test_protocol_binds_existing_storage_and_execution_contracts(
    protocol: dict[str, object],
) -> None:
    assert protocol["population"] == {
        "query_count": 1000,
        "queries_per_shard": 50,
        "shard_count": 20,
    }
    assert protocol["allocation"]["cross_filesystem_atomic_rename"] is False
    addendum = build_launch_addendum(protocol)
    assert addendum["legacy_single_volume_authorization_reusable"] is False
    assert addendum["activation_requirements"][
        "fresh_multivolume_topology"
    ]


def test_two_primary_volumes_remove_single_volume_capacity_requirement(
    protocol: dict[str, object],
) -> None:
    topology, profiles = _topology(protocol)
    validated = MultiVolumeTopology.model_validate(topology)
    report = verify_capacity(topology, profiles)
    assert report["single_volume_primary_limit_removed"] is True
    assert len(validated.primary_volume_identities) == 2
    assert all(
        requirement.required_bytes < SINGLE_VOLUME_PRIMARY_BYTES
        for requirement in validated.volume_requirements.values()
        if requirement.role == "primary"
    )
    assert [binding.shard_index for binding in validated.shard_bindings] == list(
        range(20)
    )
    assert all(
        binding.colocated_roles
        == [
            "pending",
            "generation",
            "resource_ledger",
            "provider_raw_response",
            "operation_audit_chain",
        ]
        for binding in validated.shard_bindings
    )


def test_total_capacity_cannot_hide_fragmented_volume(
    protocol: dict[str, object],
) -> None:
    topology, profiles = _topology(
        protocol,
        primary_bytes=(700_000_000_000, 30_000_000_000),
    )
    assert sum(
        item.available_bytes for item in profiles if item.role == "primary"
    ) > SINGLE_VOLUME_PRIMARY_BYTES
    with pytest.raises(
        MultiVolumeStorageError, match="volume_bytes_insufficient"
    ):
        verify_capacity(topology, profiles)


def test_inode_quota_and_writer_observations_fail_closed(
    protocol: dict[str, object],
) -> None:
    topology, profiles = _topology(
        protocol, primary_inodes=(50_000, 1)
    )
    with pytest.raises(
        MultiVolumeStorageError, match="volume_inodes_insufficient"
    ):
        verify_capacity(topology, profiles)
    unknown = [item.model_copy(deep=True) for item in _synthetic_profiles()]
    primary = next(item for item in unknown if item.role == "primary")
    primary.filesystem_quota_bytes = "not_available"
    primary.max_concurrent_writers = "not_available"
    unknown_topology = build_topology(ROOT, protocol, unknown)
    with pytest.raises(MultiVolumeStorageNotReady, match="missing_volume"):
        verify_capacity(unknown_topology, unknown)


def test_offline_volume_and_mount_replacement_are_rejected(
    protocol: dict[str, object],
) -> None:
    topology, profiles = _topology(protocol)
    offline = [item.model_copy(deep=True) for item in profiles]
    offline[0].online = False
    with pytest.raises(MultiVolumeStorageError, match="volume_offline"):
        verify_capacity(topology, offline)
    replaced = [item.model_copy(deep=True) for item in profiles]
    replaced[0].mount_identity = stable_hash({"replaced": True})
    with pytest.raises(
        MultiVolumeStorageError, match="volume_identity_drift"
    ):
        verify_capacity(topology, replaced)


def test_resume_binding_is_immutable(protocol: dict[str, object]) -> None:
    topology, _profiles = _topology(protocol)
    validated = MultiVolumeTopology.model_validate(topology)
    resume = [
        {
            "shard_index": item.shard_index,
            "volume_identity": item.volume_identity,
            "backup_volume_identity": item.backup_volume_identity,
        }
        for item in validated.shard_bindings
    ]
    verify_resume_mapping(topology, resume)
    resume[0]["volume_identity"] = validated.primary_volume_identities[-1]
    with pytest.raises(
        MultiVolumeStorageError, match="resume_volume_binding_drift"
    ):
        verify_resume_mapping(topology, resume)


def test_migration_requires_backup_restore_and_fresh_attestations() -> None:
    result = authorize_migration(
        backup_verified=True,
        empty_target=True,
        restored_hash_verified=True,
        new_host_attestation_fresh=True,
        new_storage_attestation_fresh=True,
        direct_move_requested=False,
    )
    assert result["method"] == "backup_verify_empty_restore_reattest"
    with pytest.raises(
        MultiVolumeStorageError, match="direct_cross_volume_move_forbidden"
    ):
        authorize_migration(
            backup_verified=True,
            empty_target=True,
            restored_hash_verified=True,
            new_host_attestation_fresh=True,
            new_storage_attestation_fresh=True,
            direct_move_requested=True,
        )
    with pytest.raises(
        MultiVolumeStorageError, match="migration_preconditions_incomplete"
    ):
        authorize_migration(
            backup_verified=False,
            empty_target=True,
            restored_hash_verified=True,
            new_host_attestation_fresh=True,
            new_storage_attestation_fresh=True,
            direct_move_requested=False,
        )


def test_aggregate_uses_hash_references_without_copying_history(
    protocol: dict[str, object],
) -> None:
    topology, _profiles = _topology(protocol)
    attempts = _attempts(topology)
    aggregate = verify_aggregate(topology, attempts)
    assert aggregate["copy_or_history_rewrite"] is False
    assert len(aggregate["references"]) == 20
    with pytest.raises(
        MultiVolumeStorageError, match="aggregate_partial_shard_inventory"
    ):
        verify_aggregate(topology, attempts[:-1])
    duplicate = copy.deepcopy(attempts)
    duplicate.append(copy.deepcopy(attempts[0]))
    with pytest.raises(
        MultiVolumeStorageError, match="aggregate_duplicate_shard"
    ):
        verify_aggregate(topology, duplicate)
    wrong_volume = copy.deepcopy(attempts)
    wrong_volume[0]["volume_identity"] = "f" * 64
    with pytest.raises(
        MultiVolumeStorageError, match="aggregate_reference_invalid"
    ):
        verify_aggregate(topology, wrong_volume)


def test_double_writer_is_rejected() -> None:
    active: dict[int, str] = {}
    acquire_shard_writer(active, shard_index=4, writer_identity="writer-a")
    acquire_shard_writer(active, shard_index=4, writer_identity="writer-a")
    with pytest.raises(
        MultiVolumeStorageError, match="concurrent_shard_writer_rejected"
    ):
        acquire_shard_writer(
            active, shard_index=4, writer_identity="writer-b"
        )


def test_topology_detects_duplicate_or_mixed_generation_binding(
    protocol: dict[str, object],
) -> None:
    topology, _profiles = _topology(protocol)
    changed = copy.deepcopy(topology)
    changed["shard_bindings"][1]["shard_index"] = 0
    changed["topology_sha256"] = stable_hash(
        {key: value for key, value in changed.items() if key != "topology_sha256"}
    )
    with pytest.raises(ValueError, match="shard population"):
        MultiVolumeTopology.model_validate(changed)


def test_simulation_covers_1000_queries_and_is_byte_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_run(ROOT, protocol)
    second = simulate_run(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["query_count"] == 1000
    assert first["shard_count"] == 20
    assert first["scenario_count"] == 9
    assert first["result_equivalent"] is True
    assert {row["scenario"] for row in first["scenarios"]} == {
        "aggregate_matches_uninterrupted",
        "capacity_fragmentation",
        "cross_volume_restore",
        "double_writer",
        "dual_volume_qualified_single_volume_each_insufficient",
        "inode_shortage",
        "mount_identity_replaced",
        "single_volume_enospc",
        "volume_disappeared",
    }


def test_protocol_and_profile_drift_fail_closed(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    changed = copy.deepcopy(protocol)
    changed["allocation"]["resume_mapping_mutable"] = True
    payload = copy.deepcopy(changed)
    payload.pop("protocol_sha256")
    changed["protocol_sha256"] = stable_hash(payload)
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json(changed))
    with pytest.raises(MultiVolumeStorageError, match="protocol_content_drift"):
        load_protocol(path, repository_root=ROOT)
    profiles = _synthetic_profiles()
    profile_path = tmp_path / "profiles.json"
    profile_path.write_bytes(
        canonical_json(
            {
                "schema_version": "1",
                "profiles": [
                    item.model_dump(mode="json") for item in profiles
                ],
            }
        )
    )
    assert len(load_profiles(profile_path)) == 4
    malformed = json.loads(profile_path.read_text())
    malformed["profiles"].append(copy.deepcopy(malformed["profiles"][0]))
    profile_path.write_bytes(canonical_json(malformed))
    with pytest.raises(
        MultiVolumeStorageError, match="duplicate_volume_identity"
    ):
        load_profiles(profile_path)


def test_cli_build_verify_simulate_and_readiness(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    profiles = _synthetic_profiles()
    profile_path = tmp_path / "profiles.json"
    topology_path = tmp_path / "topology.json"
    profile_path.write_bytes(
        canonical_json(
            {
                "schema_version": "1",
                "profiles": [
                    item.model_dump(mode="json") for item in profiles
                ],
            }
        )
    )
    build = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "build-topology",
            "--profiles",
            str(profile_path),
            "--output",
            str(topology_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert build.returncode == EXIT_READY
    verify = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "verify-capacity",
            "--profiles",
            str(profile_path),
            "--topology",
            str(topology_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert verify.returncode == EXIT_READY
    simulate = subprocess.run(
        [sys.executable, str(CLI), "simulate-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert simulate.returncode == EXIT_READY
    readiness_command = [sys.executable, str(CLI), "audit-readiness"]
    first = subprocess.run(
        readiness_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    second = subprocess.run(
        readiness_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert first.returncode == second.returncode == EXIT_NOT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    payload = json.loads(first.stdout)
    assert payload["controls_ready"] is True
    assert payload["primary_capacity_observed"] is True
    assert payload["primary_inode_capacity_observed"] is True
    assert "observed_primary_available_bytes_lower_bound" not in payload


def test_cli_violation_has_stable_exit_and_no_traceback(
    tmp_path: Path,
) -> None:
    missing = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "verify-capacity",
            "--profiles",
            str(tmp_path / "missing-profiles.json"),
            "--topology",
            str(tmp_path / "missing-topology.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert missing.returncode == EXIT_VIOLATION
    assert missing.stderr == b""
    assert b"Traceback" not in missing.stdout
