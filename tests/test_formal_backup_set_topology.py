from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_compaction import build_shard_state
from scholar_agent.evaluation.formal_backup_set_topology import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    BackupSetError,
    all_capacity_models,
    assigned_shards,
    audit_readiness,
    build_backup_set,
    build_topology,
    calculate_capacity,
    load_protocol,
    simulate_set,
    synthetic_profiles,
    verify_backup_set,
    verify_profiles,
    verify_set,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/formal_backup_set_topology_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_set.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR"):
            if os.environ.get(name):
                environment[name] = os.environ[name]
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _set_fixture(
    protocol: dict[str, object], member_count: int = 4
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[int, dict[int, dict[str, object]]],
]:
    topology = build_topology(protocol, member_count=member_count)
    profiles = synthetic_profiles(topology)
    states = {
        shard: build_shard_state(shard, cursor=50, generation=1)
        for shard in range(20)
    }
    manifest, archives = build_backup_set(topology, states, profiles)
    return topology, profiles, manifest, archives


@pytest.mark.parametrize(
    ("count", "maximum_bytes", "maximum_inodes"),
    [
        (2, 533_012_676_608, 56_321),
        (3, 426_130_625_878, 44_623),
        (4, 285_112_532_992, 30_413),
    ],
)
def test_capacity_models_are_worst_case_and_auditable(
    protocol: dict[str, object],
    count: int,
    maximum_bytes: int,
    maximum_inodes: int,
) -> None:
    model = calculate_capacity(protocol)["capacity_models"][str(count)]
    assert model["maximum_member_required_bytes"] == maximum_bytes
    assert model["maximum_member_required_inodes"] == maximum_inodes
    assert sum(row["required_bytes"] for row in model["members"]) == model[
        "set_required_bytes"
    ]
    assert sum(row["required_inodes"] for row in model["members"]) == model[
        "set_required_inodes"
    ]
    assert model["extra_bytes_from_per_member_staging_and_index"] > 0
    for key in (
        "compression_credit_bytes",
        "future_cleanup_credit_bytes",
        "future_deduplication_credit_bytes",
        "sparse_file_credit_bytes",
    ):
        assert model[key] == 0


@pytest.mark.parametrize("count", [2, 3, 4])
def test_assignment_is_closed_unique_and_deterministic(
    protocol: dict[str, object], count: int
) -> None:
    topology = build_topology(protocol, member_count=count)
    flattened = [
        shard for row in topology["members"] for shard in row["assigned_shards"]
    ]
    assert sorted(flattened) == list(range(20))
    assert len(flattened) == len(set(flattened))
    for index in range(count):
        assert topology["members"][index]["assigned_shards"] == assigned_shards(
            count, index
        )
    assert build_topology(protocol, member_count=count) == topology


def test_quota_pool_filesystem_and_failure_domain_must_be_distinct(
    protocol: dict[str, object],
) -> None:
    topology = build_topology(protocol, member_count=4)
    for field in (
        "quota_pool_identity",
        "filesystem_identity",
        "failure_domain_identity",
    ):
        profiles = synthetic_profiles(topology)
        profiles[1][field] = profiles[0][field]
        with pytest.raises(
            BackupSetError, match="member_capacity_or_failure_domain_overlap"
        ):
            verify_profiles(topology, profiles)


def test_directory_alias_cannot_count_as_independent_member(
    protocol: dict[str, object],
) -> None:
    topology = build_topology(protocol, member_count=2)
    profiles = synthetic_profiles(topology)
    profiles[1]["filesystem_identity"] = profiles[0]["filesystem_identity"]
    profiles[1]["quota_pool_identity"] = profiles[0]["quota_pool_identity"]
    with pytest.raises(BackupSetError, match="overlap"):
        verify_profiles(topology, profiles)


@pytest.mark.parametrize(
    "field", ["available_bytes", "available_inodes", "quota_bytes", "max_writers"]
)
def test_each_member_must_meet_its_own_capacity(
    protocol: dict[str, object], field: str
) -> None:
    topology = build_topology(protocol, member_count=3)
    profiles = synthetic_profiles(topology)
    member = topology["members"][1]
    threshold = {
        "available_bytes": member["required_bytes"],
        "available_inodes": member["required_inodes"],
        "quota_bytes": member["required_bytes"],
        "max_writers": member["required_writers"],
    }[field]
    profiles[1][field] = threshold - 1
    with pytest.raises(BackupSetError, match="member_capacity_insufficient"):
        verify_profiles(topology, profiles)


def test_complete_set_restores_1000_queries(
    protocol: dict[str, object],
) -> None:
    topology, _profiles, manifest, archives = _set_fixture(protocol)
    profiles = synthetic_profiles(topology)
    restored = verify_backup_set(topology, manifest, archives, profiles)
    assert len(restored) == 20
    assert sum(row["query_cursor"] for row in restored.values()) == 1000


def test_missing_member_and_duplicate_shard_fail_closed(
    protocol: dict[str, object],
) -> None:
    topology, _profiles, manifest, archives = _set_fixture(protocol)
    missing = copy.deepcopy(archives)
    missing.pop(3)
    with pytest.raises(BackupSetError, match="set_member_missing"):
        verify_backup_set(
            topology, manifest, missing, synthetic_profiles(topology)
        )
    duplicated = copy.deepcopy(archives)
    duplicated[1][0] = copy.deepcopy(duplicated[0][0])
    with pytest.raises(BackupSetError, match="member_archive_inventory_invalid"):
        verify_backup_set(
            topology, manifest, duplicated, synthetic_profiles(topology)
        )


def test_member_replacement_and_cross_set_mix_fail_closed(
    protocol: dict[str, object],
) -> None:
    topology, profiles, manifest, archives = _set_fixture(protocol)
    replaced_profiles = copy.deepcopy(profiles)
    replaced_profiles[0]["member_identity"] = stable_hash({"replacement": True})
    with pytest.raises(BackupSetError, match="member_identity_replaced"):
        verify_profiles(topology, replaced_profiles)
    other_topology, _other_profiles, other_manifest, _other_archives = _set_fixture(
        protocol, 3
    )
    mixed = copy.deepcopy(manifest)
    mixed["member_manifests"][0] = copy.deepcopy(
        other_manifest["member_manifests"][0]
    )
    with pytest.raises(BackupSetError, match="set_manifest_integrity_invalid"):
        verify_backup_set(topology, mixed, archives, profiles)
    assert other_topology["topology_sha256"] != topology["topology_sha256"]


def test_old_member_rollback_and_index_conflict_fail_closed(
    protocol: dict[str, object],
) -> None:
    topology, profiles, old_manifest, old_archives = _set_fixture(protocol)
    states = {
        shard: build_shard_state(
            shard,
            cursor=50,
            attempt=1 if shard == 7 else 0,
            generation=2 if shard == 7 else 1,
        )
        for shard in range(20)
    }
    new_manifest, new_archives = build_backup_set(
        topology,
        states,
        profiles,
        sequence=1,
        parent_set_root_sha256=old_manifest["set_root_sha256"],
    )
    mixed = copy.deepcopy(new_archives)
    owner = 7 % 4
    mixed[owner] = copy.deepcopy(old_archives[owner])
    with pytest.raises(BackupSetError, match="member_inventory_hash_mismatch"):
        verify_backup_set(topology, new_manifest, mixed, profiles)
    conflicted = copy.deepcopy(new_manifest)
    conflicted["member_manifests"][owner]["inventory"][0]["generation"] = 99
    with pytest.raises(BackupSetError, match="set_manifest_integrity_invalid"):
        verify_backup_set(topology, conflicted, new_archives, profiles)


def test_double_writer_fails_closed(protocol: dict[str, object]) -> None:
    topology, _profiles, manifest, archives = _set_fixture(protocol)
    tampered = copy.deepcopy(manifest)
    tampered["member_manifests"][0]["writer_lock_holders"] = 2
    tampered["member_manifests"][0]["member_root_sha256"] = stable_hash(
        {
            key: value
            for key, value in tampered["member_manifests"][0].items()
            if key != "member_root_sha256"
        }
    )
    tampered["member_roots"][0]["member_root_sha256"] = tampered[
        "member_manifests"
    ][0]["member_root_sha256"]
    tampered["set_root_sha256"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "set_root_sha256"}
    )
    with pytest.raises(BackupSetError, match="member_manifest_integrity_invalid"):
        verify_backup_set(
            topology, tampered, archives, synthetic_profiles(topology)
        )


@pytest.mark.parametrize("count", [2, 3, 4])
def test_1000_query_simulation_has_zero_duplicate_requests(
    protocol: dict[str, object], count: int
) -> None:
    report = simulate_set(protocol, member_count=count)
    assert report["query_count"] == 1000
    assert report["adapter_call_count"] == 1000
    assert report["duplicate_request_count"] == 0
    assert report["resource_ledger_conserved"] is True
    assert report["aggregate_matches_single_target"] is True
    assert report["scenario_count"] == 10


def test_all_supported_sets_verify_and_real_readiness_stays_blocked(
    protocol: dict[str, object],
) -> None:
    assert verify_set(protocol)["verified_member_counts"] == [2, 3, 4]
    readiness = audit_readiness(protocol)
    assert readiness["exit_code"] == EXIT_NOT_READY
    assert readiness["qualified_real_member_count"] == 0
    assert readiness["full1000_blocker_cleared"] is False
    assert readiness["single_target_compatibility_preserved"] is True
    assert len(readiness["missing_real_member_fields"]["4"]) == 24


def test_reports_are_byte_deterministic() -> None:
    for args in (
        ("build-topology", "--members", "3"),
        ("calculate-capacity",),
        ("verify-set",),
        ("simulate-set", "--members", "4"),
        ("audit-readiness",),
    ):
        first = _run_cli(*args)
        second = _run_cli(*args)
        assert first.returncode == second.returncode
        assert first.stdout == second.stdout
        assert first.stderr == second.stderr == b""


@pytest.mark.parametrize(
    ("args", "exit_code"),
    [
        (("build-topology",), EXIT_READY),
        (("calculate-capacity",), EXIT_READY),
        (("verify-set",), EXIT_READY),
        (("simulate-set",), EXIT_READY),
        (("audit-readiness",), EXIT_NOT_READY),
    ],
)
def test_cli_contract(args: tuple[str, ...], exit_code: int) -> None:
    result = _run_cli(*args)
    assert result.returncode == exit_code
    assert result.stderr == b""
    assert json.loads(result.stdout)["exit_code"] == exit_code


def test_cli_rejects_bad_protocol_without_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"protocol":"formal_backup_set_topology_v1"}', encoding="utf-8")
    result = _run_cli("--protocol", str(bad), "calculate-capacity")
    assert result.returncode == EXIT_VIOLATION
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"protocol":"a","protocol":"b"}', encoding="utf-8")
    result = _run_cli("--protocol", str(duplicate), "calculate-capacity")
    assert result.returncode == EXIT_VIOLATION
    assert result.stderr == b""


def test_cli_usage_error_is_stable() -> None:
    result = _run_cli()
    assert result.returncode == EXIT_USAGE
    assert result.stderr == b""
    assert json.loads(result.stdout)["status"] == "usage_error"


def test_capacity_model_digest_is_stable() -> None:
    assert stable_hash(all_capacity_models()) == (
        "569ceee71493d22b523d40bda7e9d4994e69477a2a6c0993a40dcb4225f33837"
    )


def test_readiness_freshness_and_public_contract_are_integrated() -> None:
    readiness = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text(
            encoding="utf-8"
        )
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"] == "architecture_formal_backup_set_topology_ready"
    )
    assert claim["status"] == "verified"
    assert set(claim["evidence_ids"]) == {
        "formal_backup_set_capacity",
        "formal_backup_set_protocol",
        "formal_backup_set_readiness",
        "formal_backup_set_simulation",
    }
    gate = next(
        row
        for row in readiness["read_only_gates"]
        if row["gate_id"] == "formal_backup_set_topology"
    )
    assert gate["expected_exit_code"] == EXIT_NOT_READY

    freshness = json.loads(
        (
            ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json"
        ).read_text(encoding="utf-8")
    )
    assert freshness["claim_component_bindings"][claim["claim_id"]] == [
        "formal_backup_set_topology"
    ]
    assert "formal_backup_set_readiness" in freshness["blocked_evidence_ids"]

    public = json.loads(
        (
            ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
        ).read_text(encoding="utf-8")
    )
    assert public["artifact_contracts"]["formal_backup_set_topology"] == (
        "benchmark/formal_backup_set_topology_v1_protocol.json"
    )
    assert public["cli_contracts"]["formal_backup_set_topology"]["exit_codes"] == [
        0,
        2,
        3,
        4,
    ]
