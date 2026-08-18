from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_compaction import (
    ACTIVE_SHARD_WINDOW,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    NEW_BACKUP_REQUIRED_BYTES,
    NEW_BACKUP_REQUIRED_INODES,
    OLD_BACKUP_REQUIRED_BYTES,
    BackupCompactionError,
    ContentAddressedBackup,
    audit_readiness,
    build_shard_state,
    calculate_capacity,
    load_protocol,
    simulate_compaction,
    verify_recovery,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/formal_backup_compaction_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_compaction.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _complete_store() -> tuple[ContentAddressedBackup, str]:
    store = ContentAddressedBackup()
    states = [build_shard_state(shard, cursor=50) for shard in range(20)]
    root = store.build_root(
        kind="baseline", states=states, parent_root_sha256=None
    )
    return store, store.publish(root)


def test_capacity_is_auditable_worst_case_without_speculative_credits(
    protocol: dict[str, object],
) -> None:
    report = calculate_capacity(protocol)
    capacity = report["capacity"]
    assert capacity["old_backup_required_bytes"] == OLD_BACKUP_REQUIRED_BYTES
    assert capacity["new_backup_required_bytes"] == NEW_BACKUP_REQUIRED_BYTES
    assert capacity["new_backup_required_inodes"] == NEW_BACKUP_REQUIRED_INODES
    assert NEW_BACKUP_REQUIRED_BYTES == 1_028_812_963_840
    assert ACTIVE_SHARD_WINDOW == 4
    for key in (
        "compression_credit_bytes",
        "future_cleanup_credit_bytes",
        "future_deduplication_credit_bytes",
        "sparse_file_credit_bytes",
    ):
        assert capacity[key] == 0


def test_incremental_and_compacted_roots_preserve_old_history() -> None:
    store, genesis = _complete_store()
    changed = build_shard_state(4, cursor=50, attempt=1, generation=2)
    incremental = store.build_root(
        kind="incremental",
        states=[changed],
        parent_root_sha256=genesis,
    )
    incremental_sha = store.publish(incremental)
    states = store.resolve_states(incremental_sha)
    compacted = store.build_root(
        kind="compacted_baseline",
        states=list(states.values()),
        parent_root_sha256=incremental_sha,
        supersedes_root_sha256=incremental_sha,
    )
    compacted_sha = store.publish(compacted)
    store.verify_root(genesis)
    store.verify_root(incremental_sha)
    store.verify_root(compacted_sha)
    assert store.roots[compacted_sha]["parent_root_sha256"] == incremental_sha
    assert store.roots[compacted_sha]["history_genesis_rebuilt"] is False


def test_interrupted_compaction_leaves_previous_root_active() -> None:
    store, active = _complete_store()
    states = store.resolve_states(active)
    candidate = store.build_root(
        kind="compacted_baseline",
        states=list(states.values()),
        parent_root_sha256=active,
        supersedes_root_sha256=active,
    )
    with pytest.raises(
        BackupCompactionError,
        match="compaction_interrupted_previous_root_active",
    ):
        store.publish(candidate, fault_before_publish=True)
    assert store.active_root_sha256 == active
    store.verify_root(active)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("index", "root_integrity_invalid"),
        ("state_digest", "root_state_digest_invalid"),
        ("blob", "root_blob_missing"),
        ("parent", "root_parent_missing"),
    ],
)
def test_recovery_rejects_tampered_index_state_blob_and_parent(
    mutation: str, reason: str
) -> None:
    store, genesis = _complete_store()
    changed = build_shard_state(0, cursor=50, attempt=1, generation=2)
    incremental = store.build_root(
        kind="incremental", states=[changed], parent_root_sha256=genesis
    )
    root_sha = store.publish(incremental)
    if mutation == "index":
        store.roots[root_sha]["index"][0]["generation"] = 99
    elif mutation == "state_digest":
        root = store.roots[root_sha]
        root["states"][0]["state_sha256"] = "f" * 64
        root["index"][0]["state_sha256"] = "f" * 64
        root["root_sha256"] = stable_hash(
            {key: value for key, value in root.items() if key != "root_sha256"}
        )
        root_sha = root["root_sha256"]
        store.roots[root_sha] = root
    elif mutation == "blob":
        blob = store.roots[root_sha]["states"][0]["files"][0]["sha256"]
        store.blobs.pop(blob)
    else:
        store.roots.pop(genesis)
    with pytest.raises(BackupCompactionError, match=reason):
        store.verify_root(root_sha)


def test_unique_blob_cannot_be_deleted() -> None:
    store, root_sha = _complete_store()
    blob = store.roots[root_sha]["states"][0]["files"][0]["sha256"]
    with pytest.raises(BackupCompactionError, match="unique_blob_delete_forbidden"):
        store.delete_blob(blob)


def test_full_1000_simulation_recovers_without_duplicate_calls(
    protocol: dict[str, object],
) -> None:
    simulation = simulate_compaction(protocol)
    assert simulation["query_count"] == 1000
    assert simulation["scenario_count"] == 12
    assert simulation["duplicate_request_count"] == 0
    assert simulation["resource_ledger_conserved"] is True
    assert simulation["aggregate_matches_uninterrupted"] is True
    assert simulation["history_root_rewritten"] is False
    assert set(simulation["scenarios"]) == {
        "aggregate_equivalence",
        "baseline_damage_fallback",
        "compaction_interruption",
        "index_tamper",
        "multiple_compactions",
        "normal_incremental",
        "old_root_verification",
        "parent_chain_missing",
        "restore_after_400",
        "single_shard_replacement",
        "unique_blob_delete",
        "window_4_continuous_archive",
    }


def test_recovery_and_real_readiness_are_separate(
    protocol: dict[str, object],
) -> None:
    assert verify_recovery(protocol)["exit_code"] == EXIT_READY
    readiness = audit_readiness(protocol)
    assert readiness["exit_code"] == EXIT_NOT_READY
    assert readiness["backup_failure_domain_independent"] == "not_available"
    assert readiness["full1000_blocker_cleared"] is False
    assert readiness["formal_validation_complete"] is False


def test_simulation_and_capacity_json_are_byte_deterministic() -> None:
    first = _run_cli("simulate-compaction")
    second = _run_cli("simulate-compaction")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    capacity_one = _run_cli("calculate-capacity")
    capacity_two = _run_cli("calculate-capacity")
    assert capacity_one.stdout == capacity_two.stdout


@pytest.mark.parametrize(
    ("command", "exit_code"),
    [
        ("build-policy", EXIT_READY),
        ("calculate-capacity", EXIT_READY),
        ("simulate-compaction", EXIT_READY),
        ("verify-recovery", EXIT_READY),
        ("audit-readiness", EXIT_NOT_READY),
    ],
)
def test_cli_contract(command: str, exit_code: int) -> None:
    result = _run_cli(command)
    assert result.returncode == exit_code
    assert result.stderr == b""
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == exit_code
    assert payload["formal_validation_complete"] is False


def test_cli_fail_closed_without_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"protocol":"formal_backup_compaction_v1"}', encoding="utf-8")
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
