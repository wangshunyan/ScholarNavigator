from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_execution_host_attestation import (
    EXIT_NOT_READY,
    EXIT_QUALIFIED,
    HostAttestationError,
    HostAttestationNotReady,
    _qualified_profile,
    bind_launch_authorization,
    build_launch_addendum,
    canonical_json,
    load_protocol,
    probe_host,
    simulate_profiles,
    validate_attestation,
    validate_attestation_freshness,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/formal_execution_host_attestation_v1_protocol.json"
)
CLI = ROOT / "scripts/check_formal_execution_host.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value.pop("attestation_sha256", None)
    value["attestation_sha256"] = stable_hash(value)
    return value


def _cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent)
    environment["PYTHONPATH"] = "src"
    return environment


def test_protocol_binds_full1000_storage_and_runtime_contracts(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == "e47c5393cdd9f67d38ab8e38749664ca0a3310a1"
    assert protocol["storage_requirements"] == {
        "primary": {"bytes": 713_501_442_048, "inodes": 76_980},
        "backup": {"bytes": 2_119_029_489_664, "inodes": 210_940},
    }
    assert set(protocol["bindings"]) == {
        "crash_consistency",
        "disaster_recovery",
        "execution_plan",
        "launch_control",
        "runtime_hermeticity",
        "storage_governance",
        "storage_plan",
    }
    assert protocol["host_identity"]["authentication_provided"] is False


def test_real_probe_uses_tiny_same_filesystem_fixture_and_is_redacted(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    value = probe_host(
        ROOT,
        protocol,
        primary_root=tmp_path,
        backup_root=None,
        observed_head=protocol["source_commit"],
        host_scope_identity=stable_hash({"fixture": "real-probe"}),
    )
    assert value["status"] == "not_ready_unverified_or_insufficient_host"
    assert value["backup_target_identity"] == "not_available"
    assert value["capabilities"]["atomic_replace"]["status"] == "passed"
    expected_directory_fsync = "not_available" if os.name == "nt" else "passed"
    assert value["capabilities"]["directory_fsync"]["status"] == expected_directory_fsync
    assert value["capabilities"]["advisory_lock"]["status"] == "passed"
    assert value["capabilities"]["nonempty_restore_rejection"]["status"] == "passed"
    assert not any(tmp_path.iterdir())
    encoded = canonical_json(value)
    assert str(tmp_path).encode() not in encoded
    assert b".env" not in encoded


def test_synthetic_profile_matrix_is_complete_and_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_profiles(protocol)
    second = simulate_profiles(protocol)
    assert first["exit_code"] == EXIT_QUALIFIED
    assert first["scenario_count"] == 10
    assert canonical_json(first) == canonical_json(second)
    assert all(row["status"] == "passed" for row in first["scenarios"].values())
    assert first["scenarios"]["fully_qualified"]["observed_host_status"] == "host_qualified"


@pytest.mark.parametrize(
    ("capability", "reason"),
    [
        ("primary_available_bytes", "primary_available_bytes_below_required"),
        ("primary_available_inodes", "primary_available_inodes_below_required"),
        ("directory_fsync", "directory_fsync_failed"),
        ("atomic_replace", "same_filesystem_atomic_replace_failed"),
        ("advisory_lock", "advisory_lock_unavailable"),
        ("nofile_soft_limit", "nofile_soft_limit_below_required"),
        ("independent_backup_fault_domain", "primary_backup_share_fault_domain"),
    ],
)
def test_insufficient_host_capability_cannot_be_qualified(
    protocol: dict[str, object], capability: str, reason: str
) -> None:
    value = _qualified_profile(protocol)
    value["capabilities"][capability] = {
        "status": "failed",
        "reason_code": reason,
        "observed": None,
        "required": None,
    }
    value["failed_capabilities"] = [capability]
    value["status"] = "not_ready_unverified_or_insufficient_host"
    _rehash(value)
    validated = validate_attestation(value, protocol)
    assert validated.status == "not_ready_unverified_or_insufficient_host"
    with pytest.raises(HostAttestationNotReady, match="not_qualified"):
        validate_attestation(value, protocol, require_qualified=True)


def test_missing_observation_is_not_inferred(
    protocol: dict[str, object]
) -> None:
    value = _qualified_profile(protocol)
    capability = "backup_filesystem_quota_bytes"
    value["capabilities"][capability] = {
        "status": "not_available",
        "reason_code": "backup_filesystem_quota_bytes_not_available",
        "observed": None,
        "required": 2_119_029_489_664,
    }
    value["missing_observations"] = [capability]
    value["status"] = "not_ready_unverified_or_insufficient_host"
    _rehash(value)
    validated = validate_attestation(value, protocol)
    assert validated.capabilities[capability].status == "not_available"


def test_attestation_tamper_and_protocol_binding_drift_fail_closed(
    protocol: dict[str, object],
) -> None:
    value = _qualified_profile(protocol)
    value["runtime"]["architecture"] = "tampered"
    with pytest.raises(HostAttestationError, match="schema_or_digest"):
        validate_attestation(value, protocol)

    changed = _qualified_profile(protocol)
    changed["binding_sha256s"]["storage_plan"] = "0" * 64
    _rehash(changed)
    with pytest.raises(HostAttestationError, match="bound_input"):
        validate_attestation(changed, protocol)

    missing = _qualified_profile(protocol)
    missing["capabilities"].pop("directory_fsync")
    _rehash(missing)
    with pytest.raises(HostAttestationError, match="capability_inventory"):
        validate_attestation(missing, protocol)


def test_capacity_or_resource_observation_change_invalidates_seal(
    protocol: dict[str, object],
) -> None:
    sealed = _qualified_profile(protocol)
    changed = copy.deepcopy(sealed)
    changed["capabilities"]["nofile_soft_limit"]["observed"] = 512
    _rehash(changed)
    with pytest.raises(HostAttestationError, match="observation_drift"):
        validate_attestation_freshness(sealed, changed, protocol)


def test_launch_binding_rejects_cross_host_commit_and_storage_target(
    protocol: dict[str, object],
) -> None:
    value = _qualified_profile(protocol)
    authorization = {"authorization_sha256": stable_hash({"authorization": 1})}
    expected = {
        "current_head": value["observed_head"],
        "host_scope_identity": value["host_scope_identity"],
        "authoritative_output_root_identity": value[
            "authoritative_output_root_identity"
        ],
        "primary_target_identity": value["primary_target_identity"],
        "backup_target_identity": value["backup_target_identity"],
        "current_probe_sha256": value["attestation_sha256"],
    }
    bound = bind_launch_authorization(authorization, value, protocol, **expected)
    assert bound["host_attestation_sha256"] == value["attestation_sha256"]
    assert bound["formal_validation_complete"] is False

    for key in expected:
        changed = dict(expected)
        changed[key] = stable_hash({"drift": key})
        with pytest.raises(HostAttestationError, match="drift"):
            bind_launch_authorization(authorization, value, protocol, **changed)


def test_launch_addendum_requires_fresh_qualified_attestation(
    protocol: dict[str, object],
) -> None:
    addendum = build_launch_addendum(protocol)
    requirements = addendum["launch_authorization_requirements"]
    assert requirements["attestation_status"] == "host_qualified"
    assert requirements["attestation_fresh_for_exact_head"] is True
    assert requirements["legacy_authorization_reusable"] is False


def test_cli_current_readiness_and_simulation_are_stable() -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=ROOT,
            env=_cli_environment(),
            capture_output=True,
            check=False,
        )

    first = run("simulate-profile")
    second = run("simulate-profile")
    assert first.returncode == second.returncode == EXIT_QUALIFIED
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""

    readiness = run("audit-readiness")
    assert readiness.returncode == EXIT_NOT_READY
    assert readiness.stderr == b""
    assert (
        json.loads(readiness.stdout)["status"]
        == "not_ready_unverified_or_insufficient_host"
    )


def test_cli_malformed_attestation_has_no_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "attestation.json"
    malformed.write_text('{"contract":"host_attestation_manifest_v1"}\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "verify-attestation",
            "--attestation",
            str(malformed),
        ],
        cwd=ROOT,
        env=_cli_environment(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout


def test_public_contract_readiness_and_freshness_registration() -> None:
    public = json.loads(
        (ROOT / "benchmark/public_contract_compatibility_v1_protocol.json").read_text()
    )
    assert "formal_execution_host_attestation" in public["artifact_contracts"]
    assert "formal_execution_host_attestation" in public["cli_contracts"]

    readiness = json.loads(
        (ROOT / "benchmark/validation_readiness_bundle_v1_contract.json").read_text()
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"] == "architecture_formal_execution_host_controls_ready"
    )
    assert claim["status"] == "verified"
    assert "does not qualify" in claim["boundary"]
    assert {
        row["blocker_id"] for row in readiness["blockers"]
    } == {
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    }

    addenda = json.loads(
        (ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json").read_text()
    )
    assert addenda["claim_component_bindings"][
        "architecture_formal_execution_host_controls_ready"
    ] == ["formal_execution_host_attestation"]
