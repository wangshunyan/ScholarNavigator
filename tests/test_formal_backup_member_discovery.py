from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.formal_backup_member_discovery import (
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    NOT_AVAILABLE,
    BackupMemberDiscoveryError,
    audit_readiness,
    build_candidate,
    deduplicate_candidates,
    discover,
    load_protocol,
    match_topologies,
    path_binding,
    simulate_profiles,
    synthetic_candidate,
    validate_candidate,
)
from scholar_agent.evaluation.formal_backup_set_topology import capacity_model
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/formal_backup_member_discovery_v1_protocol.json"
CLI = ROOT / "scripts/check_formal_backup_member_discovery.py"


@pytest.fixture()
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _fixture_protocol(
    protocol: dict[str, object], paths: list[Path]
) -> dict[str, object]:
    value = copy.deepcopy(protocol)
    value["registered_targets"] = [
        {
            "alias": f"backup-target-{index}",
            "path_binding_sha256": path_binding(
                f"backup-target-{index}", path
            ),
        }
        for index, path in enumerate(paths)
    ]
    return value


def _candidate(
    protocol: dict[str, object],
    index: int,
    *,
    count: int = 4,
    quota: int | str | None = None,
) -> dict[str, object]:
    model = capacity_model(count)["members"][index]
    return synthetic_candidate(
        protocol,
        alias=f"backup-target-{index}",
        identity_seed=f"member-{index}",
        available_bytes=model["required_bytes"],
        available_inodes=model["required_inodes"],
        quota_bytes=model["required_bytes"] if quota is None else quota,
    )


def _rehydrate(
    protocol: dict[str, object],
    candidate: dict[str, object],
    **changes: object,
) -> dict[str, object]:
    target_id = candidate["target_id"]
    alias = next(
        row["alias"]
        for row in protocol["registered_targets"]
        if stable_hash({"registered_alias": row["alias"]}) == target_id
    )
    observation = copy.deepcopy(candidate["observations"])
    observation.update(changes)
    return build_candidate(
        protocol,
        {"target_alias": alias, **observation},
        synthetic_only=True,
    )


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


def test_current_registry_has_no_candidates_and_is_not_ready(
    protocol: dict[str, object],
) -> None:
    assert protocol["registered_targets"] == []
    first = audit_readiness(protocol)
    second = audit_readiness(protocol)
    assert first == second
    assert first["exit_code"] == EXIT_NOT_READY
    assert first["observed_candidate_count"] == 0
    assert [row["missing_candidate_count"] for row in first["topology_match"]["plans"]] == [
        2,
        3,
        4,
    ]


def test_discovery_only_touches_registered_exact_paths(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = tmp_path / "target"
    sibling = tmp_path / "not-registered"
    target.mkdir()
    sibling.mkdir()
    fixture = _fixture_protocol(protocol, [target])
    observed: list[Path] = []

    def observer(alias: str, path: Path) -> dict[str, object]:
        observed.append(path)
        candidate = _candidate(fixture, 0, count=2)
        observations = copy.deepcopy(candidate["observations"])
        observations["synthetic_only"] = False
        return {"target_alias": alias, **observations}

    report = discover(
        fixture,
        {"backup-target-0": target},
        observer=observer,
    )
    assert observed == [target]
    assert report["exit_code"] == EXIT_READY
    serialized = json.dumps(report)
    assert str(target) not in serialized
    assert str(sibling) not in serialized
    assert report["candidates"][0]["status"] == "candidate"
    assert report["candidates"][0]["auto_registered"] is False
    assert report["candidates"][0]["auto_activated"] is False


def test_unregistered_target_and_path_rebinding_fail(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    target = tmp_path / "target"
    replacement = tmp_path / "replacement"
    target.mkdir()
    replacement.mkdir()
    fixture = _fixture_protocol(protocol, [target])
    with pytest.raises(
        BackupMemberDiscoveryError, match="unregistered_target_requested"
    ):
        discover(fixture, {"backup-target-other": target})
    with pytest.raises(
        BackupMemberDiscoveryError, match="registered_path_binding_mismatch"
    ):
        discover(fixture, {"backup-target-0": replacement})


def test_four_distinct_complete_candidates_match_topology(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    fixture = _fixture_protocol(
        protocol, [tmp_path / f"target-{index}" for index in range(4)]
    )
    candidates = [_candidate(fixture, index) for index in range(4)]
    match = match_topologies(fixture, candidates)
    assert match["deduplicated_candidate_count"] == 4
    assert match["plans"][2]["status"] == "candidate_topology_match"
    assert len(match["plans"][2]["assignment"]) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quota_bytes", NOT_AVAILABLE),
        ("available_bytes", 0),
        ("available_inodes", 0),
        ("failure_domain_identity", NOT_AVAILABLE),
        ("target_present", False),
    ],
)
def test_missing_or_insufficient_evidence_stays_candidate_only(
    protocol: dict[str, object],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _fixture_protocol(
        protocol, [tmp_path / f"target-{index}" for index in range(4)]
    )
    candidates = [_candidate(fixture, index) for index in range(4)]
    candidates[0] = _rehydrate(fixture, candidates[0], **{field: value})
    assert candidates[0]["candidate_complete"] is False or value == 0
    assert (
        match_topologies(fixture, candidates)["plans"][2]["status"]
        == "not_ready_missing_qualified_candidates"
    )


@pytest.mark.parametrize(
    "field",
    [
        "device_identity",
        "filesystem_identity",
        "quota_pool_identity",
        "failure_domain_identity",
        "management_domain_identity",
    ],
)
def test_aliases_and_shared_capacity_domains_are_deduplicated(
    protocol: dict[str, object], tmp_path: Path, field: str
) -> None:
    fixture = _fixture_protocol(
        protocol, [tmp_path / f"target-{index}" for index in range(4)]
    )
    candidates = [_candidate(fixture, index) for index in range(4)]
    candidates[1] = _rehydrate(
        fixture,
        candidates[1],
        **{field: candidates[0]["observations"][field]},
    )
    groups = deduplicate_candidates(fixture, candidates)
    assert len(groups) == 3
    assert sorted(row["alias_count"] for row in groups) == [1, 1, 2]
    assert (
        match_topologies(fixture, candidates)["plans"][2]["status"]
        == "not_ready_missing_qualified_candidates"
    )


def test_candidate_identity_replacement_is_detected(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    fixture = _fixture_protocol(protocol, [tmp_path / "target"])
    candidate = _candidate(fixture, 0, count=2)
    candidate["target_identity"] = "0" * 64
    candidate["candidate_sha256"] = stable_hash(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_sha256"
        }
    )
    with pytest.raises(
        BackupMemberDiscoveryError, match="candidate_semantic_drift"
    ):
        validate_candidate(fixture, candidate)


def test_simulation_matrix_is_complete_and_deterministic(
    protocol: dict[str, object],
) -> None:
    first = simulate_profiles(protocol)
    second = simulate_profiles(protocol)
    assert first == second
    assert first["exit_code"] == EXIT_READY
    assert first["passed_count"] == first["scenario_count"] == 9
    assert first["scenarios"]["alias_duplicate"]["deduplicated_candidate_count"] == 3


def test_cli_exit_codes_and_stable_json() -> None:
    first = _run_cli("audit-readiness")
    second = _run_cli("audit-readiness")
    assert first.returncode == second.returncode == EXIT_NOT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert json.loads(first.stdout)["reason_code"] == (
        "no_protocol_registered_real_targets"
    )
    simulation = _run_cli("simulate-profiles")
    assert simulation.returncode == EXIT_READY
    assert json.loads(simulation.stdout)["passed_count"] == 9
    usage = _run_cli("discover", "--target", "malformed")
    assert usage.returncode == EXIT_USAGE
    missing = _run_cli("verify-candidate", "--candidate", "missing.json")
    assert missing.returncode == EXIT_VIOLATION
    assert missing.stderr == b""


def test_protocol_drift_fails_closed(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    bad = copy.deepcopy(protocol)
    bad["policy"]["candidate_is_qualified_member"] = True
    bad["protocol_sha256"] = stable_hash(
        {key: value for key, value in bad.items() if key != "protocol_sha256"}
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BackupMemberDiscoveryError, match="protocol_policy_invalid"):
        load_protocol(path, repository_root=ROOT)
