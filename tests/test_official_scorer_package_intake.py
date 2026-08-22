from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.official_scorer_package_intake import (
    OfficialScorerIntakeError,
    audit_readiness,
    build_contract,
    build_kit,
    build_synthetic_package,
    canonical_json,
    conformance_dry_run,
    import_dry_run,
    load_protocol,
    package_template,
    sha256_bytes,
    simulate_matrix,
    verify_candidate_package,
    verify_kit,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/official_scorer_package_intake_v1_protocol.json"


@pytest.fixture
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _kit(tmp_path: Path, protocol: dict[str, object]) -> tuple[Path, dict[str, object]]:
    kit = tmp_path / "kit.zip"
    challenge = sha256_bytes(b"test-official-scorer-challenge")
    build_kit(ROOT, protocol, challenge=challenge, output=kit)
    return kit, verify_kit(kit, protocol, repository_root=ROOT)


def _package(
    tmp_path: Path,
    contract: dict[str, object],
    scenario: str = "qualified",
) -> Path:
    package = tmp_path / f"{scenario}.zip"
    build_synthetic_package(contract, package, scenario=scenario)
    return package


def _repack(path: Path, mutate) -> None:
    with zipfile.ZipFile(path) as archive:
        files = {info.filename: archive.read(info) for info in archive.infolist()}
    mutate(files)
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)


def test_protocol_and_real_readiness_remain_fail_closed(
    protocol: dict[str, object],
) -> None:
    report = audit_readiness(protocol)
    assert report["exit_code"] == 3
    assert report["formal_validation_complete"] is False
    assert report["official_score_generated"] is False
    assert report["blocked_reasons"] == [
        "official_input_schema_missing",
        "official_metric_namespace_missing",
        "official_output_schema_missing",
        "official_package_missing",
        "verified_official_origin_missing",
    ]
    assert report["formal_blockers"] == [
        "full1000_incomplete",
        "human_precision_missing",
        "official_scorer_schema_missing",
    ]


def test_kit_is_deterministic_and_runs_without_repository(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    challenge = sha256_bytes(b"isolated-kit")
    build_kit(ROOT, protocol, challenge=challenge, output=first)
    build_kit(ROOT, protocol, challenge=challenge, output=second)
    assert first.read_bytes() == second.read_bytes()

    for ordinal in (1, 2):
        isolated = tmp_path / f"isolated-{ordinal}"
        isolated.mkdir()
        with zipfile.ZipFile(first) as archive:
            archive.extract("verify.py", isolated)
        env = {
            "HOME": str(isolated / "home"),
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": str(ordinal),
            "TMPDIR": str(isolated),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(isolated / "verify.py"),
                "--kit",
                str(first),
            ],
            cwd=isolated,
            env=env,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        assert json.loads(completed.stdout)["status"] == (
            "official_scorer_intake_kit_verified"
        )


def test_candidate_package_and_conformance_are_strict_and_deterministic(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit, contract = _kit(tmp_path, protocol)
    package = _package(tmp_path, contract)
    manifest, files = verify_candidate_package(
        kit,
        package,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    assert manifest["origin_status"] == "unverified_origin"
    assert set(files) == {"intake_manifest.json", "scorer.py"}
    first = conformance_dry_run(
        kit,
        package,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    second = conformance_dry_run(
        kit,
        package,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["worker_audit"] == {
        "blocked_file_operations": 0,
        "blocked_network_operations": 0,
        "blocked_subprocess_operations": 0,
        "input_mutation_count": 0,
    }
    assert first["official_score_generated"] is False


def test_synthetic_matrix_closes_all_intake_and_sandbox_attacks(
    protocol: dict[str, object],
) -> None:
    first = simulate_matrix(ROOT, protocol)
    second = simulate_matrix(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == first["passed_count"] == 11
    assert first["scenarios"][0]["scenario"] == "qualified"
    assert first["scenarios"][0]["observed"] == "passed"
    assert all(row["observed"] == "rejected" for row in first["scenarios"][1:])


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("entrypoint_tamper", "package_inventory_invalid"),
        ("cross_version_mixing", "package_manifest_binding_invalid"),
        ("revoked_reuse", "package_revoked"),
    ],
)
def test_package_schema_version_hash_and_lifecycle_fail_closed(
    tmp_path: Path,
    protocol: dict[str, object],
    scenario: str,
    reason: str,
) -> None:
    kit, contract = _kit(tmp_path, protocol)
    package = _package(tmp_path, contract, scenario)
    with pytest.raises(OfficialScorerIntakeError, match=reason):
        verify_candidate_package(
            kit,
            package,
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_archive_traversal_duplicate_and_schema_duplicate_keys_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit, contract = _kit(tmp_path, protocol)
    package = _package(tmp_path, contract)
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../scorer.py", b"x")
        archive.writestr("intake_manifest.json", b"{}")
    with pytest.raises(OfficialScorerIntakeError, match="unsafe_package_path"):
        verify_candidate_package(
            kit, traversal, protocol, repository_root=ROOT, allow_synthetic=True
        )

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("scorer.py", b"x")
        archive.writestr("scorer.py", b"y")
    with pytest.raises(OfficialScorerIntakeError, match="archive_member_unsafe"):
        verify_candidate_package(
            kit, duplicate, protocol, repository_root=ROOT, allow_synthetic=True
        )

    def duplicate_key(files: dict[str, bytes]) -> None:
        files["intake_manifest.json"] = b'{"package_protocol":"a","package_protocol":"b"}'

    _repack(package, duplicate_key)
    with pytest.raises(OfficialScorerIntakeError, match="package_manifest_invalid"):
        verify_candidate_package(
            kit, package, protocol, repository_root=ROOT, allow_synthetic=True
        )


def test_unknown_material_and_origin_cannot_clear_official_blocker(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit, contract = _kit(tmp_path, protocol)
    unknown = _package(tmp_path, contract, "unverified_origin")
    manifest, _ = verify_candidate_package(
        kit,
        unknown,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    assert manifest["origin_status"] == "unverified_origin"
    with pytest.raises(OfficialScorerIntakeError, match="synthetic_package_not_real"):
        verify_candidate_package(
            kit, unknown, protocol, repository_root=ROOT, allow_synthetic=False
        )


def test_challenge_is_single_use_and_cross_challenge_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit, contract = _kit(tmp_path, protocol)
    package = _package(tmp_path, contract)
    ledger = tmp_path / "ledger.json"
    import_dry_run(
        kit,
        package,
        ledger,
        protocol,
        repository_root=ROOT,
        allow_synthetic=True,
    )
    with pytest.raises(OfficialScorerIntakeError, match="challenge_replay"):
        import_dry_run(
            kit,
            package,
            ledger,
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )

    other_contract = build_contract(
        protocol, challenge=sha256_bytes(b"other-challenge")
    )
    other_kit = tmp_path / "other-kit.zip"
    build_kit(
        ROOT,
        protocol,
        challenge=other_contract["challenge"],
        output=other_kit,
    )
    with pytest.raises(OfficialScorerIntakeError, match="binding"):
        verify_candidate_package(
            other_kit,
            package,
            protocol,
            repository_root=ROOT,
            allow_synthetic=True,
        )


def test_cli_exit_codes_stderr_and_readiness_are_deterministic(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts/check_official_scorer_intake.py"
    commands = [
        (["audit-readiness"], 3),
        (["verify-package"], 4),
        (
            [
                "verify-package",
                "--kit",
                str(tmp_path / "missing-kit"),
                "--package",
                str(tmp_path / "missing-package"),
            ],
            2,
        ),
    ]
    for arguments, expected in commands:
        first = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        second = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert first.returncode == second.returncode == expected
        assert first.stdout == second.stdout
        assert first.stderr == second.stderr == b""
        assert json.loads(first.stdout)["exit_code"] == expected


def test_kit_contains_no_sensitive_or_machine_specific_material(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit, _contract = _kit(tmp_path, protocol)
    raw = kit.read_bytes()
    assert b".env" not in raw
    assert str(ROOT).encode() not in raw
    home = os.environ.get("HOME")
    if home:
        assert home.encode() not in raw
    template = package_template(
        build_contract(protocol, challenge=sha256_bytes(b"privacy"))
    )
    assert template["metrics"] == "not_provided"
    assert template["origin_evidence_type"] == "unverified_origin"
