from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.external_scorer_handoff import (
    ExternalScorerError,
    audit_real_readiness,
    canonical_handoff,
    create_package_manifest,
    load_protocol,
    run_scorer,
    run_synthetic_matrix,
    stable_json_bytes,
    synthetic_handoff,
    synthetic_scorer_source,
    validate_handoff,
    validate_output,
    verify_package,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark" / "external_scorer_handoff_v1_protocol.json"


@pytest.fixture
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _package(tmp_path: Path, scenario: str = "valid") -> Path:
    package = tmp_path / f"package-{scenario}"
    create_package_manifest(
        package,
        scorer_name="synthetic-strict-scorer",
        scorer_version="1",
        entrypoint_source=synthetic_scorer_source(scenario),
    )
    return package


def _handoff_file(tmp_path: Path) -> Path:
    path = tmp_path / "handoff.json"
    path.write_bytes(stable_json_bytes(synthetic_handoff()))
    return path


def test_canonical_handoff_binds_lineage_order_and_top20(protocol: dict[str, object]) -> None:
    value = synthetic_handoff()
    validate_handoff(value, protocol["resource_limits"])
    assert value["query_count"] == 3
    assert [row["query_order"] for row in value["queries"]] == [0, 1, 2]
    assert len(value["lineage"]["run_manifest_sha256"]) == 64

    broken = json.loads(json.dumps(value))
    broken["queries"][0]["results"][0]["rank"] = 2
    with pytest.raises(ExternalScorerError, match="hash_mismatch|rank_invalid"):
        validate_handoff(broken, protocol["resource_limits"])


def test_duplicate_and_missing_query_identity_are_rejected() -> None:
    base = synthetic_handoff()["queries"]
    duplicate = json.loads(json.dumps(base))
    duplicate[1]["query_identity"] = duplicate[0]["query_identity"]
    with pytest.raises(ExternalScorerError, match="identity_or_order"):
        canonical_handoff(
            duplicate,
            run_manifest_sha256="1" * 64,
            commit_generation_sha256="2" * 64,
        )
    missing = json.loads(json.dumps(base))
    del missing[0]["query_identity"]
    with pytest.raises(ExternalScorerError, match="schema"):
        canonical_handoff(
            missing,
            run_manifest_sha256="1" * 64,
            commit_generation_sha256="2" * 64,
        )


def test_package_hash_and_cross_version_mixing_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    package = _package(tmp_path)
    verify_package(package, protocol)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["manifest_version"] = "scorer_package_manifest_v2"
    (package / "manifest.json").write_bytes(stable_json_bytes(manifest))
    with pytest.raises(ExternalScorerError, match="manifest_schema"):
        verify_package(package, protocol)


def test_valid_scorer_is_deterministic_and_input_is_immutable(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    package = _package(tmp_path)
    handoff = _handoff_file(tmp_path)
    before = handoff.read_bytes()
    first = run_scorer(package, handoff, protocol, repository_root=ROOT, run_ordinal=1)
    second = run_scorer(package, handoff, protocol, repository_root=ROOT, run_ordinal=2)
    assert first["output_bytes"] == second["output_bytes"]
    assert handoff.read_bytes() == before
    assert first["worker_audit"] == {
        "blocked_file_operations": 0,
        "blocked_network_operations": 0,
        "blocked_subprocess_operations": 0,
        "input_mutation_count": 0,
    }


def test_output_validation_rejects_nonfinite_unknown_and_partial(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    package = _package(tmp_path)
    manifest = verify_package(package, protocol)
    handoff = synthetic_handoff()
    valid = {
        "schema_version": "synthetic_scorer_output_v1",
        "scorer_name": "synthetic-strict-scorer",
        "scorer_version": "1",
        "metric_namespace": "synthetic_handoff",
        "query_results": [
            {
                "query_identity": query["query_identity"],
                "values": {"synthetic_handoff.result_count": len(query["results"])},
            }
            for query in handoff["queries"]
        ],
    }
    validate_output(valid, handoff, manifest)
    nonfinite = json.loads(json.dumps(valid))
    nonfinite["query_results"][0]["values"]["synthetic_handoff.result_count"] = float("nan")
    with pytest.raises(ExternalScorerError, match="non_finite"):
        validate_output(nonfinite, handoff, manifest)
    unknown = json.loads(json.dumps(valid))
    unknown["query_results"][0]["values"]["forged.metric"] = 1
    with pytest.raises(ExternalScorerError, match="unknown_metric"):
        validate_output(unknown, handoff, manifest)
    with pytest.raises(ExternalScorerError, match="coverage"):
        validate_output({**valid, "query_results": valid["query_results"][:-1]}, handoff, manifest)


def test_full_synthetic_matrix_closes_all_scenarios(
    protocol: dict[str, object]
) -> None:
    report = run_synthetic_matrix(protocol, repository_root=ROOT)
    assert report["status"] == "handoff_chain_verified"
    assert report["scenario_count"] == 15
    assert report["scenarios"][0]["scenario"] == "valid"
    assert report["scenarios"][0]["observed"] == "passed"
    assert all(row["observed"] == "rejected" for row in report["scenarios"][1:])
    assert report["execution"]["network_request_count"] == 0
    assert report["execution"]["quality_metric_count"] == 0


def test_matrix_report_is_byte_deterministic(protocol: dict[str, object]) -> None:
    first = run_synthetic_matrix(protocol, repository_root=ROOT)
    second = run_synthetic_matrix(protocol, repository_root=ROOT)
    assert stable_json_bytes(first) == stable_json_bytes(second)


def test_real_readiness_remains_blocked(protocol: dict[str, object]) -> None:
    report = audit_real_readiness(protocol, repository_root=ROOT)
    assert report["exit_code"] == 3
    assert report["formal_validation_complete"] is False
    assert report["official_score_generated"] is False
    assert "official_scorer_package" in report["blocked_reasons"]
    assert "complete_full1000_authoritative_input" in report["blocked_reasons"]


def test_cli_blocked_readiness_and_usage_are_stable(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "check_external_scorer_handoff.py"
    first = subprocess.run(
        [sys.executable, str(script), "audit-readiness"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, str(script), "audit-readiness"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert first.returncode == second.returncode == 3
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["status"] == "blocked_missing_official_scorer_or_complete_input"

    usage = subprocess.run(
        [sys.executable, str(script), "run"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert usage.returncode == 4
    assert json.loads(usage.stdout)["status"] == "usage_error"
