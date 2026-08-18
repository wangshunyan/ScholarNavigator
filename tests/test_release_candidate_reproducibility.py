from __future__ import annotations

import copy
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.release_candidate_reproducibility import (
    DIST_INFO,
    PROTOCOL,
    ReleaseCandidateError,
    ReleaseCandidateNotReady,
    _tar_bytes,
    audit_readiness,
    build_node_sbom,
    build_python_lock,
    build_source_archive,
    build_wheel,
    canonical_json,
    compare_outputs,
    load_contract,
    materialize_source,
    sha256_file,
    stable_digest,
    summarize_double_build_report,
    verify_output,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmark/release_candidate_reproducibility_v1_contract.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    # The a743 contract is immutable historical evidence.  Newer HEADs may
    # still verify its blobs without treating it as the current build input.
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_binds_exact_git_source_and_excludes_local_state(contract: dict[str, object]) -> None:
    assert contract["protocol"] == PROTOCOL
    assert contract["source_commit"] == "a743c59c719dd10db742cbd526f8f09c5ba13839"
    paths = [row["path"] for row in contract["source_manifest"]]
    assert len(paths) == len(set(paths)) == 312
    assert not any(path.startswith("third_party/") or path == ".env" for path in paths)
    assert stable_digest(contract["source_manifest"]) == contract["source_manifest_sha256"]


def test_materialized_source_is_exact_and_independent(tmp_path: Path, contract: dict[str, object]) -> None:
    first = tmp_path / "one/source"
    second = tmp_path / "other-parent/two/source"
    materialize_source(ROOT, contract, first)
    materialize_source(ROOT, contract, second)
    for row in contract["source_manifest"]:
        assert sha256_file(first / row["path"]) == row["sha256"]
        assert (first / row["path"]).read_bytes() == (second / row["path"]).read_bytes()


def test_wheel_and_source_archive_are_byte_deterministic(tmp_path: Path, contract: dict[str, object]) -> None:
    first = tmp_path / "a/source"
    second = tmp_path / "different/b/source"
    materialize_source(ROOT, contract, first)
    materialize_source(ROOT, contract, second)
    wheel_a = tmp_path / "a.whl"
    wheel_b = tmp_path / "b.whl"
    source_a = tmp_path / "a.tar.gz"
    source_b = tmp_path / "b.tar.gz"
    build_wheel(first, wheel_a, contract)
    build_wheel(second, wheel_b, contract)
    build_source_archive(first, source_a, contract)
    build_source_archive(second, source_b, contract)
    assert wheel_a.read_bytes() == wheel_b.read_bytes()
    assert source_a.read_bytes() == source_b.read_bytes()
    with zipfile.ZipFile(wheel_a) as archive:
        assert "scholar_agent/__init__.py" in archive.namelist()
        assert f"{DIST_INFO}/METADATA" in archive.namelist()


def test_dependency_closure_is_explicit_and_unknown_licenses_are_not_guessed() -> None:
    python_lock = build_python_lock(ROOT / "requirements.txt")
    node = build_node_sbom(ROOT / "frontend/package-lock.json")
    assert python_lock["complete"] is True
    assert python_lock["missing_packages"] == []
    assert python_lock["unpinned_direct_requirements"] == []
    assert all("==" in value for value in python_lock["direct_requirements"])
    assert any(item["license"] == "unknown" for item in python_lock["packages"])
    assert node["package_count"] == 434
    assert all(item["version"] != "unknown" for item in node["packages"])


def _fixture_release(path: Path, contract: dict[str, object], *, marker: bytes = b"stable") -> None:
    path.mkdir()
    wheel = path / "spar_scholar_agent-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("scholar_agent/__init__.py", marker)
        archive.writestr(f"{DIST_INFO}/METADATA", b"Metadata-Version: 2.3\n")
    for name in ("frontend-static.tar.gz", "source.tar.gz"):
        (path / name).write_bytes(_tar_bytes({"member.txt": marker}, contract))
    sbom = canonical_json({"schema_version": "1"})
    (path / "sbom.json").write_bytes(sbom)
    artifacts = [
        {"path": item.name, "size": item.stat().st_size, "sha256": sha256_file(item)}
        for item in sorted(path.iterdir())
    ]
    manifest = {
        "schema_version": "1",
        "protocol": PROTOCOL,
        "artifacts": artifacts,
        "dependency_violations": [],
    }
    (path / "release-manifest.json").write_bytes(canonical_json(manifest))
    inner = {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}
    (path / "spar-release-candidate.tar.gz").write_bytes(_tar_bytes(inner, contract))


def test_verify_accepts_complete_fixture_and_detects_content_missing(tmp_path: Path, contract: dict[str, object]) -> None:
    release = tmp_path / "release"
    _fixture_release(release, contract)
    assert verify_output(release, contract)["exit_code"] == 0
    (release / "source.tar.gz").unlink()
    report = verify_output(release, contract)
    assert report["exit_code"] == 2
    assert any(item["invariant"] == "output_member_set" for item in report["violations"])


@pytest.mark.parametrize(
    "name",
    [
        "build-time.txt",
        "absolute-path.txt",
        "extra-file.txt",
        "source-tamper.txt",
        "random-build-id.txt",
        "cross-commit.txt",
    ],
)
def test_compare_rejects_controlled_supply_chain_drift(tmp_path: Path, name: str) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / name).write_bytes(b"expected")
    (second / name).write_bytes(b"drift")
    report = compare_outputs(first, second)
    assert report["exit_code"] == 2
    assert report["differences"] == [
        {
            "path": name,
            "first_sha256": sha256_file(first / name),
            "second_sha256": sha256_file(second / name),
        }
    ]


def test_compare_is_order_and_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for name in ("z", "a", "m"):
        (first / name).write_bytes(name.encode())
        (second / name).write_bytes(name.encode())
    one = compare_outputs(first, second)
    two = compare_outputs(first, second)
    assert one == two
    assert canonical_json(one) == canonical_json(two)
    assert one["exit_code"] == 0


def test_manifest_tamper_and_cross_version_are_rejected(tmp_path: Path) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    value["source_manifest"][0]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReleaseCandidateError, match="source_manifest_digest"):
        load_contract(path, ROOT)

    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    value["protocol"] = "release_candidate_reproducibility_v2"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReleaseCandidateError, match="version"):
        load_contract(path, ROOT)


def test_duplicate_source_and_prohibited_archive_path_are_rejected(tmp_path: Path, contract: dict[str, object]) -> None:
    value = copy.deepcopy(contract)
    value["source_manifest"].append(copy.deepcopy(value["source_manifest"][0]))
    value["source_manifest_sha256"] = stable_digest(value["source_manifest"])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReleaseCandidateError, match="duplicate"):
        load_contract(path, ROOT)
    with pytest.raises(ReleaseCandidateError, match="prohibited"):
        _tar_bytes({"third_party/injected.txt": b"bad"}, contract)


def test_tracked_python_lock_matches_contract(contract: dict[str, object]) -> None:
    path = ROOT / contract["python_lock"]["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    assert stable_digest(value) == contract["python_lock"]["sha256"]


def test_toolchain_or_source_commit_drift_is_not_ready(tmp_path: Path) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    value["source_commit"] = "0" * 40
    path = tmp_path / "other-head.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReleaseCandidateNotReady, match="source_head"):
        load_contract(path, ROOT)


def test_double_build_summary_keeps_stable_evidence_only(contract: dict[str, object]) -> None:
    artifact = {"path": "source.tar.gz", "size": 7, "sha256": "a" * 64}
    report = {
        "status": "build_or_supply_chain_violation",
        "exit_code": 2,
        "source_commit": contract["source_commit"],
        "comparison": {
            "differences": [
                {"path": "frontend-static.tar.gz", "first_sha256": "b" * 64, "second_sha256": "c" * 64}
            ]
        },
        "artifacts": [artifact],
        "dependency_violations": [],
        "sbom_summary": {"python_package_count": 1},
        "execution": contract["execution"],
        "formal_validation_complete": False,
    }
    first = summarize_double_build_report(report)
    second = summarize_double_build_report(copy.deepcopy(report))
    assert canonical_json(first) == canonical_json(second)
    assert first["qualification"] == "not_qualified"
    assert first["differing_artifact_paths"] == ["frontend-static.tar.gz"]
    assert first["reproducible_artifacts"] == [artifact]
    assert "first_sha256" not in json.dumps(first)


def test_current_evidence_keeps_readiness_blocked(contract: dict[str, object]) -> None:
    evidence = ROOT / "benchmark/release_candidate_reproducibility_v1_evidence/current.json"
    report = audit_readiness(ROOT, contract, evidence)
    assert report["exit_code"] == 2
    assert report["status"] == "build_or_supply_chain_violation"
    assert report["formal_validation_complete"] is False
    assert "release_candidate_not_reproducible" in report["violations"]
