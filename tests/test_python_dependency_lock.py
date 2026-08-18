from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.python_dependency_lock import (
    EXIT_NOT_READY,
    DependencyLockError,
    DependencyLockNotReady,
    _cycles,
    _read_direct,
    build_manifest,
    lock_text,
    offline_install,
    verify_manifest,
    verify_wheel_metadata,
)
from scholar_agent.evaluation.release_candidate_reproducibility import (
    summarize_double_build_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict[str, object]:
    return json.loads(
        (ROOT / "benchmark/python_dependency_lock_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )


def _manifest() -> dict[str, object]:
    return json.loads(
        (ROOT / "benchmark/python_dependency_lock_v1_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def test_direct_declarations_require_exact_versions_and_normalized_unique_names(
    tmp_path: Path,
) -> None:
    unpinned = tmp_path / "unpinned.txt"
    unpinned.write_text("fastapi>=1\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="direct_dependency_not_exact"):
        _read_direct(unpinned, "runtime")

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("rank_bm25==0.2.2\nrank-bm25==0.2.2\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="duplicate_direct_dependency"):
        _read_direct(duplicate, "runtime")


def test_environment_and_installed_metadata_are_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    protocol["environment"] = {**protocol["environment"], "python_version": "0.0"}
    with pytest.raises(DependencyLockNotReady, match="environment_identity_mismatch"):
        build_manifest(ROOT, protocol)


def test_tracked_manifest_and_locks_are_deterministic() -> None:
    manifest = _manifest()
    report = verify_manifest(ROOT, _protocol(), manifest)
    assert report["lock_qualified"] is True
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["violations"] == []
    assert lock_text(manifest, "runtime") == (
        ROOT / "requirements-runtime.lock"
    ).read_bytes()
    assert lock_text(manifest, "development") == (
        ROOT / "requirements-dev.lock"
    ).read_bytes()
    assert "pytest==" not in lock_text(manifest, "runtime").decode()
    assert "httpx==" not in lock_text(manifest, "runtime").decode()
    assert lock_text(manifest, "runtime") == lock_text(manifest, "runtime")


def test_manifest_drift_and_development_leak_are_rejected() -> None:
    manifest = _manifest()
    drifted = json.loads(json.dumps(manifest))
    drifted["packages"][0]["version"] = "999"
    report = verify_manifest(ROOT, _protocol(), drifted)
    assert report["exit_code"] == 2
    assert "manifest_drift" in report["violations"]


def test_missing_local_artifacts_return_not_ready_without_creating_venvs() -> None:
    manifest = _manifest()
    report = offline_install(ROOT, _protocol(), manifest, {})
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["venv_results"] == []
    assert len(report["missing_artifacts"]) == manifest["package_count"]


def test_wheel_metadata_must_equal_runtime_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    wheel = tmp_path / "fixture.whl"
    metadata = "\n".join(
        [
            "Metadata-Version: 2.3",
            "Name: fixture",
            "Version: 1",
            *[f"Requires-Dist: {item}" for item in manifest["runtime_requires_dist"]],
            "",
        ]
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("spar_scholar_agent-0.1.0.dist-info/METADATA", metadata)
    assert verify_wheel_metadata(wheel, manifest)["passed"] is True

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "spar_scholar_agent-0.1.0.dist-info/METADATA",
            metadata + "Requires-Dist: pytest==9.1.1\n",
        )
    report = verify_wheel_metadata(wheel, manifest)
    assert report["passed"] is False
    assert "wheel_requires_dist_mismatch" in report["violations"]
    assert report["development_leaks"] == ["pytest==9.1.1"]


def test_dependency_cycles_are_reported_deterministically() -> None:
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    assert _cycles(graph) == [["a", "b", "c", "a"]]
    assert _cycles(dict(reversed(list(graph.items())))) == [["a", "b", "c", "a"]]


def test_reproducible_bytes_do_not_hide_dependency_qualification_failure() -> None:
    report = summarize_double_build_report(
        {
            "status": "build_or_supply_chain_violation",
            "exit_code": 2,
            "source_commit": "0" * 40,
            "comparison": {"differences": []},
            "artifacts": [],
            "dependency_violations": ["python_offline_install_not_qualified"],
        }
    )
    assert report["qualification"] == "not_qualified"
