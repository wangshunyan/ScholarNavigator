from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.package_server_evidence import EvidenceError, build_bundle, build_legacy_inventory, inspect_run
from scholar_agent.evaluation.crash_consistency import BenchmarkRunCommitStore


def _write_run(root: Path, *, complete: bool = True) -> Path:
    run = root / "contest_qual200_dense_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"dataset_source_path": "/home/example/data.jsonl", "code": {"commit": "abc"}}),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(json.dumps({"aggregate": {"f1": 0.1}}), encoding="utf-8")
    (run / "results.jsonl").write_text(json.dumps({"case_id": "q1", "status": "success"}) + "\n", encoding="utf-8")
    (run / "resource_ledger.json").write_text(json.dumps({"contract": "resource_ledger_v1"}), encoding="utf-8")
    (run / "summary.md").write_text("internal engineering metric only\n", encoding="utf-8")
    if complete:
        (run / ".run_complete").write_text("complete\n", encoding="utf-8")
    return run


def test_bundle_redacts_absolute_paths_and_keeps_hashes(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    bundle = tmp_path / "evidence.zip"

    report = build_bundle(run, bundle)

    assert report["schema_version"] == "server_evidence_bundle_v1"
    assert report["run"]["source_path_recorded"] is False
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        config = archive.read("run/config.json").decode("utf-8")
        manifest = json.loads(archive.read("manifest.json"))
    assert "/home/example" not in config
    assert "<redacted-path>" in config
    config_row = next(item for item in manifest["run"]["files"] if item["path"] == "config.json")
    assert config_row["source_sha256"] != config_row["exported_sha256"]


def test_incomplete_or_credential_bearing_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="completion_marker_missing"):
        inspect_run(_write_run(tmp_path / "incomplete", complete=False))

    run = _write_run(tmp_path / "credential")
    (run / "config.json").write_text(json.dumps({"api_key": "not-exportable"}), encoding="utf-8")
    with pytest.raises(EvidenceError, match="credential_field_present"):
        inspect_run(run)


def test_committed_generation_is_accepted_and_torn_compatibility_view_rejected(
    tmp_path: Path,
) -> None:
    run = tmp_path / "committed-run"
    run.mkdir()
    store = BenchmarkRunCommitStore(run)
    state = store.initialize(
        run_id=run.name,
        expected_query_ids=["q1"],
        config={"dataset": "fixture", "case_ids": ["q1"]},
        dataset_report={"case_count": 1},
    )
    state = store.commit_record({"case_id": "q1", "status": "succeeded"})
    state = store.commit_completion(
        {"metrics.json": b"{}\n", "resource_ledger.json": b"{}\n"}
    )
    store.materialize_compatibility_view(state)

    inspection = inspect_run(run)

    assert inspection["completion_markers"] == ["run_commit_generation"]
    assert inspection["committed_completion"]["record_count"] == 1
    bundle = tmp_path / "committed-evidence.zip"
    build_bundle(run, bundle)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["run"]["committed_completion"]["generation"] == state.generation

    (run / "results.jsonl").write_text('{"case_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(EvidenceError, match="committed_compatibility_view_drift:results.jsonl"):
        inspect_run(run)


def test_bundle_redacts_sensitive_diagnostic_text_in_result_rows(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    (run / "results.jsonl").write_text(
        json.dumps({"case_id": "q1", "error_message": "API key unavailable"}) + "\n",
        encoding="utf-8",
    )

    bundle = tmp_path / "evidence.zip"
    build_bundle(run, bundle)

    with zipfile.ZipFile(bundle) as archive:
        result = archive.read("run/results.jsonl").decode("utf-8")
    assert "API key" not in result
    assert "<redacted-sensitive-text>" in result


def test_legacy_inventory_excludes_results_and_gold_diagnostics(tmp_path: Path) -> None:
    run = _write_run(tmp_path, complete=False)
    (run / "gold_diagnostics.jsonl").write_text('{"gold":"not-exported"}\n', encoding="utf-8")
    bundle = tmp_path / "legacy.zip"

    report = build_legacy_inventory(run, bundle)

    assert report["schema_version"] == "server_legacy_inventory_v1"
    assert report["run"]["completion_status"] == "unverified_legacy_inventory"
    with zipfile.ZipFile(bundle) as archive:
        assert "run/results.jsonl" not in archive.namelist()
        assert "run/gold_diagnostics.jsonl" not in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["run"]["result_artifact"]["source_bytes"] > 0
