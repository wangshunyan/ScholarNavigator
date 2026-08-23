from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.package_server_evidence import EvidenceError, build_bundle, inspect_run


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
