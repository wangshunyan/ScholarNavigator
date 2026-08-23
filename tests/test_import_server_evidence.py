from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.import_server_evidence import ImportError, import_bundle
from scripts.package_server_evidence import build_bundle


def _run(root: Path) -> Path:
    run = root / "run-001"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"commit": "abc"}), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({"f1": 0.1}), encoding="utf-8")
    (run / "results.jsonl").write_text('{"status":"success"}\n', encoding="utf-8")
    (run / "resource_ledger.json").write_text(json.dumps({"calls": 0}), encoding="utf-8")
    (run / ".run_complete").write_text("complete\n", encoding="utf-8")
    return run


def test_import_verifies_and_materializes_redacted_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "evidence.zip"
    build_bundle(_run(tmp_path), bundle)
    destination = tmp_path / "outputs" / "imported_server_evidence" / "run-001"
    report = import_bundle(bundle, destination)
    assert report["status"] == "ready"
    assert report["network_requests"] == 0
    assert (destination / "manifest.json").is_file()
    assert json.loads((destination / "config.json").read_text()) == {"commit": "abc"}


def test_import_rejects_tampered_member(tmp_path: Path) -> None:
    bundle = tmp_path / "evidence.zip"
    build_bundle(_run(tmp_path), bundle)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "run/metrics.json":
                payload = b"{}"
            target.writestr(info, payload)
    with pytest.raises(ImportError, match="exported_(?:size|hash)_mismatch:metrics.json"):
        import_bundle(tampered, tmp_path / "destination")
