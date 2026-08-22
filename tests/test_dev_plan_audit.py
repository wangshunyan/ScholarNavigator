from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scholar_agent.evaluation.dev_plan_audit import audit_benchmark_artifacts


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_distinguishes_missing_external_and_hash_drift(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    tracked = benchmark / "tracked.json"
    tracked.write_text("{}", encoding="utf-8")
    manifest = benchmark / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tracked": {"path": "benchmark/tracked.json", "sha256": _digest(tracked)},
                "external": {"path": "outputs/run/results.jsonl", "sha256": "0" * 64},
                "outside": {"path": "../outside.json"},
            }
        ),
        encoding="utf-8",
    )

    report = audit_benchmark_artifacts(tmp_path)
    statuses = {item["location"]: item["status"] for item in report["records"]}

    assert statuses["$.tracked.path"] == "present"
    assert statuses["$.external.path"] == "missing_external"
    assert statuses["$.outside.path"] == "unsafe_absolute"
    assert report["passed"] is False


def test_audit_is_deterministic_and_reports_unhashed_paths(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "present.txt").write_text("ok", encoding="utf-8")
    (benchmark / "manifest.json").write_text(
        json.dumps({"artifact": {"path": "benchmark/present.txt"}}),
        encoding="utf-8",
    )

    first = audit_benchmark_artifacts(tmp_path)
    second = audit_benchmark_artifacts(tmp_path)

    assert first == second
    assert first["records"][0]["status"] == "present_unhashed"
    assert first["passed"] is True
