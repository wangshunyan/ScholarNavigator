from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scholar_agent.evaluation.snapshot_resume import load_resume_manifest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "benchmark" / "autoscholar_full1000_resume"


def test_tracked_resume_audit_is_closed_and_manifest_covers_every_incomplete_key() -> None:
    summary = json.loads((ARTIFACT_ROOT / "summary.json").read_text(encoding="utf-8"))
    manifest = load_resume_manifest(ARTIFACT_ROOT / "resume_manifest.json")
    rows = [
        json.loads(line)
        for line in (ARTIFACT_ROOT / "key_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    counts = Counter(row["classification"] for row in rows)
    row_keys = [row["key"] for row in rows]
    resume_keys = [request.key for request in manifest.requests]
    expected_resume_keys = {
        row["key"] for row in rows if row["classification"] != "success"
    }

    assert len(rows) == len(set(row_keys)) == 6858
    assert counts == {
        "success": 1272,
        "failed": 16,
        "missing": 5,
        "not_started": 5565,
    }
    assert summary["classification_counts"] == dict(sorted(counts.items()))
    assert manifest.required_key_count == 6858
    assert len(resume_keys) == len(set(resume_keys)) == 5586
    assert set(resume_keys) == expected_resume_keys
    assert summary["schedule"]["max_consecutive_same_source"] == 1
    assert summary["schedule"]["max_consecutive_same_case"] == 1
    assert summary["execution"] == {
        "dataset_adapter_invoked": False,
        "effectiveness_metrics_generated": False,
        "evaluator_invoked": False,
        "gold_fields_accessed": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }
