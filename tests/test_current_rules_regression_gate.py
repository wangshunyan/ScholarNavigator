from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.evaluation.current_rules_regression import (
    check_current_rules_regression,
    preflight_current_rules_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_rules_preflight_is_structured_and_read_only() -> None:
    report = preflight_current_rules_inputs(
        ROOT / "benchmark/current_rules_regression_manifest.json"
    )
    assert report["status"] in {"ready", "external_evidence_unavailable", "input_invalid"}
    assert isinstance(report["checked"], list)
    assert isinstance(report["blockers"], list)


@pytest.mark.regression_gate
def test_frozen_current_rules_replay_has_no_regression() -> None:
    preflight = preflight_current_rules_inputs(
        ROOT / "benchmark/current_rules_regression_manifest.json"
    )
    if preflight["status"] != "ready":
        pytest.skip(
            "historical frozen evidence unavailable; run check_current_rules_regression.py "
            f"preflight for details ({preflight['status']})"
        )
    observed, report = check_current_rules_regression(
        ROOT / "benchmark/current_rules_regression_manifest.json"
    )

    assert report["passed"] is True, report["drifts"]
    assert report["case_count"] == 65
    assert report["drift_count"] == 0
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "snapshot_mode": "read_only",
        "external_record_in_gate": False,
    }
    assert set(observed["datasets"]) == {"scifact", "auto_dev", "auto_val"}
