from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.evaluation.current_rules_regression import (
    check_current_rules_regression,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.regression_gate
def test_frozen_current_rules_replay_has_no_regression() -> None:
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
