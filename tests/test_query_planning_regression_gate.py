from pathlib import Path

import pytest

from scholar_agent.evaluation.query_planning_regression import (
    check_planning_regression,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.planning_regression
def test_frozen_autoscholar_1000_query_planning_gate(tmp_path: Path) -> None:
    report = check_planning_regression(
        REPOSITORY_ROOT / "benchmark" / "autoscholar_query_planning_manifest.json",
        tmp_path / "planning-gate",
    )

    assert report["passed"] is True
    assert report["case_count"] == 1000
    assert report["success_count"] == 1000
    assert report["error_count"] == 0
    assert report["drift_count"] == 0
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "connector_invoked": False,
        "evaluator_invoked": False,
        "gold_fields_accessed": False,
    }
