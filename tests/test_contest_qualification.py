from __future__ import annotations

from pathlib import Path

from scripts import check_contest_qualification as qualification


def _metrics(delta: float) -> dict:
    rows = []
    for index in range(200):
        baseline = 0.0 if index < 100 else 0.2
        rows.append(
            {
                "case_id": f"q-{index}",
                "metrics": {
                    "f1_at_k": {"20": baseline + delta},
                    "recall_at_k": {"20": baseline + delta},
                },
            }
        )
    return {"case_statistics": {"total_case_count": 200, "failed_case_count": 0}, "per_case": rows}


def _run(delta: float) -> dict:
    return {
        "config_hashes": {
            "dataset": "auto_scholar_query",
            "dataset_split": "test",
            "offset": 0,
            "limit": 200,
            "top_k": 20,
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "data_hashes": {"pasa": "same"},
        },
        "metrics": _metrics(delta),
        "resource_report": {"status": "passed"},
    }


def test_qualification_requires_positive_bootstrap_interval(monkeypatch) -> None:
    def fake_load(path: Path, _expected: str) -> dict:
        return _run(0.0 if path.name == qualification.EXPECTED_BASELINE else 0.1)

    monkeypatch.setattr(qualification, "_load_run", fake_load)

    report = qualification.check_qualification(
        Path(qualification.EXPECTED_BASELINE),
        Path("contest_qual200_dense_v1"),
    )

    assert report["eligible_for_full_1000"] is True
    assert report["strict_positive_improvement"] == {
        "f1_at_20": True,
        "recall_at_20": True,
    }


def test_qualification_rejects_no_metric_improvement(monkeypatch) -> None:
    monkeypatch.setattr(qualification, "_load_run", lambda *_: _run(0.0))

    report = qualification.check_qualification(
        Path(qualification.EXPECTED_BASELINE),
        Path("contest_qual200_reranker_v1"),
    )

    assert report["eligible_for_full_1000"] is False
    assert not any(report["strict_positive_improvement"].values())
