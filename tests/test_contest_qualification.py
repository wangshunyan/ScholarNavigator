from __future__ import annotations

import json
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


def test_reranker_audit_rejects_fallback_and_accepts_real_inference(tmp_path: Path) -> None:
    path = tmp_path / "contest_qual200_reranker_v2"
    path.mkdir()
    rows = []
    for index in range(200):
        rows.append(
            {
                "case_id": f"q-{index}",
                "diagnostics": {
                    "local_model_batch_count": 15,
                    "local_model_fallback_count": 0,
                    "local_model_inference_success_count": 1,
                    "local_model_candidate_count": 120,
                    "local_model_prompt_version": "qwen3-reranker-v1",
                    "local_model_device": "cuda:0",
                    "local_model_max_length": 2048,
                    "local_model_fingerprint": "model-fingerprint",
                    "local_model_latency_seconds": 0.5,
                    "local_model_batch_size": 8,
                    "local_model_candidate_limit": 120,
                    "local_model_peak_vram_bytes": 123456,
                },
            }
        )
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    passed = qualification._audit_reranker_run(path)
    assert passed["status"] == "passed"
    assert passed["fallback_count"] == 0

    rows[0]["diagnostics"]["local_model_fallback_count"] = 1
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    failed = qualification._audit_reranker_run(path)
    assert failed["status"] == "failed"
    assert "reranker_fallback_detected" in failed["reasons"]


def test_gpu_isolated_reranker_retry_is_an_explicit_candidate() -> None:
    assert "contest_qual200_reranker_v2_gpu1" in qualification.EXPECTED_CANDIDATES
    assert "contest_qual200_reranker_v3_gpu1" in qualification.EXPECTED_CANDIDATES
    assert "contest_qual200_reranker_v4_gpu1" in qualification.EXPECTED_CANDIDATES
