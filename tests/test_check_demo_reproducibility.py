from __future__ import annotations

from scripts.check_demo_reproducibility import validate_demo_manifest


def _manifest(**overrides):
    payload = {
        "schema_version": "search-batch-manifest-v1",
        "case_count": 5,
        "executed_case_count": 5,
        "partial": False,
        "succeeded_count": 5,
        "failed_count": 0,
        "network_requests": 0,
        "llm_calls": 0,
        "gold_or_qrels_loaded": False,
        "case_summaries": [
            {"case_id": f"demo_{index:02d}", "status": "succeeded", "visible_result_count": 5}
            for index in range(1, 6)
        ],
    }
    payload.update(overrides)
    return payload


def test_validate_demo_manifest_accepts_complete_gold_blind_run() -> None:
    assert validate_demo_manifest(_manifest()) == []


def test_validate_demo_manifest_rejects_empty_case_and_network_use() -> None:
    manifest = _manifest(network_requests=1)
    manifest["case_summaries"][2]["visible_result_count"] = 0
    reasons = validate_demo_manifest(manifest)
    assert "demo_network_requests_nonzero" in reasons
    assert "demo_case_no_visible_results:demo_03" in reasons

