from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_contest_llm_run import audit_run


def _row(index: int, *, calls: int = 1, supplemental: int = 2) -> dict:
    return {
        "case_id": f"q-{index}",
        "status": "succeeded",
        "cost_report": {"llm_call_count": calls},
        "stage_diagnostics": {
            "initial_query_planning": {
                "planning": {
                    "policy": "llm_semantic",
                    "llm_call_attempted": True,
                    "fallback_used": False,
                    "original_query_retained": True,
                    "prompt_version": "llm-query-planning-v1",
                    "model": "configured-model",
                    "llm_schema_version": "1",
                    "llm_temperature": 0.0,
                    "llm_max_supplemental_queries": 2,
                    "llm_max_tokens": 512,
                    "llm_prompt_tokens": 10,
                    "llm_completion_tokens": 5,
                    "llm_total_tokens": 15,
                    "recorded_llm_latency_seconds": 0.2,
                    "llm_http_attempts": 1,
                    "llm_http_429_count": 0,
                    "llm_retry_after_seconds": [],
                    "llm_retry_wait_seconds": 0.0,
                    "llm_provider_failure_class": None,
                    "llm_provider_cache_hit": False,
                },
                "subqueries": [
                    {"purpose": "original_query"},
                    *[{"purpose": "llm_semantic:topic"} for _ in range(supplemental)],
                ],
            }
        },
    }


def _run(tmp_path: Path, *, expected_rows: int = 1000) -> Path:
    run_name = "contest_full_dense_reranker_llm_v4" if expected_rows == 1000 else "contest_qual200_dense_reranker_llm_v16"
    path = tmp_path / run_name
    (path / ".run_commits" / "generations" / "generation-00001002").mkdir(parents=True)
    (path / ".run_commits" / "generations" / "generation-00001002" / "RUN_COMPLETED").write_text("{}", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"query_planning_policy": "llm_semantic", "limit": expected_rows, "top_k": 20, "budgets": {"max_llm_calls": 1, "max_search_rounds": 3}}), encoding="utf-8")
    (path / "metrics.json").write_text("{}", encoding="utf-8")
    (path / "resource_ledger.json").write_text("{}", encoding="utf-8")
    (path / "results.jsonl").write_text("\n".join(json.dumps(_row(index)) for index in range(expected_rows)) + "\n", encoding="utf-8")
    return path


def test_audit_accepts_complete_controlled_llm_run(tmp_path: Path) -> None:
    report = audit_run(_run(tmp_path))
    assert report["status"] == "passed"
    assert report["claimable_live_llm_effect"] is True
    assert report["transport_telemetry"]["available_row_count"] == 1000
    assert report["transport_telemetry"]["http_attempt_total"] == 1000


def test_audit_accepts_200_query_qualification_only_when_requested(tmp_path: Path) -> None:
    path = _run(tmp_path, expected_rows=200)
    qualification = audit_run(path, expected_rows=200)
    full = audit_run(path)

    assert qualification["status"] == "passed"
    assert qualification["expected_row_count"] == 200
    assert full["status"] == "failed"
    assert "full1000_or_topk_contract_drift" in full["reasons"]


def test_audit_accepts_feedback_smoke_with_normal_skips(tmp_path: Path) -> None:
    path = _run(tmp_path, expected_rows=5)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config.update(
        {"enable_query_evolution": True, "query_evolution_policy": "llm_feedback"}
    )
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [_row(index, calls=0) for index in range(5)]
    for row in rows:
        row["stage_diagnostics"] = {
            "query_evolution": {
                "llm_feedback": [
                    {
                        "policy": "llm_feedback",
                        "eligible_for_feedback": False,
                        "skipped_reason": "coverage_sufficient",
                        "llm_call_attempted": False,
                        "fallback_used": False,
                        "original_query_retained": True,
                        "supplemental_query_count": 0,
                        "accepted_query_count": 0,
                        "http_attempts": 0,
                        "http_429_count": 0,
                        "retry_after_seconds": [],
                        "retry_wait_seconds": 0.0,
                        "provider_failure_class": None,
                        "provider_cache_hit": False,
                    }
                ]
            }
        }
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = audit_run(path, expected_rows=5)

    assert report["status"] == "passed"
    assert report["feedback_eligible_count"] == 0
    assert report["feedback_skipped_count"] == 5


def test_audit_rejects_multiple_calls_or_supplemental_queries(tmp_path: Path) -> None:
    path = _run(tmp_path)
    rows = [_row(index, calls=2 if index == 0 else 1, supplemental=3 if index == 1 else 2) for index in range(1000)]
    (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = audit_run(path)
    assert report["status"] == "failed"
    assert "per_query_llm_call_count_exceeded" in report["reasons"]
    assert "supplemental_query_count_exceeded" in report["reasons"]


def test_audit_rejects_a_completed_run_with_fallback(tmp_path: Path) -> None:
    path = _run(tmp_path)
    rows = [_row(index) for index in range(1000)]
    rows[0]["stage_diagnostics"]["initial_query_planning"]["planning"]["fallback_used"] = True
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = audit_run(path)
    assert report["status"] == "failed"
    assert "llm_fallback_detected" in report["reasons"]


def test_audit_rejects_missing_execution_transport_or_schema_metadata(
    tmp_path: Path,
) -> None:
    path = _run(tmp_path)
    rows = [_row(index) for index in range(1000)]
    planning = rows[0]["stage_diagnostics"]["initial_query_planning"]["planning"]
    del planning["llm_prompt_tokens"]
    del planning["llm_http_attempts"]
    del planning["llm_schema_version"]
    planning["llm_temperature"] = 0.2
    planning["llm_max_supplemental_queries"] = 1
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = audit_run(path)

    assert report["status"] == "failed"
    assert "llm_execution_metadata_missing" in report["reasons"]
    assert "llm_transport_metadata_missing" in report["reasons"]
    assert "llm_schema_contract_missing" in report["reasons"]
    assert "llm_temperature_contract_drift" in report["reasons"]
    assert "llm_supplemental_budget_metadata_missing" in report["reasons"]


def test_contest_runners_enforce_the_per_query_llm_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    shell = (root / "scripts" / "run_contest_benchmark.sh").read_text(encoding="utf-8")
    powershell = (root / "scripts" / "run_contest_benchmark.ps1").read_text(encoding="utf-8")
    assert 'ARGS+=("--max-llm-calls" "1" "--max-search-rounds" "3")' in shell
    assert 'dense_reranker_llm_feedback' in shell
    assert 'ARGS+=("--enable-query-evolution" "--query-evolution-policy" "llm_feedback")' in shell
    assert "SCHOLARNAVIGATOR_RUN_LOG_PATH" in shell
    assert '"--max-llm-calls", "1",' in powershell
    assert '"--max-search-rounds", "3"' in powershell
    assert '"dense_reranker_llm_feedback"' in powershell
    assert '"--query-evolution-policy", "llm_feedback"' in powershell
    assert "SCHOLARNAVIGATOR_RUN_LOG_PATH" in powershell
