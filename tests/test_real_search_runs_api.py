from __future__ import annotations

import json
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats  # noqa: E402
from scholar_agent.agents.synthesis import synthesize_answer  # noqa: E402
from scholar_agent.app.api import routes  # noqa: E402
from scholar_agent.app.main import app  # noqa: E402
from scholar_agent.core.api_schemas import (  # noqa: E402
    CostReport,
    RunProgress,
    SearchRunCreateRequest,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers  # noqa: E402
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics  # noqa: E402
from scholar_agent.core.search_schemas import (  # noqa: E402
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
    SearchBudget,
    SearchSubquery,
    TimeRange,
)
from scholar_agent.services.search_service import (  # noqa: E402
    SearchCancelled,
    SearchServiceOutput,
)


client = TestClient(app)


def test_real_search_run_is_created_before_background_result_is_ready(
    monkeypatch,
) -> None:
    release = threading.Event()
    captured: dict[str, object] = {}
    monkeypatch.delenv("REAL_SEARCH_MAX_WORKERS", raising=False)

    class BlockingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            captured["max_workers"] = kwargs.get("max_workers")

        def run_search(
            self,
            query: str,
            top_k: int = 20,
            run_profile: str = "balanced",
            enable_refchain: bool = False,
            enable_query_evolution: bool = False,
            query_evolution_policy: str = "coverage_gap",
            query_planning_policy: str = "current_rules",
            judgement_policy: str = "current_rules",
            enable_synthesis: bool = True,
            current_year: int | None = None,
            enable_llm_query_understanding: bool | None = None,
            enable_llm_judgement: bool | None = None,
            sources_override: list[str] | None = None,
            explicit_constraints: QueryConstraint | None = None,
            budget: SearchBudget | None = None,
            event_callback=None,
            should_cancel=None,
        ) -> SearchServiceOutput:
            captured.update(
                {
                    "query": query,
                    "top_k": top_k,
                    "run_profile": run_profile,
                    "enable_refchain": enable_refchain,
                    "enable_query_evolution": enable_query_evolution,
                    "query_evolution_policy": query_evolution_policy,
                    "query_planning_policy": query_planning_policy,
                    "judgement_policy": judgement_policy,
                    "enable_synthesis": enable_synthesis,
                    "current_year": current_year,
                    "enable_llm_query_understanding": enable_llm_query_understanding,
                    "enable_llm_judgement": enable_llm_judgement,
                    "sources_override": sources_override,
                    "explicit_constraints": explicit_constraints,
                    "budget": budget,
                }
            )
            assert release.wait(timeout=2)
            assert not should_cancel()
            output = _fake_output(query, top_k=top_k)
            _emit_fake_search_events(event_callback, output)
            return output

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", BlockingSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={
            "query": "latest LLM reranking retrieval papers",
            "top_k": 5,
            "run_profile": "high_recall",
            "constraints": {
                "time_range": {"start_year": 2023, "end_year": 2026},
                "venues": ["SIGIR"],
                "must_have_terms": ["reranking"],
                "excluded_terms": ["vision"],
                "datasets": ["LitSearch"],
                "paper_types": ["Systematic Review"],
            },
            "source_preferences": [
                "arxiv",
                "semantic_scholar",
                "openalex",
                "arxiv",
            ],
            "budgets": {
                "max_search_rounds": 1,
                "max_candidate_papers": 17,
                "max_llm_calls": 3,
                "max_total_tokens": 1234,
                "max_latency_seconds": 12.5,
            },
            "options": {
                "judgement_policy": "calibrated_rules_v1",
                "enable_query_evolution": True,
                "enable_refchain": False,
                "enable_llm_query_understanding": True,
                "enable_llm_judgement": True,
            },
        },
    )

    assert create_response.status_code == 201
    create_body = create_response.json()
    run_id = create_body["run_id"]
    assert run_id.startswith("run_real_")
    assert create_body["status"] in {"queued", "running"}
    assert create_body["links"]["self"] == f"/api/v1/real/search/runs/{run_id}"

    early_status = client.get(f"/api/v1/real/search/runs/{run_id}")
    assert early_status.status_code == 200
    assert early_status.json()["status"] in {"queued", "running"}

    not_ready = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert not_ready.status_code == 409
    assert not_ready.json()["detail"] == "result not ready"

    release.set()
    status_body = _wait_for_status(run_id, "succeeded")
    assert status_body["current_stage"] == "synthesis"
    assert status_body["progress"]["candidate_paper_count"] == 2
    assert status_body["progress"]["judged_paper_count"] == 2
    assert status_body["cost_report"]["judged_paper_count"] == 2
    assert captured == {
        "max_workers": 2,
        "query": "latest LLM reranking retrieval papers",
        "top_k": 5,
        "run_profile": "high_recall",
        "enable_refchain": False,
        "enable_query_evolution": True,
        "query_evolution_policy": "coverage_gap",
        "query_planning_policy": "current_rules",
        "judgement_policy": "calibrated_rules_v1",
        "enable_synthesis": True,
        "current_year": None,
        "enable_llm_query_understanding": True,
        "enable_llm_judgement": True,
        "sources_override": ["arxiv", "semantic_scholar", "openalex"],
        "explicit_constraints": QueryConstraint(
            time_range=TimeRange(
                start_year=2023,
                end_year=2026,
                label="explicit",
            ),
            venues=["SIGIR"],
            must_include_terms=["reranking"],
            exclude_terms=["vision"],
            datasets=["LitSearch"],
            paper_types=["review"],
        ),
        "budget": SearchBudget(
            max_search_rounds=1,
            max_candidate_papers=17,
            max_llm_calls=3,
            max_total_tokens=1234,
            max_latency_seconds=12.5,
        ),
    }

    result_response = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert result_response.status_code == 200
    result_body = result_response.json()
    assert result_body["run_id"] == run_id
    assert result_body["status"] == "succeeded"
    assert result_body["synthesis"] is not None
    assert result_body["synthesis"]["evidence_table"][0]["citation_key"] == "R1"
    assert result_body["highly_relevant_papers"][0]["paper"]["title"] == "Real High"
    assert result_body["partially_relevant_papers"][0]["paper"]["title"] == "Real Partial"


def test_real_search_events_replay_started_and_completed(monkeypatch) -> None:
    class FastSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            output = _fake_output(query)
            _emit_fake_search_events(kwargs.get("event_callback"), output)
            return output

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FastSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers", "top_k": 5},
    )
    assert create_response.status_code == 201
    run_id = create_response.json()["run_id"]
    status = _wait_for_status(run_id, "succeeded")
    assert status["progress"]["completed_stages"] == [
        "query_understanding",
        "retrieval",
        "deduplication",
        "judgement",
        "reranking",
        "synthesis",
    ]
    assert status["progress"]["skipped_stages"] == [
        "query_evolution",
        "refchain",
    ]

    with client.stream("GET", f"/api/v1/real/search/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = _parse_sse_events(text)
    connector_events = [
        event for event in events if event["event"] == "connector_completed"
    ]
    warning_events = [event for event in events if event["event"] == "warning"]
    cost_events = [event for event in events if event["event"] == "cost_updated"]

    assert "event: run_started" in text
    assert "event: query_understanding_started" in text
    assert "event: query_understanding_completed" in text
    assert "event: retrieval_started" in text
    assert "event: retrieval_completed" in text
    assert "event: judgement_started" in text
    assert "event: judgement_completed" in text
    assert "event: reranking_started" in text
    assert "event: reranking_completed" in text
    assert "event: synthesis_started" in text
    assert "event: synthesis_completed" in text
    assert "event: stage_started" not in text
    assert "event: stage_completed" not in text
    assert "event: connector_completed" in text
    assert "event: warning" in text
    assert "event: cost_updated" in text
    assert '"stage": "synthesis"' in text
    assert "event: run_completed" in text
    assert '"status": "succeeded"' in text
    assert sum(event["event"] == "run_completed" for event in events) == 1
    assert len(connector_events) == 2
    assert connector_events[0]["payload"] == {
        "stage": "retrieval",
        "query_index": 0,
        "query": "LLM reranking retrieval papers",
        "connector": "openalex",
        "source": "openalex",
        "returned_count": 1,
        "latency_seconds": 0.1,
        "request_count": 1,
        "retry_count": 0,
        "error_count": 0,
        "cache_hit": False,
        "cache_hit_count": 0,
        "rate_limit_wait_seconds": 0.0,
        "error_message": None,
        "run_id": run_id,
        "timestamp": connector_events[0]["payload"]["timestamp"],
    }
    assert connector_events[1]["payload"]["source"] == "arxiv"
    assert connector_events[1]["payload"]["cache_hit"] is True
    assert warning_events[0]["payload"]["message"] == "real_search_warning"
    assert cost_events[0]["payload"]["run_id"] == run_id
    assert cost_events[0]["payload"]["cost_report"]["judged_paper_count"] == 2


def test_route_does_not_fabricate_pipeline_events_after_service_returns(
    monkeypatch,
) -> None:
    class SilentSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            return _fake_output(query)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", SilentSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "silent fixture", "top_k": 5},
    )
    run_id = create_response.json()["run_id"]
    _wait_for_status(run_id, "succeeded")

    with client.stream("GET", f"/api/v1/real/search/runs/{run_id}/events") as response:
        events = _parse_sse_events("".join(response.iter_text()))

    event_names = [event["event"] for event in events]
    assert event_names == ["run_started", "cost_updated", "run_completed"]


def test_cancelling_succeeded_run_does_not_change_terminal_state(monkeypatch) -> None:
    class FastSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            return _fake_output(query)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FastSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "already completed", "top_k": 5},
    )
    run_id = create_response.json()["run_id"]
    _wait_for_status(run_id, "succeeded")

    cancel_response = client.post(f"/api/v1/real/search/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "succeeded"

    with client.stream("GET", f"/api/v1/real/search/runs/{run_id}/events") as response:
        events = _parse_sse_events("".join(response.iter_text()))
    assert sum(event["event"] == "run_completed" for event in events) == 1
    assert all(event["event"] != "run_cancelled" for event in events)


def test_real_search_failed_run_records_error_and_failed_status(monkeypatch) -> None:
    class FailingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, *args, **kwargs) -> SearchServiceOutput:
            raise RuntimeError("service exploded")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FailingSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers", "top_k": 5},
    )
    assert create_response.status_code == 201
    run_id = create_response.json()["run_id"]

    status_body = _wait_for_status(run_id, "failed")
    assert status_body["current_stage"] == "failed"

    result_response = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert result_response.status_code == 500
    assert result_response.json()["detail"] == "service exploded"

    with client.stream("GET", f"/api/v1/real/search/runs/{run_id}/events") as response:
        text = "".join(response.iter_text())
    events = _parse_sse_events(text)

    assert "event: error" in text
    assert "service exploded" in text
    assert "event: run_completed" in text
    assert '"status": "failed"' in text
    assert sum(event["event"] == "error" for event in events) == 1
    assert sum(event["event"] == "run_completed" for event in events) == 1


def test_running_real_search_can_be_cancelled_and_ignores_late_result(
    monkeypatch,
) -> None:
    release = threading.Event()
    service_entered = threading.Event()

    class BlockingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            service_entered.set()
            assert release.wait(timeout=2)
            if kwargs["should_cancel"]():
                raise SearchCancelled("fixture:after_wait")
            return _fake_output(query)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", BlockingSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers", "top_k": 5},
    )
    assert create_response.status_code == 201
    run_id = create_response.json()["run_id"]
    assert service_entered.wait(timeout=2)

    cancel_response = client.post(f"/api/v1/real/search/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    cancel_body = cancel_response.json()
    assert cancel_body["status"] == "cancelled"
    assert cancel_body["current_stage"] == "cancelled"

    result_response = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert result_response.status_code == 409
    assert result_response.json()["detail"] == "run cancelled"

    repeat_cancel = client.post(f"/api/v1/real/search/runs/{run_id}/cancel")
    assert repeat_cancel.status_code == 200
    assert repeat_cancel.json()["status"] == "cancelled"
    routes._fail_real_run(run_id, "late background failure")

    release.set()
    time.sleep(0.1)
    final_status = client.get(f"/api/v1/real/search/runs/{run_id}")
    assert final_status.status_code == 200
    assert final_status.json()["status"] == "cancelled"
    assert final_status.json()["current_stage"] == "cancelled"

    with client.stream("GET", f"/api/v1/real/search/runs/{run_id}/events") as response:
        text = "".join(response.iter_text())
    events = _parse_sse_events(text)

    assert "event: run_cancelled" in text
    assert "event: run_completed" in text
    assert '"status": "cancelled"' in text
    assert "event: cost_updated" not in text
    assert '"status": "succeeded"' not in text
    assert "event: error" not in text
    assert sum(event["event"] == "run_cancelled" for event in events) == 1
    assert sum(event["event"] == "run_completed" for event in events) == 1


def test_queued_real_search_can_be_cancelled_before_worker_starts(monkeypatch) -> None:
    release_first = threading.Event()
    first_entered = threading.Event()

    class BlockingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            first_entered.set()
            assert release_first.wait(timeout=2)
            return _fake_output(query)

    monkeypatch.setenv("REAL_SEARCH_BACKGROUND_WORKERS", "1")
    monkeypatch.setattr("scholar_agent.app.api.routes._REAL_SEARCH_EXECUTOR", None)
    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", BlockingSearchService)

    first_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "first blocking run", "top_k": 5},
    )
    assert first_response.status_code == 201
    assert first_entered.wait(timeout=2)

    second_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "queued run to cancel", "top_k": 5},
    )
    assert second_response.status_code == 201
    second_run_id = second_response.json()["run_id"]

    cancel_response = client.post(f"/api/v1/real/search/runs/{second_run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"

    release_first.set()
    _wait_for_status(first_response.json()["run_id"], "succeeded")
    time.sleep(0.1)

    status_response = client.get(f"/api/v1/real/search/runs/{second_run_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "cancelled"

    result_response = client.get(f"/api/v1/real/search/runs/{second_run_id}/result")
    assert result_response.status_code == 409
    assert result_response.json()["detail"] == "run cancelled"


def test_cancel_unknown_real_run_returns_404() -> None:
    response = client.post("/api/v1/real/search/runs/run_real_missing/cancel")

    assert response.status_code == 404


def test_real_search_unknown_run_id_returns_404() -> None:
    status_response = client.get("/api/v1/real/search/runs/run_real_missing")
    result_response = client.get("/api/v1/real/search/runs/run_real_missing/result")
    events_response = client.get("/api/v1/real/search/runs/run_real_missing/events")

    assert status_response.status_code == 404
    assert result_response.status_code == 404
    assert events_response.status_code == 404


def test_real_search_empty_query_returns_400_without_creating_run(monkeypatch) -> None:
    class FailingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("empty query should not instantiate SearchService")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FailingSearchService)

    response = client.post(
        "/api/v1/real/search/runs",
        json={"query": " ", "top_k": 5},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query must not be empty"


def test_background_value_error_marks_run_failed(monkeypatch) -> None:
    class ValueErrorSearchService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_search(self, *args, **kwargs) -> SearchServiceOutput:
            raise ValueError("query understanding failed")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", ValueErrorSearchService)

    create_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers", "top_k": 5},
    )
    assert create_response.status_code == 201
    run_id = create_response.json()["run_id"]

    status_body = _wait_for_status(run_id, "failed")
    assert status_body["current_stage"] == "failed"

    result_response = client.get(f"/api/v1/real/search/runs/{run_id}/result")
    assert result_response.status_code == 500
    assert result_response.json()["detail"] == "query understanding failed"


def test_real_search_uses_real_search_max_workers_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            captured["max_workers"] = kwargs.get("max_workers")

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            return _fake_output(query)

    monkeypatch.setenv("REAL_SEARCH_MAX_WORKERS", "7")
    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers", "top_k": 5},
    )

    assert response.status_code == 201
    _wait_for_status(response.json()["run_id"], "succeeded")
    assert captured["max_workers"] == 7


def test_real_search_normalizes_invalid_or_small_max_workers_env(monkeypatch) -> None:
    captured: list[int | None] = []

    class FakeSearchService:
        def __init__(self, *args, **kwargs) -> None:
            captured.append(kwargs.get("max_workers"))

        def run_search(self, query: str, *args, **kwargs) -> SearchServiceOutput:
            return _fake_output(query)

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FakeSearchService)

    monkeypatch.setenv("REAL_SEARCH_MAX_WORKERS", "not-an-int")
    invalid_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers"},
    )
    assert invalid_response.status_code == 201
    _wait_for_status(invalid_response.json()["run_id"], "succeeded")

    monkeypatch.setenv("REAL_SEARCH_MAX_WORKERS", "0")
    small_response = client.post(
        "/api/v1/real/search/runs",
        json={"query": "LLM reranking retrieval papers"},
    )
    assert small_response.status_code == 201
    _wait_for_status(small_response.json()["run_id"], "succeeded")

    assert captured[-2:] == [2, 1]


def test_legacy_mock_api_is_removed_and_does_not_call_search_service(monkeypatch) -> None:
    class FailingSearchService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("legacy mock API must not instantiate SearchService")

    monkeypatch.setattr("scholar_agent.app.api.routes.SearchService", FailingSearchService)

    create_response = client.post(
        "/api/v1/search/runs",
        json={"query": "请帮我搜索关于 LLM reranking 的代表性论文"},
    )
    status_response = client.get("/api/v1/search/runs/run_missing")
    result_response = client.get("/api/v1/search/runs/run_missing/result")
    events_response = client.get("/api/v1/search/runs/run_missing/events")

    assert create_response.status_code in {404, 405}
    assert status_response.status_code in {404, 405}
    assert result_response.status_code in {404, 405}
    assert events_response.status_code in {404, 405}


def test_real_run_cleanup_ttl_removes_old_succeeded_run(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "1")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "0")
        _store_real_run("run_real_old", status="succeeded", age_seconds=10)
        _store_real_run("run_real_new", status="succeeded", age_seconds=0)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == ["run_real_old"]
        with routes._REAL_RUNS_LOCK:
            assert "run_real_old" not in routes._REAL_RUNS
            assert "run_real_new" in routes._REAL_RUNS
    finally:
        _clear_real_runs()


def test_real_run_cleanup_ttl_does_not_remove_running_run(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "1")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "0")
        _store_real_run("run_real_running", status="running", age_seconds=10)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == []
        with routes._REAL_RUNS_LOCK:
            assert "run_real_running" in routes._REAL_RUNS
    finally:
        _clear_real_runs()


def test_real_run_cleanup_max_count_removes_oldest_terminal_runs(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "0")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "2")
        _store_real_run("run_real_running", status="running", age_seconds=100)
        _store_real_run("run_real_old_failed", status="failed", age_seconds=90)
        _store_real_run("run_real_old_cancelled", status="cancelled", age_seconds=80)
        _store_real_run("run_real_new_succeeded", status="succeeded", age_seconds=1)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == ["run_real_old_failed", "run_real_old_cancelled"]
        with routes._REAL_RUNS_LOCK:
            assert set(routes._REAL_RUNS) == {
                "run_real_running",
                "run_real_new_succeeded",
            }
    finally:
        _clear_real_runs()


def test_real_run_cleanup_max_count_keeps_queued_and_running(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "0")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "1")
        _store_real_run("run_real_queued", status="queued", age_seconds=100)
        _store_real_run("run_real_running", status="running", age_seconds=90)
        _store_real_run("run_real_succeeded", status="succeeded", age_seconds=80)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == ["run_real_succeeded"]
        with routes._REAL_RUNS_LOCK:
            assert set(routes._REAL_RUNS) == {
                "run_real_queued",
                "run_real_running",
            }
    finally:
        _clear_real_runs()


def test_real_run_cleanup_invalid_env_falls_back_to_defaults(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "not-an-int")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "not-an-int")
        _store_real_run("run_real_expired", status="succeeded", age_seconds=4000)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == ["run_real_expired"]
        with routes._REAL_RUNS_LOCK:
            assert routes._REAL_RUNS == {}
    finally:
        _clear_real_runs()


def test_real_run_cleanup_nonpositive_env_disables_rules(monkeypatch) -> None:
    _clear_real_runs()
    try:
        monkeypatch.setenv("REAL_SEARCH_RUN_TTL_SECONDS", "0")
        monkeypatch.setenv("REAL_SEARCH_MAX_STORED_RUNS", "-1")
        _store_real_run("run_real_old_one", status="succeeded", age_seconds=4000)
        _store_real_run("run_real_old_two", status="failed", age_seconds=3000)

        deleted = routes._cleanup_real_runs(now=routes._now())

        assert deleted == []
        with routes._REAL_RUNS_LOCK:
            assert set(routes._REAL_RUNS) == {
                "run_real_old_one",
                "run_real_old_two",
            }
    finally:
        _clear_real_runs()


def _wait_for_status(run_id: str, expected_status: str) -> dict[str, object]:
    deadline = time.monotonic() + 3
    latest: dict[str, object] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/real/search/runs/{run_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] == expected_status:
            return latest
        time.sleep(0.03)
    raise AssertionError(f"timed out waiting for {expected_status}; latest={latest}")


def _fake_output(query: str, top_k: int = 5) -> SearchServiceOutput:
    search_plan = _search_plan(query, top_k=top_k)
    high = _ranked(_paper("Real High", doi="10.123/high"), rank=1, final_score=0.91)
    partial = _ranked(
        _paper("Real Partial", doi="10.123/partial"),
        rank=2,
        category="partially_relevant",
        final_score=0.62,
    )
    output = SearchServiceOutput(
        search_plan=search_plan,
        retrieval_outputs=[
            RetrievalOutput(
                query=query,
                requested_sources=["openalex", "arxiv"],
                raw_count=2,
                deduplicated_count=2,
                papers=[high.paper, partial.paper],
                source_stats=[],
                warnings=[],
                latency_seconds=0.01,
            )
        ],
        raw_count=2,
        deduplicated_count=2,
        judgements=[_judgement(high), _judgement(partial)],
        ranked_papers=[high, partial],
        warnings=["real_search_warning"],
        source_stats=[
            SourceStats(
                source="openalex",
                returned_count=1,
                latency_seconds=0.1,
                diagnostics=ConnectorDiagnostics(
                    request_count=1,
                    latency_seconds=0.1,
                ),
            ),
            SourceStats(
                source="arxiv",
                returned_count=1,
                latency_seconds=0.1,
                cache_hit=True,
                diagnostics=ConnectorDiagnostics(cache_hit_count=1),
            ),
        ],
        latency_seconds=0.25,
    )
    output.synthesis_output = synthesize_answer(output)
    return output


def _emit_fake_search_events(event_callback, output: SearchServiceOutput) -> None:
    if event_callback is None:
        return
    event_callback(
        "query_understanding_started",
        {"stage": "query_understanding"},
    )
    event_callback(
        "query_understanding_completed",
        {
            "stage": "query_understanding",
            "subquery_count": len(output.search_plan.subqueries),
            "latency_seconds": 0.01,
        },
    )
    event_callback(
        "retrieval_started",
        {"stage": "retrieval", "round_index": 1, "query_count": 1},
    )
    for stats in output.source_stats:
        event_callback(
            "connector_started",
            {
                "stage": "retrieval",
                "query_index": 0,
                "query": output.search_plan.query_analysis.original_query,
                "connector": stats.source,
                "source": stats.source,
            },
        )
        event_callback(
            "connector_completed",
            {
                "stage": "retrieval",
                "query_index": 0,
                "query": output.search_plan.query_analysis.original_query,
                "connector": stats.source,
                "source": stats.source,
                "returned_count": stats.returned_count,
                "latency_seconds": stats.latency_seconds,
                "request_count": stats.diagnostics.request_count,
                "retry_count": stats.diagnostics.retry_count,
                "error_count": stats.diagnostics.error_count,
                "cache_hit": stats.cache_hit,
                "cache_hit_count": stats.diagnostics.cache_hit_count,
                "rate_limit_wait_seconds": (
                    stats.diagnostics.rate_limit_wait_seconds
                ),
                "error_message": stats.error_message,
            },
        )
    event_callback(
        "retrieval_completed",
        {
            "stage": "retrieval",
            "round_index": 1,
            "raw_candidate_count": output.raw_count,
            "latency_seconds": 0.1,
        },
    )
    event_callback(
        "deduplication_completed",
        {
            "stage": "deduplication",
            "round_index": 1,
            "deduplicated_candidate_count": output.deduplicated_count,
        },
    )
    event_callback(
        "judgement_started",
        {"stage": "judgement", "candidate_paper_count": output.deduplicated_count},
    )
    event_callback(
        "judgement_completed",
        {"stage": "judgement", "judged_paper_count": len(output.judgements)},
    )
    event_callback(
        "reranking_started",
        {"stage": "reranking", "judged_paper_count": len(output.judgements)},
    )
    event_callback(
        "reranking_completed",
        {"stage": "reranking", "ranked_paper_count": len(output.ranked_papers)},
    )
    event_callback(
        "query_evolution_skipped",
        {"stage": "query_evolution", "reason": "disabled"},
    )
    event_callback(
        "refchain_skipped",
        {"stage": "refchain", "reason": "disabled"},
    )
    event_callback(
        "synthesis_started",
        {"stage": "synthesis", "ranked_paper_count": len(output.ranked_papers)},
    )
    event_callback(
        "synthesis_completed",
        {"stage": "synthesis", "status": "completed"},
    )
    for warning in output.warnings:
        event_callback("warning", {"stage": "completed", "message": warning})


def _search_plan(query: str, top_k: int = 5) -> SearchPlan:
    query_analysis = QueryAnalysis(
        original_query=query,
        language="en",
        intent="recent_progress",
        domain="machine_learning",
        constraints=QueryConstraint(
            time_range=TimeRange(start_year=2023, end_year=2026),
            methods=["reranking"],
            must_include_terms=["LLM", "retrieval"],
        ),
    )
    return SearchPlan(
        query_analysis=query_analysis,
        subqueries=[
            SearchSubquery(
                query=query,
                source_hints=["openalex", "arxiv"],
                priority=1,
                purpose="original_query",
            )
        ],
        selected_sources=["openalex", "arxiv"],
        limit_per_source=20,
        top_k=top_k,
        run_profile="balanced",
    )


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in text.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ").strip()
            continue
        if line.startswith("data: "):
            current_data.append(line.removeprefix("data: ").strip())
            continue
        if line == "" and current_event is not None:
            events.append(
                {
                    "event": current_event,
                    "payload": json.loads("".join(current_data) or "{}"),
                }
            )
            current_event = None
            current_data = []
    return events


def _clear_real_runs() -> None:
    with routes._REAL_RUNS_LOCK:
        routes._REAL_RUNS.clear()


def _store_real_run(run_id: str, *, status: str, age_seconds: int) -> None:
    timestamp = routes._now() - timedelta(seconds=age_seconds)
    with routes._REAL_RUNS_LOCK:
        routes._REAL_RUNS[run_id] = routes.RealRun(
            run_id=run_id,
            request=SearchRunCreateRequest(query=f"fixture query {run_id}"),
            status=status,
            current_stage=status,
            progress=RunProgress(),
            cost_report=CostReport(),
            result=None,
            events=[],
            error_message=None,
            cancel_requested=False,
            created_at=timestamp,
            updated_at=timestamp,
        )


def _paper(title: str, *, doi: str) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=2025,
        venue="ACL",
        abstract="A paper about LLM reranking for scientific literature retrieval.",
        identifiers=PaperIdentifiers(doi=doi, openalex_id=f"W-{title.upper()}"),
        sources=["fixture"],
        citation_count=12,
    )


def _ranked(
    paper: Paper,
    *,
    rank: int,
    category: str = "highly_relevant",
    final_score: float,
) -> RankedPaper:
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=final_score,
        category=category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=final_score,
            authority_score=0.5,
            timeliness_score=0.8,
            metadata_score=0.9,
            final_score=final_score,
            relevance_weight=0.65,
            authority_weight=0.1,
            timeliness_weight=0.15,
            metadata_weight=0.1,
        ),
        ranking_reason="Matched the query using fixture metadata.",
        evidence=[
            EvidenceItem(
                source="abstract",
                text="A paper about LLM reranking for scientific literature retrieval.",
                confidence=0.9,
            )
        ],
        matched_terms=["LLM", "reranking", "retrieval"],
    )


def _judgement(ranked: RankedPaper) -> JudgementResult:
    return JudgementResult(
        paper=ranked.paper,
        score=ranked.final_score,
        category=ranked.category,
        reasoning="Fixture judgement.",
        evidence=ranked.evidence,
        matched_terms=ranked.matched_terms,
    )
