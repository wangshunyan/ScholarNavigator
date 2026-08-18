from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_benchmark
from scholar_agent.agents import retriever as retriever_module
from scholar_agent.agents.llm_query_planning import LLMPlanningRequest
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.llm_planning_snapshots import (
    LLMPlanningSnapshotRuntime,
    LLMPlanningSnapshotStore,
    llm_planning_snapshot_key,
)
from scholar_agent.evaluation.snapshots import (
    SnapshotAwareReferenceFetcher,
    SnapshotAwareRetriever,
    SnapshotManifest,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.snapshots.store import (
    SnapshotMissingError,
    connector_version,
    utc_now,
)
from scholar_agent.evaluation.snapshots.schemas import SnapshotPlanEntry, SnapshotPlanRound
from scholar_agent.prompts.loader import load_prompt, render_messages
from scholar_agent.services.search_service import SearchService


def _response() -> dict[str, object]:
    return {
        "intent_summary": "graph retrieval",
        "facets": [],
        "supplemental_queries": [
            {
                "query": "graph representation learning retrieval benchmark",
                "purpose": "terminology expansion",
                "covered_facets": ["topic"],
                "retained_must_have_terms": [],
                "terminology_expansions": [],
            }
        ],
        "warnings": [],
    }


def _request(**updates: object) -> LLMPlanningRequest:
    prompt = load_prompt("llm_query_planning")
    values = {
        "provider": "test_provider",
        "model": "semantic-v1",
        "base_url_host": "llm.example.test",
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_hash": prompt.content_hash,
        "input_payload": {
            "original_query": "graph retrieval",
            "explicit_constraints": None,
            "rule_analysis": {"facets": [{"facet_type": "topic", "terms": ["graph"]}]},
            "run_profile": "balanced",
            "max_supplemental_queries": 2,
        },
        "run_profile": "balanced",
        "max_supplemental_queries": 2,
        "temperature": 0,
        "max_tokens": 512,
    }
    values.update(updates)
    return LLMPlanningRequest.model_validate(values)


class Client:
    provider = "test_provider"
    model = "semantic-v1"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001, ARG002
        self.calls += 1
        assert temperature == 0
        if self.error is not None:
            raise self.error
        self.token_usage.prompt_tokens += 13
        self.token_usage.completion_tokens += 8
        self.token_usage.total_tokens += 21
        return deepcopy(_response())


def _execute(runtime: LLMPlanningSnapshotRuntime, request: LLMPlanningRequest, client):  # noqa: ANN001, ANN202
    runtime.begin_case("case-1")
    return runtime.execute(
        request,
        render_messages(request.prompt_name, request.input_payload),
        client,
        timeout=10,
    )


def test_llm_planning_snapshot_key_is_stable() -> None:
    request = _request()

    assert llm_planning_snapshot_key(request) == llm_planning_snapshot_key(request)


def test_llm_planning_snapshot_key_is_namespaced_by_policy() -> None:
    semantic_key, _ = llm_planning_snapshot_key(_request())
    rewrite_key, _ = llm_planning_snapshot_key(
        _request(query_planning_policy="llm_constrained_rewrite")
    )

    assert semantic_key != rewrite_key


@pytest.mark.parametrize(
    "field_update",
    [
        {"model": "semantic-v2"},
        {"prompt_hash": "a" * 64},
        {
            "input_payload": {
                "original_query": "graph retrieval",
                "explicit_constraints": {"venues": ["ACL"]},
                "rule_analysis": {"facets": []},
                "run_profile": "balanced",
                "max_supplemental_queries": 2,
            }
        },
    ],
)
def test_llm_planning_snapshot_key_changes_with_semantic_inputs(
    field_update: dict[str, object],
) -> None:
    original_key, _ = llm_planning_snapshot_key(_request())
    changed_key, _ = llm_planning_snapshot_key(_request(**field_update))

    assert changed_key != original_key


def test_record_writes_success_snapshot_and_cost(tmp_path: Path) -> None:
    runtime = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path),
        mode="record",
    )
    client = Client()
    execution = _execute(runtime, _request(), client)
    report = runtime.finish_case()

    assert execution.snapshot_status == "record"
    assert execution.total_tokens == 21
    assert client.calls == 1
    assert report.snapshot_writes == 1
    assert report.live_call_count == 1


def test_replay_reads_snapshot_without_live_call(tmp_path: Path) -> None:
    request = _request()
    record = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path), mode="record"
    )
    _execute(record, request, Client())
    replay = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path), mode="replay"
    )

    execution = _execute(replay, request, None)
    report = replay.finish_case()

    assert execution.replayed is True
    assert execution.llm_call_attempted is False
    assert report.snapshot_hits == 1
    assert report.replay_execution_request_count == 0
    assert report.replay_execution_network_wait_seconds == 0


def test_replay_missing_never_calls_network_and_emits_plan_entry(tmp_path: Path) -> None:
    runtime = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path), mode="replay", group_name="llm_semantic"
    )
    request = _request()

    with pytest.raises(SnapshotMissingError, match="llm_planning_snapshot_missing"):
        _execute(runtime, request, None)

    entries = runtime.plan_entries()
    assert len(entries) == 1
    assert entries[0].entry_type == "llm_planning"
    assert entries[0].dependency_keys == []
    assert runtime.finish_case().replay_execution_request_count == 0

    runtime.begin_case("case-2")
    plan = analyze_query(
        "graph retrieval",
        query_planning_policy="llm_semantic",
        llm_planning_runtime=runtime,
    )
    assert plan.query_planning.fallback_reason == "snapshot_missing"
    assert plan.query_planning.snapshot_status == "missing"
    assert plan.query_planning.snapshot_key is not None


def test_record_missing_replays_existing_success_only(tmp_path: Path) -> None:
    store = LLMPlanningSnapshotStore(tmp_path)
    request = _request()
    _execute(LLMPlanningSnapshotRuntime(store, mode="record"), request, Client())
    client = Client()
    runtime = LLMPlanningSnapshotRuntime(store, mode="record-missing")

    execution = _execute(runtime, request, client)

    assert execution.replayed is True
    assert client.calls == 0


def test_record_missing_recovers_failed_entry(tmp_path: Path) -> None:
    store = LLMPlanningSnapshotStore(tmp_path)
    request = _request()
    failing = LLMPlanningSnapshotRuntime(store, mode="record")
    with pytest.raises(RuntimeError):
        _execute(failing, request, Client(error=RuntimeError("temporary")))
    recovery_client = Client()
    recovery = LLMPlanningSnapshotRuntime(store, mode="record-missing")

    execution = _execute(recovery, request, recovery_client)

    assert execution.snapshot_status == "record"
    assert recovery_client.calls == 1
    key, _ = llm_planning_snapshot_key(request)
    assert store.read(key).status == "success"


def test_live_mode_never_writes_snapshot(tmp_path: Path) -> None:
    runtime = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path), mode="live"
    )
    _execute(runtime, _request(), Client())

    assert not (tmp_path / "llm_planning").exists()


def test_snapshot_contains_no_secret_gold_or_full_prompt(tmp_path: Path) -> None:
    store = LLMPlanningSnapshotStore(tmp_path)
    request = _request()
    _execute(LLMPlanningSnapshotRuntime(store, mode="record"), request, Client())
    path = next((tmp_path / "llm_planning").glob("*.json"))
    text = path.read_text(encoding="utf-8")

    assert "api_key" not in text.casefold()
    assert "authorization" not in text.casefold()
    assert "gold" not in text.casefold()
    assert "qrels" not in text.casefold()
    assert "你是学术检索" not in text
    assert "input_payload" not in text


def test_snapshot_content_hash_is_verified(tmp_path: Path) -> None:
    store = LLMPlanningSnapshotStore(tmp_path)
    request = _request()
    _execute(LLMPlanningSnapshotRuntime(store, mode="record"), request, Client())
    path = next((tmp_path / "llm_planning").glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    key, _ = llm_planning_snapshot_key(request)
    with pytest.raises(Exception, match="hash_mismatch"):
        store.read(key)


def test_query_planning_record_then_replay_is_fully_offline(tmp_path: Path) -> None:
    store = LLMPlanningSnapshotStore(tmp_path)
    record_runtime = LLMPlanningSnapshotRuntime(store, mode="record")
    record_runtime.begin_case("case-1")
    client = Client()
    recorded = analyze_query(
        "graph retrieval",
        query_planning_policy="llm_semantic",
        llm_client=client,
        llm_planning_runtime=record_runtime,
    )
    replay_runtime = LLMPlanningSnapshotRuntime(store, mode="replay")
    replay_runtime.begin_case("case-1")
    replayed = analyze_query(
        "graph retrieval",
        query_planning_policy="llm_semantic",
        llm_client=None,
        llm_planning_runtime=replay_runtime,
    )

    assert client.calls == 1
    assert recorded.subqueries == replayed.subqueries
    assert replayed.query_planning.replayed is True
    assert replayed.query_planning.llm_call_attempted is False
    assert replay_runtime.finish_case().replay_execution_request_count == 0


def test_search_service_double_replay_is_fully_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_root = tmp_path / "retrieval"
    retrieval_store = SnapshotStore(retrieval_root)
    now = utc_now()
    retrieval_store.ensure_manifest(
        SnapshotManifest(
            snapshot_name="double-replay",
            dataset="auto_scholar_query",
            split="test",
            offset=0,
            limit=1,
            sources=["arxiv"],
            adapter_policy="adaptive",
            run_profile="balanced",
            budgets={},
            llm_enabled=True,
            query_understanding_prompt={},
            llm_query_planning_prompt={},
            judgement_prompt={},
            connector_versions={"arxiv": connector_version("arxiv")},
            code_hash="test",
            git_commit="test",
            dirty_worktree=False,
            created_at=now,
            updated_at=now,
        )
    )

    paper = Paper(
        title="Graph Retrieval with Representation Learning",
        authors=["Test Author"],
        year=2024,
        abstract="Graph representation learning for retrieval benchmarks.",
        identifiers=PaperIdentifiers(arxiv_id="2401.00001"),
        sources=["arxiv"],
    )
    monkeypatch.setattr(
        retriever_module,
        "search_arxiv_detailed",
        lambda query, limit: ConnectorSearchResult(papers=[paper]),
    )
    retrieval_record = SnapshotRuntime(
        retrieval_store,
        mode="record",
        group_name="llm_semantic",
        query_planning_policy="llm_semantic",
        query_planner_version="1.3.0",
    )
    llm_store = LLMPlanningSnapshotStore(tmp_path / "llm")
    llm_record = LLMPlanningSnapshotRuntime(
        llm_store,
        mode="record",
        group_name="llm_semantic",
    )
    retrieval_record.begin_case("case-1")
    llm_record.begin_case("case-1")
    recorded = SearchService(
        retriever=SnapshotAwareRetriever(retrieval_record),
        reference_fetcher=SnapshotAwareReferenceFetcher(
            retrieval_record,
            lambda paper, limit: pytest.fail("reference network forbidden"),
        ),
        llm_client=Client(),
        llm_planning_runtime=llm_record,
        max_workers=1,
    ).run_search(
        "graph retrieval",
        top_k=5,
        sources_override=["arxiv"],
        query_planning_policy="llm_semantic",
        enable_query_evolution=False,
        enable_refchain=False,
        enable_synthesis=False,
    )

    def network_forbidden(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        pytest.fail("network forbidden during double replay")

    for name in (
        "search_arxiv_detailed",
        "search_openalex_detailed",
        "search_semantic_scholar_detailed",
        "search_pubmed_detailed",
    ):
        monkeypatch.setattr(retriever_module, name, network_forbidden)

    class ForbiddenLLM:
        provider = "test_provider"
        model = "semantic-v1"

        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            pytest.fail("LLM network forbidden during replay")

    retrieval_replay = SnapshotRuntime(
        retrieval_store,
        mode="replay",
        group_name="llm_semantic",
        query_planning_policy="llm_semantic",
        query_planner_version="1.3.0",
    )
    llm_replay = LLMPlanningSnapshotRuntime(
        llm_store,
        mode="replay",
        group_name="llm_semantic",
    )
    retrieval_replay.begin_case("case-1")
    llm_replay.begin_case("case-1")
    replayed = SearchService(
        retriever=SnapshotAwareRetriever(retrieval_replay),
        reference_fetcher=SnapshotAwareReferenceFetcher(
            retrieval_replay,
            network_forbidden,
        ),
        llm_client=ForbiddenLLM(),
        llm_planning_runtime=llm_replay,
        max_workers=1,
    ).run_search(
        "graph retrieval",
        top_k=5,
        sources_override=["arxiv"],
        query_planning_policy="llm_semantic",
        enable_query_evolution=False,
        enable_refchain=False,
        enable_synthesis=False,
    )

    assert replayed.search_plan.subqueries == recorded.search_plan.subqueries
    assert replayed.ranked_papers == recorded.ranked_papers
    llm_cost = llm_replay.finish_case()
    retrieval_cost = retrieval_replay.finish_case()
    assert llm_cost.replay_execution_request_count == 0
    assert llm_cost.replay_execution_retry_count == 0
    assert llm_cost.replay_execution_network_wait_seconds == 0
    assert retrieval_cost.replay_execution_request_count == 0
    assert retrieval_cost.replay_execution_retry_count == 0
    assert retrieval_cost.replay_execution_network_wait_seconds == 0


def test_dynamic_plan_freezes_llm_before_retrieval_and_records_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        limit=1,
        output_root=tmp_path / "runs",
        run_id="plan",
        sources=["arxiv"],
        query_planning_policy="llm_semantic",
        retrieval_mode="plan",
        snapshot_dir=tmp_path / "retrieval-snapshot",
        llm_mode="replay",
        llm_snapshot_dir=tmp_path / "llm-snapshot",
    )
    llm_entry = SnapshotPlanEntry(
        key="a" * 64,
        entry_type="llm_planning",
        source="llm",
        limit=0,
        connector_version="1",
        required_by_group="llm_semantic",
        case_id="case-1",
        stage="llm_query_planning",
        generated_by="llm_query_planning",
        query_planning_policy="llm_semantic",
        query_planner_version="1.0.0",
        priority=1,
        llm_request=_request().model_dump(mode="json"),
    )
    retrieval_entry = SnapshotPlanEntry(
        key="b" * 64,
        entry_type="retrieval",
        source="arxiv",
        adapted_query="graph retrieval",
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
        required_by_group="llm_semantic",
        case_id="case-1",
        stage="initial_retrieval",
        generated_by="initial_retrieval",
        query_planning_policy="llm_semantic",
        query_planner_version="1.3.0",
        priority=2,
    )

    class RetrievalPlan:
        def plan_entries(self):  # noqa: ANN202
            return [retrieval_entry]

    class LLMPlan:
        def __init__(self, missing: bool) -> None:
            self.missing = missing

        def plan_entries(self):  # noqa: ANN202
            return [llm_entry] if self.missing else []

        def dependency_keys(self, case_id: str) -> list[str]:
            assert case_id == "case-1"
            return [llm_entry.key]

    monkeypatch.setattr(run_benchmark, "write_coverage_artifacts", lambda *a, **k: {})
    run_benchmark._write_snapshot_plan_artifacts(  # noqa: SLF001
        options,
        RetrievalPlan(),  # type: ignore[arg-type]
        LLMPlan(True),  # type: ignore[arg-type]
    )
    path = (
        options.snapshot_dir
        / "plans"
        / "llm_semantic"
        / "plan_round_1.json"
    )
    first = SnapshotPlanRound.model_validate_json(path.read_text())
    assert [entry.entry_type for entry in first.entries] == ["llm_planning"]

    run_benchmark._write_snapshot_plan_artifacts(  # noqa: SLF001
        options,
        RetrievalPlan(),  # type: ignore[arg-type]
        LLMPlan(False),  # type: ignore[arg-type]
    )
    second = SnapshotPlanRound.model_validate_json(path.read_text())
    assert [entry.entry_type for entry in second.entries] == ["retrieval"]
    assert second.entries[0].dependency_keys == [llm_entry.key]
