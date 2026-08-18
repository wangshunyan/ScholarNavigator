from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_benchmark
from scripts import collect_benchmark_snapshot_plan as collector_module
from scripts.collect_benchmark_snapshot_plan import collect_plan
from scripts.prepare_benchmark_ablation_snapshots import iterate_group
from scholar_agent.agents import retriever as retriever_module
from scholar_agent.agents.query_evolution import evolve_queries
from scholar_agent.agents.refchain import expand_refchain
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.snapshots import (
    SnapshotManifest,
    SnapshotPlanEntry,
    SnapshotPlanRound,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.snapshots.planning import (
    ABLATION_GROUPS,
    SnapshotCollectionLimits,
    atomic_write_json,
    plan_group_root,
    write_coverage_artifacts,
)
from scholar_agent.evaluation.snapshots.store import (
    connector_version,
    retrieval_snapshot_key,
    utc_now,
)
from scholar_agent.services.search_service import SearchService


def _paper(title: str = "Graph Neural Networks for Molecular Prediction") -> Paper:
    slug = title.casefold().replace(" ", "-")[:24]
    return Paper(
        title=title,
        authors=["Ada Researcher"],
        year=2024,
        abstract=(
            "Graph neural networks are evaluated for molecular property prediction "
            "with benchmark datasets and reproducible methods."
        ),
        citation_count=42,
        identifiers=PaperIdentifiers(
            doi=f"10.1234/{slug}",
            openalex_id=f"W{abs(hash(title)) % 100000}",
        ),
        sources=["openalex"],
    )


def _manifest(root: Path) -> SnapshotManifest:
    now = utc_now()
    return SnapshotManifest(
        snapshot_name=root.name,
        dataset="auto_scholar_query",
        split="development",
        offset=0,
        limit=1,
        sources=["openalex"],
        adapter_policy="adaptive",
        run_profile="balanced",
        budgets={
            "max_search_rounds": 2,
            "max_candidate_papers": 150,
            "max_llm_calls": 4,
            "max_total_tokens": 12000,
            "max_latency_seconds": 90.0,
        },
        llm_enabled=False,
        query_understanding_prompt={"name": "query_understanding"},
        judgement_prompt={"name": "relevance_judgement"},
        connector_versions={
            "openalex": connector_version("openalex"),
            "openalex_references": connector_version(
                "openalex",
                references=True,
            ),
        },
        code_hash="a" * 64,
        dirty_worktree=False,
        created_at=now,
        updated_at=now,
    )


def _store(root: Path) -> SnapshotStore:
    store = SnapshotStore(root)
    if not store.manifest_path.exists():
        store.ensure_manifest(_manifest(root))
    return store


def _plan_entry(
    token: str,
    *,
    source: str = "arxiv",
    priority: int = 1,
    present: bool = False,
    query_evolution_policy: str | None = None,
    query_planning_policy: str | None = None,
    query_planner_version: str | None = None,
) -> SnapshotPlanEntry:
    adapted_query = f"query-{token[:4]}"
    key, _ = retrieval_snapshot_key(
        source=source,
        adapted_query=adapted_query,
        limit=5,
        adapter_policy="adaptive",
        connector_version=connector_version(source),
        query_evolution_policy=query_evolution_policy,
        query_planning_policy=query_planning_policy,
        query_planner_version=query_planner_version,
    )
    return SnapshotPlanEntry(
        key=key,
        entry_type="retrieval",
        source=source,
        adapted_query=adapted_query,
        limit=5,
        adapter_policy="adaptive",
        connector_version=connector_version(source),
        required_by_group="query_evolution_only",
        case_id="case-0",
        stage="query_evolution",
        origin_subquery="graph molecular prediction",
        generated_by="query_evolution",
        query_evolution_policy=query_evolution_policy,  # type: ignore[arg-type]
        query_planning_policy=query_planning_policy,  # type: ignore[arg-type]
        query_planner_version=query_planner_version,
        dependency_keys=[],
        priority=priority,
        already_present=present,
    )


def _write_plan(
    root: Path,
    entries: list[SnapshotPlanEntry],
    *,
    group: str = "query_evolution_only",
    round_index: int = 1,
) -> Path:
    plan = SnapshotPlanRound(
        snapshot_name=root.name,
        group=group,
        round_index=round_index,
        entries=entries,
        missing_retrieval_count=sum(
            item.entry_type == "retrieval" and not item.already_present
            for item in entries
        ),
        missing_reference_count=sum(
            item.entry_type == "reference" and not item.already_present
            for item in entries
        ),
        network_request_count=0,
        converged=all(item.already_present for item in entries),
        created_at=utc_now(),
    )
    path = plan_group_root(root, group) / f"plan_round_{round_index}.json"
    atomic_write_json(path, plan.model_dump(mode="json"))
    return path


def _observe_missing_plan(root: Path, entries: list[SnapshotPlanEntry]) -> None:
    runtime = SnapshotRuntime(
        _store(root),
        mode="plan",
        group_name="query_evolution_only",
    )
    runtime.begin_case("case-0")
    for entry in entries:
        runtime.search(
            entry.source,
            entry.adapted_query or "",
            entry.limit,
            "adaptive",
            lambda query, limit: pytest.fail("plan must not call network"),
            stage=entry.stage,
            origin_subquery=entry.origin_subquery,
            generated_by="query_evolution",
        )
    runtime.finish_group(completed=True)


def test_collector_preserves_coverage_gap_policy_and_key(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entry = _plan_entry(
        "coverage-gap",
        query_evolution_policy="coverage_gap",
    )
    path = _write_plan(
        root,
        [entry],
        group="query_evolution_coverage_gap",
    )

    result = collect_plan(
        path,
        root,
        searchers={
            "arxiv": lambda query, limit: ConnectorSearchResult(
                papers=[_paper()],
                diagnostics=ConnectorDiagnostics(request_count=1),
            )
        },
    )

    assert result["coverage"]["query_evolution_policy"] == "coverage_gap"
    assert SnapshotStore(root).read_retrieval(entry.key).status == "success"


def test_collector_preserves_facet_planning_policy_version_and_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entry = _plan_entry(
        "facet-plan",
        query_planning_policy="facet_balanced",
        query_planner_version="1.0.0",
    )
    path = _write_plan(root, [entry], group="facet_balanced")

    result = collect_plan(
        path,
        root,
        searchers={
            "arxiv": lambda query, limit: ConnectorSearchResult(
                papers=[_paper()],
                diagnostics=ConnectorDiagnostics(request_count=1),
            )
        },
    )

    assert result["coverage"]["query_planning_policy"] == "facet_balanced"
    assert result["coverage"]["query_planner_version"] == "1.0.0"
    assert SnapshotStore(root).read_retrieval(entry.key).status == "success"


def _success(title: str = "Collected Paper") -> ConnectorSearchResult:
    return ConnectorSearchResult(
        papers=[_paper(title)],
        diagnostics=ConnectorDiagnostics(request_count=1, latency_seconds=0.1),
        latency_seconds=0.1,
    )


def _failure() -> ConnectorSearchResult:
    return ConnectorSearchResult(
        error_message="HTTP 429",
        warnings=["http_status:429"],
        diagnostics=ConnectorDiagnostics(request_count=1, error_count=1),
    )


def test_plan_mode_never_calls_live_and_records_structured_missing_key(
    tmp_path: Path,
) -> None:
    runtime = SnapshotRuntime(
        _store(tmp_path / "snapshot"),
        mode="plan",
        group_name="query_evolution_only",
    )
    runtime.begin_case("case-7")
    calls = 0

    def forbidden(query: str, limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return _success()

    result = runtime.search(
        "openalex",
        "graph molecular prediction",
        5,
        "adaptive",
        forbidden,
        stage="query_evolution",
        origin_subquery="graph methods",
        generated_by="query_evolution",
    )
    entry = runtime.plan_entries()[0]

    assert calls == 0
    assert result.snapshot_provenance == "snapshot_plan"
    assert entry.case_id == "case-7"
    assert entry.generated_by == "query_evolution"
    assert entry.already_present is False
    assert runtime.finish_case().replay_execution_request_count == 0


def test_plan_dependency_uses_present_upstream_retrieval_keys(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    record = SnapshotRuntime(_store(root), mode="record", group_name="baseline")
    record.search("openalex", "initial", 5, "adaptive", lambda q, n: _success())
    record.finish_group(completed=True)
    plan = SnapshotRuntime(_store(root), mode="plan", group_name="query_evolution_only")
    plan.begin_case("case-0")
    plan.search("openalex", "initial", 5, "adaptive", lambda q, n: _failure())
    plan.search(
        "openalex",
        "evolved",
        5,
        "adaptive",
        lambda q, n: pytest.fail("network forbidden"),
        generated_by="query_evolution",
    )
    entries = plan.plan_entries()

    assert entries[1].dependency_keys == [entries[0].key]
    assert entries[0].already_present is True
    assert entries[1].already_present is False


def test_reference_plan_waits_until_evolved_retrieval_dependencies_exist(
    tmp_path: Path,
) -> None:
    runtime = SnapshotRuntime(
        _store(tmp_path / "snapshot"),
        mode="plan",
        group_name="query_evolution_plus_refchain",
    )
    runtime.begin_case("case-0")
    runtime.search(
        "openalex",
        "missing evolved",
        5,
        "adaptive",
        lambda q, n: pytest.fail("network forbidden"),
        generated_by="query_evolution",
    )
    result = runtime.fetch_references(
        _paper(),
        5,
        lambda paper, limit: pytest.fail("network forbidden"),
    )

    assert result.error_message == "snapshot_plan_dependency_missing:refchain"
    assert all(item.entry_type != "reference" for item in runtime.plan_entries())


def test_reference_snapshot_preserves_partial_batch_diagnostics(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    record = SnapshotRuntime(_store(root), mode="record", group_name="refchain_only")
    record.begin_case("case-0")

    result = record.fetch_references(
        _paper(),
        5,
        lambda paper, limit: ConnectorSearchResult(
            papers=[_paper()],
            error_message="OpenAlex reference missing work id:W2",
            warnings=["OpenAlex reference supplemental requests:1 (success:0)"],
            diagnostics=ConnectorDiagnostics(request_count=3, error_count=1),
            reference_batch_status="partial_success",
            missing_reference_ids=["W2"],
            reference_batch_count=1,
            supplemental_request_count=1,
        ),
    )
    record.finish_case()

    stored = SnapshotStore(root).read_reference(result.snapshot_key or "")
    assert stored.reference_batch_status == "partial_success"
    assert stored.missing_reference_ids == ["W2"]
    assert stored.reference_batch_count == 1
    assert stored.supplemental_request_count == 1

    replay = SnapshotRuntime(_store(root), mode="replay", group_name="refchain_only")
    replay.begin_case("case-0")
    replayed = replay.fetch_references(
        _paper(),
        5,
        lambda paper, limit: pytest.fail("reference replay must not call network"),
    )
    assert replayed.reference_batch_status == "partial_success"
    assert replayed.missing_reference_ids == ["W2"]
    assert replayed.supplemental_request_count == 1


def test_present_and_failed_snapshot_are_covered_but_not_both_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    record = SnapshotRuntime(_store(root), mode="record", group_name="baseline")
    record.search("openalex", "failed", 5, "adaptive", lambda q, n: _failure())
    observation = record.finish_group(completed=True)
    report = SnapshotStore(root).inspect()["groups"]["baseline"]

    assert observation.replay_ready is True
    assert report["present_failed_entries"] == 1
    assert report["present_success_entries"] == 0
    assert report["missing_entries"] == 0


def test_repeated_plan_is_stable_and_present_key_is_not_missing(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    record = SnapshotRuntime(_store(root), mode="record", group_name="baseline")
    record.search("openalex", "same", 5, "adaptive", lambda q, n: _success())
    record.finish_group(completed=True)
    rows: list[list[dict[str, object]]] = []
    for _ in range(2):
        plan = SnapshotRuntime(_store(root), mode="plan", group_name="baseline")
        plan.begin_case("case-0")
        plan.search("openalex", "same", 5, "adaptive", lambda q, n: _failure())
        plan.finish_group(completed=True)
        rows.append([item.model_dump(mode="json") for item in plan.plan_entries()])

    assert rows[0] == rows[1]
    assert rows[0][0]["already_present"] is True
    assert SnapshotStore(root).inspect()["groups"]["baseline"]["missing_entries"] == 0


def test_collector_records_only_missing_and_preserves_priority(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entries = [
        _plan_entry("b" * 64, source="openalex", priority=2),
        _plan_entry("a" * 64, source="arxiv", priority=1),
    ]
    _observe_missing_plan(root, entries)
    plan_path = _write_plan(root, entries)
    calls: list[str] = []
    result = collect_plan(
        plan_path,
        root,
        searchers={
            "arxiv": lambda q, n: (calls.append("arxiv") or _success()),
            "openalex": lambda q, n: (calls.append("openalex") or _success()),
        },
    )

    assert calls == ["arxiv", "openalex"]
    assert result["collected_entry_count"] == 2
    assert result["covered_success"] == 2
    assert result["missing_entries"] == 0


def test_collector_request_limit_stops_before_second_request(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entries = [_plan_entry("a" * 64), _plan_entry("b" * 64)]
    _observe_missing_plan(root, entries)
    calls: list[str] = []
    result = collect_plan(
        _write_plan(root, entries),
        root,
        max_new_requests=1,
        searchers={"arxiv": lambda q, n: (calls.append(q) or _success())},
    )

    assert len(calls) == 1
    assert result["stop_reason"] == "snapshot_collection_request_limit"
    assert result["missing_entries"] == 1


def test_collector_failure_limit_freezes_failure_and_stops(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entries = [_plan_entry("a" * 64), _plan_entry("b" * 64)]
    _observe_missing_plan(root, entries)
    result = collect_plan(
        _write_plan(root, entries),
        root,
        max_new_failed_entries=1,
        source_failure_limit=3,
        searchers={"arxiv": lambda q, n: _failure()},
    )

    assert result["covered_failed"] == 1
    assert result["stop_reason"] == "snapshot_collection_failure_limit"
    assert result["missing_entries"] == 1


def test_collector_can_disable_openalex_connector_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entry = _plan_entry("a" * 64, source="openalex")
    _observe_missing_plan(root, [entry])
    observed_retries: list[int] = []

    def search(query: str, limit: int, *, max_retries: int) -> ConnectorSearchResult:
        del query, limit
        observed_retries.append(max_retries)
        return _failure()

    monkeypatch.setattr(collector_module, "search_openalex_detailed", search)
    result = collect_plan(
        _write_plan(root, [entry]),
        root,
        openalex_max_retries=0,
    )

    assert observed_retries == [0]
    assert result["covered_failed"] == 1
    assert result["missing_entries"] == 0


def test_collector_time_limit_prevents_first_request(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entry = _plan_entry("a" * 64)
    _observe_missing_plan(root, [entry])
    ticks = iter([0.0, 2.0, 2.0, 2.0])
    result = collect_plan(
        _write_plan(root, [entry]),
        root,
        max_collection_seconds=1.0,
        searchers={"arxiv": lambda q, n: pytest.fail("request must not start")},
        clock=lambda: next(ticks),
    )

    assert result["request_count"] == 0
    assert result["stop_reason"] == "snapshot_collection_time_limit"


def test_source_cooldown_skips_same_source_but_continues_other_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entries = [
        _plan_entry("a" * 64, source="openalex", priority=1),
        _plan_entry("b" * 64, source="openalex", priority=2),
        _plan_entry("c" * 64, source="arxiv", priority=3),
    ]
    _observe_missing_plan(root, entries)
    calls: list[str] = []
    result = collect_plan(
        _write_plan(root, entries),
        root,
        source_failure_limit=1,
        searchers={
            "openalex": lambda q, n: (calls.append("openalex") or _failure()),
            "arxiv": lambda q, n: (calls.append("arxiv") or _success()),
        },
    )

    assert calls == ["openalex", "arxiv"]
    assert result["blocked_sources"] == ["openalex"]
    assert result["stop_reason"] == "snapshot_collection_source_cooldown"


def test_cancelled_collection_resumes_by_skipping_atomic_completed_entry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    entries = [_plan_entry("a" * 64), _plan_entry("b" * 64)]
    _observe_missing_plan(root, entries)
    path = _write_plan(root, entries)
    checks = iter([False, True])
    first = collect_plan(
        path,
        root,
        searchers={"arxiv": lambda q, n: _success()},
        cancel_check=lambda: next(checks),
    )
    calls = 0

    def search(query: str, limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return _success()

    second = collect_plan(path, root, searchers={"arxiv": search})

    assert first["stop_reason"] == "snapshot_collection_cancelled"
    assert calls == 1
    assert second["skipped_present_count"] == 1
    assert second["missing_entries"] == 0


def test_new_snapshot_invalidates_affected_replay_verified_group(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    runtime = SnapshotRuntime(_store(root), mode="record", group_name="baseline")
    runtime.search("openalex", "key", 5, "adaptive", lambda q, n: _failure())
    runtime.finish_group(completed=True)
    replay = SnapshotRuntime(_store(root), mode="replay", group_name="baseline")
    replay.search("openalex", "key", 5, "adaptive", lambda q, n: _success())
    replay.finish_group(completed=True)
    assert _store(root).read_manifest().groups["baseline"].replay_verified is True

    retry = SnapshotRuntime(
        _store(root),
        mode="record-missing",
        group_name="baseline",
        retry_failed_snapshots=True,
    )
    retry.search("openalex", "key", 5, "adaptive", lambda q, n: _success())

    assert _store(root).read_manifest().groups["baseline"].replay_verified is False


def test_fixed_point_iteration_converges_after_replan(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _store(root)

    def planner(group: str, round_index: int) -> Path:
        entries = [] if round_index == 2 else [_plan_entry("a" * 64)]
        return _write_plan(root, entries, round_index=round_index)

    result = iterate_group(
        group="query_evolution_only",
        snapshot_dir=root,
        limits=SnapshotCollectionLimits(max_plan_rounds=2),
        plan_round=planner,
        collect=lambda *args, **kwargs: {
            "request_count": 1,
            "failed_entry_count": 0,
            "stop_reason": None,
        },
    )

    assert result["replay_ready"] is True
    assert result["plan_rounds"] == 2
    assert result["stop_reason"] is None


def test_fixed_point_iteration_marks_nonconvergence(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    runtime = SnapshotRuntime(
        _store(root),
        mode="plan",
        group_name="query_evolution_only",
    )
    runtime.finish_group(completed=True)
    result = iterate_group(
        group="query_evolution_only",
        snapshot_dir=root,
        limits=SnapshotCollectionLimits(max_plan_rounds=1),
        plan_round=lambda group, round_index: _write_plan(
            root,
            [_plan_entry("a" * 64)],
        ),
        collect=lambda *args, **kwargs: {
            "request_count": 1,
            "failed_entry_count": 0,
            "stop_reason": None,
        },
    )

    assert result["stop_reason"] == "snapshot_plan_not_converged"
    assert _store(root).read_manifest().groups[
        "query_evolution_only"
    ].stop_reason == "snapshot_plan_not_converged"


def test_global_coverage_always_contains_four_ablation_groups(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    runtime = SnapshotRuntime(_store(root), mode="record", group_name="baseline")
    runtime.search("openalex", "base", 5, "adaptive", lambda q, n: _success())
    runtime.finish_group(completed=True)
    coverage = write_coverage_artifacts(root, group="baseline", round_index=1)

    assert tuple(coverage) == ABLATION_GROUPS
    assert coverage["baseline"]["replay_ready"] is True
    assert coverage["refchain_only"]["stop_reason"] == "not_planned"


def test_plan_and_snapshot_payload_exclude_sensitive_or_gold_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _store(root)
    path = _write_plan(root, [_plan_entry("a" * 64)])
    serialized = path.read_text(encoding="utf-8")

    assert "gold" not in serialized.casefold()
    assert "prompt" not in serialized.casefold()
    assert "api_key" not in serialized.casefold()
    assert "authorization" not in serialized.casefold()


def test_search_service_query_evolution_and_refchain_signatures_have_no_gold() -> None:
    symbols = (SearchService.run_search, evolve_queries, expand_refchain)
    assert all("gold" not in inspect.signature(symbol).parameters for symbol in symbols)


def test_real_pipeline_discovers_qe_and_refchain_keys_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "qid": "case-0",
                "question": "graph neural networks for molecular prediction",
                "answer": ["Graph Neural Networks for Molecular Prediction"],
                "answer_arxiv_id": ["2401.00001"],
                "source_meta": {"published_time": "20240101"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def baseline_search(query: str, limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return _success("Graph Neural Networks for Molecular Prediction")

    monkeypatch.setattr(retriever_module, "search_openalex_detailed", baseline_search)
    base = run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        dataset_path=dataset,
        limit=1,
        output_root=tmp_path / "runs",
        run_id="record",
        sources=["openalex"],
        max_workers=1,
        retrieval_mode="record",
        snapshot_dir=tmp_path / "snapshot",
    )
    run_benchmark.run_benchmark(base)
    assert calls > 0
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: pytest.fail("plan accessed network"),
    )

    qe = run_benchmark.run_benchmark(
        base.model_copy(
            update={
                "run_id": "qe-plan",
                "retrieval_mode": "plan",
                "enable_query_evolution": True,
                "query_evolution_policy": "seed_expansion",
            }
        )
    )
    ref = run_benchmark.run_benchmark(
        base.model_copy(
            update={
                "run_id": "ref-plan",
                "retrieval_mode": "plan",
                "enable_refchain": True,
            }
        )
    )
    del qe, ref
    qe_plan = SnapshotPlanRound.model_validate_json(
        (plan_group_root(base.snapshot_dir, "query_evolution_only") / "plan_round_1.json").read_text()
    )
    ref_plan = SnapshotPlanRound.model_validate_json(
        (plan_group_root(base.snapshot_dir, "refchain_only") / "plan_round_1.json").read_text()
    )

    assert any(
        item.generated_by == "query_evolution" and not item.already_present
        for item in qe_plan.entries
    )
    assert any(
        item.entry_type == "reference" and not item.already_present
        for item in ref_plan.entries
    )
    assert qe_plan.network_request_count == 0
    assert ref_plan.network_request_count == 0


def test_combined_discovers_refchain_only_after_qe_snapshots_are_collected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "qid": "case-0",
                "question": "graph neural networks for molecular prediction",
                "answer": ["Graph Neural Networks for Molecular Prediction"],
                "answer_arxiv_id": ["2401.00001"],
                "source_meta": {"published_time": "20240101"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: _success("Graph Neural Networks for Molecular Prediction"),
    )
    base = run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        dataset_path=dataset,
        limit=1,
        output_root=tmp_path / "runs",
        run_id="record",
        sources=["openalex"],
        max_workers=1,
        retrieval_mode="record",
        snapshot_dir=tmp_path / "snapshot",
    )
    run_benchmark.run_benchmark(base)
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: pytest.fail("plan accessed network"),
    )
    combined = base.model_copy(
        update={
            "run_id": "combined-plan-1",
            "retrieval_mode": "plan",
                "enable_query_evolution": True,
                "query_evolution_policy": "seed_expansion",
                "enable_refchain": True,
            "plan_round": 1,
        }
    )
    run_benchmark.run_benchmark(combined)
    first_path = plan_group_root(
        base.snapshot_dir,
        "query_evolution_plus_refchain",
    ) / "plan_round_1.json"
    first = SnapshotPlanRound.model_validate_json(first_path.read_text())
    assert any(item.generated_by == "query_evolution" for item in first.entries)
    assert all(item.entry_type != "reference" for item in first.entries)

    collect_plan(
        first_path,
        base.snapshot_dir,
        searchers={"openalex": lambda query, limit: _success("Evolved Graph Method")},
    )
    run_benchmark.run_benchmark(
        combined.model_copy(update={"run_id": "combined-plan-2", "plan_round": 2})
    )
    second = SnapshotPlanRound.model_validate_json(
        (
            plan_group_root(base.snapshot_dir, "query_evolution_plus_refchain")
            / "plan_round_2.json"
        ).read_text()
    )

    references = [item for item in second.entries if item.entry_type == "reference"]
    assert references
    assert all(item.dependency_keys for item in references)


def test_snapshot_plan_collector_cli_loads_project_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[Path] = []
    monkeypatch.setattr(
        collector_module,
        "load_project_env",
        lambda root: loaded.append(Path(root)) or True,
    )
    monkeypatch.setattr(
        collector_module,
        "collect_plan",
        lambda *args, **kwargs: {"stop_reason": None},
    )

    assert collector_module.main(
        [
            "--collect-plan",
            str(tmp_path / "plan.json"),
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
        ]
    ) == 0
    assert loaded == [collector_module.REPO_ROOT]
