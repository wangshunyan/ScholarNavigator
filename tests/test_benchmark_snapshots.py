from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_benchmark
from scripts.inspect_benchmark_snapshot import inspect_snapshot
from scholar_agent.agents import retriever as retriever_module
from scholar_agent.agents.retriever import RetrievalRunContext
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.pipeline_diagnostics import PipelineDiagnosticsCollector
from scholar_agent.core.search_schemas import SearchBudget
from scholar_agent.evaluation.snapshots import (
    SnapshotAwareReferenceFetcher,
    SnapshotAwareRetriever,
    SnapshotConflictError,
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotMissingError,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.snapshots.store import (
    canonical_seed_identifier,
    connector_version,
    entry_content_hash,
    normalize_snapshot_query,
    reference_snapshot_key,
    retrieval_snapshot_key,
    utc_now,
)
from scholar_agent.services.search_service import SearchService
from scholar_agent.services.search_budget import SearchBudgetRuntime


@pytest.fixture(autouse=True)
def no_real_project_env_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI unit tests isolated from the developer's local .env."""

    monkeypatch.setattr(run_benchmark, "load_project_env", lambda root: False)


def _paper(
    title: str = "Snapshot Paper", *, s2orc_corpus_id: str | None = None
) -> Paper:
    return Paper(
        title=title,
        authors=["Snapshot Author"],
        year=2024,
        abstract="deterministic retrieval snapshot",
        identifiers=PaperIdentifiers(
            doi="10.1234/snapshot",
            openalex_id="W123",
            s2orc_corpus_id=s2orc_corpus_id,
        ),
        sources=["openalex"],
    )


def _manifest(root: Path, **updates: object) -> SnapshotManifest:
    now = utc_now()
    manifest = SnapshotManifest(
        snapshot_name=root.name,
        dataset="auto_scholar_query",
        split="development",
        offset=0,
        limit=10,
        sources=["openalex", "arxiv"],
        adapter_policy="adaptive",
        run_profile="balanced",
        budgets={"max_search_rounds": 2},
        llm_enabled=False,
        query_understanding_prompt={
            "name": "query_understanding",
            "version": "1.0.0",
            "hash": "a" * 64,
        },
        judgement_prompt={
            "name": "relevance_judgement",
            "version": "1.0.0",
            "hash": "b" * 64,
        },
        connector_versions={
            "openalex": connector_version("openalex"),
            "arxiv": connector_version("arxiv"),
            "openalex_references": connector_version("openalex", references=True),
        },
        code_hash="c" * 64,
        git_commit="d" * 40,
        dirty_worktree=False,
        created_at=now,
        updated_at=now,
    )
    return manifest.model_copy(update=updates)


def _runtime(
    root: Path,
    mode: str,
    *,
    group: str = "baseline",
    retry_failed: bool = False,
    query_evolution_policy: str = "off",
    query_planning_policy: str = "current_rules",
    query_planner_version: str = "1.0.0",
) -> SnapshotRuntime:
    store = SnapshotStore(root)
    if not store.manifest_path.exists():
        store.ensure_manifest(_manifest(root))
    return SnapshotRuntime(
        store,
        mode=mode,  # type: ignore[arg-type]
        group_name=group,
        retry_failed_snapshots=retry_failed,
        query_evolution_policy=query_evolution_policy,  # type: ignore[arg-type]
        query_planning_policy=query_planning_policy,  # type: ignore[arg-type]
        query_planner_version=query_planner_version,
    )


def _success_result(
    title: str = "Snapshot Paper", *, s2orc_corpus_id: str | None = None
) -> ConnectorSearchResult:
    return ConnectorSearchResult(
        papers=[_paper(title, s2orc_corpus_id=s2orc_corpus_id)],
        diagnostics=ConnectorDiagnostics(
            request_count=2,
            retry_count=1,
            rate_limit_wait_seconds=0.25,
            latency_seconds=0.5,
        ),
        latency_seconds=0.5,
    )


def _record_search(runtime: SnapshotRuntime, result: ConnectorSearchResult | None = None):
    return runtime.search(
        "openalex",
        'title.search:"Graph  Embedding" AND year:2024',
        20,
        "adaptive",
        lambda query, limit: result or _success_result(),
    )


def test_query_normalization_preserves_syntax_quotes_and_order() -> None:
    normalized = normalize_snapshot_query(
        '  title.search:"Graph\tEmbedding"  AND author.id:A1  '
    )
    reversed_query = normalize_snapshot_query(
        'author.id:A1 AND title.search:"Graph Embedding"'
    )

    assert normalized == 'title.search:"Graph Embedding" AND author.id:A1'
    assert normalized != reversed_query


def test_old_snapshot_without_s2orc_field_keeps_original_hash(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    runtime = _runtime(root, "record")
    _record_search(runtime)
    path = next((root / "retrieval").glob("*.json"))
    key = path.stem
    payload = json.loads(path.read_text(encoding="utf-8"))
    for paper in payload["papers"]:
        paper["identifiers"].pop("s2orc_corpus_id", None)
    payload["content_hash"] = entry_content_hash(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restored = SnapshotStore(root).read_retrieval(key)

    assert restored.papers[0].identifiers.s2orc_corpus_id is None


def test_retrieval_key_is_stable_and_has_no_case_or_gold_input() -> None:
    kwargs = {
        "source": "openalex",
        "adapted_query": 'title.search:"Graph Embedding"',
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": connector_version("openalex"),
    }
    first = retrieval_snapshot_key(**kwargs)
    second = retrieval_snapshot_key(**kwargs)

    assert first == second
    assert len(first[0]) == 64
    assert "qid" not in json.dumps(kwargs).casefold()
    assert "gold" not in json.dumps(kwargs).casefold()


def test_coverage_gap_has_distinct_key_while_seed_keeps_legacy_key() -> None:
    kwargs = {
        "source": "arxiv",
        "adapted_query": "graph neural networks QM9",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": connector_version("arxiv"),
    }

    legacy = retrieval_snapshot_key(**kwargs)[0]
    seed = retrieval_snapshot_key(
        **kwargs,
        query_evolution_policy="seed_expansion",
    )[0]
    gap = retrieval_snapshot_key(
        **kwargs,
        query_evolution_policy="coverage_gap",
    )[0]

    assert seed == legacy
    assert gap != legacy


def test_facet_planner_has_versioned_key_while_current_rules_stays_legacy() -> None:
    kwargs = {
        "source": "arxiv",
        "adapted_query": "graph neural networks molecular prediction",
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": connector_version("arxiv"),
    }

    legacy = retrieval_snapshot_key(**kwargs)[0]
    current = retrieval_snapshot_key(
        **kwargs,
        query_planning_policy="current_rules",
        query_planner_version="1.0.0",
    )[0]
    facet_v1 = retrieval_snapshot_key(
        **kwargs,
        query_planning_policy="facet_balanced",
        query_planner_version="1.0.0",
    )[0]
    facet_v2 = retrieval_snapshot_key(
        **kwargs,
        query_planning_policy="facet_balanced",
        query_planner_version="2.0.0",
    )[0]
    controlled_v1 = retrieval_snapshot_key(
        **kwargs,
        query_planning_policy="controlled_relaxation",
        query_planner_version="1.0.0",
    )[0]
    controlled_v2 = retrieval_snapshot_key(
        **kwargs,
        query_planning_policy="controlled_relaxation",
        query_planner_version="2.0.0",
    )[0]

    assert current == legacy
    assert facet_v1 != legacy
    assert facet_v2 != facet_v1
    assert controlled_v1 not in {legacy, facet_v1}
    assert controlled_v2 != controlled_v1


def test_manifest_group_records_query_planning_policy_and_version(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        tmp_path / "snapshot",
        "record",
        group="facet_balanced",
        query_planning_policy="facet_balanced",
        query_planner_version="1.0.0",
    )
    _record_search(runtime)

    observation = runtime.finish_group(completed=True)

    assert observation.query_planning_policy == "facet_balanced"
    assert observation.query_planner_version == "1.0.0"


def test_manifest_group_records_query_evolution_policy(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path / "snapshot",
        "record",
        group="query_evolution_coverage_gap",
        query_evolution_policy="coverage_gap",
    )
    _record_search(runtime)

    observation = runtime.finish_group(completed=True)

    assert observation.query_evolution_policy == "coverage_gap"
    stored = runtime.store.read_manifest().groups[
        "query_evolution_coverage_gap"
    ]
    assert stored.query_evolution_policy == "coverage_gap"


def test_semantic_seed_plan_freezes_initial_retrieval_to_baseline_group(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    baseline = _runtime(root, "record", group="baseline")
    baseline.search(
        "openalex",
        "shared first round",
        5,
        "safe_original",
        lambda query, limit: _success_result("Shared Candidate"),
    )
    baseline_group = baseline.finish_group(completed=True)

    unrelated = _runtime(root, "record", group="prf_v1")
    unrelated.search(
        "openalex",
        "existing but not baseline",
        5,
        "safe_original",
        lambda query, limit: _success_result("Unpaired Candidate"),
    )
    unrelated.finish_group(completed=True)

    plan = _runtime(root, "plan", group="semantic_seed_expansion")
    shared = plan.search(
        "openalex",
        "shared first round",
        5,
        "safe_original",
        lambda query, limit: pytest.fail("plan must not call a connector"),
    )
    unpaired = plan.search(
        "openalex",
        "existing but not baseline",
        5,
        "safe_original",
        lambda query, limit: pytest.fail("plan must not call a connector"),
    )
    semantic_group = plan.finish_group(completed=True)

    assert [entry.key for entry in plan.plan_entries()] == [
        baseline_group.retrieval_keys[0]
    ]
    assert semantic_group.retrieval_keys == baseline_group.retrieval_keys
    assert shared.papers[0].title == "Shared Candidate"
    assert unpaired.papers == []
    assert unpaired.error_message is not None
    assert "snapshot_paired_baseline_not_started" in unpaired.error_message
    assert semantic_group.missing_retrieval_keys == []

    replay = _runtime(root, "replay", group="semantic_seed_expansion")
    assert replay.has_recorded_retrieval_request(
        "openalex", "shared first round", 5, "safe_original"
    )
    assert not replay.has_recorded_retrieval_request(
        "openalex", "existing but not baseline", 5, "safe_original"
    )


def test_judgement_config_is_audited_without_changing_retrieval_keys(
    tmp_path: Path,
) -> None:
    observations = []
    for name, policy, config_hash in (
        ("current", "current_rules", "a" * 64),
        ("calibrated", "calibrated_rules_v1", "b" * 64),
    ):
        root = tmp_path / name
        store = SnapshotStore(root)
        store.ensure_manifest(_manifest(root))
        runtime = SnapshotRuntime(
            store,
            mode="record",
            group_name="baseline",
            judgement_policy=policy,
            judgement_config_hash=config_hash,
        )
        _record_search(runtime)
        observations.append(runtime.finish_group(completed=True))

    current, calibrated = observations
    assert current.retrieval_keys == calibrated.retrieval_keys
    assert current.judgement_policy == "current_rules"
    assert calibrated.judgement_policy == "calibrated_rules_v1"
    assert current.judgement_config_hash != calibrated.judgement_config_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "arxiv"),
        ("adapted_query", '"Embedding Graph"'),
        ("limit", 19),
        ("adapter_policy", "hybrid"),
        ("connector_version", "search-v2"),
    ],
)
def test_retrieval_key_dimensions_do_not_collide(field: str, value: object) -> None:
    base = {
        "source": "openalex",
        "adapted_query": '"Graph Embedding"',
        "limit": 20,
        "adapter_policy": "adaptive",
        "connector_version": connector_version("openalex"),
    }
    changed = {**base, field: value}
    assert retrieval_snapshot_key(**base)[0] != retrieval_snapshot_key(**changed)[0]


def test_reference_key_uses_canonical_seed_and_limit() -> None:
    seed = canonical_seed_identifier(_paper())
    assert seed == "openalex:w123"
    first = reference_snapshot_key(
        seed_identifier=seed,
        limit=10,
        connector_version=connector_version("openalex", references=True),
    )
    second = reference_snapshot_key(
        seed_identifier=seed,
        limit=11,
        connector_version=connector_version("openalex", references=True),
    )
    assert first != second


def test_record_and_replay_return_same_normalized_response_without_live_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    recorded = _record_search(
        _runtime(root, "record"),
        _success_result(s2orc_corpus_id="123456"),
    )
    calls = 0

    def forbidden(query: str, limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        raise AssertionError("network forbidden")

    replay = _runtime(root, "replay")
    replayed = replay.search(
        "openalex",
        'title.search:"Graph  Embedding" AND year:2024',
        20,
        "adaptive",
        forbidden,
    )

    assert calls == 0
    assert replayed.papers == recorded.papers
    assert replayed.papers[0].identifiers.s2orc_corpus_id == "123456"
    assert replayed.snapshot_provenance == "snapshot_replay"
    assert replayed.diagnostics == ConnectorDiagnostics()


def test_replay_missing_key_fails_clearly_and_never_calls_live(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "snapshot", "replay")
    with pytest.raises(SnapshotMissingError, match="snapshot_missing:retrieval"):
        runtime.search(
            "openalex",
            "missing",
            20,
            "adaptive",
            lambda query, limit: pytest.fail("live fallback was called"),
        )
    report = runtime.finish_case()
    assert len(report.missing_retrieval_keys) == 1
    with pytest.raises(Exception, match="snapshot_missing:retrieval"):
        runtime.assert_case_complete()


def test_record_missing_replays_existing_and_records_only_absent_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _record_search(_runtime(root, "record"))
    runtime = _runtime(root, "record-missing")
    calls: list[str] = []
    _record_search(runtime)
    runtime.search(
        "openalex",
        "new query",
        20,
        "adaptive",
        lambda query, limit: (calls.append(query) or _success_result("New Paper")),
    )
    report = runtime.finish_case()

    assert calls == ["new query"]
    assert report.retrieval_snapshot_hits == 1
    assert report.retrieval_snapshot_writes == 1


def test_final_failure_is_replayed_unless_retry_failed_is_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    failed = ConnectorSearchResult(
        error_message="temporary upstream failure",
        diagnostics=ConnectorDiagnostics(request_count=3, retry_count=2, error_count=3),
    )
    _record_search(_runtime(root, "record"), failed)
    calls = 0

    def recovered(query: str, limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return _success_result("Recovered")

    replay_failed = _runtime(root, "record-missing")
    assert _record_search(replay_failed, _success_result()).error_message is not None
    assert calls == 0

    retry = _runtime(root, "record-missing", retry_failed=True)
    result = retry.search(
        "openalex",
        'title.search:"Graph  Embedding" AND year:2024',
        20,
        "adaptive",
        recovered,
    )
    assert result.error_message is None
    assert result.papers[0].title == "Recovered"
    assert calls == 1


def test_record_rejects_content_conflict_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _record_search(_runtime(root, "record"), _success_result("First"))
    runtime = _runtime(root, "record")
    with pytest.raises(SnapshotConflictError, match="snapshot_content_conflict"):
        _record_search(runtime, _success_result("Second"))


def test_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _record_search(_runtime(root, "record"))
    assert not list(root.rglob("*.tmp"))
    assert len(list((root / "retrieval").glob("*.json"))) == 1


def test_tampered_content_hash_and_schema_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _record_search(_runtime(root, "record"))
    path = next((root / "retrieval").glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["papers"][0]["title"] = "Tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="hash_mismatch"):
        SnapshotStore(root).read_retrieval(path.stem)

    payload["schema_version"] = "999"
    payload["content_hash"] = entry_content_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="schema_incompatible"):
        SnapshotStore(root).read_retrieval(path.stem)


def test_manifest_rejects_incompatible_collection_config(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    store = SnapshotStore(root)
    store.ensure_manifest(_manifest(root))
    with pytest.raises(SnapshotConflictError, match="sources"):
        store.ensure_manifest(_manifest(root, sources=["pubmed"]))


def test_manifest_allows_versioned_query_planner_upgrade(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    store = SnapshotStore(root)
    store.ensure_manifest(_manifest(root, query_planner_version="1.0.0"))

    upgraded = store.ensure_manifest(
        _manifest(root, query_planner_version="1.1.0")
    )

    assert upgraded.query_planner_version == "1.1.0"
    assert store.read_manifest().query_planner_version == "1.1.0"


def test_manifest_allows_additive_connector_version_without_invalidating_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    store = SnapshotStore(root)
    original = _manifest(root)
    store.ensure_manifest(original)

    upgraded = store.ensure_manifest(
        original.model_copy(
            update={
                "connector_versions": {
                    **original.connector_versions,
                    "semantic_scholar_recommendations": "recommendations-v1",
                }
            }
        )
    )

    assert upgraded.connector_versions["semantic_scholar_recommendations"] == (
        "recommendations-v1"
    )


def test_manifest_rejects_changed_existing_connector_version(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    store = SnapshotStore(root)
    original = _manifest(root)
    store.ensure_manifest(original)

    with pytest.raises(SnapshotConflictError, match="connector_versions"):
        store.ensure_manifest(
            original.model_copy(
                update={
                    "connector_versions": {
                        **original.connector_versions,
                        "openalex": "search-breaking-v2",
                    }
                }
            )
        )


def test_replay_cost_is_zero_while_recorded_live_cost_is_preserved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _record_search(_runtime(root, "record"))
    replay = _runtime(root, "replay")
    _record_search(replay)
    cost = replay.finish_case()

    assert cost.retrieval_snapshot_hits == 1
    assert cost.replay_execution_request_count == 0
    assert cost.replay_execution_retry_count == 0
    assert cost.replay_execution_network_wait_seconds == 0
    assert cost.recorded_search_request_count == 2
    assert cost.recorded_retry_count == 1
    assert cost.recorded_rate_limit_wait_seconds == pytest.approx(0.25)


def test_reference_record_replay_preserves_compatibility_and_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    record = _runtime(root, "record", group="refchain_only")
    fetcher = SnapshotAwareReferenceFetcher(
        record,
        lambda paper, limit: ConnectorSearchResult(
            papers=[_paper("Reference")],
            diagnostics=ConnectorDiagnostics(request_count=1),
        ),
    )
    recorded = fetcher(_paper(), 5)

    replay = _runtime(root, "replay", group="refchain_only")
    replay_fetcher = SnapshotAwareReferenceFetcher(
        replay,
        lambda paper, limit: pytest.fail("reference network forbidden"),
    )
    replayed = replay_fetcher(_paper(), 5)

    assert replayed.papers == recorded.papers
    assert replayed.snapshot_hit is True
    assert replay.finish_case().reference_snapshot_hits == 1
    assert replayed.diagnostics.request_count == 0


def test_snapshot_payload_has_no_secret_gold_prompt_or_header_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    secret = "secret-value"
    result = ConnectorSearchResult(
        error_message=f"api_key={secret}",
        warnings=[f"Authorization: {secret}"],
        diagnostics=ConnectorDiagnostics(error_count=1),
    )
    _record_search(_runtime(root, "record"), result)
    text = next((root / "retrieval").glob("*.json")).read_text(encoding="utf-8")
    fields = set(json.loads(text))

    assert secret not in text
    assert not fields.intersection({"gold", "qrels", "prompt", "headers", "api_key"})


def test_inspector_reports_entries_failures_cost_and_group_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    runtime = _runtime(root, "record")
    _record_search(runtime)
    runtime.finish_group(completed=True)
    report = inspect_snapshot(root)

    assert report["retrieval_entries"] == 1
    assert report["successful_entries"] == 1
    assert report["request_count_recorded"] == 2
    assert report["groups"]["baseline"]["collection_completed"] is True
    assert report["groups"]["baseline"]["replay_ready"] is True
    assert report["invalid_entries"] == 0


def test_recorded_elapsed_time_can_drive_replay_budget_without_sleeping() -> None:
    runtime = SearchBudgetRuntime(
        SearchBudget(max_latency_seconds=10),
        elapsed_seconds_provider=lambda: 12.0,
    )
    assert runtime.latency_stop_reason() == "budget_stop:max_latency_seconds"
    assert runtime.status().elapsed_seconds >= 12.0


def test_snapshot_aware_retriever_replay_never_invokes_connector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    record = _runtime(root, "record")
    record.search(
        "openalex",
        "snapshot query",
        5,
        "safe_original",
        lambda query, limit: _success_result(),
    )
    replay = _runtime(root, "replay")
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: pytest.fail("connector must not run"),
    )
    output = SnapshotAwareRetriever(replay)(
        "snapshot query",
        limit_per_source=5,
        sources=["openalex"],
        query_adapter_policy="safe_original",
    )

    assert [paper.title for paper in output.papers] == ["Snapshot Paper"]
    assert output.source_stats[0].snapshot_hit is True
    assert output.source_stats[0].diagnostics.request_count == 0


@pytest.mark.parametrize(
    "order",
    [
        [
            ("semantic_scholar", "semantic failed"),
            ("openalex", "openalex success"),
            ("semantic_scholar", "semantic success"),
            ("openalex", "openalex outage"),
        ],
        [
            ("openalex", "openalex outage"),
            ("semantic_scholar", "semantic success"),
            ("openalex", "openalex success"),
            ("semantic_scholar", "semantic failed"),
        ],
    ],
)
def test_snapshot_replay_consumes_recorded_terminals_independent_of_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: list[tuple[str, str]],
) -> None:
    root = tmp_path / "snapshot"
    record = _runtime(root, "record")
    recorded = {
        ("semantic_scholar", "semantic failed"): ConnectorSearchResult(
            error_message="Semantic Scholar search failed: HTTP Error 429: ",
            warnings=["Semantic Scholar search failed: HTTP Error 429:"],
            diagnostics=ConnectorDiagnostics(
                request_count=2,
                retry_count=1,
                error_count=1,
            ),
        ),
        ("semantic_scholar", "semantic success"): _success_result(
            "Semantic Success"
        ),
        ("openalex", "openalex success"): _success_result("OpenAlex Success"),
        ("openalex", "openalex outage"): ConnectorSearchResult(
            error_message="source_outage: OpenAlex HTTP Error 429",
            warnings=["source_outage:openalex"],
            diagnostics=ConnectorDiagnostics(
                request_count=2,
                retry_count=1,
                error_count=1,
            ),
        ),
    }
    for (source, query), result in recorded.items():
        record.search(
            source,
            query,
            5,
            "safe_original",
            lambda adapted_query, limit, *, value=result: value,
        )
    record.finish_group(completed=True)

    live_calls: list[tuple[str, str]] = []

    def forbidden(source: str):
        def call(query: str, limit: int) -> ConnectorSearchResult:
            live_calls.append((source, query))
            raise AssertionError("network forbidden during replay")

        return call

    monkeypatch.setattr(
        retriever_module,
        "search_semantic_scholar_detailed",
        forbidden("semantic_scholar"),
    )
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        forbidden("openalex"),
    )

    runs: list[
        dict[tuple[str, str], tuple[bool, str | None, str | None, str | None]]
    ] = []
    for _ in range(2):
        replay = _runtime(root, "replay")
        retriever = SnapshotAwareRetriever(replay)
        context = RetrievalRunContext()
        observed = {}
        for source, query in order:
            output = retriever(
                query,
                limit_per_source=5,
                sources=[source],
                query_adapter_policy="safe_original",
                run_context=context,
            )
            stats = output.source_stats[0]
            observed[(source, query)] = (
                stats.snapshot_hit,
                stats.error_message,
                output.papers[0].title if output.papers else None,
                stats.terminal_status,
            )
            assert stats.source_skipped_reason is None
            assert stats.diagnostics.request_count == 0
        costs = replay.finish_case()
        assert costs.retrieval_snapshot_hits == len(recorded)
        assert costs.retrieval_snapshot_writes == 0
        assert costs.replay_execution_request_count == 0
        runs.append(observed)

    assert live_calls == []
    assert runs[0] == runs[1]
    assert runs[0][("semantic_scholar", "semantic failed")][1] == (
        "Semantic Scholar search failed: HTTP Error 429: "
    )
    assert runs[0][("semantic_scholar", "semantic failed")][3] == "failed"
    assert runs[0][("semantic_scholar", "semantic success")][2] == (
        "Semantic Success"
    )
    assert runs[0][("semantic_scholar", "semantic success")][3] == "success"
    assert runs[0][("openalex", "openalex success")][2] == "OpenAlex Success"
    assert runs[0][("openalex", "openalex success")][3] == "success"
    assert runs[0][("openalex", "openalex outage")][1] == (
        "source_outage: OpenAlex HTTP Error 429"
    )
    assert runs[0][("openalex", "openalex outage")][3] == "source_outage"


def test_snapshot_replay_reports_missing_key_after_failed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    failed = ConnectorSearchResult(
        error_message="Semantic Scholar search failed: HTTP Error 429: ",
        diagnostics=ConnectorDiagnostics(request_count=2, retry_count=1, error_count=1),
    )
    record = _runtime(root, "record")
    record.search(
        "semantic_scholar",
        "recorded failure",
        5,
        "safe_original",
        lambda query, limit: failed,
    )
    record.search(
        "semantic_scholar",
        "required but missing",
        5,
        "safe_original",
        lambda query, limit: _success_result("Missing Entry"),
    )
    group = record.finish_group(completed=True)
    missing_key = group.retrieval_keys[1]
    (root / "retrieval" / f"{missing_key}.json").unlink()
    monkeypatch.setattr(
        retriever_module,
        "search_semantic_scholar_detailed",
        lambda query, limit: pytest.fail("network forbidden during replay"),
    )
    replay = _runtime(root, "replay")
    retriever = SnapshotAwareRetriever(replay)
    context = RetrievalRunContext()

    failed_output = retriever(
        "recorded failure",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )
    missing_output = retriever(
        "required but missing",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )

    assert failed_output.source_stats[0].snapshot_hit is True
    assert failed_output.source_stats[0].error_message == failed.error_message
    assert missing_output.source_stats[0].snapshot_hit is False
    assert missing_output.source_stats[0].terminal_status == "missing"
    assert "snapshot_missing:retrieval" in (
        missing_output.source_stats[0].error_message or ""
    )
    assert missing_output.source_stats[0].source_skipped_reason is None
    collector = PipelineDiagnosticsCollector(enabled=True)
    collector.register_retrieval(
        "initial_retrieval",
        [failed_output, missing_output],
        origin_kind_by_query={
            "recorded failure": "initial_query",
            "required but missing": "initial_generated_subquery",
        },
    )
    assert [
        call.terminal_status for call in collector.snapshots[0].retrieval_calls
    ] == ["failed", "missing"]
    costs = replay.finish_case()
    assert costs.retrieval_snapshot_hits == 1
    assert costs.retrieval_snapshot_writes == 0
    assert len(costs.missing_retrieval_keys) == 1


def test_snapshot_replay_marks_nonfrozen_path_without_creating_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    failed = ConnectorSearchResult(
        error_message="Semantic Scholar search failed: HTTP Error 429: ",
        diagnostics=ConnectorDiagnostics(request_count=2, retry_count=1, error_count=1),
    )
    record = _runtime(root, "record")
    record.search(
        "semantic_scholar",
        "frozen failure",
        5,
        "safe_original",
        lambda query, limit: failed,
    )
    record.finish_group(completed=True)
    monkeypatch.setattr(
        retriever_module,
        "search_semantic_scholar_detailed",
        lambda query, limit: pytest.fail("network forbidden during replay"),
    )
    replay = _runtime(root, "replay")
    retriever = SnapshotAwareRetriever(replay)
    context = RetrievalRunContext()

    retriever(
        "frozen failure",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )
    unrecorded = retriever(
        "path not present in frozen group",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )

    stats = unrecorded.source_stats[0]
    assert stats.terminal_status == "not_started"
    assert stats.logical_call_executed is False
    assert stats.source_skipped_reason == "snapshot_key_not_recorded"
    assert stats.snapshot_hit is False
    costs = replay.finish_case()
    assert costs.retrieval_snapshot_hits == 1
    assert costs.retrieval_snapshot_writes == 0
    assert costs.missing_retrieval_keys == []


def test_snapshot_record_keeps_live_cooldown_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def rate_limited(query: str, limit: int) -> ConnectorSearchResult:
        calls.append(query)
        return ConnectorSearchResult(
            error_message="Semantic Scholar search failed: HTTP Error 429: ",
            diagnostics=ConnectorDiagnostics(request_count=2, retry_count=1, error_count=1),
        )

    monkeypatch.setattr(
        retriever_module,
        "search_semantic_scholar_detailed",
        rate_limited,
    )
    record = _runtime(tmp_path / "snapshot", "record")
    retriever = SnapshotAwareRetriever(record)
    context = RetrievalRunContext()

    first = retriever(
        "first live record",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )
    second = retriever(
        "second live record",
        limit_per_source=5,
        sources=["semantic_scholar"],
        query_adapter_policy="safe_original",
        run_context=context,
    )

    assert calls == ["first live record"]
    assert first.source_stats[0].snapshot_hit is False
    assert second.source_stats[0].source_skipped_reason == "run_rate_limit_cooldown"
    assert record.finish_case().retrieval_snapshot_writes == 1


def test_four_ablation_groups_can_share_one_manifest(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    store = SnapshotStore(root)
    store.ensure_manifest(_manifest(root))
    groups = {
        run_benchmark._ablation_group_name(  # noqa: SLF001
            run_benchmark.BenchmarkRunOptions(
                dataset="auto_scholar_query",
                    run_id=f"group-{qe}-{ref}",
                    enable_query_evolution=qe,
                    query_evolution_policy="seed_expansion",
                    enable_refchain=ref,
            )
        )
        for qe in (False, True)
        for ref in (False, True)
    }
    assert groups == {
        "baseline",
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    }


def test_runner_record_then_replay_is_offline_and_output_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "qid": "case-0",
                "question": "snapshot query",
                "answer": ["Snapshot Paper"],
                "answer_arxiv_id": ["2401.00001"],
                "source_meta": {"published_time": "20240101"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: (calls.append(query) or _success_result()),
    )
    base = run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        dataset_path=dataset,
        limit=1,
        output_root=tmp_path / "runs",
        run_id="record",
        sources=["openalex"],
        max_workers=1,
        query_adapter_policy="safe_original",
        retrieval_mode="record",
        snapshot_dir=tmp_path / "snapshot",
    )
    recorded = run_benchmark.run_benchmark(base)
    assert calls

    monkeypatch.setattr(
        retriever_module,
        "search_openalex_detailed",
        lambda query, limit: pytest.fail("network forbidden during replay"),
    )
    replayed = run_benchmark.run_benchmark(
        base.model_copy(update={"run_id": "replay", "retrieval_mode": "replay"})
    )

    assert replayed.result_rows[0]["status"] == "succeeded"
    assert replayed.result_rows[0]["result"][
        "highly_relevant_papers"
    ] == recorded.result_rows[0]["result"]["highly_relevant_papers"]
    assert replayed.result_rows[0]["result"][
        "partially_relevant_papers"
    ] == recorded.result_rows[0]["result"]["partially_relevant_papers"]
    costs = replayed.metrics["snapshot_costs"]
    assert costs["retrieval_snapshot_hits"] > 0
    assert costs["replay_execution_request_count"] == 0
    assert replayed.result_rows[0]["cost_report"]["search_api_call_count"] == 0


def test_cli_exposes_all_snapshot_modes_and_controls() -> None:
    parser = run_benchmark._parser()  # noqa: SLF001
    parsed = parser.parse_args(
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "snapshot-cli",
            "--retrieval-mode",
            "record-missing",
            "--snapshot-dir",
            "snapshots/example",
            "--retry-failed-snapshots",
            "--overwrite-snapshots",
        ]
    )
    assert parsed.retrieval_mode == "record-missing"
    assert parsed.snapshot_dir == "snapshots/example"
    assert parsed.retry_failed_snapshots is True
    assert parsed.overwrite_snapshots is True


def test_replay_cli_returns_nonzero_when_any_case_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_benchmark,
        "run_benchmark",
        lambda options: SimpleNamespace(
            run_dir=Path("offline-run"),
            result_rows=[{"status": "failed"}],
        ),
    )
    code = run_benchmark.main(
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "incomplete-replay",
            "--retrieval-mode",
            "replay",
            "--snapshot-dir",
            "snapshot",
        ]
    )
    assert code == 2


def test_benchmark_cli_bootstraps_project_env_before_running(monkeypatch) -> None:
    loaded_roots: list[Path] = []

    def fake_load_project_env(root) -> bool:  # noqa: ANN001
        loaded_roots.append(Path(root))
        return True

    monkeypatch.setattr(run_benchmark, "load_project_env", fake_load_project_env)
    monkeypatch.setattr(
        run_benchmark,
        "run_benchmark",
        lambda options: SimpleNamespace(
            run_dir=Path("offline-run"),
            result_rows=[{"status": "succeeded"}],
        ),
    )

    code = run_benchmark.main(
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "env-bootstrap",
        ]
    )

    assert code == 0
    assert loaded_roots == [run_benchmark.REPO_ROOT]
