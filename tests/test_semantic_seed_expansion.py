from __future__ import annotations

from scholar_agent.agents.semantic_seed_expansion import expand_semantic_seeds
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.dedup import deduplicate_papers_with_audit
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    RankedPaper,
    RerankScoreBreakdown,
)
from scholar_agent.evaluation.snapshots import (
    SnapshotManifest,
    SnapshotRuntime,
    SnapshotStore,
)
from scholar_agent.evaluation.snapshots.store import connector_version, utc_now
from scholar_agent.evaluation.stage_diagnostics import (
    aggregate_stage_diagnostics,
    analyze_search_stages,
)
from scholar_agent.services.search_service import SearchService


def _ranked(
    rank: int,
    *,
    semantic_id: str | None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    pubmed_id: str | None = None,
    corpus_id: str | None = None,
) -> RankedPaper:
    paper = Paper(
        title=f"Seed {rank}",
        authors=["Ada"],
        year=2024,
        identifiers=PaperIdentifiers(
            semantic_scholar_id=semantic_id,
            doi=doi,
            arxiv_id=arxiv_id,
            pubmed_id=pubmed_id,
            s2orc_corpus_id=corpus_id,
        ),
        sources=["semantic_scholar"],
    )
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=0.8,
        category="highly_relevant",
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.8,
            authority_score=0.5,
            timeliness_score=0.5,
            metadata_score=0.5,
            final_score=0.8,
            relevance_weight=0.62,
            authority_weight=0.25,
            timeliness_weight=0.08,
            metadata_weight=0.05,
        ),
        ranking_reason="test",
        evidence=[EvidenceItem(source="title", text=paper.title, confidence=0.8)],
    )


def test_expansion_selects_first_three_unique_exact_s2_ids() -> None:
    captured: list[tuple[list[str], int]] = []
    ranked = [
        _ranked(1, semantic_id=None),
        _ranked(2, semantic_id="S2-A"),
        _ranked(3, semantic_id="s2-a"),
        _ranked(4, semantic_id="S2-B"),
        _ranked(5, semantic_id="S2-C"),
        _ranked(6, semantic_id="S2-D"),
    ]

    output = expand_semantic_seeds(
        ranked,
        lambda seeds, limit: (
            captured.append((list(seeds), limit))
            or ConnectorSearchResult(papers=[Paper(title="Recommendation")])
        ),
    )

    assert captured == [(["S2-A", "S2-B", "S2-C"], 100)]
    assert [seed.rank for seed in output.record.seeds] == [2, 4, 5]
    assert output.record.status == "success"
    assert len(output.recommendations) == 1


def test_expansion_without_s2_seed_does_not_call_source() -> None:
    called = False

    def fetcher(seeds: list[str], limit: int) -> ConnectorSearchResult:
        nonlocal called
        called = True
        return ConnectorSearchResult()

    output = expand_semantic_seeds(
        [_ranked(1, semantic_id=None, doi="10.1/only-doi")],
        fetcher,
    )

    assert called is False
    assert output.recommendations == []
    assert output.record.status == "no_eligible_seed"
    assert output.record.skip_reason == "no_eligible_seed"


def test_expansion_source_failure_preserves_empty_delta() -> None:
    output = expand_semantic_seeds(
        [_ranked(1, semantic_id="S2-A")],
        lambda seeds, limit: ConnectorSearchResult(
            error_message="HTTP 429",
            diagnostics=ConnectorDiagnostics(
                request_count=2,
                retry_count=1,
                error_count=1,
            ),
        ),
    )

    assert output.recommendations == []
    assert output.record.status == "source_failure"
    assert output.record.skip_reason == "source_failure"
    assert output.diagnostics.request_count == 2


def test_expansion_resolves_all_supported_exact_identifier_types_in_rank_order() -> None:
    ranked = [
        _ranked(1, semantic_id=None, doi="https://doi.org/10.1/doi"),
        _ranked(2, semantic_id=None, arxiv_id="ARXIV:2401.00001v2"),
        _ranked(3, semantic_id=None, pubmed_id="PMID:123"),
        _ranked(4, semantic_id=None, corpus_id="CorpusId:456"),
    ]
    resolution_calls: list[list[str]] = []
    recommendation_calls: list[list[str]] = []

    def resolver(identifiers: list[str]) -> ConnectorSearchResult:
        resolution_calls.append(list(identifiers))
        return ConnectorSearchResult(
            papers=[
                Paper(
                    title="DOI mapped",
                    identifiers=PaperIdentifiers(
                        doi="10.1/doi", semantic_scholar_id="S2-1"
                    ),
                ),
                Paper(
                    title="arXiv mapped",
                    identifiers=PaperIdentifiers(
                        arxiv_id="2401.00001", semantic_scholar_id="S2-2"
                    ),
                ),
                Paper(
                    title="PMID mapped",
                    identifiers=PaperIdentifiers(
                        pubmed_id="123", semantic_scholar_id="S2-3"
                    ),
                ),
                Paper(
                    title="Corpus mapped",
                    identifiers=PaperIdentifiers(
                        s2orc_corpus_id="456", semantic_scholar_id="S2-4"
                    ),
                ),
            ],
            diagnostics=ConnectorDiagnostics(request_count=1),
            reference_batch_status="success",
        )

    output = expand_semantic_seeds(
        ranked,
        lambda seeds, limit: (
            recommendation_calls.append(list(seeds)) or ConnectorSearchResult()
        ),
        resolve_seed_ids=resolver,
    )

    assert resolution_calls == [[
        "DOI:10.1/doi",
        "ARXIV:2401.00001",
        "PMID:123",
        "CorpusId:456",
    ]]
    assert recommendation_calls == [["S2-1", "S2-2", "S2-3"]]
    assert [seed.source for seed in output.record.seeds] == [
        "resolved",
        "resolved",
        "resolved",
    ]
    assert output.record.resolved_candidate_count == 4
    assert output.record.resolved_seed_count == 3
    assert output.record.resolution_status == "success"


def test_seed_resolution_rejects_conflicts_and_deduplicates_mapping() -> None:
    ranked = [
        _ranked(
            1,
            semantic_id=None,
            doi="10.1/shared",
            arxiv_id="2401.00001",
        ),
        _ranked(2, semantic_id=None, doi="10.1/shared"),
    ]
    recommendation_calls = 0

    def recommend(seeds: list[str], limit: int) -> ConnectorSearchResult:
        nonlocal recommendation_calls
        recommendation_calls += 1
        return ConnectorSearchResult()

    output = expand_semantic_seeds(
        ranked,
        recommend,
        resolve_seed_ids=lambda identifiers: ConnectorSearchResult(
            papers=[
                Paper(
                    title="Conflicting mapping",
                    identifiers=PaperIdentifiers(
                        doi="10.1/shared",
                        arxiv_id="2401.99999",
                        semantic_scholar_id="S2-CONFLICT",
                    ),
                )
            ],
            reference_batch_status="success",
        ),
    )

    assert recommendation_calls == 0
    assert output.record.status == "no_eligible_seed"
    assert output.record.resolution_conflict_count == 1
    assert output.record.resolution_request_identifier_count == 1
    assert output.record.resolutions[0].identity_rule == "conflicting_stable_identifier"


def test_seed_resolution_partial_success_keeps_valid_exact_mapping() -> None:
    output = expand_semantic_seeds(
        [
            _ranked(1, semantic_id=None, doi="10.1/one"),
            _ranked(2, semantic_id=None, doi="10.1/missing"),
        ],
        lambda seeds, limit: ConnectorSearchResult(),
        resolve_seed_ids=lambda identifiers: ConnectorSearchResult(
            papers=[
                Paper(
                    title="Resolved",
                    identifiers=PaperIdentifiers(
                        doi="10.1/one", semantic_scholar_id="S2-ONE"
                    ),
                )
            ],
            reference_batch_status="partial_success",
            missing_reference_ids=["DOI:10.1/missing"],
        ),
    )

    assert [seed.semantic_scholar_id for seed in output.record.seeds] == ["S2-ONE"]
    assert output.record.resolution_status == "partial_success"
    assert output.record.resolution_missing_count == 1


def test_recommendation_dedup_keeps_duplicate_merged_and_conflict_separate() -> None:
    initial = Paper(
        title="Initial",
        identifiers=PaperIdentifiers(
            semantic_scholar_id="S2-A",
            doi="10.1/a",
        ),
    )
    duplicate = Paper(
        title="Duplicate",
        identifiers=PaperIdentifiers(semantic_scholar_id="s2-a"),
    )
    conflict = Paper(
        title="Conflict",
        identifiers=PaperIdentifiers(
            semantic_scholar_id="S2-A",
            doi="10.1/conflict",
        ),
    )

    papers, audit = deduplicate_papers_with_audit([initial, duplicate, conflict])

    assert len(papers) == 2
    assert len(audit) == 1
    assert audit[0]["rule"] == "shared_stable_identifier"


def test_recommendation_snapshot_record_and_replay_is_deterministic(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    now = utc_now()
    store.ensure_manifest(
        SnapshotManifest(
            snapshot_name=tmp_path.name,
            dataset="auto_scholar_query",
            split="development",
            offset=0,
            limit=1,
            sources=["semantic_scholar"],
            adapter_policy="adaptive",
            run_profile="balanced",
            budgets={"max_search_rounds": 2},
            llm_enabled=False,
            query_understanding_prompt={"name": "query_understanding"},
            judgement_prompt={"name": "relevance_judgement"},
            connector_versions={
                "semantic_scholar": connector_version("semantic_scholar"),
                "semantic_scholar_recommendations": connector_version(
                    "semantic_scholar_recommendations"
                ),
            },
            code_hash="test",
            dirty_worktree=False,
            created_at=now,
            updated_at=now,
        )
    )
    calls = 0

    def fetcher(seeds: list[str], limit: int) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return ConnectorSearchResult(
            papers=[
                Paper(
                    title="Recorded recommendation",
                    identifiers=PaperIdentifiers(semantic_scholar_id="REC-1"),
                )
            ],
            diagnostics=ConnectorDiagnostics(request_count=1),
        )

    record = SnapshotRuntime(
        store,
        mode="record-missing",
        group_name="semantic_seed_expansion",
    )
    record.begin_case("case-1")
    recorded = record.fetch_recommendations(["Seed-A", "seed-a", "Seed-B"], 100, fetcher)
    record.finish_group(completed=True)

    replay = SnapshotRuntime(
        store,
        mode="replay",
        group_name="semantic_seed_expansion",
    )
    replay.begin_case("case-1")
    replayed = replay.fetch_recommendations(
        ["Seed-A", "Seed-B"],
        100,
        lambda seeds, limit: (_ for _ in ()).throw(AssertionError("HTTP called")),
    )

    assert calls == 1
    assert recorded.snapshot_provenance == "snapshot_record"
    assert replayed.snapshot_provenance == "snapshot_replay"
    assert replayed.snapshot_hit is True
    assert replayed.diagnostics.request_count == 0
    assert replayed.recorded_diagnostics == ConnectorDiagnostics(request_count=1)
    assert replayed.papers == recorded.papers
    entry = store.read_reference(str(recorded.snapshot_key))
    assert entry.source == "semantic_scholar"
    assert entry.seed_identifiers == ["Seed-A", "Seed-B"]
    assert entry.connector_version == "recommendations-v1"


def test_seed_resolution_snapshot_record_missing_cache_and_replay(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    now = utc_now()
    store.ensure_manifest(
        SnapshotManifest(
            snapshot_name=tmp_path.name,
            dataset="auto_scholar_query",
            split="development",
            offset=0,
            limit=1,
            sources=["semantic_scholar"],
            adapter_policy="adaptive",
            run_profile="balanced",
            budgets={"max_search_rounds": 2},
            llm_enabled=False,
            query_understanding_prompt={"name": "query_understanding"},
            judgement_prompt={"name": "relevance_judgement"},
            connector_versions={
                "semantic_scholar": connector_version("semantic_scholar"),
                "semantic_scholar_paper_batch": connector_version(
                    "semantic_scholar_paper_batch"
                ),
            },
            code_hash="test",
            dirty_worktree=False,
            created_at=now,
            updated_at=now,
        )
    )
    calls = 0

    def resolver(identifiers: list[str]) -> ConnectorSearchResult:
        nonlocal calls
        calls += 1
        return ConnectorSearchResult(
            papers=[
                Paper(
                    title="Resolved seed",
                    identifiers=PaperIdentifiers(
                        doi="10.1/seed", semantic_scholar_id="S2-SEED"
                    ),
                )
            ],
            diagnostics=ConnectorDiagnostics(request_count=1),
            reference_batch_status="success",
            reference_batch_count=1,
        )

    first = SnapshotRuntime(
        store,
        mode="record-missing",
        group_name="semantic_seed_expansion",
    )
    first.begin_case("case-1")
    recorded = first.resolve_semantic_seed_ids(
        ["DOI:10.1/seed", "doi:10.1/seed"], resolver
    )
    first.finish_group(completed=True)

    cached = SnapshotRuntime(
        store,
        mode="record-missing",
        group_name="semantic_seed_expansion",
    )
    cached.begin_case("case-2")
    cached_result = cached.resolve_semantic_seed_ids(["DOI:10.1/seed"], resolver)

    replay = SnapshotRuntime(
        store,
        mode="replay",
        group_name="semantic_seed_expansion",
    )
    replay.begin_case("case-3")
    replayed = replay.resolve_semantic_seed_ids(
        ["DOI:10.1/seed"],
        lambda identifiers: (_ for _ in ()).throw(AssertionError("HTTP called")),
    )

    assert calls == 1
    assert recorded.snapshot_provenance == "snapshot_record"
    assert cached_result.snapshot_provenance == "snapshot_replay"
    assert replayed.snapshot_provenance == "snapshot_replay"
    assert replayed.diagnostics.request_count == 0
    assert replayed.recorded_diagnostics == ConnectorDiagnostics(request_count=1)
    assert replayed.reference_batch_status == "success"
    entry = store.read_reference(str(recorded.snapshot_key))
    assert entry.seed_identifiers == ["DOI:10.1/seed"]
    assert entry.connector_version == "paper-batch-v1"


def test_search_service_expands_after_unchanged_initial_ranking() -> None:
    initial = Paper(
        title="Graph retrieval systems",
        authors=["Ada"],
        year=2024,
        abstract="Graph retrieval systems for scientific search.",
        identifiers=PaperIdentifiers(semantic_scholar_id="SEED-1"),
        sources=["semantic_scholar"],
    )
    recommended = Paper(
        title="Related graph retrieval method",
        authors=["Bob"],
        year=2023,
        abstract="A related graph retrieval method.",
        identifiers=PaperIdentifiers(semantic_scholar_id="REC-1"),
        sources=["semantic_scholar"],
    )
    recommendation_calls: list[tuple[list[str], int]] = []

    def retrieve(query: str, *, limit_per_source: int, sources: list[str]):
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=1,
            deduplicated_count=1,
            papers=[initial],
            source_stats=[
                SourceStats(
                    source="semantic_scholar",
                    query=query,
                    returned_count=1,
                    diagnostic_papers=[initial],
                )
            ],
        )

    service = SearchService(
        retriever=lambda query, limit_per_source, sources: retrieve(
            query,
            limit_per_source=limit_per_source,
            sources=sources,
        ),
        recommendation_fetcher=lambda seeds, limit: (
            recommendation_calls.append((list(seeds), limit))
            or ConnectorSearchResult(
                papers=[recommended, initial],
                diagnostics=ConnectorDiagnostics(request_count=1),
            )
        ),
        max_workers=1,
    )

    baseline = service.run_search(
        "graph retrieval systems",
        enable_synthesis=False,
        sources_override=["semantic_scholar"],
        collect_diagnostics=True,
    )
    expanded = service.run_search(
        "graph retrieval systems",
        enable_semantic_seed_expansion=True,
        enable_synthesis=False,
        sources_override=["semantic_scholar"],
        collect_diagnostics=True,
    )

    assert recommendation_calls == [(["SEED-1"], 100)]
    assert baseline.semantic_seed_expansion_output is None
    assert expanded.semantic_seed_expansion_output is not None
    assert expanded.semantic_seed_expansion_output.record.status == "success"
    assert expanded.semantic_seed_expansion_output.record.new_unique_candidate_count == 1
    assert expanded.semantic_seed_expansion_output.record.duplicate_candidate_count == 1
    snapshots = {item.stage: item for item in expanded.stage_snapshots}
    assert [item.title for item in snapshots["initial_reranked"].candidates] == [
        "Graph retrieval systems"
    ]
    assert {
        item.title
        for item in snapshots["post_semantic_seed_expansion_deduplicated"].candidates
    } == {"Graph retrieval systems", "Related graph retrieval method"}
    stage_diagnostics = analyze_search_stages(
        EvalQuery(
            query_id="case-1",
            query="graph retrieval systems",
            gold_papers=[
                EvalGoldPaper(
                    title=recommended.title,
                    semantic_scholar_id="REC-1",
                )
            ],
        ),
        expanded,
        result_policy="highly_and_partial",
    )
    assert stage_diagnostics["stage_metrics"]["candidate_recall"][
        "initial_deduplicated"
    ] == 0.0
    assert stage_diagnostics["stage_metrics"]["candidate_recall"][
        "post_semantic_seed_expansion_deduplicated"
    ] == 1.0
    assert stage_diagnostics["semantic_seed_expansion"][
        "new_unique_reference_gold_count"
    ] == 1
    assert stage_diagnostics["semantic_seed_expansion"][
        "candidate_recall_before"
    ] == 0.0
    assert stage_diagnostics["semantic_seed_expansion"][
        "candidate_recall_after"
    ] == 1.0
    assert stage_diagnostics["semantic_seed_expansion"][
        "initial_gold_lost_after_expansion_count"
    ] == 0
    assert stage_diagnostics["semantic_seed_expansion"]["triggered"] is True
    assert stage_diagnostics["semantic_seed_expansion"][
        "seed_with_supported_identifier_count"
    ] == 1
    aggregate, _, _ = aggregate_stage_diagnostics([stage_diagnostics])
    assert aggregate["semantic_seed_expansion"]["triggered_case_count"] == 1
    assert aggregate["semantic_seed_expansion"][
        "seed_with_supported_identifier_count"
    ] == 1


def test_search_service_resolves_seed_without_mutating_initial_candidate() -> None:
    initial = Paper(
        title="Stable DOI seed",
        authors=["Ada"],
        year=2024,
        abstract="Stable seed for scientific retrieval.",
        identifiers=PaperIdentifiers(doi="10.1/seed"),
        sources=["openalex"],
    )
    recommendation_calls: list[list[str]] = []
    resolution_calls: list[list[str]] = []

    def retriever(query: str, limit_per_source: int, sources: list[str]):
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=1,
            deduplicated_count=1,
            papers=[initial],
            source_stats=[SourceStats(source="openalex", returned_count=1)],
        )

    service = SearchService(
        retriever=lambda query, limit_per_source, sources: retriever(
            query, limit_per_source, sources
        ),
        semantic_seed_resolver=lambda identifiers: (
            resolution_calls.append(list(identifiers))
            or ConnectorSearchResult(
                papers=[
                    Paper(
                        title="Official mapping",
                        identifiers=PaperIdentifiers(
                            doi="10.1/seed", semantic_scholar_id="S2-SEED"
                        ),
                    )
                ],
                diagnostics=ConnectorDiagnostics(request_count=1),
                reference_batch_status="success",
            )
        ),
        recommendation_fetcher=lambda seeds, limit: (
            recommendation_calls.append(list(seeds)) or ConnectorSearchResult()
        ),
        max_workers=1,
    )

    baseline = service.run_search(
        "stable seed",
        enable_synthesis=False,
        sources_override=["openalex"],
    )
    expanded = service.run_search(
        "stable seed",
        enable_semantic_seed_expansion=True,
        enable_synthesis=False,
        sources_override=["openalex"],
        collect_diagnostics=True,
    )

    assert resolution_calls == [["DOI:10.1/seed"]]
    assert recommendation_calls == [["S2-SEED"]]
    assert baseline.all_ranked_papers[0].paper.identifiers.semantic_scholar_id is None
    snapshots = {item.stage: item for item in expanded.stage_snapshots}
    assert (
        snapshots["initial_reranked"]
        .candidates[0]
        .identifiers.semantic_scholar_id
        is None
    )
    assert expanded.semantic_seed_expansion_output is not None
    record = expanded.semantic_seed_expansion_output.record
    assert record.resolved_candidate_count == 1
    assert record.resolved_seed_count == 1
    assert record.resolutions[0].shared_identifiers == ["doi:10.1/seed"]


def test_search_service_recommendation_failure_keeps_initial_results() -> None:
    initial = Paper(
        title="Initial stable result",
        authors=["Ada"],
        year=2024,
        identifiers=PaperIdentifiers(semantic_scholar_id="SEED-1"),
        sources=["semantic_scholar"],
    )

    def retriever(query: str, limit_per_source: int, sources: list[str]):
        return RetrievalOutput(
            query=query,
            requested_sources=sources,
            raw_count=1,
            deduplicated_count=1,
            papers=[initial],
            source_stats=[SourceStats(source="semantic_scholar", returned_count=1)],
        )

    service = SearchService(
        retriever=lambda query, limit_per_source, sources: retriever(
            query, limit_per_source, sources
        ),
        recommendation_fetcher=lambda seeds, limit: ConnectorSearchResult(
            error_message="Semantic Scholar recommendations returned 429",
            diagnostics=ConnectorDiagnostics(request_count=2, retry_count=1, error_count=1),
        ),
        max_workers=1,
    )
    baseline = service.run_search(
        "stable result",
        enable_synthesis=False,
        sources_override=["semantic_scholar"],
    )
    expanded = service.run_search(
        "stable result",
        enable_semantic_seed_expansion=True,
        enable_synthesis=False,
        sources_override=["semantic_scholar"],
        collect_diagnostics=True,
    )

    assert expanded.semantic_seed_expansion_output is not None
    assert expanded.semantic_seed_expansion_output.record.status == "source_failure"
    assert [item.paper for item in expanded.all_ranked_papers] == [
        item.paper for item in baseline.all_ranked_papers
    ]
    diagnostics = analyze_search_stages(
        EvalQuery(
            query_id="case-failure",
            query="stable result",
            gold_papers=[
                EvalGoldPaper(
                    title=initial.title,
                    semantic_scholar_id="SEED-1",
                )
            ],
        ),
        expanded,
        result_policy="highly_and_partial",
    )["semantic_seed_expansion"]
    assert diagnostics["candidate_recall_before"] == 1.0
    assert diagnostics["candidate_recall_after"] == 1.0
    assert diagnostics["initial_gold_lost_after_expansion_count"] == 0
