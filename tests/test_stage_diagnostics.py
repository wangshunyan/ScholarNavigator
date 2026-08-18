from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_benchmark_runs import compare_runs  # noqa: E402
from scholar_agent.agents.retriever import (  # noqa: E402
    QueryAdaptationProvenance,
    RetrievalOutput,
    SourceStats,
)
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics  # noqa: E402
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery  # noqa: E402
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers  # noqa: E402
from scholar_agent.core.pipeline_diagnostics import (  # noqa: E402
    CandidateProvenance,
    DiagnosticCandidate,
    PipelineDiagnosticsCollector,
    StageCandidateSnapshot,
)
from scholar_agent.core.search_schemas import (  # noqa: E402
    EvidenceItem,
    QueryAnalysis,
    QueryConstraint,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
)
from scholar_agent.evaluation.stage_diagnostics import (  # noqa: E402
    aggregate_stage_diagnostics,
    analyze_search_stages,
    classify_bottlenecks,
)
from scholar_agent.services.search_service import SearchServiceOutput  # noqa: E402


def test_provenance_merges_across_sources_and_subqueries() -> None:
    collector = PipelineDiagnosticsCollector(True)
    paper = _paper("Shared", "2401.00001", source="arxiv")
    openalex_copy = paper.model_copy(update={"sources": ["openalex"]})
    outputs = [
        _retrieval("original", paper, "arxiv"),
        _retrieval("expanded", openalex_copy, "openalex"),
    ]

    collector.register_retrieval(
        "initial_retrieval",
        outputs,
        origin_kind_by_query={
            "original": "initial_query",
            "expanded": "initial_generated_subquery",
        },
    )

    candidate = collector.snapshots[0].candidates[0]
    assert candidate.sources == ["arxiv", "openalex"]
    assert {(item.origin_kind, item.source) for item in candidate.provenance} == {
        ("initial_query", "arxiv"),
        ("initial_generated_subquery", "openalex"),
    }


def test_run_dedupe_preserves_adaptation_provenance() -> None:
    collector = PipelineDiagnosticsCollector(True)
    paper = _paper("Shared", "2401.00001", source="openalex")
    first = RetrievalOutput(
        query="long natural language query",
        requested_sources=["openalex"],
        raw_count=1,
        deduplicated_count=1,
        papers=[paper],
        source_stats=[
            SourceStats(
                source="openalex",
                query="long natural language query",
                adapted_query="graph retrieval",
                adaptation_strategy="openalex_sanitized_core_terms",
                query_provenance=[
                    QueryAdaptationProvenance(
                        origin_subquery="long natural language query",
                        adaptation_strategy="compact_core",
                        purpose="original_query",
                    )
                ],
                returned_count=1,
                diagnostic_papers=[paper],
                diagnostics=ConnectorDiagnostics(request_count=1),
            )
        ],
    )
    duplicate = RetrievalOutput(
        query="graph retrieval",
        requested_sources=["openalex"],
        raw_count=0,
        deduplicated_count=0,
        source_stats=[
            SourceStats(
                source="openalex",
                query="graph retrieval",
                adapted_query="graph retrieval",
                adaptation_strategy="openalex_sanitized_core_terms",
                query_provenance=[
                    QueryAdaptationProvenance(
                        origin_subquery="long natural language query",
                        adaptation_strategy="compact_core",
                        purpose="original_query",
                    ),
                    QueryAdaptationProvenance(
                        origin_subquery="graph retrieval",
                        adaptation_strategy="safe_original",
                        purpose="generic_rephrasing",
                    ),
                    QueryAdaptationProvenance(
                        origin_subquery="graph retrieval",
                        adaptation_strategy="compact_core",
                        purpose="generic_rephrasing",
                    ),
                ],
                run_dedupe_hit=True,
                source_skipped_reason="duplicate_adapted_query",
                diagnostic_papers=[paper],
            )
        ],
    )

    collector.register_retrieval(
        "initial_retrieval",
        [first, duplicate],
        origin_kind_by_query={
            "long natural language query": "initial_query",
            "graph retrieval": "initial_generated_subquery",
        },
    )

    snapshot = collector.snapshots[0]
    candidate = snapshot.candidates[0]
    assert len(candidate.provenance) == 3
    assert {item.origin_subquery for item in candidate.provenance} == {
        "long natural language query",
        "graph retrieval",
    }
    assert any(
        item.source_skipped_reason == "duplicate_adapted_query"
        for item in candidate.provenance
    )
    assert {
        (item.adaptation_strategy, item.purpose)
        for item in candidate.provenance
        if item.origin_subquery == "graph retrieval"
    } == {
        ("safe_original", "generic_rephrasing"),
        ("compact_core", "generic_rephrasing"),
    }
    assert all(
        item.source_skipped_reason is None
        for item in candidate.provenance
        if item.origin_subquery == "long natural language query"
    )
    assert snapshot.retrieval_calls[1].run_dedupe_hit is True
    assert snapshot.retrieval_calls[1].adapted_query == "graph retrieval"


def test_gold_first_found_stage_is_query_evolution() -> None:
    gold = _gold("Gold", "2401.00001")
    candidate = _candidate("Gold", "2401.00001", category="partially_relevant", rank=2)
    snapshots = _base_snapshots(initial=[])
    snapshots.extend(
        [
            StageCandidateSnapshot(
                stage="query_evolution_retrieval",
                candidates=[candidate],
            ),
            StageCandidateSnapshot(
                stage="post_evolution_deduplicated",
                candidates=[candidate],
            ),
            StageCandidateSnapshot(
                stage="post_evolution_judged",
                candidates=[candidate],
            ),
            StageCandidateSnapshot(
                stage="post_evolution_reranked",
                candidates=[candidate],
            ),
            StageCandidateSnapshot(stage="final_ranked", candidates=[candidate]),
        ]
    )
    output = _output(snapshots, ranked=[_ranked("Gold", "2401.00001", 2)])

    result = analyze_search_stages(_query([gold]), output, result_policy="highly_and_partial")

    assert result["gold_diagnostics"][0]["first_found_stage"] == (
        "query_evolution_retrieval"
    )


def test_not_retrieved_drop_reason() -> None:
    result = analyze_search_stages(
        _query([_gold("Missing", "2401.00001")]),
        _output(_base_snapshots(initial=[])),
        result_policy="highly_and_partial",
    )

    assert result["gold_diagnostics"][0]["drop_reason"] == "not_retrieved"


def test_failed_gold_crosswalk_is_not_reported_as_retrieval_miss() -> None:
    gold = EvalGoldPaper(
        title="Unresolved official identity",
        s2orc_corpus_id="123",
        metadata={"evaluator_crosswalk": {"status": "failed"}},
    )
    result = analyze_search_stages(
        _query([gold]),
        _output(_base_snapshots(initial=[])),
        result_policy="highly_and_partial",
    )

    assert result["evaluable_gold_count"] == 0
    assert result["gold_diagnostics"][0]["drop_reason"] == (
        "identity_crosswalk_failed"
    )
    assert result["stage_metrics"]["candidate_recall"]["initial_retrieval"] is None
    assert result["stage_metrics"]["recall_at_k"]["final_returned"]["20"] is None


@pytest.mark.parametrize(
    ("category", "reason"),
    [
        ("weakly_relevant", "judged_weakly_relevant"),
        ("irrelevant", "judged_irrelevant"),
        ("insufficient_evidence", "insufficient_evidence"),
    ],
)
def test_judgement_false_negative_drop_reasons(category: str, reason: str) -> None:
    candidate = _candidate("Gold", "2401.00001", category=category, rank=1)
    snapshots = _base_snapshots(initial=[candidate], final=[candidate])
    result = analyze_search_stages(
        _query([_gold("Gold", "2401.00001")]),
        _output(snapshots),
        result_policy="highly_and_partial",
    )

    assert result["gold_diagnostics"][0]["drop_reason"] == reason
    assert result["judgement"]["gold_false_negative_count"] == 1
    assert result["judgement"]["gold_false_negative_rate"] == 1.0


def test_outside_top_k_and_returned_are_distinct() -> None:
    returned = _candidate(
        "Returned",
        "2401.00001",
        category="partially_relevant",
        rank=3,
    )
    outside = _candidate(
        "Outside",
        "2401.00002",
        category="partially_relevant",
        rank=25,
    )
    snapshots = _base_snapshots(
        initial=[returned, outside],
        final=[returned, outside],
    )
    output = _output(
        snapshots,
        ranked=[_ranked("Returned", "2401.00001", 3)],
    )

    result = analyze_search_stages(
        _query(
            [
                _gold("Returned", "2401.00001"),
                _gold("Outside", "2401.00002"),
            ]
        ),
        output,
        result_policy="highly_and_partial",
    )
    reasons = [item["drop_reason"] for item in result["gold_diagnostics"]]

    assert reasons == ["returned", "outside_final_top_k"]
    assert result["reranking"]["gold_in_top_5"] == 1
    assert result["reranking"]["gold_outside_top_20"] == 1
    assert result["reranking"]["average_gold_rank"] == 14
    assert result["reranking"]["median_gold_rank"] == 14


def test_stage_candidate_recall_and_ranked_recall_at_k() -> None:
    found = _candidate("Found", "2401.00001", category="partially_relevant", rank=7)
    distractors = [
        _candidate(
            f"Distractor {index}",
            f"2501.{index:05d}",
            category="partially_relevant",
            rank=index,
        )
        for index in range(1, 7)
    ]
    candidates = [*distractors, found]
    output = _output(_base_snapshots(initial=candidates, final=candidates))
    result = analyze_search_stages(
        _query(
            [
                _gold("Found", "2401.00001"),
                _gold("Missing", "2401.00002"),
            ]
        ),
        output,
        result_policy="highly_and_partial",
    )

    metrics = result["stage_metrics"]
    assert metrics["candidate_recall"]["initial_retrieval"] == 0.5
    assert metrics["recall_at_k"]["initial_reranked"]["5"] == 0.0
    assert metrics["recall_at_k"]["initial_reranked"]["10"] == 0.5


def test_source_unique_gold_contribution_and_overlap() -> None:
    arxiv_only = _candidate(
        "Arxiv Only",
        "2401.00001",
        sources=["arxiv"],
    )
    shared = _candidate(
        "Shared",
        "2401.00002",
        sources=["arxiv", "openalex"],
    )
    output = _output(
        _base_snapshots(initial=[arxiv_only, shared], final=[arxiv_only, shared]),
        selected_sources=["arxiv", "openalex"],
        source_stats=[
            _source_stat("arxiv", returned=2),
            _source_stat("openalex", returned=1),
        ],
    )
    result = analyze_search_stages(
        _query(
            [
                _gold("Arxiv Only", "2401.00001"),
                _gold("Shared", "2401.00002"),
            ]
        ),
        output,
        result_policy="highly_and_partial",
    )
    sources = result["source_contribution"]

    assert sources["sources"]["arxiv"]["gold_hit_count"] == 2
    assert sources["sources"]["arxiv"]["unique_gold_hit_count"] == 1
    assert sources["sources"]["arxiv"]["gold_recall_contribution"] == 0.5
    assert sources["sources"]["openalex"]["unique_gold_hit_count"] == 0
    assert sources["overlap"]["arxiv ∩ openalex"] == {
        "candidate_count": 1,
        "gold_hit_count": 1,
    }


def test_aggregate_drop_reasons_and_bottleneck_labels() -> None:
    cases = [
        {
            "gold_diagnostics": [
                {"drop_reason": "not_retrieved"},
                {"drop_reason": "judged_weakly_relevant"},
            ],
            "stage_metrics": {
                "candidate_recall": {"initial_retrieval": 0.25},
                "recall_at_k": {"final_returned": {"5": 0.0, "10": 0.0, "20": 0.0, "50": 0.0}},
            },
            "judgement": {
                "retrieved_gold_count": 1,
                "gold_judged_weakly_relevant": 1,
                "gold_false_negative_count": 1,
            },
            "reranking": {"eligible_gold_count": 0, "gold_ranks": []},
            "source_contribution": {"sources": {}, "overlap": {}},
            "budget_stopped": False,
        }
    ]

    stage_metrics, error_analysis, _ = aggregate_stage_diagnostics(cases)

    assert error_analysis["drop_reasons"] == {
        "judged_weakly_relevant": 1,
        "not_retrieved": 1,
    }
    assert "retrieval_recall_bottleneck" in stage_metrics["bottleneck_labels"]
    assert "judgement_false_negative_bottleneck" in stage_metrics[
        "bottleneck_labels"
    ]
    assert "insufficient_sample" in stage_metrics["bottleneck_labels"]
    assert stage_metrics["sample_warning"] == "small_sample_diagnostic_only"


def test_bottleneck_rules_cover_reranking_source_and_budget() -> None:
    labels = classify_bottlenecks(
        {
            "case_count": 30,
            "initial_retrieval_recall": 0.9,
            "judgement": {"gold_false_negative_rate": 0.1},
            "reranking": {"eligible_gold_count": 4, "outside_top_20_rate": 0.5},
            "source_contribution": {"source_error_rate": 0.25},
            "budget_stop_rate": 0.2,
        }
    )

    assert labels == [
        "reranking_bottleneck",
        "source_reliability_bottleneck",
        "budget_bottleneck",
    ]


def test_comparison_reads_runs_and_rejects_incompatible_budget(
    tmp_path: Path,
) -> None:
    first = _comparison_run(tmp_path / "a", api=2.0, f1=0.1)
    second = _comparison_run(tmp_path / "b", api=3.0, f1=0.2)

    report = compare_runs([first, second])

    assert "a" in report and "b" in report
    assert "ΔF1@20" in report
    assert "策略（policy）" in report
    assert "compact 执行率" in report
    assert "0.100" in report

    config_path = second / "config.json"
    config = json.loads(config_path.read_text())
    config["budgets"]["max_candidate_papers"] = 999
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="budgets"):
        compare_runs([first, second])


def test_source_ablation_allows_only_source_configuration_to_differ(
    tmp_path: Path,
) -> None:
    first = _comparison_run(tmp_path / "arxiv", api=2.0, f1=0.1)
    second = _comparison_run(tmp_path / "local_arxiv", api=2.0, f1=0.2)
    config_path = second / "config.json"
    config = json.loads(config_path.read_text())
    config["sources"] = ["local_bm25", "arxiv"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="sources"):
        compare_runs([first, second])

    report = compare_runs([first, second], allow_source_difference=True)

    assert "Benchmark 来源消融对比" in report
    assert "检索源" in report
    assert "local_bm25, arxiv" in report


def test_adaptive_diagnostics_count_execution_new_candidates_and_posthoc_gold() -> None:
    compact_paper = _paper("Compact Gold", "2401.00002")
    compact_candidate = _candidate("Compact Gold", "2401.00002")
    compact_candidate.provenance[0].adaptation_strategy = "compact_core"
    snapshots = _base_snapshots(initial=[compact_candidate])
    output = _output(
        snapshots,
        source_stats=[
            SourceStats(
                source="arxiv",
                adaptation_strategy="safe_original",
                diagnostic_papers=[_paper("Safe Only", "2401.00001")],
                diagnostics=ConnectorDiagnostics(request_count=1),
            ),
            SourceStats(
                source="arxiv",
                adaptation_strategy="compact_core",
                logical_call_executed=True,
                compact_query_executed=True,
                triggered_by=["adaptive_low_candidate_count"],
                diagnostic_papers=[compact_paper],
                diagnostics=ConnectorDiagnostics(request_count=1),
            ),
        ],
    )

    result = analyze_search_stages(
        _query([_gold("Compact Gold", "2401.00002")]),
        output,
        result_policy="highly_and_partial",
    )

    adaptive = result["retrieval_diagnostics"]["adaptive"]
    assert adaptive["compact_decision_count"] == 1
    assert adaptive["compact_executed_count"] == 1
    assert adaptive["compact_execution_ratio"] == 1.0
    assert adaptive["compact_added_unique_candidate_count"] == 1
    assert result["query_strategy_contribution"]["compact_gold_increment"] == 1


def _query(gold: list[EvalGoldPaper]) -> EvalQuery:
    return EvalQuery(query_id="case", query="fixture query", gold_papers=gold)


def _gold(title: str, arxiv_id: str) -> EvalGoldPaper:
    return EvalGoldPaper(title=title, arxiv_id=arxiv_id)


def _paper(title: str, arxiv_id: str, *, source: str = "arxiv") -> Paper:
    return Paper(
        title=title,
        year=2024,
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=[source],
    )


def _candidate(
    title: str,
    arxiv_id: str,
    *,
    category: str | None = None,
    rank: int | None = None,
    sources: list[str] | None = None,
) -> DiagnosticCandidate:
    candidate_sources = sources or ["arxiv"]
    return DiagnosticCandidate(
        title=title,
        year=2024,
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=candidate_sources,
        provenance=[
            CandidateProvenance(
                origin_kind="initial_query",
                origin_stage="initial_retrieval",
                origin_subquery="fixture query",
                source=source,
            )
            for source in candidate_sources
        ],
        category=category,
        judgement_score=0.8 if category else None,
        rank=rank,
        final_score=0.8 if rank else None,
    )


def _ranked(title: str, arxiv_id: str, rank: int) -> RankedPaper:
    paper = _paper(title, arxiv_id)
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=0.8,
        category="partially_relevant",
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.8,
            authority_score=0.5,
            timeliness_score=0.5,
            metadata_score=1.0,
            final_score=0.8,
            relevance_weight=0.65,
            authority_weight=0.1,
            timeliness_weight=0.15,
            metadata_weight=0.1,
        ),
        ranking_reason="fixture",
        evidence=[EvidenceItem(source="title", text=title, confidence=1.0)],
    )


def _retrieval(query: str, paper: Paper, source: str) -> RetrievalOutput:
    return RetrievalOutput(
        query=query,
        requested_sources=[source],
        raw_count=1,
        deduplicated_count=1,
        papers=[paper],
    )


def _base_snapshots(
    *,
    initial: list[DiagnosticCandidate],
    final: list[DiagnosticCandidate] | None = None,
) -> list[StageCandidateSnapshot]:
    judged = [
        item.model_copy(
            update={
                "category": item.category or "partially_relevant",
                "judgement_score": item.judgement_score or 0.8,
            }
        )
        for item in initial
    ]
    ranked = [
        item.model_copy(
            update={
                "rank": item.rank or index,
                "category": item.category or "partially_relevant",
                "final_score": item.final_score or 0.8,
            }
        )
        for index, item in enumerate(initial, start=1)
    ]
    snapshots = [
        StageCandidateSnapshot(stage="initial_retrieval", candidates=initial),
        StageCandidateSnapshot(stage="initial_deduplicated", candidates=initial),
        StageCandidateSnapshot(stage="initial_judged", candidates=judged),
        StageCandidateSnapshot(stage="initial_reranked", candidates=ranked),
    ]
    snapshots.extend(
        StageCandidateSnapshot(stage=stage, status="skipped", skipped_reason="disabled")
        for stage in (
            "query_evolution_retrieval",
            "post_evolution_deduplicated",
            "post_evolution_judged",
            "post_evolution_reranked",
            "refchain_retrieval",
            "post_refchain_deduplicated",
            "post_refchain_judged",
            "post_refchain_reranked",
        )
    )
    snapshots.append(
        StageCandidateSnapshot(
            stage="final_ranked",
            candidates=ranked if final is None else final,
        )
    )
    return snapshots


def _output(
    snapshots: list[StageCandidateSnapshot],
    *,
    ranked: list[RankedPaper] | None = None,
    selected_sources: list[str] | None = None,
    source_stats: list[SourceStats] | None = None,
) -> SearchServiceOutput:
    sources = selected_sources or ["arxiv"]
    analysis = QueryAnalysis(
        original_query="fixture query",
        language="en",
        intent="survey",
        domain="machine_learning",
        constraints=QueryConstraint(),
    )
    plan = SearchPlan(
        query_analysis=analysis,
        subqueries=[
            SearchSubquery(
                query="fixture query",
                source_hints=sources,
                purpose="original_query",
            )
        ],
        selected_sources=sources,
        top_k=20,
    )
    return SearchServiceOutput(
        search_plan=plan,
        ranked_papers=ranked or [],
        all_ranked_papers=ranked or [],
        source_stats=source_stats or [_source_stat(source) for source in sources],
        stage_snapshots=snapshots,
    )


def _source_stat(source: str, *, returned: int = 0) -> SourceStats:
    return SourceStats(
        source=source,
        returned_count=returned,
        diagnostics=ConnectorDiagnostics(request_count=1),
    )


def _comparison_run(path: Path, *, api: float, f1: float) -> Path:
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "x",
        "case_ids": ["case-0"],
        "offset": 0,
        "limit": 1,
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "budgets": {"max_candidate_papers": 150},
        "llm": {"llm_enabled": False},
        "query_adapter_policy": "adaptive",
    }
    metrics = {
        "case_statistics": {"success_rate": 1.0},
        "end_to_end_metrics": {
            "f1_at_k": {"5": f1, "10": f1, "20": f1},
            "precision_at_k": {"20": f1},
            "recall_at_k": {"20": f1},
        },
        "benchmark_statistics": {
            "average_api_calls": api,
            "average_latency_seconds": api,
        },
    }
    stage = {
        "initial_retrieval_recall": f1,
        "final_returned_recall": {"20": f1},
        "judgement": {"gold_false_negative_rate": 0.0},
        "reranking": {"average_gold_rank": 1.0},
        "source_contribution": {"source_error_rate": 0.0},
        "retrieval_diagnostics": {
            "adaptive": {
                "compact_execution_ratio": 0.5,
                "compact_average_added_unique_candidates": 2.0,
            }
        },
        "query_strategy_contribution": {"compact_gold_increment": 1},
        "bottleneck_labels": ["insufficient_sample"],
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "stage_metrics.json").write_text(json.dumps(stage))
    return path
