from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_benchmark_runs import build_ablation_comparison  # noqa: E402
from scholar_agent.agents.query_evolution import evolve_queries
from scholar_agent.agents.refchain import expand_refchain
from scholar_agent.agents.retriever import SourceStats
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.pipeline_diagnostics import (
    CandidateProvenance,
    DiagnosticCandidate,
    RetrievalCallTrace,
    StageCandidateSnapshot,
)
from scholar_agent.core.search_schemas import (
    EvolvedSubquery,
    QueryAnalysis,
    QueryConstraint,
    QueryEvolutionRecord,
    QueryFacet,
    QueryPlanningResult,
    RankedPaper,
    RefChainOutput,
    RefChainRecord,
    RefChainSeed,
    RefChainSeedDiagnostic,
    ReferenceEdge,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
)
from scholar_agent.evaluation.stage_diagnostics import (
    aggregate_stage_diagnostics,
    analyze_search_stages,
    classify_module_outcome,
)
from scholar_agent.services.search_service import SearchService, SearchServiceOutput


def test_initial_query_planning_reports_per_query_yield_and_aggregate_cost() -> None:
    original = "graph retrieval"
    method_query = "graph retrieval contrastive learning"
    shared = _paper("Shared", "W1")
    exclusive = _paper("Method Exclusive", "W2")
    gold = EvalGoldPaper(title="Method Exclusive", openalex_id="W2")
    output = _output(
        snapshots=[
            StageCandidateSnapshot(
                stage="initial_retrieval",
                candidates=[
                    _candidate(
                        shared,
                        category="partially_relevant",
                        provenance=_initial_provenance(original),
                    ),
                    _candidate(
                        shared,
                        category="partially_relevant",
                        provenance=_initial_provenance(method_query),
                    ),
                    _candidate(
                        exclusive,
                        category="highly_relevant",
                        provenance=_initial_provenance(method_query),
                    ),
                ],
                retrieval_calls=[
                    RetrievalCallTrace(
                        origin_subquery=original,
                        source="openalex",
                        adapted_query=original,
                        request_count=1,
                        returned_count=1,
                        recorded_request_count=1,
                        recorded_latency_seconds=0.2,
                    ),
                    RetrievalCallTrace(
                        origin_subquery=method_query,
                        source="openalex",
                        adapted_query="graph retrieval contrastive",
                        request_count=1,
                        returned_count=2,
                        recorded_request_count=1,
                        recorded_latency_seconds=0.3,
                    ),
                ],
            )
        ],
    )
    output.search_plan = SearchPlan(
        query_analysis=output.search_plan.query_analysis.model_copy(
            update={"original_query": original}
        ),
        subqueries=[
            SearchSubquery(
                query=original,
                source_hints=["openalex"],
                purpose="original_query",
                facet_types=["topic"],
                provenance=["original_query"],
            ),
            SearchSubquery(
                query=method_query,
                source_hints=["openalex"],
                purpose="facet_method",
                facet_types=["topic", "method"],
                provenance=["rules:method:contrastive learning"],
            ),
        ],
        selected_sources=["openalex"],
        query_planning_policy="facet_balanced",
        query_planning=QueryPlanningResult(
            policy="facet_balanced",
            facets=[
                QueryFacet(
                    facet_type="method",
                    terms=["contrastive learning"],
                    confidence=1.0,
                    source="rules",
                )
            ],
            selected_subqueries=[
                SearchSubquery(
                    query=original,
                    source_hints=["openalex"],
                    purpose="original_query",
                    facet_types=["topic"],
                    provenance=["original_query"],
                ),
                SearchSubquery(
                    query=method_query,
                    source_hints=["openalex"],
                    purpose="facet_method",
                    facet_types=["topic", "method"],
                    provenance=["rules:method:contrastive learning"],
                ),
            ],
            identified_facet_count=1,
            selected_facet_count=1,
        ),
    )

    case = analyze_search_stages(
        _query(gold),
        output,
        result_policy="highly_and_partial",
    )
    planning = case["initial_query_planning"]

    assert planning["policy"] == "facet_balanced"
    assert planning["subquery_count"] == 2
    assert planning["adapted_query_count"] == 2
    assert planning["unique_candidate_count"] == 2
    assert planning["unique_gold_count"] == 1
    assert planning["recorded_request_count"] == 2
    assert planning["recorded_latency_seconds"] == pytest.approx(0.5)
    method = planning["subqueries"][1]
    assert method["exclusive_candidate_count"] == 1
    assert method["post_run_unique_gold_hit_count"] == 1
    assert method["dimension_coverage"]["method"] is True

    aggregate, _, _ = aggregate_stage_diagnostics([case])
    aggregate_planning = aggregate["initial_query_planning"]
    assert aggregate_planning["policies"] == {"facet_balanced": 1}
    assert aggregate_planning["effective_request_count"] == 2
    assert aggregate_planning["facet_contribution"]["method"] == {
        "unique_candidate_count": 1,
        "unique_gold_count": 1,
    }


def test_query_evolution_counts_seeds_queries_candidates_and_posthoc_gold() -> None:
    initial = _candidate(_paper("Initial", "W1"), category="highly_relevant", rank=1)
    evolved_paper = _paper("Evolved Gold", "W2")
    evolved = _candidate(
        evolved_paper,
        category="partially_relevant",
        provenance=_provenance("query_evolution", "evolved query"),
    )
    output = _output(
        enable_qe=True,
        snapshots=[
            _snapshot("initial_deduplicated", [initial]),
            _snapshot("initial_reranked", [initial]),
            StageCandidateSnapshot(
                stage="query_evolution_retrieval",
                candidates=[evolved],
                retrieval_calls=[
                    RetrievalCallTrace(
                        origin_subquery="evolved query",
                        source="openalex",
                        request_count=1,
                        returned_count=1,
                    )
                ],
            ),
            _snapshot("post_evolution_deduplicated", [initial, evolved]),
            _snapshot("post_evolution_judged", [initial, evolved]),
            _snapshot("post_evolution_reranked", [initial, evolved]),
            _snapshot("final_ranked", [initial, evolved]),
        ],
        query_records=[
            QueryEvolutionRecord(
                seed_count=1,
                generated_queries=[
                    EvolvedSubquery(
                        query="evolved query",
                        source_hints=["openalex"],
                        purpose="query_evolution_from_seed_title",
                        seed_paper_titles=["Initial"],
                    )
                ],
            )
        ],
        ranked=[_ranked(evolved_paper, category="partially_relevant")],
        stage_latencies={
            "query_evolution": 0.1,
            "query_evolution_retrieval": 0.4,
            "query_evolution_judgement": 0.2,
            "query_evolution_reranking": 0.1,
            "judgement": 0.2,
            "reranking": 0.1,
        },
        search_diagnostics=ConnectorDiagnostics(request_count=2),
    )

    result = analyze_search_stages(
        _query(EvalGoldPaper(title="Evolved Gold", openalex_id="W2")),
        output,
        result_policy="highly_and_partial",
    )
    qe = result["query_evolution"]

    assert qe["eligible_seed_count"] == 1
    assert qe["selected_seed_count"] == 1
    assert qe["generated_query_count"] == 1
    assert qe["executed_query_count"] == 1
    assert qe["evolved_new_unique_candidate_count"] == 1
    assert qe["evolved_new_unique_gold_count"] == 1
    assert qe["queries"][0]["seed_titles"] == ["Initial"]
    assert qe["queries"][0]["new_unique_candidate_count"] == 1
    assert qe["queries"][0]["partially_relevant_count"] == 1
    assert qe["queries"][0]["post_run_gold_hit_count"] == 1
    assert qe["queries"][0]["post_run_unique_gold_hit_count"] == 1
    assert qe["queries"][0]["ineffective_reasons"] == []
    assert result["stage_costs"]["query_evolution_api_calls"] == 1
    assert result["stage_costs"]["query_evolution_latency_seconds"] == 0.8
    assert qe["gold_filtered_by_judgement_count"] == 0
    assert qe["gold_lost_by_top_k_count"] == 0


def test_duplicate_evolved_query_is_skipped_without_new_request() -> None:
    initial = _candidate(_paper("Initial", "W1"), category="highly_relevant", rank=1)
    output = _output(
        enable_qe=True,
        snapshots=[
            _snapshot("initial_deduplicated", [initial]),
            _snapshot("initial_reranked", [initial]),
            StageCandidateSnapshot(
                stage="query_evolution_retrieval",
                retrieval_calls=[
                    RetrievalCallTrace(
                        origin_subquery="executed",
                        source="openalex",
                        request_count=1,
                    )
                ],
            ),
        ],
        query_records=[
            QueryEvolutionRecord(
                seed_count=1,
                generated_queries=[
                    _evolved("executed"),
                    _evolved("duplicate"),
                ],
            )
        ],
    )

    result = analyze_search_stages(_query(), output, result_policy="highly_and_partial")

    assert result["query_evolution"]["duplicate_query_count"] == 1
    assert result["query_evolution"]["executed_query_count"] == 1
    assert result["stage_costs"]["query_evolution_api_calls"] == 1


def test_refchain_counts_identifier_support_new_references_and_posthoc_gold() -> None:
    supported = _paper("Supported Seed", "W1")
    unsupported = Paper(
        title="Unsupported Seed",
        authors=["A"],
        year=2024,
        identifiers=PaperIdentifiers(arxiv_id="2401.00001"),
    )
    reference = _paper("Reference Gold", "W3")
    initial = [
        _candidate(supported, category="highly_relevant", rank=1),
        _candidate(unsupported, category="partially_relevant", rank=2),
    ]
    ref_candidate = _candidate(
        reference,
        category="weakly_relevant",
        provenance=_provenance("refchain", "refchain"),
    )
    record = RefChainRecord(
        seeds=[
            _seed(supported, 1, "highly_relevant"),
            _seed(unsupported, 2, "partially_relevant"),
        ],
        seed_diagnostics=[
            RefChainSeedDiagnostic(
                seed_id="openalex:w1",
                seed_rank=1,
                seed_category="highly_relevant",
                seed_score=0.8,
                identifier_type="openalex",
                request_count=1,
                references_returned=1,
                unique_references_returned=1,
            ),
            RefChainSeedDiagnostic(
                seed_rank=2,
                seed_category="partially_relevant",
                seed_score=0.7,
                skip_reason="unsupported_identifier",
            ),
        ],
        reference_edges=[
            ReferenceEdge(
                seed_paper_id="openalex:w1",
                reference_paper_id="openalex:w3",
            )
        ],
        raw_reference_count=1,
        returned_reference_count=1,
        diagnostics=ConnectorDiagnostics(request_count=1),
    )
    output = _output(
        enable_refchain=True,
        snapshots=[
            _snapshot("initial_deduplicated", initial),
            _snapshot("initial_reranked", initial),
            _snapshot("refchain_retrieval", [ref_candidate]),
            _snapshot("post_refchain_deduplicated", [*initial, ref_candidate]),
            _snapshot("post_refchain_judged", [*initial, ref_candidate]),
            _snapshot("post_refchain_reranked", [*initial, ref_candidate]),
            _snapshot("final_ranked", [*initial, ref_candidate]),
        ],
        refchain_output=RefChainOutput(
            references=[reference],
            reference_edges=list(record.reference_edges),
            record=record,
            diagnostics=ConnectorDiagnostics(request_count=1),
        ),
        reference_diagnostics=ConnectorDiagnostics(request_count=1),
    )

    result = analyze_search_stages(
        _query(EvalGoldPaper(title="Reference Gold", openalex_id="W3")),
        output,
        result_policy="highly_and_partial",
    )
    refchain = result["refchain"]

    assert refchain["eligible_seed_count"] == 2
    assert refchain["selected_seed_count"] == 2
    assert refchain["seed_with_supported_identifier_count"] == 1
    assert refchain["seed_without_supported_identifier_count"] == 1
    assert refchain["new_unique_reference_count"] == 1
    assert refchain["new_unique_reference_gold_count"] == 1
    assert refchain["seeds"][0]["new_unique_references"] == 1
    assert refchain["seeds"][1]["skip_reason"] == "unsupported_identifier"
    assert refchain["gold_found_but_filtered_count"] == 1
    assert refchain["gold_filtered_by_judgement_count"] == 1
    assert refchain["gold_lost_by_top_k_count"] == 0
    assert "gold_found_but_filtered" in refchain["conclusions"]


def test_duplicate_reference_is_not_counted_as_refchain_gain() -> None:
    paper = _paper("Already Retrieved", "W1")
    candidate = _candidate(paper, category="highly_relevant", rank=1)
    record = RefChainRecord(
        seeds=[_seed(paper, 1, "highly_relevant")],
        seed_diagnostics=[
            RefChainSeedDiagnostic(
                seed_id="openalex:w1",
                seed_rank=1,
                seed_category="highly_relevant",
                seed_score=0.8,
                identifier_type="openalex",
                request_count=1,
                references_returned=1,
                unique_references_returned=1,
            )
        ],
        reference_edges=[
            ReferenceEdge(
                seed_paper_id="openalex:w1",
                reference_paper_id="openalex:w1",
            )
        ],
        raw_reference_count=1,
        returned_reference_count=1,
    )
    output = _output(
        enable_refchain=True,
        snapshots=[
            _snapshot("initial_deduplicated", [candidate]),
            _snapshot("initial_reranked", [candidate]),
            _snapshot("refchain_retrieval", [candidate]),
            _snapshot("post_refchain_deduplicated", [candidate]),
        ],
        refchain_output=RefChainOutput(
            references=[paper],
            reference_edges=list(record.reference_edges),
            record=record,
        ),
    )

    result = analyze_search_stages(_query(), output, result_policy="highly_and_partial")

    assert result["refchain"]["new_unique_reference_count"] == 0
    assert result["refchain"]["seeds"][0]["new_unique_references"] == 0
    assert result["refchain"]["seeds"][0]["skip_reason"] == (
        "all_references_duplicate"
    )


def test_zero_gain_costs_are_null_and_outcome_rules_are_stable() -> None:
    result = analyze_search_stages(
        _query(),
        _output(),
        result_policy="highly_and_partial",
    )

    assert result["stage_costs"]["query_evolution"][
        "api_per_new_unique_gold"
    ] is None
    assert result["stage_costs"]["refchain"]["api_per_0_01_recall_gain"] is None
    assert classify_module_outcome(
        {
            "enabled": True,
            "selected_seed_count": 1,
            "executed_query_count": 1,
            "evolved_new_unique_candidate_count": 4,
            "evolved_new_unique_gold_count": 0,
        },
        {"api_per_0_01_recall_gain": None},
        case_count=10,
    ) == ["new_candidates_but_no_gold", "insufficient_sample"]


def test_compare_script_requires_and_outputs_complete_four_group_matrix(
    tmp_path: Path,
) -> None:
    runs = [
        _comparison_run(tmp_path, "baseline", False, False),
        _comparison_run(tmp_path, "query_evolution_only", True, False),
        _comparison_run(tmp_path, "refchain_only", False, True),
        _comparison_run(tmp_path, "combined", True, True),
    ]

    data, markdown, diagnostics = build_ablation_comparison(
        runs,
        split="development",
    )

    assert [item["configuration"] for item in data["configurations"]] == [
        "baseline",
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    ]
    assert "QE +候选/+gold" in markdown
    assert len(diagnostics) == 4
    assert all(item["sample_warning"] == "small_sample_diagnostic_only" for item in diagnostics)


def test_ablation_comparison_rejects_inconsistent_shared_configuration(
    tmp_path: Path,
) -> None:
    runs = [
        _comparison_run(tmp_path, "baseline", False, False),
        _comparison_run(tmp_path, "query_evolution_only", True, False),
        _comparison_run(tmp_path, "refchain_only", False, True),
        _comparison_run(tmp_path, "combined", True, True),
    ]
    config_path = runs[-1] / "config.json"
    config = json.loads(config_path.read_text())
    config["sources"] = ["arxiv"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="sources"):
        build_ablation_comparison(runs, split="validation")


def test_incomplete_ablation_report_never_fabricates_missing_metrics(
    tmp_path: Path,
) -> None:
    baseline = _comparison_run(tmp_path, "baseline", False, False)

    data, markdown, diagnostics = build_ablation_comparison(
        [baseline],
        split="development",
        incomplete_reason="来源持续限流，已安全停止",
    )

    assert data["status"] == "incomplete"
    assert len(data["configurations"]) == 1
    assert data["missing_configurations"] == [
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    ]
    assert "未完成组没有指标" in markdown
    assert len(diagnostics) == 1


def test_gold_and_benchmark_names_do_not_enter_production_module_decisions() -> None:
    source = "\n".join(
        [
            inspect.getsource(SearchService.run_search),
            inspect.getsource(evolve_queries),
            inspect.getsource(expand_refchain),
        ]
    ).casefold()

    assert "gold" not in source
    assert "autoscholarquery" not in source


def _output(
    *,
    enable_qe: bool = False,
    enable_refchain: bool = False,
    snapshots: list[StageCandidateSnapshot] | None = None,
    query_records: list[QueryEvolutionRecord] | None = None,
    refchain_output: RefChainOutput | None = None,
    ranked: list[RankedPaper] | None = None,
    stage_latencies: dict[str, float] | None = None,
    search_diagnostics: ConnectorDiagnostics | None = None,
    reference_diagnostics: ConnectorDiagnostics | None = None,
) -> SearchServiceOutput:
    analysis = QueryAnalysis(
        original_query="fixture query",
        language="en",
        intent="paper_finding",
        domain="computer_science",
        constraints=QueryConstraint(),
    )
    return SearchServiceOutput(
        search_plan=SearchPlan(
            query_analysis=analysis,
            subqueries=[
                SearchSubquery(
                    query="fixture query",
                    source_hints=["openalex"],
                    purpose="original_query",
                )
            ],
            selected_sources=["openalex"],
            enable_query_evolution=enable_qe,
            enable_refchain=enable_refchain,
        ),
        query_evolution_records=query_records or [],
        refchain_output=refchain_output,
        ranked_papers=ranked or [],
        all_ranked_papers=ranked or [],
        stage_snapshots=snapshots or [],
        stage_latencies=stage_latencies or {},
        source_stats=[SourceStats(source="openalex")],
        search_diagnostics=search_diagnostics or ConnectorDiagnostics(),
        reference_diagnostics=reference_diagnostics or ConnectorDiagnostics(),
        latency_seconds=1.0,
    )


def _paper(title: str, openalex_id: str) -> Paper:
    return Paper(
        title=title,
        authors=["A"],
        year=2024,
        abstract="evidence",
        identifiers=PaperIdentifiers(openalex_id=openalex_id),
        sources=["openalex"],
    )


def _candidate(
    paper: Paper,
    *,
    category: str,
    rank: int | None = None,
    provenance: list[CandidateProvenance] | None = None,
) -> DiagnosticCandidate:
    return DiagnosticCandidate(
        identifiers=paper.identifiers,
        title=paper.title,
        year=paper.year,
        sources=list(paper.sources),
        provenance=provenance or [],
        category=category,
        judgement_score=0.8,
        final_score=0.8,
        rank=rank,
    )


def _provenance(kind: str, query: str) -> list[CandidateProvenance]:
    return [
        CandidateProvenance(
            origin_kind=kind,  # type: ignore[arg-type]
            origin_stage=(
                "query_evolution_retrieval" if kind == "query_evolution" else "refchain_retrieval"
            ),
            origin_subquery=query,
            source="openalex",
        )
    ]


def _initial_provenance(query: str) -> list[CandidateProvenance]:
    return [
        CandidateProvenance(
            origin_kind=(
                "initial_query" if query == "graph retrieval" else "initial_generated_subquery"
            ),
            origin_stage="initial_retrieval",
            origin_subquery=query,
            source="openalex",
        )
    ]


def _snapshot(stage: str, candidates: list[DiagnosticCandidate]) -> StageCandidateSnapshot:
    return StageCandidateSnapshot(stage=stage, candidates=candidates)


def _query(gold: EvalGoldPaper | None = None) -> EvalQuery:
    return EvalQuery(
        query_id="case-0",
        query="fixture query",
        gold_papers=[gold] if gold else [],
    )


def _evolved(query: str) -> EvolvedSubquery:
    return EvolvedSubquery(
        query=query,
        source_hints=["openalex"],
        purpose="query_evolution_from_seed_title",
    )


def _seed(paper: Paper, rank: int, category: str) -> RefChainSeed:
    del category
    return RefChainSeed(paper=paper, rank=rank, score=0.8, reason="fixture")


def _ranked(paper: Paper, *, category: str) -> RankedPaper:
    return RankedPaper(
        rank=1,
        paper=paper,
        final_score=0.8,
        category=category,  # type: ignore[arg-type]
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.8,
            authority_score=0.5,
            timeliness_score=0.5,
            metadata_score=0.5,
            final_score=0.8,
            relevance_weight=0.7,
            authority_weight=0.1,
            timeliness_weight=0.1,
            metadata_weight=0.1,
        ),
        ranking_reason="fixture",
    )


def _comparison_run(
    root: Path,
    name: str,
    enable_qe: bool,
    enable_refchain: bool,
) -> Path:
    path = root / name
    path.mkdir()
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "sha",
        "case_count": 1,
        "case_ids": ["case-0"],
        "offset": 0,
        "limit": 1,
        "sources": ["arxiv", "openalex"],
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "budgets": {"max_candidate_papers": 150},
        "llm": {"llm_enabled": False},
        "query_adapter_policy": "adaptive",
        "enable_query_evolution": enable_qe,
        "enable_refchain": enable_refchain,
    }
    metrics = {
        "case_statistics": {"success_rate": 1.0},
        "end_to_end_metrics": {
            "f1_at_k": {"5": 0.1, "10": 0.1, "20": 0.1},
            "precision_at_k": {"20": 0.1},
            "recall_at_k": {"5": 0.1, "10": 0.1, "20": 0.1},
        },
        "benchmark_statistics": {
            "average_api_calls": 1.0,
            "average_latency_seconds": 1.0,
        },
        "efficiency": {
            "avg_search_api_call_count": 1.0,
            "avg_reference_api_call_count": 0.0,
        },
    }
    stage = {
        "initial_retrieval_recall": 0.1,
        "post_evolution_recall": 0.1,
        "post_refchain_recall": 0.1,
        "final_returned_recall": {"20": 0.1},
        "source_contribution": {"source_error_rate": 0.0},
        "query_evolution": {
            "evolved_new_unique_candidate_count": int(enable_qe),
            "evolved_new_unique_gold_count": 0,
            "conclusions": ["insufficient_sample"],
        },
        "refchain": {
            "new_unique_reference_count": int(enable_refchain),
            "new_unique_reference_gold_count": 0,
            "conclusions": ["insufficient_sample"],
        },
        "stage_costs": {
            "average_query_evolution_latency_seconds": 0.1 if enable_qe else 0.0,
            "average_refchain_latency_seconds": 0.1 if enable_refchain else 0.0,
            "query_evolution": {},
            "refchain": {},
        },
    }
    result = {
        "case_id": "case-0",
        "status": "succeeded",
        "stage_diagnostics": {
            "query_evolution": stage["query_evolution"],
            "refchain": stage["refchain"],
            "stage_costs": stage["stage_costs"],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "stage_metrics.json").write_text(json.dumps(stage))
    (path / "results.jsonl").write_text(json.dumps(result) + "\n")
    return path
