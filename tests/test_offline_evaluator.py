from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.evaluation.offline_evaluator import evaluate_search_service_offline
from scholar_agent.evaluation import offline_evaluator
from scripts import evaluate_search_batch
from scholar_agent.services.search_service import SearchService


def make_paper(
    title: str,
    *,
    doi: str,
    openalex_id: str | None = None,
    citation_count: int = 10,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=2025,
        venue="ACL",
        abstract=(
            "This paper studies LLM reranking methods for scientific literature "
            "retrieval with robust evidence and ranking."
        ),
        identifiers=PaperIdentifiers(doi=doi, openalex_id=openalex_id),
        sources=["fixture"],
        citation_count=citation_count,
    )


BASELINE_PAPER = make_paper(
    "LLM Reranking for Scientific Literature Retrieval",
    doi="10.123/baseline",
    openalex_id="WBASE",
    citation_count=100,
)
EVOLVED_PAPER = make_paper(
    "Recent Advances in LLM Reranking for Scientific Literature Retrieval",
    doi="10.123/evolved",
    openalex_id="WEVOLVED",
    citation_count=80,
)
REFCHAIN_PAPER = make_paper(
    "Reference Evidence for LLM Reranking in Literature Retrieval",
    doi="10.123/refchain",
    openalex_id="WREF",
    citation_count=60,
)


def make_output(
    query: str,
    papers: list[Paper],
    *,
    warning: str | None = None,
    error_message: str | None = None,
) -> RetrievalOutput:
    warnings = [warning] if warning else []
    return RetrievalOutput(
        query=query,
        requested_sources=["openalex", "arxiv"],
        raw_count=len(papers),
        deduplicated_count=len(papers),
        papers=papers,
        source_stats=[
            SourceStats(
                source="fixture",
                returned_count=len(papers),
                latency_seconds=0.01,
                error_message=error_message,
                diagnostics=ConnectorDiagnostics(
                    request_count=1,
                    error_count=int(error_message is not None),
                    latency_seconds=0.01,
                ),
            )
        ],
        warnings=warnings,
        latency_seconds=0.01,
    )


def make_eval_query() -> EvalQuery:
    return EvalQuery(
        query_id="q1",
        query="latest LLM reranking methods for scientific literature retrieval",
        gold_papers=[
            EvalGoldPaper(doi="10.123/baseline"),
            EvalGoldPaper(doi="10.123/evolved"),
            EvalGoldPaper(doi="10.123/refchain"),
        ],
        top_k_values=[5, 10, 20],
        current_year=2026,
    )


def make_fake_retriever(
    *,
    include_warning: bool = False,
) -> Callable[[str, int, list[str] | None], RetrievalOutput]:
    initial_queries = {
        subquery.query
        for subquery in analyze_query(
            make_eval_query().query,
            current_year=2026,
        ).subqueries
    }

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source, sources
        if query not in initial_queries:
            return make_output(query, [EVOLVED_PAPER])
        return make_output(
            query,
            [BASELINE_PAPER],
            warning="fixture_openalex_warning" if include_warning else None,
            error_message="fixture HTTP 503" if include_warning else None,
        )

    return fake_retriever


def fake_reference_fetcher(paper: Paper, limit: int = 20) -> list[Paper]:
    assert limit > 0
    if paper.identifiers.doi in {"10.123/baseline", "10.123/evolved"}:
        return [REFCHAIN_PAPER]
    return []


def forbidden_reference_fetcher(paper: Paper, limit: int = 20) -> list[Paper]:
    del paper, limit
    raise AssertionError("reference fetcher should not run for baseline/evolution")


def test_baseline_can_run_offline() -> None:
    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=make_fake_retriever(),
        reference_fetcher=forbidden_reference_fetcher,
        groups=["baseline"],
        max_workers=1,
    )

    baseline = result.query_results[0].group_results["baseline"]

    assert not baseline.failed
    assert baseline.metrics.raw_count > 0
    assert baseline.metrics.recall_at_k[20] == pytest.approx(1 / 3)
    assert baseline.metrics.precision_at_k[5] > 0
    assert baseline.ranked_paper_ids
    assert result.aggregate_metrics["baseline"].recall_at_k[20] == pytest.approx(1 / 3)


def test_query_evolution_adds_candidates_and_can_improve_recall() -> None:
    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=make_fake_retriever(),
        reference_fetcher=forbidden_reference_fetcher,
        groups=["baseline", "query_evolution_only"],
        max_workers=1,
    )

    groups = result.query_results[0].group_results
    baseline = groups["baseline"]
    evolved = groups["query_evolution_only"]

    assert evolved.raw_count > baseline.raw_count
    assert evolved.metrics.recall_at_k[20] > baseline.metrics.recall_at_k[20]
    assert "doi:10.123/evolved" in evolved.ranked_paper_ids


def test_refchain_references_participate_in_scoring() -> None:
    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=make_fake_retriever(),
        reference_fetcher=fake_reference_fetcher,
        groups=["query_evolution_only", "query_evolution_plus_refchain"],
        max_workers=1,
    )

    groups = result.query_results[0].group_results
    evolved = groups["query_evolution_only"]
    refchain = groups["query_evolution_plus_refchain"]

    assert refchain.raw_count > evolved.raw_count
    assert refchain.metrics.recall_at_k[20] > evolved.metrics.recall_at_k[20]
    assert "doi:10.123/refchain" in refchain.ranked_paper_ids
    assert refchain.metrics.per_source_returned_count["refchain"] >= 1


def test_connector_warning_and_error_enter_error_metrics() -> None:
    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=make_fake_retriever(include_warning=True),
        reference_fetcher=forbidden_reference_fetcher,
        groups=["baseline"],
        max_workers=1,
    )

    baseline = result.query_results[0].group_results["baseline"]

    assert "fixture_openalex_warning" in baseline.warnings
    assert baseline.metrics.source_error_count > 0
    assert baseline.metrics.source_call_count > 0
    assert baseline.metrics.source_error_rate > 0
    assert not any("source_call_count_unavailable" in item for item in baseline.warnings)
    assert baseline.metrics.warning_count == 1
    assert baseline.metrics.query_warning_rate == 1.0


def test_evaluator_uses_injected_fixtures_without_real_connectors() -> None:
    retriever_calls: list[str] = []
    reference_calls: list[str] = []

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source, sources
        retriever_calls.append(query)
        return make_output(query, [BASELINE_PAPER])

    def fake_fetcher(paper: Paper, limit: int = 20) -> list[Paper]:
        del limit
        reference_calls.append(paper.identifiers.doi or paper.title)
        return [REFCHAIN_PAPER]

    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=fake_retriever,
        reference_fetcher=fake_fetcher,
        groups=["refchain_only"],
        max_workers=1,
    )

    assert retriever_calls
    assert reference_calls
    assert not result.query_results[0].group_results["refchain_only"].failed


def test_all_four_ablation_groups_complete_with_expected_behavior() -> None:
    result = evaluate_search_service_offline(
        [make_eval_query()],
        retriever=make_fake_retriever(),
        reference_fetcher=fake_reference_fetcher,
        max_workers=1,
    )

    groups = result.query_results[0].group_results
    assert set(groups) == {
        "baseline",
        "query_evolution_only",
        "refchain_only",
        "query_evolution_plus_refchain",
    }
    assert all(not group.failed for group in groups.values())
    assert groups["query_evolution_only"].raw_count > groups["baseline"].raw_count
    assert groups["refchain_only"].raw_count > groups["baseline"].raw_count
    assert groups["query_evolution_plus_refchain"].raw_count > groups[
        "query_evolution_only"
    ].raw_count
    assert set(result.aggregate_reports) == set(groups)


def test_four_ablation_group_switches_are_exact() -> None:
    assert offline_evaluator._GROUP_OPTIONS == {
        "baseline": (False, False),
        "query_evolution_only": (True, False),
        "refchain_only": (False, True),
        "query_evolution_plus_refchain": (True, True),
    }


def test_offline_and_batch_metrics_agree_for_same_ranked_input() -> None:
    query = make_eval_query()
    offline = evaluate_search_service_offline(
        [query],
        retriever=make_fake_retriever(),
        reference_fetcher=forbidden_reference_fetcher,
        groups=["baseline"],
        max_workers=1,
    )
    batch = evaluate_search_batch.evaluate_batch_results(
        [
            {
                "case_id": query.query_id,
                "query": query.query,
                "status": "succeeded",
                "result": {
                    "highly_relevant_papers": [
                        {"rank": 1, "paper": BASELINE_PAPER.model_dump(mode="json")}
                    ],
                    "partially_relevant_papers": [],
                },
            }
        ],
        [
            {
                "case_id": query.query_id,
                "relevant_papers": [
                    paper.model_dump(mode="json") for paper in query.gold_papers
                ],
            }
        ],
        k_values=query.top_k_values,
    )

    offline_metrics = offline.query_results[0].group_results["baseline"].metrics
    assert batch["end_to_end_metrics"]["recall_at_k"] == {
        str(k): value for k, value in offline_metrics.recall_at_k.items()
    }
    assert batch["end_to_end_metrics"]["precision_at_k"] == {
        str(k): value for k, value in offline_metrics.precision_at_k.items()
    }
    assert batch["end_to_end_metrics"]["f1_at_k"] == {
        str(k): value for k, value in offline_metrics.f1_at_k.items()
    }


def test_offline_failed_case_is_zero_only_in_end_to_end(monkeypatch) -> None:
    original_run_search = SearchService.run_search

    def run_search_with_failure(self, query: str, *args, **kwargs):
        if query == "forced failure":
            raise RuntimeError("fixture failure")
        return original_run_search(self, query, *args, **kwargs)

    monkeypatch.setattr(SearchService, "run_search", run_search_with_failure)
    failed_query = make_eval_query().model_copy(
        update={"query_id": "q2", "query": "forced failure"}
    )

    result = evaluate_search_service_offline(
        [make_eval_query(), failed_query],
        retriever=make_fake_retriever(),
        reference_fetcher=forbidden_reference_fetcher,
        groups=["baseline"],
        max_workers=1,
    )
    report = result.aggregate_reports["baseline"]

    assert report.case_statistics.evaluated_success_count == 1
    assert report.case_statistics.failed_case_count == 1
    assert report.success_only_metrics.f1_at_k[20] > 0
    assert report.end_to_end_metrics.f1_at_k[20] == pytest.approx(
        report.success_only_metrics.f1_at_k[20] / 2
    )
