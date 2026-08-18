"""Offline SearchService evaluator using shared selection and metric policy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from scholar_agent.agents.refchain import ReferenceFetcher
from scholar_agent.core.evaluation_schemas import (
    EvalAggregateReport,
    EvalCaseEfficiency,
    EvalCaseStatistics,
    EvalGroupName,
    EvalGroupResult,
    EvalMetricSet,
    EvalQuery,
    EvalQueryResult,
    EvalSuiteResult,
)
from scholar_agent.evaluation.metrics import (
    aggregate_efficiency,
    average_metric_sets,
    candidate_count_metrics,
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    zero_metric_set,
)
from scholar_agent.evaluation.selection import (
    DEFAULT_RESULT_POLICY,
    ResultPolicy,
    select_ranked_results,
)
from scholar_agent.services.search_service import (
    RetrieverFn,
    SearchService,
    SearchServiceOutput,
)


DEFAULT_GROUPS: tuple[EvalGroupName, ...] = (
    "baseline",
    "query_evolution_only",
    "refchain_only",
    "query_evolution_plus_refchain",
)

_GROUP_OPTIONS: dict[EvalGroupName, tuple[bool, bool]] = {
    "baseline": (False, False),
    "query_evolution_only": (True, False),
    "refchain_only": (False, True),
    "query_evolution_plus_refchain": (True, True),
}


class OfflineSearchEvaluator:
    def __init__(
        self,
        *,
        retriever: RetrieverFn,
        reference_fetcher: ReferenceFetcher | None = None,
        max_workers: int = 4,
        groups: Sequence[EvalGroupName] = DEFAULT_GROUPS,
        result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    ) -> None:
        self._retriever = retriever
        self._reference_fetcher = reference_fetcher or _empty_reference_fetcher
        self._max_workers = max_workers
        self._groups = tuple(groups)
        self._result_policy = result_policy

    def evaluate(self, eval_queries: Iterable[EvalQuery]) -> EvalSuiteResult:
        queries = list(eval_queries)
        query_results: list[EvalQueryResult] = []
        for eval_query in queries:
            group_results = {
                group: self._evaluate_group(eval_query, group)
                for group in self._groups
            }
            query_results.append(
                EvalQueryResult(
                    query_id=eval_query.query_id,
                    query=eval_query.query,
                    group_results=group_results,
                )
            )

        reports = _aggregate_reports(query_results, queries, self._groups)
        return EvalSuiteResult(
            query_results=query_results,
            aggregate_metrics={
                group: report.end_to_end_metrics
                for group, report in reports.items()
            },
            aggregate_reports=reports,
        )

    def _evaluate_group(
        self,
        eval_query: EvalQuery,
        group: EvalGroupName,
    ) -> EvalGroupResult:
        top_k = max(eval_query.top_k_values or [20])
        enable_query_evolution, enable_refchain = _GROUP_OPTIONS[group]
        try:
            output = SearchService(
                retriever=self._retriever,
                reference_fetcher=self._reference_fetcher,
                max_workers=self._max_workers,
            ).run_search(
                eval_query.query,
                top_k=top_k,
                run_profile=eval_query.run_profile,
                enable_query_evolution=enable_query_evolution,
                query_evolution_policy="seed_expansion",
                enable_refchain=enable_refchain,
                current_year=eval_query.current_year,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed case
            return _failed_group_result(eval_query, group, exc)

        ranked = select_ranked_results(output, policy=self._result_policy)
        metrics = _score_output(eval_query, output, ranked)
        efficiency = _output_efficiency(output, len(ranked))
        ranked_ids = [
            paper_id
            for paper_id in (canonical_paper_id(item) for item in ranked)
            if paper_id is not None
        ]
        return EvalGroupResult(
            query_id=eval_query.query_id,
            group=group,
            metrics=metrics,
            ranked_paper_ids=ranked_ids,
            warnings=[*output.warnings, *efficiency.warnings],
            source_stats=[_model_to_dict(item) for item in output.source_stats],
            raw_count=output.raw_count,
            deduplicated_count=output.deduplicated_count,
            latency_seconds=output.latency_seconds,
            efficiency=efficiency,
        )


def evaluate_search_service_offline(
    eval_queries: Iterable[EvalQuery],
    *,
    retriever: RetrieverFn,
    reference_fetcher: ReferenceFetcher | None = None,
    max_workers: int = 4,
    groups: Sequence[EvalGroupName] = DEFAULT_GROUPS,
    result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
) -> EvalSuiteResult:
    return OfflineSearchEvaluator(
        retriever=retriever,
        reference_fetcher=reference_fetcher,
        max_workers=max_workers,
        groups=groups,
        result_policy=result_policy,
    ).evaluate(eval_queries)


def _score_output(
    eval_query: EvalQuery,
    output: SearchServiceOutput,
    ranked: Sequence[Any],
) -> EvalMetricSet:
    metrics = evaluate_ranking(ranked, eval_query.gold_papers, eval_query.top_k_values)
    counts = candidate_count_metrics(
        output.raw_count,
        output.deduplicated_count,
        ranked_count=len(ranked),
        source_stats=output.source_stats,
    )
    source_call_count = (
        output.search_diagnostics.request_count
        + output.reference_diagnostics.request_count
    )
    source_error_count = (
        output.search_diagnostics.error_count
        + output.reference_diagnostics.error_count
    )
    return metrics.model_copy(
        update={
            "raw_count": counts["raw_count"],
            "deduplicated_count": counts["deduplicated_count"],
            "ranked_count": counts["ranked_count"],
            "duplicate_count": counts["duplicate_count"],
            "duplicate_ratio": counts["duplicate_ratio"],
            "per_source_returned_count": counts["per_source_returned_count"],
            "source_call_count": source_call_count,
            "source_error_count": source_error_count,
            "source_error_rate": (
                source_error_count / source_call_count if source_call_count else 0.0
            ),
            "warning_count": len(output.warnings),
            "query_warning_rate": float(bool(output.warnings)),
        }
    )


def _output_efficiency(
    output: SearchServiceOutput,
    returned_result_count: int,
) -> EvalCaseEfficiency:
    search = output.search_diagnostics
    reference = output.reference_diagnostics
    external_requests = search.request_count + reference.request_count
    return EvalCaseEfficiency(
        latency_seconds=output.latency_seconds,
        api_call_count=external_requests + output.llm_call_count,
        search_api_call_count=search.request_count,
        reference_api_call_count=reference.request_count,
        retry_count=search.retry_count + reference.retry_count,
        error_count=search.error_count + reference.error_count,
        llm_call_count=output.llm_call_count,
        llm_total_tokens=output.llm_total_tokens,
        search_rounds=output.budget_status.completed_search_rounds,
        raw_count=output.raw_count,
        deduplicated_count=output.deduplicated_count,
        returned_result_count=returned_result_count,
        cache_hit_count=search.cache_hit_count + reference.cache_hit_count,
        rate_limit_wait_seconds=(
            search.rate_limit_wait_seconds + reference.rate_limit_wait_seconds
        ),
        source_call_count=external_requests,
        source_error_count=search.error_count + reference.error_count,
    )


def _failed_group_result(
    eval_query: EvalQuery,
    group: EvalGroupName,
    exc: Exception,
) -> EvalGroupResult:
    error_message = f"{type(exc).__name__}: {exc}"
    metrics = zero_metric_set(eval_query.top_k_values).model_copy(
        update={
            "failed_case_count": 1,
            "failed_case_rate": 1.0,
            "warning_count": 1,
            "query_warning_rate": 1.0,
        }
    )
    return EvalGroupResult(
        query_id=eval_query.query_id,
        group=group,
        metrics=metrics,
        warnings=[error_message],
        failed=True,
        error_message=error_message,
    )


def _aggregate_reports(
    query_results: Sequence[EvalQueryResult],
    eval_queries: Sequence[EvalQuery],
    groups: Sequence[EvalGroupName],
) -> dict[EvalGroupName, EvalAggregateReport]:
    gold_by_query = {
        item.query_id: evaluable_gold_count(item.gold_papers) > 0
        for item in eval_queries
    }
    k_values = sorted(
        {
            k
            for query in eval_queries
            for k in (query.top_k_values or [5, 10, 20])
            if k > 0
        }
    ) or [5, 10, 20]
    reports: dict[EvalGroupName, EvalAggregateReport] = {}
    for group in groups:
        results = [
            item.group_results[group]
            for item in query_results
            if group in item.group_results
        ]
        gold_results = [item for item in results if gold_by_query.get(item.query_id)]
        successful = [item for item in gold_results if not item.failed]
        failed_count = sum(item.failed for item in gold_results)
        missing_gold_count = len(results) - len(gold_results)
        total = len(results)
        statistics = EvalCaseStatistics(
            total_case_count=total,
            gold_case_count=len(gold_results),
            evaluated_success_count=len(successful),
            failed_case_count=failed_count,
            missing_result_count=0,
            missing_gold_count=missing_gold_count,
            success_rate=len(successful) / total if total else 0.0,
            failed_case_rate=failed_count / total if total else 0.0,
            missing_result_rate=0.0,
        )
        reports[group] = EvalAggregateReport(
            success_only_metrics=(
                average_metric_sets([item.metrics for item in successful])
                if successful
                else zero_metric_set(k_values)
            ),
            end_to_end_metrics=(
                average_metric_sets([item.metrics for item in gold_results])
                if gold_results
                else zero_metric_set(k_values)
            ),
            case_statistics=statistics,
            efficiency=aggregate_efficiency(
                [item.efficiency for item in successful]
            ),
        )
    return reports


def _model_to_dict(item: Any) -> dict[str, Any]:
    return item.model_dump() if hasattr(item, "model_dump") else dict(item)


def _empty_reference_fetcher(*_: Any, **__: Any) -> list[Any]:
    return []
