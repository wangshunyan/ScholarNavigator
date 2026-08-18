"""Rule-based single-layer reference expansion."""

from __future__ import annotations

import time
from collections.abc import Callable

from scholar_agent.connectors.schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    QueryAnalysis,
    RankedPaper,
    RefChainOptions,
    RefChainOutput,
    RefChainRecord,
    RefChainSeed,
    RefChainSeedDiagnostic,
    ReferenceEdge,
)


ReferenceFetcher = Callable[[Paper, int], list[Paper] | ConnectorSearchResult]
BudgetCheck = Callable[[], str | None]
CancelCheck = Callable[[], None]


class RefChainAgent:
    """Expand references for relevant ranked papers without recursion."""

    def expand(
        self,
        query_analysis: QueryAnalysis,
        ranked_papers: list[RankedPaper],
        fetch_references: ReferenceFetcher,
        options: RefChainOptions | None = None,
        budget_check: BudgetCheck | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> RefChainOutput:
        del query_analysis  # Reserved for future domain-aware seed policies.
        start = time.perf_counter()
        options = options or RefChainOptions()
        warnings: list[str] = []
        skipped_reasons: list[str] = []
        references: list[Paper] = []
        reference_edges: list[ReferenceEdge] = []
        connector_diagnostics: list[ConnectorDiagnostics] = []
        recorded_connector_diagnostics: list[ConnectorDiagnostics] = []
        recorded_latency_seconds = 0.0
        seed_diagnostics: list[RefChainSeedDiagnostic] = []
        seen_reference_ids: set[str] = set()

        seeds = _select_seeds(ranked_papers, options)
        if not seeds:
            warnings.append("refchain_no_eligible_seed")
            skipped_reasons.append("refchain_no_eligible_seed")

        for ranked in seeds:
            if cancel_check is not None:
                cancel_check()
            if budget_check is not None:
                budget_reason = budget_check()
                if budget_reason is not None:
                    warnings.append(budget_reason)
                    skipped_reasons.append(budget_reason)
                    seed_diagnostics.append(
                        _seed_diagnostic(ranked, skip_reason="budget_stop")
                    )
                    break
            if len(references) >= options.max_total_references:
                warnings.append("refchain_total_reference_limit_reached")
                seed_diagnostics.append(
                    _seed_diagnostic(ranked, skip_reason="budget_stop")
                )
                break

            seed_id = _paper_identifier(ranked.paper)
            if seed_id is None:
                warning = f"refchain_seed_missing_supported_identifier:{ranked.rank}"
                warnings.append(warning)
                skipped_reasons.append(warning)
                seed_diagnostics.append(
                    _seed_diagnostic(ranked, skip_reason="unsupported_identifier")
                )
                continue

            remaining = options.max_total_references - len(references)
            per_seed_limit = min(options.max_references_per_seed, remaining)
            if per_seed_limit <= 0:
                warnings.append("refchain_total_reference_limit_reached")
                seed_diagnostics.append(
                    _seed_diagnostic(ranked, skip_reason="budget_stop")
                )
                break

            try:
                fetch_result = fetch_references(ranked.paper, per_seed_limit)
            except Exception as exc:  # noqa: BLE001 - isolate one seed failure
                warning = f"refchain_seed_failed:{ranked.rank}:{exc}"
                warnings.append(warning)
                skipped_reasons.append(warning)
                connector_diagnostics.append(ConnectorDiagnostics(error_count=1))
                seed_diagnostics.append(
                    _seed_diagnostic(ranked, skip_reason="source_failure")
                )
                continue

            request_count = 0
            recorded_diagnostics = ConnectorDiagnostics()
            recorded_latency = 0.0
            snapshot_key: str | None = None
            fetch_error: str | None = None
            if isinstance(fetch_result, ConnectorSearchResult):
                connector_diagnostics.append(fetch_result.diagnostics)
                recorded_diagnostics = (
                    fetch_result.recorded_diagnostics or ConnectorDiagnostics()
                )
                recorded_connector_diagnostics.append(recorded_diagnostics)
                recorded_latency = fetch_result.recorded_latency_seconds
                snapshot_key = fetch_result.snapshot_key
                recorded_latency_seconds += recorded_latency
                warnings.extend(fetch_result.warnings)
                request_count = fetch_result.diagnostics.request_count
                fetch_error = fetch_result.error_message
                if fetch_result.error_message:
                    skipped_reasons.append(fetch_result.error_message)
                fetched = fetch_result.papers
            else:
                fetched = fetch_result

            if cancel_check is not None:
                cancel_check()

            returned_for_seed = 0
            unique_for_seed = 0
            for reference in fetched[:per_seed_limit]:
                if len(references) >= options.max_total_references:
                    warnings.append("refchain_total_reference_limit_reached")
                    break
                reference_id = _paper_identifier(reference)
                if reference_id is None:
                    warning = f"refchain_reference_missing_identifier:{ranked.rank}"
                    warnings.append(warning)
                    skipped_reasons.append(warning)
                    continue
                references.append(reference)
                returned_for_seed += 1
                if reference_id not in seen_reference_ids:
                    unique_for_seed += 1
                    seen_reference_ids.add(reference_id)
                reference_edges.append(
                    ReferenceEdge(
                        seed_paper_id=seed_id,
                        reference_paper_id=reference_id,
                        source="openalex",
                    )
                )
            skip_reason = _seed_fetch_skip_reason(
                fetch_error,
                batch_status=(
                    fetch_result.reference_batch_status
                    if isinstance(fetch_result, ConnectorSearchResult)
                    else None
                ),
                returned_for_seed=returned_for_seed,
                unique_for_seed=unique_for_seed,
            )
            seed_diagnostics.append(
                _seed_diagnostic(
                    ranked,
                    request_count=request_count,
                    snapshot_key=snapshot_key,
                    recorded_diagnostics=recorded_diagnostics,
                    recorded_latency_seconds=recorded_latency,
                    references_returned=returned_for_seed,
                    unique_references_returned=unique_for_seed,
                    skip_reason=skip_reason,
                )
            )

        diagnosed_ranks = {item.seed_rank for item in seed_diagnostics}
        for ranked in seeds:
            if ranked.rank not in diagnosed_ranks:
                seed_diagnostics.append(
                    _seed_diagnostic(ranked, skip_reason="budget_stop")
                )

        latency_seconds = time.perf_counter() - start
        diagnostics = merge_connector_diagnostics(connector_diagnostics)
        recorded_diagnostics = merge_connector_diagnostics(
            recorded_connector_diagnostics
        )
        record = RefChainRecord(
            seeds=[_to_seed(seed) for seed in seeds],
            seed_diagnostics=seed_diagnostics,
            reference_edges=reference_edges,
            raw_reference_count=len(references),
            returned_reference_count=len(references),
            skipped_reasons=_dedupe(skipped_reasons),
            warnings=_dedupe(warnings),
            latency_seconds=latency_seconds,
            diagnostics=diagnostics,
            recorded_diagnostics=recorded_diagnostics,
            recorded_latency_seconds=recorded_latency_seconds,
        )
        return RefChainOutput(
            references=references,
            reference_edges=reference_edges,
            record=record,
            warnings=record.warnings,
            latency_seconds=latency_seconds,
            diagnostics=diagnostics,
            recorded_diagnostics=recorded_diagnostics,
            recorded_latency_seconds=recorded_latency_seconds,
        )


def expand_refchain(
    query_analysis: QueryAnalysis,
    ranked_papers: list[RankedPaper],
    fetch_references: ReferenceFetcher,
    options: RefChainOptions | None = None,
    budget_check: BudgetCheck | None = None,
    cancel_check: CancelCheck | None = None,
) -> RefChainOutput:
    """Expand one layer of references using an injected fetcher."""

    return RefChainAgent().expand(
        query_analysis=query_analysis,
        ranked_papers=ranked_papers,
        fetch_references=fetch_references,
        options=options,
        budget_check=budget_check,
        cancel_check=cancel_check,
    )


def _select_seeds(
    ranked_papers: list[RankedPaper],
    options: RefChainOptions,
) -> list[RankedPaper]:
    seeds: list[RankedPaper] = []
    for ranked in ranked_papers:
        if ranked.category == "highly_relevant":
            seeds.append(ranked)
        elif (
            ranked.category == "partially_relevant"
            and ranked.final_score >= options.min_seed_score
        ):
            seeds.append(ranked)
        if len(seeds) >= options.max_seed_papers:
            break
    return seeds


def _to_seed(ranked: RankedPaper) -> RefChainSeed:
    return RefChainSeed(
        paper=ranked.paper,
        rank=ranked.rank,
        score=ranked.final_score,
        reason=ranked.ranking_reason,
    )


def _paper_identifier(paper: Paper) -> str | None:
    identifiers = paper.identifiers
    if identifiers.openalex_id:
        return f"openalex:{identifiers.openalex_id.casefold()}"
    if identifiers.doi:
        return f"doi:{identifiers.doi.casefold()}"
    return None


def _identifier_type(paper: Paper) -> str | None:
    if paper.identifiers.openalex_id:
        return "openalex"
    if paper.identifiers.doi:
        return "doi"
    return None


def _seed_diagnostic(
    ranked: RankedPaper,
    *,
    request_count: int = 0,
    snapshot_key: str | None = None,
    recorded_diagnostics: ConnectorDiagnostics | None = None,
    recorded_latency_seconds: float = 0.0,
    references_returned: int = 0,
    unique_references_returned: int = 0,
    skip_reason: str | None = None,
) -> RefChainSeedDiagnostic:
    recorded = recorded_diagnostics or ConnectorDiagnostics()
    return RefChainSeedDiagnostic(
        seed_id=_paper_identifier(ranked.paper),
        seed_rank=ranked.rank,
        seed_category=ranked.category,
        seed_score=ranked.final_score,
        identifier_type=_identifier_type(ranked.paper),
        request_count=request_count,
        snapshot_key=snapshot_key,
        recorded_request_count=recorded.request_count,
        recorded_retry_count=recorded.retry_count,
        recorded_error_count=recorded.error_count,
        recorded_latency_seconds=recorded_latency_seconds,
        references_returned=references_returned,
        unique_references_returned=unique_references_returned,
        skip_reason=skip_reason,
    )


def _seed_fetch_skip_reason(
    error_message: str | None,
    *,
    batch_status: str | None = None,
    returned_for_seed: int,
    unique_for_seed: int,
) -> str | None:
    normalized = (error_message or "").casefold()
    if "cooldown" in normalized or "429" in normalized:
        return "source_cooldown"
    if batch_status == "partial_success" and returned_for_seed:
        return "partial_success_missing_ids"
    if error_message:
        return "source_failure"
    if returned_for_seed == 0:
        return "no_references"
    if unique_for_seed == 0:
        return "all_references_duplicate"
    return None


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped
