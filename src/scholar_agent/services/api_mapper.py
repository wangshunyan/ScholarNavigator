"""Map internal SearchService output into the public API result schema."""

from __future__ import annotations

from collections import defaultdict

from scholar_agent.core import api_schemas as api
from scholar_agent.core.identity import paper_identifier_set
from scholar_agent.core.paper_schemas import Paper as InternalPaper
from scholar_agent.core.result_lineage import (
    ranked_result_authority_digest,
    result_identity,
)
from scholar_agent.core.untrusted_metadata import (
    opaque_record_identity,
    protect_text,
    protect_url,
    safe_diagnostic_message,
    stable_sha256,
)
from scholar_agent.core.search_schemas import (
    EvidenceItem as InternalEvidenceItem,
    QueryAnalysis as InternalQueryAnalysis,
    RankedPaper as InternalRankedPaper,
    SearchPlan as InternalSearchPlan,
)
from scholar_agent.core.synthesis_schemas import (
    CitationCoverage as InternalCitationCoverage,
    SynthesisEvidenceRow as InternalSynthesisEvidenceRow,
    SynthesisFinding as InternalSynthesisFinding,
    SynthesisOutput as InternalSynthesisOutput,
)
from scholar_agent.evaluation.selection import (
    DEFAULT_RESULT_POLICY,
    select_ranked_results,
)
from scholar_agent.services.search_service import SearchServiceOutput


_METHOD_CLUSTER_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reranking", ("reranking", "re-ranking", "rerank", "re-rank")),
    ("retrieval", ("retrieval", "retriever")),
    (
        "RAG",
        (
            "rag",
            "retrieval-augmented",
            "retrieval augmented generation",
            "retrieval-augmented generation",
        ),
    ),
    (
        "citation graph",
        ("citation graph", "citation network", "reference chain", "refchain"),
    ),
    ("benchmark", ("benchmark", "evaluation", "dataset", "trec")),
    ("agent", ("agent", "agents")),
    ("recommendation", ("recommendation", "recommender", "recommend")),
)
_PUBLIC_QUERY_IDENTITY = f"query:{stable_sha256({'boundary': 'public_api'})}"


def map_search_service_output_to_api_result(
    run_id: str,
    output: SearchServiceOutput,
    *,
    status: str = "succeeded",
    partial: bool = False,
) -> api.SearchRunResultResponse:
    """Map internal no-LLM SearchService output into the API result contract."""

    highly_relevant: list[api.RankedPaper] = []
    partially_relevant: list[api.RankedPaper] = []
    missing_evidence = _missing_evidence(output)

    selected = select_ranked_results(output, policy=DEFAULT_RESULT_POLICY)
    selected_object_ids = {id(item) for item in selected}
    for ranked in output.ranked_papers:
        mapped = map_ranked_paper(ranked)
        if id(ranked) in selected_object_ids and ranked.category == "highly_relevant":
            highly_relevant.append(mapped)
        elif (
            id(ranked) in selected_object_ids
            and ranked.category == "partially_relevant"
        ):
            partially_relevant.append(mapped)
        elif ranked.category in {
            "weakly_relevant",
            "irrelevant",
            "insufficient_evidence",
        }:
            missing_evidence.append(_filtered_paper_diagnostic(ranked))
            identifier = _filtered_paper_identifier(ranked.paper)
            if identifier:
                missing_evidence.append(
                    f"filtered_paper_identifier:{ranked.rank}:{identifier}"
                )
            missing_evidence.extend(
                f"filtered_paper_warning:{ranked.rank}:{warning}"
                for warning in ranked.warnings
            )

    all_visible = highly_relevant + partially_relevant
    return api.SearchRunResultResponse(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        partial=partial or output.budget_status.exhausted,
        query_analysis=map_query_analysis(output.search_plan.query_analysis),
        search_plan=map_search_plan(output.search_plan, output),
        highly_relevant_papers=highly_relevant,
        partially_relevant_papers=partially_relevant,
        method_clusters=_method_clusters(output.search_plan.query_analysis, all_visible),
        timeline=_timeline(all_visible),
        citation_graph=_citation_graph(all_visible, output),
        warnings=[safe_diagnostic_message(item) for item in output.warnings],
        missing_evidence=_dedupe(missing_evidence),
        synthesis=map_synthesis_output(output.synthesis_output),
        retrieval_diagnostics=_retrieval_diagnostics(output),
        budget_status=output.budget_status,
        cost_report=_cost_report(output),
        judgement_policy=output.judgement_policy,
        judgement_config_hash=output.judgement_config_hash,
    )


def map_final_ranked_papers(
    ranked_papers: list[InternalRankedPaper],
) -> list[api.RankedPaper]:
    """Map the formal public result set using the shared production selector.

    This narrow helper is also the delivery audit boundary: it applies the same
    Top-K/category contract as the API without constructing unrelated run
    diagnostics or introducing a second result-selection implementation.
    """

    selected = select_ranked_results(
        {"ranked_papers": ranked_papers},
        policy=DEFAULT_RESULT_POLICY,
    )
    return [map_ranked_paper(item) for item in selected]


def map_paper(paper: InternalPaper) -> api.Paper:
    """Map an internal Paper into an API Paper."""

    record_identity = opaque_record_identity(paper)
    return api.Paper(
        title=_public_text(paper.title, "paper.title", record_identity),
        authors=[
            _public_text(author, "paper.author", record_identity)
            for author in paper.authors
        ],
        year=paper.year,
        venue=(
            _public_text(paper.venue, "paper.venue", record_identity)
            if paper.venue is not None
            else None
        ),
        abstract=_public_text(paper.abstract, "paper.abstract", record_identity),
        identifiers=api.PaperIdentifiers(
            doi=paper.identifiers.doi,
            arxiv_id=paper.identifiers.arxiv_id,
            semantic_scholar_id=paper.identifiers.semantic_scholar_id,
            s2orc_corpus_id=paper.identifiers.s2orc_corpus_id,
            openalex_id=paper.identifiers.openalex_id,
            pubmed_id=paper.identifiers.pubmed_id,
        ),
        urls=api.PaperUrls(
            landing_page=_public_url(
                paper.urls.landing_page,
                "paper.urls.landing_page",
                record_identity,
            ),
            pdf=_public_url(
                paper.urls.pdf,
                "paper.urls.pdf",
                record_identity,
            ),
        ),
        sources=list(paper.sources),
        full_text_evidence=[
            api.FullTextEvidenceDocument(
                schema_version=document.schema_version,
                source_url=_public_url(
                    document.source.source_url,
                    "paper.urls.landing_page",
                    record_identity,
                )
                or "",
                license_id=document.source.license_id,
                content_sha256=document.source.content_sha256,
                paragraphs=[
                    api.FullTextParagraphEvidence(
                        evidence_id=paragraph.evidence_id,
                        paragraph_index=paragraph.paragraph_index,
                        text=_public_text(
                            paragraph.text,
                            "paper.abstract",
                            record_identity,
                        ),
                        text_sha256=paragraph.text_sha256,
                        start_char=paragraph.start_char,
                        end_char=paragraph.end_char,
                    )
                    for paragraph in document.paragraphs
                ],
            )
            for document in paper.full_text_evidence
        ],
    )


def map_ranked_paper(ranked: InternalRankedPaper) -> api.RankedPaper:
    """Map an internal RankedPaper into an API RankedPaper."""

    return api.RankedPaper(
        result_identity=result_identity(ranked.paper),
        authority_digest=ranked_result_authority_digest(ranked),
        rank=ranked.rank,
        paper=map_paper(ranked.paper),
        relevance_score=ranked.final_score,
        category=ranked.category,
        matched_constraints=list(ranked.matched_terms),
        ranking_reason=ranked.ranking_reason,
        evidence=[map_evidence_item(item) for item in ranked.evidence],
        rrf_score=ranked.rrf_score,
        rrf_contributions=[
            api.RRFListContribution(**item.model_dump(mode="json"))
            for item in ranked.rrf_contributions
        ],
        original_rank=ranked.original_rank,
        rrf_top_20_change=ranked.rrf_top_20_change,
        rrf_rank_change_reason=ranked.rrf_rank_change_reason,
        quality_policy=ranked.quality_policy,
        quality_score=ranked.quality_score,
        quality_contribution=ranked.quality_contribution,
        quality_config_hash=ranked.quality_config_hash,
        quality_rank_change_reason=ranked.quality_rank_change_reason,
        quality_signals=[
            api.QualitySignal(**signal.model_dump(mode="json"))
            for signal in ranked.quality_signals
        ],
    )


def map_evidence_item(item: InternalEvidenceItem) -> api.EvidenceItem:
    """Map an internal evidence item into an API evidence item."""

    return api.EvidenceItem(
        source=item.source,
        text=_public_text(
            item.text,
            "paper.abstract",
            f"record:{stable_sha256({'evidence': item.model_dump(mode='json')})}",
        ),
        confidence=item.confidence,
    )


def map_synthesis_output(
    synthesis: InternalSynthesisOutput | None,
) -> api.SynthesisOutput | None:
    """Map internal synthesis output into the optional API synthesis object."""

    if synthesis is None:
        return None
    return api.SynthesisOutput(
        answer_summary=_public_generated_text(synthesis.answer_summary),
        status=synthesis.status,
        key_findings=[
            map_synthesis_finding(finding) for finding in synthesis.key_findings
        ],
        evidence_table=[
            map_synthesis_evidence_row(row) for row in synthesis.evidence_table
        ],
        citation_coverage=map_citation_coverage(synthesis.citation_coverage),
        limitations=[_public_generated_text(item) for item in synthesis.limitations],
        warnings=[safe_diagnostic_message(item) for item in synthesis.warnings],
    )


def map_synthesis_evidence_row(
    row: InternalSynthesisEvidenceRow,
) -> api.SynthesisEvidenceRow:
    """Map an internal synthesis evidence row into the API schema."""

    return api.SynthesisEvidenceRow(
        row_id=row.row_id,
        citation_key=row.citation_key,
        rank=row.rank,
        paper_title=_public_generated_text(row.paper_title),
        year=row.year,
        venue=_public_generated_text(row.venue) if row.venue is not None else None,
        sources=list(row.sources),
        identifiers=api.PaperIdentifiers(
            doi=row.identifiers.doi,
            arxiv_id=row.identifiers.arxiv_id,
            semantic_scholar_id=row.identifiers.semantic_scholar_id,
            s2orc_corpus_id=row.identifiers.s2orc_corpus_id,
            openalex_id=row.identifiers.openalex_id,
            pubmed_id=row.identifiers.pubmed_id,
        ),
        category=row.category,
        final_score=row.final_score,
        evidence_source=row.evidence_source,
        evidence_text=_public_generated_text(row.evidence_text),
        supported_terms=list(row.supported_terms),
        supported_claim=_public_generated_text(row.supported_claim),
    )


def map_synthesis_finding(
    finding: InternalSynthesisFinding,
) -> api.SynthesisFinding:
    """Map an internal synthesis finding into the API schema."""

    return api.SynthesisFinding(
        text=_public_generated_text(finding.text),
        citation_keys=list(finding.citation_keys),
        confidence=finding.confidence,
        evidence_row_ids=list(finding.evidence_row_ids),
    )


def map_citation_coverage(
    coverage: InternalCitationCoverage,
) -> api.CitationCoverage:
    """Map internal citation coverage counters into the API schema."""

    return api.CitationCoverage(
        ranked_paper_count=coverage.ranked_paper_count,
        cited_paper_count=coverage.cited_paper_count,
        evidence_row_count=coverage.evidence_row_count,
        cited_evidence_row_count=coverage.cited_evidence_row_count,
        missing_evidence_count=coverage.missing_evidence_count,
        source_error_count=coverage.source_error_count,
        coverage_ratio=coverage.coverage_ratio,
    )


def map_query_analysis(query_analysis: InternalQueryAnalysis) -> api.QueryAnalysis:
    """Map internal query analysis into the API query analysis schema."""

    constraints = query_analysis.constraints
    time_range = constraints.time_range
    constraint_payload = {
        "time_range": (
            {
                "start_year": time_range.start_year,
                "end_year": time_range.end_year,
                "label": time_range.label,
            }
            if time_range is not None
            else None
        ),
        "venues": list(constraints.venues),
        "methods": list(constraints.methods),
        "datasets": list(constraints.datasets),
        "domains": list(constraints.domains),
        "must_have_terms": list(constraints.must_include_terms),
        "excluded_terms": list(constraints.exclude_terms),
        "paper_types": list(constraints.paper_types),
        "language": query_analysis.language,
        "needs_expansion": query_analysis.needs_expansion,
    }
    topics = _dedupe(
        constraints.methods
        + constraints.datasets
        + constraints.domains
        + constraints.must_include_terms
        + list(constraints.paper_types)
    )
    return api.QueryAnalysis(
        intent_type=query_analysis.intent,
        domain=query_analysis.domain,
        research_topics=topics,
        constraints=constraint_payload,
    )


def map_search_plan(
    search_plan: InternalSearchPlan,
    output: SearchServiceOutput | None = None,
) -> api.SearchPlan:
    """Map internal search plan into the API search plan schema."""

    expanded_queries = [subquery.query for subquery in search_plan.subqueries]
    if output is not None:
        for record in output.query_evolution_records:
            expanded_queries.extend(query.query for query in record.generated_queries)
    return api.SearchPlan(
        expanded_queries=_dedupe(expanded_queries),
        source_preferences=list(search_plan.selected_sources),
        max_rounds=_search_rounds(output, search_plan),
        query_planning_policy=search_plan.query_planning_policy,
        ranking_policy=search_plan.ranking_policy,
        query_planning=search_plan.query_planning,
        query_evolution_policy=search_plan.query_evolution_policy,
        enable_semantic_seed_expansion=(
            search_plan.enable_semantic_seed_expansion
        ),
    )


def _cost_report(output: SearchServiceOutput) -> api.CostReport:
    search = output.search_diagnostics
    reference = output.reference_diagnostics
    connector_request_count = search.request_count + reference.request_count
    return api.CostReport(
        api_call_count=connector_request_count + output.llm_call_count,
        logical_search_call_count=sum(
            stat.source not in {"refchain", "semantic_seed_expansion"}
            and stat.logical_call_executed
            for stat in output.source_stats
        ),
        search_api_call_count=search.request_count,
        reference_api_call_count=reference.request_count,
        retry_count=search.retry_count + reference.retry_count,
        error_count=search.error_count + reference.error_count,
        llm_call_count=output.llm_call_count,
        llm_prompt_tokens=output.llm_prompt_tokens,
        llm_completion_tokens=output.llm_completion_tokens,
        llm_total_tokens=output.llm_total_tokens,
        estimated_input_tokens=output.llm_prompt_tokens,
        estimated_output_tokens=output.llm_completion_tokens,
        estimated_total_tokens=output.llm_total_tokens,
        latency_seconds=output.latency_seconds,
        cache_hit_count=search.cache_hit_count + reference.cache_hit_count,
        rate_limit_wait_seconds=(
            search.rate_limit_wait_seconds + reference.rate_limit_wait_seconds
        ),
        search_rounds=_search_rounds(output, output.search_plan),
        judged_paper_count=len(output.judgements),
        raw_candidate_count=output.raw_count,
        deduplicated_candidate_count=output.deduplicated_count,
    )


def _retrieval_diagnostics(output: SearchServiceOutput) -> api.RetrievalDiagnostics:
    return api.RetrievalDiagnostics(
        raw_count=output.raw_count,
        deduplicated_count=output.deduplicated_count,
        source_stats=[
            api.RetrievalSourceStats(
                source=stats.source,
                returned_count=stats.returned_count,
                latency_seconds=stats.latency_seconds,
                cache_hit=stats.cache_hit,
                logical_call_executed=stats.logical_call_executed,
                adaptation_strategy=stats.adaptation_strategy,
                triggered_by=list(stats.triggered_by),
                safe_original_candidate_count=stats.safe_original_candidate_count,
                safe_original_core_term_coverage=(
                    stats.safe_original_core_term_coverage
                ),
                safe_original_constraint_coverage=(
                    stats.safe_original_constraint_coverage
                ),
                sufficiency_reasons=list(stats.sufficiency_reasons),
                compact_query_executed=stats.compact_query_executed,
                compact_query_skipped_reason=stats.compact_query_skipped_reason,
                error_message=(
                    safe_diagnostic_message(stats.error_message)
                    if stats.error_message is not None
                    else None
                ),
                diagnostics=stats.diagnostics,
            )
            for stats in output.source_stats
        ],
    )


def _search_rounds(
    output: SearchServiceOutput | None,
    search_plan: InternalSearchPlan,
) -> int:
    if output is None:
        return 1 if search_plan.subqueries else 0
    return output.budget_status.completed_search_rounds


def _method_clusters(
    query_analysis: InternalQueryAnalysis,
    ranked_papers: list[api.RankedPaper],
) -> list[api.MethodCluster]:
    clusters: list[api.MethodCluster] = []

    for name, aliases in _method_cluster_candidates(query_analysis):
        paper_ranks = sorted(
            paper.rank for paper in ranked_papers if _paper_matches_any_alias(paper, aliases)
        )
        if paper_ranks:
            clusters.append(
                api.MethodCluster(
                    name=name,
                    paper_ranks=paper_ranks,
                    summary=(
                        f"Ranks {_rank_list(paper_ranks)} discuss {name} based on "
                        "title, abstract, or evidence signals."
                    ),
                )
            )

    clusters = _dedupe_method_clusters(clusters)
    if clusters or not ranked_papers:
        return clusters

    ranks = sorted(paper.rank for paper in ranked_papers)
    return [
        api.MethodCluster(
            name="general",
            paper_ranks=ranks,
            summary=(
                f"Ranks {_rank_list(ranks)} are grouped as general results because "
                "no method-specific keyword evidence was available."
            ),
        )
    ]


def _timeline(ranked_papers: list[api.RankedPaper]) -> list[api.TimelineItem]:
    ranks_by_year: dict[int, list[int]] = defaultdict(list)
    for paper in ranked_papers:
        year = paper.paper.year
        if isinstance(year, int) and year > 0:
            ranks_by_year[year].append(paper.rank)
    return [
        api.TimelineItem(
            year=year,
            paper_ranks=sorted(ranks),
            summary=f"Ranks {_rank_list(sorted(ranks))} were published in {year}.",
        )
        for year, ranks in sorted(ranks_by_year.items())
    ]


def _method_cluster_candidates(
    query_analysis: InternalQueryAnalysis,
) -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    known_names = {name.casefold() for name, _ in _METHOD_CLUSTER_KEYWORDS}

    for name, aliases in _METHOD_CLUSTER_KEYWORDS:
        candidates.append((name, aliases))

    for method in query_analysis.constraints.methods:
        method_name = method.strip()
        if not method_name or method_name.casefold() in known_names:
            continue
        candidates.append((method_name, (method_name,)))

    return candidates


def _paper_matches_any_alias(
    ranked_paper: api.RankedPaper,
    aliases: tuple[str, ...],
) -> bool:
    searchable_text = _cluster_search_text(ranked_paper)
    return any(alias.casefold() in searchable_text for alias in aliases)


def _cluster_search_text(ranked_paper: api.RankedPaper) -> str:
    evidence_text = " ".join(item.text for item in ranked_paper.evidence)
    return " ".join(
        [
            ranked_paper.paper.title,
            ranked_paper.paper.abstract,
            evidence_text,
        ]
    ).casefold()


def _dedupe_method_clusters(
    clusters: list[api.MethodCluster],
) -> list[api.MethodCluster]:
    by_name: dict[str, api.MethodCluster] = {}
    for cluster in clusters:
        key = cluster.name.casefold()
        if key not in by_name:
            by_name[key] = cluster
            continue
        merged_ranks = sorted(set(by_name[key].paper_ranks + cluster.paper_ranks))
        by_name[key] = api.MethodCluster(
            name=by_name[key].name,
            paper_ranks=merged_ranks,
            summary=(
                f"Ranks {_rank_list(merged_ranks)} discuss {by_name[key].name} "
                "based on title, abstract, or evidence signals."
            ),
        )
    return list(by_name.values())


def _rank_list(ranks: list[int]) -> str:
    return ", ".join(f"R{rank}" for rank in ranks)


def _citation_graph(
    ranked_papers: list[api.RankedPaper],
    output: SearchServiceOutput,
) -> api.CitationGraph:
    node_by_id: dict[str, api.CitationGraphNode] = {}

    for ranked in ranked_papers:
        node_id = _paper_node_id(ranked.paper)
        if node_id is None:
            continue
        node_by_id[node_id] = api.CitationGraphNode(
            id=node_id,
            label=ranked.paper.title,
            rank=ranked.rank,
        )

    edges: list[api.CitationGraphEdge] = []
    if output.refchain_output is not None:
        for edge in output.refchain_output.reference_edges:
            if edge.seed_paper_id not in node_by_id:
                node_by_id[edge.seed_paper_id] = api.CitationGraphNode(
                    id=edge.seed_paper_id,
                    label=edge.seed_paper_id,
                )
            if edge.reference_paper_id not in node_by_id:
                node_by_id[edge.reference_paper_id] = api.CitationGraphNode(
                    id=edge.reference_paper_id,
                    label=edge.reference_paper_id,
                )
            edges.append(
                api.CitationGraphEdge(
                    source=edge.seed_paper_id,
                    target=edge.reference_paper_id,
                    relation=edge.relation,
                )
            )

    return api.CitationGraph(
        nodes=list(node_by_id.values()),
        edges=edges,
    )


def _missing_evidence(output: SearchServiceOutput) -> list[str]:
    missing: list[str] = []
    missing.extend(
        safe_diagnostic_message(warning)
        for warning in output.warnings
        if not _is_budget_diagnostic(warning)
    )
    for stage, seconds in output.stage_latencies.items():
        missing.append(f"stage_latency:{stage}:{seconds:.6f}")

    for stats in output.source_stats:
        if stats.error_message and not _is_budget_diagnostic(stats.error_message):
            missing.append(
                f"source_error:{stats.source}:"
                f"{safe_diagnostic_message(stats.error_message)}"
            )

    for record in output.query_evolution_records:
        missing.append(
            "query_evolution:"
            f"round={record.round_index}:"
            f"seed_count={record.seed_count}:"
            f"generated_count={len(record.generated_queries)}"
        )
        missing.extend(
            f"query_evolution_warning:{warning}"
            for warning in record.warnings
            if not _is_budget_diagnostic(warning)
        )
        missing.extend(
            f"query_evolution_skipped:{reason}"
            for reason in record.skipped_reasons
            if not _is_budget_diagnostic(reason)
        )

    if output.refchain_output is not None:
        record = output.refchain_output.record
        missing.append(
            "refchain:"
            f"seed_count={len(record.seeds)}:"
            f"returned_reference_count={record.returned_reference_count}"
        )
        missing.extend(
            f"refchain_warning:{warning}"
            for warning in record.warnings
            if not _is_budget_diagnostic(warning)
        )
        missing.extend(
            f"refchain_skipped:{reason}"
            for reason in record.skipped_reasons
            if not _is_budget_diagnostic(reason)
        )

    return missing


def _is_budget_diagnostic(value: str) -> bool:
    return "budget_stop:" in value or "budget_diagnostic:" in value


def _filtered_paper_diagnostic(ranked: InternalRankedPaper) -> str:
    title = _public_text(
        ranked.paper.title,
        "paper.title",
        opaque_record_identity(ranked.paper),
    )
    return (
        f"filtered_paper:{ranked.rank}:{ranked.category}:"
        f"{ranked.final_score:.4f}:{title}"
    )


def _filtered_paper_identifier(paper: InternalPaper) -> str | None:
    identifiers = paper_identifier_set(paper)
    for prefix in ("arxiv:", "doi:", "openalex:", "s2:", "pubmed:"):
        matches = sorted(item for item in identifiers if item.startswith(prefix))
        if matches:
            return matches[0]
    return None


def _paper_node_id(paper: api.Paper) -> str | None:
    identifiers = paper_identifier_set(paper)
    for prefix in ("arxiv:", "doi:", "openalex:", "s2:", "pubmed:"):
        matches = sorted(item for item in identifiers if item.startswith(prefix))
        if matches:
            return matches[0]
    return None


def _public_text(value: object, field: str, record_identity: str) -> str:
    return protect_text(
        value,
        field=field,
        query_identity=_PUBLIC_QUERY_IDENTITY,
        result_identity=record_identity,
    )


def _public_url(value: object, field: str, record_identity: str) -> str | None:
    if value is None:
        return None
    return protect_url(
        value,
        field=field,
        query_identity=_PUBLIC_QUERY_IDENTITY,
        result_identity=record_identity,
    )


def _public_generated_text(value: object) -> str:
    return _public_text(
        value,
        "paper.abstract",
        f"record:{stable_sha256({'generated_text': str(value)})}",
    )


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped
