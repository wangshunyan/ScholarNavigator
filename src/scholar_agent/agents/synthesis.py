"""Rule-based citation-backed synthesis agent."""

from __future__ import annotations

from collections.abc import Iterable

from scholar_agent.core.search_schemas import EvidenceItem, RankedPaper
from scholar_agent.core.synthesis_schemas import (
    CitationCoverage,
    SynthesisEvidenceRow,
    SynthesisFinding,
    SynthesisOptions,
    SynthesisOutput,
)
from scholar_agent.core.untrusted_metadata import safe_diagnostic_message
from scholar_agent.services.search_service import SearchServiceOutput


SUPPORTED_EVIDENCE_SOURCES = {"title", "abstract", "venue", "metadata"}


def synthesize_answer(
    search_output: SearchServiceOutput,
    options: SynthesisOptions | None = None,
) -> SynthesisOutput:
    """Create a deterministic, metadata-grounded synthesis from search output."""

    opts = options or SynthesisOptions()
    limitations = _collect_limitations(search_output)
    evidence_rows, filtered_warnings = _build_evidence_rows(
        search_output.ranked_papers,
        opts,
    )
    warnings = _dedupe(filtered_warnings)

    if not evidence_rows:
        limitations = _dedupe(
            limitations
            + [
                "insufficient_evidence:no_supported_evidence_rows",
                "full_text_evidence_unavailable",
            ]
        )
        return SynthesisOutput(
            answer_summary=(
                "Insufficient evidence: no citation-backed evidence rows were "
                "available from the ranked papers, so no answer summary was "
                "generated."
            ),
            status="insufficient_evidence",
            key_findings=[],
            evidence_table=[],
            citation_coverage=_coverage(search_output, [], []),
            limitations=limitations,
            warnings=warnings,
        )

    limitations.extend(_evidence_limitations(evidence_rows))
    findings = _build_findings(evidence_rows, opts)
    cited_keys = _cited_keys(findings)
    answer_summary = _answer_summary(search_output, evidence_rows, findings)

    return SynthesisOutput(
        answer_summary=answer_summary,
        status="succeeded",
        key_findings=findings,
        evidence_table=evidence_rows,
        citation_coverage=_coverage(search_output, evidence_rows, cited_keys),
        limitations=_dedupe(limitations),
        warnings=warnings,
    )


def _build_evidence_rows(
    ranked_papers: list[RankedPaper],
    options: SynthesisOptions,
) -> tuple[list[SynthesisEvidenceRow], list[str]]:
    rows: list[SynthesisEvidenceRow] = []
    warnings: list[str] = []
    cited_paper_count = 0

    for ranked in ranked_papers:
        if cited_paper_count >= options.max_cited_papers:
            break
        if ranked.category in {"irrelevant", "insufficient_evidence"}:
            continue

        valid_evidence = _valid_evidence_items(ranked.evidence)
        unsupported_count = len(ranked.evidence) - len(valid_evidence)
        if unsupported_count:
            warnings.append(
                f"unsupported_evidence_filtered:rank={ranked.rank}:"
                f"count={unsupported_count}"
            )
        if not valid_evidence:
            continue

        citation_key = f"R{ranked.rank}"
        cited_paper_count += 1
        for evidence_index, evidence in enumerate(
            valid_evidence[: options.max_evidence_rows_per_paper],
            start=1,
        ):
            row_id = f"{citation_key}-E{evidence_index}"
            evidence_text = _clip(evidence.text, options.evidence_snippet_chars)
            rows.append(
                SynthesisEvidenceRow(
                    row_id=row_id,
                    citation_key=citation_key,
                    rank=ranked.rank,
                    paper_title=ranked.paper.title,
                    year=ranked.paper.year,
                    venue=ranked.paper.venue,
                    sources=list(ranked.paper.sources),
                    identifiers=ranked.paper.identifiers,
                    category=ranked.category,
                    final_score=ranked.final_score,
                    evidence_source=evidence.source,
                    evidence_text=evidence_text,
                    supported_terms=list(ranked.matched_terms),
                    supported_claim=_supported_claim(ranked, evidence),
                )
            )

    return rows, warnings


def _valid_evidence_items(evidence_items: list[EvidenceItem]) -> list[EvidenceItem]:
    valid: list[EvidenceItem] = []
    for item in evidence_items:
        source = str(item.source)
        if source not in SUPPORTED_EVIDENCE_SOURCES:
            continue
        if not item.text.strip():
            continue
        valid.append(item)
    return valid


def _supported_claim(ranked: RankedPaper, evidence: EvidenceItem) -> str:
    terms = ", ".join(ranked.matched_terms[:3])
    if terms:
        return (
            f"{ranked.paper.title} has {evidence.source} evidence related to "
            f"{terms}."
        )
    return f"{ranked.paper.title} has {evidence.source} evidence relevant to the query."


def _build_findings(
    evidence_rows: list[SynthesisEvidenceRow],
    options: SynthesisOptions,
) -> list[SynthesisFinding]:
    findings: list[SynthesisFinding] = []
    seen_keys: set[str] = set()

    for row in evidence_rows:
        if len(findings) >= options.max_findings:
            break
        finding_key = row.citation_key
        if finding_key in seen_keys:
            continue
        seen_keys.add(finding_key)
        topic = _topic_label(row)
        findings.append(
            SynthesisFinding(
                text=(
                    f"{row.paper_title} provides {row.evidence_source} evidence "
                    f"for {topic} [{row.citation_key}]."
                ),
                citation_keys=[row.citation_key],
                confidence=row.final_score,
                evidence_row_ids=[row.row_id],
            )
        )

    return [finding for finding in findings if finding.citation_keys]


def _topic_label(row: SynthesisEvidenceRow) -> str:
    if row.supported_terms:
        return ", ".join(row.supported_terms[:3])
    if row.venue:
        return f"evidence from {row.venue}"
    return "the search query"


def _answer_summary(
    search_output: SearchServiceOutput,
    evidence_rows: list[SynthesisEvidenceRow],
    findings: list[SynthesisFinding],
) -> str:
    query = search_output.search_plan.query_analysis.original_query
    intent = search_output.search_plan.query_analysis.intent
    domain = search_output.search_plan.query_analysis.domain
    top_keys = _dedupe(row.citation_key for row in evidence_rows)[:3]
    cited = ", ".join(f"[{key}]" for key in top_keys)
    themes = _top_terms(evidence_rows)
    theme_text = ", ".join(themes) if themes else "the retrieved evidence"
    finding_count = len(findings)
    return (
        f"For the query \"{query}\", the current {domain} search evidence "
        f"supports a {intent} synthesis around {theme_text}. The strongest "
        f"citation-backed candidates are {cited}. {finding_count} finding(s) "
        "were generated only from ranked-paper evidence rows."
    )


def _top_terms(evidence_rows: list[SynthesisEvidenceRow]) -> list[str]:
    counts: dict[str, int] = {}
    for row in evidence_rows:
        for term in row.supported_terms:
            key = term.strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return [
        term
        for term, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:4]
    ]


def _coverage(
    search_output: SearchServiceOutput,
    evidence_rows: list[SynthesisEvidenceRow],
    cited_keys: Iterable[str],
) -> CitationCoverage:
    cited_key_set = set(cited_keys)
    cited_paper_keys = {row.citation_key for row in evidence_rows}
    cited_evidence_count = sum(
        1 for row in evidence_rows if row.citation_key in cited_key_set
    )
    ranked_count = len(search_output.ranked_papers)
    coverage_ratio = (
        len(cited_paper_keys) / ranked_count if ranked_count > 0 else 0.0
    )
    return CitationCoverage(
        ranked_paper_count=ranked_count,
        cited_paper_count=len(cited_paper_keys),
        evidence_row_count=len(evidence_rows),
        cited_evidence_row_count=cited_evidence_count,
        missing_evidence_count=len(search_output.warnings)
        + sum(1 for stats in search_output.source_stats if stats.error_message),
        source_error_count=sum(
            1 for stats in search_output.source_stats if stats.error_message
        ),
        coverage_ratio=round(min(1.0, coverage_ratio), 4),
    )


def _collect_limitations(search_output: SearchServiceOutput) -> list[str]:
    limitations: list[str] = []
    limitations.extend(
        safe_diagnostic_message(item) for item in search_output.warnings
    )
    for stats in search_output.source_stats:
        if stats.error_message:
            limitations.append(
                f"source_error:{stats.source}:"
                f"{safe_diagnostic_message(stats.error_message)}"
            )
    if search_output.refchain_output is None:
        limitations.append("refchain_not_enabled_or_not_available")
    else:
        limitations.extend(search_output.refchain_output.warnings)
    return _dedupe(limitations)


def _evidence_limitations(evidence_rows: list[SynthesisEvidenceRow]) -> list[str]:
    limitations = ["full_text_evidence_unavailable"]
    sources = {row.evidence_source for row in evidence_rows}
    metadata_like = {"title", "venue", "metadata"}
    if sources and sources.issubset(metadata_like):
        limitations.append("metadata_only_evidence:no_abstract_or_full_text_evidence_used")
    elif "metadata" in sources:
        limitations.append("metadata_evidence_used")
    return limitations


def _cited_keys(findings: list[SynthesisFinding]) -> list[str]:
    keys: list[str] = []
    for finding in findings:
        keys.extend(finding.citation_keys)
    return _dedupe(keys)


def _clip(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _dedupe(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped
