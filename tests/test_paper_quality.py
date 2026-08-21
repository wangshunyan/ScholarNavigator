from __future__ import annotations

from scholar_agent.core.full_text_evidence import build_paragraph_evidence
from scholar_agent.core.paper_quality import assess_paper_quality
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


def test_quality_report_is_explainable_and_deterministic() -> None:
    paper = Paper(
        title="A paper",
        abstract="A complete abstract.",
        year=2025,
        venue="Venue",
        sources=["openalex", "arxiv", "openalex"],
        identifiers=PaperIdentifiers(doi="10.1/example", arxiv_id="2501.00001"),
        full_text_evidence=[
            build_paragraph_evidence(
                "Licensed full text.",
                source_url="https://example.test/paper.txt",
                license_id="CC-BY-4.0",
                license_verified=True,
            )
        ],
    )

    first = assess_paper_quality(paper)
    second = assess_paper_quality(paper)

    assert first == second
    assert first.quality_score == 0.9166666666666666
    by_name = {signal.name: signal for signal in first.signals}
    assert by_name["metadata_completeness"].value == 1.0
    assert by_name["source_corroboration"].value == 1.0
    assert by_name["stable_identifier_coverage"].value == 2 / 3
    assert by_name["licensed_full_text_evidence"].value == 1.0
    assert by_name["retraction_status"].state == "unknown"
    assert by_name["duplicate_risk"].state == "unknown"


def test_missing_metadata_is_not_a_relevance_or_external_risk_claim() -> None:
    report = assess_paper_quality(Paper(title="Title only", citation_count=-1))
    by_name = {signal.name: signal for signal in report.signals}

    assert report.quality_scope == "metadata_only_not_relevance_or_retraction_claim"
    assert by_name["metadata_completeness"].state == "present"
    assert by_name["metadata_completeness"].value == 0.25
    assert by_name["stable_identifier_coverage"].state == "missing"
    assert by_name["retraction_status"].state == "unknown"
    assert by_name["duplicate_risk"].state == "unknown"


def test_source_case_and_duplicates_do_not_inflate_corroboration() -> None:
    report = assess_paper_quality(
        Paper(title="Source test", sources=[" OpenAlex ", "openalex", ""])
    )
    signal = next(item for item in report.signals if item.name == "source_corroboration")

    assert signal.value == 0.5
    assert signal.detail == "unique_sources:1"
