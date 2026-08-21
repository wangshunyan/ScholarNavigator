from __future__ import annotations

import pytest

from scholar_agent.core.full_text_evidence import build_paragraph_evidence
from scholar_agent.core.identity import paper_identifier_set
from scholar_agent.core.paper_quality import VerifiedQualityEvidence, assess_paper_quality
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


def test_verified_risk_evidence_is_opt_in_and_changes_only_known_signal() -> None:
    paper = Paper(
        title="Evidence-backed paper",
        abstract="Abstract",
        year=2025,
        identifiers=PaperIdentifiers(doi="10.1/evidence"),
    )
    identifier = next(iter(paper_identifier_set(paper)))
    evidence = [
        VerifiedQualityEvidence(
            paper_identifier=identifier,
            signal_name="retraction_status",
            state="clear",
            source="verified_registry",
            source_record_id="registry-record-1",
        ),
        VerifiedQualityEvidence(
            paper_identifier=identifier,
            signal_name="duplicate_risk",
            state="flagged",
            source="verified_registry",
            source_record_id="registry-record-2",
        ),
    ]

    without = assess_paper_quality(paper)
    with_evidence = assess_paper_quality(paper, verified_evidence=evidence)
    signals = {item.name: item for item in with_evidence.signals}

    assert without.external_risk_data_available is False
    assert with_evidence.external_risk_data_available is True
    assert with_evidence.quality_scope == "metadata_plus_verified_risk_not_relevance_claim"
    assert signals["retraction_status"].state == "present"
    assert signals["retraction_status"].value == 1.0
    assert signals["duplicate_risk"].state == "missing"
    assert signals["duplicate_risk"].value == 0.0
    assert "registry-record-1" not in signals["retraction_status"].detail


def test_unmatched_external_evidence_remains_unknown_without_penalty() -> None:
    paper = Paper(
        title="No matching evidence",
        identifiers=PaperIdentifiers(doi="10.1/known"),
    )
    evidence = VerifiedQualityEvidence(
        paper_identifier="doi:10.1/other",
        signal_name="retraction_status",
        state="flagged",
        source="registry",
        source_record_id="other-record",
    )

    first = assess_paper_quality(paper)
    second = assess_paper_quality(paper, verified_evidence=[evidence])

    assert second == first
    assert all(
        item.state == "unknown"
        for item in second.signals
        if item.name in {"retraction_status", "duplicate_risk"}
    )


def test_conflicting_verified_evidence_fails_closed() -> None:
    paper = Paper(
        title="Conflicting evidence",
        identifiers=PaperIdentifiers(doi="10.1/conflict"),
    )
    identifier = next(iter(paper_identifier_set(paper)))
    evidence = [
        VerifiedQualityEvidence(
            paper_identifier=identifier,
            signal_name="duplicate_risk",
            state="clear",
            source="registry_a",
            source_record_id="a",
        ),
        VerifiedQualityEvidence(
            paper_identifier=identifier,
            signal_name="duplicate_risk",
            state="flagged",
            source="registry_b",
            source_record_id="b",
        ),
    ]

    with pytest.raises(ValueError, match="conflicting_verified_quality_evidence"):
        assess_paper_quality(paper, verified_evidence=evidence)
