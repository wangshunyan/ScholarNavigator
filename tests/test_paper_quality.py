from __future__ import annotations

import json
import hashlib

import pytest

from scholar_agent.core.full_text_evidence import build_paragraph_evidence
from scholar_agent.core.identity import paper_identifier_set
from scholar_agent.core.paper_quality import (
    VerifiedQualityEvidence,
    assess_paper_quality,
    bind_verified_quality_evidence_to_candidates,
    load_verified_quality_evidence_ledger,
)
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
    assert by_name["author_metadata"].state == "missing"
    assert by_name["doi_metadata"].state == "present"
    assert by_name["identity_consistency"].state == "unknown"
    assert by_name["retraction_status"].state == "unknown"
    assert by_name["duplicate_risk"].state == "unknown"


def test_missing_metadata_is_not_a_relevance_or_external_risk_claim() -> None:
    report = assess_paper_quality(Paper(title="Title only", citation_count=-1))
    by_name = {signal.name: signal for signal in report.signals}

    assert report.quality_scope == "metadata_only_not_relevance_or_retraction_claim"
    assert by_name["metadata_completeness"].state == "present"
    assert by_name["metadata_completeness"].value == 0.25
    assert by_name["stable_identifier_coverage"].state == "missing"
    assert by_name["author_metadata"].state == "missing"
    assert by_name["doi_metadata"].state == "missing"
    assert by_name["identity_consistency"].state == "missing"
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


def test_arxiv_doi_identity_consistency_is_explicit_but_not_scored() -> None:
    matching = Paper(
        title="Matching identity",
        identifiers=PaperIdentifiers(
            arxiv_id="2401.00001",
            doi="10.48550/arxiv.2401.00001",
        ),
    )
    conflicting = matching.model_copy(
        update={"identifiers": PaperIdentifiers(
            arxiv_id="2401.00001",
            doi="10.48550/arxiv.2401.00002",
        )}
    )

    matching_report = assess_paper_quality(matching)
    conflicting_report = assess_paper_quality(conflicting)
    matching_signal = next(
        item for item in matching_report.signals if item.name == "identity_consistency"
    )
    conflicting_signal = next(
        item for item in conflicting_report.signals if item.name == "identity_consistency"
    )

    assert matching_signal.state == "present"
    assert conflicting_signal.state == "missing"
    for report in (matching_report, conflicting_report):
        scored = [
            signal.value
            for signal in report.signals
            if signal.name
            in {
                "metadata_completeness",
                "source_corroboration",
                "stable_identifier_coverage",
                "licensed_full_text_evidence",
                "retraction_status",
                "duplicate_risk",
            }
            and signal.value is not None
        ]
        assert report.quality_score == sum(scored) / len(scored)


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


def test_quality_evidence_ledger_is_strict_and_semantically_stable(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        {
            "paper_identifier": "doi:10.1/first",
            "signal_name": "retraction_status",
            "state": "clear",
            "source": "registry_a",
            "source_record_id": "a-1",
        },
        {
            "paper_identifier": "arxiv:2401.12345",
            "signal_name": "duplicate_risk",
            "state": "flagged",
            "source": "registry_b",
            "source_record_id": "b-1",
        },
    ]
    first.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    second.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n", encoding="utf-8")

    first_ledger = load_verified_quality_evidence_ledger(first)
    second_ledger = load_verified_quality_evidence_ledger(second)

    assert first_ledger.record_count == 2
    assert first_ledger.file_sha256 != second_ledger.file_sha256
    assert first_ledger.semantic_sha256 == second_ledger.semantic_sha256
    assert first_ledger.evidence[0].paper_identifier == "doi:10.1/first"
    report = assess_paper_quality(
        Paper(title="Ledger paper", identifiers=PaperIdentifiers(doi="10.1/first")),
        verified_evidence=first_ledger.evidence,
    )
    retraction = next(
        signal for signal in report.signals if signal.name == "retraction_status"
    )
    assert retraction.value == 1.0


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        ('{"paper_identifier":"doi:https://doi.org/10.1/x","signal_name":"retraction_status","state":"clear","source":"registry","source_record_id":"1"}\n', "schema_invalid"),
        ('{"paper_identifier":"doi:10.1/x","paper_identifier":"doi:10.1/y","signal_name":"retraction_status","state":"clear","source":"registry","source_record_id":"1"}\n', "invalid_line"),
        ('{"paper_identifier":"doi:10.1/x","signal_name":"retraction_status","state":"clear","source":"registry","source_record_id":"1"}\n{"paper_identifier":"doi:10.1/x","signal_name":"retraction_status","state":"clear","source":"registry","source_record_id":"2"}\n', "duplicate_signal"),
        ("\n", "empty"),
    ],
)
def test_quality_evidence_ledger_rejects_ambiguous_or_invalid_rows(
    tmp_path,
    contents: str,
    reason: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=f"quality_evidence_ledger_{reason}"):
        load_verified_quality_evidence_ledger(path)


def test_quality_evidence_binding_requires_exported_candidate_identity(
    tmp_path,
) -> None:
    identifiers = tmp_path / "candidates.txt"
    identifiers_text = "arxiv:2401.00001\n"
    identifiers.write_text(identifiers_text, encoding="utf-8")
    report = tmp_path / "candidates.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "quality-evidence-candidate-export-v1",
                "source_kind": "completed_run_initial_reranked_candidates_only",
                "gold_or_query_content_loaded": False,
                "identifiers_sha256": hashlib.sha256(
                    identifiers.read_bytes()
                ).hexdigest(),
                "valid_arxiv_identifier_count": 1,
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "paper_identifier": "arxiv:2401.00001",
                "signal_name": "retraction_status",
                "state": "flagged",
                "source": "crossref",
                "source_record_id": "crossref-work:10.1/example",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded, binding = bind_verified_quality_evidence_to_candidates(
        ledger, identifiers, report
    )

    assert loaded.record_count == 1
    assert binding.candidate_identifier_count == 1
    assert binding.matched_evidence_count == 1
    assert binding.ledger_semantic_sha256 == loaded.semantic_sha256

    ledger.write_text(
        ledger.read_text(encoding="utf-8").replace("2401.00001", "2401.00002"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quality_evidence_not_in_exported_candidates"):
        bind_verified_quality_evidence_to_candidates(ledger, identifiers, report)


def test_quality_evidence_binding_rejects_tampered_candidate_report(tmp_path) -> None:
    identifiers = tmp_path / "candidates.txt"
    identifiers.write_text("arxiv:2401.00001\n", encoding="utf-8")
    report = tmp_path / "candidates.json"
    report.write_text("{}", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "paper_identifier": "arxiv:2401.00001",
                "signal_name": "retraction_status",
                "state": "flagged",
                "source": "crossref",
                "source_record_id": "crossref-work:10.1/example",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality_evidence_candidate_report_mismatch"):
        bind_verified_quality_evidence_to_candidates(ledger, identifiers, report)
