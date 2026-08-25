from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from scholar_agent.core.quality_evidence_sources import (
    ArxivCrossrefRetractionCollection,
    ArxivDoiResolution,
    CrossrefRetractionCollection,
    CrossrefRetractionLookup,
    collect_crossref_retraction_evidence,
    collect_arxiv_crossref_retraction_evidence,
)
from scholar_agent.core.paper_quality import VerifiedQualityEvidence
from scripts import collect_crossref_retraction_evidence as collection_script


class _Headers:
    def __init__(self, content_type: str = "application/json") -> None:
        self.content_type = content_type

    def get_content_type(self) -> str:
        return self.content_type


class _Response:
    def __init__(self, value: object, *, content_type: str = "application/json") -> None:
        self.headers = _Headers(content_type)
        self._body = BytesIO(json.dumps(value).encode("utf-8"))

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


_ARXIV_DOI_FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2401.00001v2</id>
    <arxiv:doi>10.1000/explicit-retraction</arxiv:doi>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2401.00002</id>
  </entry>
</feed>'''


class _BytesResponse:
    def __init__(self, body: bytes, *, content_type: str) -> None:
        self.headers = _Headers(content_type)
        self._body = BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_crossref_collector_emits_only_explicit_retraction_relation() -> None:
    requests = []

    def opener(request, timeout: float):  # noqa: ANN001
        requests.append((request, timeout))
        if request.full_url.endswith("10.1%2Fretracted"):
            return _Response(
                {"message": {"relation": {"is-retracted-by": [{"id": "10.2/n"}]}}}
            )
        return _Response({"message": {"relation": {}}})

    collection = collect_crossref_retraction_evidence(
        ["doi:10.1/unflagged", "doi:10.1/retracted", "doi:10.1/retracted"],
        opener=opener,
    )

    assert [item.paper_identifier for item in collection.evidence] == [
        "doi:10.1/retracted"
    ]
    evidence = collection.evidence[0]
    assert evidence.signal_name == "retraction_status"
    assert evidence.state == "flagged"
    assert evidence.source == "crossref"
    assert collection.outcome_counts() == {
        "flagged": 1,
        "no_explicit_retraction_relation": 1,
    }
    assert len(requests) == 2
    assert all(timeout == 10.0 for _, timeout in requests)
    assert all(request.get_header("Accept") == "application/json" for request, _ in requests)


def test_crossref_collector_accepts_explicit_retraction_update() -> None:
    collection = collect_crossref_retraction_evidence(
        ["doi:10.1/retracted"],
        opener=lambda *_args, **_kwargs: _Response(
            {"message": {"update-to": [{"type": "retraction"}]}}
        ),
    )

    assert collection.lookups[0].outcome == "flagged"
    assert collection.evidence[0].source_record_id == "crossref-work:10.1/retracted"


def test_crossref_collector_accepts_direct_retraction_relation() -> None:
    collection = collect_crossref_retraction_evidence(
        ["doi:10.1007/s11613-016-0476-y"],
        opener=lambda *_args, **_kwargs: _Response(
            {
                "message": {
                    "relation": {
                        "retraction": [{"id-type": "doi", "id": "10.1007/retraction"}]
                    }
                }
            }
        ),
    )

    assert collection.lookups[0].outcome == "flagged"
    assert collection.evidence[0].paper_identifier == "doi:10.1007/s11613-016-0476-y"


def test_crossref_collector_keeps_request_failures_unknown() -> None:
    def opener(request, timeout: float):  # noqa: ANN001
        if request.full_url.endswith("10.1%2Fmissing"):
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        raise URLError("offline")

    collection = collect_crossref_retraction_evidence(
        ["doi:10.1/missing", "doi:10.1/offline"], opener=opener
    )

    assert collection.evidence == ()
    assert collection.outcome_counts() == {"network_error": 1, "not_found": 1}


@pytest.mark.parametrize(
    ("identifiers", "reason"),
    [
        ([], "crossref_paper_identifier_required"),
        (["10.1/not-prefixed"], "canonical_doi_paper_identifier_required"),
        (["doi:https://doi.org/10.1/not-canonical"], "canonical_doi_paper_identifier_required"),
        (["arxiv:2401.00001"], "canonical_doi_paper_identifier_required"),
    ],
)
def test_crossref_collector_requires_canonical_doi_identifiers(
    identifiers: list[str], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        collect_crossref_retraction_evidence(identifiers)


def test_crossref_collector_rejects_non_json_response() -> None:
    collection = collect_crossref_retraction_evidence(
        ["doi:10.1/plain"],
        opener=lambda *_args, **_kwargs: _Response({}, content_type="text/html"),
    )

    assert collection.evidence == ()
    assert collection.lookups[0].outcome == "invalid_response"


def test_arxiv_crossref_collector_binds_flag_to_original_arxiv_identifier() -> None:
    arxiv_requests = []
    crossref_requests = []

    def arxiv_opener(request, timeout: float):  # noqa: ANN001
        arxiv_requests.append((request, timeout))
        return _BytesResponse(_ARXIV_DOI_FEED, content_type="application/atom+xml")

    def crossref_opener(request, timeout: float):  # noqa: ANN001
        crossref_requests.append((request, timeout))
        return _Response(
            {"message": {"update-to": [{"type": "retraction"}]}}
        )

    collection = collect_arxiv_crossref_retraction_evidence(
        ["arxiv:2401.00002", "arxiv:2401.00001"],
        arxiv_opener=arxiv_opener,
        crossref_opener=crossref_opener,
    )

    assert collection.evidence[0].paper_identifier == "arxiv:2401.00001"
    assert collection.evidence[0].source_record_id == (
        "crossref-work:10.1000/explicit-retraction"
    )
    assert collection.outcome_counts() == {
        "arxiv:no_doi": 1,
        "arxiv:resolved": 1,
        "crossref:flagged": 1,
    }
    assert len(arxiv_requests) == 1
    assert len(crossref_requests) == 1
    assert arxiv_requests[0][0].get_header("Accept") == "application/atom+xml"
    assert crossref_requests[0][0].get_header("Accept") == "application/json"


def test_arxiv_crossref_collector_keeps_unavailable_or_missing_metadata_unknown() -> None:
    collection = collect_arxiv_crossref_retraction_evidence(
        ["arxiv:2401.00001", "arxiv:2401.00002"],
        arxiv_opener=lambda *_args, **_kwargs: _BytesResponse(
            _ARXIV_DOI_FEED, content_type="application/atom+xml"
        ),
        crossref_opener=lambda request, timeout: (_ for _ in ()).throw(
            URLError("offline")
        ),
    )

    assert collection.evidence == ()
    assert collection.outcome_counts() == {
        "arxiv:no_doi": 1,
        "arxiv:resolved": 1,
        "crossref:network_error": 1,
    }


@pytest.mark.parametrize(
    ("identifiers", "reason"),
    [
        ([], "arxiv_paper_identifier_required"),
        (["2401.00001"], "canonical_arxiv_paper_identifier_required"),
        (["arxiv:https://arxiv.org/abs/2401.00001"], "canonical_arxiv_paper_identifier_required"),
        ([f"arxiv:2401.{index:05d}" for index in range(21)], "arxiv_identifier_batch_too_large"),
    ],
)
def test_arxiv_crossref_collector_has_exact_bounded_inputs(
    identifiers: list[str], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        collect_arxiv_crossref_retraction_evidence(identifiers)


def test_collection_writer_does_not_create_an_empty_ledger(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    collection = CrossrefRetractionCollection(
        evidence=(),
        lookups=(
            CrossrefRetractionLookup(
                paper_identifier="doi:10.1/unknown",
                outcome="no_explicit_retraction_relation",
            ),
        ),
    )

    wrote = collection_script.write_collection_outputs(
        collection,
        input_file_sha256="a" * 64,
        identifier_count=1,
        ledger_output=ledger_path,
        report_output=report_path,
    )

    assert wrote is False
    assert not ledger_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "no_explicit_evidence"
    assert report["outcome_counts"] == {"no_explicit_retraction_relation": 1}


def test_collection_writer_writes_only_ledger_records_and_compact_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    collection = CrossrefRetractionCollection(
        evidence=(
            VerifiedQualityEvidence(
                paper_identifier="doi:10.1/retracted",
                signal_name="retraction_status",
                state="flagged",
                source="crossref",
                source_record_id="crossref-work:10.1/retracted",
            ),
        ),
        lookups=(
            CrossrefRetractionLookup(
                paper_identifier="doi:10.1/retracted", outcome="flagged"
            ),
        ),
    )

    wrote = collection_script.write_collection_outputs(
        collection,
        input_file_sha256="b" * 64,
        identifier_count=1,
        ledger_output=ledger_path,
        report_output=report_path,
    )

    assert wrote is True
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["state"] == "flagged"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "flagged_evidence_count": 1,
        "input_file_sha256": "b" * 64,
        "input_identifier_count": 1,
        "outcome_counts": {"flagged": 1},
        "schema_version": "crossref-retraction-evidence-report-v1",
        "source": "crossref",
        "status": "evidence_written",
    }


def test_collection_cli_uses_explicit_identifier_kind(monkeypatch, tmp_path: Path) -> None:
    identifiers = tmp_path / "arxiv.txt"
    identifiers.write_text("arxiv:2401.00001\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    observed = {}

    def fake_arxiv(identifier_rows, *, timeout_seconds):  # noqa: ANN001
        observed["identifiers"] = identifier_rows
        observed["timeout"] = timeout_seconds
        return CrossrefRetractionCollection(evidence=(), lookups=())

    monkeypatch.setattr(
        collection_script,
        "collect_arxiv_crossref_retraction_evidence",
        fake_arxiv,
    )

    result = collection_script.main(
        [
            "--paper-identifiers",
            str(identifiers),
            "--identifier-kind",
            "arxiv",
            "--ledger-output",
            str(ledger_path),
            "--report-output",
            str(report_path),
        ]
    )

    assert result == 2
    assert observed == {"identifiers": ["arxiv:2401.00001"], "timeout": 10.0}
    assert not ledger_path.exists()


def test_collection_writer_identifies_arxiv_then_crossref_provenance(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    collection = ArxivCrossrefRetractionCollection(
        evidence=(),
        arxiv_resolutions=(
            ArxivDoiResolution("arxiv:2401.00001", "no_doi"),
        ),
        crossref_lookups=(),
    )

    collection_script.write_collection_outputs(
        collection,
        input_file_sha256="c" * 64,
        identifier_count=1,
        ledger_output=tmp_path / "ledger.jsonl",
        report_output=report_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["source"] == (
        "arxiv_then_crossref"
    )
