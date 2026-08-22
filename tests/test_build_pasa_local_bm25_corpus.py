from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pasa_local_bm25_corpus import convert_corpus  # noqa: E402


def test_convert_id2paper_json_to_arxiv_identity_corpus(tmp_path: Path) -> None:
    source = tmp_path / "id2paper.json"
    source.write_text(
        json.dumps(
            {
                "internal-a": {
                    "title": "Retrieval paper",
                    "abstract": "BM25 and neural search",
                    "arxiv_id": "2501.00001",
                    "doi": "10.123/example",
                },
                "internal-b": {
                    "title": "Missing arXiv paper",
                    "abstract": "This row is not useful for AutoScholar arXiv matching.",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pasa.jsonl"
    report = convert_corpus(source, output, identity="arxiv_id")

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert report.input_records == 2
    assert report.output_records == 1
    assert report.skip_reasons == {"missing_identity:arxiv_id": 1}
    assert rows == [
        {
            "_id": "2501.00001",
            "abstract": "BM25 and neural search",
            "arxiv_id": "2501.00001",
            "doi": "10.123/example",
            "title": "Retrieval paper",
        }
    ]


def test_convert_zip_member_with_external_ids(tmp_path: Path) -> None:
    archive = tmp_path / "cs_paper_2nd.zip"
    with zipfile.ZipFile(archive, mode="w") as handle:
        handle.writestr(
            "paper_database/id2paper.json",
            json.dumps(
                {
                    "a": {
                        "paper_title": "External ID paper",
                        "paper_abstract": "Hybrid retrieval",
                        "externalids": {"ArXiv": "2409.12345"},
                    }
                }
            ),
        )
    output = tmp_path / "local.jsonl"
    report_path = tmp_path / "report.json"
    report = convert_corpus(
        archive,
        output,
        identity="arxiv_id",
        report_path=report_path,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert row["_id"] == "2409.12345"
    assert row["title"] == "External ID paper"
    assert report.output_records == 1
    assert persisted_report["output_sha256"] == report.output_sha256


def test_convert_preserves_optional_ranking_metadata(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "source-1",
                    "title": "Metadata paper",
                    "abstract": "An abstract",
                    "arxiv_id": "2501.00003v2",
                    "authors": [["Ada", "Lovelace"], ["Alan", "Turing"]],
                    "published": "2024-05-01",
                    "journal_ref": "Journal of Reproducible Search",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "local.jsonl"
    convert_corpus(source, output, identity="arxiv_id")
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["arxiv_id"] == "2501.00003v2"
    assert row["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert row["year"] == 2024
    assert row["venue"] == "Journal of Reproducible Search"


def test_convert_official_pasa_title_index_to_arxiv_identity_corpus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "id2paper.json"
    source.write_text(
        json.dumps(
            {
                "2501.00001": "A paper indexed by arXiv ID",
                "2501.00002": "Another indexed paper",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pasa.jsonl"
    report = convert_corpus(source, output, identity="arxiv_id")

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert report.input_records == 2
    assert report.output_records == 2
    assert report.skip_reasons == {}
    assert rows == [
        {
            "_id": "2501.00001",
            "abstract": "",
            "arxiv_id": "2501.00001",
            "title": "A paper indexed by arXiv ID",
        },
        {
            "_id": "2501.00002",
            "abstract": "",
            "arxiv_id": "2501.00002",
            "title": "Another indexed paper",
        },
    ]
