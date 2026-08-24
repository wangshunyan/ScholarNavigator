from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pasa_semantic_corpus import (  # noqa: E402
    build_semantic_corpus,
    normalize_arxiv_id,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_normalize_arxiv_id_removes_transport_syntax_and_version() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2501.00001v2") == "2501.00001"
    assert normalize_arxiv_id("arXiv:HEP-TH/9901001v1") == "hep-th/9901001"
    assert normalize_arxiv_id("not-an-arxiv-id") is None


def test_build_joins_exact_id_and_keeps_pasa_title(tmp_path: Path) -> None:
    pasa = tmp_path / "id2paper.json"
    pasa.write_text(
        json.dumps(
            {
                "2501.00001": "PaSa authoritative title",
                "2501.00002": "No metadata title",
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": "https://arxiv.org/abs/2501.00001v3",
                "title": "Different external title",
                "abstract": "Exact ID abstract",
                "categories": ["cs.IR"],
                "authors": ["A. Author"],
            },
            {
                "id": "2501.00003",
                "title": "Unmatched",
                "abstract": "Does not belong to PaSa.",
            },
            {
                "title": "Same title must not match",
                "abstract": "No identifier means no join.",
            },
        ],
    )
    output = tmp_path / "semantic.jsonl"
    report = build_semantic_corpus(metadata, pasa, output)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "_id": "2501.00001",
            "abstract": "Exact ID abstract",
            "arxiv_id": "2501.00001",
            "authors": ["A. Author"],
            "categories": ["cs.IR"],
            "title": "PaSa authoritative title",
        }
    ]
    assert report.input_rows == 3
    assert report.metadata_missing_id_rows == 1
    assert report.exact_matches == 1
    assert report.unmatched_metadata_ids == 1
    assert report.pasa_without_metadata == 1
    assert report.coverage == 0.5
    assert report.field_completeness["title"] == 1.0
    assert report.field_completeness["abstract"] == 1.0


def test_identical_duplicate_metadata_is_deduplicated(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {"id": "2501.00001", "title": "Paper", "abstract": "Abstract"},
            {"id": "2501.00001v1", "title": "Paper", "abstract": "Abstract"},
        ],
    )

    report = build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")

    assert report.duplicate_metadata_rows == 1
    assert report.output_rows == 1


def test_build_preserves_optional_ranking_metadata(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": "2501.00001",
                "abstract": "Abstract",
                "published": "2024-03-01",
                "journal-ref": "Example Conference",
                "doi": "https://doi.org/10.1234/example",
            }
        ],
    )

    report = build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")
    row = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))

    assert row["year"] == 2024
    assert row["venue"] == "Example Conference"
    assert row["doi"] == "10.1234/example"
    assert report.output_year_rows == 1
    assert report.output_venue_rows == 1
    assert report.output_doi_rows == 1
    assert report.field_completeness["year"] == 1.0
    assert report.field_completeness["venue"] == 1.0
    assert report.field_completeness["doi"] == 1.0


def test_snapshot_uses_first_version_year_before_metadata_update_date(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"0704.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    _write_jsonl(
        metadata,
        [
            {
                "id": "0704.00001",
                "abstract": "Abstract",
                "update_date": "2008-11-26",
                "versions": [
                    {"version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT"},
                    {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
                ],
            }
        ],
    )

    build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")

    row = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    assert row["year"] == 2007


def test_conflicting_duplicate_metadata_fails(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {"id": "2501.00001", "abstract": "First"},
            {"id": "2501.00001v2", "abstract": "Second"},
        ],
    )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")


def test_conflicting_official_records_choose_latest_update_date(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": "2501.00001",
                "abstract": "Older abstract",
                "update_date": "2020-01-01",
            },
            {
                "id": "2501.00001",
                "abstract": "Latest abstract",
                "update_date": "2026-06-03",
            },
        ],
    )

    report = build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")

    row = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    assert row["abstract"] == "Latest abstract"
    assert report.conflicting_metadata_ids == 1


def test_official_json_snapshot_name_is_streamed_as_jsonl(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    _write_jsonl(
        metadata,
        [
            {"id": "2501.00001", "abstract": "Abstract"},
            {"id": "2501.00002", "abstract": "Unmatched"},
        ],
    )

    report = build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")

    assert report.input_rows == 2
    assert report.exact_matches == 1
    assert report.output_rows == 1


def test_evaluator_gold_paths_are_rejected(tmp_path: Path) -> None:
    pasa = tmp_path / "pasa.json"
    pasa.write_text(json.dumps({"2501.00001": "Paper"}), encoding="utf-8")
    metadata = tmp_path / "AutoScholarQuery_test.jsonl"
    _write_jsonl(metadata, [{"id": "2501.00001", "abstract": "Gold"}])

    with pytest.raises(ValueError, match="gold/qrels"):
        build_semantic_corpus(metadata, pasa, tmp_path / "output.jsonl")

    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(metadata.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="gold/qrels"):
        build_semantic_corpus(qrels, pasa, tmp_path / "qrels-output.jsonl")
