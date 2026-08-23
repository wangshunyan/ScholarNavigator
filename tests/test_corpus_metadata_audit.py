from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.evaluation.corpus_metadata_audit import audit_jsonl_corpus


def test_audit_reports_field_completeness_and_identity_duplicates(tmp_path: Path) -> None:
    corpus = tmp_path / "papers.jsonl"
    rows = [
        {"arxiv_id": "https://arxiv.org/abs/2501.00001v2", "title": "A", "abstract": "x"},
        {"arxiv_id": "2501.00001", "title": "A2", "abstract": "y", "year": 2025},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = audit_jsonl_corpus(corpus)

    assert report["row_count"] == 2
    assert report["unique_identity_count"] == 1
    assert report["duplicate_identity_rows"] == 1
    assert report["field_completeness"]["abstract"] == 1.0
    assert report["field_completeness"]["year"] == 0.5
    assert report["passed"] is False


def test_audit_rejects_invalid_rows(tmp_path: Path) -> None:
    corpus = tmp_path / "papers.jsonl"
    corpus.write_text('{"arxiv_id":"2501.00001","title":"A"}\nnot-json\n', encoding="utf-8")

    report = audit_jsonl_corpus(corpus)

    assert report["invalid_json_rows"] == 1
    assert report["invalid_identity_rows"] == 0
    assert report["passed"] is False


def test_audit_can_fail_closed_on_required_sorting_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "papers.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "arxiv_id": "2501.00001",
                "title": "A",
                "abstract": "x",
                "authors": ["Author"],
                "year": 2025,
                "venue": "ACL",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_jsonl_corpus(
        corpus,
        required_fields=("title", "abstract", "authors", "year", "venue", "doi"),
    )

    assert report["structural_passed"] is True
    assert report["required_fields_complete"] is False
    assert report["passed"] is False
