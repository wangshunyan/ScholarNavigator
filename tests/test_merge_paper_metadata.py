from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from merge_paper_metadata import MetadataMergeError, merge_rows


def test_merge_fills_only_missing_fields_and_preserves_order() -> None:
    base = [
        {"arxiv_id": "2401.00001", "title": "Base title", "abstract": ""},
        {"arxiv_id": "2401.00002v2", "title": "Second", "abstract": "old"},
    ]
    metadata = [
        {
            "arxiv_id": "2401.00002",
            "abstract": "new abstract",
            "authors": ["A. Author"],
            "year": 2024,
            "doi": "10.1234/example",
        },
        {"arxiv_id": "2401.99999", "title": "not in base"},
    ]
    merged, report = merge_rows(base, metadata)
    assert [row["arxiv_id"] for row in merged] == ["2401.00001", "2401.00002"]
    assert merged[0]["title"] == "Base title"
    assert merged[1]["abstract"] == "old"
    assert merged[1]["authors"] == ["A. Author"]
    assert report["unmatched_metadata_count"] == 1
    assert report["conflict_count"] == 1


def test_merge_can_overwrite_or_reject_conflicts() -> None:
    base = [{"arxiv_id": "2401.00001", "title": "old"}]
    metadata = [{"arxiv_id": "2401.00001", "title": "new"}]
    merged, report = merge_rows(base, metadata, overwrite=True)
    assert merged[0]["title"] == "new"
    assert report["conflict_count"] == 1
    try:
        merge_rows(base, metadata, reject_conflicts=True)
    except MetadataMergeError as exc:
        assert str(exc) == "conflict:2401.00001:title"
    else:  # pragma: no cover
        raise AssertionError("expected conflict rejection")


def test_merge_rejects_duplicate_or_invalid_identity() -> None:
    try:
        merge_rows(
            [{"arxiv_id": "2401.00001"}, {"arxiv_id": "2401.00001v2"}],
            [],
        )
    except MetadataMergeError as exc:
        assert str(exc) == "duplicate_base_id:2401.00001"
    else:  # pragma: no cover
        raise AssertionError("expected duplicate rejection")
