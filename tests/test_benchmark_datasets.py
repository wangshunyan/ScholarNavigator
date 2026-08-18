from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.datasets import inspect_dataset, load_dataset
from scholar_agent.evaluation.datasets.auto_scholar_query import (
    load_auto_scholar_query,
)


ROOT = Path(__file__).resolve().parents[1]


def test_auto_scholar_query_adapter_loads_real_dataset() -> None:
    queries = load_dataset("auto_scholar_query")

    assert len(queries) == 1000
    assert len({item.query_id for item in queries}) == 1000
    assert sum(len(item.gold_papers) for item in queries) == 2403
    assert queries[0].query_id == "AutoScholarQuery_test_0"
    assert queries[0].query == (
        "Can you tell me some papers about hybrid architectures in "
        "reconstruction-based techniques?"
    )


def test_adapter_preserves_gold_identifier_title_and_binary_grade() -> None:
    first = load_dataset("auto_scholar_query")[0]

    assert first.gold_papers[0].title == (
        "Multivariate Time-series Anomaly Detection via Graph Attention Network"
    )
    assert first.gold_papers[0].arxiv_id == "2009.02040"
    assert first.gold_papers[0].relevance_grade == 1.0
    assert first.gold_papers[0].metadata["label_type"] == "binary_gold"
    assert first.metadata["split"] == "test"


def test_adapter_is_deterministic() -> None:
    first = load_dataset("auto_scholar_query")[:3]
    second = load_dataset("auto_scholar_query")[:3]

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


def test_adapter_rejects_missing_query(tmp_path: Path) -> None:
    path = _write_rows(
        tmp_path / "missing-query.jsonl",
        [{"qid": "case-1", "answer": ["Paper"], "answer_arxiv_id": ["1"]}],
    )

    with pytest.raises(ValueError, match="missing question"):
        load_auto_scholar_query(path)


@pytest.mark.parametrize(
    "row",
    [
        {"qid": "case-1", "question": "Query", "answer": []},
        {
            "qid": "case-1",
            "question": "Query",
            "answer": ["Paper"],
            "answer_arxiv_id": [],
        },
    ],
)
def test_adapter_rejects_missing_gold_by_dataset_rule(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    path = _write_rows(tmp_path / "missing-gold.jsonl", [row])

    with pytest.raises(ValueError, match="gold must not be empty|must be a list"):
        load_auto_scholar_query(path)


def test_adapter_rejects_duplicate_case_id(tmp_path: Path) -> None:
    row = {
        "qid": "duplicate",
        "question": "Query",
        "answer": ["Paper"],
        "answer_arxiv_id": ["2401.00001"],
    }
    path = _write_rows(tmp_path / "duplicate.jsonl", [row, row])

    with pytest.raises(ValueError, match="duplicate qid duplicate"):
        load_auto_scholar_query(path)


def test_dataset_inspection_reports_valid_invalid_and_duplicate_cases(
    tmp_path: Path,
) -> None:
    valid = {
        "qid": "case-1",
        "question": "Original query",
        "answer": ["Paper"],
        "answer_arxiv_id": ["2401.00001"],
    }
    invalid = {
        "qid": "case-2",
        "question": "",
        "answer": [],
        "answer_arxiv_id": [],
    }
    path = _write_rows(tmp_path / "inspect.jsonl", [valid, valid, invalid])

    report = inspect_dataset("auto_scholar_query", path=path)

    assert report.case_count == 3
    assert report.query_count == 2
    assert report.gold_paper_count == 2
    assert report.cases_without_gold == 1
    assert report.gold_with_arxiv_id == 2
    assert report.invalid_case_count == 1
    assert report.duplicate_case_id_count == 1


def test_real_dataset_inspection_counts_are_stable() -> None:
    report = inspect_dataset("auto_scholar_query")

    assert report.source_path == str(
        ROOT / "benchmark" / "AutoScholarQuery_test.jsonl"
    )
    assert report.case_count == 1000
    assert report.query_count == 1000
    assert report.gold_paper_count == 2403
    assert report.cases_without_gold == 0
    assert report.gold_with_arxiv_id == 2403
    assert report.invalid_case_count == 0
    assert report.duplicate_case_id_count == 0


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
