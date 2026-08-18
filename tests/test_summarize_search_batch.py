from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import summarize_search_batch  # noqa: E402


def test_normal_jsonl_generates_markdown_file(tmp_path: Path) -> None:
    input_path = _write_jsonl(tmp_path / "batch.jsonl", _sample_rows())
    output_path = tmp_path / "reports" / "summary.md"

    code = summarize_search_batch.main(
        ["--input", str(input_path), "--output", str(output_path), "--top-n", "5"]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert code == 0
    assert "# ScholarNavigator Batch Search Summary" in markdown
    assert "- Total cases: 3" in markdown
    assert "- Succeeded: 2" in markdown
    assert "- Failed: 1" in markdown
    assert "## Cost / Efficiency" in markdown
    assert "- Succeeded cases: 2" in markdown
    assert "- Failed cases: 1" in markdown
    assert "- Average latency seconds: 2.000" in markdown
    assert "- Total LLM calls: 1" in markdown
    assert "- Total LLM prompt tokens: 100" in markdown
    assert "- Total LLM completion tokens: 20" in markdown
    assert "- Total LLM tokens: 120" in markdown
    assert (
        "| case_001 | succeeded | 1.000 | 1 | 1 | succeeded | query one; query one refined | arxiv,semantic_scholar | 6 | 5 | - |"
        in markdown
    )
    assert (
        "| case_003 | failed | 3.000 | 0 | 0 | - | - | - | - | - | forced failure |"
        in markdown
    )
    assert "## Source Reliability" in markdown
    assert "| arxiv | 2 | 2 | 0 | 0 | 5 | 0.150 | - |" in markdown
    assert "| semantic_scholar | 2 | 0 | 2 | 1 | 0 | 0.550 | HTTP 429 (1); source_cooldown_skip:semantic_scholar (1) |" in markdown


def test_without_output_prints_to_stdout(tmp_path: Path, capsys) -> None:
    input_path = _write_jsonl(tmp_path / "batch.jsonl", _sample_rows())

    code = summarize_search_batch.main(["--input", str(input_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "# ScholarNavigator Batch Search Summary" in captured.out
    assert "## Case Summary" in captured.out


def test_summary_counts_success_rate_latency_and_costs() -> None:
    summary = summarize_search_batch.summarize_rows(_sample_rows(), top_n=5)

    assert summary["total_cases"] == 3
    assert summary["succeeded_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["success_rate"] == 2 / 3
    assert summary["latency"] == {"average": 2.0, "min": 1.0, "max": 3.0}
    assert summary["cost_totals"]["api_call_count"] == 5
    assert summary["cost_totals"]["search_api_call_count"] == 5
    assert summary["cost_totals"]["cache_hit_count"] == 1
    assert summary["cost_totals"]["llm_call_count"] == 1
    assert summary["cost_totals"]["llm_prompt_tokens"] == 100
    assert summary["cost_totals"]["llm_completion_tokens"] == 20
    assert summary["cost_totals"]["llm_total_tokens"] == 120
    assert summary["cost_totals"]["estimated_total_tokens"] == 30
    assert summary["case_summaries"][0]["expanded_queries"] == (
        "query one; query one refined"
    )
    assert summary["case_summaries"][0]["source_preferences"] == (
        "arxiv,semantic_scholar"
    )
    assert summary["case_summaries"][0]["raw_count"] == "6"
    assert summary["case_summaries"][0]["deduplicated_count"] == "5"


def test_top_papers_missing_evidence_and_source_errors_are_counted() -> None:
    summary = summarize_search_batch.summarize_rows(_sample_rows(), top_n=5)

    assert summary["top_papers"][0] == ("Paper A", 2)
    assert ("Paper B", 1) in summary["top_papers"]
    assert ("warning_common", 2) in summary["warning_counts"]
    assert ("source_error:openalex:HTTP 503", 1) in summary["source_error_counts"]


def test_source_reliability_is_aggregated_from_retrieval_diagnostics() -> None:
    summary = summarize_search_batch.summarize_rows(_sample_rows(), top_n=5)
    reliability = {
        item["source"]: item for item in summary["source_reliability"]
    }

    assert reliability["arxiv"]["call_count"] == 2
    assert reliability["arxiv"]["success_count"] == 2
    assert reliability["arxiv"]["error_count"] == 0
    assert reliability["arxiv"]["cooldown_skip_count"] == 0
    assert reliability["arxiv"]["total_returned_count"] == 5
    assert reliability["arxiv"]["avg_latency_seconds"] == pytest.approx(0.15)
    assert reliability["arxiv"]["top_error_messages"] == []

    assert reliability["semantic_scholar"]["call_count"] == 2
    assert reliability["semantic_scholar"]["success_count"] == 0
    assert reliability["semantic_scholar"]["error_count"] == 2
    assert reliability["semantic_scholar"]["cooldown_skip_count"] == 1
    assert reliability["semantic_scholar"]["total_returned_count"] == 0
    assert reliability["semantic_scholar"]["avg_latency_seconds"] == pytest.approx(0.55)
    assert reliability["semantic_scholar"]["top_error_messages"] == [
        ("HTTP 429", 1),
        ("source_cooldown_skip:semantic_scholar", 1),
    ]


def test_source_reliability_handles_legacy_results_without_diagnostics() -> None:
    summary = summarize_search_batch.summarize_rows(
        [
            {
                "case_id": "legacy",
                "query": "query",
                "status": "succeeded",
                "result": {
                    "highly_relevant_papers": [],
                    "partially_relevant_papers": [],
                    "missing_evidence": [],
                    "cost_report": {},
                    "synthesis": None,
                },
                "error": None,
                "latency_seconds": 0.1,
            }
        ]
    )

    assert summary["source_reliability"] == []
    assert summary["cost_totals"]["llm_call_count"] == 0
    assert summary["cost_totals"]["llm_prompt_tokens"] == 0
    assert summary["cost_totals"]["llm_completion_tokens"] == 0
    assert summary["cost_totals"]["llm_total_tokens"] == 0


def test_missing_input_file_returns_nonzero(tmp_path: Path) -> None:
    code = summarize_search_batch.main(
        ["--input", str(tmp_path / "missing.jsonl")]
    )

    assert code == 1


def test_invalid_jsonl_returns_nonzero(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"case_id": "ok"}\n{bad-json}\n', encoding="utf-8")

    code = summarize_search_batch.main(["--input", str(input_path)])

    assert code == 1


def test_non_object_jsonl_returns_nonzero(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('["not", "object"]\n', encoding="utf-8")

    code = summarize_search_batch.main(["--input", str(input_path)])

    assert code == 1


def test_succeeded_with_null_result_does_not_crash(tmp_path: Path) -> None:
    input_path = _write_jsonl(
        tmp_path / "batch.jsonl",
        [
            {
                "case_id": "case_null",
                "query": "query",
                "status": "succeeded",
                "result": None,
                "error": None,
                "latency_seconds": 0.5,
            }
        ],
    )
    output_path = tmp_path / "summary.md"

    code = summarize_search_batch.main(
        ["--input", str(input_path), "--output", str(output_path)]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert code == 0
    assert "| case_null | succeeded | 0.500 | 0 | 0 | - | - | - | - | - | - |" in markdown
    assert "succeeded_result_missing" in markdown


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "case_001",
            "query": "query one",
            "status": "succeeded",
            "result": _result(
                high_titles=["Paper A"],
                partial_titles=["Paper B"],
                missing_evidence=["warning_common"],
                cost_report={
                    "api_call_count": 2,
                    "search_api_call_count": 2,
                    "cache_hit_count": 1,
                    "llm_call_count": 1,
                    "llm_prompt_tokens": 100,
                    "llm_completion_tokens": 20,
                    "llm_total_tokens": 120,
                    "estimated_input_tokens": 10,
                    "estimated_output_tokens": 5,
                    "estimated_total_tokens": 15,
                },
                synthesis_status="succeeded",
                expanded_queries=["query one", "query one refined"],
                source_preferences=["arxiv", "semantic_scholar"],
                raw_count=6,
                deduplicated_count=5,
                source_stats=[
                    {
                        "source": "arxiv",
                        "returned_count": 5,
                        "latency_seconds": 0.1,
                        "cache_hit": False,
                        "error_message": None,
                    },
                    {
                        "source": "semantic_scholar",
                        "returned_count": 0,
                        "latency_seconds": 0.6,
                        "cache_hit": False,
                        "error_message": "HTTP 429",
                    },
                ],
            ),
            "error": None,
            "latency_seconds": 1.0,
        },
        {
            "case_id": "case_002",
            "query": "query two",
            "status": "succeeded",
            "result": _result(
                high_titles=["Paper A"],
                partial_titles=[],
                missing_evidence=[
                    "warning_common",
                    "source_error:openalex:HTTP 503",
                ],
                cost_report={
                    "api_call_count": 3,
                    "search_api_call_count": 3,
                    "cache_hit_count": 0,
                    "estimated_input_tokens": 9,
                    "estimated_output_tokens": 6,
                    "estimated_total_tokens": 15,
                },
                synthesis_status="insufficient_evidence",
                expanded_queries=["query two"],
                source_preferences=["openalex", "arxiv"],
                raw_count=3,
                deduplicated_count=2,
                source_stats=[
                    {
                        "source": "arxiv",
                        "returned_count": 0,
                        "latency_seconds": 0.2,
                        "cache_hit": True,
                        "error_message": None,
                    },
                    {
                        "source": "semantic_scholar",
                        "returned_count": 0,
                        "latency_seconds": 0.5,
                        "cache_hit": False,
                        "error_message": "source_cooldown_skip:semantic_scholar",
                    },
                ],
            ),
            "error": None,
            "latency_seconds": 2.0,
        },
        {
            "case_id": "case_003",
            "query": "query three",
            "status": "failed",
            "result": None,
            "error": "forced failure",
            "latency_seconds": 3.0,
        },
    ]


def _result(
    *,
    high_titles: list[str],
    partial_titles: list[str],
    missing_evidence: list[str],
    cost_report: dict[str, int],
    synthesis_status: str,
    expanded_queries: list[str] | None = None,
    source_preferences: list[str] | None = None,
    raw_count: int | None = None,
    deduplicated_count: int | None = None,
    source_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "highly_relevant_papers": [_paper(title) for title in high_titles],
        "partially_relevant_papers": [_paper(title) for title in partial_titles],
        "missing_evidence": missing_evidence,
        "cost_report": cost_report,
        "synthesis": {"status": synthesis_status},
        "search_plan": {
            "expanded_queries": expanded_queries or [],
            "source_preferences": source_preferences or [],
        },
        "retrieval_diagnostics": {
            "raw_count": raw_count,
            "deduplicated_count": deduplicated_count,
            "source_stats": source_stats or [],
        },
    }


def _paper(title: str) -> dict[str, Any]:
    return {"paper": {"title": title}}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
