from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_search_batch  # noqa: E402


def test_complete_match_metrics_are_correct(tmp_path: Path) -> None:
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl",
        [
            _batch_row(
                "case_001",
                high=[
                    _ranked("Paper A", year=2025, doi="10.1000/a"),
                    _ranked("Paper B", year=2024, doi="10.1000/b"),
                ],
            )
        ],
    )
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {
                "case_id": "case_001",
                "relevant_papers": [
                    {"title": "Paper A", "year": 2025, "doi": "10.1000/a"},
                    {"title": "Paper B", "year": 2024, "doi": "10.1000/b"},
                ],
            }
        ],
    )
    output_path = tmp_path / "reports" / "eval.json"

    code = evaluate_search_batch.main(
        [
            "--batch-results",
            str(batch_path),
            "--gold",
            str(gold_path),
            "--output",
            str(output_path),
            "--k",
            "2",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["case_count"] == 1
    assert payload["evaluated_case_count"] == 1
    assert payload["aggregate"]["recall_at_k"]["2"] == pytest.approx(1.0)
    assert payload["aggregate"]["precision_at_k"]["2"] == pytest.approx(1.0)
    assert payload["aggregate"]["f1_at_k"]["2"] == pytest.approx(1.0)
    assert payload["aggregate"]["mrr"] == pytest.approx(1.0)
    assert payload["aggregate"]["ndcg_at_k"]["2"] == pytest.approx(1.0)


def test_doi_arxiv_match_but_unproven_title_only_record_stays_unmatched() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[
                _ranked("DOI Paper", year=2025, doi="https://doi.org/10.1000/ABC"),
                _ranked("Arxiv Paper", year=2024, arxiv_id="2501.00001v2"),
                _ranked("Fallback Paper!", year=2023),
            ],
        )
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {"title": "DOI Paper", "year": 2025, "doi": "10.1000/abc"},
                {"title": "Arxiv Paper", "year": 2024, "arxiv_id": "2501.00001"},
                {"title": "Fallback Paper", "year": 2023},
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[3],
    )

    assert result["per_case"][0]["matched_ids"] == [
        "doi:10.1000/abc",
        "arxiv:2501.00001",
    ]
    assert result["aggregate"]["recall_at_k"]["3"] == pytest.approx(2 / 3)


def test_arxiv_doi_matches_result_arxiv_id() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[
                _ranked("RAGAS", year=2023, arxiv_id="2309.15217"),
                _ranked("Ordinary DOI Paper", year=2025, doi="10.1000/ABC"),
            ],
        )
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {
                    "title": "RAGAS",
                    "year": 2023,
                    "doi": "10.48550/arXiv.2309.15217",
                },
                {
                    "title": "Ordinary DOI Paper",
                    "year": 2025,
                    "doi": "https://doi.org/10.1000/abc",
                },
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[2],
    )

    assert result["per_case"][0]["matched_ids"] == [
        "arxiv:2309.15217",
        "doi:10.1000/abc",
    ]
    assert result["aggregate"]["recall_at_k"]["2"] == pytest.approx(1.0)


def test_gold_arxiv_id_version_matches_result_without_version() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[_ranked("Versioned arXiv Paper", year=2024, arxiv_id="2501.00001")],
        )
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {
                    "title": "Versioned arXiv Paper",
                    "year": 2024,
                    "arxiv_id": "2501.00001v2",
                }
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )

    assert result["per_case"][0]["matched_ids"] == ["arxiv:2501.00001"]
    assert result["aggregate"]["recall_at_k"]["1"] == pytest.approx(1.0)


def test_semantic_scholar_id_conflict_with_doi_does_not_match() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[
                _ranked(
                    "Entity-Duet Neural Ranking",
                    year=2018,
                    doi="10.18653/v1/P18-1223",
                    semantic_scholar_id="4d91b5b2f4306f92f556a866a770ecd0fc22731e",
                )
            ],
        )
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {
                    "title": "Entity-Duet Neural Ranking",
                    "year": 2018,
                    "doi": "10.0000/different",
                    "semantic_scholar_id": "4d91b5b2f4306f92f556a866a770ecd0fc22731e",
                }
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )

    assert result["per_case"][0]["matched_ids"] == []
    assert result["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)


def test_arxiv_id_conflict_with_doi_does_not_match() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[
                _ranked(
                    "arXiv Paper",
                    year=2024,
                    doi="10.0000/predicted",
                    arxiv_id="2501.00001",
                )
            ],
        )
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {
                    "title": "arXiv Paper",
                    "year": 2024,
                    "doi": "10.0000/gold",
                    "arxiv_id": "2501.00001v2",
                }
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )

    assert result["per_case"][0]["matched_ids"] == []
    assert result["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)


def test_title_fallback_requires_author_evidence() -> None:
    title = "Fallback Only Paper"
    no_id_match = evaluate_search_batch.evaluate_batch_results(
        [_batch_row("case_001", high=[_ranked(title, year=2024)])],
        evaluate_search_batch.load_gold_rows(
            _write_jsonl_for_rows(
                [
                    {
                        "case_id": "case_001",
                        "relevant_papers": [{"title": title, "year": 2024}],
                    }
                ]
            )
        ),
        k_values=[1],
    )
    one_side_has_id = evaluate_search_batch.evaluate_batch_results(
        [_batch_row("case_001", high=[_ranked(title, year=2024, doi="10.1000/p")])],
        evaluate_search_batch.load_gold_rows(
            _write_jsonl_for_rows(
                [
                    {
                        "case_id": "case_001",
                        "relevant_papers": [{"title": title, "year": 2024}],
                    }
                ]
            )
        ),
        k_values=[1],
    )

    assert no_id_match["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)
    assert no_id_match["per_case"][0]["matched_ids"] == []
    assert one_side_has_id["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)
    assert one_side_has_id["per_case"][0]["matched_ids"] == []


def test_different_reliable_ids_do_not_match_by_title() -> None:
    title = "Same Title Different Identifiers"
    batch_rows = [
        _batch_row("case_001", high=[_ranked(title, year=2024, doi="10.1000/a")])
    ]
    gold_rows = [
        {
            "case_id": "case_001",
            "relevant_papers": [
                {
                    "title": title,
                    "year": 2024,
                    "doi": "10.1000/b",
                    "semantic_scholar_id": "different",
                }
            ],
        }
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )

    assert result["per_case"][0]["matched_ids"] == []
    assert result["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)


def test_failed_missing_gold_and_missing_result_cases_are_tracked() -> None:
    batch_rows = [
        _batch_row("case_ok", high=[_ranked("Paper A", doi="10.1/a")]),
        {
            "case_id": "case_failed",
            "query": "failed query",
            "status": "failed",
            "result": None,
            "error": "connector failed",
            "latency_seconds": 0.2,
        },
        _batch_row("case_missing_gold", high=[_ranked("Paper X", doi="10.1/x")]),
    ]
    gold_rows = [
        {"case_id": "case_ok", "relevant_papers": [{"doi": "10.1/a"}]},
        {"case_id": "case_failed", "relevant_papers": [{"doi": "10.1/f"}]},
        {"case_id": "case_missing_result", "relevant_papers": [{"doi": "10.1/m"}]},
    ]

    result = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )

    assert result["case_count"] == 3
    assert result["evaluated_case_count"] == 1
    assert result["failed_cases"] == [
        {
            "case_id": "case_failed",
            "query": "failed query",
            "status": "failed",
            "error": "connector failed",
        }
    ]
    assert result["missing_gold_cases"] == ["case_missing_gold"]
    assert result["missing_result_cases"] == ["case_missing_result"]
    assert result["aggregate"]["recall_at_k"]["1"] == pytest.approx(1.0)
    assert result["success_only_metrics"]["recall_at_k"]["1"] == pytest.approx(1.0)
    assert result["end_to_end_metrics"]["recall_at_k"]["1"] == pytest.approx(1 / 3)
    assert result["case_statistics"] == {
        "total_case_count": 4,
        "gold_case_count": 3,
        "evaluated_success_count": 1,
        "failed_case_count": 1,
        "missing_result_count": 1,
        "missing_gold_count": 1,
        "success_rate": 0.25,
        "failed_case_rate": 0.25,
        "missing_result_rate": 0.25,
    }


def test_include_partial_controls_ranked_list() -> None:
    batch_rows = [
        _batch_row(
            "case_001",
            high=[],
            partial=[_ranked("Partial Paper", doi="10.1/partial")],
        )
    ]
    gold_rows = [{"case_id": "case_001", "relevant_papers": [{"doi": "10.1/partial"}]}]

    default_policy = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
    )
    highly_only = evaluate_search_batch.evaluate_batch_results(
        batch_rows,
        evaluate_search_batch.load_gold_rows(_write_jsonl_for_rows(gold_rows)),
        k_values=[1],
        result_policy="highly_only",
    )

    assert default_policy["aggregate"]["recall_at_k"]["1"] == pytest.approx(1.0)
    assert default_policy["per_case"][0]["ranked_count"] == 1
    assert highly_only["aggregate"]["recall_at_k"]["1"] == pytest.approx(0.0)
    assert highly_only["per_case"][0]["ranked_count"] == 0


def test_succeeded_but_invalid_result_counts_as_missing_result() -> None:
    result = evaluate_search_batch.evaluate_batch_results(
        [
            {
                "case_id": "case_001",
                "query": "query",
                "status": "succeeded",
                "result": {},
            }
        ],
        [{"case_id": "case_001", "relevant_papers": [{"doi": "10.1/gold"}]}],
        k_values=[1],
    )

    assert result["missing_result_cases"] == ["case_001"]
    assert result["case_statistics"]["evaluated_success_count"] == 0
    assert result["end_to_end_metrics"]["f1_at_k"]["1"] == 0.0


def test_empty_gold_is_excluded_from_metric_denominators() -> None:
    result = evaluate_search_batch.evaluate_batch_results(
        [_batch_row("case_without_gold", high=[])],
        [{"case_id": "case_without_gold", "relevant_papers": []}],
        k_values=[1],
    )

    assert result["missing_gold_cases"] == ["case_without_gold"]
    assert result["case_statistics"]["gold_case_count"] == 0
    assert result["case_statistics"]["evaluated_success_count"] == 0
    assert result["success_only_metrics"]["f1_at_k"] == {"1": 0.0}


def test_failed_case_without_gold_is_counted_but_not_scored() -> None:
    result = evaluate_search_batch.evaluate_batch_results(
        [
            {
                "case_id": "case_001",
                "query": "query",
                "status": "timeout",
                "result": None,
                "error": "timeout",
            }
        ],
        [],
        k_values=[5],
    )

    assert result["missing_gold_cases"] == ["case_001"]
    assert result["case_statistics"]["failed_case_count"] == 1
    assert result["case_statistics"]["gold_case_count"] == 0
    assert result["end_to_end_metrics"]["f1_at_k"] == {"5": 0.0}


def test_batch_efficiency_uses_observed_cost_fields() -> None:
    row = _batch_row("case_001", high=[_ranked("Paper", doi="10.1/p")])
    row["latency_seconds"] = 1.25
    row["result"]["cost_report"] = {
        "llm_call_count": 3,
        "llm_total_tokens": 420,
        "api_call_count": 9,
        "search_api_call_count": 4,
        "reference_api_call_count": 2,
        "retry_count": 1,
        "error_count": 1,
        "search_rounds": 2,
        "cache_hit_count": 1,
        "rate_limit_wait_seconds": 0.5,
    }
    row["result"]["retrieval_diagnostics"] = {
        "raw_count": 12,
        "deduplicated_count": 8,
        "source_stats": [
            {"source": "openalex", "error_message": "timeout"},
            {"source": "arxiv", "error_message": None},
        ],
    }

    result = evaluate_search_batch.evaluate_batch_results(
        [row],
        [{"case_id": "case_001", "relevant_papers": [{"doi": "10.1/p"}]}],
    )
    efficiency = result["aggregate_report"]["efficiency"]

    assert efficiency["average_latency_seconds"] == pytest.approx(1.25)
    assert efficiency["total_llm_call_count"] == 3
    assert efficiency["total_llm_total_tokens"] == 420
    assert efficiency["average_search_rounds"] == 2
    assert efficiency["total_raw_count"] == 12
    assert efficiency["total_deduplicated_count"] == 8
    assert efficiency["total_returned_result_count"] == 1
    assert efficiency["total_cache_hit_count"] == 1
    assert efficiency["avg_api_call_count"] == 9
    assert efficiency["avg_search_api_call_count"] == 4
    assert efficiency["avg_reference_api_call_count"] == 2
    assert efficiency["avg_retry_count"] == 1
    assert efficiency["avg_error_count"] == 1
    assert efficiency["avg_cache_hit_count"] == 1
    assert efficiency["avg_rate_limit_wait_seconds"] == pytest.approx(0.5)
    assert efficiency["avg_llm_call_count"] == 3
    assert efficiency["avg_llm_total_tokens"] == 420
    assert efficiency["total_source_call_count"] == 6
    assert efficiency["total_source_error_count"] == 1
    assert efficiency["warnings"] == []


def test_cli_legacy_include_partial_and_conflict(tmp_path: Path) -> None:
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl",
        [_batch_row("case_001", partial=[_ranked("Partial", doi="10.1/p")])],
    )
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"case_id": "case_001", "relevant_papers": [{"doi": "10.1/p"}]}],
    )
    output_path = tmp_path / "report.json"

    code = evaluate_search_batch.main(
        [
            "--batch-results",
            str(batch_path),
            "--gold",
            str(gold_path),
            "--output",
            str(output_path),
            "--include-partial",
        ]
    )
    conflict = evaluate_search_batch.main(
        [
            "--batch-results",
            str(batch_path),
            "--gold",
            str(gold_path),
            "--include-partial",
            "--result-policy",
            "highly_only",
        ]
    )

    assert code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["config"][
        "result_policy"
    ] == "highly_and_partial"
    assert conflict == 1


def test_cli_invalid_result_policy_returns_parser_error(tmp_path: Path) -> None:
    batch_path = _write_jsonl(tmp_path / "batch.jsonl", [])
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])

    with pytest.raises(SystemExit) as exc_info:
        evaluate_search_batch.main(
            [
                "--batch-results",
                str(batch_path),
                "--gold",
                str(gold_path),
                "--result-policy",
                "invalid",
            ]
        )

    assert exc_info.value.code == 2


def test_invalid_jsonl_returns_nonzero(tmp_path: Path) -> None:
    batch_path = tmp_path / "bad.jsonl"
    batch_path.write_text('{"case_id": "ok"}\n{bad-json}\n', encoding="utf-8")
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"case_id": "ok", "relevant_papers": []}],
    )

    code = evaluate_search_batch.main(
        ["--batch-results", str(batch_path), "--gold", str(gold_path)]
    )

    assert code == 1


def test_non_object_jsonl_returns_nonzero(tmp_path: Path) -> None:
    batch_path = tmp_path / "batch.jsonl"
    batch_path.write_text('["not", "object"]\n', encoding="utf-8")
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"case_id": "ok", "relevant_papers": []}],
    )

    code = evaluate_search_batch.main(
        ["--batch-results", str(batch_path), "--gold", str(gold_path)]
    )

    assert code == 1


def test_missing_file_returns_nonzero(tmp_path: Path) -> None:
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"case_id": "ok", "relevant_papers": []}],
    )

    code = evaluate_search_batch.main(
        [
            "--batch-results",
            str(tmp_path / "missing.jsonl"),
            "--gold",
            str(gold_path),
        ]
    )

    assert code == 1


def test_stdout_output_without_output_path(tmp_path: Path, capsys) -> None:
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl",
        [_batch_row("case_001", high=[_ranked("Paper A", doi="10.1/a")])],
    )
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"case_id": "case_001", "relevant_papers": [{"doi": "10.1/a"}]}],
    )

    code = evaluate_search_batch.main(
        ["--batch-results", str(batch_path), "--gold", str(gold_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["evaluated_case_count"] == 1
    assert payload["aggregate"]["mrr"] == pytest.approx(1.0)


def _batch_row(
    case_id: str,
    *,
    high: list[dict[str, Any]] | None = None,
    partial: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": f"query for {case_id}",
        "status": "succeeded",
        "result": {
            "highly_relevant_papers": high or [],
            "partially_relevant_papers": partial or [],
        },
        "error": None,
        "latency_seconds": 0.1,
    }


def _ranked(
    title: str,
    *,
    year: int = 2025,
    doi: str | None = None,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    pubmed_id: str | None = None,
) -> dict[str, Any]:
    return {
        "paper": {
            "title": title,
            "year": year,
            "identifiers": {
                "doi": doi,
                "arxiv_id": arxiv_id,
                "openalex_id": openalex_id,
                "semantic_scholar_id": semantic_scholar_id,
                "pubmed_id": pubmed_id,
            },
        }
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_jsonl_for_rows(rows: list[dict[str, Any]]) -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "rows.jsonl"
    return _write_jsonl(path, rows)
