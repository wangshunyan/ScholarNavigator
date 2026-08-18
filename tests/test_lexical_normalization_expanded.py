from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scholar_agent.evaluation.lexical_normalization_expanded import (
    METRIC_VERSION,
    build_stratified_statistics,
    resolve_record_terminals,
    write_expanded_lexical_audit,
)


ROOT = Path(__file__).resolve().parents[1]


class _Store:
    def __init__(self, entries: dict[str, SimpleNamespace]) -> None:
        self.entries = entries
        self.read_keys: list[str] = []

    def read_retrieval(self, key: str) -> SimpleNamespace:
        self.read_keys.append(key)
        return self.entries[key]


def _row(calls: list[dict]) -> dict:
    return {
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": "initial_retrieval",
                    "retrieval_calls": calls,
                }
            ]
        }
    }


def _call(
    source: str,
    key: str | None,
    *,
    query: str = "query",
    terminal: str | None = None,
) -> dict:
    return {
        "source": source,
        "adapted_query": query,
        "logical_call_executed": True,
        "snapshot_key": key,
        "terminal_status": terminal,
    }


def _case(
    *,
    successful_sources: int,
    overlap: bool,
    baseline_recall: float,
    experiment_recall: float,
) -> dict:
    return {
        "successful_source_count": successful_sources,
        "overlaps_prior_auto_dev_val": overlap,
        "evaluable_gold_count": 1,
        "candidate_gold_count": 1,
        "baseline": {
            "recall_at_20": baseline_recall,
            "f1_at_20": baseline_recall / 2,
        },
        "experiment": {
            "recall_at_20": experiment_recall,
            "f1_at_20": experiment_recall / 2,
        },
    }


def test_record_terminals_resolve_success_failure_and_not_started() -> None:
    success = "a" * 64
    failed = "b" * 64
    store = _Store(
        {
            success: SimpleNamespace(
                source="arxiv", adapted_query="query", status="success"
            ),
            failed: SimpleNamespace(
                source="openalex", adapted_query="query", status="failed"
            ),
        }
    )
    original = _row(
        [
            _call("arxiv", success),
            _call("openalex", failed),
            _call("pubmed", None, terminal="timeout"),
        ]
    )

    prepared, audit = resolve_record_terminals(
        original,
        store=store,
        configured_sources=["openalex", "arxiv", "semantic_scholar", "pubmed"],
    )

    assert store.read_keys == [success, failed]
    assert audit["source_states"] == {
        "openalex": "failed",
        "arxiv": "success",
        "semantic_scholar": "not_started",
        "pubmed": "not_started",
    }
    assert audit["successful_source_count"] == 1
    assert audit["snapshot_key_count"] == 2
    prepared_calls = prepared["stage_diagnostics"]["snapshots"][0][
        "retrieval_calls"
    ]
    assert prepared_calls[0]["terminal_status"] == "success"
    assert prepared_calls[1]["terminal_status"] == "failed"
    assert original["stage_diagnostics"]["snapshots"][0]["retrieval_calls"][0][
        "terminal_status"
    ] is None


def test_no_snapshot_keys_is_closed_as_no_successful_source() -> None:
    _, audit = resolve_record_terminals(
        _row([_call("subquery", None, terminal="timeout")]),
        store=_Store({}),
        configured_sources=["openalex", "arxiv", "semantic_scholar", "pubmed"],
    )
    assert audit["successful_source_count"] == 0
    assert set(audit["source_states"].values()) == {"not_started"}


def test_record_terminal_mismatch_and_duplicate_key_are_rejected() -> None:
    key = "c" * 64
    store = _Store(
        {
            key: SimpleNamespace(
                source="arxiv", adapted_query="query", status="success"
            )
        }
    )
    with pytest.raises(ValueError, match="disagrees"):
        resolve_record_terminals(
            _row([_call("arxiv", key, terminal="failed")]),
            store=store,
            configured_sources=["arxiv"],
        )
    with pytest.raises(ValueError, match="duplicate Snapshot key"):
        resolve_record_terminals(
            _row([_call("arxiv", key), _call("arxiv", key)]),
            store=store,
            configured_sources=["arxiv"],
        )


def test_stratified_paired_statistics_are_deterministic() -> None:
    manifest = json.loads(
        (ROOT / "benchmark/lexical_normalization_record160_manifest.json").read_text()
    )
    manifest["bootstrap"]["iterations"] = 100
    manifest["permutation_test"]["iterations"] = 100
    cases = [
        _case(
            successful_sources=1,
            overlap=True,
            baseline_recall=0.0,
            experiment_recall=1.0,
        ),
        _case(
            successful_sources=2,
            overlap=False,
            baseline_recall=1.0,
            experiment_recall=1.0,
        ),
        _case(
            successful_sources=3,
            overlap=False,
            baseline_recall=1.0,
            experiment_recall=0.0,
        ),
        _case(
            successful_sources=4,
            overlap=False,
            baseline_recall=0.0,
            experiment_recall=0.0,
        ),
    ]

    first = build_stratified_statistics(cases, manifest)
    second = build_stratified_statistics(cases, manifest)

    assert first == second
    assert first["all_160"]["query_count"] == 4
    assert first["new_excluding_prior_auto_dev_val"]["query_count"] == 3
    assert first["by_successful_source_count"]["1"]["query_count"] == 1
    assert first["all_160"]["metrics"]["candidate_recall"][
        "mean_paired_difference"
    ] == 0.0
    assert first["all_160"]["metrics"]["recall_at_20"]["outcomes"] == {
        "improved": 1,
        "tied": 2,
        "regressed": 1,
    }


def test_expanded_manifest_freezes_v2_and_zero_io() -> None:
    manifest = json.loads(
        (ROOT / "benchmark/lexical_normalization_record160_manifest.json").read_text()
    )
    assert manifest["metric_version"] == METRIC_VERSION
    assert manifest["policy"]["default"] == "off"
    assert manifest["inclusion"] == {
        "expected_main_case_count": 160,
        "expected_no_success_case_count": 2,
        "main": "at_least_one_configured_source_has_a_success_snapshot",
        "no_success": "separate_not_scored",
        "pairing_mismatch": "stop_that_case_and_report",
    }
    assert manifest["frozen_invariants"]["network_request_count"] == 0
    assert manifest["frozen_invariants"]["llm_request_count"] == 0
    assert manifest["frozen_invariants"]["snapshot_write_count"] == 0

    result = json.loads(
        (ROOT / "benchmark/lexical_normalization_record160_result.json").read_text()
    )
    assert result["metric_version"] == METRIC_VERSION
    assert result["closure"] == {
        "included_main_analysis": 160,
        "no_successful_source": 2,
        "record_case_count": 162,
        "successful_source_count_strata": {
            "1": 57,
            "2": 72,
            "3": 30,
            "4": 1,
        },
    }
    assert result["interpretation"]["default_remains_off"] is True
    assert result["execution"]["network_request_count"] == 0
    assert result["execution"]["llm_request_count"] == 0
    assert result["execution"]["snapshot_write_count"] == 0


def test_expanded_audit_writer_is_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"case_id": "q", "analysis_status": "included_main_analysis"}]
    candidates = [{"case_id": "q", "candidate_id": "doi:10.1/example"}]
    aggregate = {"metric_version": METRIC_VERSION, "case_count": 1}
    manifest = ROOT / "benchmark/lexical_normalization_record160_manifest.json"
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_expanded_lexical_audit(
        first, cases, candidates, aggregate, manifest
    )
    write_expanded_lexical_audit(
        second, cases, candidates, aggregate, manifest
    )

    for name in (
        "case_comparison.jsonl",
        "candidate_diagnostics.jsonl",
        "aggregate.json",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
