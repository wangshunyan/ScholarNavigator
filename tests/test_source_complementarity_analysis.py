from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_source_complementarity import (  # noqa: E402
    build_source_complementarity_analysis,
)
from scholar_agent.services.search_service import SearchService  # noqa: E402


def test_analysis_reports_source_exclusive_gold_and_acceptance(tmp_path: Path) -> None:
    paths = _six_runs(tmp_path)

    result = build_source_complementarity_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    validation = result["validation"]
    contribution = validation["source_gold_contribution"]
    assert contribution["arxiv_exclusive_gold_count"] == 1
    assert contribution["openalex_exclusive_gold_count"] == 1
    assert contribution["overlap_gold_count"] == 1
    assert result["validation_acceptance"]["accepted"] is True
    assert result["validation_acceptance"]["high_recall_profile_candidate"] is True
    assert result["product_default_sources_changed"] is False


def test_analysis_writes_all_required_outputs_deterministically(tmp_path: Path) -> None:
    paths = _six_runs(tmp_path)
    first = tmp_path / "analysis-first"
    second = tmp_path / "analysis-second"

    build_source_complementarity_analysis(**paths, output_dir=first)
    build_source_complementarity_analysis(**paths, output_dir=second)

    names = (
        "development_comparison.json",
        "validation_comparison.json",
        "source_gold_contribution.jsonl",
        "error_analysis.json",
        "summary.md",
    )
    assert all((first / name).is_file() for name in names)
    assert {
        name: _sha(first / name) for name in names
    } == {
        name: _sha(second / name) for name in names
    }


def test_gold_diagnostics_distinguish_identifier_mismatch_and_stage_loss(
    tmp_path: Path,
) -> None:
    paths = _six_runs(tmp_path)
    output = tmp_path / "analysis"

    result = build_source_complementarity_analysis(**paths, output_dir=output)
    rows = _jsonl(output / "source_gold_contribution.jsonl")
    validation_rows = [row for row in rows if row["split"] == "validation"]

    mismatch = validation_rows[3]
    assert mismatch["source_classification"] == "openalex_identifier_mismatch"
    assert mismatch["openalex_identifier_mismatch"] is True
    assert "openalex_identifier_mismatch" in mismatch["diagnostic_labels"]
    losses = result["validation"]["groups"]["arxiv_openalex"]["stage_losses"]
    assert losses == {
        "retrieval_missing_gold_count": 17,
        "judgement_filtered_gold_count": 0,
        "top_20_filtered_gold_count": 0,
        "returned_gold_count": 3,
        "drop_reasons": {"not_retrieved": 17, "returned": 3},
    }


def test_failed_snapshot_is_covered_and_counted_as_source_error(tmp_path: Path) -> None:
    paths = _six_runs(tmp_path, failed_openalex_snapshot=True)

    result = build_source_complementarity_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    costs = result["validation"]["groups"]["openalex_only"][
        "source_recorded_costs"
    ]["openalex"]
    assert costs["snapshot_entry_count"] == 1
    assert costs["failed_entry_count"] == 1
    assert costs["request_count"] == 2
    assert costs["retry_count"] == 1
    assert costs["error_count"] == 1
    assert costs["error_rate"] == 0.5


def test_multi_source_dedup_and_openalex_identifier_completeness(
    tmp_path: Path,
) -> None:
    paths = _six_runs(tmp_path)

    result = build_source_complementarity_analysis(
        **paths,
        output_dir=tmp_path / "analysis",
    )

    combined = result["development"]["groups"]["arxiv_openalex"]
    assert combined["deduplication"] == {
        "raw_candidate_count": 40,
        "deduplicated_candidate_count": 20,
        "removed_duplicate_count": 20,
        "deduplication_rate": 0.5,
    }
    identifiers = combined["openalex_identifier_completeness"]
    assert identifiers["unique_openalex_candidate_count"] == 20
    assert identifiers["any_stable_identifier_rate"] == 1.0
    assert identifiers["openalex_id_rate"] == 1.0
    assert identifiers["doi_rate"] == 0.0


def test_analysis_rejects_group_configuration_drift(tmp_path: Path) -> None:
    paths = _six_runs(tmp_path)
    config_path = paths["validation_openalex"] / "config.json"
    config = _json(config_path)
    config["budgets"]["max_candidate_papers"] = 999
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="incompatible"):
        build_source_complementarity_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def test_analysis_rejects_nonzero_replay_network_cost(tmp_path: Path) -> None:
    paths = _six_runs(tmp_path)
    metrics_path = paths["validation_combined"] / "metrics.json"
    metrics = _json(metrics_path)
    metrics["snapshot_costs"]["replay_execution_request_count"] = 1
    _write_json(metrics_path, metrics)

    with pytest.raises(ValueError, match="replay used network"):
        build_source_complementarity_analysis(
            **paths,
            output_dir=tmp_path / "analysis",
        )


def test_gold_analysis_is_not_imported_by_production_search() -> None:
    source = inspect.getsource(SearchService).casefold()

    assert "source_complementarity" not in source
    assert "source_gold_contribution" not in source


def _six_runs(
    root: Path,
    *,
    failed_openalex_snapshot: bool = False,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    names = {
        "development_arxiv": ("development", "arxiv_only"),
        "development_openalex": ("development", "openalex_only"),
        "development_combined": ("development", "arxiv_openalex"),
        "validation_arxiv": ("validation", "arxiv_only"),
        "validation_openalex": ("validation", "openalex_only"),
        "validation_combined": ("validation", "arxiv_openalex"),
    }
    for name, (split, group) in names.items():
        paths[name] = _run(
            root,
            name,
            split=split,
            group=group,
            failed_snapshot=(failed_openalex_snapshot and name == "validation_openalex"),
        )
    return paths


def _run(
    root: Path,
    name: str,
    *,
    split: str,
    group: str,
    failed_snapshot: bool,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    offset = 90 if split == "development" else 110
    sources = {
        "arxiv_only": ["arxiv"],
        "openalex_only": ["openalex"],
        "arxiv_openalex": ["arxiv", "openalex"],
    }[group]
    snapshot_dir = root / f"{name}-snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "retrieval").mkdir()
    keys = []
    for source in sources:
        key = f"{group}-{source}"
        keys.append(key)
        failed = failed_snapshot and source == "openalex"
        _write_json(
            snapshot_dir / "retrieval" / f"{key}.json",
            {
                "key": key,
                "source": source,
                "status": "failed" if failed else "success",
                "recorded_latency_seconds": 2.0,
                "diagnostics": {
                    "request_count": 2 if failed else 1,
                    "retry_count": 1 if failed else 0,
                    "error_count": 1 if failed else 0,
                    "rate_limit_wait_seconds": 0.5,
                },
            },
        )
    _write_json(
        snapshot_dir / "manifest.json",
        {
            "groups": {
                "baseline": {
                    "retrieval_keys": keys,
                    "replay_ready": True,
                    "missing_key_count": 0,
                }
            }
        },
    )

    case_ids = [f"AutoScholarQuery_test_{index}" for index in range(offset, offset + 20)]
    config = {
        "dataset": "auto_scholar_query",
        "dataset_sha256": "fixed-sha",
        "case_ids": case_ids,
        "offset": offset,
        "limit": 20,
        "sources": sources,
        "query_adapter_policy": "adaptive",
        "query_planning_policy": "current_rules",
        "query_planner_version": "1.4.0",
        "judgement_policy": "current_rules",
        "judgement_config_hash": "rules-hash",
        "run_profile": "balanced",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": 2026,
        "max_workers": 1,
        "budgets": {"max_candidate_papers": 200, "max_search_rounds": 2},
        "diagnostics": True,
        "llm": {"query_understanding": False, "judgement": False},
        "enable_query_evolution": False,
        "query_evolution_policy": "off",
        "enable_refchain": False,
        "runtime_code_hash": "runtime-hash",
        "retrieval_mode": "replay",
        "snapshot": {
            "directory": str(snapshot_dir),
            "group": "baseline",
            "name": snapshot_dir.name,
        },
    }
    found_indexes = {
        "arxiv_only": {0, 1},
        "openalex_only": {0, 2},
        "arxiv_openalex": {0, 1, 2},
    }[group]
    results = [
        _result(
            case_id,
            index=index,
            group=group,
            found=index in found_indexes,
        )
        for index, case_id in enumerate(case_ids)
    ]
    unique_gold = len(found_indexes)
    metric = {"arxiv_only": 0.1, "openalex_only": 0.08, "arxiv_openalex": 0.2}[group]
    recorded_requests = {"arxiv_only": 10, "openalex_only": 12, "arxiv_openalex": 20}[group]
    metrics = {
        "case_count": 20,
        "end_to_end_metrics": {
            "f1_at_k": {"5": metric, "10": metric, "20": metric},
            "precision_at_k": {"20": metric},
            "recall_at_k": {"20": metric},
            "mrr": metric,
            "ndcg_at_k": {"20": metric},
        },
        "snapshot_costs": {
            "retrieval_snapshot_hits": recorded_requests,
            "recorded_search_request_count": recorded_requests,
            "recorded_retry_count": 0,
            "recorded_error_count": 0,
            "recorded_rate_limit_wait_seconds": 1.0,
            "recorded_latency_seconds": 10.0,
            "replay_execution_request_count": 0,
            "replay_execution_retry_count": 0,
            "replay_execution_network_wait_seconds": 0.0,
        },
    }
    stage_metrics = {
        "case_count": 20,
        "gold_count": 20,
        "initial_retrieval_recall": unique_gold / 20,
        "initial_query_planning": {"unique_candidate_count": 20},
        "judgement": {"retrieved_gold_count": unique_gold},
        "source_contribution": {
            "sources": {},
            "overlap": {},
            "source_error_rate": 0.0,
        },
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "stage_metrics.json", stage_metrics)
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    return run_dir


def _result(case_id: str, *, index: int, group: str, found: bool) -> dict[str, object]:
    gold_title = f"Gold Paper {index}"
    source = "openalex" if group == "openalex_only" else "arxiv"
    candidate = {
        "identifiers": {
            "doi": None,
            "arxiv_id": None,
            "semantic_scholar_id": None,
            "openalex_id": f"W{index}" if source == "openalex" else None,
            "pubmed_id": None,
        },
        "title": gold_title if index == 3 and group == "openalex_only" else f"Candidate {index}",
        "year": 2026,
        "sources": [source],
    }
    if group == "arxiv_openalex":
        openalex = {
            **candidate,
            "identifiers": {**candidate["identifiers"], "openalex_id": f"W{index}"},
            "sources": ["openalex"],
        }
        raw_candidates = [candidate, openalex]
        deduplicated = [openalex]
    else:
        raw_candidates = [candidate]
        deduplicated = [candidate]
    drop_reason = "returned" if found else (
        "identifier_not_matched"
        if index == 3 and group == "openalex_only"
        else "not_retrieved"
    )
    return {
        "case_id": case_id,
        "query": f"Fixture query {index}",
        "status": "succeeded",
        "stage_diagnostics": {
            "snapshots": [
                {"stage": "initial_retrieval", "candidates": raw_candidates},
                {"stage": "initial_deduplicated", "candidates": deduplicated},
            ],
            "gold_diagnostics": [
                {
                    "case_id": case_id,
                    "query": f"Fixture query {index}",
                    "gold_id": f"arxiv:{index:04d}.00001",
                    "gold_title": gold_title,
                    "found": found,
                    "final_rank": 1 if found else None,
                    "drop_reason": drop_reason,
                }
            ],
        },
    }


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
