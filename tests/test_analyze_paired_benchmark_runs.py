from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_paired_benchmark_runs import analyze_paired_runs


def _write_run(
    root: Path,
    name: str,
    *,
    candidate: bool = False,
    drift: bool = False,
    runtime_code_hash: str = "runtime-hash",
) -> Path:
    run = root / name
    run.mkdir()
    case_ids = ["case-0", "case-1"]
    config = {
        "dataset": "fixture",
        "dataset_split": "test",
        "dataset_sha256": "dataset-hash",
        "case_ids": case_ids,
        "offset": 0,
        "limit": 2,
        "query_adapter_policy": "adaptive",
        "run_profile": "high_recall",
        "result_policy": "highly_and_partial",
        "top_k": 20,
        "current_year": None,
        "max_workers": 1,
        "budgets": {"max_candidate_papers": 300},
        "diagnostics": True,
        "llm": {"requested": False},
        "prompts": [],
        "runtime_code_hash": runtime_code_hash,
        "sources": ["candidate"] if candidate else ["baseline"],
        "ranking_policy": "candidate" if candidate else "current_rules",
    }
    if drift:
        config["run_profile"] = "fast"
    rows = []
    for index, case_id in enumerate(case_ids):
        base = 0.1 * (index + 1)
        gain = 0.05 if candidate else 0.0
        rows.append(
            {
                "case_id": case_id,
                "metrics": {
                    "f1_at_k": {"5": base + gain, "10": base + gain, "20": base + gain},
                    "recall_at_k": {"20": base + gain},
                },
            }
        )
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"per_case": rows}), encoding="utf-8"
    )
    return run


def test_analyze_paired_runs_reports_deterministic_ci(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)

    report = analyze_paired_runs(baseline, candidate, seed=7, iterations=200)

    assert report["shared_inputs_match"] is True
    assert report["query_count"] == 2
    assert report["strict_positive_improvement"]["f1_at_20"] is True
    assert report["comparisons"]["f1_at_20"]["seed"] == 7
    assert report["comparisons"]["f1_at_20"]["iterations"] == 200
    assert report["reported_config_differences"]["sources"]["candidate"] == [
        "candidate"
    ]


def test_analyze_paired_runs_rejects_shared_config_drift(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True, drift=True)

    with pytest.raises(ValueError, match="shared_config_drift:run_profile"):
        analyze_paired_runs(baseline, candidate)


def test_analyze_paired_runs_can_opt_into_one_strategy_difference(
    tmp_path: Path,
) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)
    baseline_config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    candidate_config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    baseline_config["query_planning_policy"] = "current_rules"
    candidate_config["query_planning_policy"] = "facet_balanced"
    candidate_config["ranking_policy"] = baseline_config["ranking_policy"]
    (baseline / "config.json").write_text(json.dumps(baseline_config), encoding="utf-8")
    (candidate / "config.json").write_text(json.dumps(candidate_config), encoding="utf-8")

    with pytest.raises(ValueError, match="shared_config_drift:query_planning_policy"):
        analyze_paired_runs(baseline, candidate)

    report = analyze_paired_runs(
        baseline,
        candidate,
        allow_strategy_difference=True,
        seed=7,
        iterations=20,
    )
    assert report["strategy_difference_allowed"] is True
    assert report["strategy_differences"] == ["query_planning_policy"]


def test_analyze_paired_runs_reports_runtime_code_change(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", runtime_code_hash="old-code")
    candidate = _write_run(
        tmp_path,
        "candidate",
        candidate=True,
        runtime_code_hash="new-code",
    )

    report = analyze_paired_runs(baseline, candidate, seed=7, iterations=200)

    assert report["shared_inputs_match"] is True
    assert report["reported_config_differences"]["runtime_code_hash"] == {
        "baseline": "old-code",
        "candidate": "new-code",
    }


def test_disabled_llm_provider_label_does_not_break_offline_pair(
    tmp_path: Path,
) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)
    for path, provider in (
        (baseline / "config.json", "openai_compatible"),
        (candidate / "config.json", "disabled"),
    ):
        config = json.loads(path.read_text(encoding="utf-8"))
        config["llm"] = {
            "requested": False,
            "llm_enabled": False,
            "runtime_available": False,
            "model": None,
            "provider": provider,
        }
        path.write_text(json.dumps(config), encoding="utf-8")

    report = analyze_paired_runs(baseline, candidate, seed=7, iterations=20)
    assert report["shared_inputs_match"] is True


def test_analyze_paired_runs_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline")
    candidate = _write_run(tmp_path, "candidate", candidate=True)
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    metrics["per_case"][1]["case_id"] = "case-0"
    (candidate / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="per_case_metric_identity_invalid"):
        analyze_paired_runs(baseline, candidate)


def _write_strict_reranker_pair(root: Path) -> tuple[Path, Path]:
    case_ids = [f"case-{index}" for index in range(200)]
    hybrid = {
        "connector_version": "local-hybrid-v3",
        "bm25_corpus_sha256": "b" * 64,
        "bm25_document_count": 569432,
        "semantic_corpus_sha256": "s" * 64,
        "semantic_corpus_size_bytes": 100,
        "semantic_document_count": 569432,
        "semantic_abstract_document_count": 569432,
        "embedding_dimension": 384,
        "model_fingerprint": "m" * 64,
        "index_fingerprint": "i" * 64,
        "fusion": {"method": "reciprocal_rank_fusion", "rrf_k": 60},
        "query_input": "current_rules_generated_text_only",
        "evaluator_data_access": False,
        "reranker_model_path": None,
        "reranker_batch_size": 8,
        "reranker_candidate_limit": 120,
        "reranker_device": "auto",
    }

    def write(name: str, *, reranker: bool) -> Path:
        run = root / name
        run.mkdir()
        local_hybrid = dict(hybrid)
        if reranker:
            local_hybrid.update(
                {"reranker_model_path": "<redacted-path>", "reranker_device": "cuda:1"}
            )
        config = {
            "dataset": "fixture",
            "dataset_split": "test",
            "dataset_sha256": "d" * 64,
            "case_ids": case_ids,
            "case_count": 200,
            "offset": 0,
            "limit": 200,
            "query_adapter_policy": "adaptive",
            "run_profile": "high_recall",
            "result_policy": "highly_and_partial",
            "top_k": 20,
            "current_year": 2026,
            "max_workers": 1,
            "budgets": {"max_candidate_papers": 300},
            "diagnostics": True,
            "llm": {"requested": False},
            "prompts": [],
            "runtime_code_hash": "runtime" * 9,
            "code": {"commit": "c" * 40, "dirty": False},
            "sources": ["local_hybrid"],
            "local_hybrid": local_hybrid,
        }
        rows = [
            {
                "case_id": case_id,
                "metrics": {"f1_at_k": {"5": 0.1, "10": 0.1, "20": 0.1}, "recall_at_k": {"20": 0.1}},
            }
            for case_id in case_ids
        ]
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "metrics.json").write_text(
            json.dumps({"per_case": rows, "case_statistics": {"total_case_count": 200, "failed_case_count": 0}}),
            encoding="utf-8",
        )
        result_rows = []
        for case_id in case_ids:
            diagnostics = {}
            if reranker:
                diagnostics = {
                    "local_model_batch_count": 1,
                    "local_model_fallback_count": 0,
                    "local_model_inference_success_count": 1,
                    "local_model_candidate_count": 120,
                    "local_model_prompt_version": "qwen3-reranker-v1",
                    "local_model_device": "cuda:1",
                    "local_model_max_length": 2048,
                    "local_model_fingerprint": "f" * 64,
                    "local_model_latency_seconds": 0.1,
                    "local_model_batch_size": 8,
                    "local_model_candidate_limit": 120,
                    "local_model_peak_vram_bytes": 1,
                }
            result_rows.append({"case_id": case_id, "diagnostics": diagnostics})
        (run / "results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in result_rows) + "\n", encoding="utf-8"
        )
        return run

    return write("baseline", reranker=False), write("candidate", reranker=True)


def test_strict_reranker_only_accepts_clean_200_query_pair(tmp_path: Path) -> None:
    baseline, candidate = _write_strict_reranker_pair(tmp_path)

    for run, process_id in ((baseline, 101), (candidate, 202)):
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["execution"] = {
            "process_id": process_id,
            "launch_command": ["run_benchmark.py", "--run-id", run.name],
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

    report = analyze_paired_runs(
        baseline, candidate, strict_reranker_only=True, iterations=20
    )

    assert report["strict_reranker_only"] is True
    assert report["reranker_audit"]["status"] == "passed"


def test_strict_reranker_only_rejects_hybrid_input_drift(tmp_path: Path) -> None:
    baseline, candidate = _write_strict_reranker_pair(tmp_path)
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    config["local_hybrid"]["index_fingerprint"] = "other-index"
    (candidate / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="strict_reranker_only_hybrid_drift:index_fingerprint"):
        analyze_paired_runs(baseline, candidate, strict_reranker_only=True)
