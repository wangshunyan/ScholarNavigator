from __future__ import annotations

import json
from pathlib import Path

from scripts import check_contest_qualification as qualification


def _metrics(delta: float) -> dict:
    rows = []
    for index in range(200):
        baseline = 0.0 if index < 100 else 0.2
        rows.append(
            {
                "case_id": f"q-{index}",
                "metrics": {
                    "f1_at_k": {"20": baseline + delta},
                    "recall_at_k": {"20": baseline + delta},
                },
            }
        )
    return {"case_statistics": {"total_case_count": 200, "failed_case_count": 0}, "per_case": rows}


def _run(delta: float) -> dict:
    return {
        "config": {
            "judgement_config": {
                "config_version": "current-rules-v1",
                "partially_relevant_threshold": 0.45,
            }
        },
        "config_hashes": {
            "dataset": "auto_scholar_query",
            "dataset_split": "test",
            "offset": 0,
            "limit": 200,
            "top_k": 20,
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "data_hashes": {"pasa": "same"},
        },
        "metrics": _metrics(delta),
        "resource_report": {"status": "passed"},
    }


def test_qualification_requires_positive_bootstrap_interval(monkeypatch) -> None:
    def fake_load(path: Path, _expected: str) -> dict:
        return _run(0.0 if path.name == qualification.EXPECTED_BASELINE else 0.1)

    monkeypatch.setattr(qualification, "_load_run", fake_load)

    report = qualification.check_qualification(
        Path(qualification.EXPECTED_BASELINE),
        Path("contest_qual200_dense_v1"),
    )

    assert report["eligible_for_full_1000"] is True
    assert report["strict_positive_improvement"] == {
        "f1_at_20": True,
        "recall_at_20": True,
    }


def test_qualification_rejects_no_metric_improvement(monkeypatch) -> None:
    monkeypatch.setattr(qualification, "_load_run", lambda *_: _run(0.0))

    report = qualification.check_qualification(
        Path(qualification.EXPECTED_BASELINE),
        Path("contest_qual200_reranker_v1"),
    )

    assert report["eligible_for_full_1000"] is False
    assert not any(report["strict_positive_improvement"].values())


def test_reranker_audit_rejects_fallback_and_accepts_real_inference(tmp_path: Path) -> None:
    path = tmp_path / "contest_qual200_reranker_v2"
    path.mkdir()
    rows = []
    for index in range(200):
        rows.append(
            {
                "case_id": f"q-{index}",
                "diagnostics": {
                    "local_model_batch_count": 15,
                    "local_model_fallback_count": 0,
                    "local_model_inference_success_count": 1,
                    "local_model_candidate_count": 120,
                    "local_model_prompt_version": "qwen3-reranker-v1",
                    "local_model_device": "cuda:0",
                    "local_model_max_length": 2048,
                    "local_model_fingerprint": "model-fingerprint",
                    "local_model_latency_seconds": 0.5,
                    "local_model_batch_size": 8,
                    "local_model_candidate_limit": 120,
                    "local_model_peak_vram_bytes": 123456,
                },
            }
        )
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    passed = qualification._audit_reranker_run(path)
    assert passed["status"] == "passed"
    assert passed["fallback_count"] == 0

    rows[0]["diagnostics"]["local_model_fallback_count"] = 1
    (path / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    failed = qualification._audit_reranker_run(path)
    assert failed["status"] == "failed"
    assert "reranker_fallback_detected" in failed["reasons"]

    assert qualification._audit_reranker_run(path, expected_rows=1000)["status"] == "failed"


def test_gpu_isolated_reranker_retry_is_an_explicit_candidate() -> None:
    assert "contest_qual200_reranker_v2_gpu1" in qualification.EXPECTED_CANDIDATES
    assert "contest_qual200_reranker_v3_gpu1" in qualification.EXPECTED_CANDIDATES
    assert "contest_qual200_reranker_v4_gpu1" in qualification.EXPECTED_CANDIDATES


def test_soft_judgement_qualification_requires_reranker_v4_baseline(
    monkeypatch,
) -> None:
    baseline = _run(0.0)
    candidate = _run(0.1)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.35,
    }

    def fake_load(path: Path, expected: str) -> dict:
        assert path.name == expected
        return baseline if path.name == qualification.SOFT_JUDGEMENT_BASELINE else candidate

    monkeypatch.setattr(qualification, "_load_run", fake_load)
    report = qualification.check_qualification(
        Path(qualification.SOFT_JUDGEMENT_BASELINE),
        Path(qualification.SOFT_JUDGEMENT_CANDIDATE),
    )

    assert report["baseline_run_id"] == qualification.SOFT_JUDGEMENT_BASELINE
    assert report["eligible_for_full_1000"] is True


def test_soft_judgement_qualification_rejects_unreviewed_config_delta(
    monkeypatch,
) -> None:
    baseline = _run(0.0)
    candidate = _run(0.1)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.30,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: baseline if path.name == qualification.SOFT_JUDGEMENT_BASELINE else candidate,
    )

    try:
        qualification.check_qualification(
            Path(qualification.SOFT_JUDGEMENT_BASELINE),
            Path(qualification.SOFT_JUDGEMENT_CANDIDATE),
        )
    except ValueError as exc:
        assert str(exc) == "soft_judgement_delta_not_allowlisted"
    else:
        raise AssertionError("unreviewed soft judgement delta was accepted")


def _rrf_soft_run(
    delta: float, *, recall: float, fn_rate: float, latency: float = 4.0
) -> dict:
    run = _run(delta)
    run["config"].update(
        {
            "judgement_policy": "current_rules",
            "local_hybrid": {
                "fusion": {
                    "method": "reciprocal_rank_fusion",
                    "rrf_k": 60,
                    "bm25_candidate_limit": 60,
                    "semantic_candidate_limit": 60,
                }
            },
        }
    )
    run["stage_metrics"] = {
        "case_count": 200,
        "initial_retrieval_recall": recall,
        "judgement": {"gold_false_negative_rate": fn_rate},
    }
    run["metrics"]["benchmark_statistics"] = {
        "average_latency_seconds": latency,
    }
    run["reranker_audit"] = {"status": "passed"}
    return run


def test_rrf_soft_qualification_requires_stage_metric_improvements(monkeypatch) -> None:
    baseline = _rrf_soft_run(0.0, recall=0.40, fn_rate=0.30)
    candidate = _rrf_soft_run(0.1, recall=0.42, fn_rate=0.20)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.35,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline if path.name == qualification.RRF_SOFT_BASELINE else candidate
        ),
    )

    report = qualification.check_qualification(
        Path(qualification.RRF_SOFT_BASELINE),
        Path(qualification.RRF_SOFT_CANDIDATE),
    )

    assert report["eligible_for_full_1000"] is True
    assert report["stage_metric_gate"] == {
        "baseline_initial_retrieval_recall": 0.40,
        "candidate_initial_retrieval_recall": 0.42,
        "candidate_recall_non_regressed": True,
        "baseline_gold_false_negative_rate": 0.30,
        "candidate_gold_false_negative_rate": 0.20,
        "gold_false_negative_rate_decreased": True,
        "passed": True,
    }
    assert report["efficiency_gate"] == {
        "baseline_average_latency_seconds": 4.0,
        "candidate_average_latency_seconds": 4.0,
        "maximum_average_latency_seconds": 4.4,
        "maximum_multiplier": 1.10,
        "passed": True,
    }


def test_rrf_soft_qualification_rejects_candidate_recall_regression(monkeypatch) -> None:
    baseline = _rrf_soft_run(0.0, recall=0.40, fn_rate=0.30)
    candidate = _rrf_soft_run(0.1, recall=0.39, fn_rate=0.20)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.35,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline if path.name == qualification.RRF_SOFT_BASELINE else candidate
        ),
    )

    report = qualification.check_qualification(
        Path(qualification.RRF_SOFT_BASELINE),
        Path(qualification.RRF_SOFT_CANDIDATE),
    )

    assert report["stage_metric_gate"]["candidate_recall_non_regressed"] is False
    assert report["eligible_for_full_1000"] is False


def test_rrf_soft_qualification_rejects_non_decreasing_false_negative_rate(
    monkeypatch,
) -> None:
    baseline = _rrf_soft_run(0.0, recall=0.40, fn_rate=0.30)
    candidate = _rrf_soft_run(0.1, recall=0.42, fn_rate=0.30)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.35,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline if path.name == qualification.RRF_SOFT_BASELINE else candidate
        ),
    )

    report = qualification.check_qualification(
        Path(qualification.RRF_SOFT_BASELINE),
        Path(qualification.RRF_SOFT_CANDIDATE),
    )

    assert report["stage_metric_gate"]["gold_false_negative_rate_decreased"] is False
    assert report["eligible_for_full_1000"] is False


def test_rrf_soft_qualification_rejects_latency_regression(monkeypatch) -> None:
    baseline = _rrf_soft_run(0.0, recall=0.40, fn_rate=0.30, latency=4.0)
    candidate = _rrf_soft_run(0.1, recall=0.42, fn_rate=0.20, latency=4.5)
    candidate["config"]["judgement_config"] = {
        "config_version": "soft-current-rules-v1",
        "partially_relevant_threshold": 0.35,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline if path.name == qualification.RRF_SOFT_BASELINE else candidate
        ),
    )

    report = qualification.check_qualification(
        Path(qualification.RRF_SOFT_BASELINE),
        Path(qualification.RRF_SOFT_CANDIDATE),
    )

    assert report["efficiency_gate"]["passed"] is False
    assert report["eligible_for_full_1000"] is False


def test_quality_soft_qualification_allows_only_reviewed_policy_delta(
    monkeypatch,
) -> None:
    baseline = _run(0.0)
    baseline["config"]["ranking_policy"] = "current_rules"
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "ranking_policy": "quality_soft_v1",
            "quality_soft_ranking": qualification.quality_soft_ranking_catalog(),
        }
    )

    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: baseline if path.name == qualification.QUALITY_SOFT_BASELINE else candidate,
    )
    report = qualification.check_qualification(
        Path(qualification.QUALITY_SOFT_BASELINE),
        Path(qualification.QUALITY_SOFT_CANDIDATE),
    )

    assert report["candidate_run_id"] == qualification.QUALITY_SOFT_CANDIDATE
    assert report["eligible_for_full_1000"] is True


def test_quality_soft_qualification_rejects_catalog_drift(monkeypatch) -> None:
    baseline = _run(0.0)
    baseline["config"]["ranking_policy"] = "current_rules"
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "ranking_policy": "quality_soft_v1",
            "quality_soft_ranking": {
                **qualification.quality_soft_ranking_catalog(),
                "quality_weight": 0.02,
                "hard_filtering": True,
            },
        }
    )
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: baseline if path.name == qualification.QUALITY_SOFT_BASELINE else candidate,
    )

    try:
        qualification.check_qualification(
            Path(qualification.QUALITY_SOFT_BASELINE),
            Path(qualification.QUALITY_SOFT_CANDIDATE),
        )
    except ValueError as exc:
        assert str(exc) == "quality_soft_catalog_drift"
    else:
        raise AssertionError("quality catalog drift was accepted")


def test_llm_qualification_requires_reranker_baseline_and_llm_audit(
    monkeypatch,
) -> None:
    assert (
        qualification._expected_baseline_for(qualification.LLM_QUALIFICATION_CANDIDATE)
        == qualification.LLM_QUALIFICATION_BASELINE
    )
    baseline = _run(0.0)
    baseline["config"].update(
        {
            "query_planning_policy": "current_rules",
            "budgets": {"max_llm_calls": 0, "max_search_rounds": 1},
        }
    )
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "query_planning_policy": "llm_semantic",
            "llm_mode": "openai_compatible",
            "budgets": {"max_llm_calls": 1, "max_search_rounds": 3},
        }
    )
    candidate["reranker_audit"] = {"status": "passed"}
    candidate["llm_audit"] = {"status": "passed", "fallback_count": 0}

    def fake_load(path: Path, expected: str) -> dict:
        assert path.name == expected
        return baseline if path.name == qualification.LLM_QUALIFICATION_BASELINE else candidate

    monkeypatch.setattr(qualification, "_load_run", fake_load)
    report = qualification.check_qualification(
        Path(qualification.LLM_QUALIFICATION_BASELINE),
        Path(qualification.LLM_QUALIFICATION_CANDIDATE),
    )

    assert report["eligible_for_full_1000"] is True
    assert report["llm_audit_passed"] is True


def test_shared_config_ignores_bm25_path_but_keeps_corpus_identity() -> None:
    baseline = {
        "local_hybrid": {
            "bm25_corpus_path": "/first-worktree/datasets/local_bm25/pasa_papers.jsonl",
            "bm25_corpus_sha256": "same-corpus",
            "bm25_document_count": 569432,
            "reranker_device": "cuda:1",
        }
    }
    candidate = {
        "local_hybrid": {
            "bm25_corpus_path": "/second-worktree/datasets/local_bm25/pasa_papers.jsonl",
            "bm25_corpus_sha256": "same-corpus",
            "bm25_document_count": 569432,
            "reranker_device": "cuda:0",
        }
    }

    assert qualification._shared_config_matches(
        baseline,
        candidate,
        ("local_hybrid",),
    )

    candidate["local_hybrid"]["bm25_corpus_sha256"] = "other-corpus"
    assert not qualification._shared_config_matches(
        baseline,
        candidate,
        ("local_hybrid",),
    )


def test_llm_qualification_rejects_failed_llm_audit(monkeypatch) -> None:
    baseline = _run(0.0)
    baseline["config"].update(
        {
            "query_planning_policy": "current_rules",
            "budgets": {"max_llm_calls": 0, "max_search_rounds": 1},
        }
    )
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "query_planning_policy": "llm_semantic",
            "llm_mode": "openai_compatible",
            "budgets": {"max_llm_calls": 1, "max_search_rounds": 3},
        }
    )
    candidate["reranker_audit"] = {"status": "passed"}
    candidate["llm_audit"] = {"status": "failed", "fallback_count": 1}
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: baseline if path.name == qualification.LLM_QUALIFICATION_BASELINE else candidate,
    )

    report = qualification.check_qualification(
        Path(qualification.LLM_QUALIFICATION_BASELINE),
        Path(qualification.LLM_QUALIFICATION_CANDIDATE),
    )

    assert report["llm_audit_passed"] is False
    assert report["eligible_for_full_1000"] is False


def test_feedback_llm_qualification_requires_live_feedback_evidence(
    monkeypatch,
) -> None:
    assert (
        qualification._expected_baseline_for(
            qualification.LLM_FEEDBACK_QUALIFICATION_CANDIDATE
        )
        == qualification.LLM_FEEDBACK_QUALIFICATION_BASELINE
    )
    baseline = _run(0.0)
    baseline["config"].update(
        {
            "enable_query_evolution": False,
            "query_evolution_policy": "off",
            "query_planning_policy": "current_rules",
            "ranking_policy": "current_rules",
            "budgets": {"max_llm_calls": 0, "max_search_rounds": 1},
        }
    )
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "enable_query_evolution": True,
            "query_evolution_policy": "llm_feedback",
            "query_planning_policy": "current_rules",
            "ranking_policy": "current_rules",
            "llm_mode": "record",
            "llm": {"feedback_query_evolution": True},
            "budgets": {"max_llm_calls": 1, "max_search_rounds": 3},
        }
    )
    candidate["reranker_audit"] = {"status": "passed"}
    candidate["llm_audit"] = {
        "status": "passed",
        "fallback_count": 0,
        "claimable_live_llm_effect": True,
        "llm_call_attempted_count": 125,
        "feedback_eligible_count": 125,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline
            if path.name == qualification.LLM_FEEDBACK_QUALIFICATION_BASELINE
            else candidate
        ),
    )

    report = qualification.check_qualification(
        Path(qualification.LLM_FEEDBACK_QUALIFICATION_BASELINE),
        Path(qualification.LLM_FEEDBACK_QUALIFICATION_CANDIDATE),
    )

    assert report["eligible_for_full_1000"] is True
    assert report["llm_audit_passed"] is True
    assert report["live_llm_effect_verified"] is True


def test_feedback_llm_qualification_rejects_replay_or_zero_live_calls(
    monkeypatch,
) -> None:
    baseline = _run(0.0)
    baseline["config"].update(
        {
            "enable_query_evolution": False,
            "query_evolution_policy": "off",
            "query_planning_policy": "current_rules",
            "ranking_policy": "current_rules",
            "budgets": {"max_llm_calls": 0, "max_search_rounds": 1},
        }
    )
    candidate = _run(0.1)
    candidate["config"].update(
        {
            "enable_query_evolution": True,
            "query_evolution_policy": "llm_feedback",
            "query_planning_policy": "current_rules",
            "ranking_policy": "current_rules",
            "llm_mode": "replay",
            "llm": {"feedback_query_evolution": True},
            "budgets": {"max_llm_calls": 1, "max_search_rounds": 3},
        }
    )
    candidate["reranker_audit"] = {"status": "passed"}
    candidate["llm_audit"] = {
        "status": "passed",
        "fallback_count": 0,
        "claimable_live_llm_effect": False,
        "llm_call_attempted_count": 0,
        "feedback_eligible_count": 0,
    }
    monkeypatch.setattr(
        qualification,
        "_load_run",
        lambda path, _: (
            baseline
            if path.name == qualification.LLM_FEEDBACK_QUALIFICATION_BASELINE
            else candidate
        ),
    )

    try:
        qualification.check_qualification(
            Path(qualification.LLM_FEEDBACK_QUALIFICATION_BASELINE),
            Path(qualification.LLM_FEEDBACK_QUALIFICATION_CANDIDATE),
        )
    except ValueError as exc:
        assert str(exc) == "llm_feedback_qualification_llm_mode_invalid"
    else:
        raise AssertionError("replay feedback run was accepted for live qualification")
