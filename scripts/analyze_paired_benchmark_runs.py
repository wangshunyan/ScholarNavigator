#!/usr/bin/env python3
"""Produce a deterministic paired analysis for two completed benchmark runs.

This utility consumes only ``config.json`` and evaluator-produced
``metrics.json`` from each run.  Its optional strict reranker mode also reads
the candidate's already-produced result diagnostics to verify real neural
inference; it never loads query gold, qrels, documents, or an online
connector, so it cannot leak evaluator data into retrieval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


_SHARED_FIELDS = (
    "dataset",
    "dataset_split",
    "dataset_sha256",
    "case_ids",
    "offset",
    "limit",
    "query_adapter_policy",
    "run_profile",
    "result_policy",
    "top_k",
    "current_year",
    "max_workers",
    "budgets",
    "diagnostics",
    "llm",
    "prompts",
)
_STRATEGY_FIELDS = (
    "query_planning_policy",
    "query_evolution_policy",
    "judgement_policy",
)
_REPORTED_DIFFERENCES = (
    "runtime_code_hash",
    "sources",
    "ranking_policy",
    "judgement_policy",
    "query_planning_policy",
    "query_evolution_policy",
    "local_bm25",
    "local_hybrid",
)
_RERANKER_ONLY_HYBRID_FIELDS = (
    "connector_version",
    "bm25_corpus_sha256",
    "bm25_document_count",
    "semantic_corpus_sha256",
    "semantic_corpus_size_bytes",
    "semantic_document_count",
    "semantic_abstract_document_count",
    "embedding_dimension",
    "model_fingerprint",
    "index_fingerprint",
    "fusion",
    "query_input",
    "evaluator_data_access",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--strict-reranker-only",
        action="store_true",
        help="require a clean 200-query Hybrid pair differing only by neural reranking",
    )
    parser.add_argument(
        "--allow-strategy-difference",
        action="store_true",
        help=(
            "allow exactly one ranking/planning/judgement policy to differ; "
            "all dataset, query, budget and asset inputs remain shared"
        ),
    )
    return parser


def analyze_paired_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    seed: int = 20260818,
    iterations: int = 5000,
    strict_reranker_only: bool = False,
    allow_strategy_difference: bool = False,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed_must_be_non_negative")
    if iterations <= 0:
        raise ValueError("iterations_must_be_positive")
    baseline = _load_run(baseline_dir)
    candidate = _load_run(candidate_dir)
    baseline_config = baseline["config"]
    candidate_config = candidate["config"]
    shared_fields = tuple(
        field
        for field in _SHARED_FIELDS
        if not (allow_strategy_difference and field in _STRATEGY_FIELDS)
    )
    mismatched = [
        field
        for field in shared_fields
        if baseline_config.get(field) != candidate_config.get(field)
    ]
    if mismatched:
        raise ValueError("shared_config_drift:" + ",".join(mismatched))
    strategy_differences = [
        field
        for field in _STRATEGY_FIELDS
        if baseline_config.get(field) != candidate_config.get(field)
    ]
    if len(strategy_differences) > 1:
        raise ValueError(
            "strategy_config_drift_multiple:" + ",".join(strategy_differences)
        )
    if strategy_differences and not allow_strategy_difference:
        raise ValueError("shared_config_drift:" + ",".join(strategy_differences))
    reranker_audit = None
    if strict_reranker_only:
        reranker_audit = _validate_strict_reranker_only_pair(baseline, candidate)

    baseline_cases = _case_metrics(baseline["metrics"])
    candidate_cases = _case_metrics(candidate["metrics"])
    if list(baseline_cases) != list(candidate_cases):
        raise ValueError("query_order_or_identity_drift")
    differences = {
        metric: [
            candidate_cases[case_id][metric] - baseline_cases[case_id][metric]
            for case_id in baseline_cases
        ]
        for metric in ("f1_at_5", "f1_at_10", "f1_at_20", "recall_at_20")
    }
    comparisons = {
        metric: _bootstrap_delta(values, seed=seed, iterations=iterations)
        for metric, values in differences.items()
    }
    return {
        "schema_version": "paired-benchmark-analysis-v1",
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "query_count": len(baseline_cases),
        "baseline_runtime_code_hash": baseline_config.get("runtime_code_hash"),
        "candidate_runtime_code_hash": candidate_config.get("runtime_code_hash"),
        "shared_inputs_match": True,
        "strategy_difference_allowed": allow_strategy_difference,
        "strategy_differences": strategy_differences,
        "reported_config_differences": {
            field: {
                "baseline": _redact_config_value(
                    baseline_config.get(field)
                ),
                "candidate": _redact_config_value(
                    candidate_config.get(field)
                ),
            }
            for field in _REPORTED_DIFFERENCES
            if baseline_config.get(field) != candidate_config.get(field)
        },
        "comparisons": comparisons,
        "strict_positive_improvement": {
            metric: value["mean_delta"] > 0 and value["ci95"]["low"] > 0
            for metric, value in comparisons.items()
        },
        "strict_reranker_only": strict_reranker_only,
        "reranker_audit": reranker_audit,
        "internal_metric_scope": "not_official_competition_scorer",
    }


def _load_run(path: Path) -> dict[str, Any]:
    run_dir = path.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    run_id = str(run_dir.name)
    if not isinstance(config, dict) or not isinstance(metrics, dict):
        raise ValueError("benchmark_run_json_must_be_objects")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "config": config,
        "metrics": metrics,
    }


def _case_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = metrics.get("per_case")
    if not isinstance(rows, list) or not rows:
        raise ValueError("per_case_metrics_missing")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("per_case_metric_row_invalid")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in result:
            raise ValueError("per_case_metric_identity_invalid")
        values = row.get("metrics")
        if not isinstance(values, dict):
            raise ValueError("per_case_metric_values_missing")
        f1 = values.get("f1_at_k") or {}
        recall = values.get("recall_at_k") or {}
        result[case_id] = {
            "f1_at_5": float(f1.get("5", 0.0)),
            "f1_at_10": float(f1.get("10", 0.0)),
            "f1_at_20": float(f1.get("20", 0.0)),
            "recall_at_20": float(recall.get("20", 0.0)),
        }
    return result


def _validate_strict_reranker_only_pair(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Prove that a 200-query Hybrid pair differs only by neural reranking."""

    baseline_config = baseline["config"]
    candidate_config = candidate["config"]
    for name, run in (("baseline", baseline), ("candidate", candidate)):
        config = run["config"]
        if config.get("case_count") != 200 or config.get("limit") != 200:
            raise ValueError(f"strict_reranker_only_requires_200_queries:{name}")
        if config.get("sources") != ["local_hybrid"]:
            raise ValueError(f"strict_reranker_only_requires_local_hybrid:{name}")
        code = config.get("code")
        if not isinstance(code, dict) or code.get("dirty") is not False:
            raise ValueError(f"strict_reranker_only_requires_clean_code:{name}")
        statistics = run["metrics"].get("case_statistics")
        if not isinstance(statistics, dict) or int(
            statistics.get("total_case_count", 0)
        ) != 200 or int(statistics.get("failed_case_count", -1)) != 0:
            raise ValueError(f"strict_reranker_only_requires_complete_success:{name}")
    if baseline_config.get("runtime_code_hash") != candidate_config.get(
        "runtime_code_hash"
    ) or baseline_config["code"].get("commit") != candidate_config["code"].get("commit"):
        raise ValueError("strict_reranker_only_code_drift")
    top_level_drift = [
        key
        for key in sorted(set(baseline_config) | set(candidate_config))
        if key not in {"started_at", "resume_signature", "local_hybrid"}
        and baseline_config.get(key) != candidate_config.get(key)
    ]
    if top_level_drift:
        raise ValueError(
            "strict_reranker_only_config_drift:" + ",".join(top_level_drift)
        )
    baseline_hybrid = baseline_config.get("local_hybrid")
    candidate_hybrid = candidate_config.get("local_hybrid")
    if not isinstance(baseline_hybrid, dict) or not isinstance(candidate_hybrid, dict):
        raise ValueError("strict_reranker_only_hybrid_config_missing")
    hybrid_drift = [
        key
        for key in _RERANKER_ONLY_HYBRID_FIELDS
        if baseline_hybrid.get(key) != candidate_hybrid.get(key)
    ]
    if hybrid_drift:
        raise ValueError("strict_reranker_only_hybrid_drift:" + ",".join(hybrid_drift))
    if baseline_hybrid.get("reranker_model_path") is not None:
        raise ValueError("strict_reranker_only_baseline_reranker_enabled")
    if not isinstance(candidate_hybrid.get("reranker_model_path"), str) or not candidate_hybrid.get("reranker_model_path"):
        raise ValueError("strict_reranker_only_candidate_reranker_missing")
    if candidate_hybrid.get("reranker_batch_size") != 8 or candidate_hybrid.get("reranker_candidate_limit") != 120:
        raise ValueError("strict_reranker_only_candidate_reranker_limits_invalid")
    from scripts.check_contest_qualification import _audit_reranker_run

    audit = _audit_reranker_run(candidate["run_dir"], expected_rows=200)
    if audit["status"] != "passed":
        raise ValueError("strict_reranker_only_candidate_audit_failed:" + ",".join(audit["reasons"]))
    return audit


def _bootstrap_delta(
    differences: list[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    if not differences:
        raise ValueError("paired_difference_empty")
    rng = random.Random(seed)
    count = len(differences)
    samples = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    ]
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return {
        "mean_delta": sum(differences) / count,
        "ci95": {"low": samples[low_index], "high": samples[high_index]},
        "seed": seed,
        "iterations": iterations,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"benchmark_run_input_unavailable:{path.name}") from exc


def _redact_config_value(value: Any, *, key: str = "") -> Any:
    """Keep comparison context while excluding machine-specific paths."""

    lowered = key.casefold()
    if lowered.endswith("path") or lowered.endswith("dir"):
        return "<redacted-path>" if value is not None else None
    if isinstance(value, dict):
        return {
            str(item_key): _redact_config_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_config_value(item, key=key) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = analyze_paired_runs(
            args.baseline,
            args.candidate,
            seed=args.seed,
            iterations=args.iterations,
            strict_reranker_only=args.strict_reranker_only,
            allow_strategy_difference=args.allow_strategy_difference,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
