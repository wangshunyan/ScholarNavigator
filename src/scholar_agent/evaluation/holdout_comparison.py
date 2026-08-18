"""固定 holdout 上的 Judgement 策略对比与确定性配对 bootstrap。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scholar_agent.evaluation.snapshots import SnapshotStore


HOLDOUT_OFFSET = 20
HOLDOUT_LIMIT = 30
HOLDOUT_CASE_IDS = tuple(
    f"AutoScholarQuery_test_{index}"
    for index in range(HOLDOUT_OFFSET, HOLDOUT_OFFSET + HOLDOUT_LIMIT)
)
BOOTSTRAP_SEED = 20260720
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_METRICS = ("f1_at_20", "recall_at_20", "mrr", "ndcg_at_20")
JUDGEMENT_DROP_REASONS = {
    "judged_weakly_relevant",
    "judged_irrelevant",
    "insufficient_evidence",
    "not_in_return_categories",
}


def analyze_holdout_runs(
    current_run_dir: Path | str,
    calibrated_run_dir: Path | str,
    snapshot_dir: Path | str,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """校验两次 replay 并返回对比、逐查询诊断、错误和覆盖报告。"""

    current = _load_run(current_run_dir)
    calibrated = _load_run(calibrated_run_dir)
    _validate_protocol(current, calibrated)
    current_rows = _rows_by_case(current["results"])
    calibrated_rows = _rows_by_case(calibrated["results"])
    _validate_candidate_identity(current_rows, calibrated_rows)
    _assert_zero_replay_cost(current["metrics"])
    _assert_zero_replay_cost(calibrated["metrics"])

    per_query = _per_query_rows(current, calibrated, current_rows, calibrated_rows)
    bootstrap = paired_bootstrap(
        per_query,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    current_summary = _run_summary(current, current_rows)
    calibrated_summary = _run_summary(calibrated, calibrated_rows)
    comparison = {
        "protocol": {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "case_ids": list(HOLDOUT_CASE_IDS),
            "sources": ["arxiv"],
            "query_planning_policy": "current_rules",
            "query_adapter_policy": "adaptive",
            "query_evolution_policy": "off",
            "refchain": False,
            "llm": False,
            "top_k": 20,
            "result_policy": "highly_and_partial",
            "run_profile": "balanced",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
            "holdout_recalibration_forbidden": True,
        },
        "current_rules": current_summary,
        "calibrated_rules_v1": calibrated_summary,
        "deltas": _summary_deltas(current_summary, calibrated_summary),
        "paired_bootstrap": bootstrap,
        "query_slices": query_slice_analysis(per_query),
        "candidate_snapshot_hash": _candidate_snapshot_hash(current_rows),
        "candidate_recall_identical": (
            current_summary["candidate_recall"]
            == calibrated_summary["candidate_recall"]
        ),
        "replay_zero_network": True,
        "conclusion": _conclusion(current_summary, calibrated_summary, bootstrap),
    }
    error_summary = {
        "current_rules": _error_summary(current, current_rows),
        "calibrated_rules_v1": _error_summary(calibrated, calibrated_rows),
        "stage_loss_deltas": {
            key: (
                _error_summary(calibrated, calibrated_rows)[key]
                - _error_summary(current, current_rows)[key]
            )
            for key in (
                "retrieval_missing_gold_count",
                "judgement_filtered_gold_count",
                "reranking_filtered_gold_count",
            )
        },
    }
    coverage = snapshot_coverage(snapshot_dir)
    return comparison, per_query, error_summary, coverage


def paired_bootstrap(
    per_query: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """按 case 成对重采样，输出 calibrated-current 的均值差与区间。"""

    if not per_query:
        raise ValueError("paired bootstrap requires at least one case")
    if iterations < 100:
        raise ValueError("paired bootstrap requires at least 100 iterations")
    randomizer = random.Random(seed)
    count = len(per_query)
    output: dict[str, Any] = {
        "seed": seed,
        "iterations": iterations,
        "case_count": count,
        "interval": "percentile_95",
        "metrics": {},
        "small_sample_warning": "holdout30_diagnostic_only",
    }
    for metric in BOOTSTRAP_METRICS:
        differences = [
            float(row["calibrated_rules_v1"][metric])
            - float(row["current_rules"][metric])
            for row in per_query
        ]
        sampled = []
        for _ in range(iterations):
            sampled.append(
                sum(differences[randomizer.randrange(count)] for _ in range(count))
                / count
            )
        sampled.sort()
        output["metrics"][metric] = {
            "current_mean": _average(
                row["current_rules"][metric] for row in per_query
            ),
            "calibrated_mean": _average(
                row["calibrated_rules_v1"][metric] for row in per_query
            ),
            "mean_difference": _average(differences),
            "ci_95_low": _percentile(sampled, 0.025),
            "ci_95_high": _percentile(sampled, 0.975),
            "bootstrap_positive_share": (
                sum(value > 0 for value in sampled) / len(sampled)
            ),
        }
    return output


def query_slice_analysis(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """使用查询本身的规则分面与长度标签汇总，不读取 gold 定义分组。"""

    dimensions = (
        "topic_structure",
        "method_presence",
        "dataset_presence",
        "must_have_presence",
        "paper_type_presence",
        "query_length_bin",
    )
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_query:
            groups[str(row["query_features"][dimension])].append(row)
        output[dimension] = {
            label: _slice_summary(rows)
            for label, rows in sorted(groups.items())
        }
    return output


def query_features(result_row: dict[str, Any]) -> dict[str, Any]:
    """从规则 QueryAnalysis/Planning 构造与 gold 无关的稳定分组标签。"""

    result = result_row.get("result") or {}
    analysis = result.get("query_analysis") or {}
    constraints = analysis.get("constraints") or {}
    planning = (result.get("search_plan") or {}).get("query_planning") or {}
    facets = planning.get("facets") or []
    facet_types = sorted(
        {
            str(facet.get("facet_type"))
            for facet in facets
            if facet.get("facet_type") and facet.get("terms")
        }
    )
    word_count = len(re.findall(r"[A-Za-z0-9]+", str(result_row.get("query") or "")))
    return {
        "facet_types": facet_types,
        "topic_structure": (
            "compound_query" if len(facet_types) >= 2 else "single_topic"
        ),
        "method_presence": (
            "has_method" if "method" in facet_types else "no_method"
        ),
        "dataset_presence": (
            "has_dataset" if "dataset" in facet_types else "no_dataset"
        ),
        "must_have_presence": (
            "has_must_have"
            if constraints.get("must_have_terms")
            else "no_must_have"
        ),
        "paper_type_presence": (
            "has_paper_type" if "paper_type" in facet_types else "no_paper_type"
        ),
        "query_word_count": word_count,
        "query_length_bin": (
            "short_1_10"
            if word_count <= 10
            else "medium_11_18"
            if word_count <= 18
            else "long_19_plus"
        ),
    }


def snapshot_coverage(snapshot_dir: Path | str) -> dict[str, Any]:
    root = Path(snapshot_dir).expanduser().resolve()
    store = SnapshotStore(root)
    inspection = store.inspect()
    manifest = store.read_manifest()
    group = inspection.get("groups", {}).get("baseline", {})
    manifest_group = manifest.groups.get("baseline")
    collection_paths = sorted(root.glob("plans/baseline/*/collection_result.json"))
    collections = [
        payload
        for path in collection_paths
        if "collected_entry_count" in (payload := _read_json(path))
    ]
    collection_rounds = [_collection_round_summary(item) for item in collections]
    group_summary = {
        key: group.get(key)
        for key in (
            "completed",
            "collection_completed",
            "plan_rounds",
            "last_plan_round",
            "required_key_count",
            "present_success_entries",
            "present_failed_entries",
            "missing_entries",
            "replay_ready",
            "replay_verified",
            "stop_reason",
        )
    }
    return {
        "snapshot_dir": str(root),
        "snapshot_name": manifest.snapshot_name,
        "dataset": manifest.dataset,
        "offset": manifest.offset,
        "limit": manifest.limit,
        "sources": list(manifest.sources),
        "manifest_sha256": hashlib.sha256(
            store.manifest_path.read_bytes()
        ).hexdigest(),
        "group": group_summary,
        "replay_ready": bool(group.get("replay_ready")),
        "replay_verified": bool(group.get("replay_verified")),
        "required_key_count": int(group.get("required_key_count") or 0),
        "success_key_count": int(group.get("present_success_entries") or 0),
        "failed_key_count": int(group.get("present_failed_entries") or 0),
        "missing_key_count": int(group.get("missing_entries") or 0),
        "judgement_policy_metadata": (
            manifest_group.judgement_policy if manifest_group else None
        ),
        "collection_rounds": collection_rounds,
        "recorded_request_count": sum(
            int(item.get("request_count") or 0) for item in collections
        ),
        "recorded_failed_entry_count": sum(
            int(item.get("failed_entry_count") or 0) for item in collections
        ),
        "recorded_elapsed_seconds": sum(
            float(item.get("elapsed_seconds") or 0.0) for item in collections
        ),
    }


def _collection_round_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """只保留复现采集成本与覆盖状态，避免复制全部快照键。"""

    return {
        key: payload.get(key)
        for key in (
            "round_index",
            "collected_entry_count",
            "skipped_present_count",
            "request_count",
            "failed_entry_count",
            "missing_entries",
            "elapsed_seconds",
            "stop_reason",
            "blocked_sources",
            "source_failure_counts",
        )
    }


def _load_run(path: Path | str) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    return {
        "path": str(root),
        "config": _read_json(root / "config.json"),
        "metrics": _read_json(root / "metrics.json"),
        "stage_metrics": _read_json(root / "stage_metrics.json"),
        "error_analysis": _read_json(root / "error_analysis.json"),
        "results": _read_jsonl(root / "results.jsonl"),
    }


def _validate_protocol(current: dict[str, Any], calibrated: dict[str, Any]) -> None:
    for run, expected_policy in (
        (current, "current_rules"),
        (calibrated, "calibrated_rules_v1"),
    ):
        config = run["config"]
        expected = {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "sources": ["arxiv"],
            "query_planning_policy": "current_rules",
            "query_adapter_policy": "adaptive",
            "query_evolution_policy": "off",
            "enable_refchain": False,
            "top_k": 20,
            "result_policy": "highly_and_partial",
            "run_profile": "balanced",
            "retrieval_mode": "replay",
            "judgement_policy": expected_policy,
        }
        mismatched = [key for key, value in expected.items() if config.get(key) != value]
        if mismatched:
            raise ValueError("holdout protocol mismatch:" + ",".join(mismatched))
        if tuple(config.get("case_ids") or ()) != HOLDOUT_CASE_IDS:
            raise ValueError("holdout case ids are not fixed offset=20 limit=30")
    comparable = (
        "dataset_sha256",
        "case_ids",
        "sources",
        "run_profile",
        "top_k",
        "budgets",
        "query_adapter_policy",
        "query_planning_policy",
        "query_planner_version",
        "result_policy",
        "snapshot",
    )
    differences = [
        key
        for key in comparable
        if current["config"].get(key) != calibrated["config"].get(key)
    ]
    if differences:
        raise ValueError("holdout runs are incompatible:" + ",".join(differences))


def _validate_candidate_identity(
    current: dict[str, dict[str, Any]],
    calibrated: dict[str, dict[str, Any]],
) -> None:
    if tuple(current) != HOLDOUT_CASE_IDS or tuple(calibrated) != HOLDOUT_CASE_IDS:
        raise ValueError("holdout result order changed")
    for case_id in HOLDOUT_CASE_IDS:
        left = _candidate_fingerprints(current[case_id])
        right = _candidate_fingerprints(calibrated[case_id])
        if left != right:
            raise ValueError(f"retrieval candidates differ between policies:{case_id}")


def _candidate_fingerprints(row: dict[str, Any]) -> list[str]:
    diagnostics = row.get("stage_diagnostics") or {}
    snapshots = diagnostics.get("snapshots") or []
    stage = next(
        (item for item in snapshots if item.get("stage") == "initial_deduplicated"),
        {},
    )
    return [
        _stable_hash(
            {
                "identifiers": item.get("identifiers") or {},
                "title": item.get("title"),
                "year": item.get("year"),
            }
        )
        for item in stage.get("candidates") or []
    ]


def _candidate_snapshot_hash(rows: dict[str, dict[str, Any]]) -> str:
    return _stable_hash(
        {
            case_id: _candidate_fingerprints(row)
            for case_id, row in rows.items()
        }
    )


def _assert_zero_replay_cost(metrics: dict[str, Any]) -> None:
    costs = metrics.get("snapshot_costs") or {}
    for key in (
        "replay_execution_request_count",
        "replay_execution_retry_count",
        "replay_execution_network_wait_seconds",
    ):
        if float(costs.get(key) or 0.0) != 0.0:
            raise ValueError(f"holdout replay executed network work:{key}")


def _per_query_rows(
    current: dict[str, Any],
    calibrated: dict[str, Any],
    current_rows: dict[str, dict[str, Any]],
    calibrated_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    current_metrics = {
        row["case_id"]: row for row in current["metrics"].get("per_case") or []
    }
    calibrated_metrics = {
        row["case_id"]: row
        for row in calibrated["metrics"].get("per_case") or []
    }
    output = []
    for case_id in HOLDOUT_CASE_IDS:
        current_row = current_rows[case_id]
        calibrated_row = calibrated_rows[case_id]
        current_summary = _case_summary(current_row, current_metrics[case_id])
        calibrated_summary = _case_summary(
            calibrated_row,
            calibrated_metrics[case_id],
        )
        output.append(
            {
                "case_id": case_id,
                "query": current_row.get("query"),
                "query_features": query_features(current_row),
                "candidate_hash": _stable_hash(_candidate_fingerprints(current_row)),
                "current_rules": current_summary,
                "calibrated_rules_v1": calibrated_summary,
                "deltas": {
                    metric: calibrated_summary[metric] - current_summary[metric]
                    for metric in BOOTSTRAP_METRICS
                },
            }
        )
    return output


def _case_summary(row: dict[str, Any], metric_row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("stage_diagnostics") or {}
    gold = diagnostics.get("gold_diagnostics") or []
    judgement = diagnostics.get("judgement") or {}
    reranking = diagnostics.get("reranking") or {}
    returned = _returned_count(row)
    returned_gold = sum(item.get("drop_reason") == "returned" for item in gold)
    nested = metric_row.get("metrics") or {}
    return {
        "candidate_recall": float(
            ((diagnostics.get("stage_metrics") or {}).get("candidate_recall") or {}).get(
                "initial_retrieval"
            )
            or 0.0
        ),
        "f1_at_5": _metric(nested, "f1_at_k", 5),
        "f1_at_10": _metric(nested, "f1_at_k", 10),
        "f1_at_20": _metric(nested, "f1_at_k", 20),
        "precision_at_5": _metric(nested, "precision_at_k", 5),
        "precision_at_10": _metric(nested, "precision_at_k", 10),
        "precision_at_20": _metric(nested, "precision_at_k", 20),
        "recall_at_5": _metric(nested, "recall_at_k", 5),
        "recall_at_10": _metric(nested, "recall_at_k", 10),
        "recall_at_20": _metric(nested, "recall_at_k", 20),
        "mrr": float(nested.get("mrr") or 0.0),
        "ndcg_at_20": _metric(nested, "ndcg_at_k", 20),
        "gold_count": len(gold),
        "gold_retrieved_count": int(judgement.get("retrieved_gold_count") or 0),
        "gold_retained_count": int(
            judgement.get("gold_judged_highly_relevant") or 0
        )
        + int(judgement.get("gold_judged_partially_relevant") or 0),
        "gold_judgement_filtered_count": int(
            judgement.get("gold_false_negative_count") or 0
        ),
        "gold_reranking_filtered_count": int(
            reranking.get("gold_outside_top_20") or 0
        ),
        "returned_count": returned,
        "benchmark_non_gold_returned_count": max(0, returned - returned_gold),
        "category_distribution": _case_category_distribution(row),
        "drop_reasons": dict(
            sorted(Counter(str(item.get("drop_reason")) for item in gold).items())
        ),
    }


def _run_summary(run: dict[str, Any], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = run["metrics"]
    ranking = metrics.get("end_to_end_metrics") or {}
    stage = run["stage_metrics"]
    judgement = stage.get("judgement") or {}
    reranking = stage.get("reranking") or {}
    costs = metrics.get("snapshot_costs") or {}
    total_returned = sum(_returned_count(row) for row in rows.values())
    returned_gold = sum(
        item.get("drop_reason") == "returned"
        for row in rows.values()
        for item in (row.get("stage_diagnostics") or {}).get("gold_diagnostics") or []
    )
    return {
        "judgement_policy": run["config"].get("judgement_policy"),
        "judgement_config_hash": run["config"].get("judgement_config_hash"),
        "candidate_recall": float(stage.get("initial_retrieval_recall") or 0.0),
        "f1_at_5": _metric(ranking, "f1_at_k", 5),
        "f1_at_10": _metric(ranking, "f1_at_k", 10),
        "f1_at_20": _metric(ranking, "f1_at_k", 20),
        "precision_at_5": _metric(ranking, "precision_at_k", 5),
        "precision_at_10": _metric(ranking, "precision_at_k", 10),
        "precision_at_20": _metric(ranking, "precision_at_k", 20),
        "recall_at_5": _metric(ranking, "recall_at_k", 5),
        "recall_at_10": _metric(ranking, "recall_at_k", 10),
        "recall_at_20": _metric(ranking, "recall_at_k", 20),
        "mrr": float(ranking.get("mrr") or 0.0),
        "ndcg_at_20": _metric(ranking, "ndcg_at_k", 20),
        "gold_count": int(stage.get("gold_count") or 0),
        "gold_retrieved_count": int(judgement.get("retrieved_gold_count") or 0),
        "gold_retained_count": int(
            judgement.get("gold_judged_highly_relevant") or 0
        )
        + int(judgement.get("gold_judged_partially_relevant") or 0),
        "gold_filtered_by_judgement_count": int(
            judgement.get("gold_false_negative_count") or 0
        ),
        "gold_judgement_false_negative_rate": float(
            judgement.get("gold_false_negative_rate") or 0.0
        ),
        "gold_filtered_by_reranking_count": int(
            reranking.get("gold_outside_top_20") or 0
        ),
        "average_returned_paper_count": total_returned / HOLDOUT_LIMIT,
        "benchmark_non_gold_returned_count": total_returned - returned_gold,
        "category_distribution": _aggregate_category_distribution(rows),
        "source_error_rate": float(
            (stage.get("source_contribution") or {}).get("source_error_rate") or 0.0
        ),
        "recorded_cost": {
            "search_request_count": float(
                costs.get("recorded_search_request_count") or 0.0
            ),
            "retry_count": float(costs.get("recorded_retry_count") or 0.0),
            "error_count": float(costs.get("recorded_error_count") or 0.0),
            "rate_limit_wait_seconds": float(
                costs.get("recorded_rate_limit_wait_seconds") or 0.0
            ),
            "latency_seconds": float(costs.get("recorded_latency_seconds") or 0.0),
        },
        "replay_execution_cost": {
            "snapshot_hits": float(costs.get("retrieval_snapshot_hits") or 0.0),
            "http_requests": float(
                costs.get("replay_execution_request_count") or 0.0
            ),
            "retries": float(costs.get("replay_execution_retry_count") or 0.0),
            "network_wait_seconds": float(
                costs.get("replay_execution_network_wait_seconds") or 0.0
            ),
        },
    }


def _summary_deltas(current: dict[str, Any], calibrated: dict[str, Any]) -> dict[str, float]:
    fields = (
        "candidate_recall",
        "f1_at_5",
        "f1_at_10",
        "f1_at_20",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr",
        "ndcg_at_20",
        "gold_judgement_false_negative_rate",
        "average_returned_paper_count",
    )
    return {
        field: float(calibrated[field]) - float(current[field])
        for field in fields
    }


def _slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("candidate_recall", *BOOTSTRAP_METRICS)
    current = {
        metric: _average(row["current_rules"][metric] for row in rows)
        for metric in metrics
    }
    calibrated = {
        metric: _average(row["calibrated_rules_v1"][metric] for row in rows)
        for metric in metrics
    }
    return {
        "case_count": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "current_rules": current,
        "calibrated_rules_v1": calibrated,
        "deltas": {
            metric: calibrated[metric] - current[metric] for metric in metrics
        },
    }


def _error_summary(run: dict[str, Any], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    drop_reasons = Counter(
        str(item.get("drop_reason"))
        for row in rows.values()
        for item in (row.get("stage_diagnostics") or {}).get("gold_diagnostics") or []
    )
    return {
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "retrieval_missing_gold_count": sum(
            count
            for reason, count in drop_reasons.items()
            if reason
            in {
                "not_retrieved",
                "source_failed",
                "identifier_not_matched",
                "budget_stopped_before_retrieval",
            }
        ),
        "judgement_filtered_gold_count": sum(
            drop_reasons[reason] for reason in JUDGEMENT_DROP_REASONS
        ),
        "reranking_filtered_gold_count": drop_reasons["outside_final_top_k"],
        "returned_gold_count": drop_reasons["returned"],
        "bottleneck_labels": list(run["stage_metrics"].get("bottleneck_labels") or []),
        "source_error_rate": float(
            (run["stage_metrics"].get("source_contribution") or {}).get(
                "source_error_rate"
            )
            or 0.0
        ),
    }


def _conclusion(
    current: dict[str, Any],
    calibrated: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    advantages = [
        calibrated["f1_at_20"] > current["f1_at_20"],
        calibrated["recall_at_20"] > current["recall_at_20"],
        calibrated["gold_judgement_false_negative_rate"]
        < current["gold_judgement_false_negative_rate"],
        any(
            bootstrap["metrics"][metric]["ci_95_low"] > 0
            for metric in BOOTSTRAP_METRICS
        ),
    ]
    retrieval_ratio = (
        current["gold_retrieved_count"] / current["gold_count"]
        if current["gold_count"]
        else 0.0
    )
    return {
        "calibrated_has_stable_advantage": all(advantages),
        "product_default": "current_rules",
        "retrieval_recall_bottleneck": retrieval_ratio < 0.5,
        "retrieved_gold_share": retrieval_ratio,
        "evidence_scope": "holdout30_diagnostic_only",
        "recalibration_performed": False,
    }


def _returned_count(row: dict[str, Any]) -> int:
    result = row.get("result") or {}
    return len(result.get("highly_relevant_papers") or []) + len(
        result.get("partially_relevant_papers") or []
    )


def _case_category_distribution(row: dict[str, Any]) -> dict[str, int]:
    diagnostics = row.get("stage_diagnostics") or {}
    snapshot = next(
        (
            item
            for item in diagnostics.get("snapshots") or []
            if item.get("stage") == "initial_judged"
        ),
        {},
    )
    return dict(
        sorted(
            Counter(
                str(item.get("category"))
                for item in snapshot.get("candidates") or []
                if item.get("category")
            ).items()
        )
    )


def _aggregate_category_distribution(
    rows: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows.values():
        counts.update(_case_category_distribution(row))
    return dict(sorted(counts.items()))


def _rows_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("case_id")): row for row in rows}
    return {case_id: indexed[case_id] for case_id in HOLDOUT_CASE_IDS}


def _metric(metrics: dict[str, Any], name: str, k: int) -> float:
    values = metrics.get(name) or {}
    return float(values.get(str(k), values.get(k, 0.0)) or 0.0)


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object:{path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
