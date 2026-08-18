#!/usr/bin/env python3
"""比较 arXiv、OpenAlex 与双源检索的冻结回放结果。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.analyze_query_planning_policies import (  # noqa: E402
    _at_k,
    _load_run,
    _write_json,
    _write_jsonl,
)
from scholar_agent.core.dedup import normalize_title  # noqa: E402


GROUP_SOURCES = {
    "arxiv_only": ["arxiv"],
    "openalex_only": ["openalex"],
    "arxiv_openalex": ["arxiv", "openalex"],
}
SPLIT_OFFSETS = {"development": 90, "validation": 110}
COMPARABLE_FIELDS = (
    "dataset",
    "dataset_sha256",
    "case_ids",
    "offset",
    "limit",
    "query_adapter_policy",
    "query_planning_policy",
    "query_planner_version",
    "judgement_policy",
    "judgement_config_hash",
    "run_profile",
    "result_policy",
    "top_k",
    "current_year",
    "max_workers",
    "budgets",
    "diagnostics",
    "llm",
    "enable_query_evolution",
    "query_evolution_policy",
    "enable_refchain",
    "runtime_code_hash",
)
RETRIEVAL_DROP_REASONS = {
    "not_retrieved",
    "source_failed",
    "identifier_not_matched",
    "budget_stopped_before_retrieval",
}
JUDGEMENT_DROP_REASONS = {
    "judged_weakly_relevant",
    "judged_irrelevant",
    "insufficient_evidence",
    "not_in_return_categories",
}
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_runs/source_complementarity")


def build_source_complementarity_analysis(
    *,
    development_arxiv: Path,
    development_openalex: Path,
    development_combined: Path,
    validation_arxiv: Path,
    validation_openalex: Path,
    validation_combined: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    runs = {
        "development": {
            "arxiv_only": _load_run(development_arxiv),
            "openalex_only": _load_run(development_openalex),
            "arxiv_openalex": _load_run(development_combined),
        },
        "validation": {
            "arxiv_only": _load_run(validation_arxiv),
            "openalex_only": _load_run(validation_openalex),
            "arxiv_openalex": _load_run(validation_combined),
        },
    }
    for split, split_runs in runs.items():
        _validate_split(split, split_runs)

    comparisons = {
        split: _split_comparison(split, split_runs)
        for split, split_runs in runs.items()
    }
    gold_rows = [
        row
        for split, split_runs in runs.items()
        for row in _source_gold_rows(split, split_runs)
    ]
    error_analysis = {
        split: {
            group: {
                "stage_losses": summary["stage_losses"],
                "source_recorded_costs": summary["source_recorded_costs"],
                "source_error_rate": summary["source_error_rate"],
            }
            for group, summary in comparison["groups"].items()
        }
        for split, comparison in comparisons.items()
    }
    acceptance = _validation_acceptance(comparisons["validation"])
    result = {
        "protocol": {
            "dataset": "auto_scholar_query",
            "development": {"offset": 90, "limit": 20},
            "validation": {"offset": 110, "limit": 20},
            "query_planning_policy": "current_rules",
            "query_adapter_policy": "adaptive",
            "judgement_policy": "current_rules",
            "query_evolution_policy": "off",
            "refchain": False,
            "llm": False,
            "run_profile": "balanced",
            "top_k": 20,
            "result_policy": "highly_and_partial",
            "validation_policy": "single_frozen_evaluation",
        },
        "development": comparisons["development"],
        "validation": comparisons["validation"],
        "validation_acceptance": acceptance,
        "product_default_sources_changed": False,
        "limitations": [
            "固定 20+20 小样本只用于来源互补性诊断，不代表完整 Benchmark。",
            "gold 仅在 SearchService 完成后参与匹配和来源归因。",
            "通过门槛也只形成 high_recall profile 候选，不直接修改默认来源。",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "development_comparison.json", comparisons["development"])
    _write_json(output_dir / "validation_comparison.json", comparisons["validation"])
    _write_jsonl(output_dir / "source_gold_contribution.jsonl", gold_rows)
    _write_json(output_dir / "error_analysis.json", error_analysis)
    (output_dir / "summary.md").write_text(
        _summary_markdown(result),
        encoding="utf-8",
    )
    return result


def _validate_split(split: str, runs: dict[str, dict[str, Any]]) -> None:
    if set(runs) != set(GROUP_SOURCES):
        raise ValueError(f"{split} requires all three source groups")
    baseline = runs["arxiv_only"]["config"]
    for group, run in runs.items():
        config = run["config"]
        mismatched = [
            field
            for field in COMPARABLE_FIELDS
            if config.get(field) != baseline.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"incompatible {split} source runs ({group}): " + ",".join(mismatched)
            )
        if config.get("sources") != GROUP_SOURCES[group]:
            raise ValueError(f"unexpected sources for {group}: {config.get('sources')}")
        if config.get("offset") != SPLIT_OFFSETS[split] or config.get("limit") != 20:
            raise ValueError(f"unexpected fixed {split} split")
        if len(config.get("case_ids") or []) != 20:
            raise ValueError(f"{split} source comparison requires exactly 20 cases")
        if config.get("dataset") != "auto_scholar_query":
            raise ValueError("source comparison requires auto_scholar_query")
        if config.get("query_planning_policy") != "current_rules":
            raise ValueError("source comparison requires current_rules planning")
        if config.get("query_adapter_policy") != "adaptive":
            raise ValueError("source comparison requires adaptive adapter")
        if config.get("judgement_policy") != "current_rules":
            raise ValueError("source comparison requires current_rules judgement")
        if (
            config.get("enable_query_evolution")
            or config.get("query_evolution_policy") != "off"
            or config.get("enable_refchain")
        ):
            raise ValueError("source comparison requires later stages off")
        llm = config.get("llm") or {}
        if llm.get("query_understanding") or llm.get("judgement"):
            raise ValueError("source comparison requires LLM off")
        if config.get("run_profile") != "balanced" or config.get("top_k") != 20:
            raise ValueError("source comparison requires balanced top20")
        if config.get("result_policy") != "highly_and_partial":
            raise ValueError("source comparison result policy mismatch")
        if config.get("retrieval_mode") != "replay":
            raise ValueError("final source comparison requires replay runs")
        _assert_replay_zero_network(run["metrics"])


def _split_comparison(split: str, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {
        group: _run_summary(run, split=split, group=group)
        for group, run in runs.items()
    }
    gold_rows = _source_gold_rows(split, runs)
    contribution = Counter(row["source_classification"] for row in gold_rows)
    labels = Counter(
        label for row in gold_rows for label in row["diagnostic_labels"]
    )
    arxiv_gold = {
        row["gold_key"] for row in gold_rows if row["found_by_arxiv_only"]
    }
    openalex_gold = {
        row["gold_key"] for row in gold_rows if row["found_by_openalex_only"]
    }
    return {
        "split": split,
        "groups": summaries,
        "source_gold_contribution": {
            "arxiv_exclusive_gold_count": len(arxiv_gold - openalex_gold),
            "openalex_exclusive_gold_count": len(openalex_gold - arxiv_gold),
            "overlap_gold_count": len(arxiv_gold & openalex_gold),
            "not_found_by_either_count": sum(
                not row["found_by_arxiv_only"]
                and not row["found_by_openalex_only"]
                for row in gold_rows
            ),
            "classification_counts": dict(sorted(contribution.items())),
            "diagnostic_label_counts": dict(sorted(labels.items())),
        },
    }


def _run_summary(
    run: dict[str, Any],
    *,
    split: str,
    group: str,
) -> dict[str, Any]:
    metrics = run["metrics"]
    end_to_end = metrics.get("end_to_end_metrics") or metrics.get("aggregate") or {}
    stage = run["stage_metrics"]
    planning = stage.get("initial_query_planning") or {}
    snapshot_costs = metrics.get("snapshot_costs") or {}
    gold_count = int(stage.get("gold_count") or 0)
    judgement = stage.get("judgement") or {}
    source_contribution = stage.get("source_contribution") or {}
    return {
        "split": split,
        "group": group,
        "run_dir": run["run_dir"],
        "case_count": int(stage.get("case_count") or metrics.get("case_count") or 0),
        "gold_count": gold_count,
        "candidate_recall": stage.get("initial_retrieval_recall"),
        "f1_at_5": _at_k(end_to_end, "f1_at_k", 5),
        "f1_at_10": _at_k(end_to_end, "f1_at_k", 10),
        "f1_at_20": _at_k(end_to_end, "f1_at_k", 20),
        "precision_at_20": _at_k(end_to_end, "precision_at_k", 20),
        "recall_at_20": _at_k(end_to_end, "recall_at_k", 20),
        "mrr": float(end_to_end.get("mrr") or 0.0),
        "ndcg_at_20": _at_k(end_to_end, "ndcg_at_k", 20),
        "unique_candidate_count": int(planning.get("unique_candidate_count") or 0),
        "unique_gold_count": int(judgement.get("retrieved_gold_count") or 0),
        "deduplication": _deduplication_summary(run["results"]),
        "openalex_identifier_completeness": _openalex_identifier_completeness(
            run["results"]
        ),
        "source_contribution": source_contribution,
        "source_error_rate": float(source_contribution.get("source_error_rate") or 0.0),
        "source_recorded_costs": _snapshot_source_costs(run["config"]),
        "stage_losses": _stage_losses(run["results"]),
        "recorded_live_cost": {
            "search_request_count": float(
                snapshot_costs.get("recorded_search_request_count") or 0.0
            ),
            "retry_count": float(snapshot_costs.get("recorded_retry_count") or 0.0),
            "error_count": float(snapshot_costs.get("recorded_error_count") or 0.0),
            "rate_limit_wait_seconds": float(
                snapshot_costs.get("recorded_rate_limit_wait_seconds") or 0.0
            ),
            "latency_seconds": float(
                snapshot_costs.get("recorded_latency_seconds") or 0.0
            ),
        },
        "replay_execution_cost": {
            "snapshot_hits": float(
                snapshot_costs.get("retrieval_snapshot_hits") or 0.0
            ),
            "http_requests": float(
                snapshot_costs.get("replay_execution_request_count") or 0.0
            ),
            "retries": float(
                snapshot_costs.get("replay_execution_retry_count") or 0.0
            ),
            "network_wait_seconds": float(
                snapshot_costs.get("replay_execution_network_wait_seconds") or 0.0
            ),
        },
    }


def _source_gold_rows(
    split: str,
    runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = {
        group: {str(row["case_id"]): row for row in run["results"]}
        for group, run in runs.items()
    }
    case_ids = list(runs["arxiv_only"]["config"].get("case_ids") or [])
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        grouped = {group: values[case_id] for group, values in results.items()}
        gold_lists = {
            group: (row.get("stage_diagnostics") or {}).get("gold_diagnostics") or []
            for group, row in grouped.items()
        }
        counts = {group: len(values) for group, values in gold_lists.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"gold diagnostics mismatch for {case_id}: {counts}")
        for index in range(counts["arxiv_only"]):
            gold = {group: values[index] for group, values in gold_lists.items()}
            identity = {
                (item.get("gold_id"), item.get("gold_title")) for item in gold.values()
            }
            if len(identity) != 1:
                raise ValueError(f"gold ordering mismatch for {case_id}:{index}")
            arxiv_found = bool(gold["arxiv_only"].get("found"))
            openalex_found = bool(gold["openalex_only"].get("found"))
            combined = gold["arxiv_openalex"]
            mismatch = (
                gold["openalex_only"].get("drop_reason") == "identifier_not_matched"
                or _title_seen(
                    grouped["openalex_only"],
                    str(gold["openalex_only"].get("gold_title") or ""),
                )
            ) and not openalex_found
            source_failed = any(
                item.get("drop_reason") == "source_failed" for item in gold.values()
            )
            classification = _source_classification(
                arxiv_found=arxiv_found,
                openalex_found=openalex_found,
                identifier_mismatch=mismatch,
                source_failed=source_failed,
            )
            pipeline = _pipeline_classification(str(combined.get("drop_reason") or ""))
            labels = [classification]
            if mismatch and "openalex_identifier_mismatch" not in labels:
                labels.append("openalex_identifier_mismatch")
            if source_failed and "source_failure" not in labels:
                labels.append("source_failure")
            if pipeline == "ranked_below_limit":
                labels.append("ranked_below_limit")
            gold_id = str(combined.get("gold_id") or "")
            gold_title = str(combined.get("gold_title") or "")
            rows.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "query": grouped["arxiv_openalex"].get("query"),
                    "gold_index": index,
                    "gold_key": gold_id or f"title:{normalize_title(gold_title)}",
                    "gold_id": gold_id or None,
                    "gold_title": gold_title or None,
                    "found_by_arxiv_only": arxiv_found,
                    "found_by_openalex_only": openalex_found,
                    "found_by_combined": bool(combined.get("found")),
                    "source_classification": classification,
                    "pipeline_classification": pipeline,
                    "openalex_identifier_mismatch": mismatch,
                    "source_failure": source_failed,
                    "diagnostic_labels": labels,
                    "arxiv_drop_reason": gold["arxiv_only"].get("drop_reason"),
                    "openalex_drop_reason": gold["openalex_only"].get("drop_reason"),
                    "combined_drop_reason": combined.get("drop_reason"),
                    "combined_final_rank": combined.get("final_rank"),
                }
            )
    return rows


def _source_classification(
    *,
    arxiv_found: bool,
    openalex_found: bool,
    identifier_mismatch: bool,
    source_failed: bool,
) -> str:
    if arxiv_found and openalex_found:
        return "found_by_both"
    if arxiv_found:
        return "found_by_arxiv_only"
    if openalex_found:
        return "found_by_openalex_only"
    if identifier_mismatch:
        return "openalex_identifier_mismatch"
    if source_failed:
        return "source_failure"
    return "not_found_by_either"


def _pipeline_classification(drop_reason: str) -> str:
    if drop_reason == "returned":
        return "returned"
    if drop_reason == "outside_final_top_k":
        return "ranked_below_limit"
    if drop_reason in RETRIEVAL_DROP_REASONS:
        return "retrieval_missing"
    if drop_reason in JUDGEMENT_DROP_REASONS:
        return "judgement_filtered"
    if drop_reason == "removed_or_merged_by_dedup":
        return "deduplicated"
    return "other"


def _stage_losses(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        str(gold.get("drop_reason") or "unknown")
        for row in results
        for gold in (row.get("stage_diagnostics") or {}).get("gold_diagnostics") or []
    )
    return {
        "retrieval_missing_gold_count": sum(reasons[reason] for reason in RETRIEVAL_DROP_REASONS),
        "judgement_filtered_gold_count": sum(reasons[reason] for reason in JUDGEMENT_DROP_REASONS),
        "top_20_filtered_gold_count": reasons["outside_final_top_k"],
        "returned_gold_count": reasons["returned"],
        "drop_reasons": dict(sorted(reasons.items())),
    }


def _deduplication_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    raw = deduplicated = 0
    for row in results:
        snapshots = {
            str(item.get("stage")): item
            for item in (row.get("stage_diagnostics") or {}).get("snapshots") or []
        }
        raw += len((snapshots.get("initial_retrieval") or {}).get("candidates") or [])
        deduplicated += len(
            (snapshots.get("initial_deduplicated") or {}).get("candidates") or []
        )
    removed = max(0, raw - deduplicated)
    return {
        "raw_candidate_count": raw,
        "deduplicated_candidate_count": deduplicated,
        "removed_duplicate_count": removed,
        "deduplication_rate": removed / raw if raw else 0.0,
    }


def _openalex_identifier_completeness(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in results:
        snapshot = next(
            (
                item
                for item in (row.get("stage_diagnostics") or {}).get("snapshots") or []
                if item.get("stage") == "initial_retrieval"
            ),
            None,
        )
        for candidate in (snapshot or {}).get("candidates") or []:
            if "openalex" not in {
                str(source).casefold() for source in candidate.get("sources") or []
            }:
                continue
            identifiers = candidate.get("identifiers") or {}
            key = str(
                identifiers.get("openalex_id")
                or identifiers.get("doi")
                or identifiers.get("arxiv_id")
                or f"{normalize_title(str(candidate.get('title') or ''))}:{candidate.get('year')}"
            )
            candidates.setdefault(key.casefold(), candidate)
    total = len(candidates)

    def count(field: str) -> int:
        return sum(bool((row.get("identifiers") or {}).get(field)) for row in candidates.values())

    any_stable = sum(
        any(bool(value) for value in (row.get("identifiers") or {}).values())
        for row in candidates.values()
    )
    return {
        "unique_openalex_candidate_count": total,
        "any_stable_identifier_count": any_stable,
        "any_stable_identifier_rate": any_stable / total if total else None,
        "openalex_id_count": count("openalex_id"),
        "openalex_id_rate": count("openalex_id") / total if total else None,
        "doi_count": count("doi"),
        "doi_rate": count("doi") / total if total else None,
        "arxiv_id_count": count("arxiv_id"),
        "arxiv_id_rate": count("arxiv_id") / total if total else None,
    }


def _title_seen(result: dict[str, Any], gold_title: str) -> bool:
    normalized = normalize_title(gold_title)
    if not normalized:
        return False
    for snapshot in (result.get("stage_diagnostics") or {}).get("snapshots") or []:
        if snapshot.get("stage") != "initial_retrieval":
            continue
        if any(
            normalize_title(str(candidate.get("title") or "")) == normalized
            for candidate in snapshot.get("candidates") or []
        ):
            return True
    return False


def _snapshot_source_costs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot = config.get("snapshot") or {}
    directory = Path(str(snapshot.get("directory") or ""))
    group = str(snapshot.get("group") or "")
    if not directory.is_dir() or not group:
        raise ValueError("run config is missing a readable snapshot directory/group")
    manifest = _read_json(directory / "manifest.json")
    coverage = (manifest.get("groups") or {}).get(group) or {}
    if (
        not coverage.get("replay_ready")
        or int(coverage.get("missing_key_count") or 0) != 0
    ):
        raise ValueError(f"snapshot group is not replay-ready: {directory}:{group}")
    totals: dict[str, Counter[str]] = {}
    status_counts: dict[str, Counter[str]] = {}
    for key in coverage.get("retrieval_keys") or []:
        entry = _read_json(directory / "retrieval" / f"{key}.json")
        source = str(entry.get("source") or "unknown")
        diagnostics = entry.get("diagnostics") or {}
        bucket = totals.setdefault(source, Counter())
        bucket["snapshot_entry_count"] += 1
        bucket["request_count"] += int(diagnostics.get("request_count") or 0)
        bucket["retry_count"] += int(diagnostics.get("retry_count") or 0)
        bucket["error_count"] += int(diagnostics.get("error_count") or 0)
        bucket["latency_milliseconds"] += round(
            float(entry.get("recorded_latency_seconds") or 0.0) * 1000
        )
        bucket["rate_limit_wait_milliseconds"] += round(
            float(diagnostics.get("rate_limit_wait_seconds") or 0.0) * 1000
        )
        status_counts.setdefault(source, Counter())[str(entry.get("status") or "unknown")] += 1
    output: dict[str, dict[str, Any]] = {}
    for source, values in sorted(totals.items()):
        requests = int(values["request_count"])
        output[source] = {
            "snapshot_entry_count": int(values["snapshot_entry_count"]),
            "success_entry_count": int(status_counts[source]["success"]),
            "failed_entry_count": int(status_counts[source]["failed"]),
            "request_count": requests,
            "retry_count": int(values["retry_count"]),
            "error_count": int(values["error_count"]),
            "error_rate": int(values["error_count"]) / requests if requests else 0.0,
            "recorded_latency_seconds": values["latency_milliseconds"] / 1000,
            "rate_limit_wait_seconds": values["rate_limit_wait_milliseconds"] / 1000,
        }
    return output


def _validation_acceptance(validation: dict[str, Any]) -> dict[str, Any]:
    arxiv = validation["groups"]["arxiv_only"]
    combined = validation["groups"]["arxiv_openalex"]
    contribution = validation["source_gold_contribution"]
    arxiv_api = float(arxiv["recorded_live_cost"]["search_request_count"])
    combined_api = float(combined["recorded_live_cost"]["search_request_count"])
    checks = {
        "at_least_one_new_unique_gold": (
            int(combined["unique_gold_count"]) >= int(arxiv["unique_gold_count"]) + 1
        ),
        "openalex_has_exclusive_gold": int(
            contribution["openalex_exclusive_gold_count"]
        ) >= 1,
        "recall_at_20_non_regression": float(combined["recall_at_20"])
        >= float(arxiv["recall_at_20"]),
        "f1_at_20_non_regression": float(combined["f1_at_20"])
        >= float(arxiv["f1_at_20"]),
        "api_calls_within_2x": (
            combined_api == 0.0 if arxiv_api == 0.0 else combined_api <= arxiv_api * 2
        ),
        "source_error_rate_controlled": float(combined["source_error_rate"]) < 0.2,
        "frozen_replay_zero_network": all(
            float(summary["replay_execution_cost"][field]) == 0.0
            for summary in validation["groups"].values()
            for field in ("http_requests", "retries", "network_wait_seconds")
        ),
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "checks": checks,
        "unique_gold_gain_vs_arxiv": int(combined["unique_gold_count"])
        - int(arxiv["unique_gold_count"]),
        "recorded_api_ratio_vs_arxiv": (
            combined_api / arxiv_api if arxiv_api else None
        ),
        "high_recall_profile_candidate": accepted,
        "decision": (
            "标记 arxiv+openalex 为 high_recall profile 候选"
            if accepted
            else "不形成 high_recall profile 候选，保持默认来源策略"
        ),
    }


def _assert_replay_zero_network(metrics: dict[str, Any]) -> None:
    costs = metrics.get("snapshot_costs") or {}
    nonzero = {
        field: float(costs.get(field) or 0.0)
        for field in (
            "replay_execution_request_count",
            "replay_execution_retry_count",
            "replay_execution_network_wait_seconds",
        )
        if float(costs.get(field) or 0.0) != 0.0
    }
    if nonzero:
        raise ValueError(f"replay used network: {nonzero}")


def _summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 来源互补性对比",
        "",
        "> 开发集冻结分析规则后，验证集只评估一次；gold 不进入生产检索。",
        "",
        "| 切片 | 分组 | 候选 Recall | F1@5 | F1@10 | F1@20 | P@20 | R@20 | MRR | nDCG@20 | 候选 | gold | 记录 API | 错误率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "validation"):
        for group in GROUP_SOURCES:
            row = result[split]["groups"][group]
            values = [
                split,
                group,
                _fmt(row["candidate_recall"]),
                _fmt(row["f1_at_5"]),
                _fmt(row["f1_at_10"]),
                _fmt(row["f1_at_20"]),
                _fmt(row["precision_at_20"]),
                _fmt(row["recall_at_20"]),
                _fmt(row["mrr"]),
                _fmt(row["ndcg_at_20"]),
                str(row["unique_candidate_count"]),
                str(row["unique_gold_count"]),
                _fmt(row["recorded_live_cost"]["search_request_count"]),
                _fmt(row["source_error_rate"]),
            ]
            lines.append("| " + " | ".join(values) + " |")
    acceptance = result["validation_acceptance"]
    lines.extend(
        [
            "",
            "## 验收",
            "",
            f"- 结论：{acceptance['decision']}。",
            f"- 双源相对 arXiv unique gold 增量：{acceptance['unique_gold_gain_vs_arxiv']}。",
            "- 产品默认来源未修改。",
            "- 所有最终 Replay 的 HTTP、重试和网络等待均为 0。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-arxiv", type=Path, required=True)
    parser.add_argument("--development-openalex", type=Path, required=True)
    parser.add_argument("--development-combined", type=Path, required=True)
    parser.add_argument("--validation-arxiv", type=Path, required=True)
    parser.add_argument("--validation-openalex", type=Path, required=True)
    parser.add_argument("--validation-combined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_source_complementarity_analysis(
        development_arxiv=args.development_arxiv,
        development_openalex=args.development_openalex,
        development_combined=args.development_combined,
        validation_arxiv=args.validation_arxiv,
        validation_openalex=args.validation_openalex,
        validation_combined=args.validation_combined,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["validation_acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
