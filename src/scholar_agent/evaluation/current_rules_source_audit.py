"""Pure-Replay source contribution and ablation audit for ``current_rules``.

The module reads frozen Benchmark results and retrieval snapshots only.  Source
failures, skipped requests, missing snapshots, and inconsistent terminals remain
explicit and are never converted to zero-contribution evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.current_rules_subquery_audit import (
    AUDIT_SCHEMA_VERSION,
    AuditDataset,
    _Cell,
    _average_metrics,
    _candidate_limit,
    _empty_costs,
    _metric_outcome,
    _read_cell,
    _read_json,
    _read_rows,
    _sha256,
    _sum_costs,
)
from scholar_agent.evaluation.datasets import load_dataset
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    _PreparedCounterfactual,
    _rank_pool,
    first_gold_rank,
    identity_cluster_labels,
    prepare_counterfactual_baseline,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.metrics import matched_paper_ids
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import SnapshotError


SourceStatus = Literal[
    "success",
    "failed",
    "not_started",
    "missing_snapshot",
    "terminal_inconsistent",
]


def run_source_audit(
    datasets: Sequence[AuditDataset],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for spec in sorted(datasets, key=lambda item: item.name):
        dataset_rows, input_row = _audit_dataset(spec)
        rows.extend(dataset_rows)
        inputs.append(input_row)
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["case_order"])))
    return rows, _aggregate(rows, inputs)


def write_source_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in case_rows
    )
    _atomic_write_text(root / "case_source_audit.jsonl", payload)
    _atomic_write_json(root / "aggregate.json", aggregate)


def source_terminal_status(cells: Sequence[_Cell]) -> tuple[SourceStatus, list[str]]:
    """Collapse all planned subquery cells for one case/source conservatively."""

    if any(cell.status == "missing_snapshot" for cell in cells):
        return "missing_snapshot", _source_reasons(cells, "missing_snapshot")
    if any(cell.status == "terminal_inconsistent" for cell in cells):
        return "terminal_inconsistent", _source_reasons(
            cells, "terminal_inconsistent"
        )
    if any(cell.status == "source_failure" for cell in cells):
        return "failed", _source_reasons(cells, "source_failure")
    if any(cell.status == "completed" for cell in cells):
        return "success", []
    return "not_started", ["no_source_request_started"]


def build_source_pool(
    selected: Sequence[dict[str, Any]],
    source_order: Sequence[str],
    cells: dict[tuple[str, str], _Cell],
    *,
    included_sources: Sequence[str],
    candidate_limit: int | None,
) -> list[Paper]:
    """Mirror the frozen query-major/source-order candidate merge."""

    included = set(included_sources)
    outputs: list[Paper] = []
    for subquery in sorted(
        selected, key=lambda item: (int(item.get("priority") or 0), str(item["query"]))
    ):
        per_query: list[Paper] = []
        for source in source_order:
            if source not in included:
                continue
            cell = cells[(str(subquery["query"]), source)]
            if cell.status == "completed":
                per_query.extend(cell.observation.papers)
        outputs.extend(deduplicate_papers(per_query))
    candidates = deduplicate_papers(outputs)
    if candidate_limit is not None and len(candidates) > candidate_limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=candidate_limit,
            source_order=[item for item in source_order if item in included],
        )
    return candidates


def source_candidate_observations(
    sources: Sequence[str],
    source_pools: dict[str, list[Paper]],
    statuses: dict[str, SourceStatus],
    gold: Sequence[EvalGoldPaper],
) -> dict[str, dict[str, Any]]:
    """Compute observed overlap; strict independence requires all sources success."""

    groups = [source_pools[source] for source in sources]
    labels = identity_cluster_labels(groups)
    gold_sets = [set(matched_paper_ids(group, gold)) for group in groups]
    all_sources_success = all(statuses[source] == "success" for source in sources)
    result: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        other_candidates = set().union(
            *(
                set(values)
                for offset, values in enumerate(labels)
                if offset != index
            )
        )
        other_gold = set().union(
            *(
                values
                for offset, values in enumerate(gold_sets)
                if offset != index
            )
        )
        independent_candidates = sorted(set(labels[index]) - other_candidates)
        independent_gold = sorted(gold_sets[index] - other_gold)
        result[source] = {
            "candidate_ids": labels[index],
            "gold_ids": sorted(gold_sets[index]),
            "observed_independent_candidate_ids": independent_candidates,
            "observed_independent_gold_ids": independent_gold,
            "strict_independence_comparable": all_sources_success,
            "strict_independent_candidate_ids": (
                independent_candidates if all_sources_success else None
            ),
            "strict_independent_gold_ids": (
                independent_gold if all_sources_success else None
            ),
            "first_gold_rank": first_gold_rank(groups[index], gold),
        }
    return result


def _audit_dataset(spec: AuditDataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = spec.run_dir.expanduser().resolve()
    snapshot_dir = spec.snapshot_dir.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    _validate_config(config, spec.name)
    result_rows = _read_rows(run_dir / "results.jsonl")
    dataset = load_dataset(str(config["dataset"]))
    cases = {item.query_id: item for item in dataset}
    case_ids = [str(value) for value in config["case_ids"]]
    if set(case_ids) != set(result_rows):
        raise ValueError(f"result case set mismatch:{spec.name}")
    if any(case_id not in cases for case_id in case_ids):
        raise ValueError(f"dataset case missing:{spec.name}")
    store = SnapshotStore(snapshot_dir)
    rows = [
        _audit_case(
            spec.name,
            order,
            cases[case_id],
            result_rows[case_id],
            config,
            store,
        )
        for order, case_id in enumerate(case_ids)
    ]
    return rows, {
        "name": spec.name,
        "case_count": len(case_ids),
        "run_results_sha256": _sha256(run_dir / "results.jsonl"),
        "run_config_sha256": _sha256(run_dir / "config.json"),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
    }


def _audit_case(
    dataset_name: str,
    case_order: int,
    eval_query: EvalQuery,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
) -> dict[str, Any]:
    planning = row["stage_diagnostics"]["initial_query_planning"]["planning"]
    selected = list(planning["selected_subqueries"])
    sources = [str(value) for value in config["sources"]]
    cells: dict[tuple[str, str], _Cell] = {}
    for subquery in selected:
        for source in sources:
            query = str(subquery["query"])
            cells[(query, source)] = _read_cell(
                str(eval_query.query_id),
                row,
                config,
                store,
                query=query,
                source=source,
                purpose=str(subquery.get("purpose") or "unknown"),
                priority=int(subquery.get("priority") or 0),
            )

    status_rows: dict[str, tuple[SourceStatus, list[str]]] = {}
    source_pools: dict[str, list[Paper]] = {}
    raw_counts: dict[str, int] = {}
    for source in sources:
        source_cells = [
            cells[(str(subquery["query"]), source)] for subquery in selected
        ]
        status_rows[source] = source_terminal_status(source_cells)
        source_pools[source] = build_source_pool(
            selected,
            sources,
            cells,
            included_sources=[source],
            candidate_limit=None,
        )
        raw_counts[source] = sum(
            cell.observation.raw_count
            for cell in source_cells
            if cell.status == "completed"
        )
    statuses = {source: status_rows[source][0] for source in sources}
    observations = source_candidate_observations(
        sources, source_pools, statuses, eval_query.gold_papers
    )

    prepared: _PreparedCounterfactual | None = None
    reconstruction_error: str | None = None
    try:
        prepared = prepare_counterfactual_baseline(row, config, store, eval_query)
    except (SnapshotError, ValueError) as exc:
        reconstruction_error = type(exc).__name__
    source_rows: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_cells = [
            cells[(str(subquery["query"]), source)] for subquery in selected
        ]
        unique_count = len(source_pools[source])
        source_row = {
            "status": statuses[source],
            "reasons": status_rows[source][1],
            "planned_subquery_count": len(selected),
            "cell_status_counts": _cell_status_counts(source_cells),
            "raw_candidate_count": raw_counts[source],
            "unique_candidate_count": unique_count,
            "duplicate_ratio": (
                max(0, raw_counts[source] - unique_count) / raw_counts[source]
                if raw_counts[source]
                else 0.0
            ),
            **observations[source],
            "recorded_costs": _source_costs(source_cells),
            "single_source": None,
            "leave_one_out": None,
        }
        if prepared is not None:
            single_pool = build_source_pool(
                selected,
                sources,
                cells,
                included_sources=[source],
                candidate_limit=_candidate_limit(config),
            )
            leave_pool = build_source_pool(
                selected,
                sources,
                cells,
                included_sources=[item for item in sources if item != source],
                candidate_limit=_candidate_limit(config),
            )
            single_metrics, _, _ = _rank_pool(
                prepared.analysis, single_pool, eval_query.gold_papers
            )
            leave_metrics, _, _ = _rank_pool(
                prepared.analysis, leave_pool, eval_query.gold_papers
            )
            source_row["single_source"] = single_metrics
            source_row["leave_one_out"] = leave_metrics
        source_rows[source] = source_row

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": dataset_name,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "query": eval_query.query,
        "gold_count": len(eval_query.gold_papers),
        "source_order": sources,
        "source_status_counts": _source_status_counts(statuses.values()),
        "all_sources_success": all(value == "success" for value in statuses.values()),
        "baseline": prepared.baseline_metrics if prepared is not None else None,
        "reconstruction_error": reconstruction_error,
        "sources": source_rows,
    }


def _aggregate(
    case_rows: Sequence[dict[str, Any]], inputs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in sorted({str(row["dataset"]) for row in case_rows}):
        rows = [row for row in case_rows if row["dataset"] == dataset_name]
        source_order = list(rows[0]["source_order"])
        datasets[dataset_name] = {
            "case_count": len(rows),
            "source_case_count": len(rows) * len(source_order),
            "source_status_counts": _source_status_counts(
                row["sources"][source]["status"]
                for row in rows
                for source in source_order
            ),
            "all_sources_success_case_count": sum(
                bool(row["all_sources_success"]) for row in rows
            ),
            "frozen_four_source_baseline": _average_metrics(
                row["baseline"] for row in rows if row["baseline"] is not None
            ),
            "sources": {
                source: _aggregate_source(rows, source) for source in source_order
            },
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "offline_frozen_snapshot_replay",
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "inputs": list(inputs),
        "datasets": datasets,
        "cross_dataset_safe_removal": _cross_dataset_removal(datasets),
    }


def _aggregate_source(rows: Sequence[dict[str, Any]], source: str) -> dict[str, Any]:
    source_rows = [row["sources"][source] for row in rows]
    available = [
        row for row, source_row in zip(rows, source_rows)
        if row["baseline"] is not None
        and source_row["single_source"] is not None
        and source_row["leave_one_out"] is not None
    ]
    strict = [
        row for row in available if row["sources"][source]["status"] == "success"
    ]
    strict_all_sources = [row for row in available if row["all_sources_success"]]
    raw = sum(int(item["raw_candidate_count"]) for item in source_rows)
    unique = sum(int(item["unique_candidate_count"]) for item in source_rows)
    first_ranks = [
        int(item["first_gold_rank"])
        for item in source_rows
        if item["first_gold_rank"] is not None
    ]
    return {
        "status_counts": _source_status_counts(item["status"] for item in source_rows),
        "observed_contribution": {
            "raw_candidate_count": raw,
            "summed_per_case_unique_candidate_count": unique,
            "duplicate_ratio": max(0, raw - unique) / raw if raw else 0.0,
            "gold_hit_count_source_cases": sum(
                len(item["gold_ids"]) for item in source_rows
            ),
            "independent_candidate_count_observed": sum(
                len(item["observed_independent_candidate_ids"])
                for item in source_rows
            ),
            "independent_gold_count_observed": sum(
                len(item["observed_independent_gold_ids"]) for item in source_rows
            ),
            "strict_independence_case_count": len(strict_all_sources),
            "strict_independent_candidate_count": sum(
                len(row["sources"][source]["strict_independent_candidate_ids"])
                for row in strict_all_sources
            ),
            "strict_independent_gold_count": sum(
                len(row["sources"][source]["strict_independent_gold_ids"])
                for row in strict_all_sources
            ),
            "first_gold_rank_count": len(first_ranks),
            "minimum_first_gold_rank": min(first_ranks) if first_ranks else None,
            "average_first_gold_rank": (
                sum(first_ranks) / len(first_ranks) if first_ranks else None
            ),
        },
        "lower_bound_all_cases": _condition_summary(available, source),
        "strict_source_success": _condition_summary(strict, source),
        "recorded_costs_all_cases": _sum_costs(
            item["recorded_costs"] for item in source_rows
        ),
        "recorded_costs_strict_subset": _sum_costs(
            row["sources"][source]["recorded_costs"] for row in strict
        ),
    }


def _condition_summary(rows: Sequence[dict[str, Any]], source: str) -> dict[str, Any]:
    baselines = [row["baseline"] for row in rows]
    single = [row["sources"][source]["single_source"] for row in rows]
    leave = [row["sources"][source]["leave_one_out"] for row in rows]
    return {
        "case_count": len(rows),
        "baseline": _average_metrics(baselines),
        "single_source": _average_metrics(single),
        "leave_one_out": _average_metrics(leave),
        "single_source_outcomes": _outcomes(baselines, single),
        "leave_one_out_outcomes": _outcomes(baselines, leave),
        "leave_one_out_candidate_gold_lost_count": sum(
            len(set(left["candidate_gold_ids"]) - set(right["candidate_gold_ids"]))
            for left, right in zip(baselines, leave)
        ),
        "leave_one_out_top20_gold_lost_count": sum(
            len(set(left["returned_gold_ids"]) - set(right["returned_gold_ids"]))
            for left, right in zip(baselines, leave)
        ),
        "leave_one_out_top20_gold_recovered_count": sum(
            len(set(right["returned_gold_ids"]) - set(left["returned_gold_ids"]))
            for left, right in zip(baselines, leave)
        ),
        "request_savings": _sum_costs(
            row["sources"][source]["recorded_costs"] for row in rows
        ),
    }


def _outcomes(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> dict[str, int]:
    values = Counter(_metric_outcome(left, right) for left, right in zip(before, after))
    return {
        "improved": values["improved"],
        "tied": values["tied"],
        "degraded": values["degraded"],
    }


def _cross_dataset_removal(datasets: dict[str, Any]) -> list[dict[str, Any]]:
    source_sets = [set(value["sources"]) for value in datasets.values()]
    sources = sorted(set.intersection(*source_sets) if source_sets else set())
    result: list[dict[str, Any]] = []
    for source in sources:
        summaries = {
            dataset: value["sources"][source]["strict_source_success"]
            for dataset, value in sorted(datasets.items())
        }
        has_evidence = all(summary["case_count"] > 0 for summary in summaries.values())
        non_degrading = has_evidence and all(
            summary["leave_one_out_outcomes"]["degraded"] == 0
            and _metrics_non_decreasing(
                summary["baseline"], summary["leave_one_out"]
            )
            for summary in summaries.values()
        )
        request_savings = sum(
            int(summary["request_savings"]["request_count"])
            for summary in summaries.values()
        )
        result.append(
            {
                "source": source,
                "strict_case_counts": {
                    dataset: summary["case_count"]
                    for dataset, summary in summaries.items()
                },
                "strict_evidence_in_all_datasets": has_evidence,
                "non_degrading_in_all_strict_subsets": non_degrading,
                "recorded_request_savings": request_savings,
                "safe_removal_supported": non_degrading and request_savings > 0,
            }
        )
    return result


def _metrics_non_decreasing(before: dict[str, Any], after: dict[str, Any]) -> bool:
    fields = ("candidate_recall", "recall_at_20", "f1_at_20")
    if any(before[field] is None or after[field] is None for field in fields):
        return False
    return all(float(after[field]) >= float(before[field]) for field in fields)


def _source_costs(cells: Sequence[_Cell]) -> dict[str, float | int]:
    result = _empty_costs()
    seen: set[str] = set()
    for cell in cells:
        for terminal in cell.observation.terminals:
            key = str(terminal.get("key") or "")
            if terminal.get("status") == "not_started" or not key or key in seen:
                continue
            seen.add(key)
            diagnostics = terminal.get("recorded_diagnostics") or {}
            result["snapshot_key_count"] += 1
            result["request_count"] += int(diagnostics.get("request_count") or 0)
            result["retry_count"] += int(diagnostics.get("retry_count") or 0)
            result["error_count"] += int(diagnostics.get("error_count") or 0)
            result["cache_hit_count"] += int(diagnostics.get("cache_hit_count") or 0)
            result["latency_seconds"] += float(
                diagnostics.get("latency_seconds") or 0.0
            )
            result["rate_limit_wait_seconds"] += float(
                diagnostics.get("rate_limit_wait_seconds") or 0.0
            )
    return result


def _source_reasons(cells: Sequence[_Cell], status: str) -> list[str]:
    return sorted(
        {
            f"{cell.purpose}:{reason}"
            for cell in cells
            if cell.status == status
            for reason in cell.reasons
        }
    )


def _cell_status_counts(cells: Sequence[_Cell]) -> dict[str, int]:
    counts = Counter(cell.status for cell in cells)
    return {
        status: counts[status]
        for status in (
            "completed",
            "source_failure",
            "not_started",
            "missing_snapshot",
            "terminal_inconsistent",
        )
    }


def _source_status_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {
        status: counts[status]
        for status in (
            "success",
            "failed",
            "not_started",
            "missing_snapshot",
            "terminal_inconsistent",
        )
    }


def _validate_config(config: dict[str, Any], name: str) -> None:
    llm = config.get("llm") or {}
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError(f"{name} is not current_rules")
    if (
        bool(llm.get("llm_enabled"))
        or bool(config.get("enable_query_evolution"))
        or bool(config.get("enable_refchain"))
        or config.get("ranking_policy") != "current_rules"
    ):
        raise ValueError(f"{name} enables a forbidden strategy")


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
