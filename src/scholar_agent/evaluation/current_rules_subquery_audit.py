"""Offline marginal-contribution audit for frozen ``current_rules`` subqueries.

The audit reads Benchmark Replay artifacts and retrieval snapshots directly.  It
does not construct ``SearchService`` or any connector, and therefore cannot send
academic API or LLM requests.  Gold is introduced only after every frozen query
list has been reconstructed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.datasets import load_dataset
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    _PreparedCounterfactual,
    _QueryList,
    _query_list,
    _rank_pool,
    first_gold_rank,
    identity_cluster_labels,
    prepare_counterfactual_baseline,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.metrics import matched_paper_ids
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import SnapshotError, SnapshotMissingError


AUDIT_SCHEMA_VERSION = "1"
CellStatus = Literal[
    "completed",
    "source_failure",
    "not_started",
    "missing_snapshot",
    "terminal_inconsistent",
]


class AuditDataset(BaseModel):
    """One frozen current-rules run and its retrieval snapshot store."""

    name: str
    run_dir: Path
    snapshot_dir: Path


class _Cell(NamedTuple):
    case_id: str
    source: str
    query: str
    purpose: str
    priority: int
    observation: _QueryList
    status: CellStatus
    reasons: list[str]
    costs: dict[str, float | int]


def run_subquery_audit(
    datasets: Sequence[AuditDataset],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a deterministic, read-only audit over one or more frozen datasets."""

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for spec in sorted(datasets, key=lambda item: item.name):
        dataset_rows, input_row = _audit_dataset(spec)
        rows.extend(dataset_rows)
        inputs.append(input_row)
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["case_order"])))
    aggregate = _aggregate(rows, inputs)
    return rows, aggregate


def write_subquery_audit(
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
    _atomic_write_text(root / "case_audit.jsonl", payload)
    _atomic_write_json(root / "aggregate.json", aggregate)


def classify_terminals(terminals: Sequence[dict[str, Any]]) -> tuple[CellStatus, list[str]]:
    """Classify one planned source/subquery cell without treating absence as zero."""

    statuses = [str(item.get("status") or "") for item in terminals]
    if "failed" in statuses:
        return "source_failure", sorted(
            {
                str(item.get("error_type") or "source_failure")
                for item in terminals
                if item.get("status") == "failed"
            }
        )
    if "success" in statuses:
        return "completed", []
    return "not_started", ["no_adapted_query_was_started"]


def source_list_contributions(
    cells: Sequence[_Cell], gold: Sequence[EvalGoldPaper]
) -> list[dict[str, Any]]:
    """Return overlap and plan-order marginal contribution for one source."""

    ordered = sorted(cells, key=lambda item: (item.priority, item.query))
    comparable = [item for item in ordered if item.status == "completed"]
    complete_context = all(
        item.status in {"completed", "not_started"} for item in ordered
    )
    labels = identity_cluster_labels([item.observation.papers for item in comparable])
    gold_sets = [
        set(matched_paper_ids(item.observation.papers, gold)) for item in comparable
    ]
    comparable_index = {id(cell): index for index, cell in enumerate(comparable)}
    accumulated: list[Paper] = []
    prior_incomplete = False
    result: list[dict[str, Any]] = []
    for cell in ordered:
        base = {
            "source": cell.source,
            "query": cell.query,
            "purpose": cell.purpose,
            "priority": cell.priority,
            "query_kind": (
                "original" if cell.purpose == "original_query" else "derived"
            ),
            "status": cell.status,
            "reasons": cell.reasons,
            "request_terminals": cell.observation.terminals,
            "costs": cell.costs,
            "raw_candidate_count": cell.observation.raw_count,
            "unique_candidate_count": len(cell.observation.papers),
            "duplicate_ratio": (
                max(0, cell.observation.raw_count - len(cell.observation.papers))
                / cell.observation.raw_count
                if cell.observation.raw_count
                else 0.0
            ),
            "gold_ids": sorted(
                matched_paper_ids(cell.observation.papers, gold)
            ),
            "first_gold_rank": first_gold_rank(cell.observation.papers, gold),
            "independent_candidate_ids": None,
            "independent_gold_ids": None,
            "plan_order_marginal_candidate_count": None,
            "plan_order_marginal_gold_ids": None,
            "contribution_comparable": False,
        }
        if cell.status == "completed":
            index = comparable_index[id(cell)]
            if complete_context:
                other_candidates = set().union(
                    *(
                        set(value)
                        for offset, value in enumerate(labels)
                        if offset != index
                    )
                )
                other_gold = set().union(
                    *(
                        value
                        for offset, value in enumerate(gold_sets)
                        if offset != index
                    )
                )
                base["independent_candidate_ids"] = sorted(
                    set(labels[index]) - other_candidates
                )
                base["independent_gold_ids"] = sorted(
                    gold_sets[index] - other_gold
                )
            if not prior_incomplete:
                before = deduplicate_papers(accumulated)
                after = deduplicate_papers([*before, *cell.observation.papers])
                before_gold = set(matched_paper_ids(before, gold))
                after_gold = set(matched_paper_ids(after, gold))
                base["plan_order_marginal_candidate_count"] = len(after) - len(before)
                base["plan_order_marginal_gold_ids"] = sorted(
                    after_gold - before_gold
                )
                base["contribution_comparable"] = True
                accumulated = after
        elif cell.status != "not_started":
            prior_incomplete = True
        result.append(base)
    return result


def build_query_type_pool(
    selected: Sequence[dict[str, Any]],
    sources: Sequence[str],
    cells: dict[tuple[str, str], _Cell],
    *,
    candidate_limit: int,
    only_purpose: str | None = None,
    remove_purpose: str | None = None,
) -> list[Paper]:
    """Rebuild the production first-seen pool after a query-type intervention."""

    outputs: list[Paper] = []
    for subquery in sorted(
        selected, key=lambda item: (int(item.get("priority") or 0), str(item["query"]))
    ):
        purpose = str(subquery.get("purpose") or "unknown")
        if only_purpose is not None and purpose != only_purpose:
            continue
        if remove_purpose is not None and purpose == remove_purpose:
            continue
        per_query: list[Paper] = []
        for source in sources:
            cell = cells[(str(subquery["query"]), str(source))]
            if cell.status == "completed":
                per_query.extend(cell.observation.papers)
        outputs.extend(deduplicate_papers(per_query))
    candidates = deduplicate_papers(outputs)
    if len(candidates) > candidate_limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=candidate_limit,
            source_order=sources,
        )
    return candidates


def _audit_dataset(spec: AuditDataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = spec.run_dir.expanduser().resolve()
    snapshot_dir = spec.snapshot_dir.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError(f"{spec.name} is not current_rules")
    llm = config.get("llm") or {}
    if (
        bool(llm.get("llm_enabled"))
        or bool(config.get("enable_query_evolution"))
        or bool(config.get("enable_refchain"))
        or config.get("ranking_policy") != "current_rules"
    ):
        raise ValueError(f"{spec.name} enables a forbidden strategy")
    result_rows = _read_rows(run_dir / "results.jsonl")
    dataset = load_dataset(str(config["dataset"]))
    cases = {item.query_id: item for item in dataset}
    selected_ids = [str(item) for item in config["case_ids"]]
    if set(selected_ids) != set(result_rows):
        raise ValueError(f"result case set mismatch:{spec.name}")
    if any(case_id not in cases for case_id in selected_ids):
        raise ValueError(f"dataset case missing:{spec.name}")

    store = SnapshotStore(snapshot_dir)
    rows: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(selected_ids):
        rows.append(
            _audit_case(
                spec.name,
                case_order,
                cases[case_id],
                result_rows[case_id],
                config,
                store,
            )
        )
    return rows, {
        "name": spec.name,
        "run_results_sha256": _sha256(run_dir / "results.jsonl"),
        "run_config_sha256": _sha256(run_dir / "config.json"),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        "case_count": len(selected_ids),
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
    sources = [str(item) for item in config["sources"]]
    cells: dict[tuple[str, str], _Cell] = {}
    for subquery in selected:
        query = str(subquery["query"])
        purpose = str(subquery.get("purpose") or "unknown")
        priority = int(subquery.get("priority") or 0)
        for source in sources:
            cells[(query, source)] = _read_cell(
                str(eval_query.query_id),
                row,
                config,
                store,
                query=query,
                source=source,
                purpose=purpose,
                priority=priority,
            )

    source_rows: list[dict[str, Any]] = []
    for source in sources:
        source_cells = [
            cells[(str(item["query"]), source)] for item in selected
        ]
        source_rows.extend(source_list_contributions(source_cells, eval_query.gold_papers))

    prepared: _PreparedCounterfactual | None = None
    reconstruction_error: str | None = None
    try:
        prepared = prepare_counterfactual_baseline(row, config, store, eval_query)
    except (SnapshotError, ValueError) as exc:
        reconstruction_error = type(exc).__name__

    purposes = sorted(
        {
            str(item.get("purpose") or "unknown")
            for item in selected
            if str(item.get("purpose") or "unknown") != "original_query"
        }
    )
    counterfactual = {
        "baseline": prepared.baseline_metrics if prepared is not None else None,
        "only_original": None,
        "remove_by_purpose": {},
        "reconstruction_error": reconstruction_error,
    }
    candidate_limit = _candidate_limit(config)
    if prepared is not None:
        original_cells = [
            cell for cell in cells.values() if cell.purpose == "original_query"
        ]
        original_comparable = not any(
            cell.status
            in {"source_failure", "missing_snapshot", "terminal_inconsistent"}
            for cell in original_cells
        ) and any(cell.status == "completed" for cell in original_cells)
        if original_comparable:
            original_pool = build_query_type_pool(
                selected,
                sources,
                cells,
                candidate_limit=candidate_limit,
                only_purpose="original_query",
            )
            metrics, _, _ = _rank_pool(
                prepared.analysis, original_pool, eval_query.gold_papers
            )
            counterfactual["only_original"] = {
                "status": "comparable",
                "metrics": metrics,
                "request_savings": _request_savings(cells.values(), keep_original=True),
            }
        else:
            counterfactual["only_original"] = {
                "status": "incomparable",
                "reasons": _cell_failure_reasons(original_cells),
            }

        for purpose in purposes:
            target = [cell for cell in cells.values() if cell.purpose == purpose]
            target_started = any(cell.status != "not_started" for cell in target)
            target_complete = target_started and not any(
                cell.status
                in {"source_failure", "missing_snapshot", "terminal_inconsistent"}
                for cell in target
            )
            if not target_complete:
                counterfactual["remove_by_purpose"][purpose] = {
                    "status": "not_executed" if not target_started else "incomparable",
                    "reasons": _cell_failure_reasons(target),
                }
                continue
            pool = build_query_type_pool(
                selected,
                sources,
                cells,
                candidate_limit=candidate_limit,
                remove_purpose=purpose,
            )
            metrics, _, _ = _rank_pool(prepared.analysis, pool, eval_query.gold_papers)
            counterfactual["remove_by_purpose"][purpose] = {
                "status": "comparable",
                "metrics": metrics,
                "request_savings": _request_savings(
                    cells.values(), remove_purpose=purpose
                ),
            }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": dataset_name,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "query": eval_query.query,
        "gold_count": len(eval_query.gold_papers),
        "selected_subquery_count": len(selected),
        "planned_source_subquery_count": len(selected) * len(sources),
        "cell_status_counts": _status_counts(
            cell.status for cell in cells.values()
        ),
        "subquery_contributions": sorted(
            source_rows,
            key=lambda item: (
                sources.index(str(item["source"])),
                int(item["priority"]),
                str(item["query"]),
            ),
        ),
        "counterfactual": counterfactual,
    }


def _read_cell(
    case_id: str,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    *,
    query: str,
    source: str,
    purpose: str,
    priority: int,
) -> _Cell:
    try:
        observation = _query_list(row, config, store, query, source)
        observation.terminals = _enrich_terminals(observation.terminals, store)
        status, reasons = classify_terminals(observation.terminals)
        costs = _terminal_costs(observation.terminals)
    except SnapshotMissingError as exc:
        observation = _QueryList(query, source, [], 0, [])
        status = "missing_snapshot"
        reasons = [type(exc).__name__]
        costs = _empty_costs()
    except (SnapshotError, ValueError) as exc:
        observation = _QueryList(query, source, [], 0, [])
        status = "terminal_inconsistent"
        reasons = [type(exc).__name__]
        costs = _empty_costs()
    return _Cell(
        case_id,
        source,
        query,
        purpose,
        priority,
        observation,
        status,
        reasons,
        costs,
    )


def _enrich_terminals(
    terminals: Sequence[dict[str, Any]], store: SnapshotStore
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for terminal in terminals:
        enriched = dict(terminal)
        if terminal.get("status") != "not_started":
            entry = store.read_retrieval(str(terminal["key"]))
            enriched["recorded_diagnostics"] = entry.diagnostics.model_dump(
                mode="json", exclude_none=True
            )
            enriched["recorded_latency_seconds"] = entry.recorded_latency_seconds
        result.append(enriched)
    return result


def _terminal_costs(
    terminals: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    costs = _empty_costs()
    seen: set[str] = set()
    for terminal in terminals:
        key = str(terminal.get("key") or "")
        if terminal.get("status") == "not_started" or not key or key in seen:
            continue
        seen.add(key)
        diagnostics = terminal.get("recorded_diagnostics") or {}
        costs["snapshot_key_count"] += 1
        costs["request_count"] += int(diagnostics.get("request_count") or 0)
        costs["retry_count"] += int(diagnostics.get("retry_count") or 0)
        costs["error_count"] += int(diagnostics.get("error_count") or 0)
        costs["cache_hit_count"] += int(diagnostics.get("cache_hit_count") or 0)
        costs["latency_seconds"] += float(diagnostics.get("latency_seconds") or 0.0)
        costs["rate_limit_wait_seconds"] += float(
            diagnostics.get("rate_limit_wait_seconds") or 0.0
        )
    return costs


def _request_savings(
    cells: Iterable[_Cell],
    *,
    remove_purpose: str | None = None,
    keep_original: bool = False,
) -> dict[str, float | int]:
    materialized = list(cells)
    owners: dict[str, set[str]] = defaultdict(set)
    entries: dict[str, dict[str, Any]] = {}
    for cell in materialized:
        for terminal in cell.observation.terminals:
            key = str(terminal.get("key") or "")
            if terminal.get("status") == "not_started" or not key:
                continue
            owners[key].add(cell.purpose)
            entries.setdefault(key, terminal)
    if keep_original:
        removable = {
            key for key, values in owners.items() if "original_query" not in values
        }
    else:
        removable = {
            key for key, values in owners.items() if values == {remove_purpose}
        }
    result = _empty_costs()
    for key in sorted(removable):
        terminal = entries[key]
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


def _aggregate(
    case_rows: Sequence[dict[str, Any]], inputs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in case_rows}):
        rows = [row for row in case_rows if row["dataset"] == dataset]
        cells = [item for row in rows for item in row["subquery_contributions"]]
        purposes = sorted(
            {
                str(item["purpose"])
                for item in cells
                if item["purpose"] != "original_query"
            }
        )
        only_original_rows = [
            row
            for row in rows
            if (row["counterfactual"]["only_original"] or {}).get("status")
            == "comparable"
        ]
        datasets[dataset] = {
            "case_count": len(rows),
            "planned_source_subquery_count": len(cells),
            "cell_status_counts": _status_counts(
                str(item["status"]) for item in cells
            ),
            "failure_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for item in cells
                        if item["status"]
                        in {
                            "source_failure",
                            "missing_snapshot",
                            "terminal_inconsistent",
                        }
                        for reason in item["reasons"]
                    ).items()
                )
            ),
            "query_type_contribution": {
                purpose: _aggregate_query_type(cells, purpose)
                for purpose in ["original_query", *purposes]
            },
            "frozen_baseline": _average_metrics(
                row["counterfactual"]["baseline"]
                for row in rows
                if row["counterfactual"]["baseline"] is not None
            ),
            "derived_relative_to_original": _aggregate_original_counterfactual(
                only_original_rows
            ),
            "remove_by_purpose": {
                purpose: _aggregate_removal(rows, purpose) for purpose in purposes
            },
        }
    removable = _cross_dataset_removable(datasets)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "offline_frozen_snapshot_replay",
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "inputs": list(inputs),
        "datasets": datasets,
        "cross_dataset_removable_query_types": removable,
    }


def _aggregate_query_type(cells: Sequence[dict[str, Any]], purpose: str) -> dict[str, Any]:
    selected = [item for item in cells if item["purpose"] == purpose]
    completed = [item for item in selected if item["status"] == "completed"]
    comparable = [item for item in completed if item["contribution_comparable"]]
    raw = sum(int(item["raw_candidate_count"]) for item in completed)
    unique = sum(int(item["unique_candidate_count"]) for item in completed)
    independent_candidates = [
        item for item in completed if item["independent_candidate_ids"] is not None
    ]
    first_ranks = [
        int(item["first_gold_rank"])
        for item in completed
        if item["first_gold_rank"] is not None
    ]
    return {
        "planned_cell_count": len(selected),
        "completed_cell_count": len(completed),
        "incomparable_cell_count": len(selected) - len(completed),
        "raw_candidate_count": raw,
        "summed_per_list_unique_candidate_count": unique,
        "duplicate_ratio": max(0, raw - unique) / raw if raw else 0.0,
        "gold_hit_count_source_pairs": sum(len(item["gold_ids"]) for item in completed),
        "independent_candidate_count_source_pairs": sum(
            len(item["independent_candidate_ids"])
            for item in independent_candidates
        ),
        "independent_gold_count_source_pairs": sum(
            len(item["independent_gold_ids"])
            for item in independent_candidates
        ),
        "plan_order_comparable_cell_count": len(comparable),
        "plan_order_marginal_candidate_count": sum(
            int(item["plan_order_marginal_candidate_count"] or 0)
            for item in comparable
        ),
        "plan_order_marginal_gold_count_source_pairs": sum(
            len(item["plan_order_marginal_gold_ids"] or []) for item in comparable
        ),
        "first_gold_rank_count": len(first_ranks),
        "minimum_first_gold_rank": min(first_ranks) if first_ranks else None,
        "average_first_gold_rank": (
            sum(first_ranks) / len(first_ranks) if first_ranks else None
        ),
        "recorded_costs": _unique_terminal_costs(completed),
    }


def _aggregate_original_counterfactual(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (
            row["counterfactual"]["baseline"],
            row["counterfactual"]["only_original"]["metrics"],
        )
        for row in rows
    ]
    return {
        "comparable_case_count": len(rows),
        "baseline": _average_metrics(item[0] for item in pairs),
        "only_original": _average_metrics(item[1] for item in pairs),
        "derived_added_candidate_gold_count": sum(
            len(set(full["candidate_gold_ids"]) - set(original["candidate_gold_ids"]))
            for full, original in pairs
        ),
        "derived_added_top20_gold_count": sum(
            len(set(full["returned_gold_ids"]) - set(original["returned_gold_ids"]))
            for full, original in pairs
        ),
        "original_only_gold_recovered_by_budget_count": sum(
            len(set(original["returned_gold_ids"]) - set(full["returned_gold_ids"]))
            for full, original in pairs
        ),
        "request_savings": _sum_costs(
            row["counterfactual"]["only_original"]["request_savings"]
            for row in rows
        ),
    }


def _aggregate_removal(rows: Sequence[dict[str, Any]], purpose: str) -> dict[str, Any]:
    comparable = [
        row
        for row in rows
        if (
            row["counterfactual"]["remove_by_purpose"].get(purpose) or {}
        ).get("status")
        == "comparable"
    ]
    before = [row["counterfactual"]["baseline"] for row in comparable]
    after = [
        row["counterfactual"]["remove_by_purpose"][purpose]["metrics"]
        for row in comparable
    ]
    outcomes = Counter(_metric_outcome(left, right) for left, right in zip(before, after))
    statuses = Counter(
        str(
            (row["counterfactual"]["remove_by_purpose"].get(purpose) or {}).get(
                "status", "not_selected"
            )
        )
        for row in rows
    )
    return {
        "selected_case_count": len(rows) - statuses["not_selected"],
        "comparable_case_count": len(comparable),
        "status_counts": dict(sorted(statuses.items())),
        "baseline": _average_metrics(before),
        "without_query_type": _average_metrics(after),
        "outcome_counts": {
            "improved": outcomes["improved"],
            "tied": outcomes["tied"],
            "degraded": outcomes["degraded"],
        },
        "candidate_gold_lost_count": sum(
            len(set(left["candidate_gold_ids"]) - set(right["candidate_gold_ids"]))
            for left, right in zip(before, after)
        ),
        "top20_gold_lost_count": sum(
            len(set(left["returned_gold_ids"]) - set(right["returned_gold_ids"]))
            for left, right in zip(before, after)
        ),
        "top20_gold_recovered_count": sum(
            len(set(right["returned_gold_ids"]) - set(left["returned_gold_ids"]))
            for left, right in zip(before, after)
        ),
        "request_savings": _sum_costs(
            row["counterfactual"]["remove_by_purpose"][purpose]["request_savings"]
            for row in comparable
        ),
    }


def _cross_dataset_removable(datasets: dict[str, Any]) -> list[dict[str, Any]]:
    purposes = sorted(
        set.intersection(
            *(
                set(value["remove_by_purpose"])
                for value in datasets.values()
            )
        )
        if datasets
        else set()
    )
    result: list[dict[str, Any]] = []
    for purpose in purposes:
        rows = [value["remove_by_purpose"][purpose] for value in datasets.values()]
        fully_comparable = all(
            row["comparable_case_count"] > 0
            and row["comparable_case_count"] == row["selected_case_count"]
            for row in rows
        )
        non_degrading = fully_comparable and all(
            row["outcome_counts"]["degraded"] == 0
            and (row["without_query_type"]["candidate_recall"] or 0.0)
            >= (row["baseline"]["candidate_recall"] or 0.0)
            and (row["without_query_type"]["recall_at_20"] or 0.0)
            >= (row["baseline"]["recall_at_20"] or 0.0)
            and (row["without_query_type"]["f1_at_20"] or 0.0)
            >= (row["baseline"]["f1_at_20"] or 0.0)
            for row in rows
        )
        request_savings = sum(
            int(row["request_savings"]["request_count"]) for row in rows
        )
        selected_case_counts = {
            dataset: int(value["remove_by_purpose"][purpose]["selected_case_count"])
            for dataset, value in sorted(datasets.items())
        }
        result.append(
            {
                "purpose": purpose,
                "fully_comparable_across_datasets": fully_comparable,
                "non_degrading_across_datasets": non_degrading,
                "recorded_request_savings": request_savings,
                "selected_case_counts": selected_case_counts,
                "selected_case_count_total": sum(selected_case_counts.values()),
                "observed_non_degrading_with_request_savings": (
                    non_degrading and request_savings > 0
                ),
            }
        )
    return result


def _average_metrics(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(values)
    evaluable = [item for item in rows if item["candidate_recall"] is not None]
    return {
        "input_case_count": len(rows),
        "evaluable_case_count": len(evaluable),
        "candidate_recall": _average(
            item["candidate_recall"] for item in evaluable
        ),
        "recall_at_20": _average(item["recall_at_20"] for item in evaluable),
        "f1_at_20": _average(item["f1_at_20"] for item in evaluable),
        "candidate_gold_count": sum(
            len(item["candidate_gold_ids"]) for item in evaluable
        ),
        "top20_gold_count": sum(
            len(item["returned_gold_ids"]) for item in evaluable
        ),
    }


def _metric_outcome(before: dict[str, Any], after: dict[str, Any]) -> str:
    left = (
        float(before["candidate_recall"] or 0.0),
        float(before["recall_at_20"] or 0.0),
        float(before["f1_at_20"] or 0.0),
    )
    right = (
        float(after["candidate_recall"] or 0.0),
        float(after["recall_at_20"] or 0.0),
        float(after["f1_at_20"] or 0.0),
    )
    if right == left:
        return "tied"
    if all(right_value >= left_value for left_value, right_value in zip(left, right)):
        return "improved"
    return "degraded"


def _cell_failure_reasons(cells: Sequence[_Cell]) -> list[str]:
    reasons = [
        f"{cell.source}:{reason}"
        for cell in cells
        if cell.status
        in {"source_failure", "missing_snapshot", "terminal_inconsistent"}
        for reason in cell.reasons
    ]
    if not any(cell.status == "completed" for cell in cells):
        reasons.append("no_completed_source_list")
    return sorted(set(reasons))


def _empty_costs() -> dict[str, float | int]:
    return {
        "snapshot_key_count": 0,
        "request_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "cache_hit_count": 0,
        "latency_seconds": 0.0,
        "rate_limit_wait_seconds": 0.0,
    }


def _status_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
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


def _sum_costs(values: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    result = _empty_costs()
    for value in values:
        for field in result:
            result[field] += value[field]
    return result


def _unique_terminal_costs(cells: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    result = _empty_costs()
    seen: set[str] = set()
    for cell in cells:
        for terminal in cell["request_terminals"]:
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


def _average(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def _candidate_limit(config: dict[str, Any]) -> int:
    for field in ("budgets", "budget"):
        value = config.get(field)
        if isinstance(value, dict) and value.get("max_candidate_papers") is not None:
            return int(value["max_candidate_papers"])
    return 200


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["case_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case row:{path}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
