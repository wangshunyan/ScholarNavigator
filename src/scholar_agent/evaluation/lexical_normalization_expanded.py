"""Expanded offline pairing audit for lexical normalization on frozen Record data."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scholar_agent.core.evaluation_schemas import (
    DEDUPLICATED_GOLD_METRIC_VERSION,
)
from scholar_agent.evaluation.lexical_normalization_benchmark import (
    _aggregate_dataset,
    _audit_case,
)
from scholar_agent.evaluation.paired_significance import _analyze_group
from scholar_agent.evaluation.relevance_filter_audit import (
    _load_queries,
    _read_json,
    _sha256,
    _tree_sha256,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


SCHEMA_VERSION = "1"
METRIC_VERSION = DEDUPLICATED_GOLD_METRIC_VERSION
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def run_expanded_lexical_audit(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rejudge frozen Record candidates without network or Snapshot mutation."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    record_spec = manifest["frozen_record"]
    run_root = _repo_path(record_spec["run_dir"])
    snapshot_root = _repo_path(record_spec["snapshot_dir"])
    config_path = run_root / "config.json"
    results_path = run_root / "results.jsonl"
    _validate_hash(config_path, record_spec["config_sha256"])
    _validate_hash(results_path, record_spec["results_sha256"])
    _validate_hash(
        _repo_path(manifest["gold_identity_baseline"]["path"]),
        manifest["gold_identity_baseline"]["sha256"],
    )
    _validate_hash(
        _repo_path(manifest["prior_65_reference"]["path"]),
        manifest["prior_65_reference"]["sha256"],
    )
    _validate_hash(
        _repo_path(manifest["policy"]["manifest"]),
        manifest["policy"]["manifest_sha256"],
    )
    snapshot_before = _tree_sha256(snapshot_root)
    if snapshot_before != record_spec["snapshot_tree_sha256"]:
        raise ValueError("frozen Snapshot tree hash drift")
    snapshot_file_count = sum(path.is_file() for path in snapshot_root.rglob("*"))
    if snapshot_file_count != int(record_spec["snapshot_file_count"]):
        raise ValueError("frozen Snapshot file count drift")

    config = _read_json(config_path)
    _validate_record_config(config, manifest)
    rows = _read_ordered_rows(results_path)
    expected_record_count = int(record_spec["record_case_count"])
    if len(rows) != expected_record_count:
        raise ValueError("frozen Record case count drift")
    record_case_ids = [str(item.get("case_id") or "") for item in rows]
    if record_case_ids != [str(item) for item in config["case_ids"]][
        :expected_record_count
    ]:
        raise ValueError("frozen Record is not the ordered config prefix")
    if len(set(record_case_ids)) != len(record_case_ids):
        raise ValueError("duplicate frozen Record case ID")

    dataset_spec = {
        "dataset": config["dataset"],
    }
    queries = _load_queries(config, dataset_spec)
    by_case = {item.query_id: item for item in queries}
    if any(case_id not in by_case for case_id in record_case_ids):
        raise ValueError("frozen Record case missing from dataset")

    overlap_ids = {str(item) for item in manifest["prior_dev_val_overlap_case_ids"]}
    if len(overlap_ids) != len(manifest["prior_dev_val_overlap_case_ids"]):
        raise ValueError("duplicate prior overlap case ID")
    if not overlap_ids.issubset(record_case_ids):
        raise ValueError("prior dev/val overlap is outside frozen Record prefix")

    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    included_rows: list[dict[str, Any]] = []
    included_states: list[dict[str, Any]] = []
    input_key_count = 0
    source_case_terminals: Counter[str] = Counter()
    for case_order, (case_id, raw_row) in enumerate(zip(record_case_ids, rows, strict=True)):
        prepared, terminal = resolve_record_terminals(
            raw_row,
            store=store,
            configured_sources=[str(item) for item in config["sources"]],
        )
        input_key_count += int(terminal["snapshot_key_count"])
        for source, status in terminal["source_states"].items():
            source_case_terminals[f"{source}:{status}"] += 1
        base = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "autoscholar_record160",
            "case_order": case_order,
            "case_id": case_id,
            "metric_version": METRIC_VERSION,
            "successful_source_count": terminal["successful_source_count"],
            "source_states": terminal["source_states"],
            "source_terminal_signature_sha256": terminal[
                "source_terminal_signature_sha256"
            ],
            "source_terminal_parity": True,
            "overlaps_prior_auto_dev_val": case_id in overlap_ids,
        }
        if terminal["successful_source_count"] == 0:
            case_rows.append(
                {
                    **base,
                    "analysis_status": "excluded_no_successful_source",
                    "included_main_analysis": False,
                    "candidate_identity_parity": None,
                    "candidate_order_parity": None,
                    "judgement_input_parity": None,
                    "baseline_frozen_judgement_parity": None,
                }
            )
            continue

        try:
            case, candidates, state = _audit_case(
                label="autoscholar_record160",
                case_order=case_order,
                eval_query=by_case[case_id],
                row=prepared,
                config=config,
                store=store,
                required_candidate_source=None,
                prior_lexical_false_negatives=set(),
                include_instance_identity_keys=True,
            )
        except ValueError as exc:
            case_rows.append(
                {
                    **base,
                    "analysis_status": "excluded_pairing_mismatch",
                    "included_main_analysis": False,
                    "candidate_identity_parity": False,
                    "candidate_order_parity": False,
                    "judgement_input_parity": False,
                    "baseline_frozen_judgement_parity": False,
                    "pairing_failure_category": _pairing_failure_category(str(exc)),
                }
            )
            continue

        baseline_ids = list(case["baseline"]["returned_identity_keys"])
        experiment_ids = list(case["experiment"]["returned_identity_keys"])
        baseline_set = set(baseline_ids)
        experiment_set = set(experiment_ids)
        case.update(
            {
                **base,
                "analysis_status": "included_main_analysis",
                "included_main_analysis": True,
                "candidate_order_parity": True,
                "judgement_input_parity": True,
                "baseline_frozen_judgement_parity": True,
                "top_20_swaps": {
                    "admitted_count": len(experiment_set - baseline_set),
                    "removed_count": len(baseline_set - experiment_set),
                    "admitted_ids": sorted(experiment_set - baseline_set),
                    "removed_ids": sorted(baseline_set - experiment_set),
                },
            }
        )
        for item in candidates:
            item["successful_source_count"] = terminal["successful_source_count"]
            item["overlaps_prior_auto_dev_val"] = case_id in overlap_ids
        case_rows.append(case)
        candidate_rows.extend(candidates)
        included_rows.append(case)
        included_states.append(state)

    status_counts = Counter(str(item["analysis_status"]) for item in case_rows)
    if len(case_rows) != expected_record_count:
        raise ValueError("case closure failure")
    if status_counts["included_main_analysis"] != int(
        manifest["inclusion"]["expected_main_case_count"]
    ):
        raise ValueError("main analysis case count drift")
    if status_counts["excluded_no_successful_source"] != int(
        manifest["inclusion"]["expected_no_success_case_count"]
    ):
        raise ValueError("no-success case count drift")

    core = _aggregate_dataset(
        "autoscholar_record160", included_rows, candidate_rows, included_states
    )
    statistics = build_stratified_statistics(included_rows, manifest)
    transition_counts = Counter(str(item["transition"]) for item in candidate_rows)
    unmatched_admitted = [
        item
        for item in candidate_rows
        if item["transition"] == "benchmark_non_gold_admitted"
    ]
    recovered_gold = [
        item for item in candidate_rows if item["transition"] == "recovered_gold"
    ]
    source_unmatched = Counter(
        source for item in unmatched_admitted for source in item["sources"]
    )
    snapshot_after = _tree_sha256(snapshot_root)
    if snapshot_after != snapshot_before:
        raise ValueError("Snapshot tree changed during offline audit")
    prior = _read_json(_repo_path(manifest["prior_65_reference"]["path"]))
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "analysis": manifest["analysis"],
        "implementation_base_commit": manifest["implementation_base_commit"],
        "metric_version": METRIC_VERSION,
        "input": {
            "manifest_sha256": _sha256(manifest_file),
            "record_config_sha256": _sha256(config_path),
            "record_results_sha256": _sha256(results_path),
            "snapshot_tree_sha256": snapshot_before,
            "record_case_count": len(rows),
            "consumed_snapshot_key_count": input_key_count,
        },
        "closure": {
            "case_count": len(case_rows),
            "status_counts": dict(sorted(status_counts.items())),
            "source_case_terminals": dict(sorted(source_case_terminals.items())),
            "candidate_pairing_mismatch_count": status_counts[
                "excluded_pairing_mismatch"
            ],
        },
        "core_metrics": core,
        "paired_statistics": statistics,
        "admission_risk": {
            "known_gold_false_kill_recovered_count": len(recovered_gold),
            "qrels_unmatched_admitted_candidate_relation_count": len(
                unmatched_admitted
            ),
            "qrels_unmatched_admitted_by_candidate_source": dict(
                sorted(source_unmatched.items())
            ),
            "top_20_admitted_relation_count": sum(
                int(item["top_20_swaps"]["admitted_count"])
                for item in included_rows
            ),
            "top_20_removed_relation_count": sum(
                int(item["top_20_swaps"]["removed_count"])
                for item in included_rows
            ),
            "transition_counts": dict(sorted(transition_counts.items())),
            "interpretation": (
                "qrels-unmatched candidates are not reliable negative labels"
            ),
        },
        "prior_65_reference": {
            "input_query_count": prior["pairing"]["input_query_count"],
            "all_evaluable_query_count": prior["pairing"][
                "all_evaluable_query_count"
            ],
            "auto_dev": prior["datasets"]["auto_dev"],
            "auto_val": prior["datasets"]["auto_val"],
            "overlap_case_count": len(overlap_ids),
            "comparison_limit": (
                "the expanded Record prefix is order-biased and shares 15 Auto "
                "queries with the prior 65-query audit"
            ),
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "snapshot_tree_unchanged": True,
            "production_default": "off",
            "internal_metric_not_official_scorer": True,
        },
    }
    return case_rows, candidate_rows, aggregate


def resolve_record_terminals(
    row: Mapping[str, Any],
    *,
    store: Any,
    configured_sources: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve legacy Record call terminals from immutable Snapshot entries."""

    prepared = copy.deepcopy(dict(row))
    snapshots = prepared.get("stage_diagnostics", {}).get("snapshots", [])
    initial = next(
        (item for item in snapshots if item.get("stage") == "initial_retrieval"),
        None,
    )
    if initial is None:
        raise ValueError("missing initial retrieval diagnostic")
    statuses: dict[str, list[str]] = defaultdict(list)
    seen_keys: set[str] = set()
    terminal_records: list[dict[str, Any]] = []
    for call in initial.get("retrieval_calls") or []:
        if not call.get("logical_call_executed"):
            continue
        key = str(call.get("snapshot_key") or "")
        if not key:
            terminal = str(call.get("terminal_status") or "not_started")
            terminal_records.append(
                {
                    "source": str(call.get("source") or "unknown"),
                    "snapshot_key": None,
                    "status": terminal,
                }
            )
            continue
        if key in seen_keys:
            raise ValueError("duplicate Snapshot key in frozen Record case")
        seen_keys.add(key)
        entry = store.read_retrieval(key)
        source = str(call.get("source") or "")
        adapted_query = str(call.get("adapted_query") or "")
        if entry.source != source or entry.adapted_query != adapted_query:
            raise ValueError("frozen Snapshot request signature mismatch")
        recorded_terminal = call.get("terminal_status")
        if recorded_terminal not in (None, entry.status):
            raise ValueError("frozen Record terminal disagrees with Snapshot")
        call["terminal_status"] = entry.status
        statuses[source].append(entry.status)
        terminal_records.append(
            {"source": source, "snapshot_key": key, "status": entry.status}
        )
    source_states: dict[str, str] = {}
    for source in configured_sources:
        observed = statuses.get(source, [])
        if "success" in observed:
            source_states[source] = "success"
        elif "failed" in observed:
            source_states[source] = "failed"
        else:
            source_states[source] = "not_started"
    successful_source_count = sum(
        status == "success" for status in source_states.values()
    )
    return prepared, {
        "source_states": source_states,
        "successful_source_count": successful_source_count,
        "snapshot_key_count": len(seen_keys),
        "source_terminal_signature_sha256": _sha256_json(terminal_records),
    }


def build_stratified_statistics(
    case_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse the frozen paired-statistics implementation for fixed subgroups."""

    paired = [_paired_row(item) for item in case_rows]
    strata = {
        str(count): [
            item
            for item in paired
            if int(item["successful_source_count"]) == count
        ]
        for count in (1, 2, 3, 4)
    }
    overlap = [item for item in paired if item["overlaps_prior_auto_dev_val"]]
    new = [item for item in paired if not item["overlaps_prior_auto_dev_val"]]
    return {
        "protocol": {
            "metrics": list(manifest["metrics"]),
            "bootstrap": dict(manifest["bootstrap"]),
            "permutation_test": dict(manifest["permutation_test"]),
            "query_weighting": "equal",
        },
        "all_160": _analyze_group(
            paired,
            manifest,
            scope_name="record160",
            group_name="all_160",
        ),
        "by_successful_source_count": {
            name: _analyze_group(
                rows,
                manifest,
                scope_name="record160",
                group_name=f"successful_sources_{name}",
            )
            for name, rows in strata.items()
        },
        "prior_auto_dev_val_overlap": _analyze_group(
            overlap,
            manifest,
            scope_name="record160",
            group_name="prior_auto_dev_val_overlap",
        ),
        "new_excluding_prior_auto_dev_val": _analyze_group(
            new,
            manifest,
            scope_name="record160",
            group_name="new_excluding_prior_auto_dev_val",
        ),
    }


def write_expanded_lexical_audit(
    output_dir: str | Path,
    case_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    manifest_path: str | Path,
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_comparison.jsonl", case_rows)
    _write_jsonl(root / "candidate_diagnostics.jsonl", candidate_rows)
    _write_json(root / "aggregate.json", aggregate)
    _write_json(root / "manifest.json", _read_json(Path(manifest_path).resolve()))


def _paired_row(case: Mapping[str, Any]) -> dict[str, Any]:
    denominator = int(case["evaluable_gold_count"])
    candidate_recall = (
        int(case["candidate_gold_count"]) / denominator if denominator else 0.0
    )
    return {
        "successful_source_count": int(case["successful_source_count"]),
        "overlaps_prior_auto_dev_val": bool(
            case["overlaps_prior_auto_dev_val"]
        ),
        "metrics": {
            "candidate_recall": {
                "baseline": candidate_recall,
                "experiment": candidate_recall,
            },
            "recall_at_20": {
                "baseline": float(case["baseline"]["recall_at_20"]),
                "experiment": float(case["experiment"]["recall_at_20"]),
            },
            "f1_at_20": {
                "baseline": float(case["baseline"]["f1_at_20"]),
                "experiment": float(case["experiment"]["f1_at_20"]),
            },
        },
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("analysis") != "lexical_normalization_v1_record160_v1":
        raise ValueError("unexpected expanded lexical audit")
    if manifest.get("metric_version") != METRIC_VERSION:
        raise ValueError("expanded audit requires deduplicated gold v2")
    if manifest.get("policy", {}).get("default") != "off":
        raise ValueError("lexical normalization must remain default-off")
    invariants = manifest.get("frozen_invariants", {})
    if any(
        int(invariants.get(field, -1)) != 0
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise ValueError("expanded lexical audit must be zero-I/O")


def _validate_record_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if config.get("dataset") != "auto_scholar_query":
        raise ValueError("expanded audit requires AutoScholarQuery")
    if config.get("retrieval_mode") != "record-missing":
        raise ValueError("expanded audit requires the frozen Record input")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("query planning policy drift")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("ranking policy drift")
    if config.get("judgement_policy") != "current_rules":
        raise ValueError("judgement policy drift")
    if config.get("result_policy") != "highly_and_partial":
        raise ValueError("result filter policy drift")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError("Top-K drift")
    if config.get("sources") != manifest["frozen_record"]["sources"]:
        raise ValueError("source order drift")
    if (config.get("judgement_config") or {}).get(
        "lexical_normalization_policy"
    ) != "off":
        raise ValueError("frozen baseline did not use default-off lexical policy")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise ValueError("experimental retrieval strategy is enabled")


def _validate_hash(path: Path, expected: str) -> None:
    if _sha256(path) != str(expected):
        raise ValueError(f"frozen input hash drift:{path.name}")


def _pairing_failure_category(message: str) -> str:
    normalized = message.casefold()
    if "candidate" in normalized or "identity" in normalized:
        return "candidate_identity_or_order_mismatch"
    if "judgement" in normalized:
        return "baseline_judgement_mismatch"
    if "ranking" in normalized or "returned" in normalized:
        return "baseline_ranking_or_filter_mismatch"
    if "snapshot" in normalized:
        return "snapshot_reconstruction_mismatch"
    return "other_frozen_pairing_mismatch"


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _read_ordered_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
