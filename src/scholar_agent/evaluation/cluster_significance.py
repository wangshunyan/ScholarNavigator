"""Cluster-aware paired inference for frozen lexical-normalization results."""

from __future__ import annotations

import hashlib
import json
import math
import random
import socket
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.evaluation.current_rules_regression import compare_profiles


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
ANALYSIS_NAME = "lexical_normalization_v1_cluster_significance_v1"
GATE_NAME = "lexical_normalization_cluster_significance_regression"
SCHEMA_VERSION = "1"
METRICS = ("candidate_recall", "recall_at_20", "f1_at_20")
VIEWS = (
    "record160_full",
    "record160_decontaminated",
    "existing65_full",
    "existing65_decontaminated",
)


class ClusterSignificanceError(RuntimeError):
    """Raised when the frozen cluster-aware statistical contract is invalid."""


def run_cluster_significance_audit(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest, require_baseline=False)
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0}
    with _forbid_network(attempts):
        assignments = _read_jsonl(_repo_path(manifest["inputs"]["frozen_query_assignments"]["path"]))
        components = _read_jsonl(_repo_path(manifest["inputs"]["frozen_components"]["path"]))
        diagnostics = _read_jsonl(_repo_path(manifest["inputs"]["frozen_metric_diagnostics"]["path"]))
        small_cases = _read_jsonl(_repo_path(manifest["inputs"]["small65_cases"]["path"]))
        record_cases = _read_jsonl(_repo_path(manifest["inputs"]["record160_cases"]["path"]))
        historical_small = _read_json(
            _repo_path(manifest["inputs"]["historical_query_significance"]["path"])
        )
        historical_record = _read_json(
            _repo_path(
                manifest["inputs"]["historical_record160_significance"]["path"]
            )
        )
        query_rows = prepare_cluster_query_rows(
            assignments, diagnostics, small_cases, record_cases
        )
        cluster_rows, statistics_output = analyze_cluster_queries(
            query_rows, components, manifest
        )
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "input_mode": "frozen_replay_and_frozen_component_assignments",
        "components_recomputed": False,
    }
    if any(
        int(execution[field])
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise ClusterSignificanceError(f"offline invariant failed:{execution}")
    statistics_output.update(
        {
            "schema_version": SCHEMA_VERSION,
            "analysis": ANALYSIS_NAME,
            "manifest_protocol_sha256": _manifest_protocol_sha256(manifest),
            "execution": execution,
            "historical_query_level_references": {
                name: dict(manifest["inputs"][name])
                for name in (
                    "historical_query_significance",
                    "historical_record160_significance",
                )
            },
            "historical_query_level_results_read_only": {
                "existing65_full": historical_small["pooled_query_equal"],
                "record160_full": historical_record["paired_statistics_all_160"],
            },
            "interpretation_limits": list(manifest["warnings"]),
        }
    )
    _validate_closure(query_rows, cluster_rows, statistics_output)
    return query_rows, cluster_rows, statistics_output


def prepare_cluster_query_rows(
    assignments: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    small_cases: Sequence[Mapping[str, Any]],
    record_cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignment_by_id = {str(item["query_id"]): item for item in assignments}
    if len(assignment_by_id) != len(assignments):
        raise ClusterSignificanceError("duplicate frozen query assignment")
    diagnostics_by_key = {
        (str(item["scope"]), str(item["case_id"])): item for item in diagnostics
    }
    if len(diagnostics_by_key) != len(diagnostics):
        raise ClusterSignificanceError("duplicate frozen metric diagnostic")
    cases_by_scope = {
        "existing65": {str(item["case_id"]): item for item in small_cases},
        "record160": {str(item["case_id"]): item for item in record_cases},
    }
    rows: list[dict[str, Any]] = []
    for scope in ("existing65", "record160"):
        for case_id, case in sorted(
            cases_by_scope[scope].items(),
            key=lambda item: (int(item[1].get("case_order") or 0), item[0]),
        ):
            diagnostic = diagnostics_by_key.get((scope, case_id))
            if diagnostic is None:
                raise ClusterSignificanceError(f"missing frozen metric diagnostic:{scope}:{case_id}")
            dataset = str(case["dataset"])
            if dataset.startswith("auto") or dataset == "autoscholar_record160":
                assignment = assignment_by_id.get(case_id)
                if assignment is None:
                    raise ClusterSignificanceError(f"missing frozen component:{case_id}")
                component_id = str(assignment["component_id"])
                if component_id != str(diagnostic["component_id"]):
                    raise ClusterSignificanceError(f"component assignment drift:{case_id}")
                external_singleton = False
            elif dataset == "scifact":
                component_id = "external:scifact:" + hashlib.sha256(
                    case_id.encode("utf-8")
                ).hexdigest()[:20]
                external_singleton = True
            else:
                raise ClusterSignificanceError(f"unknown frozen dataset:{dataset}")
            gold_count = int(case.get("evaluable_gold_count") or 0)
            metrics: dict[str, dict[str, float | None]] = {}
            candidate = _optional_float(diagnostic.get("candidate_recall"))
            metrics["candidate_recall"] = _metric_pair(candidate, candidate)
            for metric in ("recall_at_20", "f1_at_20"):
                metrics[metric] = _metric_pair(
                    _optional_float((diagnostic.get("baseline") or {}).get(metric)),
                    _optional_float((diagnostic.get("experiment") or {}).get(metric)),
                )
            evaluable = gold_count > 0 and all(
                item["difference"] is not None for item in metrics.values()
            )
            exclusion_reasons: list[str] = []
            if not bool(diagnostic["included_full"]):
                exclusion_reasons.append(str(diagnostic.get("exclusion_reason") or "not_in_full_view"))
            if gold_count <= 0:
                exclusion_reasons.append("identity_unavailable_gold")
            if not all(item["difference"] is not None for item in metrics.values()):
                exclusion_reasons.append("missing_paired_metric")
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": scope,
                    "dataset": dataset,
                    "case_id": case_id,
                    "component_id": component_id,
                    "external_singleton_component": external_singleton,
                    "evaluable_gold_count": gold_count,
                    "included_full": bool(diagnostic["included_full"]),
                    "included_decontaminated": bool(
                        diagnostic["included_decontaminated"]
                    ),
                    "evaluable_pair": evaluable,
                    "exclusion_reasons": sorted(set(exclusion_reasons)),
                    "metrics": metrics,
                }
            )
    return rows


def analyze_cluster_queries(
    query_rows: Sequence[Mapping[str, Any]],
    full_components: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cluster_rows: list[dict[str, Any]] = []
    views: dict[str, Any] = {}
    for view in VIEWS:
        scope = "record160" if view.startswith("record160") else "existing65"
        inclusion = (
            "included_decontaminated" if view.endswith("decontaminated") else "included_full"
        )
        input_rows = [row for row in query_rows if row["scope"] == scope]
        included = [
            row for row in input_rows if row[inclusion] and row["evaluable_pair"]
        ]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in included:
            grouped[str(row["component_id"])].append(row)
        current_clusters = _build_cluster_rows(view, grouped)
        cluster_rows.extend(current_clusters)
        views[view] = _analyze_view(view, input_rows, included, current_clusters, manifest)
    power = _power_planning(views, full_components, manifest)
    return cluster_rows, {
        "statistical_protocol": {
            "metrics": list(METRICS),
            "bootstrap": dict(manifest["bootstrap"]),
            "permutation_test": dict(manifest["permutation_test"]),
            "cluster_assignment": dict(manifest["cluster_assignment"]),
            "large_cluster_sensitivity": dict(manifest["large_cluster_sensitivity"]),
            "power_planning": dict(manifest["power_planning"]),
        },
        "views": views,
        "power_planning": power,
    }


def cluster_metric_statistics(
    differences: Sequence[float],
    baseline: Sequence[float],
    experiment: Sequence[float],
    *,
    bootstrap_seed: int,
    permutation_seed: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    tie_tolerance: float,
    unit_label: str = "component",
) -> dict[str, Any]:
    if not differences or len(differences) != len(baseline) or len(baseline) != len(experiment):
        raise ValueError("cluster metric inputs must be non-empty and aligned")
    bootstrap = _bootstrap_means(
        differences, seed=bootstrap_seed, iterations=bootstrap_iterations
    )
    permutation = _sign_flip(
        differences,
        seed=permutation_seed,
        iterations=permutation_iterations,
        tolerance=tie_tolerance,
        unit_label=unit_label,
    )
    counts = Counter(
        "improved"
        if value > tie_tolerance
        else "regressed"
        if value < -tie_tolerance
        else "tied"
        for value in differences
    )
    difference_sd = statistics.stdev(differences) if len(differences) >= 2 else None
    mean_difference = statistics.fmean(differences)
    return {
        "unit_count": len(differences),
        "baseline_mean": statistics.fmean(baseline),
        "experiment_mean": statistics.fmean(experiment),
        "mean_paired_difference": mean_difference,
        "median_paired_difference": statistics.median(differences),
        "paired_difference_sd": difference_sd,
        "standardized_mean_difference_dz": (
            mean_difference / difference_sd
            if difference_sd is not None and difference_sd > 0
            else None
        ),
        "outcomes": {name: counts[name] for name in ("improved", "tied", "regressed")},
        "bootstrap_ci_95": {
            "low": _percentile(bootstrap, 0.025),
            "high": _percentile(bootstrap, 0.975),
            "iterations": bootstrap_iterations,
            "method": "component_resampling_percentile",
        },
        "cluster_sign_flip": permutation,
    }


def write_cluster_significance_audit(
    output_dir: str | Path,
    query_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    statistics_output: Mapping[str, Any],
    manifest_path: str | Path,
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "paired_queries.jsonl", query_rows)
    _write_jsonl(root / "paired_components.jsonl", cluster_rows)
    _write_json(root / "statistics.json", statistics_output)
    _write_json(root / "manifest.json", _read_json(Path(manifest_path).resolve()))


def check_cluster_significance_regression(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path).expanduser().resolve())
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    drifts: list[dict[str, Any]] = []
    try:
        _validate_manifest(manifest, require_baseline=True)
        observed_values = run_cluster_significance_audit(manifest_path)
        names = ("paired_queries", "paired_components", "statistics")
        for name, observed in zip(names, observed_values, strict=True):
            spec = manifest["baseline"]
            path = _repo_path(spec[f"{name}_path"])
            actual_hash = sha256_file(path)
            if actual_hash != str(spec[f"{name}_sha256"]):
                drifts.append(
                    {
                        "kind": "baseline_fingerprint_drift",
                        "path": f"$.baseline.{name}",
                        "expected": spec[f"{name}_sha256"],
                        "observed": actual_hash,
                    }
                )
            expected = _read_json(path) if name == "statistics" else _read_jsonl(path)
            drifts.extend(compare_profiles({name: expected}, {name: observed}, max_diffs=100))
        write_cluster_significance_audit(output / "observed", *observed_values, manifest_path)
    except (ClusterSignificanceError, ValueError, KeyError) as exc:
        drifts.append(
            {
                "kind": "input_protocol_or_cluster_drift",
                "path": "$",
                "expected": "frozen cluster-aware statistical contract",
                "observed": str(exc),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not drifts,
        "drift_count": len(drifts),
        "drifts": drifts,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
    }
    _write_json(output / "regression_report.json", report)
    return report


def _build_cluster_rows(
    view: str, grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cluster_count = len(grouped)
    query_count = sum(len(items) for items in grouped.values())
    for component_id in sorted(grouped):
        queries = sorted(grouped[component_id], key=lambda item: str(item["case_id"]))
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            baseline = statistics.fmean(
                float(row["metrics"][metric]["baseline"]) for row in queries
            )
            experiment = statistics.fmean(
                float(row["metrics"][metric]["experiment"]) for row in queries
            )
            difference = experiment - baseline
            metrics[metric] = {
                "baseline_mean": baseline,
                "experiment_mean": experiment,
                "difference": difference,
                "cluster_equal_contribution": difference / cluster_count,
                "query_equal_contribution": sum(
                    float(row["metrics"][metric]["difference"]) for row in queries
                )
                / query_count,
            }
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "view": view,
                "component_id": component_id,
                "query_count": len(queries),
                "gold_count": sum(int(row["evaluable_gold_count"]) for row in queries),
                "dataset_counts": dict(
                    sorted(Counter(str(row["dataset"]) for row in queries).items())
                ),
                "query_ids": [str(row["case_id"]) for row in queries],
                "metrics": metrics,
            }
        )
    return rows


def _analyze_view(
    view: str,
    input_rows: Sequence[Mapping[str, Any]],
    included: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    query_metrics: dict[str, Any] = {}
    for metric in METRICS:
        cluster_differences = [float(row["metrics"][metric]["difference"]) for row in clusters]
        cluster_baseline = [float(row["metrics"][metric]["baseline_mean"]) for row in clusters]
        cluster_experiment = [float(row["metrics"][metric]["experiment_mean"]) for row in clusters]
        metrics[metric] = cluster_metric_statistics(
            cluster_differences,
            cluster_baseline,
            cluster_experiment,
            bootstrap_seed=_derived_seed(int(manifest["bootstrap"]["seed"]), view, metric, "cluster_bootstrap"),
            permutation_seed=_derived_seed(int(manifest["permutation_test"]["seed"]), view, metric, "cluster_sign_flip"),
            bootstrap_iterations=int(manifest["bootstrap"]["iterations"]),
            permutation_iterations=int(manifest["permutation_test"]["iterations"]),
            tie_tolerance=float(manifest["comparison"]["tie_tolerance"]),
            unit_label="component",
        )
        metrics[metric]["large_cluster_sensitivity"] = _large_cluster_sensitivity(
            clusters, metric
        )
        query_differences = [float(row["metrics"][metric]["difference"]) for row in included]
        query_baseline = [float(row["metrics"][metric]["baseline"]) for row in included]
        query_experiment = [float(row["metrics"][metric]["experiment"]) for row in included]
        query_metrics[metric] = cluster_metric_statistics(
            query_differences,
            query_baseline,
            query_experiment,
            bootstrap_seed=_derived_seed(int(manifest["bootstrap"]["seed"]), view, metric, "query_bootstrap"),
            permutation_seed=_derived_seed(int(manifest["permutation_test"]["seed"]), view, metric, "query_sign_flip"),
            bootstrap_iterations=int(manifest["query_level_comparator"]["bootstrap_iterations"]),
            permutation_iterations=int(manifest["query_level_comparator"]["permutation_iterations"]),
            tie_tolerance=float(manifest["comparison"]["tie_tolerance"]),
            unit_label="query",
        )
        query_metrics[metric]["unit"] = "query_comparator_only"
    included_keys = {(str(row["scope"]), str(row["case_id"])) for row in included}
    exclusion_counts: Counter[str] = Counter()
    decontaminated = view.endswith("decontaminated")
    for row in input_rows:
        key = (str(row["scope"]), str(row["case_id"]))
        if key in included_keys:
            continue
        reasons = set(str(reason) for reason in row["exclusion_reasons"])
        if decontaminated and not bool(row["included_decontaminated"]):
            if bool(row["included_full"]):
                reasons.add("cross_stratum_contaminated_component")
            elif not reasons:
                reasons.add("not_in_decontaminated_view")
        if not reasons:
            reasons.add("not_evaluable_pair")
        exclusion_counts.update(reasons)
    return {
        "input_query_count": len(input_rows),
        "included_query_count": len(included),
        "excluded_query_count": len(input_rows) - len(included),
        "excluded_reason_counts": dict(sorted(exclusion_counts.items())),
        "component_count": len(clusters),
        "gold_count": sum(int(row["evaluable_gold_count"]) for row in included),
        "all_included_queries_have_one_component": len(included)
        == sum(int(row["query_count"]) for row in clusters),
        "component_equal_metrics": metrics,
        "historical_query_equal_comparator": query_metrics,
    }


def _large_cluster_sensitivity(
    clusters: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    largest = sorted(
        clusters, key=lambda row: (-int(row["query_count"]), str(row["component_id"]))
    )[0]
    differences = [float(row["metrics"][metric]["difference"]) for row in clusters]
    contributions = [
        float(row["metrics"][metric]["cluster_equal_contribution"]) for row in clusters
    ]
    absolute_total = sum(abs(value) for value in contributions)
    without = [
        float(row["metrics"][metric]["difference"])
        for row in clusters
        if str(row["component_id"]) != str(largest["component_id"])
    ]
    shares = (
        [abs(value) / absolute_total for value in contributions]
        if absolute_total > 0
        else []
    )
    return {
        "largest_query_count_component_id": largest["component_id"],
        "largest_query_count": largest["query_count"],
        "largest_component_difference": largest["metrics"][metric]["difference"],
        "mean_difference_without_largest_query_count_component": (
            statistics.fmean(without) if without else None
        ),
        "maximum_absolute_contribution_share": max(shares) if shares else None,
        "absolute_contribution_herfindahl": (
            sum(value * value for value in shares) if shares else None
        ),
        "full_mean_difference": statistics.fmean(differences),
    }


def _power_planning(
    views: Mapping[str, Any],
    full_components: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    config = manifest["power_planning"]
    recall = views["record160_full"]["component_equal_metrics"]["recall_at_20"]
    sd = recall["paired_difference_sd"]
    query_count = sum(int(row["query_count"]) for row in full_components)
    component_count = len(full_components)
    structural_ess = query_count * query_count / sum(
        int(row["query_count"]) ** 2 for row in full_components
    )
    base = {
        "method": config["method"],
        "minimum_detectable_absolute_recall_lift": float(
            config["minimum_detectable_absolute_recall_lift"]
        ),
        "alpha": float(config["alpha"]),
        "target_power": float(config["target_power"]),
        "future_query_count": query_count,
        "future_component_count": component_count,
        "structural_effective_sample_size": structural_ess,
        "pilot_view": "record160_full",
    }
    if sd is None or float(sd) <= 0:
        return {**base, "available": False, "reason": "zero_or_unavailable_pilot_variance"}
    sd_value = float(sd)
    delta = float(config["minimum_detectable_absolute_recall_lift"])
    alpha = float(config["alpha"])
    target = float(config["target_power"])
    normal = statistics.NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(target)
    required = max(2, math.ceil(((z_alpha + z_power) * sd_value / delta) ** 2))
    noncentrality = delta * math.sqrt(component_count) / sd_value
    estimated = 1 - normal.cdf(z_alpha - noncentrality) + normal.cdf(
        -z_alpha - noncentrality
    )
    return {
        **base,
        "available": True,
        "pilot_component_difference_sd": sd_value,
        "required_component_count_for_target_power": required,
        "estimated_power_at_frozen_full1000_component_count": estimated,
    }


def _metric_pair(
    baseline: float | None, experiment: float | None
) -> dict[str, float | None]:
    return {
        "baseline": baseline,
        "experiment": experiment,
        "difference": (
            experiment - baseline
            if baseline is not None and experiment is not None
            else None
        ),
    }


def _bootstrap_means(
    values: Sequence[float], *, seed: int, iterations: int
) -> list[float]:
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    randomizer = random.Random(seed)
    count = len(values)
    output = [
        sum(values[randomizer.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    ]
    output.sort()
    return output


def _sign_flip(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int,
    tolerance: float,
    unit_label: str,
) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("sign-flip iterations must be at least 100")
    observed = abs(statistics.fmean(values))
    nonzero = [value for value in values if abs(value) > tolerance]
    if not nonzero:
        return {
            "p_value_two_sided": 1.0,
            "method": f"monte_carlo_{unit_label}_sign_flip_all_ties",
            "evaluated_permutations": iterations,
            "nonzero_unit_count": 0,
        }
    randomizer = random.Random(seed)
    denominator = len(values)
    extreme = 0
    for _ in range(iterations):
        statistic = abs(
            sum(value if randomizer.getrandbits(1) else -value for value in nonzero)
            / denominator
        )
        extreme += statistic >= observed - tolerance
    return {
        "p_value_two_sided": (extreme + 1) / (iterations + 1),
        "method": f"monte_carlo_{unit_label}_sign_flip",
        "evaluated_permutations": iterations,
        "nonzero_unit_count": len(nonzero),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1 - fraction) + values[upper] * fraction)


def _derived_seed(master: int, *parts: str) -> int:
    payload = ":".join((str(master), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1:
        raise ClusterSignificanceError("invalid frozen metric")
    return result


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("analysis") != ANALYSIS_NAME:
        raise ClusterSignificanceError("unexpected cluster significance manifest")
    if tuple(manifest.get("metrics") or ()) != METRICS:
        raise ClusterSignificanceError("metric list drift")
    if int(manifest.get("bootstrap", {}).get("iterations", 0)) != 20000:
        raise ClusterSignificanceError("bootstrap iteration drift")
    if int(manifest.get("permutation_test", {}).get("iterations", 0)) != 20000:
        raise ClusterSignificanceError("sign-flip iteration drift")
    if manifest.get("cluster_assignment", {}).get("estimand") != "equal mean of component-level paired differences":
        raise ClusterSignificanceError("cluster estimand drift")
    for spec in manifest["inputs"].values():
        _validate_hash(_repo_path(spec["path"]), str(spec["sha256"]))
    implementation = manifest.get("implementation")
    if implementation:
        _validate_hash(_repo_path(implementation["path"]), str(implementation["sha256"]))
    if require_baseline and "baseline" not in manifest:
        raise ClusterSignificanceError("cluster significance baseline missing")


def _validate_closure(
    query_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    statistics_output: Mapping[str, Any],
) -> None:
    if len(query_rows) != 227:
        raise ClusterSignificanceError("frozen query row closure failure")
    if len({(row["scope"], row["case_id"]) for row in query_rows}) != 227:
        raise ClusterSignificanceError("duplicate frozen query row")
    for view in VIEWS:
        summary = statistics_output["views"][view]
        rows = [row for row in cluster_rows if row["view"] == view]
        if len(rows) != int(summary["component_count"]):
            raise ClusterSignificanceError(f"component count closure failure:{view}")
        if sum(int(row["query_count"]) for row in rows) != int(summary["included_query_count"]):
            raise ClusterSignificanceError(f"query/component closure failure:{view}")
        if not summary["all_included_queries_have_one_component"]:
            raise ClusterSignificanceError(f"component uniqueness failure:{view}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ClusterSignificanceError(f"frozen input hash drift:{path.name}")


def _manifest_protocol_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the preregistered protocol without its self-referential baseline pointers."""

    protocol = {key: value for key, value in manifest.items() if key != "baseline"}
    payload = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_signature(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise ClusterSignificanceError("network access forbidden")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
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
