"""Deterministic paired significance audit for frozen offline Benchmark results."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"
METRICS = ("candidate_recall", "recall_at_20", "f1_at_20")
DATASET_ORDER = ("scifact", "auto_dev", "auto_val")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def run_paired_significance_audit(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load frozen lexical Replay rows and compute the pre-registered audit."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    input_hashes = _validate_frozen_inputs(manifest)

    primary = manifest["inputs"]["primary_replay"]
    case_rows = _read_jsonl(_repo_path(primary["case_comparison"]["path"]))
    aggregate = _read_json(_repo_path(primary["aggregate"]["path"]))
    experiment_manifest = _read_json(
        _repo_path(manifest["inputs"]["frozen_experiment_manifest"]["path"])
    )
    query_context = _load_query_context(experiment_manifest)
    paired_rows = prepare_paired_rows(case_rows, query_context)
    _validate_against_frozen_aggregate(paired_rows, aggregate)

    output = analyze_paired_rows(paired_rows, manifest)
    output.update(
        {
            "schema_version": SCHEMA_VERSION,
            "analysis": manifest["analysis"],
            "implementation_base_commit": manifest["implementation_base_commit"],
            "manifest_sha256": _sha256(manifest_file),
            "input_sha256": input_hashes,
            "execution": {
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "input_mode": "frozen_replay_artifacts",
            },
            "interpretation_limits": list(manifest["warnings"]),
        }
    )
    return paired_rows, output


def prepare_paired_rows(
    case_rows: Sequence[Mapping[str, Any]],
    query_context: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build query pairs while explicitly excluding duplicates and invalid pairs."""

    seen_keys: set[tuple[str, str]] = set()
    normalized_queries: dict[str, list[tuple[str, str]]] = {}
    for row in case_rows:
        key = (str(row.get("dataset") or ""), str(row.get("case_id") or ""))
        if key in seen_keys:
            raise ValueError(f"duplicate case pair:{key[0]}:{key[1]}")
        seen_keys.add(key)
        context = query_context.get(key)
        if context is None:
            continue
        normalized_queries.setdefault(str(context["normalized_query"]), []).append(key)
    duplicate_keys = {
        key
        for keys in normalized_queries.values()
        if len(keys) > 1
        for key in keys
    }

    output: list[dict[str, Any]] = []
    dataset_positions = {name: index for index, name in enumerate(DATASET_ORDER)}
    ordered = sorted(
        case_rows,
        key=lambda item: (
            dataset_positions.get(str(item.get("dataset")), len(DATASET_ORDER)),
            int(item.get("case_order") or 0),
            str(item.get("case_id") or ""),
        ),
    )
    for row in ordered:
        dataset = str(row.get("dataset") or "")
        case_id = str(row.get("case_id") or "")
        key = (dataset, case_id)
        context = query_context.get(key)
        reasons: list[str] = []
        if context is None:
            reasons.append("missing_query_context")
        evaluable_gold_count = int(row.get("evaluable_gold_count") or 0)
        if evaluable_gold_count <= 0:
            reasons.append("identity_unavailable_gold")
        if key in duplicate_keys:
            reasons.append("duplicate_normalized_query")

        metrics: dict[str, dict[str, float | None]] = {}
        candidate_recall = (
            float(row.get("candidate_gold_count") or 0) / evaluable_gold_count
            if evaluable_gold_count > 0
            else None
        )
        metrics["candidate_recall"] = {
            "baseline": candidate_recall,
            "experiment": candidate_recall,
            "difference": 0.0 if candidate_recall is not None else None,
        }
        for metric in ("recall_at_20", "f1_at_20"):
            baseline = _optional_metric(row.get("baseline"), metric)
            experiment = _optional_metric(row.get("experiment"), metric)
            if baseline is None or experiment is None:
                reasons.append(f"missing_pair:{metric}")
                difference = None
            else:
                difference = experiment - baseline
            metrics[metric] = {
                "baseline": baseline,
                "experiment": experiment,
                "difference": difference,
            }

        all_evaluable = not reasons
        candidate_parity = bool(row.get("candidate_identity_parity"))
        terminal_consistent = bool(context and context.get("terminal_signature_sha256"))
        strict_reasons = list(reasons)
        if not candidate_parity:
            strict_reasons.append("candidate_identity_drift")
        if not terminal_consistent:
            strict_reasons.append("missing_shared_terminal_signature")
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
                "case_order": int(row.get("case_order") or 0),
                "case_id": case_id,
                "query_sha256": (
                    str(context["query_sha256"]) if context is not None else None
                ),
                "terminal_signature_sha256": (
                    str(context["terminal_signature_sha256"])
                    if context is not None
                    else None
                ),
                "source_terminal_counts": (
                    dict(context["source_terminal_counts"])
                    if context is not None
                    else {}
                ),
                "evaluable_gold_count": evaluable_gold_count,
                "candidate_identity_parity": candidate_parity,
                "shared_frozen_retrieval": terminal_consistent,
                "included_all_evaluable": all_evaluable,
                "included_strict_comparable": not strict_reasons,
                "all_evaluable_exclusion_reasons": sorted(set(reasons)),
                "strict_exclusion_reasons": sorted(set(strict_reasons)),
                "metrics": metrics,
            }
        )
    return output


def analyze_paired_rows(
    paired_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate dataset and equal-query pooled statistics for both scopes."""

    scopes: dict[str, Any] = {}
    for scope_name, inclusion_field in (
        ("all_evaluable", "included_all_evaluable"),
        ("strict_comparable", "included_strict_comparable"),
    ):
        included = [row for row in paired_rows if bool(row.get(inclusion_field))]
        datasets = {
            dataset: _analyze_group(
                [row for row in included if row.get("dataset") == dataset],
                manifest,
                scope_name=scope_name,
                group_name=dataset,
            )
            for dataset in DATASET_ORDER
        }
        datasets["combined_query_equal"] = _analyze_group(
            included,
            manifest,
            scope_name=scope_name,
            group_name="combined_query_equal",
        )
        scopes[scope_name] = {
            "included_query_count": len(included),
            "excluded_query_count": len(paired_rows) - len(included),
            "query_weighting": "equal",
            "datasets": datasets,
        }

    exclusion_counts = Counter(
        reason
        for row in paired_rows
        for reason in row.get("all_evaluable_exclusion_reasons", [])
    )
    strict_exclusion_counts = Counter(
        reason
        for row in paired_rows
        for reason in row.get("strict_exclusion_reasons", [])
    )
    return {
        "pairing": {
            "input_query_count": len(paired_rows),
            "dataset_query_counts": dict(
                sorted(Counter(str(row["dataset"]) for row in paired_rows).items())
            ),
            "all_evaluable_query_count": sum(
                bool(row["included_all_evaluable"]) for row in paired_rows
            ),
            "strict_comparable_query_count": sum(
                bool(row["included_strict_comparable"]) for row in paired_rows
            ),
            "all_evaluable_exclusions": dict(sorted(exclusion_counts.items())),
            "strict_exclusions": dict(sorted(strict_exclusion_counts.items())),
            "duplicate_query_count": exclusion_counts["duplicate_normalized_query"],
            "source_terminal_semantics": (
                "baseline and experiment reuse the same frozen retrieval per case"
            ),
        },
        "statistical_protocol": {
            "metrics": list(manifest["metrics"]),
            "bootstrap": dict(manifest["bootstrap"]),
            "permutation_test": dict(manifest["permutation_test"]),
            "power_planning": dict(manifest["power_planning"]),
            "combined_query_weighting": "equal",
        },
        "scopes": scopes,
    }


def paired_metric_statistics(
    baseline: Sequence[float],
    experiment: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
    permutation_seed: int,
    permutation_iterations: int,
    exact_nonzero_pair_limit: int,
    tie_tolerance: float,
    power_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute deterministic effect, CI, paired permutation and power planning."""

    if len(baseline) != len(experiment):
        raise ValueError("paired metric inputs have different lengths")
    if not baseline:
        raise ValueError("paired metric requires at least one query")
    differences = [right - left for left, right in zip(baseline, experiment, strict=True)]
    counts = Counter(
        "improved"
        if value > tie_tolerance
        else "regressed"
        if value < -tie_tolerance
        else "tied"
        for value in differences
    )
    sampled = _paired_bootstrap_means(
        differences, seed=bootstrap_seed, iterations=bootstrap_iterations
    )
    paired_sd = statistics.stdev(differences) if len(differences) >= 2 else None
    mean_difference = statistics.fmean(differences)
    standardized = (
        mean_difference / paired_sd if paired_sd is not None and paired_sd > 0 else None
    )
    permutation = _paired_permutation(
        differences,
        seed=permutation_seed,
        iterations=permutation_iterations,
        exact_nonzero_pair_limit=exact_nonzero_pair_limit,
        tolerance=tie_tolerance,
    )
    return {
        "query_count": len(differences),
        "baseline_mean": statistics.fmean(baseline),
        "experiment_mean": statistics.fmean(experiment),
        "mean_paired_difference": mean_difference,
        "median_paired_difference": statistics.median(differences),
        "paired_difference_sd": paired_sd,
        "standardized_mean_difference_dz": standardized,
        "outcomes": {
            name: counts[name] for name in ("improved", "tied", "regressed")
        },
        "bootstrap_ci_95": {
            "low": _percentile(sampled, 0.025),
            "high": _percentile(sampled, 0.975),
            "iterations": bootstrap_iterations,
            "method": "paired_query_resampling_percentile",
        },
        "paired_permutation": permutation,
        "power_planning": _power_planning(paired_sd, power_config),
    }


def write_paired_significance_audit(
    output_dir: str | Path,
    paired_rows: Sequence[Mapping[str, Any]],
    statistics_output: Mapping[str, Any],
    manifest_path: str | Path,
) -> None:
    """Write byte-stable audit artifacts."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "paired_queries.jsonl", paired_rows)
    _write_json(root / "statistics.json", statistics_output)
    manifest = _read_json(Path(manifest_path).expanduser().resolve())
    _write_json(root / "manifest.json", manifest)


def _analyze_group(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    scope_name: str,
    group_name: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not rows:
        return {"query_count": 0, "metrics": {}, "warning": "no_evaluable_pairs"}
    for metric in manifest["metrics"]:
        baseline = [float(row["metrics"][metric]["baseline"]) for row in rows]
        experiment = [float(row["metrics"][metric]["experiment"]) for row in rows]
        bootstrap_seed = _derived_seed(
            int(manifest["bootstrap"]["seed"]), scope_name, group_name, metric, "bootstrap"
        )
        permutation_seed = _derived_seed(
            int(manifest["permutation_test"]["seed"]),
            scope_name,
            group_name,
            metric,
            "permutation",
        )
        metrics[metric] = paired_metric_statistics(
            baseline,
            experiment,
            bootstrap_seed=bootstrap_seed,
            bootstrap_iterations=int(manifest["bootstrap"]["iterations"]),
            permutation_seed=permutation_seed,
            permutation_iterations=int(manifest["permutation_test"]["iterations"]),
            exact_nonzero_pair_limit=int(
                manifest["permutation_test"]["exact_nonzero_pair_limit"]
            ),
            tie_tolerance=float(manifest["comparison"]["tie_tolerance"]),
            power_config=manifest["power_planning"],
        )
    return {
        "query_count": len(rows),
        "metrics": metrics,
        "small_sample_warning": len(rows) < 30,
    }


def _paired_bootstrap_means(
    differences: Sequence[float], *, seed: int, iterations: int
) -> list[float]:
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    randomizer = random.Random(seed)
    count = len(differences)
    sampled = [
        statistics.fmean(differences[randomizer.randrange(count)] for _ in range(count))
        for _ in range(iterations)
    ]
    sampled.sort()
    return sampled


def _paired_permutation(
    differences: Sequence[float],
    *,
    seed: int,
    iterations: int,
    exact_nonzero_pair_limit: int,
    tolerance: float,
) -> dict[str, Any]:
    nonzero = [value for value in differences if abs(value) > tolerance]
    observed = abs(statistics.fmean(differences))
    if not nonzero:
        return {
            "p_value_two_sided": 1.0,
            "method": "exact_all_ties",
            "evaluated_permutations": 1,
            "nonzero_pair_count": 0,
        }
    denominator = len(differences)
    if len(nonzero) <= exact_nonzero_pair_limit:
        total = 1 << len(nonzero)
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            statistic = abs(sum(sign * value for sign, value in zip(signs, nonzero, strict=True)) / denominator)
            extreme += statistic >= observed - tolerance
        return {
            "p_value_two_sided": extreme / total,
            "method": "exact_sign_flip",
            "evaluated_permutations": total,
            "nonzero_pair_count": len(nonzero),
        }
    if iterations < 100:
        raise ValueError("permutation iterations must be at least 100")
    randomizer = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        statistic = abs(
            sum(value if randomizer.getrandbits(1) else -value for value in nonzero)
            / denominator
        )
        extreme += statistic >= observed - tolerance
    return {
        "p_value_two_sided": (extreme + 1) / (iterations + 1),
        "method": "monte_carlo_sign_flip",
        "evaluated_permutations": iterations,
        "nonzero_pair_count": len(nonzero),
    }


def _power_planning(
    paired_sd: float | None, config: Mapping[str, Any]
) -> dict[str, Any]:
    mde = float(config["minimum_detectable_absolute_lift"])
    alpha = float(config["alpha"])
    target_power = float(config["target_power"])
    future_n = int(config["future_validation_query_count"])
    base = {
        "minimum_detectable_absolute_lift": mde,
        "alpha": alpha,
        "target_power": target_power,
        "future_validation_query_count": future_n,
        "method": config["method"],
    }
    if paired_sd is None:
        return {**base, "available": False, "reason": "fewer_than_two_pairs"}
    if paired_sd <= 0:
        return {**base, "available": False, "reason": "zero_observed_variance"}
    normal = statistics.NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(target_power)
    required = max(2, math.ceil(((z_alpha + z_power) * paired_sd / mde) ** 2))
    noncentrality = mde * math.sqrt(future_n) / paired_sd
    future_power = (
        1 - normal.cdf(z_alpha - noncentrality)
        + normal.cdf(-z_alpha - noncentrality)
    )
    return {
        **base,
        "available": True,
        "observed_paired_difference_sd": paired_sd,
        "required_query_count": required,
        "estimated_power_at_future_query_count": future_power,
    }


def _load_query_context(
    experiment_manifest: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in experiment_manifest["frozen_inputs"]:
        dataset = str(spec["label"])
        run_dir = _repo_path(spec["run_dir"])
        for row in _read_jsonl(run_dir / "results.jsonl"):
            case_id = str(row.get("case_id") or "")
            query = str(row.get("query") or "")
            initial = next(
                (
                    stage
                    for stage in row.get("stage_diagnostics", {}).get("snapshots", [])
                    if stage.get("stage") == "initial_retrieval"
                ),
                None,
            )
            if initial is None:
                terminal_records: list[dict[str, Any]] = []
            else:
                terminal_records = [
                    {
                        "source": call.get("source"),
                        "origin_subquery": call.get("origin_subquery"),
                        "adapted_query": call.get("adapted_query"),
                        "logical_call_executed": call.get("logical_call_executed"),
                        "terminal_status": call.get("terminal_status"),
                        "snapshot_key": call.get("snapshot_key"),
                        "snapshot_hit": call.get("snapshot_hit"),
                        "snapshot_provenance": call.get("snapshot_provenance"),
                    }
                    for call in initial.get("retrieval_calls", [])
                ]
            key = (dataset, case_id)
            if key in output:
                raise ValueError(f"duplicate frozen result:{dataset}:{case_id}")
            output[key] = {
                "normalized_query": _normalize_query(query),
                "query_sha256": _sha256_text(query),
                "terminal_signature_sha256": (
                    _sha256_json(terminal_records) if initial is not None else None
                ),
                "source_terminal_counts": dict(
                    sorted(
                        Counter(
                            f"{item.get('source')}:{item.get('terminal_status')}"
                            for item in terminal_records
                        ).items()
                    )
                ),
            }
    return output


def _validate_against_frozen_aggregate(
    rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]
) -> None:
    for dataset in DATASET_ORDER:
        included = [
            row
            for row in rows
            if row["dataset"] == dataset and row["included_all_evaluable"]
        ]
        expected = aggregate["datasets"][dataset]
        if len(included) != int(expected["evaluable_case_count"]):
            raise ValueError(f"evaluable case count drift:{dataset}")
        candidate = statistics.fmean(
            float(row["metrics"]["candidate_recall"]["baseline"])
            for row in included
        )
        baseline_recall = statistics.fmean(
            float(row["metrics"]["recall_at_20"]["baseline"]) for row in included
        )
        experiment_recall = statistics.fmean(
            float(row["metrics"]["recall_at_20"]["experiment"])
            for row in included
        )
        if not math.isclose(candidate, float(expected["candidate_recall"]), abs_tol=1e-12):
            raise ValueError(f"candidate recall drift:{dataset}")
        if not math.isclose(
            baseline_recall, float(expected["baseline"]["recall_at_20"]), abs_tol=1e-12
        ):
            raise ValueError(f"baseline recall drift:{dataset}")
        if not math.isclose(
            experiment_recall,
            float(expected["experiment"]["recall_at_20"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"experiment recall drift:{dataset}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("analysis") != "lexical_normalization_v1_paired_significance":
        raise ValueError("unexpected significance manifest")
    if tuple(manifest.get("metrics") or ()) != METRICS:
        raise ValueError("paired significance metrics drifted")
    if manifest.get("combined_scope", {}).get("query_weighting") != "equal":
        raise ValueError("combined result must weight queries equally")
    if int(manifest.get("bootstrap", {}).get("iterations") or 0) < 100:
        raise ValueError("bootstrap iterations are not frozen")
    if int(manifest.get("permutation_test", {}).get("iterations") or 0) < 100:
        raise ValueError("permutation iterations are not frozen")


def _validate_frozen_inputs(manifest: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for group_name, group in manifest["inputs"].items():
        if group_name == "frozen_experiment_manifest":
            groups = {group_name: group}
        else:
            groups = {f"{group_name}.{name}": value for name, value in group.items()}
        for label, spec in groups.items():
            path = _repo_path(spec["path"])
            actual = _sha256(path)
            if actual != spec["sha256"]:
                raise ValueError(f"frozen input hash mismatch:{label}")
            output[label] = actual
    primary = manifest["inputs"]["primary_replay"]
    reproducibility = manifest["inputs"]["reproducibility_replay"]
    for name in ("aggregate", "case_comparison"):
        if primary[name]["sha256"] != reproducibility[name]["sha256"]:
            raise ValueError(f"frozen Replay pair differs:{name}")
    aggregate = _read_json(_repo_path(primary["aggregate"]["path"]))
    execution = aggregate.get("execution") or {}
    if any(int(execution.get(name) or 0) for name in (
        "network_request_count", "llm_request_count", "snapshot_write_count"
    )):
        raise ValueError("frozen lexical audit was not zero-I/O")
    return dict(sorted(output.items()))


def _optional_metric(value: Any, metric: str) -> float | None:
    if not isinstance(value, Mapping) or value.get(metric) is None:
        return None
    result = float(value[metric])
    if not math.isfinite(result) or result < 0 or result > 1:
        raise ValueError(f"invalid paired metric:{metric}")
    return result


def _normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
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


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
