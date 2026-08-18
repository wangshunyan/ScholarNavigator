"""Deterministic blind human-annotation tooling for paired ranking audits.

The generator reconstructs frozen candidate lists without consulting gold or
qrels. Strategy membership is stored only in the private evaluator mapping;
public annotation rows contain query and paper text only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import (
    CURRENT_RULES_CONFIG,
    LEXICAL_NORMALIZATION_V1_CONFIG,
)
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.lexical_normalization_benchmark import (
    _identity_sequence,
    _read_json,
    _read_rows,
    _sha256,
    _validate_frozen_config,
)
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _reconstruct_candidates,
)
from scholar_agent.evaluation.metrics import canonical_paper_id
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


PACKAGE_SCHEMA_VERSION = "1"
LABELS = (
    "relevant",
    "partially_relevant",
    "not_relevant",
    "insufficient_information",
)
STRATEGIES = ("baseline", "experiment")
PUBLIC_FIELDS = ("sample_id", "query", "title", "abstract", "year")


def generate_precision_annotation_package(
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_package_manifest(manifest)
    universe, population = _build_universe(manifest, manifest_file.parent)
    selected = balanced_sample(
        universe,
        dataset_order=list(manifest["sampling"]["dataset_order"]),
        stratum_order=list(manifest["sampling"]["stratum_order"]),
        maximum=int(manifest["sampling"]["maximum_query_paper_pairs"]),
        seed=str(manifest["sampling"]["seed"]),
    )
    selected = sorted(
        selected,
        key=lambda item: _digest(
            str(manifest["sampling"]["seed"]),
            "display",
            str(item["dataset"]),
            str(item["case_id"]),
            str(item["paper_identity"]),
        ),
    )
    cell_population = Counter(
        (str(item["dataset"]), str(item["stratum"])) for item in universe
    )
    cell_sample = Counter(
        (str(item["dataset"]), str(item["stratum"])) for item in selected
    )
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        sample_id = f"SPAR-LX-{index:04d}"
        public_rows.append(
            {
                "sample_id": sample_id,
                "query": item["query"],
                "title": item["paper"]["title"],
                "abstract": item["paper"]["abstract"],
                "year": item["paper"]["year"],
            }
        )
        cell = (str(item["dataset"]), str(item["stratum"]))
        private_rows.append(
            {
                "sample_id": sample_id,
                "dataset": item["dataset"],
                "case_id": item["case_id"],
                "paper_identity": item["paper_identity"],
                "stratum": item["stratum"],
                "baseline_returned": item["baseline_returned"],
                "experiment_returned": item["experiment_returned"],
                "baseline_rank": item["baseline_rank"],
                "experiment_rank": item["experiment_rank"],
                "cell_population_count": cell_population[cell],
                "cell_sample_count": cell_sample[cell],
                "inclusion_probability": cell_sample[cell]
                / cell_population[cell],
            }
        )
    assert_blinded_rows(public_rows, manifest["forbidden_public_fields"])
    annotation_one = _annotation_template(public_rows, "annotator_1")
    annotation_two = _annotation_template(public_rows, "annotator_2")
    adjudication = [
        {
            "sample_id": item["sample_id"],
            "adjudicator_id": "",
            "final_label": None,
            "rationale": "",
        }
        for item in public_rows
    ]
    private_mapping = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package": manifest["package"],
        "top_k": int(manifest["scope"]["top_k"]),
        "population": population,
        "samples": private_rows,
    }
    summary = _package_summary(
        manifest=manifest,
        universe=universe,
        selected=private_rows,
        population=population,
    )
    pending_metrics = evaluate_annotations(
        private_mapping,
        annotation_one,
        annotation_two,
        adjudication,
    )
    return {
        "manifest": manifest,
        "blind_samples": public_rows,
        "annotation_schema": _annotation_schema(manifest),
        "annotator_1": annotation_one,
        "annotator_2": annotation_two,
        "adjudication": adjudication,
        "private_mapping": private_mapping,
        "summary": summary,
        "metrics": pending_metrics,
        "readme": _readme(summary),
    }


def write_precision_annotation_package(
    output: str | Path, package: Mapping[str, Any]
) -> None:
    root = Path(output).expanduser().resolve()
    public = root / "public"
    private = root / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    _write_json(root / "manifest.json", package["manifest"])
    _write_json(root / "summary.json", package["summary"])
    (root / "README.md").write_text(str(package["readme"]), encoding="utf-8")
    _write_jsonl(public / "blind_samples.jsonl", package["blind_samples"])
    _write_json(public / "annotation_schema.json", package["annotation_schema"])
    _write_jsonl(public / "annotator_1.jsonl", package["annotator_1"])
    _write_jsonl(public / "annotator_2.jsonl", package["annotator_2"])
    _write_jsonl(public / "adjudication.jsonl", package["adjudication"])
    _write_json(private / "mapping.json", package["private_mapping"])
    _write_json(private / "metrics_pending.json", package["metrics"])


def balanced_sample(
    universe: Sequence[dict[str, Any]],
    *,
    dataset_order: Sequence[str],
    stratum_order: Sequence[str],
    maximum: int,
    seed: str,
) -> list[dict[str, Any]]:
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    cells = [(dataset, stratum) for dataset in dataset_order for stratum in stratum_order]
    queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        values = [
            dict(item)
            for item in universe
            if (str(item["dataset"]), str(item["stratum"])) == cell
        ]
        queues[cell] = sorted(
            values,
            key=lambda item: _digest(
                seed,
                "selection",
                cell[0],
                cell[1],
                _digest(str(item["query"])),
                str(item["paper_identity"]),
            ),
        )
    selected: list[dict[str, Any]] = []
    offsets = {cell: 0 for cell in cells}
    while len(selected) < maximum:
        progressed = False
        for cell in cells:
            offset = offsets[cell]
            if offset >= len(queues[cell]):
                continue
            selected.append(queues[cell][offset])
            offsets[cell] += 1
            progressed = True
            if len(selected) == maximum:
                break
        if not progressed:
            break
    return selected


def merge_strategy_candidates(
    baseline: Sequence[Any], experiment: Sequence[Any]
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    profiles = []
    for strategy, candidates in (("baseline", baseline), ("experiment", experiment)):
        for rank, value in enumerate(candidates, start=1):
            paper = _paper(value)
            profile = build_identity_profile(paper)
            match_index = next(
                (
                    index
                    for index, existing in enumerate(profiles)
                    if identity_evidence_from_profiles(existing, profile).equivalent
                ),
                None,
            )
            if match_index is None:
                clusters.append(
                    {
                        "paper": paper.model_copy(deep=True),
                        "baseline_rank": None,
                        "experiment_rank": None,
                    }
                )
                profiles.append(profile)
                match_index = len(clusters) - 1
            else:
                merged = deduplicate_papers(
                    [clusters[match_index]["paper"], paper]
                )[0]
                clusters[match_index]["paper"] = merged
                profiles[match_index] = build_identity_profile(merged)
            key = f"{strategy}_rank"
            existing_rank = clusters[match_index][key]
            clusters[match_index][key] = (
                rank if existing_rank is None else min(existing_rank, rank)
            )
    for item in clusters:
        item["paper_identity"] = canonical_paper_id(item["paper"])
        if not item["paper_identity"]:
            raise ValueError("returned candidate lacks a stable unified identity")
    return clusters


def assert_blinded_rows(
    rows: Sequence[Mapping[str, Any]], forbidden_fields: Sequence[str]
) -> None:
    forbidden = {str(item).casefold() for item in forbidden_fields}
    expected = set(PUBLIC_FIELDS)
    for row in rows:
        if set(row) != expected:
            raise ValueError("blind row fields differ from the frozen public schema")
        leaked = {str(key).casefold() for key in _recursive_keys(row)} & forbidden
        if leaked:
            raise ValueError(f"forbidden blind fields present:{sorted(leaked)}")
        if not str(row["sample_id"]).startswith("SPAR-LX-"):
            raise ValueError("invalid public sample ID")


def validate_annotation_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_sample_ids: Sequence[str],
    *,
    label_field: str = "label",
) -> dict[str, str | None]:
    expected = set(expected_sample_ids)
    observed: dict[str, str | None] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in observed:
            raise ValueError("annotation sample IDs must be non-empty and unique")
        raw = row.get(label_field)
        if raw is not None and raw not in LABELS:
            raise ValueError(f"invalid annotation label:{raw}")
        observed[sample_id] = str(raw) if raw is not None else None
    if set(observed) != expected:
        raise ValueError("annotation sample IDs do not match the package")
    return observed


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    if len(first) != len(second):
        raise ValueError("Cohen's kappa inputs must have equal length")
    if not first:
        return None
    if any(item not in LABELS for item in [*first, *second]):
        raise ValueError("Cohen's kappa received an invalid label")
    count = len(first)
    observed = sum(left == right for left, right in zip(first, second)) / count
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] / count * second_counts[label] / count
        for label in LABELS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def evaluate_annotations(
    private_mapping: Mapping[str, Any],
    annotator_one: Sequence[Mapping[str, Any]],
    annotator_two: Sequence[Mapping[str, Any]],
    adjudication: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = list(private_mapping.get("samples") or [])
    sample_ids = [str(item["sample_id"]) for item in samples]
    first_annotators = {str(item.get("annotator_id") or "") for item in annotator_one}
    second_annotators = {str(item.get("annotator_id") or "") for item in annotator_two}
    if (
        len(first_annotators) != 1
        or len(second_annotators) != 1
        or "" in first_annotators
        or "" in second_annotators
        or first_annotators == second_annotators
    ):
        raise ValueError("two distinct non-empty annotator IDs are required")
    first = validate_annotation_rows(annotator_one, sample_ids)
    second = validate_annotation_rows(annotator_two, sample_ids)
    final = validate_annotation_rows(
        adjudication,
        sample_ids,
        label_field="final_label",
    )
    first_values = list(first.values())
    second_values = list(second.values())
    if all(item is None for item in [*first_values, *second_values]):
        return _pending_metrics("pending_human_labels", len(samples))
    if any(item is None for item in [*first_values, *second_values]):
        raise ValueError("both annotators must complete every sample before scoring")
    first_labels = [str(first[sample_id]) for sample_id in sample_ids]
    second_labels = [str(second[sample_id]) for sample_id in sample_ids]
    kappa = cohen_kappa(first_labels, second_labels)
    disagreements = [
        sample_id
        for sample_id in sample_ids
        if first[sample_id] != second[sample_id]
    ]
    resolved: dict[str, str] = {}
    missing_adjudications: list[str] = []
    for sample_id in sample_ids:
        if first[sample_id] == second[sample_id]:
            if final[sample_id] is not None:
                raise ValueError("adjudication must be blank when annotators agree")
            resolved[sample_id] = str(first[sample_id])
        elif final[sample_id] is None:
            missing_adjudications.append(sample_id)
        else:
            resolved[sample_id] = str(final[sample_id])
    base = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "annotation_status": (
            "pending_adjudication" if missing_adjudications else "complete"
        ),
        "sample_count": len(samples),
        "agreement": {
            "cohen_kappa": kappa,
            "agreement_count": len(samples) - len(disagreements),
            "disagreement_count": len(disagreements),
            "missing_adjudication_count": len(missing_adjudications),
        },
    }
    if missing_adjudications:
        return {
            **base,
            "metrics": None,
            "reason": "all disagreements require independent adjudication",
        }
    return {
        **base,
        "metrics": _annotation_metrics(private_mapping, resolved),
        "reason": None,
    }


def _build_universe(
    manifest: Mapping[str, Any], manifest_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_replay = manifest["source_replay"]
    replay_root = _resolve_repo_path(str(source_replay["run_dir"]), manifest_root)
    for name, expected in (
        ("aggregate.json", source_replay["aggregate_sha256"]),
        ("case_comparison.jsonl", source_replay["case_comparison_sha256"]),
        ("candidate_diagnostics.jsonl", source_replay["candidate_diagnostics_sha256"]),
    ):
        if _sha256(replay_root / name) != expected:
            raise ValueError(f"source Replay fingerprint mismatch:{name}")
    experiment_manifest = _read_json(
        _resolve_repo_path(str(manifest["source_experiment_manifest"]), manifest_root)
    )
    specs = {str(item["label"]): item for item in experiment_manifest["frozen_inputs"]}
    rank_threshold = int(manifest["sampling"]["significant_absolute_rank_change"])
    universe: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    for dataset in manifest["sampling"]["dataset_order"]:
        spec = specs[str(dataset)]
        run_root = _resolve_repo_path(str(spec["run_dir"]), manifest_root)
        snapshot_root = _resolve_repo_path(str(spec["snapshot_dir"]), manifest_root)
        config = _read_json(run_root / "config.json")
        _validate_frozen_config(config, spec)
        rows = _read_rows(run_root / "results.jsonl")
        store = SnapshotStore(snapshot_root)
        case_ids = [str(item) for item in config["case_ids"]]
        counts = Counter()
        for case_id in case_ids:
            row = rows[case_id]
            snapshots = {
                str(item["stage"]): item
                for item in row["stage_diagnostics"]["snapshots"]
            }
            candidates = _reconstruct_candidates(
                snapshots["initial_retrieval"],
                snapshots["initial_deduplicated"],
                config,
                store,
            )
            analysis = QueryAnalysis.model_validate(
                row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
            )
            baseline = judge_papers(
                analysis,
                candidates,
                use_llm=False,
                config=CURRENT_RULES_CONFIG,
            )
            experiment = judge_papers(
                analysis,
                candidates,
                use_llm=False,
                config=LEXICAL_NORMALIZATION_V1_CONFIG,
            )
            baseline_ranked = rerank_papers(analysis, baseline, top_k=len(candidates))
            experiment_ranked = rerank_papers(
                analysis, experiment, top_k=len(candidates)
            )
            top_k = int(config["top_k"])
            baseline_returned = select_ranked_results(
                {"ranked_papers": baseline_ranked[:top_k]},
                policy=str(config["result_policy"]),
            )
            experiment_returned = select_ranked_results(
                {"ranked_papers": experiment_ranked[:top_k]},
                policy=str(config["result_policy"]),
            )
            if _identity_sequence(baseline_returned) != _identity_sequence(
                snapshots["final_returned"].get("candidates") or []
            ):
                raise ValueError(f"default-off frozen result mismatch:{dataset}:{case_id}")
            pairs = merge_strategy_candidates(
                baseline_returned, experiment_returned
            )
            counts["baseline_returned_pair_count"] += len(baseline_returned)
            counts["experiment_returned_pair_count"] += len(experiment_returned)
            for pair in pairs:
                baseline_rank = pair["baseline_rank"]
                experiment_rank = pair["experiment_rank"]
                if baseline_rank is None:
                    stratum = "normalization_added"
                elif experiment_rank is None:
                    stratum = "baseline_only"
                elif abs(int(baseline_rank) - int(experiment_rank)) >= rank_threshold:
                    stratum = "shared_rank_shifted"
                else:
                    counts["shared_below_rank_threshold_count"] += 1
                    continue
                counts[f"eligible_{stratum}_count"] += 1
                paper = pair["paper"]
                universe.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "query": str(row["query"]),
                        "paper_identity": pair["paper_identity"],
                        "paper": {
                            "title": paper.title,
                            "abstract": paper.abstract,
                            "year": paper.year,
                        },
                        "stratum": stratum,
                        "baseline_returned": baseline_rank is not None,
                        "experiment_returned": experiment_rank is not None,
                        "baseline_rank": baseline_rank,
                        "experiment_rank": experiment_rank,
                    }
                )
        datasets[str(dataset)] = {
            "case_count": len(case_ids),
            **dict(sorted(counts.items())),
        }
    return universe, {
        "datasets": datasets,
        "eligible_pair_count": len(universe),
    }


def _annotation_metrics(
    private_mapping: Mapping[str, Any], labels: Mapping[str, str]
) -> dict[str, Any]:
    samples = list(private_mapping["samples"])
    population = private_mapping["population"]["datasets"]
    top_k = int(private_mapping["top_k"])
    groups = ["overall", *population]
    report: dict[str, Any] = {}
    for group in groups:
        group_rows = [
            item for item in samples if group == "overall" or item["dataset"] == group
        ]
        if group == "overall":
            case_count = sum(int(item["case_count"]) for item in population.values())
            strategy_populations = {
                strategy: sum(
                    int(item[f"{strategy}_returned_pair_count"])
                    for item in population.values()
                )
                for strategy in STRATEGIES
            }
        else:
            case_count = int(population[group]["case_count"])
            strategy_populations = {
                strategy: int(population[group][f"{strategy}_returned_pair_count"])
                for strategy in STRATEGIES
            }
        strategies: dict[str, Any] = {}
        for strategy in STRATEGIES:
            relevant_rows = [
                item for item in group_rows if item[f"{strategy}_returned"]
            ]
            informed = [
                item
                for item in relevant_rows
                if labels[item["sample_id"]] != "insufficient_information"
            ]
            positive = sum(
                labels[item["sample_id"]]
                in {"relevant", "partially_relevant"}
                for item in informed
            )
            weighted_denominator = sum(
                item["cell_population_count"] / item["cell_sample_count"]
                for item in informed
            )
            weighted_positive = sum(
                item["cell_population_count"] / item["cell_sample_count"]
                for item in informed
                if labels[item["sample_id"]]
                in {"relevant", "partially_relevant"}
            )
            coverage_complete = len(relevant_rows) == strategy_populations[strategy]
            insufficient_count = len(relevant_rows) - len(informed)
            precision_at_20 = None
            precision_reason = "incomplete_top20_annotation_coverage"
            if coverage_complete and insufficient_count == 0:
                precision_at_20 = positive / (case_count * top_k)
                precision_reason = None
            elif coverage_complete:
                precision_reason = "insufficient_information_labels_present"
            strategies[strategy] = {
                "sampled_returned_pair_count": len(relevant_rows),
                "population_returned_pair_count": strategy_populations[strategy],
                "sufficiently_informed_count": len(informed),
                "insufficient_information_count": insufficient_count,
                "sample_precision": positive / len(informed) if informed else None,
                "stratified_precision_estimate": (
                    weighted_positive / weighted_denominator
                    if weighted_denominator
                    else None
                ),
                "sampling_frame_coverage": (
                    len(relevant_rows) / strategy_populations[strategy]
                    if strategy_populations[strategy]
                    else 1.0
                ),
                "precision_at_20": precision_at_20,
                "precision_at_20_reason": precision_reason,
            }
        added = [
            item for item in group_rows if item["stratum"] == "normalization_added"
        ]
        added_informed = [
            item
            for item in added
            if labels[item["sample_id"]] != "insufficient_information"
        ]
        report[group] = {
            "strategies": strategies,
            "normalization_added_false_admission_rate": (
                sum(
                    labels[item["sample_id"]] == "not_relevant"
                    for item in added_informed
                )
                / len(added_informed)
                if added_informed
                else None
            ),
            "normalization_added_sufficiently_informed_count": len(added_informed),
        }
    return report


def _package_summary(
    *,
    manifest: Mapping[str, Any],
    universe: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    selected_cells = Counter(
        (str(item["dataset"]), str(item["stratum"])) for item in selected
    )
    universe_cells = Counter(
        (str(item["dataset"]), str(item["stratum"])) for item in universe
    )
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package": manifest["package"],
        "sample_count": len(selected),
        "maximum_sample_count": manifest["sampling"]["maximum_query_paper_pairs"],
        "eligible_pair_count": len(universe),
        "by_dataset": dict(sorted(Counter(str(item["dataset"]) for item in selected).items())),
        "by_stratum": dict(sorted(Counter(str(item["stratum"]) for item in selected).items())),
        "cells": [
            {
                "dataset": dataset,
                "stratum": stratum,
                "population_count": universe_cells[(dataset, stratum)],
                "sample_count": selected_cells[(dataset, stratum)],
            }
            for dataset in manifest["sampling"]["dataset_order"]
            for stratum in manifest["sampling"]["stratum_order"]
        ],
        "population": population,
        "blind_public_field_count": len(manifest["blind_public_fields"]),
        "forbidden_public_field_count": len(manifest["forbidden_public_fields"]),
        "annotation_status": "pending_human_labels",
        "automatic_metric_count": 0,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_or_qrels_used_for_sampling": False,
        },
    }


def _annotation_schema(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "labels": list(LABELS),
        "definitions": {
            "relevant": "directly answers or materially supports the query",
            "partially_relevant": "addresses a meaningful part of the query but is incomplete or indirect",
            "not_relevant": "does not materially address the query",
            "insufficient_information": "title and abstract are insufficient for a reliable decision",
        },
        "independence": "annotators must work independently before adjudication",
        "adjudication": "required only where the two labels differ",
        "public_sample_fields": list(manifest["blind_public_fields"]),
        "annotation_row_fields": ["sample_id", "annotator_id", "label", "notes"],
        "adjudication_row_fields": [
            "sample_id",
            "adjudicator_id",
            "final_label",
            "rationale",
        ],
    }


def _annotation_template(
    public_rows: Sequence[Mapping[str, Any]], annotator_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": item["sample_id"],
            "annotator_id": annotator_id,
            "label": None,
            "notes": "",
        }
        for item in public_rows
    ]


def _pending_metrics(status: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "annotation_status": status,
        "sample_count": sample_count,
        "agreement": None,
        "metrics": None,
        "reason": "human labels have not been provided",
    }


def _validate_package_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("package") != "lexical_normalization_v1_blinded_precision_audit_v1":
        raise ValueError("unexpected annotation package manifest")
    if manifest.get("scope", {}).get("gold_or_qrels_used_for_sampling") is not False:
        raise ValueError("sampling must not use gold or qrels")
    if any(
        int(manifest.get("scope", {}).get(field, -1)) != 0
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise ValueError("annotation generation must be offline and read-only")
    if tuple(manifest.get("annotation", {}).get("labels") or []) != LABELS:
        raise ValueError("annotation labels differ from the frozen schema")
    if set(manifest.get("blind_public_fields") or []) != set(PUBLIC_FIELDS):
        raise ValueError("blind public fields differ from the frozen schema")


def _paper(value: Any) -> Paper:
    paper = getattr(value, "paper", value)
    return paper if isinstance(paper, Paper) else Paper.model_validate(paper)


def _recursive_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            *[str(key) for key in value],
            *[
                nested
                for item in value.values()
                for nested in _recursive_keys(item)
            ],
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [nested for item in value for nested in _recursive_keys(item)]
    return []


def _resolve_repo_path(value: str, manifest_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_root = manifest_root.parent
    return (repo_root / path).resolve()


def _digest(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _readme(summary: Mapping[str, Any]) -> str:
    return (
        "# lexical_normalization_v1 盲化 Precision 标注包\n\n"
        f"本包含 {summary['sample_count']} 个随机化 query-paper 样本。"
        "标注者只能查看 `public/blind_samples.jsonl` 与标注 Schema，"
        "不得查看 `private/`。\n\n"
        "两位标注者应分别填写 `annotator_1.jsonl` 与 `annotator_2.jsonl`；"
        "完成前不得交流。仅对分歧项填写 `adjudication.jsonl`。"
        "完成后使用同一 CLI 的 `score` 子命令计算一致性和人工指标。\n\n"
        "当前模板没有人工标签，因此不包含 Precision、误放率或相关性结论。\n"
    )
