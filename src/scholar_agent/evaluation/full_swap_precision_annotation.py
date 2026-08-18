"""Blind annotation package for every Top-20 lexical-normalization change.

The package builder consumes immutable Record/Replay artifacts.  It rebuilds
candidate text without loading evaluator gold, then uses only the frozen
baseline/experiment Top-20 identity sets to select boundary changes.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
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
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.lexical_normalization_benchmark import (
    _identity_instance_key,
    _identity_sequence,
)
from scholar_agent.evaluation.lexical_normalization_expanded import (
    _read_ordered_rows,
    _validate_record_config,
    resolve_record_terminals,
)
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _reconstruct_candidates,
)
from scholar_agent.evaluation.precision_annotation import (
    LABELS,
    PUBLIC_FIELDS,
    assert_blinded_rows,
    cohen_kappa,
    validate_annotation_rows,
)
from scholar_agent.evaluation.relevance_filter_audit import (
    _read_json,
    _sha256,
    _tree_sha256,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


SCHEMA_VERSION = "1"
PACKAGE_NAME = "lexical_normalization_v1_record160_all_top20_changes_v1"
STRATA = ("experiment_admitted", "baseline_removed")
POSITIVE_LABELS = {"relevant", "partially_relevant"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def generate_full_swap_package(manifest_path: str | Path) -> dict[str, Any]:
    """Build a deterministic, exhaustive, change-only annotation package."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    occurrences, population = _load_swap_occurrences(manifest)
    unique_items, duplicate_relations = deduplicate_swap_occurrences(occurrences)
    prior_items = _load_prior_items(manifest)
    new_items, overlaps = partition_prior_overlaps(unique_items, prior_items)
    seed = str(manifest["randomization"]["seed"])
    new_items = sorted(
        new_items,
        key=lambda item: _digest(
            seed,
            "display",
            str(item["query_fingerprint"]),
            _profile_digest(item["profile"]),
        ),
    )

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, item in enumerate(new_items, start=1):
        sample_id = f"SPAR-LX-160-{index:04d}"
        paper = item["paper"]
        public_rows.append(
            {
                "sample_id": sample_id,
                "query": item["query"],
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
            }
        )
        private_rows.append(
            {
                "sample_id": sample_id,
                "query_fingerprint": item["query_fingerprint"],
                "paper_identity": _identity_instance_key(paper),
                "occurrences": item["occurrences"],
            }
        )
    assert_blinded_rows(public_rows, manifest["forbidden_public_fields"])
    annotator_one = _annotation_template(public_rows, "annotator_1")
    annotator_two = _annotation_template(public_rows, "annotator_2")
    adjudication = [
        {
            "sample_id": row["sample_id"],
            "adjudicator_id": "",
            "final_label": None,
            "rationale": "",
        }
        for row in public_rows
    ]
    private_mapping = {
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE_NAME,
        "top_k": int(manifest["scope"]["top_k"]),
        "case_count": int(manifest["scope"]["case_count"]),
        "population": population,
        "samples": private_rows,
        "prior_package_overlaps": overlaps,
        "within_package_duplicate_relations": duplicate_relations,
    }
    summary = _summary(
        occurrences=occurrences,
        unique_items=unique_items,
        new_items=private_rows,
        overlaps=overlaps,
        duplicate_relations=duplicate_relations,
        population=population,
    )
    metrics = evaluate_full_swap_annotations(
        private_mapping,
        annotator_one,
        annotator_two,
        adjudication,
    )
    return {
        "manifest": manifest,
        "blind_samples": public_rows,
        "annotation_schema": _annotation_schema(),
        "annotator_1": annotator_one,
        "annotator_2": annotator_two,
        "adjudication": adjudication,
        "private_mapping": private_mapping,
        "summary": summary,
        "metrics": metrics,
        "readme": _readme(summary),
    }


def write_full_swap_package(output: str | Path, package: Mapping[str, Any]) -> None:
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


def deduplicate_swap_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate equivalent papers per normalized query, preserving conflicts."""

    clusters: list[dict[str, Any]] = []
    profiles: list[IdentityProfile] = []
    duplicates: list[dict[str, Any]] = []
    ordered = sorted(
        occurrences,
        key=lambda item: (
            str(item["query_fingerprint"]),
            int(item["case_order"]),
            str(item["direction"]),
            int(item["rank"]),
            _identity_instance_key(item["paper"]),
        ),
    )
    for raw in ordered:
        item = dict(raw)
        paper = _paper(item["paper"])
        profile = build_identity_profile(paper)
        match = next(
            (
                index
                for index, existing in enumerate(profiles)
                if clusters[index]["query_fingerprint"]
                == item["query_fingerprint"]
                and identity_evidence_from_profiles(existing, profile).equivalent
            ),
            None,
        )
        occurrence = _private_occurrence(item)
        if match is None:
            clusters.append(
                {
                    "query": item["query"],
                    "query_fingerprint": item["query_fingerprint"],
                    "paper": paper.model_copy(deep=True),
                    "profile": profile,
                    "occurrences": [occurrence],
                }
            )
            profiles.append(profile)
            continue
        duplicates.append(
            {
                "reason": "same_query_unified_identity_equivalent",
                "kept_case_id": clusters[match]["occurrences"][0]["case_id"],
                "duplicate_case_id": occurrence["case_id"],
                "direction": occurrence["direction"],
            }
        )
        merged = deduplicate_papers([clusters[match]["paper"], paper])[0]
        clusters[match]["paper"] = merged
        clusters[match]["profile"] = build_identity_profile(merged)
        clusters[match]["occurrences"].append(occurrence)
        profiles[match] = clusters[match]["profile"]
    return clusters, duplicates


def partition_prior_overlaps(
    items: Sequence[dict[str, Any]], prior_items: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reference already-issued blind samples instead of duplicating annotation."""

    prior_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in prior_items:
        prior_by_query[str(item["query_fingerprint"])].append(item)
    fresh: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for item in items:
        match: Mapping[str, Any] | None = None
        evidence_rule: str | None = None
        for prior in prior_by_query.get(str(item["query_fingerprint"]), []):
            evidence = identity_evidence_from_profiles(
                item["profile"], prior["profile"]
            )
            if evidence.equivalent:
                match = prior
                evidence_rule = evidence.rule
                break
        if match is None:
            fresh.append(item)
            continue
        overlaps.append(
            {
                "prior_package": "lexical_normalization_v1_blinded_precision_audit_v1",
                "prior_sample_id": match["sample_id"],
                "query_fingerprint": item["query_fingerprint"],
                "paper_identity": _identity_instance_key(item["paper"]),
                "identity_rule": evidence_rule,
                "occurrences": item["occurrences"],
                "coverage_reason": "already_present_in_prior_blind_package",
            }
        )
    return fresh, overlaps


def evaluate_full_swap_annotations(
    private_mapping: Mapping[str, Any],
    annotator_one: Sequence[Mapping[str, Any]],
    annotator_two: Sequence[Mapping[str, Any]],
    adjudication: Sequence[Mapping[str, Any]],
    *,
    prior_resolved_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    samples = list(private_mapping.get("samples") or [])
    sample_ids = [str(item["sample_id"]) for item in samples]
    first_ids = {str(item.get("annotator_id") or "") for item in annotator_one}
    second_ids = {str(item.get("annotator_id") or "") for item in annotator_two}
    if (
        len(first_ids) != 1
        or len(second_ids) != 1
        or "" in first_ids
        or "" in second_ids
        or first_ids == second_ids
    ):
        raise ValueError("two distinct non-empty annotator IDs are required")
    first = validate_annotation_rows(annotator_one, sample_ids)
    second = validate_annotation_rows(annotator_two, sample_ids)
    final = validate_annotation_rows(
        adjudication, sample_ids, label_field="final_label"
    )
    if all(value is None for value in [*first.values(), *second.values()]):
        return _pending_metrics("pending_human_labels", len(samples))
    if any(value is None for value in [*first.values(), *second.values()]):
        raise ValueError("both annotators must complete every sample before scoring")
    first_labels = [str(first[sample_id]) for sample_id in sample_ids]
    second_labels = [str(second[sample_id]) for sample_id in sample_ids]
    disagreements = [
        sample_id
        for sample_id in sample_ids
        if first[sample_id] != second[sample_id]
    ]
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for sample_id in sample_ids:
        if first[sample_id] == second[sample_id]:
            if final[sample_id] is not None:
                raise ValueError("adjudication must be blank when annotators agree")
            resolved[sample_id] = str(first[sample_id])
        elif final[sample_id] is None:
            unresolved.append(sample_id)
        else:
            resolved[sample_id] = str(final[sample_id])
    agreement = {
        "cohen_kappa": cohen_kappa(first_labels, second_labels),
        "agreement_count": len(samples) - len(disagreements),
        "disagreement_count": len(disagreements),
        "missing_adjudication_count": len(unresolved),
    }
    if unresolved:
        result = _pending_metrics("pending_adjudication", len(samples))
        result["agreement"] = agreement
        result["reason"] = "all disagreements require independent adjudication"
        return result

    prior = dict(prior_resolved_labels or {})
    required_prior = {
        str(item["prior_sample_id"])
        for item in private_mapping.get("prior_package_overlaps") or []
    }
    invalid_prior = {
        sample_id: label
        for sample_id, label in prior.items()
        if label not in LABELS
    }
    if invalid_prior:
        raise ValueError("prior resolved labels contain an invalid label")
    missing_prior = sorted(required_prior - set(prior))
    if missing_prior:
        result = _pending_metrics("pending_prior_package_labels", len(samples))
        result["agreement"] = agreement
        result["reason"] = "prior-package overlap labels are required for closed scoring"
        result["missing_prior_label_count"] = len(missing_prior)
        return result

    labelled_occurrences: list[dict[str, Any]] = []
    for sample in samples:
        for occurrence in sample["occurrences"]:
            labelled_occurrences.append(
                {**occurrence, "label": resolved[str(sample["sample_id"])]}
            )
    for overlap in private_mapping.get("prior_package_overlaps") or []:
        label = prior[str(overlap["prior_sample_id"])]
        for occurrence in overlap["occurrences"]:
            labelled_occurrences.append({**occurrence, "label": label})
    metrics = _score_occurrences(
        labelled_occurrences,
        case_count=int(private_mapping["case_count"]),
        top_k=int(private_mapping["top_k"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_status": "complete",
        "sample_count": len(samples),
        "agreement": agreement,
        "metrics": metrics,
        "reason": None,
    }


def _load_swap_occurrences(
    manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_spec = manifest["source_replay"]
    replay_root = _repo_path(replay_spec["run_dir"])
    for name, expected in (
        ("aggregate.json", replay_spec["aggregate_sha256"]),
        ("case_comparison.jsonl", replay_spec["case_comparison_sha256"]),
        ("candidate_diagnostics.jsonl", replay_spec["candidate_diagnostics_sha256"]),
        ("manifest.json", replay_spec["manifest_sha256"]),
    ):
        _validate_hash(replay_root / name, expected)
    case_rows = _read_jsonl(replay_root / "case_comparison.jsonl")
    included = {
        str(row["case_id"]): row
        for row in case_rows
        if row.get("included_main_analysis") is True
    }
    if len(included) != int(manifest["scope"]["case_count"]):
        raise ValueError("included case count drift")

    record_spec = manifest["source_record"]
    record_root = _repo_path(record_spec["run_dir"])
    config_path = record_root / "config.json"
    results_path = record_root / "results.jsonl"
    _validate_hash(config_path, record_spec["config_sha256"])
    _validate_hash(results_path, record_spec["results_sha256"])
    config = _read_json(config_path)
    source_manifest = _read_json(replay_root / "manifest.json")
    _validate_record_config(config, source_manifest)
    snapshot_root = _repo_path(record_spec["snapshot_dir"])
    if _tree_sha256(snapshot_root) != record_spec["snapshot_tree_sha256"]:
        raise ValueError("frozen Snapshot tree hash drift")
    if sum(path.is_file() for path in snapshot_root.rglob("*")) != int(
        record_spec["snapshot_file_count"]
    ):
        raise ValueError("frozen Snapshot file count drift")
    rows = _read_ordered_rows(results_path)
    by_case = {str(row["case_id"]): row for row in rows}
    store = SnapshotStore(snapshot_root)
    occurrences: list[dict[str, Any]] = []
    shared_count = 0
    baseline_pair_count = 0
    experiment_pair_count = 0
    for case_id, case in sorted(
        included.items(), key=lambda item: int(item[1]["case_order"])
    ):
        prepared, terminal = resolve_record_terminals(
            by_case[case_id], store=store, configured_sources=config["sources"]
        )
        if terminal["source_states"] != case["source_states"]:
            raise ValueError("source terminal drift in annotation input")
        snapshots = {
            str(item["stage"]): item
            for item in prepared["stage_diagnostics"]["snapshots"]
        }
        candidates = _reconstruct_candidates(
            snapshots["initial_retrieval"],
            snapshots["initial_deduplicated"],
            config,
            store,
        )
        analysis = QueryAnalysis.model_validate(
            prepared["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
        )
        baseline = rerank_papers(
            analysis,
            judge_papers(
                analysis, candidates, use_llm=False, config=CURRENT_RULES_CONFIG
            ),
            top_k=len(candidates),
        )
        experiment = rerank_papers(
            analysis,
            judge_papers(
                analysis,
                candidates,
                use_llm=False,
                config=LEXICAL_NORMALIZATION_V1_CONFIG,
            ),
            top_k=len(candidates),
        )
        top_k = int(config["top_k"])
        baseline_returned = select_ranked_results(
            {"ranked_papers": baseline[:top_k]}, policy=config["result_policy"]
        )
        experiment_returned = select_ranked_results(
            {"ranked_papers": experiment[:top_k]}, policy=config["result_policy"]
        )
        baseline_ids = _identity_sequence(baseline_returned)
        experiment_ids = _identity_sequence(experiment_returned)
        if baseline_ids != case["baseline"]["returned_identity_keys"]:
            raise ValueError("baseline Top-20 identity drift")
        if experiment_ids != case["experiment"]["returned_identity_keys"]:
            raise ValueError("experiment Top-20 identity drift")
        if baseline_ids != _identity_sequence(
            snapshots["final_returned"].get("candidates") or []
        ):
            raise ValueError("frozen baseline result drift")
        baseline_pair_count += len(baseline_ids)
        experiment_pair_count += len(experiment_ids)
        baseline_by_key = dict(zip(baseline_ids, baseline_returned, strict=True))
        experiment_by_key = dict(zip(experiment_ids, experiment_returned, strict=True))
        baseline_keys = set(baseline_ids)
        experiment_keys = set(experiment_ids)
        shared_count += len(baseline_keys & experiment_keys)
        expected_admitted = set(case["top_20_swaps"]["admitted_ids"])
        expected_removed = set(case["top_20_swaps"]["removed_ids"])
        if expected_admitted != experiment_keys - baseline_keys:
            raise ValueError("admitted Top-20 set drift")
        if expected_removed != baseline_keys - experiment_keys:
            raise ValueError("removed Top-20 set drift")
        query = str(prepared["query"])
        fingerprint = _query_fingerprint(query)
        for direction, keys, values in (
            ("experiment_admitted", expected_admitted, experiment_by_key),
            ("baseline_removed", expected_removed, baseline_by_key),
        ):
            ranks = experiment_ids if direction == "experiment_admitted" else baseline_ids
            for key in sorted(keys):
                occurrences.append(
                    {
                        "query": query,
                        "query_fingerprint": fingerprint,
                        "paper": _paper(values[key]),
                        "case_id": case_id,
                        "case_order": int(case["case_order"]),
                        "direction": direction,
                        "rank": ranks.index(key) + 1,
                        "successful_source_count": int(case["successful_source_count"]),
                        "overlaps_prior_auto_dev_val": bool(
                            case["overlaps_prior_auto_dev_val"]
                        ),
                    }
                )
    return occurrences, {
        "case_count": len(included),
        "baseline_returned_pair_count": baseline_pair_count,
        "experiment_returned_pair_count": experiment_pair_count,
        "shared_top20_pair_count": shared_count,
    }


def _load_prior_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = manifest["prior_package"]
    mapping_path = _repo_path(spec["mapping"])
    public_path = _repo_path(spec["blind_samples"])
    _validate_hash(mapping_path, spec["mapping_sha256"])
    _validate_hash(public_path, spec["blind_samples_sha256"])
    prior_mapping = _read_json(mapping_path)
    public = {row["sample_id"]: row for row in _read_jsonl(public_path)}
    items: list[dict[str, Any]] = []
    for sample in prior_mapping["samples"]:
        sample_id = str(sample["sample_id"])
        row = public[sample_id]
        paper = _prior_paper(row, str(sample["paper_identity"]))
        items.append(
            {
                "sample_id": sample_id,
                "query_fingerprint": _query_fingerprint(str(row["query"])),
                "profile": build_identity_profile(paper),
            }
        )
    return items


def _prior_paper(public: Mapping[str, Any], canonical: str) -> Paper:
    identifiers: dict[str, str] = {}
    prefix, separator, value = canonical.partition(":")
    field = {
        "doi": "doi",
        "arxiv": "arxiv_id",
        "pmid": "pubmed_id",
        "pubmed": "pubmed_id",
        "openalex": "openalex_id",
        "s2": "semantic_scholar_id",
        "s2orc": "s2orc_corpus_id",
    }.get(prefix)
    if separator and value and field:
        identifiers[field] = value
    return Paper(
        title=str(public.get("title") or ""),
        abstract=str(public.get("abstract") or ""),
        year=public.get("year"),
        identifiers=PaperIdentifiers(**identifiers),
    )


def _score_occurrences(
    occurrences: Sequence[Mapping[str, Any]], *, case_count: int, top_k: int
) -> dict[str, Any]:
    informed = [
        item for item in occurrences if item["label"] != "insufficient_information"
    ]
    by_direction: dict[str, dict[str, Any]] = {}
    for direction in STRATA:
        rows = [item for item in informed if item["direction"] == direction]
        positive = sum(item["label"] in POSITIVE_LABELS for item in rows)
        by_direction[direction] = {
            "relation_count": sum(
                item["direction"] == direction for item in occurrences
            ),
            "sufficiently_informed_count": len(rows),
            "positive_count": positive,
            "changed_item_precision": positive / len(rows) if rows else None,
            "changed_component_precision_at_20": positive / (case_count * top_k),
        }
    all_informed = len(informed) == len(occurrences)
    paired_delta = None
    if all_informed:
        per_case: dict[str, int] = defaultdict(int)
        for item in informed:
            sign = 1 if item["direction"] == "experiment_admitted" else -1
            per_case[str(item["case_id"])] += sign * (
                item["label"] in POSITIVE_LABELS
            )
        paired_delta = sum(per_case.values()) / (case_count * top_k)
    strata: dict[str, Any] = {}
    for source_count in (1, 2, 3, 4):
        rows = [
            item
            for item in occurrences
            if int(item["successful_source_count"]) == source_count
        ]
        strata[str(source_count)] = _stratum_score(rows)
    return {
        "precision_at_20": {
            "baseline": None,
            "experiment": None,
            "reason": "unchanged_top20_items_are_not_in_the_change_only_package",
        },
        "changed_components": by_direction,
        "paired_precision_at_20_difference": paired_delta,
        "paired_precision_at_20_difference_reason": (
            None if all_informed else "insufficient_information_labels_present"
        ),
        "admitted_false_admission_rate": _label_rate(
            informed,
            direction="experiment_admitted",
            labels={"not_relevant"},
        ),
        "removed_relevance_rate": _label_rate(
            informed,
            direction="baseline_removed",
            labels=POSITIVE_LABELS,
        ),
        "by_successful_source_count": strata,
        "by_prior_dev_val_overlap": {
            "overlap": _stratum_score(
                [item for item in occurrences if item["overlaps_prior_auto_dev_val"]]
            ),
            "new": _stratum_score(
                [item for item in occurrences if not item["overlaps_prior_auto_dev_val"]]
            ),
        },
    }


def _stratum_score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    informed = [item for item in rows if item["label"] != "insufficient_information"]
    admitted = [item for item in informed if item["direction"] == "experiment_admitted"]
    removed = [item for item in informed if item["direction"] == "baseline_removed"]
    return {
        "relation_count": len(rows),
        "sufficiently_informed_count": len(informed),
        "admitted_positive_count": sum(item["label"] in POSITIVE_LABELS for item in admitted),
        "removed_positive_count": sum(item["label"] in POSITIVE_LABELS for item in removed),
    }


def _label_rate(
    rows: Sequence[Mapping[str, Any]], *, direction: str, labels: set[str]
) -> float | None:
    selected = [item for item in rows if item["direction"] == direction]
    if not selected:
        return None
    return sum(item["label"] in labels for item in selected) / len(selected)


def _summary(
    *,
    occurrences: Sequence[Mapping[str, Any]],
    unique_items: Sequence[Mapping[str, Any]],
    new_items: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    duplicate_relations: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    direction_counts = Counter(str(item["direction"]) for item in occurrences)
    overlap_relations = sum(len(item["occurrences"]) for item in overlaps)
    public_relations = sum(len(item["occurrences"]) for item in new_items)
    if public_relations + overlap_relations != len(occurrences):
        raise ValueError("Top-20 change coverage does not close")
    return {
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE_NAME,
        "top20_change_relation_count": len(occurrences),
        "direction_counts": dict(sorted(direction_counts.items())),
        "unique_query_paper_count": len(unique_items),
        "public_new_sample_count": len(new_items),
        "public_new_relation_count": public_relations,
        "prior_package_overlap_sample_count": len(overlaps),
        "prior_package_overlap_relation_count": overlap_relations,
        "within_package_duplicate_relation_count": len(duplicate_relations),
        "coverage_closure": {
            "covered_by_new_public_sample": public_relations,
            "covered_by_prior_package_reference": overlap_relations,
            "uncovered_relation_count": 0,
        },
        "population": dict(population),
        "annotation_status": "pending_human_labels",
        "metrics": {
            "precision_at_20": None,
            "paired_precision_at_20_difference": None,
            "admitted_false_admission_rate": None,
            "removed_relevance_rate": None,
            "cohen_kappa": None,
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_or_qrels_used_for_generation": False,
        },
    }


def _pending_metrics(status: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_status": status,
        "sample_count": sample_count,
        "agreement": {"cohen_kappa": None},
        "metrics": {
            "precision_at_20": {"baseline": None, "experiment": None},
            "changed_components": None,
            "paired_precision_at_20_difference": None,
            "admitted_false_admission_rate": None,
            "removed_relevance_rate": None,
            "by_successful_source_count": None,
            "by_prior_dev_val_overlap": None,
        },
        "reason": "human labels have not been provided",
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("package") != PACKAGE_NAME:
        raise ValueError("unexpected full-swap annotation manifest")
    scope = manifest.get("scope") or {}
    if scope.get("gold_or_qrels_used_for_generation") is not False:
        raise ValueError("annotation generation must not use gold or qrels")
    if any(
        int(scope.get(field, -1)) != 0
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise ValueError("annotation generation must be offline and read-only")
    if tuple(manifest.get("annotation", {}).get("labels") or []) != LABELS:
        raise ValueError("annotation labels differ from the frozen schema")
    if set(manifest.get("blind_public_fields") or []) != set(PUBLIC_FIELDS):
        raise ValueError("blind public fields differ from the frozen schema")


def _annotation_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "labels": list(LABELS),
        "definitions": {
            "relevant": "directly answers or materially supports the query",
            "partially_relevant": "addresses a meaningful part of the query but is incomplete or indirect",
            "not_relevant": "does not materially address the query",
            "insufficient_information": "title and abstract are insufficient for a reliable decision",
        },
        "independence": "annotators work independently before adjudication",
        "adjudication": "required only where the two labels differ",
        "public_sample_fields": list(PUBLIC_FIELDS),
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
            "sample_id": row["sample_id"],
            "annotator_id": annotator_id,
            "label": None,
            "notes": "",
        }
        for row in public_rows
    ]


def _private_occurrence(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": item["case_id"],
        "case_order": int(item["case_order"]),
        "direction": item["direction"],
        "rank": int(item["rank"]),
        "successful_source_count": int(item["successful_source_count"]),
        "overlaps_prior_auto_dev_val": bool(item["overlaps_prior_auto_dev_val"]),
    }


def _query_fingerprint(query: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", query).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _profile_digest(profile: IdentityProfile) -> str:
    return _digest(
        *sorted(profile.identifiers),
        profile.title,
        str(profile.year),
        *sorted(profile.authors),
    )


def _paper(value: Any) -> Paper:
    paper = getattr(value, "paper", value)
    return paper if isinstance(paper, Paper) else Paper.model_validate(paper)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _validate_hash(path: Path, expected: str) -> None:
    if _sha256(path) != str(expected):
        raise ValueError(f"frozen input hash drift:{path.name}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
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


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _readme(summary: Mapping[str, Any]) -> str:
    return (
        "# Top-20 变更盲化人工标注包\n\n"
        f"公共目录包含 {summary['public_new_sample_count']} 个随机化 query-paper 项；"
        f"另有 {summary['prior_package_overlap_sample_count']} 个项引用既有盲标包，"
        "不会重复标注。公共样本仅包含 sample_id、query、title、abstract、year。\n\n"
        "两位标注者应独立填写各自模板，仅对分歧项进行仲裁。私有 mapping 只能由"
        "评测负责人在标注完成后使用。当前没有人工标签，因此 Precision、差值、"
        "误放率、相关率与 Cohen's kappa 均为 null。\n"
    )
