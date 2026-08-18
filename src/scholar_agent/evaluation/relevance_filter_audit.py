"""Pure-offline audit of relevance-filter false negatives.

The audit reconstructs papers and rankings from frozen Benchmark Replay
artifacts.  Gold is only used after reconstruction to label candidate rows and
to calculate evaluator-only, one-rule-at-a-time counterfactuals.  The module
does not import SearchService, connectors, providers, or Snapshot writers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from scholar_agent.agents.judgement import DOMAIN_TERMS, STOPWORDS
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    JudgementFeatureVector,
    JudgementResult,
    QueryAnalysis,
    RankedPaper,
)
from scholar_agent.evaluation.datasets.beir_scifact import load_beir_scifact_enriched
from scholar_agent.evaluation.datasets.registry import load_dataset
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _candidate_key,
    _reconstruct_candidates,
    _reconstruct_ranking,
)
from scholar_agent.evaluation.metrics import (
    average_metric_sets,
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_SCHEMA_VERSION = "1"
RETURN_CATEGORIES = {"highly_relevant", "partially_relevant"}
NEGATIVE_RULE_COMPONENTS = (
    "constraint_coverage_adjustment",
    "venue_mismatch_penalty",
    "temporal_mismatch_penalty",
    "paper_type_mismatch_penalty",
    "missing_abstract_penalty",
    "missing_metadata_penalty",
    "hard_constraint_adjustment",
)
RootCause = Literal[
    "query_parsing_missing",
    "field_text_missing",
    "morphology_or_abbreviation_mismatch",
    "constraint_penalty",
    "fixed_threshold",
    "category_priority",
    "other",
]
ROOT_CAUSES = (
    "query_parsing_missing",
    "field_text_missing",
    "morphology_or_abbreviation_mismatch",
    "constraint_penalty",
    "fixed_threshold",
    "category_priority",
    "other",
)


def classify_filtered_gold_root(
    *,
    query_term_count: int,
    structured_term_count: int,
    title_present: bool,
    abstract_present: bool,
    lexical_gap_terms: Sequence[str],
    negative_components: Mapping[str, float],
    hard_constraint_failures: Sequence[str],
    score: float,
    partial_threshold: float,
    rank: int | None,
    top_k: int = 20,
) -> RootCause:
    """Assign one deterministic primary cause to a filtered gold candidate."""

    if query_term_count == 0 and structured_term_count == 0:
        return "query_parsing_missing"
    if not title_present or not abstract_present:
        return "field_text_missing"
    if lexical_gap_terms:
        return "morphology_or_abbreviation_mismatch"
    if hard_constraint_failures or any(
        value < 0 for value in negative_components.values()
    ):
        return "constraint_penalty"
    if rank is not None and rank > top_k:
        return "category_priority"
    if score < partial_threshold:
        return "fixed_threshold"
    return "other"


def lexical_gap_terms(
    expected_terms: Sequence[str],
    matched_terms: Sequence[str],
    *,
    title: str,
    abstract: str,
) -> list[str]:
    """Find conservative punctuation/morphology/acronym matches missed exactly.

    This is diagnostic only.  It does not alter production matching and only
    reports a gap when a normalized whole-token form or an explicit acronym
    expansion can be observed in the candidate text.
    """

    matched = {_lexical_key(item) for item in matched_terms if _lexical_key(item)}
    text = f"{title} {abstract}"
    tokens = {_simple_stem(item) for item in _word_tokens(text)}
    compact_tokens = {_lexical_key(item) for item in _word_tokens(text)}
    acronyms = _explicit_acronyms(text)
    initialisms = _text_initialisms(text)
    gaps: list[str] = []
    seen_gap_keys: set[str] = set()
    for term in _dedupe_strings(expected_terms):
        key = _lexical_key(term)
        if not key or key in matched:
            continue
        term_words = _word_tokens(term)
        term_stems = {_simple_stem(item) for item in term_words}
        compact = "".join(_lexical_key(item) for item in term_words)
        morphology_hit = bool(term_stems) and term_stems.issubset(tokens)
        punctuation_hit = bool(compact) and compact in compact_tokens
        abbreviation_hit = (
            key in acronyms
            or compact in acronyms
            or (_looks_like_acronym(term) and compact in initialisms)
        )
        if (
            (morphology_hit or punctuation_hit or abbreviation_hit)
            and key not in seen_gap_keys
        ):
            seen_gap_keys.add(key)
            gaps.append(term)
    return gaps


def remove_single_score_rule(
    judgements: Sequence[JudgementResult],
    rule: str,
) -> list[JudgementResult]:
    """Remove one frozen negative score component without changing thresholds."""

    if rule not in NEGATIVE_RULE_COMPONENTS:
        raise ValueError(f"unsupported counterfactual rule:{rule}")
    output: list[JudgementResult] = []
    for judgement in judgements:
        feature = judgement.feature_vector
        if feature is None:
            output.append(judgement.model_copy(deep=True))
            continue
        component = float(feature.score_components.get(rule) or 0.0)
        if component >= 0:
            output.append(judgement.model_copy(deep=True))
            continue
        score = round(min(1.0, max(0.0, judgement.score - component)), 4)
        category = _threshold_category(feature, score)
        if feature.category_reason in {
            "missing_title_and_abstract",
            "minimum_evidence_count_not_met",
        }:
            category = "insufficient_evidence"
        components = dict(feature.score_components)
        components[rule] = 0.0
        updated_feature = feature.model_copy(
            update={
                "score_components": components,
                "final_score": score,
                "category_reason": f"audit_removed_single_rule:{rule}",
            }
        )
        output.append(
            judgement.model_copy(
                update={
                    "score": score,
                    "category": category,
                    "feature_vector": updated_feature,
                },
                deep=True,
            )
        )
    return output


def run_relevance_filter_audit(
    manifest_path: str | Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run all frozen datasets declared by an audit manifest."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("audit manifest requires datasets")
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    dataset_aggregates: dict[str, Any] = {}
    inputs: dict[str, Any] = {"manifest_sha256": _sha256(manifest_file)}
    for spec in datasets:
        if not isinstance(spec, dict):
            raise ValueError("audit dataset spec must be an object")
        cases, candidates, filtered, aggregate, fingerprints = _audit_dataset(spec)
        label = str(spec["label"])
        case_rows.extend(cases)
        candidate_rows.extend(candidates)
        filtered_rows.extend(filtered)
        dataset_aggregates[label] = aggregate
        inputs[label] = fingerprints
    overall = _aggregate_all(
        manifest=manifest,
        datasets=dataset_aggregates,
        filtered_rows=filtered_rows,
        inputs=inputs,
    )
    return case_rows, candidate_rows, filtered_rows, overall


def write_relevance_filter_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    filtered_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_audit.jsonl", case_rows)
    _write_jsonl(root / "candidate_audit.jsonl", candidate_rows)
    _write_jsonl(root / "filtered_gold_chains.jsonl", filtered_rows)
    _atomic_write_json(root / "aggregate.json", aggregate)


def _audit_dataset(
    spec: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    label = str(spec["label"])
    run_root = Path(str(spec["run_dir"])).expanduser().resolve()
    snapshot_root = Path(str(spec["snapshot_dir"])).expanduser().resolve()
    config = _read_json(run_root / "config.json")
    _validate_config(config, spec)
    results = _read_rows(run_root / "results.jsonl")
    queries = _load_queries(config, spec)
    by_case = {item.query_id: item for item in queries}
    case_ids = [str(item) for item in config["case_ids"]]
    if set(results) != set(case_ids) or any(item not in by_case for item in case_ids):
        raise ValueError(f"frozen case set mismatch:{label}")
    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    case_states: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        case, candidates, filtered, state = _audit_case(
            label=label,
            case_order=case_order,
            eval_query=by_case[case_id],
            row=results[case_id],
            config=config,
            store=store,
            required_candidate_source=(
                str(spec["required_candidate_source"])
                if spec.get("required_candidate_source")
                else None
            ),
        )
        case_rows.append(case)
        candidate_rows.extend(candidates)
        filtered_rows.extend(filtered)
        case_states.append(state)
    gold_rows = [item for item in candidate_rows if item["candidate_label"] == "gold"]
    expected_gold = spec.get("expected_candidate_gold_count")
    expected_filtered = spec.get("expected_filtered_gold_count")
    if expected_gold is not None and len(gold_rows) != int(expected_gold):
        raise ValueError(
            f"{label} candidate gold mismatch:{len(gold_rows)} != {expected_gold}"
        )
    if expected_filtered is not None and len(filtered_rows) != int(expected_filtered):
        raise ValueError(
            f"{label} filtered gold mismatch:"
            f"{len(filtered_rows)} != {expected_filtered}"
        )
    aggregate = _aggregate_dataset(
        label=label,
        case_rows=case_rows,
        candidate_rows=candidate_rows,
        filtered_rows=filtered_rows,
        case_states=case_states,
    )
    fingerprints = {
        "config_sha256": _sha256(run_root / "config.json"),
        "results_sha256": _sha256(run_root / "results.jsonl"),
        "snapshot_tree_sha256": _tree_sha256(snapshot_root),
        "snapshot_file_count": sum(path.is_file() for path in snapshot_root.rglob("*")),
    }
    return case_rows, candidate_rows, filtered_rows, aggregate, fingerprints


def _audit_case(
    *,
    label: str,
    case_order: int,
    eval_query: EvalQuery,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    required_candidate_source: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if row.get("status") != "succeeded":
        raise ValueError(f"frozen case is not succeeded:{label}:{eval_query.query_id}")
    snapshots = {
        str(item["stage"]): item for item in row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
        "final_returned",
    }
    if not required.issubset(snapshots):
        raise ValueError(f"missing frozen stage:{label}:{eval_query.query_id}")
    if any(snapshots[name].get("status") != "completed" for name in required):
        raise ValueError(f"incomplete frozen stage:{label}:{eval_query.query_id}")
    initial = snapshots["initial_retrieval"]
    dedup = snapshots["initial_deduplicated"]
    judged = snapshots["initial_judged"]
    reranked = snapshots["initial_reranked"]
    full_candidates = _reconstruct_candidates(initial, dedup, config, store)
    ranked = _reconstruct_ranking(full_candidates, judged, reranked, row)
    judgements = _frozen_judgements(full_candidates, judged)
    judgement_by_key = {_candidate_key(item.paper): item for item in judgements}
    if len(judgement_by_key) != len(judgements):
        raise ValueError(
            f"duplicate frozen judgement identity:{label}:{eval_query.query_id}"
        )
    ranked_judgements = [
        _judgement_for_ranked(item, judgement_by_key) for item in ranked
    ]
    analysis = QueryAnalysis.model_validate(
        row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    returned = list(snapshots["final_returned"].get("candidates") or [])
    current_selected = select_ranked_results(
        {"ranked_papers": ranked[: int(config["top_k"])]},
        policy=str(config["result_policy"]),
    )
    if _identity_sequence(current_selected) != _identity_sequence(returned):
        raise ValueError(
            f"frozen returned reconstruction mismatch:{label}:{eval_query.query_id}"
        )

    all_gold_flags = [
        matched_paper_ids([item.paper], eval_query.gold_papers, k=1) for item in ranked
    ]
    gold_flags = [
        matches
        if matches
        and (
            required_candidate_source is None
            or _paper_has_source(item.paper, required_candidate_source)
        )
        else []
        for item, matches in zip(ranked, all_gold_flags, strict=True)
    ]
    gold_ranks = [
        item.rank
        for item, matches in zip(ranked, gold_flags, strict=True)
        if matches
    ]
    comparison_ceiling = max(gold_ranks, default=0)
    candidates: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for judgement, ranked_item, gold_ids, any_gold_ids in zip(
        ranked_judgements, ranked, gold_flags, all_gold_flags, strict=True
    ):
        if any_gold_ids and not gold_ids:
            continue
        if not gold_ids and (
            not comparison_ceiling or ranked_item.rank >= comparison_ceiling
        ):
            continue
        audit = _candidate_audit(
            label=label,
            case_order=case_order,
            case_id=eval_query.query_id,
            analysis=analysis,
            judgement=judgement,
            ranked=ranked_item,
            returned=bool(_matching_any(returned, judgement.paper)),
            gold_ids=gold_ids,
            top_k=int(config["top_k"]),
        )
        candidates.append(audit)
        if audit["candidate_label"] == "gold" and audit["filtered"]:
            filtered.append(_filtered_chain(audit))

    variants: dict[str, Sequence[Any]] = {
        "current": current_selected,
        "remove_return_category_gate": ranked[: int(config["top_k"])],
    }
    for rule in NEGATIVE_RULE_COMPONENTS:
        changed = remove_single_score_rule(judgements, rule)
        changed_ranked = rerank_papers(analysis, changed, top_k=len(changed))
        variants[f"remove_{rule}"] = select_ranked_results(
            {"ranked_papers": changed_ranked[: int(config["top_k"])]},
            policy=str(config["result_policy"]),
        )
    variant_metrics = {
        name: _case_metrics(items, eval_query.gold_papers)
        for name, items in variants.items()
    }
    candidate_gold_ids = _dedupe_strings(
        value for matches in gold_flags for value in matches
    )
    case = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": label,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "candidate_count": len(ranked),
        "evaluable_gold_count": evaluable_gold_count(eval_query.gold_papers),
        "candidate_gold_count": len(candidate_gold_ids),
        "candidate_gold_ids": candidate_gold_ids,
        "filtered_gold_count": sum(
            item["candidate_label"] == "gold" and item["filtered"]
            for item in candidates
        ),
        "comparison_non_gold_count": sum(
            item["candidate_label"] == "higher_ranked_non_gold" for item in candidates
        ),
        "variants": variant_metrics,
    }
    state = {
        "gold": eval_query.gold_papers,
        "variants": variants,
        "candidate_gold_ids": candidate_gold_ids,
    }
    return case, candidates, filtered, state


def _candidate_audit(
    *,
    label: str,
    case_order: int,
    case_id: str,
    analysis: QueryAnalysis,
    judgement: JudgementResult,
    ranked: RankedPaper,
    returned: bool,
    gold_ids: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    feature = judgement.feature_vector
    if feature is None:
        raise ValueError(f"missing frozen judgement feature:{label}:{case_id}")
    expected = _expected_terms(analysis)
    matched = _matched_terms(feature, analysis, judgement.paper)
    expected_flat = [value for values in expected.values() for value in values]
    matched_flat = [value for values in matched.values() for value in values]
    lexical = lexical_gap_terms(
        expected_flat,
        matched_flat,
        title=judgement.paper.title,
        abstract=judgement.paper.abstract,
    )
    negatives = {
        key: float(value)
        for key, value in feature.score_components.items()
        if float(value) < 0
    }
    filtered = judgement.category not in RETURN_CATEGORIES
    root: RootCause | None = None
    if gold_ids and filtered:
        root = classify_filtered_gold_root(
            query_term_count=len(expected["topic"]),
            structured_term_count=sum(
                len(expected[name]) for name in ("must_have", "method", "dataset")
            ),
            title_present=bool(judgement.paper.title.strip()),
            abstract_present=bool(judgement.paper.abstract.strip()),
            lexical_gap_terms=lexical,
            negative_components=negatives,
            hard_constraint_failures=feature.hard_constraint_failures,
            score=judgement.score,
            partial_threshold=feature.partially_relevant_threshold,
            rank=ranked.rank,
            top_k=top_k,
        )
    triggers = _triggered_rules(judgement, ranked.rank, top_k)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": label,
        "case_order": case_order,
        "case_id": case_id,
        "candidate_label": "gold" if gold_ids else "higher_ranked_non_gold",
        "candidate_id": canonical_paper_id(judgement.paper),
        "matched_gold_ids": list(gold_ids),
        "rank": ranked.rank,
        "returned": returned,
        "filtered": filtered,
        "judgement": {
            "score": judgement.score,
            "category": judgement.category,
            "category_reason": feature.category_reason,
            "partial_threshold": feature.partially_relevant_threshold,
            "score_gap_to_partial": round(
                feature.partially_relevant_threshold - judgement.score, 6
            ),
            "score_components": dict(sorted(feature.score_components.items())),
            "rerank_score_components": ranked.score_breakdown.model_dump(mode="json"),
            "hard_constraint_failures": list(feature.hard_constraint_failures),
            "constraint_results": dict(sorted(feature.constraint_results.items())),
            "warnings": list(judgement.warnings),
        },
        "text_evidence": {
            "title_present": bool(judgement.paper.title.strip()),
            "abstract_present": bool(judgement.paper.abstract.strip()),
            "title_length": len(judgement.paper.title),
            "abstract_length": len(judgement.paper.abstract),
            "metadata_completeness": feature.metadata_completeness,
        },
        "facet_evidence": {
            name: {
                "input_terms": expected[name],
                "matched_terms": matched[name],
            }
            for name in ("topic", "must_have", "method", "dataset", "domain")
        },
        "lexical_gap_terms": lexical,
        "triggered_rules": triggers,
        "primary_root_cause": root,
    }


def _filtered_chain(audit: dict[str, Any]) -> dict[str, Any]:
    judgement = audit["judgement"]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": audit["dataset"],
        "case_order": audit["case_order"],
        "case_id": audit["case_id"],
        "candidate_id": audit["candidate_id"],
        "matched_gold_ids": audit["matched_gold_ids"],
        "primary_root_cause": audit["primary_root_cause"],
        "decision_chain": [
            {
                "step": "query_analysis",
                "facet_evidence": audit["facet_evidence"],
            },
            {"step": "field_availability", **audit["text_evidence"]},
            {
                "step": "lexical_matching",
                "lexical_gap_terms": audit["lexical_gap_terms"],
            },
            {
                "step": "rule_scoring",
                "score_components": judgement["score_components"],
                "hard_constraint_failures": judgement["hard_constraint_failures"],
                "triggered_rules": audit["triggered_rules"],
            },
            {
                "step": "threshold",
                "score": judgement["score"],
                "partial_threshold": judgement["partial_threshold"],
                "score_gap_to_partial": judgement["score_gap_to_partial"],
                "category": judgement["category"],
                "category_reason": judgement["category_reason"],
            },
            {
                "step": "category_priority_and_return",
                "rank": audit["rank"],
                "filtered": audit["filtered"],
                "returned": audit["returned"],
            },
        ],
    }


def _aggregate_dataset(
    *,
    label: str,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    filtered_rows: Sequence[dict[str, Any]],
    case_states: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gold = [item for item in candidate_rows if item["candidate_label"] == "gold"]
    non_gold = [
        item
        for item in candidate_rows
        if item["candidate_label"] == "higher_ranked_non_gold"
    ]
    variants: dict[str, Any] = {}
    variant_names = list(case_states[0]["variants"]) if case_states else []
    for name in variant_names:
        metrics = average_metric_sets(
            [
                evaluate_ranking(state["variants"][name], state["gold"], [20])
                for state in case_states
                if evaluable_gold_count(state["gold"])
            ]
        )
        returned_gold = sum(
            len(matched_paper_ids(state["variants"][name], state["gold"], k=20))
            for state in case_states
        )
        variants[name] = {
            "recall_at_20": metrics.recall_at_k.get(20, 0.0),
            "f1_at_20": metrics.f1_at_k.get(20, 0.0),
            "returned_gold_relation_count": returned_gold,
        }
    current = variants.get("current", {})
    for name, values in variants.items():
        values["delta_vs_current"] = {
            "recall_at_20": values["recall_at_20"]
            - float(current.get("recall_at_20", 0.0)),
            "f1_at_20": values["f1_at_20"] - float(current.get("f1_at_20", 0.0)),
            "gold_relations": values["returned_gold_relation_count"]
            - int(current.get("returned_gold_relation_count", 0)),
        }
    observed_roots = Counter(
        str(item["primary_root_cause"]) for item in filtered_rows
    )
    root_counts = {name: observed_roots[name] for name in ROOT_CAUSES}
    if sum(root_counts.values()) != len(filtered_rows):
        raise ValueError(f"filtered root classification not closed:{label}")
    trigger_counts = {
        "gold": _count_triggers(gold),
        "higher_ranked_non_gold": _count_triggers(non_gold),
    }
    false_negative_rate = len(filtered_rows) / len(gold) if gold else None
    return {
        "case_count": len(case_rows),
        "evaluable_query_count": sum(
            item["evaluable_gold_count"] > 0 for item in case_rows
        ),
        "evaluable_gold_relation_count": sum(
            item["evaluable_gold_count"] for item in case_rows
        ),
        "candidate_gold_relation_count": len(gold),
        "filtered_gold_relation_count": len(filtered_rows),
        "gold_false_negative_rate": false_negative_rate,
        "comparison_non_gold_count": len(non_gold),
        "filtered_root_causes": root_counts,
        "score_distributions": {
            "gold": _score_distribution(gold),
            "higher_ranked_non_gold": _score_distribution(non_gold),
            "gold_by_judgement": _scores_by_category(gold),
            "higher_ranked_non_gold_by_judgement": _scores_by_category(non_gold),
        },
        "rule_trigger_frequency": trigger_counts,
        "variants": variants,
        "candidate_recall": _average(
            item["candidate_gold_count"] / item["evaluable_gold_count"]
            for item in case_rows
            if item["evaluable_gold_count"]
        ),
    }


def _aggregate_all(
    *,
    manifest: dict[str, Any],
    datasets: dict[str, Any],
    filtered_rows: Sequence[dict[str, Any]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    scifact = datasets.get("scifact", {})
    auto_filtered = sum(
        int(datasets.get(name, {}).get("filtered_gold_relation_count", 0))
        for name in ("auto_dev", "auto_val")
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "relevance_filter_false_negative_calibration",
        "implementation_base_commit": manifest.get("implementation_base_commit"),
        "inputs": inputs,
        "datasets": datasets,
        "filtered_chain_count": len(filtered_rows),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_access": "post_reconstruction_evaluator_only",
            "counterfactual_scope": "one_frozen_rule_removed_at_a_time_no_combinations",
            "counterfactual_is_not_deployable_score": True,
        },
        "cross_dataset_evidence": {
            "scifact_filtered_gold_count": int(
                scifact.get("filtered_gold_relation_count", 0)
            ),
            "auto_filtered_gold_count": auto_filtered,
            "auto_candidate_gold_count": sum(
                int(datasets.get(name, {}).get("candidate_gold_relation_count", 0))
                for name in ("auto_dev", "auto_val")
            ),
            "systemic_rule": _systemic_rule(datasets),
            "evidence_limit": (
                "AutoScholarQuery frozen candidate-gold evidence is small; absence or "
                "presence is diagnostic, not a population estimate."
            ),
        },
    }


def _systemic_rule(datasets: Mapping[str, Any]) -> str | None:
    active: list[set[str]] = []
    for name in ("scifact", "auto_dev", "auto_val"):
        values = datasets.get(name)
        if not values or not values.get("filtered_gold_relation_count"):
            continue
        active.append(
            {
                key
                for key, count in values.get("filtered_root_causes", {}).items()
                if int(count) > 0
            }
        )
    if len(active) < 2:
        return None
    shared = set.intersection(*active)
    return sorted(shared)[0] if shared else None


def _frozen_judgements(
    candidates: Sequence[Paper], judged: dict[str, Any]
) -> list[JudgementResult]:
    diagnostics = list(judged.get("candidates") or [])
    if len(candidates) != len(diagnostics):
        raise ValueError("frozen judgement candidate count mismatch")
    output: list[JudgementResult] = []
    for paper, diagnostic in zip(candidates, diagnostics, strict=True):
        if not _equivalent(paper, diagnostic):
            raise ValueError("frozen judgement candidate order mismatch")
        feature = diagnostic.get("judgement_features")
        output.append(
            JudgementResult(
                paper=paper,
                score=float(diagnostic["judgement_score"]),
                category=str(diagnostic["category"]),
                reasoning="frozen_stage_diagnostic",
                matched_terms=list(diagnostic.get("matched_terms") or []),
                warnings=list(diagnostic.get("warnings") or []),
                feature_vector=(
                    JudgementFeatureVector.model_validate(feature) if feature else None
                ),
            )
        )
    return output


def _judgement_for_ranked(
    ranked: RankedPaper,
    judgements: Mapping[str, JudgementResult],
) -> JudgementResult:
    try:
        judgement = judgements[_candidate_key(ranked.paper)]
    except KeyError as exc:
        raise ValueError("ranked paper lacks a frozen judgement") from exc
    if (
        judgement.category != ranked.category
        or judgement.score != ranked.score_breakdown.relevance_score
    ):
        raise ValueError("ranked judgement fields mismatch")
    return judgement


def _expected_terms(analysis: QueryAnalysis) -> dict[str, list[str]]:
    return {
        "topic": _production_query_terms(analysis.original_query),
        "must_have": list(analysis.constraints.must_include_terms),
        "method": list(analysis.constraints.methods),
        "dataset": list(analysis.constraints.datasets),
        "domain": list(DOMAIN_TERMS.get(analysis.domain, ())),
    }


def _matched_terms(
    feature: JudgementFeatureVector,
    analysis: QueryAnalysis,
    paper: Paper,
) -> dict[str, list[str]]:
    domain = [
        term
        for term in DOMAIN_TERMS.get(analysis.domain, ())
        if _exact_term_in_text(term, f"{paper.title} {paper.abstract}")
    ]
    return {
        "topic": list(feature.matched_topic_terms),
        "must_have": list(feature.matched_must_have_terms),
        "method": list(feature.matched_method_terms),
        "dataset": list(feature.matched_dataset_terms),
        "domain": domain,
    }


def _triggered_rules(
    judgement: JudgementResult, rank: int, top_k: int
) -> list[str]:
    feature = judgement.feature_vector
    if feature is None:
        return ["missing_feature_vector"]
    values: list[str] = []
    values.extend(
        f"negative_component:{key}"
        for key, value in sorted(feature.score_components.items())
        if float(value) < 0
    )
    values.extend(
        f"hard_constraint:{item}" for item in feature.hard_constraint_failures
    )
    values.extend(f"warning:{item.split(':', 1)[0]}" for item in judgement.warnings)
    if judgement.category not in RETURN_CATEGORIES:
        values.append("return_category_gate")
        if rank > top_k:
            values.append("category_priority_outside_top_k")
    values.append(f"category_reason:{feature.category_reason}")
    return _dedupe_strings(values)


def _threshold_category(feature: JudgementFeatureVector, score: float) -> str:
    if score >= feature.highly_relevant_threshold:
        return "highly_relevant"
    if score >= feature.partially_relevant_threshold:
        return "partially_relevant"
    if score >= feature.weakly_relevant_threshold:
        return "weakly_relevant"
    return "irrelevant"


def _case_metrics(
    ranked: Sequence[Any], gold: Sequence[EvalGoldPaper]
) -> dict[str, Any]:
    metrics = evaluate_ranking(ranked, gold, [20])
    matched = matched_paper_ids(ranked, gold, k=20)
    return {
        "recall_at_20": metrics.recall_at_k.get(20, 0.0),
        "f1_at_20": metrics.f1_at_k.get(20, 0.0),
        "matched_gold_count": len(matched),
        "matched_gold_ids": matched,
        "ranked_ids": _identity_sequence(ranked),
    }


def _score_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(item["judgement"]["score"]) for item in rows)
    if not values:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": values[0],
        "q1": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "q3": _percentile(values, 0.75),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def _scores_by_category(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(item["judgement"]["category"]) for item in rows})
    return {
        category: _score_distribution(
            [item for item in rows if item["judgement"]["category"] == category]
        )
        for category in categories
    }


def _count_triggers(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                trigger for item in rows for trigger in item["triggered_rules"]
            ).items()
        )
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _production_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]*", text):
        normalized = token.strip(".,;:()[]{}").casefold()
        if not normalized or normalized in STOPWORDS:
            continue
        if len(normalized) <= 2 and normalized not in {"ai", "ml", "cv"}:
            continue
        terms.append(token.strip(".,;:()[]{}"))
    if "大模型" in text:
        terms.append("LLM")
    if "检索增强" in text:
        terms.append("RAG")
    if "检索" in text:
        terms.append("retrieval")
    if "重排序" in text:
        terms.append("reranking")
    if "数据集" in text:
        terms.append("dataset")
    if "评测" in text:
        terms.append("benchmark")
    return _dedupe_strings(terms)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def _lexical_key(value: str) -> str:
    return "".join(_word_tokens(value))


def _simple_stem(value: str) -> str:
    token = _lexical_key(value)
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _explicit_acronyms(text: str) -> set[str]:
    values: set[str] = set()
    for long_form, acronym in re.findall(
        r"([A-Za-z][A-Za-z -]{3,80})\s*\(([A-Za-z0-9-]{2,12})\)", text
    ):
        initials = "".join(word[0] for word in _word_tokens(long_form))
        values.add(_lexical_key(acronym))
        if initials:
            values.add(initials)
    return values


def _text_initialisms(text: str) -> set[str]:
    words = _word_tokens(text)
    values: set[str] = set()
    for width in range(2, min(7, len(words) + 1)):
        for start in range(0, len(words) - width + 1):
            values.add("".join(item[0] for item in words[start : start + width]))
    return values


def _looks_like_acronym(value: str) -> bool:
    letters = "".join(character for character in value if character.isalpha())
    return 2 <= len(letters) <= 10 and letters.upper() == letters


def _exact_term_in_text(term: str, text: str) -> bool:
    normalized = term.casefold()
    target = text.casefold()
    if re.fullmatch(r"[a-z0-9+.#-]+", normalized):
        return re.search(
            rf"(?<![a-z0-9+.#-]){re.escape(normalized)}(?![a-z0-9+.#-])",
            target,
        ) is not None
    return normalized in target


def _load_queries(config: dict[str, Any], spec: dict[str, Any]) -> list[EvalQuery]:
    if config["dataset"] == "beir_scifact":
        crosswalk = spec.get("crosswalk_path")
        if not crosswalk:
            raise ValueError("SciFact audit requires evaluator crosswalk")
        return load_beir_scifact_enriched(
            str(config["dataset_source_path"]), crosswalk_path=str(crosswalk)
        )
    return load_dataset(str(config["dataset"]), path=str(config["dataset_source_path"]))


def _validate_config(config: dict[str, Any], spec: dict[str, Any]) -> None:
    expected = spec.get("dataset")
    if expected and config.get("dataset") != expected:
        raise ValueError("audit dataset config mismatch")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("audit requires current_rules query planning")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("audit requires current_rules ranking")
    if config.get("judgement_policy") != "current_rules":
        raise ValueError("audit requires current_rules judgement")
    if config.get("result_policy") != "highly_and_partial":
        raise ValueError("audit requires highly_and_partial result policy")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError("audit requires Top-20")
    if config.get("retrieval_mode") != "replay":
        raise ValueError("audit input must be a frozen Replay")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
            "enable_pubmed_related_expansion",
            "enable_prf",
            "enable_concept_projection",
            "enable_llm_constrained_rewrite",
        )
    ):
        raise ValueError("audit requires all optional strategies disabled")


def _matching_any(candidates: Sequence[Any], target: Any) -> list[Any]:
    profile = build_identity_profile(target)
    return [
        item
        for item in candidates
        if identity_evidence_from_profiles(
            build_identity_profile(item), profile
        ).equivalent
    ]


def _paper_has_source(paper: Paper, source: str) -> bool:
    return source in paper.sources


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _identity_sequence(values: Sequence[Any]) -> list[str | None]:
    return [canonical_paper_id(item) for item in values]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _average(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object:{path}")
    return payload


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case_id = str(payload.get("case_id") or "")
        if not case_id or case_id in rows:
            raise ValueError(f"invalid or duplicate case row:{path}:{case_id}")
        rows[case_id] = payload
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
