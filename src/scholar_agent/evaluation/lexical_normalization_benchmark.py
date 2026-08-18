"""Pure-Replay benchmark for the default-off lexical normalization policy."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import (
    CURRENT_RULES_CONFIG,
    LEXICAL_NORMALIZATION_V1_CONFIG,
)
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.evaluation_schemas import (
    DEDUPLICATED_GOLD_METRIC_VERSION,
    EvalGoldPaper,
    EvalQuery,
)
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import JudgementResult, QueryAnalysis
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _reconstruct_candidates,
)
from scholar_agent.evaluation.metrics import (
    average_metric_sets,
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
)
from scholar_agent.evaluation.relevance_filter_audit import (
    _load_queries,
    _read_json,
    _read_rows,
    _sha256,
    _tree_sha256,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_SCHEMA_VERSION = "1"
METRIC_VERSION = DEDUPLICATED_GOLD_METRIC_VERSION
RETURN_CATEGORIES = {"highly_relevant", "partially_relevant"}
Transition = Literal[
    "recovered_gold",
    "lost_gold",
    "gold_rank_changed",
    "benchmark_non_gold_admitted",
    "benchmark_non_gold_removed",
    "benchmark_non_gold_rank_changed",
    "unchanged",
]


def classify_candidate_transition(
    *,
    is_gold: bool,
    baseline_returned: bool,
    experiment_returned: bool,
    baseline_rank: int,
    experiment_rank: int,
) -> Transition:
    if is_gold and not baseline_returned and experiment_returned:
        return "recovered_gold"
    if is_gold and baseline_returned and not experiment_returned:
        return "lost_gold"
    if not is_gold and not baseline_returned and experiment_returned:
        return "benchmark_non_gold_admitted"
    if not is_gold and baseline_returned and not experiment_returned:
        return "benchmark_non_gold_removed"
    if baseline_rank != experiment_rank:
        return "gold_rank_changed" if is_gold else "benchmark_non_gold_rank_changed"
    return "unchanged"


def assert_candidate_identity_parity(
    baseline: Sequence[Any], experiment: Sequence[Any]
) -> None:
    baseline_ids = [_identity_instance_key(item) for item in baseline]
    experiment_ids = [_identity_instance_key(item) for item in experiment]
    if baseline_ids != experiment_ids:
        raise ValueError("baseline and experiment candidate identity/order differ")


def run_lexical_normalization_benchmark(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    dataset_aggregates: dict[str, Any] = {}
    inputs: dict[str, Any] = {"manifest_sha256": _sha256(manifest_file)}
    for spec in manifest["frozen_inputs"]:
        cases, candidates, aggregate, fingerprints = _audit_dataset(spec)
        label = str(spec["label"])
        case_rows.extend(cases)
        candidate_rows.extend(candidates)
        dataset_aggregates[label] = aggregate
        inputs[label] = fingerprints
    aggregate = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": "lexical_normalization_v1",
        "implementation_base_commit": manifest["implementation_base_commit"],
        "inputs": inputs,
        "datasets": dataset_aggregates,
        "cross_dataset_acceptance": _cross_dataset_acceptance(dataset_aggregates),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_access": "post_candidate_and_judgement_reconstruction_only",
            "production_default": "off",
        },
    }
    return case_rows, candidate_rows, aggregate


def write_lexical_normalization_benchmark(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_comparison.jsonl", case_rows)
    _write_jsonl(root / "candidate_diagnostics.jsonl", candidate_rows)
    _write_json(root / "aggregate.json", aggregate)


def _audit_dataset(
    spec: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    label = str(spec["label"])
    run_root = Path(str(spec["run_dir"])).expanduser().resolve()
    snapshot_root = Path(str(spec["snapshot_dir"])).expanduser().resolve()
    config = _read_json(run_root / "config.json")
    _validate_frozen_config(config, spec)
    results = _read_rows(run_root / "results.jsonl")
    prior_filtered_path = Path(
        str(spec["prior_filtered_gold_path"])
    ).expanduser().resolve()
    prior_lexical_false_negatives = {
        (str(item["case_id"]), str(item["candidate_id"]))
        for item in _read_jsonl(prior_filtered_path)
        if item.get("dataset") == label
        and item.get("primary_root_cause")
        == "morphology_or_abbreviation_mismatch"
    }
    queries = _load_queries(config, spec)
    by_case = {item.query_id: item for item in queries}
    case_ids = [str(item) for item in config["case_ids"]]
    if set(results) != set(case_ids) or any(item not in by_case for item in case_ids):
        raise ValueError(f"frozen case set mismatch:{label}")
    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        case, candidates, state = _audit_case(
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
            prior_lexical_false_negatives=prior_lexical_false_negatives,
        )
        case_rows.append(case)
        candidate_rows.extend(candidates)
        states.append(state)
    aggregate = _aggregate_dataset(label, case_rows, candidate_rows, states)
    fingerprints = {
        "config_sha256": _sha256(run_root / "config.json"),
        "results_sha256": _sha256(run_root / "results.jsonl"),
        "snapshot_tree_sha256": _tree_sha256(snapshot_root),
        "prior_filtered_gold_sha256": _sha256(prior_filtered_path),
        "snapshot_file_count": sum(
            path.is_file() for path in snapshot_root.rglob("*")
        ),
    }
    return case_rows, candidate_rows, aggregate, fingerprints


def _audit_case(
    *,
    label: str,
    case_order: int,
    eval_query: EvalQuery,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    required_candidate_source: str | None,
    prior_lexical_false_negatives: set[tuple[str, str]],
    include_instance_identity_keys: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if row.get("status") != "succeeded":
        raise ValueError(f"frozen case failed:{label}:{eval_query.query_id}")
    snapshots = {
        str(item["stage"]): item for item in row["stage_diagnostics"]["snapshots"]
    }
    required_stages = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
        "final_returned",
    }
    if not required_stages.issubset(snapshots):
        raise ValueError(f"missing frozen stage:{label}:{eval_query.query_id}")
    initial = snapshots["initial_retrieval"]
    dedup = snapshots["initial_deduplicated"]
    candidates = _reconstruct_candidates(initial, dedup, config, store)
    analysis = QueryAnalysis.model_validate(
        row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    baseline_judgements = judge_papers(
        analysis,
        candidates,
        use_llm=False,
        config=CURRENT_RULES_CONFIG,
    )
    experiment_judgements = judge_papers(
        analysis,
        candidates,
        use_llm=False,
        config=LEXICAL_NORMALIZATION_V1_CONFIG,
    )
    assert_candidate_identity_parity(
        [item.paper for item in baseline_judgements],
        [item.paper for item in experiment_judgements],
    )
    _validate_frozen_judgements(
        candidates,
        baseline_judgements,
        snapshots["initial_judged"],
        label=label,
        case_id=eval_query.query_id,
    )
    baseline_ranked = rerank_papers(
        analysis, baseline_judgements, top_k=len(candidates)
    )
    experiment_ranked = rerank_papers(
        analysis, experiment_judgements, top_k=len(candidates)
    )
    _validate_frozen_ranking(
        baseline_ranked,
        snapshots["initial_reranked"],
        label=label,
        case_id=eval_query.query_id,
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
        raise ValueError(f"frozen returned mismatch:{label}:{eval_query.query_id}")

    baseline_by_key = {
        _identity_instance_key(item): item for item in baseline_ranked
    }
    experiment_by_key = {
        _identity_instance_key(item): item for item in experiment_ranked
    }
    baseline_judgement_by_key = {
        _identity_instance_key(item.paper): item for item in baseline_judgements
    }
    experiment_judgement_by_key = {
        _identity_instance_key(item.paper): item for item in experiment_judgements
    }
    if set(baseline_by_key) != set(experiment_by_key):
        raise ValueError(f"ranked candidate identity set mismatch:{label}")
    candidate_audits: list[dict[str, Any]] = []
    for key in baseline_by_key:
        baseline_ranked_item = baseline_by_key[key]
        experiment_ranked_item = experiment_by_key[key]
        baseline_judgement = baseline_judgement_by_key[key]
        experiment_judgement = experiment_judgement_by_key[key]
        gold_ids = matched_paper_ids(
            [baseline_ranked_item.paper],
            eval_query.gold_papers,
            k=1,
            metric_version=METRIC_VERSION,
        )
        lexical_matches = (
            experiment_judgement.feature_vector.lexical_normalization_matches
            if experiment_judgement.feature_vector is not None
            else []
        )
        is_gold = bool(gold_ids)
        baseline_is_returned = bool(
            _matching_any(baseline_returned, baseline_ranked_item.paper)
        )
        experiment_is_returned = bool(
            _matching_any(experiment_returned, baseline_ranked_item.paper)
        )
        transition = classify_candidate_transition(
            is_gold=is_gold,
            baseline_returned=baseline_is_returned,
            experiment_returned=experiment_is_returned,
            baseline_rank=baseline_ranked_item.rank,
            experiment_rank=experiment_ranked_item.rank,
        )
        focus_source = (
            required_candidate_source is None
            or required_candidate_source in baseline_ranked_item.paper.sources
        )
        candidate_id = canonical_paper_id(baseline_ranked_item.paper)
        previous_lexical_false_negative = bool(
            is_gold
            and focus_source
            and (eval_query.query_id, str(candidate_id))
            in prior_lexical_false_negatives
        )
        if is_gold or lexical_matches or transition != "unchanged":
            candidate_audit = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "dataset": label,
                    "case_order": case_order,
                    "case_id": eval_query.query_id,
                    "candidate_id": candidate_id,
                    "sources": sorted(baseline_ranked_item.paper.sources),
                    "is_gold": is_gold,
                    "matched_gold_ids": gold_ids,
                    "focus_source_candidate": focus_source,
                    "previous_lexical_false_negative": (
                        previous_lexical_false_negative
                    ),
                    "lexical_matches": [
                        item.model_dump(mode="json") for item in lexical_matches
                    ],
                    "baseline": _candidate_state(
                        baseline_judgement,
                        baseline_ranked_item.rank,
                        baseline_is_returned,
                    ),
                    "experiment": _candidate_state(
                        experiment_judgement,
                        experiment_ranked_item.rank,
                        experiment_is_returned,
                    ),
                    "score_delta": round(
                        experiment_judgement.score - baseline_judgement.score,
                        6,
                    ),
                    "rank_delta": baseline_ranked_item.rank
                    - experiment_ranked_item.rank,
                    "transition": transition,
                }
            if include_instance_identity_keys:
                candidate_audit["candidate_identity_key"] = key
            candidate_audits.append(candidate_audit)

    baseline_metrics = _case_metrics(
        baseline_returned,
        eval_query.gold_papers,
        include_instance_identity_keys=include_instance_identity_keys,
    )
    experiment_metrics = _case_metrics(
        experiment_returned,
        eval_query.gold_papers,
        include_instance_identity_keys=include_instance_identity_keys,
    )
    recovered_gold_ids = sorted(
        set(experiment_metrics["matched_gold_ids"])
        - set(baseline_metrics["matched_gold_ids"])
    )
    lost_gold_ids = sorted(
        set(baseline_metrics["matched_gold_ids"])
        - set(experiment_metrics["matched_gold_ids"])
    )
    comparison = _metric_comparison(baseline_metrics, experiment_metrics)
    focus_candidates = [
        item
        for item in candidates
        if required_candidate_source is None
        or required_candidate_source in item.sources
    ]
    case = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": label,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "candidate_identity_parity": True,
        "candidate_count": len(candidates),
        "focus_source_candidate_count": len(focus_candidates),
        "evaluable_gold_count": evaluable_gold_count(
            eval_query.gold_papers,
            metric_version=METRIC_VERSION,
        ),
        "candidate_gold_count": len(
            matched_paper_ids(
                focus_candidates,
                eval_query.gold_papers,
                metric_version=METRIC_VERSION,
            )
        ),
        "baseline": baseline_metrics,
        "experiment": experiment_metrics,
        "comparison": comparison,
        "recovered_unique_gold_ids": recovered_gold_ids,
        "lost_unique_gold_ids": lost_gold_ids,
        "lexical_match_count": sum(
            len(item["lexical_matches"]) for item in candidate_audits
        ),
        "recovered_gold_count": sum(
            item["transition"] == "recovered_gold" for item in candidate_audits
        ),
        "lost_gold_count": sum(
            item["transition"] == "lost_gold" for item in candidate_audits
        ),
        "benchmark_non_gold_admitted_count": sum(
            item["transition"] == "benchmark_non_gold_admitted"
            for item in candidate_audits
        ),
    }
    state = {
        "gold": eval_query.gold_papers,
        "evaluable_gold_count": evaluable_gold_count(
            eval_query.gold_papers,
            metric_version=METRIC_VERSION,
        ),
        "baseline_returned": baseline_returned,
        "experiment_returned": experiment_returned,
    }
    return case, candidate_audits, state


def _aggregate_dataset(
    label: str,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    states: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    evaluable_states = [item for item in states if item["evaluable_gold_count"]]
    baseline_metrics = average_metric_sets(
        [
            evaluate_ranking(
                item["baseline_returned"],
                item["gold"],
                [20],
                metric_version=METRIC_VERSION,
            )
            for item in evaluable_states
        ]
    )
    experiment_metrics = average_metric_sets(
        [
            evaluate_ranking(
                item["experiment_returned"],
                item["gold"],
                [20],
                metric_version=METRIC_VERSION,
            )
            for item in evaluable_states
        ]
    )
    comparison_counts = Counter(str(item["comparison"]) for item in case_rows)
    transition_counts = Counter(str(item["transition"]) for item in candidate_rows)
    lexical_matches = [
        match for item in candidate_rows for match in item["lexical_matches"]
    ]
    previous = [
        item for item in candidate_rows if item["previous_lexical_false_negative"]
    ]
    recovered = [item for item in previous if item["transition"] == "recovered_gold"]
    lost = [item for item in candidate_rows if item["transition"] == "lost_gold"]
    admitted = [
        item
        for item in candidate_rows
        if item["transition"] == "benchmark_non_gold_admitted"
    ]
    baseline = {
        "recall_at_20": baseline_metrics.recall_at_k.get(20, 0.0),
        "f1_at_20": baseline_metrics.f1_at_k.get(20, 0.0),
        "returned_gold_relation_count": sum(
            len(
                matched_paper_ids(
                    item["baseline_returned"],
                    item["gold"],
                    k=20,
                    metric_version=METRIC_VERSION,
                )
            )
            for item in states
        ),
    }
    experiment = {
        "recall_at_20": experiment_metrics.recall_at_k.get(20, 0.0),
        "f1_at_20": experiment_metrics.f1_at_k.get(20, 0.0),
        "returned_gold_relation_count": sum(
            len(
                matched_paper_ids(
                    item["experiment_returned"],
                    item["gold"],
                    k=20,
                    metric_version=METRIC_VERSION,
                )
            )
            for item in states
        ),
    }
    return {
        "dataset": label,
        "case_count": len(case_rows),
        "evaluable_case_count": len(evaluable_states),
        "identity_unavailable_case_count": len(case_rows) - len(evaluable_states),
        "candidate_identity_parity_count": sum(
            bool(item["candidate_identity_parity"]) for item in case_rows
        ),
        "evaluable_gold_relation_count": sum(
            int(item["evaluable_gold_count"]) for item in case_rows
        ),
        "candidate_recall": (
            sum(
                int(item["candidate_gold_count"])
                / int(item["evaluable_gold_count"])
                for item in case_rows
                if item["evaluable_gold_count"]
            )
            / len(evaluable_states)
            if evaluable_states
            else 0.0
        ),
        "focus_candidate_gold_relation_count": sum(
            int(item["candidate_gold_count"]) for item in case_rows
        ),
        "baseline": baseline,
        "experiment": experiment,
        "delta": {
            "recall_at_20": experiment["recall_at_20"] - baseline["recall_at_20"],
            "f1_at_20": experiment["f1_at_20"] - baseline["f1_at_20"],
            "gold_relations": experiment["returned_gold_relation_count"]
            - baseline["returned_gold_relation_count"],
        },
        "query_outcomes": {
            name: comparison_counts[name]
            for name in ("improved", "tied", "regressed")
        },
        "lexical_diagnostics": {
            "candidate_with_new_match_count": sum(
                bool(item["lexical_matches"]) for item in candidate_rows
            ),
            "new_match_count": len(lexical_matches),
            "by_facet": dict(
                sorted(Counter(str(item["facet"]) for item in lexical_matches).items())
            ),
            "by_field": dict(
                sorted(Counter(str(item["field"]) for item in lexical_matches).items())
            ),
            "total_score_impact": sum(
                float(item["score_impact"]) for item in lexical_matches
            ),
        },
        "previous_lexical_false_negatives": {
            "count": len(previous),
            "recovered_count": len(recovered),
            "not_recovered_count": len(previous) - len(recovered),
        },
        "ranking_and_admission": {
            "lost_gold_matching_candidate_count": len(lost),
            "recovered_unique_gold_relation_count": sum(
                len(item["recovered_unique_gold_ids"]) for item in case_rows
            ),
            "lost_unique_gold_relation_count": sum(
                len(item["lost_unique_gold_ids"]) for item in case_rows
            ),
            "benchmark_non_gold_admitted_count": len(admitted),
            "transition_counts": dict(sorted(transition_counts.items())),
        },
    }


def _cross_dataset_acceptance(datasets: Mapping[str, Any]) -> dict[str, Any]:
    required = ("scifact", "auto_dev", "auto_val")
    if any(name not in datasets for name in required):
        raise ValueError("all three fixed datasets are required")
    scifact = datasets["scifact"]
    auto = [datasets["auto_dev"], datasets["auto_val"]]
    scifact_improved = (
        scifact["delta"]["recall_at_20"] > 0
        or scifact["delta"]["f1_at_20"] > 0
    )
    auto_non_regression = all(
        item["delta"]["recall_at_20"] >= 0
        and item["delta"]["f1_at_20"] >= 0
        for item in auto
    )
    return {
        "scifact_improved": scifact_improved,
        "auto_dev_and_val_non_regression": auto_non_regression,
        "recommend_continue": scifact_improved and auto_non_regression,
        "default_remains_off": True,
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("experiment") != "lexical_normalization_v1":
        raise ValueError("unexpected experiment manifest")
    policy = manifest.get("policy") or {}
    if policy.get("default") != "off":
        raise ValueError("lexical normalization must remain default-off")
    if policy.get("experimental_value") != "lexical_normalization_v1":
        raise ValueError("unexpected lexical normalization policy")
    if manifest.get("frozen_invariants", {}).get("snapshot_write_count") != 0:
        raise ValueError("benchmark must prohibit Snapshot writes")


def _validate_frozen_config(config: dict[str, Any], spec: dict[str, Any]) -> None:
    if config.get("dataset") != spec.get("dataset"):
        raise ValueError("frozen dataset mismatch")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("requires current_rules queries")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("requires current_rules ranking")
    if config.get("judgement_policy") != "current_rules":
        raise ValueError("requires current_rules judgement")
    if config.get("retrieval_mode") != "replay":
        raise ValueError("requires frozen Replay input")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError("requires Top-20")
    frozen_rules = dict(config.get("judgement_config") or {})
    expected_rules = CURRENT_RULES_CONFIG.model_dump(mode="json")
    expected_rules.pop("lexical_normalization_policy", None)
    if frozen_rules != expected_rules:
        raise ValueError("frozen Judgement parameters differ from current_rules")


def _validate_frozen_judgements(
    candidates: Sequence[Paper],
    baseline: Sequence[JudgementResult],
    stage: dict[str, Any],
    *,
    label: str,
    case_id: str,
) -> None:
    frozen = list(stage.get("candidates") or [])
    if len(candidates) != len(frozen) or len(baseline) != len(frozen):
        raise ValueError(f"frozen judgement length mismatch:{label}:{case_id}")
    for paper, judgement, diagnostic in zip(
        candidates, baseline, frozen, strict=True
    ):
        if not _equivalent(paper, diagnostic):
            raise ValueError(f"frozen judgement identity mismatch:{label}:{case_id}")
        if (
            judgement.score != float(diagnostic["judgement_score"])
            or judgement.category != diagnostic["category"]
        ):
            raise ValueError(f"default-off judgement changed:{label}:{case_id}")


def _validate_frozen_ranking(
    baseline: Sequence[Any],
    stage: dict[str, Any],
    *,
    label: str,
    case_id: str,
) -> None:
    frozen = list(stage.get("candidates") or [])
    if len(baseline) != len(frozen):
        raise ValueError(f"frozen ranking length mismatch:{label}:{case_id}")
    for live, diagnostic in zip(baseline, frozen, strict=True):
        if (
            not _equivalent(live.paper, diagnostic)
            or live.rank != int(diagnostic["rank"])
            or live.category != diagnostic["category"]
            or live.final_score != float(diagnostic["final_score"])
        ):
            raise ValueError(f"default-off ranking changed:{label}:{case_id}")


def _candidate_state(
    judgement: JudgementResult,
    rank: int,
    returned: bool,
) -> dict[str, Any]:
    return {
        "score": judgement.score,
        "category": judgement.category,
        "rank": rank,
        "returned": returned,
    }


def _case_metrics(
    ranked: Sequence[Any],
    gold: Sequence[EvalGoldPaper],
    *,
    include_instance_identity_keys: bool = False,
) -> dict[str, Any]:
    metrics = evaluate_ranking(
        ranked,
        gold,
        [20],
        metric_version=METRIC_VERSION,
    )
    matched = matched_paper_ids(
        ranked,
        gold,
        k=20,
        metric_version=METRIC_VERSION,
    )
    output = {
        "recall_at_20": metrics.recall_at_k.get(20, 0.0),
        "f1_at_20": metrics.f1_at_k.get(20, 0.0),
        "matched_gold_count": len(matched),
        "matched_gold_ids": matched,
        "returned_ids": [_identity_key(item) for item in ranked],
    }
    if include_instance_identity_keys:
        output["returned_identity_keys"] = _identity_sequence(ranked)
    return output


def _metric_comparison(
    baseline: Mapping[str, Any], experiment: Mapping[str, Any]
) -> str:
    baseline_pair = (
        float(baseline["recall_at_20"]),
        float(baseline["f1_at_20"]),
    )
    experiment_pair = (
        float(experiment["recall_at_20"]),
        float(experiment["f1_at_20"]),
    )
    if experiment_pair > baseline_pair:
        return "improved"
    if experiment_pair < baseline_pair:
        return "regressed"
    return "tied"


def _matching_any(candidates: Sequence[Any], target: Any) -> list[Any]:
    profile = build_identity_profile(target)
    return [
        item
        for item in candidates
        if identity_evidence_from_profiles(
            build_identity_profile(item), profile
        ).equivalent
    ]


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _identity_key(item: Any) -> str:
    value = canonical_paper_id(item)
    if value:
        return value
    paper = getattr(item, "paper", item)
    title = getattr(paper, "title", None)
    if title is None and isinstance(paper, Mapping):
        title = paper.get("title")
    return "title:" + str(title or "").casefold()


def _identity_sequence(values: Sequence[Any]) -> list[str]:
    return [_identity_instance_key(item) for item in values]


def _identity_instance_key(item: Any) -> str:
    """Disambiguate conflicting records that share one display identifier."""

    profile = build_identity_profile(item)
    payload = json.dumps(
        {
            "identifiers": sorted(profile.identifiers),
            "title": profile.title,
            "year": profile.year,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    display = _identity_key(item)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{display}|profile:{digest}"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
