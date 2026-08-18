"""Pure-Replay separability audit for frozen candidate-ranking signals.

Candidate provenance and scores are reconstructed before evaluator gold is
loaded into the comparison. The module never invokes a connector or mutates a
Snapshot; it only validates and reads frozen Replay artifacts.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_title,
)
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _reconstruct_candidates,
    _reconstruct_ranking,
)
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    evaluable_gold_count,
    matched_paper_ids,
)
from scholar_agent.evaluation.relevance_filter_audit import (
    _load_queries,
    _read_json,
    _read_rows,
    _sha256,
    _tree_sha256,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_SCHEMA_VERSION = "1"
SIGNALS = (
    "existing_composite_score",
    "best_reciprocal_rank",
    "support_list_count",
    "support_source_count",
)
PROVENANCE_SIGNALS = SIGNALS[1:]
SignalName = Literal[
    "existing_composite_score",
    "best_reciprocal_rank",
    "support_list_count",
    "support_source_count",
]


def extract_candidate_signal_row(
    candidate: Mapping[str, Any],
    *,
    score_breakdown: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract fixed provenance signals without consulting evaluator gold."""

    candidate_id = _candidate_key(candidate)
    raw_provenance = list(candidate.get("provenance") or [])
    by_list: dict[tuple[str, str], dict[str, Any]] = {}
    incomplete_count = 0
    for raw in raw_provenance:
        source = str(raw.get("source") or "").strip()
        query = str(raw.get("adapted_query") or "").strip()
        rank = _positive_int(raw.get("source_rank"))
        if not source or not query or rank is None:
            incomplete_count += 1
        if not source or not query:
            continue
        key = (source, query)
        prior = by_list.get(key)
        row = {
            "source": source,
            "adapted_query": query,
            "source_rank": rank,
            "origin_kind": str(raw.get("origin_kind") or "unknown"),
            "origin_subquery": str(raw.get("origin_subquery") or ""),
            "purpose": str(raw.get("purpose") or "unknown"),
            "adaptation_strategy": str(
                raw.get("adaptation_strategy") or "unknown"
            ),
        }
        if prior is None or (
            rank is not None
            and (
                prior["source_rank"] is None
                or rank < int(prior["source_rank"])
            )
        ):
            by_list[key] = row

    lists = sorted(
        by_list.values(),
        key=lambda item: (
            str(item["source"]),
            str(item["adapted_query"]),
            int(item["source_rank"] or 10**9),
        ),
    )
    valid_ranks = [int(item["source_rank"]) for item in lists if item["source_rank"]]
    best_rank = min(valid_ranks) if valid_ranks else None
    if not raw_provenance:
        provenance_status = "missing"
    elif incomplete_count:
        provenance_status = "incomplete"
    else:
        provenance_status = "complete"
    sources = sorted({str(item["source"]) for item in lists})
    original_lists = sum(item["purpose"] == "original_query" for item in lists)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "title": str(candidate.get("title") or ""),
        "year": candidate.get("year"),
        "identifiers": dict(candidate.get("identifiers") or {}),
        "sources": sorted(str(value) for value in candidate.get("sources") or []),
        "filter_state": {
            "judgement_score": candidate.get("judgement_score"),
            "category": candidate.get("category"),
            "matched_terms": list(candidate.get("matched_terms") or []),
            "warnings": list(candidate.get("warnings") or []),
            "judgement_score_components": dict(
                (candidate.get("judgement_features") or {}).get(
                    "score_components"
                )
                or {}
            ),
        },
        "existing_composite_score": float(candidate.get("final_score") or 0.0),
        "existing_rank": int(candidate["rank"]),
        "score_breakdown": dict(score_breakdown or {}),
        "provenance_status": provenance_status,
        "raw_provenance_count": len(raw_provenance),
        "incomplete_provenance_count": incomplete_count,
        "provenance_lists": lists,
        "best_source_rank": best_rank,
        "best_reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
        "support_list_count": len(lists),
        "support_source_count": len(sources),
        "support_sources": sources,
        "original_query_list_count": original_lists,
        "derived_query_list_count": len(lists) - original_lists,
    }


def rank_candidate_signal_rows(
    rows: Sequence[dict[str, Any]], signal: SignalName
) -> list[dict[str, Any]]:
    """Apply one pre-declared signal with a stable, gold-free tie break."""

    if signal not in SIGNALS:
        raise ValueError(f"unsupported candidate signal:{signal}")
    if signal == "existing_composite_score":
        return sorted(
            rows,
            key=lambda item: (
                int(item["existing_rank"]),
                str(item["candidate_id"]),
            ),
        )
    return sorted(
        rows,
        key=lambda item: (
            -float(item[signal]),
            str(item["candidate_id"]),
        ),
    )


def evaluate_case_signals(
    rows: Sequence[dict[str, Any]],
    gold: Sequence[EvalGoldPaper],
    *,
    cutoffs: Sequence[int] = (20, 50, 100),
) -> dict[str, Any]:
    """Evaluate the four frozen signals with the existing identity matcher."""

    denominator = evaluable_gold_count(gold)
    rankings = {
        signal: rank_candidate_signal_rows(rows, signal)  # type: ignore[arg-type]
        for signal in SIGNALS
    }
    signal_rows: dict[str, Any] = {}
    baseline = rankings["existing_composite_score"]
    baseline_top = {
        str(item["candidate_id"]): item for item in baseline[:20]
    }
    for signal, ranked in rankings.items():
        captures: dict[str, Any] = {}
        for cutoff in cutoffs:
            matched = matched_paper_ids(ranked, gold, k=cutoff)
            captures[str(cutoff)] = {
                "matched_gold_relation_count": len(matched),
                "matched_gold_ids": matched,
                "gold_capture": len(matched) / denominator if denominator else None,
            }
        exact_ranks = _gold_match_ranks(ranked, gold)
        current_top = {
            str(item["candidate_id"]): item for item in ranked[:20]
        }
        entered_ids = sorted(set(current_top) - set(baseline_top))
        exited_ids = sorted(set(baseline_top) - set(current_top))
        signal_rows[signal] = {
            "captures": captures,
            "gold_match_ranks": exact_ranks,
            "gold_rank_distribution": _rank_distribution(exact_ranks, denominator),
            "top_20_candidate_ids": list(current_top),
            "top_20_swaps_vs_existing": {
                "entered_candidate_ids": entered_ids,
                "exited_candidate_ids": exited_ids,
                "entered_candidate_count": len(entered_ids),
                "exited_candidate_count": len(exited_ids),
                "entered_gold_candidate_count": sum(
                    _candidate_matches_any(current_top[value], gold)
                    for value in entered_ids
                ),
                "exited_gold_candidate_count": sum(
                    _candidate_matches_any(baseline_top[value], gold)
                    for value in exited_ids
                ),
            },
        }
    return {
        "evaluable_gold_count": denominator,
        "signals": signal_rows,
        "rankings": rankings,
    }


def run_candidate_ranking_signal_audit(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    cutoffs = [int(value) for value in manifest["cutoffs"]]
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {"manifest_sha256": _sha256(manifest_file)}
    for spec in manifest["frozen_inputs"]:
        cases, candidates, fingerprint = _audit_dataset(spec, cutoffs=cutoffs)
        case_rows.extend(cases)
        candidate_rows.extend(candidates)
        inputs[str(spec["label"])] = fingerprint
    case_rows.sort(key=lambda item: (str(item["dataset"]), int(item["case_order"])))
    candidate_rows.sort(
        key=lambda item: (
            str(item["dataset"]),
            int(item["case_order"]),
            int(item["existing_rank"]),
            str(item["candidate_id"]),
        )
    )
    datasets = {
        str(spec["label"]): _aggregate_dataset(
            str(spec["label"]),
            [row for row in case_rows if row["dataset"] == spec["label"]],
            [row for row in candidate_rows if row["dataset"] == spec["label"]],
            cutoffs=cutoffs,
        )
        for spec in manifest["frozen_inputs"]
    }
    aggregate = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "candidate_ranking_signal_separability",
        "implementation_base_commit": manifest["implementation_base_commit"],
        "inputs": inputs,
        "candidate_scope": manifest["candidate_scope"],
        "signals": [item["name"] for item in manifest["signals"]],
        "cutoffs": cutoffs,
        "datasets": datasets,
        "cross_dataset_consistency": _cross_dataset_consistency(datasets, cutoffs),
        "interpretation": {
            "unmatched_candidate_semantics": (
                "benchmark-unmatched only; not reliable human negatives"
            ),
            "provenance_signal_is_not_a_fitted_model": True,
            "production_change_recommended_by_audit_only": False,
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "score_components_reconstructed_from_frozen_judgements": True,
            "gold_accessed_after_candidate_reconstruction": True,
        },
    }
    return case_rows, candidate_rows, aggregate


def write_candidate_ranking_signal_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_signal_audit.jsonl", case_rows)
    _write_jsonl(root / "candidate_signal_audit.jsonl", candidate_rows)
    _write_json(root / "aggregate.json", aggregate)


def _audit_dataset(
    spec: dict[str, Any], *, cutoffs: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label = str(spec["label"])
    run_root = Path(str(spec["run_dir"])).expanduser().resolve()
    snapshot_root = Path(str(spec["snapshot_dir"])).expanduser().resolve()
    config = _read_json(run_root / "config.json")
    _validate_frozen_config(config, spec)
    results = _read_rows(run_root / "results.jsonl")
    queries = _load_queries(config, spec)
    query_by_id = {item.query_id: item for item in queries}
    case_ids = [str(value) for value in config["case_ids"]]
    if len(case_ids) != int(spec["case_count"]):
        raise ValueError(f"manifest case count mismatch:{label}")
    if set(results) != set(case_ids) or any(item not in query_by_id for item in case_ids):
        raise ValueError(f"frozen case set mismatch:{label}")
    store = SnapshotStore(snapshot_root)
    cases: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        case, rows = _audit_case(
            label=label,
            case_order=case_order,
            eval_query=query_by_id[case_id],
            row=results[case_id],
            config=config,
            store=store,
            cutoffs=cutoffs,
        )
        cases.append(case)
        candidates.extend(rows)
    return cases, candidates, {
        "config_sha256": _sha256(run_root / "config.json"),
        "results_sha256": _sha256(run_root / "results.jsonl"),
        "snapshot_tree_sha256": _tree_sha256(snapshot_root),
        "snapshot_file_count": sum(path.is_file() for path in snapshot_root.rglob("*")),
    }


def _audit_case(
    *,
    label: str,
    case_order: int,
    eval_query: EvalQuery,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    cutoffs: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if row.get("status") != "succeeded":
        raise ValueError(f"frozen case failed:{label}:{eval_query.query_id}")
    snapshots = {
        str(item["stage"]): item for item in row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
    }
    if not required.issubset(snapshots):
        raise ValueError(f"missing frozen stage:{label}:{eval_query.query_id}")
    if any(snapshots[name].get("status") != "completed" for name in required):
        raise ValueError(f"incomplete frozen stage:{label}:{eval_query.query_id}")
    initial = snapshots["initial_retrieval"]
    reconstructed = _reconstruct_candidates(
        initial, snapshots["initial_deduplicated"], config, store
    )
    ranked = _reconstruct_ranking(
        reconstructed,
        snapshots["initial_judged"],
        snapshots["initial_reranked"],
        row,
    )
    frozen = list(snapshots["initial_reranked"].get("candidates") or [])
    if len(ranked) != len(frozen):
        raise ValueError(f"candidate length mismatch:{label}:{eval_query.query_id}")

    candidate_rows: list[dict[str, Any]] = []
    for diagnostic, ranked_paper in zip(frozen, ranked, strict=True):
        if not _equivalent(diagnostic, ranked_paper.paper):
            raise ValueError(f"candidate identity mismatch:{label}:{eval_query.query_id}")
        extracted = extract_candidate_signal_row(
            diagnostic,
            score_breakdown=ranked_paper.score_breakdown.model_dump(mode="json"),
        )
        extracted.update(
            {
                "dataset": label,
                "case_order": case_order,
                "case_id": eval_query.query_id,
            }
        )
        candidate_rows.append(extracted)
    ids = [str(item["candidate_id"]) for item in candidate_rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate unified candidate identity:{label}:{eval_query.query_id}")

    # Gold enters only after every candidate signal and score component exists.
    evaluated = evaluate_case_signals(
        candidate_rows, eval_query.gold_papers, cutoffs=cutoffs
    )
    rankings = evaluated.pop("rankings")
    rank_maps = {
        signal: {
            str(item["candidate_id"]): rank
            for rank, item in enumerate(values, 1)
        }
        for signal, values in rankings.items()
    }
    for candidate in candidate_rows:
        candidate["is_benchmark_gold"] = _candidate_matches_any(
            candidate, eval_query.gold_papers
        )
        candidate["matched_gold_ids"] = matched_paper_ids(
            [candidate], eval_query.gold_papers, k=1
        )
        candidate["signal_ranks"] = {
            signal: rank_maps[signal][str(candidate["candidate_id"])]
            for signal in SIGNALS
        }

    executed = [
        call
        for call in initial.get("retrieval_calls") or []
        if call.get("logical_call_executed")
    ]
    terminals = Counter(
        str(call.get("terminal_status") or "missing") for call in executed
    )
    all_executed_success = bool(executed) and terminals == {
        "success": len(executed)
    }
    all_provenance_complete = all(
        item["provenance_status"] == "complete" for item in candidate_rows
    )
    strict_reasons: list[str] = []
    if not all_executed_success:
        strict_reasons.append("executed_retrieval_terminal_not_all_success")
    if not all_provenance_complete:
        strict_reasons.append("candidate_provenance_incomplete")
    strict = not strict_reasons
    baseline_signals = evaluated["signals"]["existing_composite_score"]
    comparisons: dict[str, Any] = {}
    for signal in PROVENANCE_SIGNALS:
        comparisons[signal] = {
            str(cutoff): _outcome(
                evaluated["signals"][signal]["captures"][str(cutoff)][
                    "matched_gold_relation_count"
                ],
                baseline_signals["captures"][str(cutoff)][
                    "matched_gold_relation_count"
                ],
            )
            for cutoff in cutoffs
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": label,
        "case_order": case_order,
        "case_id": eval_query.query_id,
        "case_status": str(row.get("status")),
        "candidate_count": len(candidate_rows),
        "evaluable_gold_count": evaluated["evaluable_gold_count"],
        "provenance_complete_candidate_count": sum(
            item["provenance_status"] == "complete" for item in candidate_rows
        ),
        "provenance_incomplete_or_missing_candidate_count": sum(
            item["provenance_status"] != "complete" for item in candidate_rows
        ),
        "executed_retrieval_call_count": len(executed),
        "executed_retrieval_terminals": dict(sorted(terminals.items())),
        "strict_comparable": strict,
        "strict_exclusion_reasons": strict_reasons,
        "signals": evaluated["signals"],
        "provenance_signal_outcomes_vs_existing": comparisons,
    }, candidate_rows


def _aggregate_dataset(
    label: str,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    return {
        "dataset": label,
        "case_count": len(case_rows),
        "candidate_count": len(candidate_rows),
        "provenance_completeness": {
            "complete": sum(
                item["provenance_status"] == "complete" for item in candidate_rows
            ),
            "incomplete": sum(
                item["provenance_status"] == "incomplete" for item in candidate_rows
            ),
            "missing": sum(
                item["provenance_status"] == "missing" for item in candidate_rows
            ),
        },
        "benchmark_unmatched_candidate_count": sum(
            not item["is_benchmark_gold"] for item in candidate_rows
        ),
        "full_observed": _aggregate_scope(case_rows, cutoffs=cutoffs),
        "strict_comparable": _aggregate_scope(
            [item for item in case_rows if item["strict_comparable"]],
            cutoffs=cutoffs,
        ),
    }


def _aggregate_scope(
    rows: Sequence[dict[str, Any]], *, cutoffs: Sequence[int]
) -> dict[str, Any]:
    evaluable = [row for row in rows if int(row["evaluable_gold_count"]) > 0]
    signals: dict[str, Any] = {}
    for signal in SIGNALS:
        captures: dict[str, Any] = {}
        outcomes: dict[str, Any] = {}
        all_ranks: list[int] = []
        total_gold = sum(int(row["evaluable_gold_count"]) for row in evaluable)
        for cutoff in cutoffs:
            matched = sum(
                int(
                    row["signals"][signal]["captures"][str(cutoff)][
                        "matched_gold_relation_count"
                    ]
                )
                for row in evaluable
            )
            macro = (
                sum(
                    float(
                        row["signals"][signal]["captures"][str(cutoff)][
                            "gold_capture"
                        ]
                    )
                    for row in evaluable
                )
                / len(evaluable)
                if evaluable
                else None
            )
            captures[str(cutoff)] = {
                "macro_gold_capture": macro,
                "micro_gold_capture": matched / total_gold if total_gold else None,
                "matched_gold_relation_count": matched,
            }
            if signal != "existing_composite_score":
                counts = Counter(
                    row["provenance_signal_outcomes_vs_existing"][signal][
                        str(cutoff)
                    ]
                    for row in evaluable
                )
                outcomes[str(cutoff)] = {
                    key: counts[key]
                    for key in ("improved", "tied", "regressed")
                }
        for row in evaluable:
            all_ranks.extend(
                int(value)
                for value in row["signals"][signal]["gold_match_ranks"]
            )
        swap_rows = [
            row["signals"][signal]["top_20_swaps_vs_existing"]
            for row in evaluable
        ]
        signals[signal] = {
            "captures": captures,
            "query_outcomes_vs_existing": outcomes,
            "gold_rank_distribution": _rank_distribution(all_ranks, total_gold),
            "matched_gold_rank_mean": (
                statistics.fmean(all_ranks) if all_ranks else None
            ),
            "matched_gold_rank_median": (
                statistics.median(all_ranks) if all_ranks else None
            ),
            "top_20_swaps_vs_existing": {
                key: sum(int(item[key]) for item in swap_rows)
                for key in (
                    "entered_candidate_count",
                    "exited_candidate_count",
                    "entered_gold_candidate_count",
                    "exited_gold_candidate_count",
                )
            },
        }
    return {
        "case_count": len(rows),
        "evaluable_case_count": len(evaluable),
        "evaluable_gold_relation_count": sum(
            int(row["evaluable_gold_count"]) for row in evaluable
        ),
        "signals": signals,
    }


def _cross_dataset_consistency(
    datasets: Mapping[str, Any], cutoffs: Sequence[int]
) -> dict[str, Any]:
    required = ("scifact_local_bm25", "auto_dev", "auto_val")
    if any(name not in datasets for name in required):
        raise ValueError("all three fixed datasets are required")
    output: dict[str, Any] = {}
    for signal in PROVENANCE_SIGNALS:
        evidence = []
        for dataset in required:
            scope = datasets[dataset]["strict_comparable"]
            current = scope["signals"]["existing_composite_score"]["captures"]
            alternative = scope["signals"][signal]["captures"]
            deltas = {
                str(cutoff): (
                    alternative[str(cutoff)]["macro_gold_capture"]
                    - current[str(cutoff)]["macro_gold_capture"]
                    if alternative[str(cutoff)]["macro_gold_capture"] is not None
                    and current[str(cutoff)]["macro_gold_capture"] is not None
                    else None
                )
                for cutoff in cutoffs
            }
            evidence.append(
                {
                    "dataset": dataset,
                    "strict_evaluable_case_count": scope[
                        "evaluable_case_count"
                    ],
                    "macro_gold_capture_deltas": deltas,
                }
            )
        values = [
            delta
            for item in evidence
            for delta in item["macro_gold_capture_deltas"].values()
            if delta is not None
        ]
        has_all_evidence = all(
            item["strict_evaluable_case_count"] > 0 for item in evidence
        )
        non_degrading = (
            has_all_evidence and bool(values) and all(value >= 0 for value in values)
        )
        output[signal] = {
            "datasets": evidence,
            "strict_evidence_in_all_datasets": has_all_evidence,
            "non_degrading_at_all_cutoffs_in_all_datasets": non_degrading,
            "improves_at_least_one_dataset_cutoff": any(
                value > 0 for value in values
            ),
            "stable_value_supported": (
                non_degrading and any(value > 0 for value in values)
            ),
        }
    return output


def _gold_match_ranks(
    ranked: Sequence[dict[str, Any]], gold: Sequence[EvalGoldPaper]
) -> list[int]:
    ranks: list[int] = []
    previous = 0
    for rank in range(1, len(ranked) + 1):
        current = len(matched_paper_ids(ranked, gold, k=rank))
        ranks.extend([rank] * max(0, current - previous))
        previous = current
    return ranks


def _rank_distribution(ranks: Sequence[int], denominator: int) -> dict[str, int]:
    return {
        "1_20": sum(value <= 20 for value in ranks),
        "21_50": sum(20 < value <= 50 for value in ranks),
        "51_100": sum(50 < value <= 100 for value in ranks),
        "over_100": sum(value > 100 for value in ranks),
        "candidate_miss": max(0, denominator - len(ranks)),
    }


def _candidate_matches_any(
    candidate: Mapping[str, Any], gold: Sequence[EvalGoldPaper]
) -> bool:
    return bool(matched_paper_ids([candidate], gold, k=1))


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    value = canonical_paper_id(candidate)
    if value:
        return value
    normalized = normalize_title(str(candidate.get("title") or ""))
    year = candidate.get("year")
    if not normalized:
        digest = hashlib.sha256(
            json.dumps(candidate, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"anonymous:{digest}"
    return f"title:{normalized}:{year if year is not None else 'unknown'}"


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _outcome(alternative: int, existing: int) -> str:
    if alternative > existing:
        return "improved"
    if alternative < existing:
        return "regressed"
    return "tied"


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("audit") != "candidate_ranking_signal_separability":
        raise ValueError("unexpected candidate signal audit manifest")
    names = [item.get("name") for item in manifest.get("signals") or []]
    if names != list(SIGNALS):
        raise ValueError("candidate signals or order drifted")
    if [int(value) for value in manifest.get("cutoffs") or []] != [20, 50, 100]:
        raise ValueError("candidate signal cutoffs drifted")
    execution = manifest.get("execution") or {}
    if any(
        int(execution.get(field, -1)) != 0
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise ValueError("candidate signal audit must remain pure Replay")


def _validate_frozen_config(config: dict[str, Any], spec: dict[str, Any]) -> None:
    if config.get("dataset") != spec.get("dataset"):
        raise ValueError("frozen dataset mismatch")
    for field in ("query_planning_policy", "ranking_policy", "judgement_policy"):
        if config.get(field) != "current_rules":
            raise ValueError(f"candidate signal audit requires current_rules:{field}")
    if (
        config.get("retrieval_mode") != "replay"
        or int(config.get("top_k") or 0) != 20
    ):
        raise ValueError("candidate signal audit requires Replay Top-20")
    enabled = (
        "enable_query_evolution",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_pubmed_related_expansion",
        "enable_prf",
        "enable_concept_projection",
        "enable_llm_constrained_rewrite",
        "enable_local_bm25_original_deepening",
    )
    if any(bool(config.get(field)) for field in enabled):
        raise ValueError("candidate signal audit requires experiments disabled")
    if config.get("lexical_normalization_policy") not in (None, "off"):
        raise ValueError("candidate signal audit requires lexical normalization off")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
