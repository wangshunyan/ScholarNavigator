"""Read-only regression gate for frozen ``current_rules`` Replay artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import (
    EvalGoldPaper,
    EvalMetricVersion,
    EvalQuery,
    LEGACY_GOLD_METRIC_VERSION,
)
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_title,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.datasets import load_beir_scifact_enriched, load_dataset
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.metrics import (
    average_metric_sets,
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
    paper_identifier_set,
    paper_title_year_key,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


SCHEMA_VERSION = "1"
GATE_NAME = "current_rules_frozen_replay_regression"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = ["openalex", "arxiv", "semantic_scholar", "pubmed"]
BASELINE_APPROVAL_TOKEN = "PROPOSE_CURRENT_RULES_REGRESSION_BASELINE"
NONDETERMINISTIC_CONFIG_FIELDS = {
    "started_at",
    "code",
    "runtime_code_hash",
    "resume_signature",
}
SET_LIKE_PATH_SUFFIXES = (
    ".candidate_identities",
    ".required_retrieval_keys",
    ".matched_gold_ids",
)


class RegressionGateError(RuntimeError):
    """Raised for malformed gate configuration, not for ordinary drift."""


def check_current_rules_regression(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the current semantic profile and compare it with the baseline."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest, require_baseline=True)
    observed = build_current_rules_profile(manifest)
    baseline_spec = manifest["baseline_profile"]
    baseline_path = _repo_path(baseline_spec["path"])
    baseline_sha = _sha256_file(baseline_path)
    baseline = _read_json(baseline_path)
    diffs = compare_profiles(baseline, observed)
    input_diffs = _input_fingerprint_diffs(manifest, observed)
    if baseline_sha != baseline_spec["sha256"]:
        input_diffs.insert(
            0,
            {
                "path": "$.baseline_profile.sha256",
                "kind": "value_changed",
                "expected": baseline_spec["sha256"],
                "actual": baseline_sha,
            },
        )
    all_diffs = [*input_diffs, *diffs]
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not all_diffs,
        "manifest_sha256": _sha256_file(manifest_file),
        "baseline_profile_sha256": baseline_sha,
        "observed_profile_sha256": _sha256_json(observed),
        "dataset_count": len(observed.get("datasets") or {}),
        "case_count": sum(
            int(item.get("summary_metrics", {}).get("case_count") or 0)
            for item in observed.get("datasets", {}).values()
        ),
        "drift_count": len(all_diffs),
        "drifts": all_diffs,
        "execution": observed["execution"],
        "official_score": False,
    }
    return observed, report


def build_current_rules_profile(
    manifest: Mapping[str, Any],
    *,
    metric_version: EvalMetricVersion = LEGACY_GOLD_METRIC_VERSION,
) -> dict[str, Any]:
    """Reconstruct candidates and metrics with current production components."""

    datasets: dict[str, Any] = {}
    for spec in manifest["datasets"]:
        label = str(spec["label"])
        if label in datasets:
            raise RegressionGateError(f"duplicate regression dataset:{label}")
        datasets[label] = _audit_dataset(spec, metric_version=metric_version)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "canonicalization": {
            "excluded_config_fields": sorted(NONDETERMINISTIC_CONFIG_FIELDS),
            "path_policy": "repository_relative_when_inside_repository",
            "candidate_identity": "all_stable_identifiers_plus_title_year_fallback",
            "float_policy": "exact_json_number",
        },
        "datasets": datasets,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": sum(
                int(item["snapshot_integrity"]["write_count"])
                for item in datasets.values()
            ),
            "snapshot_mode": "read_only",
            "external_record_in_gate": False,
        },
        "metric_semantics": {
            "candidate_recall": "macro_mean_over_queries_with_evaluable_gold",
            "recall_at_20": "macro_mean_over_queries_with_evaluable_gold",
            "f1_at_20": "macro_mean_over_queries_with_evaluable_gold",
            "official_score": False,
        },
    }


def compare_profiles(
    expected: Any,
    actual: Any,
    *,
    max_diffs: int = 100,
) -> list[dict[str, Any]]:
    """Return deterministic, minimally located semantic differences."""

    diffs: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if len(diffs) >= max_diffs:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys):
                diffs.append(
                    {
                        "path": f"{path}.{key}",
                        "kind": "missing_actual",
                        "expected": _compact(left[key]),
                        "actual": None,
                    }
                )
            for key in sorted(right_keys - left_keys):
                diffs.append(
                    {
                        "path": f"{path}.{key}",
                        "kind": "unexpected_actual",
                        "expected": None,
                        "actual": _compact(right[key]),
                    }
                )
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if any(path.endswith(suffix) for suffix in SET_LIKE_PATH_SUFFIXES):
                left_values = {_stable_json(item) for item in left}
                right_values = {_stable_json(item) for item in right}
                if left_values != right_values:
                    diffs.append(
                        {
                            "path": path,
                            "kind": "set_changed",
                            "removed": [
                                json.loads(item)
                                for item in sorted(left_values - right_values)
                            ][:20],
                            "added": [
                                json.loads(item)
                                for item in sorted(right_values - left_values)
                            ][:20],
                        }
                    )
                return
            if len(left) != len(right):
                diffs.append(
                    {
                        "path": path,
                        "kind": "length_changed",
                        "expected": len(left),
                        "actual": len(right),
                    }
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            diffs.append(
                {
                    "path": path,
                    "kind": "value_changed",
                    "expected": _compact(left),
                    "actual": _compact(right),
                }
            )

    visit(expected, actual, "$")
    return diffs[:max_diffs]


def canonicalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only declared non-semantic fields and normalize local paths."""

    canonical = copy.deepcopy(dict(config))
    for field in NONDETERMINISTIC_CONFIG_FIELDS:
        canonical.pop(field, None)
    for field in ("dataset_source_path",):
        if canonical.get(field):
            canonical[field] = _display_path(Path(str(canonical[field])))
    snapshot = canonical.get("snapshot")
    if isinstance(snapshot, dict) and snapshot.get("directory"):
        snapshot["directory"] = _display_path(Path(str(snapshot["directory"])))
    llm = canonical.get("llm")
    if isinstance(llm, dict):
        canonical["llm"] = {
            key: bool(llm.get(key))
            for key in (
                "llm_enabled",
                "requested",
                "query_understanding",
                "judgement",
                "semantic_query_planning",
                "constrained_query_rewrite",
            )
        }
    return canonical


def validate_current_rules_config(config: Mapping[str, Any]) -> list[str]:
    """Return every default-policy violation without consulting environment config."""

    violations: list[str] = []
    expected_values = {
        "sources": DEFAULT_SOURCES,
        "retrieval_mode": "replay",
        "query_planning_policy": "current_rules",
        "query_adapter_policy": "adaptive",
        "query_evolution_policy": "off",
        "ranking_policy": "current_rules",
        "judgement_policy": "current_rules",
        "result_policy": "highly_and_partial",
        "run_profile": "balanced",
        "top_k": 20,
    }
    for field, expected in expected_values.items():
        if config.get(field) != expected:
            violations.append(f"{field}:expected={expected!r}:actual={config.get(field)!r}")
    for field in (
        "enable_query_evolution",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_pubmed_related_expansion",
        "enable_prf",
        "enable_concept_projection",
        "enable_llm_constrained_rewrite",
        "enable_local_bm25",
    ):
        if bool(config.get(field, False)):
            violations.append(f"{field}:must_be_false")
    llm = config.get("llm") or {}
    for field in (
        "llm_enabled",
        "requested",
        "query_understanding",
        "judgement",
        "semantic_query_planning",
        "constrained_query_rewrite",
    ):
        if bool(llm.get(field, False)):
            violations.append(f"llm.{field}:must_be_false")
    if CURRENT_RULES_CONFIG.lexical_normalization_policy != "off":
        violations.append("runtime_judgement.lexical_normalization_policy:must_be_off")
    return sorted(violations)


def authorize_baseline_proposal(*, approval_token: str, reason: str) -> None:
    """Protect baseline capture behind a deliberate, auditable invocation."""

    if approval_token != BASELINE_APPROVAL_TOKEN:
        raise RegressionGateError("explicit baseline proposal approval is required")
    if len(reason.strip()) < 12:
        raise RegressionGateError("baseline proposal reason must be at least 12 characters")


def build_baseline_proposal(
    manifest_path: str | Path,
    *,
    approval_token: str,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a review-only proposal; never mutate the tracked baseline or manifest."""

    authorize_baseline_proposal(approval_token=approval_token, reason=reason)
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest, require_baseline=False)
    proposed = build_current_rules_profile(manifest)
    baseline_spec = manifest.get("baseline_profile") or {}
    baseline_path = _repo_path(baseline_spec["path"]) if baseline_spec.get("path") else None
    existing = _read_json(baseline_path) if baseline_path and baseline_path.is_file() else None
    diffs = compare_profiles(existing, proposed) if existing is not None else []
    audit = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "operation": "baseline_proposal_only",
        "reason": reason.strip(),
        "existing_baseline_sha256": (
            _sha256_json(existing) if existing is not None else None
        ),
        "proposed_baseline_sha256": _sha256_json(proposed),
        "drift_count": len(diffs),
        "drifts": diffs,
        "tracked_files_modified": False,
        "manual_review_required": True,
    }
    return proposed, audit


def write_gate_artifacts(
    output_dir: str | Path,
    *,
    observed: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "observed_profile.json", observed)
    _write_json(root / "regression_report.json", report)


def write_baseline_proposal(
    output_dir: str | Path,
    *,
    proposed: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "proposed_baseline.json", proposed)
    _write_json(root / "baseline_update_audit.json", audit)


def _audit_dataset(
    spec: Mapping[str, Any],
    *,
    metric_version: EvalMetricVersion,
) -> dict[str, Any]:
    label = str(spec["label"])
    run_dir = _repo_path(spec["run_dir"])
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    config = _read_json(config_path)
    violations = validate_current_rules_config(config)
    results = _rows_by_case(_read_jsonl(results_path))
    queries = _load_queries(config, spec)
    by_case = {query.query_id: query for query in queries}
    case_ids = [str(item) for item in config.get("case_ids") or []]
    if set(results) != set(case_ids):
        violations.append("results_case_set_mismatch")
    if any(case_id not in by_case for case_id in case_ids):
        violations.append("dataset_case_missing")

    snapshot_dir = _repo_path(spec["snapshot_dir"])
    tree_before = _tree_sha256(snapshot_dir)
    store = SnapshotStore(snapshot_dir)
    snapshot_manifest = store.read_manifest()
    group_name = str(spec["snapshot_group"])
    group = snapshot_manifest.groups.get(group_name)
    if group is None:
        violations.append(f"snapshot_group_missing:{group_name}")

    cases: dict[str, Any] = {}
    all_required_keys: list[str] = []
    all_metrics = []
    candidate_recalls: list[float] = []
    for case_id in case_ids:
        row = results.get(case_id)
        query = by_case.get(case_id)
        if row is None or query is None:
            cases[case_id] = {
                "status": "missing_input",
                "errors": ["missing_result_or_query"],
            }
            continue
        case = _audit_case(
            row,
            query,
            config,
            store,
            metric_version=metric_version,
        )
        cases[case_id] = case
        all_required_keys.extend(case["required_retrieval_keys"])
        if case["metrics"]["evaluable_gold_count"] > 0:
            all_metrics.append(case.pop("_metric"))
            candidate_recalls.append(float(case["metrics"]["candidate_recall"]))
        else:
            case.pop("_metric", None)

    unique_required = sorted(set(all_required_keys))
    group_keys = sorted(group.retrieval_keys) if group is not None else []
    group_reference_keys = sorted(group.reference_keys) if group is not None else []
    if group_keys != unique_required:
        violations.append("snapshot_group_required_keys_mismatch")
    if group_reference_keys:
        violations.append("unexpected_reference_keys_for_default_current_rules")
    tree_after = _tree_sha256(snapshot_dir)
    write_count = int(tree_before != tree_after)
    if write_count:
        violations.append("snapshot_tree_changed_during_gate")

    aggregate = average_metric_sets(all_metrics)
    summary = {
        "case_count": len(cases),
        "evaluable_case_count": len(all_metrics),
        "candidate_recall": (
            sum(candidate_recalls) / len(candidate_recalls)
            if candidate_recalls
            else 0.0
        ),
        "recall_at_20": aggregate.recall_at_k.get(20, 0.0),
        "f1_at_20": aggregate.f1_at_k.get(20, 0.0),
    }
    semantic_config = canonicalize_config(config)
    semantic_config["runtime_judgement_config"] = CURRENT_RULES_CONFIG.model_dump(
        mode="json"
    )
    stable_cases = {key: _without_private(value) for key, value in sorted(cases.items())}
    return {
        "label": label,
        "dataset": config.get("dataset"),
        "dataset_split": config.get("dataset_split"),
        "dataset_sha256": _sha256_file(_repo_path(config["dataset_source_path"])),
        "source_run": _display_path(run_dir),
        "source_run_inputs": {
            "config_sha256": _sha256_file(config_path),
            "results_sha256": _sha256_file(results_path),
        },
        "semantic_config": semantic_config,
        "semantic_config_sha256": _sha256_json(semantic_config),
        "configuration_violations": sorted(violations),
        "snapshot_integrity": {
            "directory": _display_path(snapshot_dir),
            "tree_sha256": tree_after,
            "file_count": sum(path.is_file() for path in snapshot_dir.rglob("*")),
            "required_retrieval_key_count": len(unique_required),
            "required_retrieval_keys": unique_required,
            "required_retrieval_keys_sha256": _sha256_json(unique_required),
            "group_required_key_parity": group_keys == unique_required,
            "missing_key_count": sum(
                len(case.get("snapshot_errors") or []) for case in stable_cases.values()
            ),
            "reference_key_count": len(group_reference_keys),
            "write_count": write_count,
        },
        "summary_metrics": summary,
        "cases": stable_cases,
        "semantic_hashes": {
            "candidate_identity_sha256": _sha256_json(
                {
                    key: value.get("candidate_identities", [])
                    for key, value in stable_cases.items()
                }
            ),
            "core_metrics_sha256": _sha256_json(
                {
                    "summary": summary,
                    "cases": {
                        key: value.get("metrics", {})
                        for key, value in stable_cases.items()
                    },
                }
            ),
            "gold_diagnostics_sha256": _sha256_json(
                {
                    key: value.get("gold_diagnostics", [])
                    for key, value in stable_cases.items()
                }
            ),
            "source_terminal_sha256": _sha256_json(
                {
                    key: value.get("source_terminals", [])
                    for key, value in stable_cases.items()
                }
            ),
        },
    }


def _audit_case(
    row: Mapping[str, Any],
    query: EvalQuery,
    config: Mapping[str, Any],
    store: SnapshotStore,
    *,
    metric_version: EvalMetricVersion,
) -> dict[str, Any]:
    errors: list[str] = []
    snapshots = {
        str(item.get("stage")): item
        for item in row.get("stage_diagnostics", {}).get("snapshots", [])
    }
    initial = snapshots.get("initial_retrieval") or {}
    raw: list[Paper] = []
    required_keys: list[str] = []
    terminals: list[dict[str, Any]] = []
    seen: set[str] = set()
    snapshot_errors: list[dict[str, str]] = []
    for call in initial.get("retrieval_calls") or []:
        key = str(call.get("snapshot_key") or "")
        terminal = {
            "source": str(call.get("source") or ""),
            "query_sha256": _sha256_text(str(call.get("adapted_query") or "")),
            "logical_call_executed": bool(call.get("logical_call_executed")),
            "recorded_terminal_status": str(call.get("terminal_status") or ""),
            "snapshot_key": key or None,
            "snapshot_terminal_status": None,
        }
        if call.get("logical_call_executed") and key:
            if key not in seen:
                required_keys.append(key)
                seen.add(key)
            try:
                entry = store.read_retrieval(key)
            except Exception as exc:  # Snapshot exceptions are normalized below.
                snapshot_errors.append(
                    {"snapshot_key": key, "error_type": type(exc).__name__}
                )
            else:
                terminal["snapshot_terminal_status"] = entry.status
                if entry.source != call.get("source"):
                    errors.append(f"snapshot_source_mismatch:{key}")
                if entry.adapted_query != call.get("adapted_query"):
                    errors.append(f"snapshot_query_mismatch:{key}")
                if entry.status != call.get("terminal_status"):
                    errors.append(f"snapshot_terminal_mismatch:{key}")
                if entry.status == "success":
                    raw.extend(item.model_copy(deep=True) for item in entry.papers)
        terminals.append(terminal)

    candidates = deduplicate_papers(raw)
    candidate_limit = int(config["budgets"]["max_candidate_papers"])
    if len(candidates) > candidate_limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=candidate_limit,
            source_order=[str(item) for item in config["sources"]],
        )
    analysis_payload = row.get("stage_diagnostics", {}).get(
        "initial_query_planning", {}
    ).get("query_analysis")
    if not analysis_payload:
        raise RegressionGateError(f"missing frozen QueryAnalysis:{query.query_id}")
    analysis = QueryAnalysis.model_validate(analysis_payload)
    judgements = judge_papers(
        analysis,
        candidates,
        use_llm=False,
        config=CURRENT_RULES_CONFIG,
    )
    ranked = rerank_papers(analysis, judgements, top_k=len(judgements))
    top_k = int(config["top_k"])
    returned = select_ranked_results(
        {"ranked_papers": ranked[:top_k]},
        policy=str(config["result_policy"]),
    )
    denominator = evaluable_gold_count(
        query.gold_papers,
        metric_version=metric_version,
    )
    candidate_matched = matched_paper_ids(
        candidates,
        query.gold_papers,
        metric_version=metric_version,
    )
    returned_matched = matched_paper_ids(
        returned,
        query.gold_papers,
        k=top_k,
        metric_version=metric_version,
    )
    metric = evaluate_ranking(
        returned,
        query.gold_papers,
        [top_k],
        metric_version=metric_version,
    )
    metrics = {
        "evaluable_gold_count": denominator,
        "candidate_recall": (
            len(candidate_matched) / denominator if denominator else None
        ),
        "recall_at_20": metric.recall_at_k.get(top_k, 0.0),
        "f1_at_20": metric.f1_at_k.get(top_k, 0.0),
        "candidate_matched_gold_ids": sorted(candidate_matched),
        "matched_gold_ids": sorted(returned_matched),
    }
    return {
        "status": str(row.get("status") or ""),
        "query_sha256": _sha256_text(query.query),
        "query_analysis_sha256": _sha256_json(analysis.model_dump(mode="json")),
        "required_retrieval_keys": sorted(required_keys),
        "source_terminals": terminals,
        "source_terminal_counts": dict(
            sorted(
                Counter(
                    f"{item['source']}:{item['snapshot_terminal_status']}"
                    for item in terminals
                    if item["logical_call_executed"]
                ).items()
            )
        ),
        "snapshot_errors": snapshot_errors,
        "errors": sorted(errors),
        "candidate_count": len(candidates),
        "candidate_identities": sorted(_paper_identity(item) for item in candidates),
        "returned_count": len(returned),
        "returned_identities": [_paper_identity(item) for item in returned],
        "metrics": metrics,
        "gold_diagnostics": _gold_diagnostics(
            query.gold_papers,
            candidates,
            ranked,
            returned,
            metric_version=metric_version,
        ),
        "_metric": metric,
    }


def _gold_diagnostics(
    gold_papers: Sequence[EvalGoldPaper],
    candidates: Sequence[Any],
    ranked: Sequence[Any],
    returned: Sequence[Any],
    *,
    metric_version: EvalMetricVersion,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for relation_index, gold in enumerate(gold_papers):
        if gold.relevance_grade <= 0:
            continue
        evaluable = evaluable_gold_count(
            [gold],
            metric_version=metric_version,
        ) > 0
        candidate_matches = _matching_positions(candidates, gold)
        ranked_matches = _matching_positions(ranked, gold)
        returned_matches = _matching_positions(returned, gold)
        terminal = (
            "identity_unavailable"
            if not evaluable
            else "not_retrieved"
            if not candidate_matches
            else "returned"
            if returned_matches
            else "candidate_filtered_or_ranked_out"
        )
        output.append(
            {
                "relation_index": relation_index,
                "gold_id": canonical_paper_id(gold),
                "evaluable": evaluable,
                "candidate_positions": candidate_matches,
                "ranked_positions": ranked_matches,
                "returned_positions": returned_matches,
                "terminal": terminal,
            }
        )
    return output


def _matching_positions(values: Sequence[Any], gold: EvalGoldPaper) -> list[int]:
    target = build_identity_profile(gold)
    return [
        index
        for index, item in enumerate(values, 1)
        if identity_evidence_from_profiles(
            build_identity_profile(getattr(item, "paper", item)), target
        ).equivalent
    ]


def _paper_identity(paper: Any) -> str:
    value = getattr(paper, "paper", paper)
    identifiers = sorted(paper_identifier_set(value))
    title_year = paper_title_year_key(value)
    payload = {
        "identifiers": identifiers,
        "title_year": title_year,
    }
    if not identifiers and not title_year:
        payload["title"] = normalize_title(str(getattr(value, "title", "") or ""))
    return _stable_json(payload)


def _load_queries(config: Mapping[str, Any], spec: Mapping[str, Any]) -> list[EvalQuery]:
    dataset_path = _repo_path(config["dataset_source_path"])
    if config["dataset"] == "beir_scifact":
        crosswalk = spec.get("crosswalk_path")
        if not crosswalk:
            raise RegressionGateError("SciFact regression requires evaluator crosswalk")
        return load_beir_scifact_enriched(
            dataset_path,
            crosswalk_path=_repo_path(crosswalk),
        )
    return load_dataset(str(config["dataset"]), path=dataset_path)


def _input_fingerprint_diffs(
    manifest: Mapping[str, Any], observed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for spec in manifest["datasets"]:
        label = str(spec["label"])
        dataset = observed["datasets"].get(label) or {}
        expected = spec["frozen_hashes"]
        actual = {
            "dataset_sha256": dataset.get("dataset_sha256"),
            "config_sha256": dataset.get("source_run_inputs", {}).get("config_sha256"),
            "results_sha256": dataset.get("source_run_inputs", {}).get("results_sha256"),
            "snapshot_tree_sha256": dataset.get("snapshot_integrity", {}).get("tree_sha256"),
        }
        if spec.get("crosswalk_path"):
            actual["crosswalk_sha256"] = _sha256_file(_repo_path(spec["crosswalk_path"]))
        for name, expected_value in expected.items():
            if actual.get(name) != expected_value:
                diffs.append(
                    {
                        "path": f"$.inputs.{label}.{name}",
                        "kind": "value_changed",
                        "expected": expected_value,
                        "actual": actual.get(name),
                    }
                )
        expected_profile = spec.get("expected_profile") or {}
        actual_profile = {
            "required_retrieval_key_count": dataset.get(
                "snapshot_integrity", {}
            ).get("required_retrieval_key_count"),
            "required_retrieval_keys_sha256": dataset.get(
                "snapshot_integrity", {}
            ).get("required_retrieval_keys_sha256"),
            "semantic_config_sha256": dataset.get("semantic_config_sha256"),
            "semantic_hashes": dataset.get("semantic_hashes"),
            "summary_metrics": dataset.get("summary_metrics"),
        }
        for drift in compare_profiles(expected_profile, actual_profile):
            drift["path"] = drift["path"].replace(
                "$",
                f"$.inputs.{label}.expected_profile",
                1,
            )
            diffs.append(drift)
    return diffs


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("gate") != GATE_NAME:
        raise RegressionGateError("unexpected regression manifest")
    if not manifest.get("datasets"):
        raise RegressionGateError("regression manifest has no datasets")
    if require_baseline:
        baseline = manifest.get("baseline_profile") or {}
        if not baseline.get("path") or not baseline.get("sha256"):
            raise RegressionGateError("regression baseline is not frozen")
    canonicalization = manifest.get("canonicalization") or {}
    if set(canonicalization.get("excluded_config_fields") or []) != NONDETERMINISTIC_CONFIG_FIELDS:
        raise RegressionGateError("non-deterministic field exclusions drifted")


def _rows_by_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in output:
            raise RegressionGateError(f"invalid or duplicate case id:{case_id}")
        output[case_id] = dict(row)
    return output


def _without_private(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _compact(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 20:
        return {"count": len(value), "head": value[:5]}
    if isinstance(value, Mapping) and len(value) > 20:
        keys = sorted(value)[:5]
        return {"count": len(value), "head": {key: value[key] for key in keys}}
    return value


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
