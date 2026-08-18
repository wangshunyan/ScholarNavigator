"""Deterministic offline audit of AutoScholarQuery query independence."""

from __future__ import annotations

import hashlib
import html
import json
import re
import socket
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.core.evaluation_schemas import EvalQuery
from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.datasets.auto_scholar_query import load_auto_scholar_query


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
AUDIT_NAME = "autoscholar_query_independence_v1"
GATE_NAME = "autoscholar_query_independence_regression"
SCHEMA_VERSION = "1"
PROTOCOL_VERSION = "autoscholar-query-independence-protocol-v1"
COMPONENT_NAMESPACE = "autoscholar-query-component-v1"
EDGE_TYPES = (
    "normalized_exact_query",
    "lexical_near_duplicate",
    "shared_gold_identity_cluster",
)
EXCLUSIVE_STRATA = ("auto_dev", "auto_val", "record160_only", "remainder")
FROZEN_MEMBERSHIPS = ("auto_dev", "auto_val", "autoscholar_record160")
_PUNCTUATION_RE = re.compile(r"[\W_]+", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


class QueryIndependenceAuditError(RuntimeError):
    """Raised when a frozen independence-audit contract is invalid."""


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def normalize_query_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return " ".join(_PUNCTUATION_RE.sub(" ", text).split())


def informative_query_tokens(value: str, protocol: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = normalize_query_text(value)
    stopwords = {str(item) for item in protocol["stopwords"]}
    return tuple(sorted({token for token in _TOKEN_RE.findall(normalized) if token not in stopwords}))


def lexical_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def build_independence_graph(
    queries: Sequence[EvalQuery],
    identity_rows: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, Sequence[str]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build query assignments, graph edges, and transitive components."""

    _validate_protocol(protocol)
    query_by_id = {query.query_id: query for query in queries}
    if len(query_by_id) != len(queries):
        raise QueryIndependenceAuditError("duplicate query ID")
    query_ids = sorted(query_by_id)
    if set(memberships) != set(query_ids):
        raise QueryIndependenceAuditError("membership/query set mismatch")
    normalized = {
        query_id: normalize_query_text(query_by_id[query_id].query)
        for query_id in query_ids
    }
    tokens = {
        query_id: informative_query_tokens(query_by_id[query_id].query, protocol)
        for query_id in query_ids
    }
    edge_evidence: dict[tuple[str, str], dict[str, Any]] = {}

    def edge(left: str, right: str) -> dict[str, Any]:
        pair = tuple(sorted((left, right)))
        return edge_evidence.setdefault(
            pair,
            {
                "left_query_id": pair[0],
                "right_query_id": pair[1],
                "edge_types": set(),
                "near_duplicate_similarity": None,
                "shared_gold_identity_cluster_ids": set(),
            },
        )

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        exact_groups[normalized[query_id]].append(query_id)
    for group in exact_groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                edge(left, right)["edge_types"].add("normalized_exact_query")

    minimum_tokens = int(protocol["near_duplicate"]["minimum_informative_token_count_per_query"])
    threshold = float(protocol["near_duplicate"]["threshold_inclusive"])
    for left_index, left in enumerate(query_ids):
        if len(tokens[left]) < minimum_tokens:
            continue
        for right in query_ids[left_index + 1 :]:
            if normalized[left] == normalized[right] or len(tokens[right]) < minimum_tokens:
                continue
            similarity = lexical_jaccard(tokens[left], tokens[right])
            if similarity >= threshold:
                item = edge(left, right)
                item["edge_types"].add("lexical_near_duplicate")
                item["near_duplicate_similarity"] = similarity

    cluster_queries: dict[str, set[str]] = defaultdict(set)
    for item in identity_rows:
        query_id = str(item["query_id"])
        if query_id not in query_by_id:
            raise QueryIndependenceAuditError(f"unknown identity query:{query_id}")
        cluster_queries[str(item["identity_cluster_id"])].add(query_id)
    for cluster_id, members in sorted(cluster_queries.items()):
        group = sorted(members)
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                item = edge(left, right)
                item["edge_types"].add("shared_gold_identity_cluster")
                item["shared_gold_identity_cluster_ids"].add(cluster_id)

    edges: list[dict[str, Any]] = []
    dsu = _DisjointSet(query_ids)
    for pair in sorted(edge_evidence):
        item = edge_evidence[pair]
        dsu.union(pair[0], pair[1])
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "left_query_id": item["left_query_id"],
                "right_query_id": item["right_query_id"],
                "edge_types": sorted(item["edge_types"]),
                "near_duplicate_similarity": item["near_duplicate_similarity"],
                "shared_gold_identity_cluster_ids": sorted(
                    item["shared_gold_identity_cluster_ids"]
                ),
            }
        )

    component_members: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        component_members[dsu.find(query_id)].append(query_id)
    edge_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in edges:
        edge_by_component[dsu.find(str(item["left_query_id"]))].append(item)

    components: list[dict[str, Any]] = []
    component_by_query: dict[str, dict[str, Any]] = {}
    for root in sorted(component_members):
        members = sorted(component_members[root])
        member_edges = edge_by_component.get(root, [])
        component_id = _component_id(members)
        partitions = sorted({_exclusive_stratum(memberships[item]) for item in members})
        overlapping_members = sorted(
            item
            for item in members
            if len(set(memberships[item]) & set(FROZEN_MEMBERSHIPS)) > 1
        )
        contaminated_reasons: list[str] = []
        if overlapping_members:
            contaminated_reasons.append("same_query_reused_across_frozen_memberships")
        if len(partitions) > 1:
            contaminated_reasons.append("component_connects_exclusive_strata")
        edge_counts = Counter(
            edge_type
            for item in member_edges
            for edge_type in item["edge_types"]
        )
        component = {
            "schema_version": SCHEMA_VERSION,
            "component_id": component_id,
            "query_count": len(members),
            "query_ids": members,
            "edge_count": len(member_edges),
            "edge_type_counts": {name: edge_counts[name] for name in EDGE_TYPES},
            "exclusive_strata": partitions,
            "frozen_memberships": sorted(
                {
                    membership
                    for query_id in members
                    for membership in memberships[query_id]
                    if membership in FROZEN_MEMBERSHIPS
                }
            ),
            "overlapping_membership_query_ids": overlapping_members,
            "cross_stratum_contaminated": bool(contaminated_reasons),
            "contamination_reasons": contaminated_reasons,
        }
        components.append(component)
        for query_id in members:
            component_by_query[query_id] = component

    neighbor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    shared_clusters_by_query: dict[str, set[str]] = defaultdict(set)
    for item in edges:
        for query_id in (str(item["left_query_id"]), str(item["right_query_id"])):
            for edge_type in item["edge_types"]:
                neighbor_counts[query_id][edge_type] += 1
            shared_clusters_by_query[query_id].update(
                str(value) for value in item["shared_gold_identity_cluster_ids"]
            )
    assignments = [
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": query_id,
            "normalized_query": normalized[query_id],
            "informative_token_count": len(tokens[query_id]),
            "component_id": component_by_query[query_id]["component_id"],
            "component_query_count": component_by_query[query_id]["query_count"],
            "cross_stratum_contaminated": component_by_query[query_id][
                "cross_stratum_contaminated"
            ],
            "contamination_reasons": component_by_query[query_id][
                "contamination_reasons"
            ],
            "frozen_memberships": sorted(set(memberships[query_id])),
            "exclusive_stratum": _exclusive_stratum(memberships[query_id]),
            "neighbor_counts": {
                name: neighbor_counts[query_id][name] for name in EDGE_TYPES
            },
            "shared_gold_identity_cluster_ids": sorted(
                shared_clusters_by_query[query_id]
            ),
        }
        for query_id in query_ids
    ]
    return assignments, edges, sorted(components, key=lambda item: item["component_id"])


def build_query_independence_audit(
    manifest: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Build the frozen 1000-query independence audit and diagnostic strata."""

    _validate_manifest(manifest, require_baseline=False)
    protocol = _read_json(_repo_path(manifest["protocol"]["path"]))
    _validate_protocol(protocol)
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0}
    with _forbid_network(attempts):
        queries = load_auto_scholar_query(_repo_path(manifest["dataset"]["path"]))
        identity_rows = _read_jsonl(
            _repo_path(manifest["gold_identity_baseline"]["path"])
        )
        small_cases = _read_jsonl(
            _repo_path(manifest["frozen_replays"]["existing65"]["path"])
        )
        record_cases = _read_jsonl(
            _repo_path(manifest["frozen_replays"]["record160"]["path"])
        )
        memberships = _build_memberships(queries, small_cases, record_cases)
        assignments, edges, components = build_independence_graph(
            queries, identity_rows, memberships, protocol
        )
        metric_rows, metric_summary = _metric_diagnostics(
            assignments, small_cases, record_cases
        )
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "retrieval_invoked": False,
        "metric_scope": "frozen internal diagnostics only; not an official score",
    }
    if any(
        int(execution[field])
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise QueryIndependenceAuditError(f"offline invariant failed:{execution}")
    summary = _summarize(
        assignments,
        edges,
        components,
        metric_rows,
        metric_summary,
        record_cases,
        execution,
    )
    _validate_closure(assignments, edges, components, metric_rows, summary, manifest)
    return assignments, edges, components, metric_rows, summary


def write_query_independence_audit(
    output_dir: str | Path,
    assignments: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "query_assignments.jsonl", assignments)
    _write_jsonl(root / "edges.jsonl", edges)
    _write_jsonl(root / "components.jsonl", components)
    _write_jsonl(root / "metric_diagnostics.jsonl", metric_rows)
    _write_json(root / "summary.json", summary)


def check_query_independence_regression(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path).expanduser().resolve())
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    drifts: list[dict[str, Any]] = []
    try:
        _validate_manifest(manifest, require_baseline=True)
        observed_values = build_query_independence_audit(manifest)
        names = (
            "query_assignments",
            "edges",
            "components",
            "metric_diagnostics",
            "summary",
        )
        observed = dict(zip(names, observed_values, strict=True))
        for name in names:
            path = _repo_path(manifest["baseline"][f"{name}_path"])
            expected_hash = str(manifest["baseline"][f"{name}_sha256"])
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                drifts.append(
                    {
                        "kind": "baseline_fingerprint_drift",
                        "path": f"$.baseline.{name}",
                        "expected": expected_hash,
                        "observed": actual_hash,
                    }
                )
            expected = _read_json(path) if name == "summary" else _read_jsonl(path)
            drifts.extend(
                compare_profiles(
                    {name: expected}, {name: observed[name]}, max_diffs=100
                )
            )
        write_query_independence_audit(output / "observed", *observed_values)
    except (QueryIndependenceAuditError, ValueError, KeyError) as exc:
        drifts.append(
            {
                "kind": "input_protocol_or_cluster_drift",
                "path": "$",
                "expected": "frozen independence audit contract",
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


def _build_memberships(
    queries: Sequence[EvalQuery],
    small_cases: Sequence[Mapping[str, Any]],
    record_cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    query_ids = {query.query_id for query in queries}
    dev = {str(row["case_id"]) for row in small_cases if row["dataset"] == "auto_dev"}
    val = {str(row["case_id"]) for row in small_cases if row["dataset"] == "auto_val"}
    record = {str(row["case_id"]) for row in record_cases}
    unknown = (dev | val | record) - query_ids
    if unknown:
        raise QueryIndependenceAuditError(f"frozen cases missing from dataset:{sorted(unknown)}")
    output: dict[str, list[str]] = {}
    for query_id in sorted(query_ids):
        labels: list[str] = []
        if query_id in dev:
            labels.append("auto_dev")
        if query_id in val:
            labels.append("auto_val")
        if query_id in record:
            labels.append("autoscholar_record160")
        output[query_id] = labels
    return output


def _exclusive_stratum(memberships: Sequence[str]) -> str:
    labels = set(memberships)
    if "auto_dev" in labels:
        return "auto_dev"
    if "auto_val" in labels:
        return "auto_val"
    if "autoscholar_record160" in labels:
        return "record160_only"
    return "remainder"


def _component_id(query_ids: Sequence[str]) -> str:
    payload = COMPONENT_NAMESPACE + "\n" + "\n".join(sorted(query_ids))
    return "component:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _metric_diagnostics(
    assignments: Sequence[Mapping[str, Any]],
    small_cases: Sequence[Mapping[str, Any]],
    record_cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment_by_id = {str(item["query_id"]): item for item in assignments}
    rows: list[dict[str, Any]] = []
    for source, cases in (("existing65", small_cases), ("record160", record_cases)):
        for item in cases:
            case_id = str(item["case_id"])
            is_auto = str(item["dataset"]).startswith("auto")
            assignment = assignment_by_id.get(case_id) if is_auto else None
            if is_auto and assignment is None:
                raise QueryIndependenceAuditError(f"missing assignment:{case_id}")
            included = (
                True
                if source == "existing65"
                else bool(item.get("included_main_analysis"))
            )
            contaminated = bool(
                assignment and assignment["cross_stratum_contaminated"]
            )
            denominator = int(item.get("evaluable_gold_count") or 0)
            candidate_recall = (
                int(item.get("candidate_gold_count") or 0) / denominator
                if denominator
                else None
            )
            baseline = item.get("baseline") or {}
            experiment = item.get("experiment") or {}
            if included and (
                baseline.get("recall_at_20") is None
                or baseline.get("f1_at_20") is None
                or experiment.get("recall_at_20") is None
                or experiment.get("f1_at_20") is None
            ):
                raise QueryIndependenceAuditError(
                    f"included metric row is incomplete:{case_id}"
                )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": source,
                    "case_id": case_id,
                    "dataset": item["dataset"],
                    "included_full": included,
                    "included_decontaminated": included and not contaminated,
                    "exclusion_reason": (
                        "cross_stratum_contaminated_component"
                        if included and contaminated
                        else (
                            "no_successful_source"
                            if not included and source == "record160"
                            else None
                        )
                    ),
                    "component_id": assignment["component_id"] if assignment else None,
                    "component_query_count": (
                        assignment["component_query_count"] if assignment else None
                    ),
                    "cross_stratum_contaminated": contaminated,
                    "candidate_recall": candidate_recall,
                    "baseline": {
                        "recall_at_20": _optional_float(
                            baseline.get("recall_at_20")
                        ),
                        "f1_at_20": _optional_float(baseline.get("f1_at_20")),
                        "final_hit": bool(baseline.get("matched_gold_ids")),
                    },
                    "experiment": {
                        "recall_at_20": _optional_float(
                            experiment.get("recall_at_20")
                        ),
                        "f1_at_20": _optional_float(experiment.get("f1_at_20")),
                        "final_hit": bool(experiment.get("matched_gold_ids")),
                    },
                    "candidate_hit": int(item.get("candidate_gold_count") or 0) > 0,
                }
            )
    return rows, {
        scope: {
            view: _aggregate_metric_rows(
                [
                    row
                    for row in rows
                    if row["scope"] == scope and row[f"included_{view}"]
                ]
            )
            for view in ("full", "decontaminated")
        }
        for scope in ("existing65", "record160")
    }


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["candidate_recall"] is not None]
    output: dict[str, Any] = {
        "case_count": len(rows),
        "evaluable_case_count": len(evaluable),
        "dataset_case_counts": dict(
            sorted(Counter(str(row["dataset"]) for row in rows).items())
        ),
        "candidate_recall": _mean(
            [float(row["candidate_recall"]) for row in evaluable]
        ),
    }
    for policy in ("baseline", "experiment"):
        output[policy] = {
            "recall_at_20": _mean(
                [float(row[policy]["recall_at_20"]) for row in evaluable]
            ),
            "f1_at_20": _mean(
                [float(row[policy]["f1_at_20"]) for row in evaluable]
            ),
            "final_hit_query_count": sum(bool(row[policy]["final_hit"]) for row in rows),
        }
    output["candidate_hit_query_count"] = sum(bool(row["candidate_hit"]) for row in rows)
    return output


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _summarize(
    assignments: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    metric_summary: Mapping[str, Any],
    record_cases: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    edge_type_counts = Counter(
        edge_type for item in edges for edge_type in item["edge_types"]
    )
    non_singletons = [item for item in components if int(item["query_count"]) > 1]
    contaminated = [item for item in components if item["cross_stratum_contaminated"]]
    assignments_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in assignments:
        assignments_by_stratum[str(item["exclusive_stratum"])].append(item)
    size_distribution = Counter(int(item["query_count"]) for item in components)
    assignment_by_id = {str(item["query_id"]): item for item in assignments}
    shared_gold_edges = [
        item for item in edges if "shared_gold_identity_cluster" in item["edge_types"]
    ]
    cross_partition_edges = [
        item
        for item in edges
        if assignment_by_id[str(item["left_query_id"])]["exclusive_stratum"]
        != assignment_by_id[str(item["right_query_id"])]["exclusive_stratum"]
    ]
    cross_partition_pairs = {
        (str(item["left_query_id"]), str(item["right_query_id"]))
        for item in cross_partition_edges
    }
    cross_shared_gold_clusters = {
        str(cluster_id)
        for item in shared_gold_edges
        if (str(item["left_query_id"]), str(item["right_query_id"]))
        in cross_partition_pairs
        for cluster_id in item["shared_gold_identity_cluster_ids"]
    }
    membership_overlap_counts = Counter(
        "+".join(sorted(item["frozen_memberships"]))
        for item in assignments
        if len(item["frozen_memberships"]) > 1
    )
    query_duplicate_components = [
        item
        for item in components
        if int(item["edge_type_counts"]["normalized_exact_query"])
        or int(item["edge_type_counts"]["lexical_near_duplicate"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "query_count": len(assignments),
        "component_count": len(components),
        "non_singleton_component_count": len(non_singletons),
        "largest_component_query_count": max(
            (int(item["query_count"]) for item in components), default=0
        ),
        "component_size_distribution": {
            str(size): count for size, count in sorted(size_distribution.items())
        },
        "edge_count": len(edges),
        "edge_type_counts": {name: edge_type_counts[name] for name in EDGE_TYPES},
        "query_duplicates": {
            "normalized_exact_query_pair_count": edge_type_counts[
                "normalized_exact_query"
            ],
            "lexical_near_duplicate_pair_count": edge_type_counts[
                "lexical_near_duplicate"
            ],
            "component_count": len(query_duplicate_components),
            "query_count": len(
                {
                    query_id
                    for item in query_duplicate_components
                    for query_id in item["query_ids"]
                }
            ),
        },
        "shared_gold": {
            "query_pair_edge_count": len(shared_gold_edges),
            "cross_exclusive_stratum_edge_count": sum(
                (str(item["left_query_id"]), str(item["right_query_id"]))
                in cross_partition_pairs
                for item in shared_gold_edges
            ),
            "identity_cluster_count": len(
                {
                    cluster_id
                    for item in shared_gold_edges
                    for cluster_id in item["shared_gold_identity_cluster_ids"]
                }
            ),
            "cross_exclusive_stratum_identity_cluster_count": len(
                cross_shared_gold_clusters
            ),
        },
        "strata": {
            name: {
                "query_count": len(rows),
                "independent_query_count": sum(
                    not row["cross_stratum_contaminated"] for row in rows
                ),
                "contaminated_query_count": sum(
                    row["cross_stratum_contaminated"] for row in rows
                ),
                "non_singleton_component_query_count": sum(
                    int(row["component_query_count"]) > 1 for row in rows
                ),
            }
            for name, rows in sorted(assignments_by_stratum.items())
        },
        "cross_stratum": {
            "contaminated_component_count": len(contaminated),
            "contaminated_query_count": len(
                {
                    query_id
                    for item in contaminated
                    for query_id in item["query_ids"]
                }
            ),
            "cross_exclusive_stratum_edge_count": len(cross_partition_edges),
            "same_query_membership_overlap_count": sum(
                len(set(item["frozen_memberships"]) & set(FROZEN_MEMBERSHIPS)) > 1
                for item in assignments
            ),
            "same_query_membership_overlap_counts": dict(
                sorted(membership_overlap_counts.items())
            ),
        },
        "frozen_metric_diagnostics": dict(metric_summary),
        "frozen_hit_concentration": _hit_concentration(metric_rows),
        "record160_closure": {
            "artifact_case_count": len(record_cases),
            "included_main_analysis_count": sum(
                bool(item.get("included_main_analysis")) for item in record_cases
            ),
            "excluded_no_success_count": sum(
                item.get("analysis_status") == "excluded_no_successful_source"
                for item in record_cases
            ),
        },
        "execution": dict(execution),
        "interpretation": {
            "data_removed": False,
            "formal_split_redrawn": False,
            "diagnostic_decontaminated_view_replaces_full": False,
            "official_score": False,
        },
    }


def _hit_concentration(
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in ("existing65", "record160"):
        rows = [
            row
            for row in metric_rows
            if row["scope"] == scope
            and row["included_full"]
            and row["component_id"] is not None
        ]
        output[scope] = {
            "by_component_size": {
                "singleton": _hit_group(
                    [row for row in rows if int(row["component_query_count"]) == 1]
                ),
                "non_singleton": _hit_group(
                    [row for row in rows if int(row["component_query_count"]) > 1]
                ),
            },
            "by_independence": {
                "independent": _hit_group(
                    [row for row in rows if not row["cross_stratum_contaminated"]]
                ),
                "cross_stratum_contaminated": _hit_group(
                    [row for row in rows if row["cross_stratum_contaminated"]]
                ),
            },
        }
    return output


def _hit_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    candidate = sum(bool(row["candidate_hit"]) for row in rows)
    baseline = sum(bool(row["baseline"]["final_hit"]) for row in rows)
    experiment = sum(bool(row["experiment"]["final_hit"]) for row in rows)
    return {
        "case_count": count,
        "candidate_hit_query_count": candidate,
        "candidate_hit_rate": candidate / count if count else None,
        "baseline_final_hit_query_count": baseline,
        "baseline_final_hit_rate": baseline / count if count else None,
        "experiment_final_hit_query_count": experiment,
        "experiment_final_hit_rate": experiment / count if count else None,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("audit") != AUDIT_NAME or protocol.get("version") != PROTOCOL_VERSION:
        raise QueryIndependenceAuditError("unexpected independence protocol")
    near = protocol.get("near_duplicate") or {}
    if (
        int(near.get("minimum_informative_token_count_per_query", 0)) != 6
        or float(near.get("threshold_inclusive", 0.0)) != 0.8
        or near.get("similarity") != "set_jaccard"
    ):
        raise QueryIndependenceAuditError("near-duplicate threshold drift")
    if tuple(protocol.get("component_definition", {}).get("edges") or []) != EDGE_TYPES:
        raise QueryIndependenceAuditError("component edge definition drift")
    execution = protocol.get("execution") or {}
    if any(
        int(execution.get(field, -1)) != 0
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise QueryIndependenceAuditError("protocol must remain zero-I/O")


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("audit") != AUDIT_NAME or manifest.get("gate") != GATE_NAME:
        raise QueryIndependenceAuditError("unexpected independence manifest")
    if int(manifest.get("dataset", {}).get("case_count", 0)) != 1000:
        raise QueryIndependenceAuditError("query count contract drift")
    for item in (
        manifest["dataset"],
        manifest["gold_identity_baseline"],
        manifest["protocol"],
        manifest["implementation"],
        manifest["frozen_replays"]["existing65"],
        manifest["frozen_replays"]["record160"],
    ):
        _validate_hash(_repo_path(item["path"]), item["sha256"])
    if require_baseline and "baseline" not in manifest:
        raise QueryIndependenceAuditError("independence baseline missing")


def _validate_closure(
    assignments: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected = int(manifest["dataset"]["case_count"])
    if len(assignments) != expected or len({row["query_id"] for row in assignments}) != expected:
        raise QueryIndependenceAuditError("query assignment closure failure")
    assigned_components = {str(row["component_id"]) for row in assignments}
    if assigned_components != {str(row["component_id"]) for row in components}:
        raise QueryIndependenceAuditError("component assignment closure failure")
    if sum(int(row["query_count"]) for row in components) != expected:
        raise QueryIndependenceAuditError("component query count closure failure")
    if len({(row["left_query_id"], row["right_query_id"]) for row in edges}) != len(edges):
        raise QueryIndependenceAuditError("duplicate graph edge")
    if len(metric_rows) != 227:
        raise QueryIndependenceAuditError("frozen metric row closure failure")
    if int(summary["query_count"]) != expected:
        raise QueryIndependenceAuditError("summary query count closure failure")


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
    if sha256_file(path) != str(expected):
        raise QueryIndependenceAuditError(f"frozen input hash drift:{path.name}")


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
        raise QueryIndependenceAuditError("network access forbidden")

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
