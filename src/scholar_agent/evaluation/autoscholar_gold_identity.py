"""Offline identity-quality audit for the frozen AutoScholarQuery gold set.

Gold is loaded only inside this evaluator-side module. The audit never imports
SearchService, a connector, ranking code, Prompt code, or Snapshot runtimes.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_arxiv_id,
    normalize_doi,
    normalize_s2orc_corpus_id,
    normalize_simple_id,
)
from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.datasets.auto_scholar_query import (
    load_auto_scholar_query,
)
from scholar_agent.evaluation.metrics import canonical_paper_id, evaluable_gold_count


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
AUDIT_IMPLEMENTATION_PATH = "src/scholar_agent/evaluation/autoscholar_gold_identity.py"
EVALUATOR_IMPLEMENTATION_PATH = "src/scholar_agent/evaluation/metrics.py"
SCHEMA_VERSION = "1"
GATE_NAME = "autoscholar_gold_identity_regression"
BASELINE_APPROVAL_TOKEN = "PROPOSE_AUTOSCHOLAR_GOLD_IDENTITY_BASELINE"
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
TERMINALS = (
    "stable_identifier_evaluable",
    "conservative_title_evidence_evaluable",
    "identifier_conflict",
    "identity_ambiguous",
    "insufficient_information",
)
_IDENTIFIER_SPECS = (
    ("doi", "doi", normalize_doi, ("doi",)),
    ("arxiv_id", "arxiv", normalize_arxiv_id, ("arxiv_id", "arxiv")),
    (
        "openalex_id",
        "openalex",
        normalize_simple_id,
        ("openalex_id", "openalex"),
    ),
    (
        "semantic_scholar_id",
        "s2",
        normalize_simple_id,
        ("semantic_scholar_id", "semanticScholarId", "s2_id"),
    ),
    (
        "s2orc_corpus_id",
        "s2orc",
        normalize_s2orc_corpus_id,
        ("s2orc_corpus_id", "s2orc_id", "corpus_id", "corpusId", "CorpusId"),
    ),
    (
        "pubmed_id",
        "pubmed",
        normalize_simple_id,
        ("pubmed_id", "pmid", "pubmed"),
    ),
)


class GoldIdentityAuditError(RuntimeError):
    """Raised when the frozen audit contract or baseline is malformed."""


@dataclass(frozen=True)
class _Relation:
    query_order: int
    query_id: str
    gold_index: int
    gold: Any
    profile: IdentityProfile
    field_values: dict[str, tuple[str, ...]]
    local_conflict_fields: tuple[str, ...]

    @property
    def relation_id(self) -> str:
        return f"{self.query_id}::gold[{self.gold_index}]"


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def classify_gold_identity(gold: Any) -> dict[str, Any]:
    """Classify one gold using only the shared conservative identity contract."""

    profile = build_identity_profile(gold)
    field_values = collect_identifier_field_values(gold)
    conflicts = sorted(field for field, values in field_values.items() if len(values) > 1)
    title_complete = bool(profile.title and profile.authors and profile.year is not None)
    if conflicts:
        terminal = "identifier_conflict"
    elif profile.identifiers:
        terminal = "stable_identifier_evaluable"
    elif title_complete:
        terminal = "conservative_title_evidence_evaluable"
    elif profile.title or profile.authors or profile.year is not None:
        terminal = "identity_ambiguous"
    else:
        terminal = "insufficient_information"
    return {
        "terminal_status": terminal,
        "stable_identifiers": sorted(profile.identifiers),
        "stable_identifier_types": sorted({
            value.split(":", 1)[0] for value in profile.identifiers
        }),
        "identifier_field_values": {
            field: list(values) for field, values in sorted(field_values.items())
        },
        "identifier_conflict_fields": conflicts,
        "title_evidence": {
            "normalized_title": profile.title,
            "normalized_authors": sorted(profile.authors),
            "year": profile.year,
            "complete": title_complete,
        },
        "missing_fields": _missing_identity_fields(gold, profile, field_values),
        "current_evaluator_evaluable": evaluable_gold_count([gold]) == 1,
    }


def collect_identifier_field_values(gold: Any) -> dict[str, tuple[str, ...]]:
    """Collect every supported representation so same-field conflicts are visible."""

    containers = _identity_containers(gold)
    output: dict[str, tuple[str, ...]] = {}
    for field, _prefix, normalizer, aliases in _IDENTIFIER_SPECS:
        values: set[str] = set()
        for container in containers:
            for alias in aliases:
                raw = _value(container, alias)
                if raw is None:
                    continue
                normalized = normalizer(raw)
                if normalized:
                    values.add(normalized)
        output[field] = tuple(sorted(values))
    return output


def build_gold_identity_audit(
    manifest: Mapping[str, Any],
    *,
    queries: Sequence[EvalQuery] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the 2403-relation audit without retrieval or effectiveness metrics."""

    _validate_manifest(manifest, require_baseline=False)
    dataset_path = _repo_path(manifest["dataset"]["path"])
    query_manifest_path = _repo_path(manifest["query_manifest"]["path"])
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0}
    with _forbid_network(attempts):
        loaded = list(queries) if queries is not None else load_auto_scholar_query(dataset_path)
        query_manifest = _read_jsonl(query_manifest_path)
        _validate_query_manifest(loaded, query_manifest)
        gold_rows, query_rows, identity_summary = analyze_gold_relations(loaded)
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "retrieval_invoked": False,
        "effectiveness_metrics_generated": False,
        "gold_scope": "offline_evaluator_input_only",
    }
    if any(
        int(execution[field])
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise GoldIdentityAuditError(f"offline execution invariant failed:{execution}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit": GATE_NAME,
        "dataset": {
            "name": manifest["dataset"]["name"],
            "split": manifest["dataset"]["split"],
            "path": manifest["dataset"]["path"],
            "sha256": sha256_file(dataset_path),
            "case_count": len(loaded),
            "gold_relation_count": len(gold_rows),
        },
        "query_manifest": {
            "path": manifest["query_manifest"]["path"],
            "sha256": sha256_file(query_manifest_path),
            "case_count": len(query_manifest),
            "order_matches_dataset": True,
        },
        **identity_summary,
        "identity_implementation": {
            "path": manifest["identity_implementation"]["path"],
            "sha256": sha256_file(
                _repo_path(manifest["identity_implementation"]["path"])
            ),
            "fuzzy_title_matching": False,
            "external_crosswalk": False,
        },
        "evaluator_implementation": {
            "path": manifest["evaluator_implementation"]["path"],
            "sha256": sha256_file(
                _repo_path(manifest["evaluator_implementation"]["path"])
            ),
        },
        "audit_implementation": {
            "path": manifest["audit_implementation"]["path"],
            "sha256": sha256_file(
                _repo_path(manifest["audit_implementation"]["path"])
            ),
        },
        "execution": execution,
        "official_effectiveness_score": False,
    }
    _validate_summary_closure(gold_rows, query_rows, summary)
    return gold_rows, query_rows, summary


def analyze_gold_relations(
    queries: Sequence[EvalQuery],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Analyze identity, duplicates, and evaluator denominator semantics."""

    relations = _relations(queries)
    conflicts = _cross_record_conflicts(relations)
    clusters = _identity_clusters(relations, conflicts)
    cluster_members: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in clusters.items():
        cluster_members[cluster_id].append(index)
    query_ids_by_cluster = {
        cluster_id: sorted({relations[index].query_id for index in indices})
        for cluster_id, indices in cluster_members.items()
    }
    gold_rows: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        local = classify_gold_identity(relation.gold)
        conflict_ids = sorted(
            relations[value].relation_id for value in conflicts.get(index, set())
        )
        if conflict_ids:
            local["terminal_status"] = "identifier_conflict"
        cluster_id = clusters[index]
        same_query_members = [
            value
            for value in cluster_members[cluster_id]
            if relations[value].query_id == relation.query_id
        ]
        safe_evaluable = local["terminal_status"] in {
            "stable_identifier_evaluable",
            "conservative_title_evidence_evaluable",
        }
        gold_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "relation_id": relation.relation_id,
                "query_order": relation.query_order,
                "query_id": relation.query_id,
                "gold_index": relation.gold_index,
                "gold_input_sha256": sha256_json(_json_value(relation.gold)),
                **local,
                "cross_record_conflict_relation_ids": conflict_ids,
                "safe_evaluator_evaluable": safe_evaluable,
                "canonical_identity": canonical_paper_id(relation.gold),
                "identity_cluster_id": cluster_id,
                "identity_cluster_relation_count": len(cluster_members[cluster_id]),
                "identity_cluster_query_count": len(query_ids_by_cluster[cluster_id]),
                "duplicate_within_query": len(same_query_members) > 1,
                "duplicate_across_queries": len(query_ids_by_cluster[cluster_id]) > 1,
                "relevance_grade": float(_value(relation.gold, "relevance_grade") or 0.0),
            }
        )

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        by_query[str(row["query_id"])].append(row)
    query_rows: list[dict[str, Any]] = []
    query_by_id = {item.query_id: item for item in queries}
    for query_order, query in enumerate(queries):
        rows = by_query[query.query_id]
        evaluator_count = evaluable_gold_count(query.gold_papers)
        evaluator_clusters = {
            str(row["identity_cluster_id"])
            for row in rows
            if row["current_evaluator_evaluable"]
        }
        safe_clusters = {
            str(row["identity_cluster_id"])
            for row in rows
            if row["safe_evaluator_evaluable"]
        }
        terminals = Counter(str(row["terminal_status"]) for row in rows)
        query_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_order": query_order,
                "query_id": query.query_id,
                "query_input_sha256": sha256_text(query_by_id[query.query_id].query),
                "raw_gold_relation_count": len(rows),
                "current_evaluator_gold_count": evaluator_count,
                "current_evaluator_unique_identity_count": len(evaluator_clusters),
                "current_evaluator_duplicate_denominator_count": max(
                    0, evaluator_count - len(evaluator_clusters)
                ),
                "safe_evaluable_relation_count": sum(
                    bool(row["safe_evaluator_evaluable"]) for row in rows
                ),
                "safe_unique_identity_count": len(safe_clusters),
                "terminal_counts": {
                    name: terminals[name] for name in TERMINALS
                },
                "zero_raw_gold": not rows,
                "zero_current_evaluable_gold": evaluator_count == 0,
                "zero_safe_evaluable_gold": not safe_clusters,
            }
        )

    terminal_counts = Counter(str(row["terminal_status"]) for row in gold_rows)
    identifier_types = Counter(
        value for row in gold_rows for value in row["stable_identifier_types"]
    )
    missing_fields = Counter(
        value for row in gold_rows for value in row["missing_fields"]
    )
    unique_clusters = set(clusters.values())
    safe_clusters = {
        str(row["identity_cluster_id"])
        for row in gold_rows
        if row["safe_evaluator_evaluable"]
    }
    current_evaluator_clusters = {
        str(row["identity_cluster_id"])
        for row in gold_rows
        if row["current_evaluator_evaluable"]
    }
    repeated_clusters = {
        cluster_id
        for cluster_id, query_ids in query_ids_by_cluster.items()
        if len(query_ids) > 1
    }
    title_to_identifiers: dict[str, set[str]] = defaultdict(set)
    identifier_to_titles: dict[str, set[str]] = defaultdict(set)
    for row in gold_rows:
        title = str(row["title_evidence"]["normalized_title"])
        for identifier in row["stable_identifiers"]:
            if title:
                title_to_identifiers[title].add(str(identifier))
                identifier_to_titles[str(identifier)].add(title)
    summary = {
        "gold_relation_count": len(gold_rows),
        "terminal_counts": {name: terminal_counts[name] for name in TERMINALS},
        "terminal_count_closed": sum(terminal_counts.values()) == len(gold_rows),
        "stable_identifier_type_relation_counts": dict(sorted(identifier_types.items())),
        "missing_field_relation_counts": dict(sorted(missing_fields.items())),
        "safe_evaluable_relation_count": sum(
            bool(row["safe_evaluator_evaluable"]) for row in gold_rows
        ),
        "safe_evaluable_relation_rate": (
            sum(bool(row["safe_evaluator_evaluable"]) for row in gold_rows)
            / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "current_evaluator_evaluable_relation_count": sum(
            bool(row["current_evaluator_evaluable"]) for row in gold_rows
        ),
        "global_unique_identity_count": len(unique_clusters),
        "safe_global_unique_identity_count": len(safe_clusters),
        "current_evaluator_global_unique_identity_count": len(
            current_evaluator_clusters
        ),
        "current_evaluator_global_repeated_relation_count": sum(
            bool(row["current_evaluator_evaluable"]) for row in gold_rows
        )
        - len(current_evaluator_clusters),
        "safe_evaluator_deduplicated_query_denominator_count": sum(
            int(row["safe_unique_identity_count"]) for row in query_rows
        ),
        "global_duplicate_relation_count": len(gold_rows) - len(unique_clusters),
        "within_query_duplicate_relation_count": sum(
            int(row["current_evaluator_duplicate_denominator_count"])
            for row in query_rows
        ),
        "queries_with_duplicate_evaluator_relations": sum(
            int(row["current_evaluator_duplicate_denominator_count"]) > 0
            for row in query_rows
        ),
        "cross_query_repeated_identity_count": len(repeated_clusters),
        "cross_query_repeated_relation_count": sum(
            len(cluster_members[cluster_id]) for cluster_id in repeated_clusters
        ),
        "normalized_title_multiple_identifier_count": sum(
            len(values) > 1 for values in title_to_identifiers.values()
        ),
        "stable_identifier_multiple_title_variant_count": sum(
            len(values) > 1 for values in identifier_to_titles.values()
        ),
        "query_count": len(query_rows),
        "raw_gold_per_query_distribution": _count_distribution(
            row["raw_gold_relation_count"] for row in query_rows
        ),
        "current_evaluator_gold_per_query_distribution": _count_distribution(
            row["current_evaluator_gold_count"] for row in query_rows
        ),
        "safe_unique_gold_per_query_distribution": _count_distribution(
            row["safe_unique_identity_count"] for row in query_rows
        ),
        "zero_raw_gold_query_count": sum(row["zero_raw_gold"] for row in query_rows),
        "zero_current_evaluable_gold_query_count": sum(
            row["zero_current_evaluable_gold"] for row in query_rows
        ),
        "zero_safe_evaluable_gold_query_count": sum(
            row["zero_safe_evaluable_gold"] for row in query_rows
        ),
        "per_gold_identity_hashes_sha256": sha256_json(
            [
                {
                    "relation_id": row["relation_id"],
                    "identity_sha256": sha256_json(_identity_projection(row)),
                }
                for row in gold_rows
            ]
        ),
    }
    return gold_rows, query_rows, summary


def _count_distribution(values: Iterator[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(value): counts[value] for value in sorted(counts)}


def check_gold_identity_regression(
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, require_baseline=True)
    gold_rows, query_rows, summary = build_gold_identity_audit(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_audit(output_dir, gold_rows, query_rows, summary)
    expected_gold = _read_jsonl(_repo_path(manifest["baseline"]["gold_rows_path"]))
    expected_queries = _read_jsonl(
        _repo_path(manifest["baseline"]["query_rows_path"])
    )
    expected_summary = _read_json(_repo_path(manifest["baseline"]["summary_path"]))
    drifts: list[dict[str, Any]] = []
    drifts.extend(_fingerprint_drifts(manifest))
    drifts.extend(_keyed_diffs(expected_gold, gold_rows, "relation_id", "gold"))
    drifts.extend(_keyed_diffs(expected_queries, query_rows, "query_id", "query"))
    for drift in compare_profiles(expected_summary, summary, max_diffs=200):
        drift["path"] = drift["path"].replace("$", "$.summary", 1)
        drifts.append(drift)
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not drifts,
        "case_count": len(query_rows),
        "gold_relation_count": len(gold_rows),
        "drift_count": len(drifts),
        "drifts": drifts[:200],
        "artifacts": {
            "gold_rows_sha256": sha256_file(output_dir / "gold_identity.jsonl"),
            "query_rows_sha256": sha256_file(output_dir / "query_identity.jsonl"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "gold_hashes_sha256": sha256_file(output_dir / "gold_hashes.jsonl"),
        },
        "execution": summary["execution"],
    }
    _write_json(output_dir / "regression_report.json", report)
    return report


def write_gold_identity_audit(
    output_dir: Path,
    gold_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_audit(output_dir, gold_rows, query_rows, summary)


def propose_gold_identity_baseline(
    manifest_path: Path,
    output_dir: Path,
    *,
    approval_token: str,
    reason: str,
) -> dict[str, Any]:
    if approval_token != BASELINE_APPROVAL_TOKEN:
        raise GoldIdentityAuditError("baseline proposal approval token rejected")
    if len(reason.strip()) < 12:
        raise GoldIdentityAuditError("baseline proposal reason is too short")
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, require_baseline=False)
    if output_dir.exists():
        raise GoldIdentityAuditError("baseline proposal output already exists")
    gold_rows, query_rows, summary = build_gold_identity_audit(manifest)
    write_gold_identity_audit(output_dir, gold_rows, query_rows, summary)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "reason": reason.strip(),
        "approval_token_verified": True,
        "tracked_files_mutated": False,
        "gold_relation_count": len(gold_rows),
        "case_count": len(query_rows),
        "gold_rows_sha256": sha256_file(output_dir / "gold_identity.jsonl"),
        "query_rows_sha256": sha256_file(output_dir / "query_identity.jsonl"),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
    }
    _write_json(output_dir / "baseline_update_audit.json", audit)
    return audit


def _relations(queries: Sequence[EvalQuery]) -> list[_Relation]:
    relations: list[_Relation] = []
    for query_order, query in enumerate(queries):
        for gold_index, gold in enumerate(query.gold_papers):
            field_values = collect_identifier_field_values(gold)
            relations.append(
                _Relation(
                    query_order=query_order,
                    query_id=query.query_id,
                    gold_index=gold_index,
                    gold=gold,
                    profile=build_identity_profile(gold),
                    field_values=field_values,
                    local_conflict_fields=tuple(
                        sorted(
                            field
                            for field, values in field_values.items()
                            if len(values) > 1
                        )
                    ),
                )
            )
    return relations


def _cross_record_conflicts(
    relations: Sequence[_Relation],
) -> dict[int, set[int]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for index, relation in enumerate(relations):
        for identifier in relation.profile.identifiers:
            groups[f"id:{identifier}"].add(index)
        title_key = _strict_title_key(relation.profile)
        if title_key:
            groups[f"title:{title_key}"].add(index)
    conflicts: dict[int, set[int]] = defaultdict(set)
    for indices in groups.values():
        ordered = sorted(indices)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                evidence = identity_evidence_from_profiles(
                    relations[left].profile, relations[right].profile
                )
                if evidence.conflicting_identifiers:
                    conflicts[left].add(right)
                    conflicts[right].add(left)
    return conflicts


def _identity_clusters(
    relations: Sequence[_Relation],
    conflicts: Mapping[int, set[int]],
) -> dict[int, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, relation in enumerate(relations):
        for identifier in relation.profile.identifiers:
            groups[f"id:{identifier}"].append(index)
        title_key = _strict_title_key(relation.profile)
        if title_key:
            groups[f"title:{title_key}"].append(index)
    dsu = _DisjointSet(len(relations))
    for indices in groups.values():
        ordered = sorted(set(indices))
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                if right in conflicts.get(left, set()):
                    continue
                evidence = identity_evidence_from_profiles(
                    relations[left].profile, relations[right].profile
                )
                if evidence.equivalent:
                    dsu.union(left, right)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(relations)):
        members[dsu.find(index)].append(index)
    cluster_ids: dict[int, str] = {}
    for indices in members.values():
        identifiers = sorted(
            {
                identifier
                for index in indices
                for identifier in relations[index].profile.identifiers
            }
        )
        title_keys = sorted(
            {
                value
                for index in indices
                if (value := _strict_title_key(relations[index].profile))
            }
        )
        evidence = identifiers or title_keys or sorted(
            relations[index].relation_id for index in indices
        )
        cluster_id = "identity:" + sha256_text("\n".join(evidence))[:20]
        for index in indices:
            cluster_ids[index] = cluster_id
    return cluster_ids


def _keyed_diffs(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    key: str,
    namespace: str,
) -> list[dict[str, Any]]:
    left = {str(item[key]): item for item in expected}
    right = {str(item[key]): item for item in actual}
    if len(left) != len(expected) or len(right) != len(actual):
        raise GoldIdentityAuditError(f"duplicate regression key:{namespace}:{key}")
    diffs: list[dict[str, Any]] = []
    for value in sorted(set(left) - set(right)):
        diffs.append(
            {
                "path": f"$.{namespace}[{value}]",
                "kind": f"{namespace}_removed",
                "expected": value,
                "actual": None,
            }
        )
    for value in sorted(set(right) - set(left)):
        diffs.append(
            {
                "path": f"$.{namespace}[{value}]",
                "kind": f"{namespace}_added",
                "expected": None,
                "actual": value,
            }
        )
    for value in sorted(set(left) & set(right)):
        for drift in compare_profiles(left[value], right[value], max_diffs=50):
            drift["path"] = drift["path"].replace(
                "$", f"$.{namespace}[{value}]", 1
            )
            diffs.append(drift)
    return diffs[:200]


def _fingerprint_drifts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "dataset_sha256": manifest["dataset"]["sha256"],
        "query_manifest_sha256": manifest["query_manifest"]["sha256"],
        "identity_implementation_sha256": manifest["identity_implementation"][
            "sha256"
        ],
        "evaluator_implementation_sha256": manifest["evaluator_implementation"][
            "sha256"
        ],
        "audit_implementation_sha256": manifest["audit_implementation"]["sha256"],
        "baseline_gold_rows_sha256": manifest["baseline"]["gold_rows_sha256"],
        "baseline_query_rows_sha256": manifest["baseline"]["query_rows_sha256"],
        "baseline_summary_sha256": manifest["baseline"]["summary_sha256"],
    }
    actual = {
        "dataset_sha256": sha256_file(_repo_path(manifest["dataset"]["path"])),
        "query_manifest_sha256": sha256_file(
            _repo_path(manifest["query_manifest"]["path"])
        ),
        "identity_implementation_sha256": sha256_file(
            _repo_path(manifest["identity_implementation"]["path"])
        ),
        "evaluator_implementation_sha256": sha256_file(
            _repo_path(manifest["evaluator_implementation"]["path"])
        ),
        "audit_implementation_sha256": sha256_file(
            _repo_path(manifest["audit_implementation"]["path"])
        ),
        "baseline_gold_rows_sha256": sha256_file(
            _repo_path(manifest["baseline"]["gold_rows_path"])
        ),
        "baseline_query_rows_sha256": sha256_file(
            _repo_path(manifest["baseline"]["query_rows_path"])
        ),
        "baseline_summary_sha256": sha256_file(
            _repo_path(manifest["baseline"]["summary_path"])
        ),
    }
    drifts = compare_profiles(expected, actual)
    for drift in drifts:
        drift["path"] = drift["path"].replace("$", "$.fingerprints", 1)
    return drifts


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GoldIdentityAuditError("unsupported gold identity manifest")
    if manifest.get("gate") != GATE_NAME:
        raise GoldIdentityAuditError("unexpected gold identity gate")
    if int(manifest["dataset"]["case_count"]) <= 0:
        raise GoldIdentityAuditError("gold identity audit case count must be positive")
    if int(manifest["dataset"]["gold_relation_count"]) <= 0:
        raise GoldIdentityAuditError("gold relation count must be positive")
    for field, expected_path in (
        ("audit_implementation", AUDIT_IMPLEMENTATION_PATH),
        ("evaluator_implementation", EVALUATOR_IMPLEMENTATION_PATH),
    ):
        if manifest.get(field, {}).get("path") != expected_path:
            raise GoldIdentityAuditError(f"unexpected {field} path")
    execution = manifest.get("execution") or {}
    for field in ("network_request_count", "llm_request_count", "snapshot_write_count"):
        if int(execution.get(field) or 0):
            raise GoldIdentityAuditError(f"offline execution drift:{field}")
    if require_baseline:
        required = {
            "gold_rows_path",
            "gold_rows_sha256",
            "query_rows_path",
            "query_rows_sha256",
            "summary_path",
            "summary_sha256",
        }
        if not required.issubset(manifest.get("baseline") or {}):
            raise GoldIdentityAuditError("gold identity baseline is incomplete")


def _validate_query_manifest(
    queries: Sequence[EvalQuery], rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(queries) != len(rows):
        raise GoldIdentityAuditError("query manifest case count drifted")
    for index, (query, row) in enumerate(zip(queries, rows, strict=True)):
        if row.get("query_id") != query.query_id or row.get("query") != query.query:
            raise GoldIdentityAuditError(f"query manifest order drifted at index {index}")


def _validate_summary_closure(
    gold_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    if len(gold_rows) != int(summary["gold_relation_count"]):
        raise GoldIdentityAuditError("gold relation count is not closed")
    if len(query_rows) != int(summary["query_count"]):
        raise GoldIdentityAuditError("query count is not closed")
    if sum(int(value) for value in summary["terminal_counts"].values()) != len(
        gold_rows
    ):
        raise GoldIdentityAuditError("gold terminal classification is not closed")
    if any(
        sum(int(value) for value in row["terminal_counts"].values())
        != int(row["raw_gold_relation_count"])
        for row in query_rows
    ):
        raise GoldIdentityAuditError("query terminal classification is not closed")


def _write_audit(
    output_dir: Path,
    gold_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    _write_jsonl(output_dir / "gold_identity.jsonl", gold_rows)
    _write_jsonl(output_dir / "query_identity.jsonl", query_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(
        output_dir / "gold_hashes.jsonl",
        [
            {
                "relation_id": row["relation_id"],
                "identity_sha256": sha256_json(_identity_projection(row)),
            }
            for row in gold_rows
        ],
    )


def _identity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "terminal_status": row["terminal_status"],
        "stable_identifiers": row["stable_identifiers"],
        "identifier_conflict_fields": row["identifier_conflict_fields"],
        "title_evidence": row["title_evidence"],
        "identity_cluster_id": row["identity_cluster_id"],
        "current_evaluator_evaluable": row["current_evaluator_evaluable"],
        "safe_evaluator_evaluable": row["safe_evaluator_evaluable"],
    }


def _identity_containers(gold: Any) -> list[Any]:
    metadata = _value(gold, "metadata") or {}
    containers = [gold, _value(gold, "identifiers"), metadata]
    containers.append(_value(metadata, "identifiers"))
    return [item for item in containers if item is not None]


def _missing_identity_fields(
    gold: Any,
    profile: IdentityProfile,
    field_values: Mapping[str, Sequence[str]],
) -> list[str]:
    missing: list[str] = []
    if not profile.title:
        missing.append("title")
    if not profile.authors:
        missing.append("authors")
    if profile.year is None:
        missing.append("year")
    missing.extend(field for field, values in field_values.items() if not values)
    return sorted(missing)


def _strict_title_key(profile: IdentityProfile) -> str | None:
    if not profile.title or not profile.authors or profile.year is None:
        return None
    return f"{profile.title}|{profile.year}|{'|'.join(sorted(profile.authors))}"


def _value(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _json_value(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_signature(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts["network"] += 1
        raise GoldIdentityAuditError("network access forbidden in gold identity audit")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
