"""Deterministic field-level lineage for production paper deduplication.

The source records in this contract are connector-mapped :class:`Paper`
objects, not raw HTTP payloads.  The module deliberately records only the
fields needed to reconstruct the production merge and never carries request
headers, credentials, absolute paths, or unrelated response bodies.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.core.identity import (
    IdentityEvidence,
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_arxiv_id,
    normalize_doi,
    normalize_s2orc_corpus_id,
    normalize_simple_id,
    normalize_title,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.untrusted_metadata import UntrustedMetadataIsolationDocument


RESULT_LINEAGE_CONTRACT = "result_lineage_v1"
RESULT_LINEAGE_SCHEMA_VERSION = "1"
IDENTITY_NORMALIZATION_VERSION = "unified_identity_v1"
FIELD_MERGE_VERSION = "paper_field_merge_v1"
RESULT_LINEAGE_ARTIFACT_NAME = "result_lineage.jsonl"

FieldState = Literal["missing", "null", "empty", "value", "invalid"]
TerminalStatus = Literal[
    "success", "success_empty", "partial_completion", "failed", "not_started"
]

_FIELD_PATHS = (
    "title",
    "authors",
    "year",
    "venue",
    "abstract",
    "identifiers.doi",
    "identifiers.arxiv_id",
    "identifiers.semantic_scholar_id",
    "identifiers.s2orc_corpus_id",
    "identifiers.openalex_id",
    "identifiers.pubmed_id",
    "urls.landing_page",
    "urls.pdf",
    "sources",
    "citation_count",
)


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def opaque_query_identity(query: str) -> str:
    """Return a stable identity without retaining or exposing query text."""

    normalized = " ".join(str(query).split())
    return f"query:{stable_sha256({'query': normalized})}"


def run_manifest_output_spec() -> dict[str, str]:
    """Return the mandatory run_manifest_v1 registration for lineage output."""

    return {
        "path": RESULT_LINEAGE_ARTIFACT_NAME,
        "role": RESULT_LINEAGE_CONTRACT,
        "format": "jsonl",
    }


class SourceTerminal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    status: TerminalStatus
    reason: str | None = Field(default=None, pattern=r"^[a-z0-9_.:-]+$")
    contributed_record_count: int = Field(ge=0)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_index: int = Field(ge=0)
    record_ref: str
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: list[str]
    paper: Paper
    field_states: dict[str, FieldState]
    identity_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ClusterMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_ref: str
    action: Literal["cluster_created", "merged"]
    matched_cluster_record_refs: list[str] = Field(default_factory=list)
    rule: str
    shared_identifiers: list[str] = Field(default_factory=list)
    conflicting_identifiers: list[str] = Field(default_factory=list)
    normalized_title: str | None = None
    author_overlap: list[str] = Field(default_factory=list)
    year: int | None = None


class RejectedIdentityComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_record_ref: str
    right_record_ref: str
    rule: str
    conflicting_identifiers: list[str] = Field(default_factory=list)


class FieldCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_ref: str
    state: FieldState
    value: Any = None
    normalized_value: Any = None
    normalization_steps: list[str] = Field(default_factory=list)


class RejectedFieldCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_ref: str
    reason: str


class FieldDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    status: Literal["selected", "no_contribution", "conflict_resolved"]
    selected_value: Any = None
    selected_record_refs: list[str] = Field(default_factory=list)
    selection_rule: str
    candidates: list[FieldCandidate]
    rejected: list[RejectedFieldCandidate] = Field(default_factory=list)
    deterministic_steps: list[str] = Field(default_factory=list)


class ResultLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_identity: str
    final_result: Paper
    contributing_sources: list[str]
    contributing_record_refs: list[str]
    cluster_members: list[ClusterMember]
    field_decisions: list[FieldDecision]


class ResultLineageDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["result_lineage_v1"] = RESULT_LINEAGE_CONTRACT
    schema_version: Literal["1"] = RESULT_LINEAGE_SCHEMA_VERSION
    query_identity: str
    identity_normalization_version: Literal[
        "unified_identity_v1"
    ] = IDENTITY_NORMALIZATION_VERSION
    field_merge_version: Literal["paper_field_merge_v1"] = FIELD_MERGE_VERSION
    source_terminals: list[SourceTerminal]
    source_records: list[SourceRecord]
    invalid_records: list[dict[str, Any]] = Field(default_factory=list)
    rejected_identity_comparisons: list[RejectedIdentityComparison]
    results: list[ResultLineage]
    final_result_order: list[str]
    final_results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    untrusted_metadata_isolation: UntrustedMetadataIsolationDocument | None = None


def build_result_lineage_document(
    *,
    query_identity: str,
    input_papers: Sequence[Paper],
    final_papers: Sequence[Paper],
    cluster_indexes: Sequence[Sequence[int]],
    cluster_member_evidence: Sequence[Sequence[IdentityEvidence]],
    source_terminals: Sequence[Mapping[str, Any]] | None = None,
    untrusted_metadata_isolation: Mapping[str, Any] | None = None,
) -> ResultLineageDocument:
    """Build lineage from the exact clusters produced by production dedup."""

    records = _source_records(input_papers)
    terminals = _source_terminals(records, source_terminals)
    result_rows: list[ResultLineage] = []
    for output, indexes, evidences in zip(
        final_papers, cluster_indexes, cluster_member_evidence, strict=True
    ):
        members = [records[index] for index in indexes]
        result_rows.append(
            ResultLineage(
                result_identity=result_identity(output),
                final_result=output.model_copy(deep=True),
                contributing_sources=sorted(
                    {source for item in members for source in item.sources}
                ),
                contributing_record_refs=[item.record_ref for item in members],
                cluster_members=_cluster_members(members, evidences),
                field_decisions=[
                    _field_decision(path, members, output) for path in _FIELD_PATHS
                ],
            )
        )
    final_order = [item.result_identity for item in result_rows]
    return ResultLineageDocument(
        query_identity=query_identity,
        source_terminals=terminals,
        source_records=records,
        rejected_identity_comparisons=_rejected_identity_comparisons(records),
        results=result_rows,
        final_result_order=final_order,
        final_results_sha256=stable_sha256(
            [paper.model_dump(mode="json") for paper in final_papers]
        ),
        untrusted_metadata_isolation=(
            UntrustedMetadataIsolationDocument.model_validate(
                untrusted_metadata_isolation
            )
            if untrusted_metadata_isolation is not None
            else None
        ),
    )


def result_identity(paper: Paper) -> str:
    profile = build_identity_profile(paper)
    payload: dict[str, Any]
    if profile.identifiers:
        payload = {"stable_identifiers": sorted(profile.identifiers)}
    else:
        payload = {
            "normalized_title": profile.title,
            "authors": sorted(profile.authors),
            "year": profile.year,
        }
    return f"result:{stable_sha256(payload)}"


def ranked_result_authority_digest(ranked: Any) -> str:
    """Hash one authoritative ranked result before public display transforms."""

    payload = (
        ranked.model_dump(mode="json")
        if hasattr(ranked, "model_dump")
        else ranked
    )
    return stable_sha256({"ranked_result": payload})


def restrict_result_lineage_document(
    document: Mapping[str, Any], final_papers: Sequence[Paper]
) -> dict[str, Any]:
    """Restrict candidate lineage to the papers returned by the final ranker.

    The operation only selects already reconstructed results; it never merges
    or enriches fields and therefore cannot affect the production output.
    """

    parsed = ResultLineageDocument.model_validate(document)
    by_identity = {item.result_identity: item for item in parsed.results}
    identities = [result_identity(paper) for paper in final_papers]
    if len(identities) != len(set(identities)):
        raise ValueError("final_result_identity_duplicate")
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError("final_result_lineage_missing")
    restricted = parsed.model_copy(
        update={
            "final_result_order": identities,
            "final_results_sha256": stable_sha256(
                [paper.model_dump(mode="json") for paper in final_papers]
            ),
        },
        deep=True,
    )
    return restricted.model_dump(mode="json")


def _source_records(papers: Sequence[Paper]) -> list[SourceRecord]:
    occurrences: Counter[tuple[str, str]] = Counter()
    records: list[SourceRecord] = []
    for index, paper in enumerate(papers):
        payload = paper.model_dump(mode="json")
        digest = stable_sha256(payload)
        sources = [value for value in paper.sources if value.strip()]
        source_key = "+".join(sources) if sources else "unattributed"
        occurrences[(source_key, digest)] += 1
        record_ref = (
            f"record:{source_key}:{digest[:16]}:"
            f"{occurrences[(source_key, digest)]:04d}"
        )
        profile = build_identity_profile(paper)
        records.append(
            SourceRecord(
                input_index=index,
                record_ref=record_ref,
                source_record_sha256=digest,
                sources=sources,
                paper=paper.model_copy(deep=True),
                field_states={
                    path: _field_state(_path_value(payload, path), present=True)
                    for path in _FIELD_PATHS
                },
                identity_summary_sha256=stable_sha256(
                    {
                        "identifiers": sorted(profile.identifiers),
                        "title": profile.title,
                        "authors": sorted(profile.authors),
                        "year": profile.year,
                    }
                ),
            )
        )
    return records


def _source_terminals(
    records: Sequence[SourceRecord],
    values: Sequence[Mapping[str, Any]] | None,
) -> list[SourceTerminal]:
    counts = Counter(source for item in records for source in item.sources)
    if values is None:
        return [
            SourceTerminal(
                source=source,
                status="success",
                contributed_record_count=count,
            )
            for source, count in sorted(counts.items())
        ]
    terminals = [SourceTerminal.model_validate(item) for item in values]
    return sorted(terminals, key=lambda item: item.source)


def _cluster_members(
    records: Sequence[SourceRecord], evidences: Sequence[IdentityEvidence]
) -> list[ClusterMember]:
    rows: list[ClusterMember] = []
    for index, (record, evidence) in enumerate(zip(records, evidences, strict=True)):
        rows.append(
            ClusterMember(
                record_ref=record.record_ref,
                action="cluster_created" if index == 0 else "merged",
                matched_cluster_record_refs=[
                    item.record_ref for item in records[:index]
                ],
                rule=evidence.rule,
                shared_identifiers=list(evidence.shared_identifiers),
                conflicting_identifiers=list(evidence.conflicting_identifiers),
                normalized_title=evidence.title,
                author_overlap=list(evidence.author_overlap),
                year=evidence.year,
            )
        )
    return rows


def _field_decision(
    path: str, records: Sequence[SourceRecord], output: Paper
) -> FieldDecision:
    candidates = [
        _field_candidate(path, item) for item in records
    ]
    output_value = _path_value(output.model_dump(mode="json"), path)
    selected_indexes, rule = _selected_indexes(path, candidates, output_value)
    selected_refs = [records[index].record_ref for index in selected_indexes]
    nonempty_values = {
        json.dumps(item.normalized_value, ensure_ascii=False, sort_keys=True)
        for item in candidates
        if item.state == "value"
    }
    if not selected_refs and all(item.state != "value" for item in candidates):
        status = "no_contribution"
    elif len(nonempty_values) > 1:
        status = "conflict_resolved"
    else:
        status = "selected"
    rejected = [
        RejectedFieldCandidate(
            record_ref=record.record_ref,
            reason=_rejection_reason(path, candidate, rule),
        )
        for index, (record, candidate) in enumerate(zip(records, candidates, strict=True))
        if index not in selected_indexes
    ]
    return FieldDecision(
        field=path,
        status=status,
        selected_value=output_value,
        selected_record_refs=selected_refs,
        selection_rule=rule,
        candidates=candidates,
        rejected=rejected,
        deterministic_steps=[FIELD_MERGE_VERSION],
    )


def _field_candidate(path: str, record: SourceRecord) -> FieldCandidate:
    payload = record.paper.model_dump(mode="json")
    value = _path_value(payload, path)
    normalized, steps = _normalized_field_value(path, value)
    return FieldCandidate(
        record_ref=record.record_ref,
        state=_field_state(value, present=True),
        value=value,
        normalized_value=normalized,
        normalization_steps=steps,
    )


def _selected_indexes(
    path: str, candidates: Sequence[FieldCandidate], output_value: Any
) -> tuple[list[int], str]:
    values = [item.value for item in candidates]
    if path == "sources":
        selected: list[int] = []
        seen: set[str] = set()
        for index, value in enumerate(values):
            added = False
            for source in value or []:
                key = str(source).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    added = True
            if added:
                selected.append(index)
        return selected, "stable_unique_first_seen"
    if not values:
        return [], "no_source_candidate"
    winner = 0
    rule = "first_seen"
    for index in range(1, len(values)):
        left = values[winner]
        right = values[index]
        choose_right, current_rule = _choose_right(path, left, right)
        rule = current_rule
        if choose_right:
            winner = index
    if _field_state(output_value, present=True) != "value":
        return [], rule
    return [winner], rule


def _choose_right(path: str, left: Any, right: Any) -> tuple[bool, str]:
    if path == "title":
        left_placeholder = normalize_title(left).startswith("untitled")
        right_placeholder = normalize_title(right).startswith("untitled")
        if left_placeholder != right_placeholder:
            return left_placeholder, "prefer_non_placeholder_title"
        return (
            len(str(right).strip()) > len(str(left).strip()),
            "longer_trimmed_title_then_first_seen",
        )
    if path == "authors":
        return len(right or []) > len(left or []), "more_authors_then_first_seen"
    if path == "abstract":
        return (
            len(str(right or "").strip()) > len(str(left or "").strip()),
            "longer_trimmed_abstract_then_first_seen",
        )
    if path == "citation_count":
        return (right or 0) > (left or 0), "maximum_citation_count_then_first_seen"
    if path == "year":
        return left is None and right is not None, "first_non_null_then_first_seen"
    return (not bool(left) and bool(right)), "first_nonempty_then_first_seen"


def _rejection_reason(path: str, candidate: FieldCandidate, rule: str) -> str:
    if candidate.state == "null":
        return "null_not_selected"
    if candidate.state == "empty":
        return "empty_not_selected"
    if path.startswith("identifiers."):
        return "conflicting_or_later_identifier"
    return f"not_selected_by:{rule}"


def _normalized_field_value(path: str, value: Any) -> tuple[Any, list[str]]:
    if path == "title":
        return normalize_title(value), ["normalize_title_v1"]
    if path == "authors":
        return [normalize_title(item) for item in value or []], [
            "normalize_author_v1"
        ]
    if path == "identifiers.doi":
        return normalize_doi(value), ["normalize_doi_v1"]
    if path == "identifiers.arxiv_id":
        return normalize_arxiv_id(value), ["normalize_arxiv_id_v1"]
    if path == "identifiers.s2orc_corpus_id":
        return normalize_s2orc_corpus_id(value), ["normalize_s2orc_corpus_id_v1"]
    if path.startswith("identifiers."):
        return normalize_simple_id(value), ["normalize_simple_id_v1"]
    if path == "sources":
        return [str(item).strip().lower() for item in value or []], [
            "source_key_strip_casefold_v1"
        ]
    return value, []


def _rejected_identity_comparisons(
    records: Sequence[SourceRecord],
) -> list[RejectedIdentityComparison]:
    profiles = [build_identity_profile(item.paper) for item in records]
    rows: list[RejectedIdentityComparison] = []
    for left_index, left in enumerate(profiles):
        for right_index in range(left_index + 1, len(profiles)):
            evidence = identity_evidence_from_profiles(left, profiles[right_index])
            if evidence.equivalent or evidence.rule == "no_identity_evidence":
                continue
            rows.append(
                RejectedIdentityComparison(
                    left_record_ref=records[left_index].record_ref,
                    right_record_ref=records[right_index].record_ref,
                    rule=evidence.rule,
                    conflicting_identifiers=list(evidence.conflicting_identifiers),
                )
            )
    return rows


def _field_state(value: Any, *, present: bool) -> FieldState:
    if not present:
        return "missing"
    if value is None:
        return "null"
    if value == "" or value == [] or value == {}:
        return "empty"
    return "value"


def _path_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value
