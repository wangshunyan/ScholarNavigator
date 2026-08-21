"""原子快照存储、稳定键、完整性校验与覆盖率检查。"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.metrics import paper_identifier_set
from scholar_agent.evaluation.snapshots.schemas import (
    CONNECTOR_VERSIONS,
    QUERY_ADAPTER_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    ReferenceSnapshotEntry,
    RetrievalSnapshotEntry,
    SnapshotGroupObservation,
    SnapshotManifest,
    SnapshotPlanRound,
)


class SnapshotError(RuntimeError):
    """快照错误基类。"""


class SnapshotMissingError(SnapshotError):
    """Replay 请求缺少快照。"""


class SnapshotConflictError(SnapshotError):
    """同一键对应不同内容且未允许覆盖。"""


class SnapshotIntegrityError(SnapshotError):
    """快照 Schema、键或内容哈希无效。"""


EntryT = TypeVar("EntryT", RetrievalSnapshotEntry, ReferenceSnapshotEntry)
EntryKind = Literal["retrieval", "references"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_snapshot_query(query: str) -> str:
    """只规范 Unicode、控制字符和空白，保留字段语法、引号与词序。"""

    normalized = unicodedata.normalize("NFKC", str(query))
    visible = "".join(
        character
        for character in normalized
        if character in "\t\n\r" or unicodedata.category(character) != "Cc"
    )
    return " ".join(visible.split())


def retrieval_snapshot_key(
    *,
    source: str,
    adapted_query: str,
    limit: int,
    adapter_policy: str,
    query_adapter_version: str = QUERY_ADAPTER_VERSION,
    connector_version: str,
    query_evolution_policy: str | None = None,
    query_planning_policy: str | None = None,
    query_planner_version: str | None = None,
    schema_version: str = SNAPSHOT_SCHEMA_VERSION,
) -> tuple[str, str]:
    normalized_query = normalize_snapshot_query(adapted_query)
    payload = {
        "adapter_policy": adapter_policy,
        "adapted_query": normalized_query,
        "connector_version": connector_version,
        "limit": int(limit),
        "query_adapter_version": query_adapter_version,
        "schema_version": schema_version,
        "source": source.strip().casefold(),
    }
    # 旧 seed_expansion 与初始检索继续使用历史键；新策略单独命名空间化。
    if query_evolution_policy in {"coverage_gap", "llm_feedback"}:
        payload["query_evolution_policy"] = query_evolution_policy
    # current_rules 保留历史键；候选 planner 使用独立版本化命名空间。
    if query_planning_policy in {
        "concept_projection",
        "controlled_relaxation",
        "disjunctive_facets",
        "current_plus_disjunctive",
        "facet_union",
        "facet_balanced",
        "llm_semantic",
        "llm_constrained_rewrite",
    }:
        payload["query_planning_policy"] = query_planning_policy
        payload["query_planner_version"] = query_planner_version
    return _stable_hash(payload), normalized_query


def canonical_seed_identifier(paper: Paper) -> str | None:
    identifiers = paper_identifier_set(paper)
    for prefix in ("openalex:", "doi:"):
        matches = sorted(value for value in identifiers if value.startswith(prefix))
        if matches:
            return matches[0]
    return None


def reference_snapshot_key(
    *,
    seed_identifier: str,
    limit: int,
    connector_version: str,
    source: str = "openalex",
    schema_version: str = SNAPSHOT_SCHEMA_VERSION,
) -> str:
    payload = {
        "connector_version": connector_version,
        "limit": int(limit),
        "schema_version": schema_version,
        "seed_identifier": seed_identifier.strip().casefold(),
        "source": source.strip().casefold(),
    }
    return _stable_hash(payload)


def entry_content_hash(entry: BaseModel | dict[str, Any]) -> str:
    payload = (
        entry.model_dump(mode="json") if isinstance(entry, BaseModel) else dict(entry)
    )
    payload.pop("content_hash", None)
    payload.pop("recorded_at", None)
    return _stable_hash(payload)


class SnapshotStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.retrieval_dir = self.root / "retrieval"
        self.reference_dir = self.root / "references"
        self.manifest_path = self.root / "manifest.json"

    def read_retrieval(self, key: str) -> RetrievalSnapshotEntry:
        return self._read_entry("retrieval", key, RetrievalSnapshotEntry)

    def read_reference(self, key: str) -> ReferenceSnapshotEntry:
        return self._read_entry("references", key, ReferenceSnapshotEntry)

    def write_retrieval(
        self,
        entry: RetrievalSnapshotEntry,
        *,
        overwrite: bool = False,
    ) -> bool:
        return self._write_entry("retrieval", entry, overwrite=overwrite)

    def write_reference(
        self,
        entry: ReferenceSnapshotEntry,
        *,
        overwrite: bool = False,
    ) -> bool:
        return self._write_entry("references", entry, overwrite=overwrite)

    def ensure_manifest(self, manifest: SnapshotManifest) -> SnapshotManifest:
        if self.manifest_path.is_file():
            existing = self.read_manifest()
            mismatched = [
                field
                for field in (
                    "snapshot_name",
                    "schema_version",
                    "dataset",
                    "split",
                    "offset",
                    "limit",
                    "sources",
                    "adapter_policy",
                    "query_adapter_version",
                    "run_profile",
                    "budgets",
                    "llm_enabled",
                    "query_understanding_prompt",
                    "llm_query_planning_prompt",
                    "judgement_prompt",
                )
                if getattr(existing, field) != getattr(manifest, field)
            ]
            connector_conflicts = [
                name
                for name, version in existing.connector_versions.items()
                if manifest.connector_versions.get(name) != version
            ]
            if connector_conflicts:
                mismatched.append("connector_versions")
            if mismatched:
                raise SnapshotConflictError(
                    "snapshot_manifest_incompatible:" + ",".join(mismatched)
                )
            merged_connector_versions = {
                **existing.connector_versions,
                **manifest.connector_versions,
            }
            if (
                existing.query_planner_version != manifest.query_planner_version
                or existing.connector_versions != merged_connector_versions
            ):
                existing = existing.model_copy(
                    update={
                        "query_planner_version": manifest.query_planner_version,
                        "connector_versions": merged_connector_versions,
                        "updated_at": utc_now(),
                    }
                )
                self._write_manifest(self._with_entry_counts(existing))
            return existing
        self._write_manifest(self._with_entry_counts(manifest))
        return manifest

    def read_manifest(self) -> SnapshotManifest:
        try:
            return SnapshotManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise SnapshotIntegrityError("snapshot_manifest_invalid") from exc

    def update_group(
        self,
        group_name: str,
        observation: SnapshotGroupObservation,
    ) -> SnapshotManifest:
        manifest = self.read_manifest()
        groups = dict(manifest.groups)
        groups[group_name] = observation
        updated = self._with_entry_counts(
            manifest.model_copy(
                update={"groups": groups, "updated_at": utc_now()}
            )
        )
        self._write_manifest(updated)
        return updated

    def invalidate_verified_groups(self, key: str) -> list[str]:
        """新条目刷新共享覆盖，并使引用它的已验证组失效。"""

        manifest = self.read_manifest()
        groups = dict(manifest.groups)
        invalidated: list[str] = []
        for name, observation in groups.items():
            if (
                key not in observation.retrieval_keys
                and key not in observation.reference_keys
            ):
                continue
            missing_retrieval = [
                item
                for item in observation.retrieval_keys
                if not (self.retrieval_dir / f"{item}.json").is_file()
            ]
            missing_references = [
                item
                for item in observation.reference_keys
                if not (self.reference_dir / f"{item}.json").is_file()
            ]
            success_count = 0
            failed_count = 0
            for item in observation.retrieval_keys:
                try:
                    status = self.read_retrieval(item).status
                except SnapshotMissingError:
                    continue
                success_count += status == "success"
                failed_count += status == "failed"
            for item in observation.reference_keys:
                try:
                    status = self.read_reference(item).status
                except SnapshotMissingError:
                    continue
                success_count += status == "success"
                failed_count += status == "failed"
            missing_count = len(missing_retrieval) + len(missing_references)
            replay_ready = bool(
                observation.retrieval_keys or observation.reference_keys
            ) and missing_count == 0
            groups[name] = observation.model_copy(
                update={
                    "missing_retrieval_keys": missing_retrieval,
                    "missing_reference_keys": missing_references,
                    "collection_completed": replay_ready,
                    "replay_ready": replay_ready,
                    "replay_verified": False,
                    "required_key_count": len(observation.retrieval_keys)
                    + len(observation.reference_keys),
                    "success_key_count": success_count,
                    "failed_key_count": failed_count,
                    "missing_key_count": missing_count,
                    "stop_reason": None if replay_ready else observation.stop_reason,
                    "updated_at": utc_now(),
                }
            )
            invalidated.append(name)
        if invalidated:
            self._write_manifest(
                self._with_entry_counts(
                    manifest.model_copy(
                        update={"groups": groups, "updated_at": utc_now()}
                    )
                )
            )
        return invalidated

    def inspect(self) -> dict[str, Any]:
        manifest = self.read_manifest()
        retrieval = self._inspect_kind("retrieval", RetrievalSnapshotEntry)
        references = self._inspect_kind("references", ReferenceSnapshotEntry)
        retrieval_keys = {entry.key for entry in retrieval["entries"]}
        reference_keys = {entry.key for entry in references["entries"]}
        entry_by_key = {entry.key: entry for entry in [
            *retrieval["entries"],
            *references["entries"],
        ]}
        planned_source_by_key = self._planned_source_by_key()
        groups = {
            name: {
                "completed": observation.completed,
                "collection_started": observation.collection_started,
                "collection_completed": observation.collection_completed,
                "replay_ready": bool(
                    (observation.retrieval_keys or observation.reference_keys)
                    and not (
                        (set(observation.missing_retrieval_keys) - retrieval_keys)
                        | (set(observation.retrieval_keys) - retrieval_keys)
                    )
                    and not (
                        (set(observation.missing_reference_keys) - reference_keys)
                        | (set(observation.reference_keys) - reference_keys)
                    )
                ),
                "replay_verified": observation.replay_verified,
                "retrieval_key_count": len(observation.retrieval_keys),
                "reference_key_count": len(observation.reference_keys),
                "required_retrieval_keys": list(observation.retrieval_keys),
                "required_reference_keys": list(observation.reference_keys),
                "required_key_count": len(observation.retrieval_keys)
                + len(observation.reference_keys),
                "present_success_entries": sum(
                    entry_by_key[key].status == "success"
                    for key in [
                        *observation.retrieval_keys,
                        *observation.reference_keys,
                    ]
                    if key in entry_by_key
                ),
                "present_failed_entries": sum(
                    entry_by_key[key].status == "failed"
                    for key in [
                        *observation.retrieval_keys,
                        *observation.reference_keys,
                    ]
                    if key in entry_by_key
                ),
                "missing_entries": len(
                    (set(observation.missing_retrieval_keys) - retrieval_keys)
                    | (set(observation.retrieval_keys) - retrieval_keys)
                )
                + len(
                    (set(observation.missing_reference_keys) - reference_keys)
                    | (set(observation.reference_keys) - reference_keys)
                ),
                "missing_retrieval_keys": sorted(
                    (set(observation.missing_retrieval_keys) - retrieval_keys)
                    | (set(observation.retrieval_keys) - retrieval_keys)
                ),
                "missing_reference_keys": sorted(
                    (set(observation.missing_reference_keys) - reference_keys)
                    | (set(observation.reference_keys) - reference_keys)
                ),
                "plan_rounds": observation.plan_rounds,
                "last_plan_round": observation.last_plan_round,
                "stop_reason": observation.stop_reason,
                "missing_keys_by_source": _count_sources(
                    [
                        planned_source_by_key.get(key, "unknown")
                        for key in sorted(
                            (set(observation.missing_retrieval_keys) - retrieval_keys)
                            | (set(observation.retrieval_keys) - retrieval_keys)
                            | (set(observation.missing_reference_keys) - reference_keys)
                            | (set(observation.reference_keys) - reference_keys)
                        )
                    ]
                ),
                "failed_keys_by_source": _count_sources(
                    [
                        entry_by_key[key].source
                        for key in [
                            *observation.retrieval_keys,
                            *observation.reference_keys,
                        ]
                        if key in entry_by_key
                        and entry_by_key[key].status == "failed"
                    ]
                ),
            }
            for name, observation in sorted(manifest.groups.items())
        }
        entries = [*retrieval.pop("entries"), *references.pop("entries")]
        diagnostics = [entry.diagnostics for entry in entries]
        return {
            "snapshot_name": manifest.snapshot_name,
            "schema_version": manifest.schema_version,
            "retrieval_entries": retrieval["entry_count"],
            "reference_entries": references["entry_count"],
            "successful_entries": sum(entry.status == "success" for entry in entries),
            "failed_entries": sum(entry.status == "failed" for entry in entries),
            "sources": sorted({entry.source for entry in entries}),
            "request_count_recorded": sum(item.request_count for item in diagnostics),
            "retry_count_recorded": sum(item.retry_count for item in diagnostics),
            "missing_keys_observed": sorted(
                {
                    key
                    for observation in manifest.groups.values()
                    for key in [
                        *observation.missing_retrieval_keys,
                        *observation.missing_reference_keys,
                    ]
                }
            ),
            "duplicate_keys": retrieval["duplicate_keys"] + references["duplicate_keys"],
            "invalid_entries": retrieval["invalid_entries"] + references["invalid_entries"],
            "hash_mismatch_entries": retrieval["hash_mismatch_entries"]
            + references["hash_mismatch_entries"],
            "groups": groups,
        }

    def _planned_source_by_key(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        plans_root = self.root / "plans"
        for path in sorted(plans_root.rglob("plan_round_*.json")):
            try:
                plan = SnapshotPlanRound.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError):
                continue
            for entry in plan.entries:
                sources[entry.key] = entry.source
        return sources

    def _read_entry(
        self,
        kind: EntryKind,
        key: str,
        model: type[EntryT],
    ) -> EntryT:
        path = self._entry_path(kind, key)
        if not path.is_file():
            raise SnapshotMissingError(f"snapshot_missing:{kind}:{key}")
        try:
            raw_text = path.read_text(encoding="utf-8")
            raw_payload = json.loads(raw_text)
            entry = model.model_validate(raw_payload)
        except (OSError, ValidationError) as exc:
            raise SnapshotIntegrityError(f"snapshot_invalid:{kind}:{key}") from exc
        except json.JSONDecodeError as exc:
            raise SnapshotIntegrityError(f"snapshot_invalid:{kind}:{key}") from exc
        if entry.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotIntegrityError(f"snapshot_schema_incompatible:{kind}:{key}")
        if entry.key != key:
            raise SnapshotIntegrityError(f"snapshot_key_mismatch:{kind}:{key}")
        parsed_hash_matches = entry.content_hash == entry_content_hash(entry)
        legacy_reference_matches = (
            kind == "references"
            and not any(
                field in raw_payload
                for field in (
                    "reference_batch_status",
                    "missing_reference_ids",
                    "reference_batch_count",
                    "supplemental_request_count",
                )
            )
            and entry.content_hash == entry_content_hash(raw_payload)
        )
        legacy_s2orc_matches = _legacy_s2orc_hash_matches(entry, raw_payload)
        if not (
            parsed_hash_matches or legacy_reference_matches or legacy_s2orc_matches
        ):
            raise SnapshotIntegrityError(f"snapshot_hash_mismatch:{kind}:{key}")
        return entry

    def _write_entry(
        self,
        kind: EntryKind,
        entry: EntryT,
        *,
        overwrite: bool,
    ) -> bool:
        if entry.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotIntegrityError("snapshot_schema_incompatible")
        expected_hash = entry_content_hash(entry)
        if entry.content_hash != expected_hash:
            raise SnapshotIntegrityError("snapshot_content_hash_invalid")
        path = self._entry_path(kind, entry.key)
        if path.is_file():
            existing = self._read_entry(
                kind,
                entry.key,
                RetrievalSnapshotEntry if kind == "retrieval" else ReferenceSnapshotEntry,
            )
            if existing.content_hash == entry.content_hash:
                return False
            if not overwrite:
                raise SnapshotConflictError(f"snapshot_content_conflict:{kind}:{entry.key}")
        self._atomic_write_json(path, entry.model_dump(mode="json"))
        return True

    def _inspect_kind(self, kind: EntryKind, model: type[EntryT]) -> dict[str, Any]:
        directory = self.retrieval_dir if kind == "retrieval" else self.reference_dir
        entries: list[EntryT] = []
        seen: set[str] = set()
        duplicate_keys = 0
        invalid_entries = 0
        hash_mismatch_entries = 0
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            key = path.stem
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            try:
                entries.append(self._read_entry(kind, key, model))
            except SnapshotIntegrityError as exc:
                invalid_entries += 1
                if "hash_mismatch" in str(exc):
                    hash_mismatch_entries += 1
        return {
            "entry_count": len(entries),
            "entries": entries,
            "duplicate_keys": duplicate_keys,
            "invalid_entries": invalid_entries,
            "hash_mismatch_entries": hash_mismatch_entries,
        }

    def _with_entry_counts(self, manifest: SnapshotManifest) -> SnapshotManifest:
        retrieval_count = len(list(self.retrieval_dir.glob("*.json"))) if self.retrieval_dir.is_dir() else 0
        reference_count = len(list(self.reference_dir.glob("*.json"))) if self.reference_dir.is_dir() else 0
        return manifest.model_copy(
            update={
                "retrieval_entry_count": retrieval_count,
                "reference_entry_count": reference_count,
            }
        )

    def _write_manifest(self, manifest: SnapshotManifest) -> None:
        self._atomic_write_json(
            self.manifest_path,
            manifest.model_dump(mode="json"),
        )

    def _entry_path(self, kind: EntryKind, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise SnapshotIntegrityError("snapshot_key_invalid")
        directory = self.retrieval_dir if kind == "retrieval" else self.reference_dir
        return directory / f"{key}.json"

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _legacy_s2orc_hash_matches(
    entry: RetrievalSnapshotEntry | ReferenceSnapshotEntry,
    raw_payload: dict[str, Any],
) -> bool:
    """Accept an old hash only when it differs by the new empty ID field."""

    raw_papers = raw_payload.get("papers")
    if not isinstance(raw_papers, list) or not raw_papers:
        return False
    if any(
        isinstance(paper, dict)
        and isinstance(paper.get("identifiers"), dict)
        and "s2orc_corpus_id" in paper["identifiers"]
        for paper in raw_papers
    ):
        return False
    compatible = entry.model_dump(mode="json")
    for paper in compatible.get("papers", []):
        identifiers = paper.get("identifiers") if isinstance(paper, dict) else None
        if not isinstance(identifiers, dict):
            return False
        if identifiers.get("s2orc_corpus_id") is not None:
            return False
        identifiers.pop("s2orc_corpus_id", None)
    return (
        entry.content_hash == entry_content_hash(raw_payload)
        and entry.content_hash == entry_content_hash(compatible)
    )


def connector_version(source: str, *, references: bool = False) -> str:
    if source == "local_bm25" and not references:
        from scholar_agent.connectors.local_bm25 import (
            local_bm25_connector_version,
        )

        return local_bm25_connector_version()
    if source == "local_hybrid" and not references:
        from scholar_agent.connectors.local_hybrid import (
            local_hybrid_connector_version,
        )

        return local_hybrid_connector_version()
    key = "openalex_references" if references else source
    try:
        return CONNECTOR_VERSIONS[key]
    except KeyError as exc:
        raise ValueError(f"snapshot_unsupported_source:{source}") from exc


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_sources(sources: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))
