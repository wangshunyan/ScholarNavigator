"""固定 holdout30 的事后 Retrieval 召回审计。

本模块只读取已经完成的 SearchService 结果，并把 gold 用于离线分析请求；
不会被生产检索路径导入，也不会改变查询或排序策略。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request

from pydantic import BaseModel, Field, ValidationError

from scholar_agent.connectors.arxiv import (
    ARXIV_NS,
    ARXIV_QUERY_URL,
    ATOM_NS,
    _request_feed_detailed,
)
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.evaluation_schemas import EvalQuery
from scholar_agent.evaluation.holdout_comparison import (
    HOLDOUT_CASE_IDS,
    HOLDOUT_LIMIT,
    HOLDOUT_OFFSET,
    query_features,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_SCHEMA_VERSION = "1"
AUDIT_SNAPSHOT_NAME = "retrieval_recall_audit_holdout30_v1"
AUDIT_MAX_RESULTS = 100
METADATA_BATCH_SIZE = 25
AuditMode = Literal["plan", "record-missing", "replay"]
AuditRequestKind = Literal[
    "metadata_by_id",
    "current_query",
    "exact_title",
    "normalized_title",
    "title_core_terms",
]
FailureReason = Literal[
    "source_unavailable",
    "identifier_available_but_query_not_matched",
    "lexical_mismatch",
    "query_over_restrictive",
    "query_over_broad_ranked_below_limit",
    "adapter_term_loss",
    "result_limit_truncation",
    "metadata_mismatch",
    "source_error",
    "unknown",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "provide",
    "reference",
    "references",
    "research",
    "that",
    "the",
    "their",
    "these",
    "to",
    "using",
    "what",
    "which",
    "with",
    "work",
    "works",
    "you",
}


class AuditRequest(BaseModel):
    key: str = Field(min_length=64, max_length=64)
    kind: AuditRequestKind
    query: str | None = None
    arxiv_ids: list[str] = Field(default_factory=list)
    max_results: int = Field(default=AUDIT_MAX_RESULTS, ge=1, le=100)


class AuditPaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    categories: list[str] = Field(default_factory=list)
    doi: str | None = None


class AuditSnapshotEntry(BaseModel):
    schema_version: str = AUDIT_SCHEMA_VERSION
    key: str = Field(min_length=64, max_length=64)
    request: AuditRequest
    status: Literal["success", "failed"]
    papers: list[AuditPaperMetadata] = Field(default_factory=list)
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)
    recorded_at: str
    content_hash: str = Field(min_length=64, max_length=64)


class AuditRuntimeCost(BaseModel):
    snapshot_hits: int = Field(default=0, ge=0)
    snapshot_writes: int = Field(default=0, ge=0)
    execution_request_count: int = Field(default=0, ge=0)
    execution_retry_count: int = Field(default=0, ge=0)
    execution_error_count: int = Field(default=0, ge=0)
    execution_network_wait_seconds: float = Field(default=0.0, ge=0.0)
    execution_latency_seconds: float = Field(default=0.0, ge=0.0)


class AuditSnapshotStore:
    """带内容哈希的最小审计快照存储。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries_dir = self.root / "entries"
        self.manifest_path = self.root / "manifest.json"
        self.plan_path = self.root / "plan.json"
        self.collection_path = self.root / "collection_result.json"

    def read(self, key: str) -> AuditSnapshotEntry:
        path = self._path(key)
        if not path.is_file():
            raise ValueError(f"audit_snapshot_missing:{key}")
        try:
            entry = AuditSnapshotEntry.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ValueError(f"audit_snapshot_invalid:{key}") from exc
        if entry.key != key or entry.request.key != key:
            raise ValueError(f"audit_snapshot_key_mismatch:{key}")
        if entry.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError(f"audit_snapshot_schema_incompatible:{key}")
        if entry.content_hash != audit_entry_hash(entry):
            raise ValueError(f"audit_snapshot_hash_mismatch:{key}")
        return entry

    def read_optional(self, key: str) -> AuditSnapshotEntry | None:
        return self.read(key) if self._path(key).is_file() else None

    def write(self, entry: AuditSnapshotEntry) -> bool:
        existing = self.read_optional(entry.key)
        if existing is not None:
            if existing.content_hash != entry.content_hash:
                raise ValueError(f"audit_snapshot_content_conflict:{entry.key}")
            return False
        _atomic_write_json(self._path(entry.key), entry.model_dump(mode="json"))
        return True

    def entries(self) -> list[AuditSnapshotEntry]:
        return [self.read(path.stem) for path in sorted(self.entries_dir.glob("*.json"))]

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("audit_snapshot_key_invalid")
        return self.entries_dir / f"{key}.json"


class AuditSnapshotRuntime:
    """审计请求的 plan、串行 record-missing 和严格 replay。"""

    def __init__(self, store: AuditSnapshotStore, *, mode: AuditMode) -> None:
        self.store = store
        self.mode = mode
        self.cost = AuditRuntimeCost()

    def resolve(
        self,
        request: AuditRequest,
        fetcher: Callable[[AuditRequest], AuditSnapshotEntry],
    ) -> AuditSnapshotEntry | None:
        existing = self.store.read_optional(request.key)
        if self.mode == "plan":
            return existing
        if existing is not None:
            self.cost = self.cost.model_copy(
                update={"snapshot_hits": self.cost.snapshot_hits + 1}
            )
            return existing
        if self.mode == "replay":
            raise ValueError(f"audit_snapshot_missing:{request.key}")
        entry = fetcher(request)
        wrote = self.store.write(entry)
        diagnostics = entry.diagnostics
        self.cost = AuditRuntimeCost(
            snapshot_hits=self.cost.snapshot_hits,
            snapshot_writes=self.cost.snapshot_writes + int(wrote),
            execution_request_count=(
                self.cost.execution_request_count + diagnostics.request_count
            ),
            execution_retry_count=(
                self.cost.execution_retry_count + diagnostics.retry_count
            ),
            execution_error_count=(
                self.cost.execution_error_count + diagnostics.error_count
            ),
            execution_network_wait_seconds=(
                self.cost.execution_network_wait_seconds
                + diagnostics.rate_limit_wait_seconds
            ),
            execution_latency_seconds=(
                self.cost.execution_latency_seconds + entry.recorded_latency_seconds
            ),
        )
        return entry


def make_audit_request(
    kind: AuditRequestKind,
    *,
    query: str | None = None,
    arxiv_ids: Iterable[str] = (),
    max_results: int = AUDIT_MAX_RESULTS,
) -> AuditRequest:
    normalized_ids = [_normalize_arxiv_id(value) for value in arxiv_ids]
    normalized_query = " ".join(str(query or "").split()) or None
    payload = {
        "arxiv_ids": normalized_ids,
        "kind": kind,
        "max_results": max_results,
        "query": normalized_query,
        "schema_version": AUDIT_SCHEMA_VERSION,
    }
    return AuditRequest(
        key=_stable_hash(payload),
        kind=kind,
        query=normalized_query,
        arxiv_ids=normalized_ids,
        max_results=max_results,
    )


def build_audit_requests(
    queries: list[EvalQuery],
    result_rows: dict[str, dict[str, Any]],
    retrieval_store: SnapshotStore,
) -> tuple[list[AuditRequest], dict[str, Any]]:
    """构造固定请求集；gold 只在这里进入离线审计。"""

    _validate_holdout_queries(queries)
    unique_gold: dict[str, str] = {}
    case_current_requests: dict[str, list[str]] = {}
    gold_requests: dict[str, dict[str, str]] = {}
    requests: dict[str, AuditRequest] = {}

    for query in queries:
        row = result_rows[query.query_id]
        observed = (row.get("snapshot_cost_report") or {}).get(
            "observed_retrieval_keys"
        ) or []
        current_keys: list[str] = []
        for snapshot_key in observed:
            source_entry = retrieval_store.read_retrieval(str(snapshot_key))
            if source_entry.source != "arxiv":
                continue
            request = make_audit_request(
                "current_query",
                query=source_entry.adapted_query,
            )
            requests.setdefault(request.key, request)
            current_keys.append(request.key)
        case_current_requests[query.query_id] = list(dict.fromkeys(current_keys))

        for gold in query.gold_papers:
            if not gold.arxiv_id:
                continue
            arxiv_id = _normalize_arxiv_id(gold.arxiv_id)
            unique_gold.setdefault(arxiv_id, gold.title or "")
            exact = make_audit_request(
                "exact_title",
                query=exact_title_query(gold.title or ""),
            )
            normalized = make_audit_request(
                "normalized_title",
                query=normalized_title_query(gold.title or ""),
            )
            core = make_audit_request(
                "title_core_terms",
                query=title_core_query(gold.title or ""),
            )
            for request in (exact, normalized, core):
                requests.setdefault(request.key, request)
            gold_requests[arxiv_id] = {
                "exact_title": exact.key,
                "normalized_title": normalized.key,
                "title_core_terms": core.key,
            }

    gold_ids = list(unique_gold)
    metadata_keys: list[str] = []
    for index in range(0, len(gold_ids), METADATA_BATCH_SIZE):
        request = make_audit_request(
            "metadata_by_id",
            arxiv_ids=gold_ids[index : index + METADATA_BATCH_SIZE],
            max_results=min(METADATA_BATCH_SIZE, len(gold_ids) - index),
        )
        requests.setdefault(request.key, request)
        metadata_keys.append(request.key)

    ordered = sorted(
        requests.values(),
        key=lambda item: (
            _kind_priority(item.kind),
            item.query or "",
            item.arxiv_ids,
            item.key,
        ),
    )
    index = {
        "case_current_requests": case_current_requests,
        "gold_requests": gold_requests,
        "metadata_requests": metadata_keys,
        "gold_ids": gold_ids,
    }
    return ordered, index


def write_audit_plan(
    store: AuditSnapshotStore,
    requests: list[AuditRequest],
    *,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    present = []
    missing = []
    failed = []
    required_keys = {request.key for request in requests}
    for request in requests:
        entry = store.read_optional(request.key)
        if entry is None:
            missing.append(request.key)
        elif entry.status == "failed":
            present.append(request.key)
            failed.append(request.key)
        else:
            present.append(request.key)
    stored_keys = {entry.key for entry in store.entries()}
    manifest = {
        "snapshot_name": AUDIT_SNAPSHOT_NAME,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol": {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "case_ids": list(HOLDOUT_CASE_IDS),
            "source": "arxiv",
            "max_results": AUDIT_MAX_RESULTS,
            "metadata_batch_size": METADATA_BATCH_SIZE,
            "gold_post_search_only": True,
            "production_integration": False,
        },
        "input_metadata": input_metadata,
        "request_count": len(requests),
        "request_kind_counts": dict(
            sorted(Counter(request.kind for request in requests).items())
        ),
        "request_keys": [request.key for request in requests],
        "stored_entry_count": len(stored_keys),
        "unused_entry_count": len(stored_keys - required_keys),
        "present_count": len(present),
        "failed_count": len(failed),
        "missing_count": len(missing),
        "replay_ready": not missing,
    }
    plan = {
        "snapshot_name": AUDIT_SNAPSHOT_NAME,
        "requests": [request.model_dump(mode="json") for request in requests],
        "present_keys": present,
        "failed_keys": failed,
        "missing_keys": missing,
    }
    _atomic_write_json(store.manifest_path, manifest)
    _atomic_write_json(store.plan_path, plan)
    return manifest


def collect_audit_requests(
    store: AuditSnapshotStore,
    requests: list[AuditRequest],
    *,
    max_new_requests: int | None = None,
    source_failure_limit: int = 3,
    progress: Callable[[int, int, AuditRequest, AuditSnapshotEntry], None] | None = None,
) -> dict[str, Any]:
    runtime = AuditSnapshotRuntime(store, mode="record-missing")
    missing = [request for request in requests if store.read_optional(request.key) is None]
    selected = missing[:max_new_requests] if max_new_requests is not None else missing
    started = time.perf_counter()
    status_counts: Counter[str] = Counter()
    attempted = 0
    consecutive_failures = 0
    stop_reason = None
    for index, request in enumerate(selected, start=1):
        entry = runtime.resolve(request, fetch_arxiv_audit_request)
        if entry is None:
            raise AssertionError("record-missing must return an entry")
        status_counts[entry.status] += 1
        attempted += 1
        consecutive_failures = consecutive_failures + 1 if entry.status == "failed" else 0
        if progress is not None:
            progress(index, len(selected), request, entry)
        if consecutive_failures >= source_failure_limit:
            stop_reason = f"source_failure_limit:{source_failure_limit}"
            break
    remaining = [request.key for request in requests if store.read_optional(request.key) is None]
    result = {
        "selected_request_count": len(selected),
        "attempted_request_count": attempted,
        "status_counts": dict(sorted(status_counts.items())),
        "remaining_missing_count": len(remaining),
        "remaining_missing_keys": remaining,
        "replay_ready": not remaining,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - started,
        "execution_cost": runtime.cost.model_dump(mode="json"),
    }
    _atomic_write_json(store.collection_path, result)
    return result


def fetch_arxiv_audit_request(request: AuditRequest) -> AuditSnapshotEntry:
    """执行单个审计专用 arXiv 请求并冻结规范化元数据。"""

    params: dict[str, str]
    if request.kind == "metadata_by_id":
        params = {
            "id_list": ",".join(request.arxiv_ids),
            "start": "0",
            "max_results": str(request.max_results),
        }
    else:
        if not request.query:
            raise ValueError("audit search request requires query")
        params = {
            "search_query": request.query,
            "start": "0",
            "max_results": str(request.max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    raw_request = Request(
        f"{ARXIV_QUERY_URL}?{urlencode(params)}",
        headers={"User-Agent": "ScholarNavigator-RetrievalAudit"},
    )
    started = time.perf_counter()
    payload, error_message, warnings, diagnostics = _request_feed_detailed(
        raw_request,
        max_retries=1,
    )
    papers: list[AuditPaperMetadata] = []
    if payload is not None:
        try:
            papers = parse_arxiv_audit_feed(payload)
        except (ET.ParseError, ValueError) as exc:
            error_message = f"arxiv_audit_parse_failed:{exc}"
            warnings = [*warnings, error_message]
            diagnostics = diagnostics.model_copy(
                update={"error_count": diagnostics.error_count + 1}
            )
    elapsed = max(time.perf_counter() - started, diagnostics.latency_seconds)
    entry = AuditSnapshotEntry(
        key=request.key,
        request=request,
        status="failed" if error_message else "success",
        papers=papers,
        error_message=_sanitize(error_message),
        warnings=[_sanitize(item) or "" for item in warnings],
        diagnostics=diagnostics,
        recorded_latency_seconds=elapsed,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        content_hash="0" * 64,
    )
    return entry.model_copy(update={"content_hash": audit_entry_hash(entry)})


def parse_arxiv_audit_feed(payload: bytes | str) -> list[AuditPaperMetadata]:
    root = ET.fromstring(payload)
    papers: list[AuditPaperMetadata] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        landing_page = _element_text(entry.find(f"{ATOM_NS}id")) or ""
        arxiv_id = _normalize_arxiv_id(landing_page.rstrip("/").split("/")[-1])
        title = _normalize_space(_element_text(entry.find(f"{ATOM_NS}title")))
        if not arxiv_id or not title:
            continue
        published = _element_text(entry.find(f"{ATOM_NS}published")) or ""
        authors = [
            name
            for author in entry.findall(f"{ATOM_NS}author")
            if (
                name := _normalize_space(
                    _element_text(author.find(f"{ATOM_NS}name"))
                )
            )
        ]
        categories = list(
            dict.fromkeys(
                term
                for category in entry.findall(f"{ATOM_NS}category")
                if (term := _normalize_space(category.attrib.get("term")))
            )
        )
        papers.append(
            AuditPaperMetadata(
                arxiv_id=arxiv_id,
                title=title,
                abstract=(
                    _normalize_space(_element_text(entry.find(f"{ATOM_NS}summary")))
                    or ""
                ),
                authors=authors,
                year=_parse_year(published),
                categories=categories,
                doi=_normalize_space(_element_text(entry.find(f"{ARXIV_NS}doi"))),
            )
        )
    return papers


def analyze_retrieval_recall(
    queries: list[EvalQuery],
    result_rows: dict[str, dict[str, Any]],
    requests: list[AuditRequest],
    request_index: dict[str, Any],
    store: AuditSnapshotStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """严格从快照 replay，返回 gold、query 和聚合诊断。"""

    runtime = AuditSnapshotRuntime(store, mode="replay")
    entries: dict[str, AuditSnapshotEntry] = {}
    for request in requests:
        entry = runtime.resolve(request, fetch_arxiv_audit_request)
        if entry is None:
            raise AssertionError("replay must return an entry")
        entries[request.key] = entry
    if any(
        (
            runtime.cost.execution_request_count,
            runtime.cost.execution_retry_count,
            runtime.cost.execution_network_wait_seconds,
        )
    ):
        raise ValueError("audit replay executed network work")

    metadata: dict[str, AuditPaperMetadata] = {}
    metadata_errors: set[str] = set()
    for key in request_index["metadata_requests"]:
        entry = entries[key]
        if entry.status == "failed":
            metadata_errors.update(entry.request.arxiv_ids)
        for paper in entry.papers:
            metadata[paper.arxiv_id] = paper

    gold_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for query in queries:
        result_row = result_rows[query.query_id]
        current_entries = [
            entries[key]
            for key in request_index["case_current_requests"][query.query_id]
        ]
        case_gold_rows = []
        for gold_index, gold in enumerate(query.gold_papers):
            arxiv_id = _normalize_arxiv_id(gold.arxiv_id or "")
            gold_metadata = metadata.get(arxiv_id)
            oracle_keys = request_index["gold_requests"].get(arxiv_id) or {}
            oracle_entries = {
                kind: entries[key] for kind, key in oracle_keys.items()
            }
            row = _gold_audit_row(
                query=query,
                result_row=result_row,
                gold_index=gold_index,
                arxiv_id=arxiv_id,
                dataset_title=gold.title or "",
                metadata=gold_metadata,
                metadata_source_error=arxiv_id in metadata_errors,
                current_entries=current_entries,
                oracle_entries=oracle_entries,
            )
            gold_rows.append(row)
            case_gold_rows.append(row)
        query_rows.append(_query_audit_row(query, result_row, case_gold_rows))

    aggregate = _aggregate_audit(gold_rows, query_rows, entries, runtime.cost)
    return gold_rows, query_rows, aggregate


def lexical_overlap(query: str, document: str) -> dict[str, Any]:
    query_tokens = content_tokens(query)
    document_tokens = set(content_tokens(document))
    matched = [token for token in query_tokens if token in document_tokens]
    return {
        "query_tokens": query_tokens,
        "matched_tokens": matched,
        "missing_tokens": [token for token in query_tokens if token not in document_tokens],
        "query_coverage": len(matched) / len(query_tokens) if query_tokens else 0.0,
        "document_jaccard": (
            len(set(matched)) / len(set(query_tokens) | document_tokens)
            if query_tokens or document_tokens
            else 0.0
        ),
    }


def adapter_term_loss(
    original_query: str,
    adapted_queries: Iterable[str],
    gold_text: str,
) -> dict[str, Any]:
    original = content_tokens(original_query)
    gold_tokens = set(content_tokens(gold_text))
    relevant_original = [token for token in original if token in gold_tokens]
    adapted_tokens = {
        token
        for query in adapted_queries
        for token in content_tokens(_strip_arxiv_field_syntax(query))
    }
    lost = [token for token in relevant_original if token not in adapted_tokens]
    return {
        "relevant_original_terms": relevant_original,
        "retained_terms": [
            token for token in relevant_original if token in adapted_tokens
        ],
        "lost_terms": lost,
        "loss_detected": bool(lost),
    }


def classify_failure(signals: dict[str, Any]) -> FailureReason:
    """按可观测证据和固定优先级返回单一失败原因。"""

    if signals.get("source_error"):
        return "source_error"
    if not signals.get("identifier_available"):
        return "source_unavailable"
    if signals.get("metadata_mismatch"):
        return "metadata_mismatch"
    current_rank = signals.get("current_query_rank")
    if isinstance(current_rank, int) and 21 <= current_rank <= 50:
        return "result_limit_truncation"
    if isinstance(current_rank, int) and 51 <= current_rank <= 100:
        return "query_over_broad_ranked_below_limit"
    if signals.get("adapter_term_loss"):
        return "adapter_term_loss"
    if float(signals.get("lexical_query_coverage") or 0.0) < 0.15:
        return "lexical_mismatch"
    if signals.get("query_over_restrictive"):
        return "query_over_restrictive"
    if signals.get("any_title_oracle_hit"):
        return "identifier_available_but_query_not_matched"
    return "unknown"


def recall_at_k(ranks: Iterable[int | None], k: int) -> dict[str, Any]:
    items = list(ranks)
    recovered = sum(isinstance(rank, int) and rank <= k for rank in items)
    return {
        "k": k,
        "recovered_count": recovered,
        "total_count": len(items),
        "recall": recovered / len(items) if items else 0.0,
    }


def exact_title_query(title: str) -> str:
    phrase = _normalize_space(title) or ""
    phrase = phrase.replace('"', " ")
    return f'ti:"{phrase}"'


def normalized_title_query(title: str) -> str:
    tokens = _all_tokens(title)
    if not tokens:
        return "ti:unknown"
    return f'ti:"{" ".join(tokens)}"'


def title_core_query(title: str) -> str:
    tokens = content_tokens(title)
    if not tokens:
        tokens = _all_tokens(title)
    unique = list(dict.fromkeys(tokens))
    selected = sorted(unique, key=lambda token: (-len(token), unique.index(token)))[:4]
    selected_set = set(selected)
    ordered = [token for token in unique if token in selected_set]
    return " AND ".join(f"all:{token}" for token in ordered) or "all:unknown"


def content_tokens(value: str) -> list[str]:
    return list(
        dict.fromkeys(
            token
            for token in _all_tokens(value)
            if token not in _STOPWORDS and len(token) > 1
        )
    )


def audit_entry_hash(entry: AuditSnapshotEntry | dict[str, Any]) -> str:
    payload = (
        entry.model_dump(mode="json") if isinstance(entry, BaseModel) else dict(entry)
    )
    payload.pop("content_hash", None)
    payload.pop("recorded_at", None)
    return _stable_hash(payload)


def write_audit_outputs(
    output_dir: Path | str,
    gold_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output_dir).expanduser().resolve()
    _atomic_write_jsonl(root / "gold_audit.jsonl", gold_rows)
    _atomic_write_jsonl(root / "query_summary.jsonl", query_rows)
    _atomic_write_json(root / "aggregate.json", aggregate)
    _atomic_write_text(root / "summary.md", audit_summary_markdown(aggregate))


def audit_summary_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# holdout30 检索召回失败审计",
        "",
        "> 仅作固定保留集事后诊断；gold 和 oracle 查询不进入生产检索。",
        "",
        f"- gold：{aggregate['gold_count']}；原候选已召回："
        f"{aggregate['existing_retrieved_gold_count']}。",
        f"- arXiv ID 可用：{aggregate['identifier_available_count']}"
        f"（{aggregate['identifier_available_rate']:.6f}）。",
        "- exact/normalized/core title 可找回："
        f"{aggregate['exact_title_recovered_count']}/"
        f"{aggregate['normalized_title_recovered_count']}/"
        f"{aggregate['title_core_recovered_count']}。",
        "",
        "## 当前查询的离线 Top-K 上限",
        "",
        "| K | 找回 gold | Recall |",
        "|---:|---:|---:|",
    ]
    for key in ("20", "50", "100"):
        item = aggregate["current_query_oracle_recall_at_k"][key]
        lines.append(f"| {key} | {item['recovered_count']} | {item['recall']:.6f} |")
    lines.extend(["", "## 未召回原因", ""])
    for reason, count in aggregate["failure_reason_distribution"].items():
        lines.append(f"- `{reason}`：{count}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 主导问题：`{aggregate['primary_bottleneck_family']}`"
            f"（细分类：`{aggregate['primary_bottleneck']}`）。",
            f"- 下一受控实验：{aggregate['next_experiment_recommendation']}",
            "- Replay 执行期 HTTP、重试和网络等待均为 0。",
        ]
    )
    return "\n".join(lines) + "\n"


def _gold_audit_row(
    *,
    query: EvalQuery,
    result_row: dict[str, Any],
    gold_index: int,
    arxiv_id: str,
    dataset_title: str,
    metadata: AuditPaperMetadata | None,
    metadata_source_error: bool,
    current_entries: list[AuditSnapshotEntry],
    oracle_entries: dict[str, AuditSnapshotEntry],
) -> dict[str, Any]:
    metadata_text = (
        f"{metadata.title} {metadata.abstract}" if metadata is not None else dataset_title
    )
    current_rank = _minimum_rank(current_entries, arxiv_id)
    exact_rank = _minimum_rank([oracle_entries.get("exact_title")], arxiv_id)
    normalized_rank = _minimum_rank(
        [oracle_entries.get("normalized_title")], arxiv_id
    )
    core_rank = _minimum_rank([oracle_entries.get("title_core_terms")], arxiv_id)
    original_overlap = lexical_overlap(query.query, metadata_text)
    planning = ((result_row.get("stage_diagnostics") or {}).get(
        "initial_query_planning"
    ) or {})
    subqueries = planning.get("subqueries") or []
    logical_queries = [str(item.get("query") or "") for item in subqueries]
    adapted_queries = list(
        dict.fromkeys(
            entry.request.query or ""
            for entry in current_entries
            if entry.request.query
        )
    )
    term_loss = adapter_term_loss(query.query, adapted_queries, metadata_text)
    existing_retrieved = _existing_candidate_has_gold(result_row, arxiv_id)
    metadata_mismatch = bool(
        metadata
        and dataset_title
        and lexical_overlap(dataset_title, metadata.title)["query_coverage"] < 0.5
    )
    oracle_error = any(entry.status == "failed" for entry in oracle_entries.values())
    current_error = bool(current_entries) and all(
        entry.status == "failed" for entry in current_entries
    )
    adapted_overlap = max(
        (
            lexical_overlap(_strip_arxiv_field_syntax(item), metadata_text)[
                "query_coverage"
            ]
            for item in adapted_queries
        ),
        default=0.0,
    )
    any_oracle = any(
        isinstance(rank, int) and rank <= AUDIT_MAX_RESULTS
        for rank in (exact_rank, normalized_rank, core_rank)
    )
    signals = {
        "source_error": metadata_source_error or (current_error and oracle_error),
        "identifier_available": metadata is not None,
        "metadata_mismatch": metadata_mismatch,
        "current_query_rank": current_rank,
        "adapter_term_loss": term_loss["loss_detected"],
        "lexical_query_coverage": original_overlap["query_coverage"],
        "query_over_restrictive": (
            current_rank is None
            and any_oracle
            and max((len(content_tokens(item)) for item in adapted_queries), default=0)
            >= 6
            and adapted_overlap < 0.4
        ),
        "any_title_oracle_hit": any_oracle,
        "adapted_query_has_and": any(
            " AND " in item.upper() for item in adapted_queries
        ),
        "current_query_failed_request_count": sum(
            entry.status == "failed" for entry in current_entries
        ),
    }
    failure_reason = None if existing_retrieved else classify_failure(signals)
    return {
        "case_id": query.query_id,
        "gold_index": gold_index,
        "query": query.query,
        "arxiv_id": arxiv_id,
        "dataset_title": dataset_title,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "identifier_available": metadata is not None,
        "existing_candidate_retrieved": existing_retrieved,
        "current_query_rank": current_rank,
        "exact_title_rank": exact_rank,
        "normalized_title_rank": normalized_rank,
        "title_core_rank": core_rank,
        "logical_queries": logical_queries,
        "adapted_queries": adapted_queries,
        "logical_query_overlap": [
            {
                "query": item,
                "query_coverage": lexical_overlap(item, metadata_text)[
                    "query_coverage"
                ],
            }
            for item in logical_queries
        ],
        "adapted_query_overlap": [
            {
                "query": item,
                "query_coverage": lexical_overlap(
                    _strip_arxiv_field_syntax(item), metadata_text
                )["query_coverage"],
            }
            for item in adapted_queries
        ],
        "lexical_overlap": original_overlap,
        "facet_coverage": _facet_coverage(planning, metadata_text),
        "adapter_term_loss": term_loss,
        "signals": signals,
        "failure_reason": failure_reason,
    }


def _query_audit_row(
    query: EvalQuery,
    result_row: dict[str, Any],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = Counter(
        str(row["failure_reason"])
        for row in gold_rows
        if row.get("failure_reason")
    )
    return {
        "case_id": query.query_id,
        "query": query.query,
        "query_features": query_features(result_row),
        "gold_count": len(gold_rows),
        "identifier_available_count": sum(
            bool(row["identifier_available"]) for row in gold_rows
        ),
        "existing_retrieved_gold_count": sum(
            bool(row["existing_candidate_retrieved"]) for row in gold_rows
        ),
        "current_top20_count": sum(
            isinstance(row["current_query_rank"], int)
            and row["current_query_rank"] <= 20
            for row in gold_rows
        ),
        "current_top50_count": sum(
            isinstance(row["current_query_rank"], int)
            and row["current_query_rank"] <= 50
            for row in gold_rows
        ),
        "current_top100_count": sum(
            isinstance(row["current_query_rank"], int)
            and row["current_query_rank"] <= 100
            for row in gold_rows
        ),
        "failure_reason_distribution": dict(sorted(reasons.items())),
    }


def _aggregate_audit(
    gold_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    entries: dict[str, AuditSnapshotEntry],
    replay_cost: AuditRuntimeCost,
) -> dict[str, Any]:
    failure_reasons = Counter(
        str(row["failure_reason"])
        for row in gold_rows
        if row.get("failure_reason")
    )
    recorded = AuditRuntimeCost()
    for entry in entries.values():
        diagnostics = entry.diagnostics
        recorded = AuditRuntimeCost(
            snapshot_hits=0,
            snapshot_writes=0,
            execution_request_count=(
                recorded.execution_request_count + diagnostics.request_count
            ),
            execution_retry_count=(
                recorded.execution_retry_count + diagnostics.retry_count
            ),
            execution_error_count=(
                recorded.execution_error_count + diagnostics.error_count
            ),
            execution_network_wait_seconds=(
                recorded.execution_network_wait_seconds
                + diagnostics.rate_limit_wait_seconds
            ),
            execution_latency_seconds=(
                recorded.execution_latency_seconds + entry.recorded_latency_seconds
            ),
        )
    query_type_failures: dict[str, Any] = {}
    for dimension in (
        "topic_structure",
        "method_presence",
        "dataset_presence",
        "must_have_presence",
        "paper_type_presence",
        "query_length_bin",
    ):
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        case_counts: Counter[str] = Counter()
        for row in query_rows:
            label = str(row["query_features"][dimension])
            case_counts[label] += 1
            groups[label].update(row["failure_reason_distribution"])
        query_type_failures[dimension] = {
            label: {
                "case_count": case_counts[label],
                "gold_failure_count": sum(groups[label].values()),
                "failure_reasons": dict(sorted(groups[label].items())),
            }
            for label in sorted(groups)
        }
    ranks = [row["current_query_rank"] for row in gold_rows]
    oracle_ranks = {
        "exact_title": [row["exact_title_rank"] for row in gold_rows],
        "normalized_title": [row["normalized_title_rank"] for row in gold_rows],
        "title_core_terms": [row["title_core_rank"] for row in gold_rows],
    }
    primary = failure_reasons.most_common(1)[0][0] if failure_reasons else "none"
    cause_families = {
        "query_construction": (
            failure_reasons["identifier_available_but_query_not_matched"]
            + failure_reasons["query_over_restrictive"]
            + failure_reasons["adapter_term_loss"]
        ),
        "lexical_mismatch": failure_reasons["lexical_mismatch"],
        "ranking_cutoff": (
            failure_reasons["result_limit_truncation"]
            + failure_reasons["query_over_broad_ranked_below_limit"]
        ),
        "source_availability": (
            failure_reasons["source_error"]
            + failure_reasons["source_unavailable"]
        ),
        "metadata_or_unknown": (
            failure_reasons["metadata_mismatch"] + failure_reasons["unknown"]
        ),
    }
    primary_family = max(
        cause_families,
        key=lambda item: (cause_families[item], item),
    )
    return {
        "protocol": {
            "dataset": "auto_scholar_query",
            "offset": HOLDOUT_OFFSET,
            "limit": HOLDOUT_LIMIT,
            "case_ids": list(HOLDOUT_CASE_IDS),
            "source": "arxiv",
            "oracle_max_results": AUDIT_MAX_RESULTS,
            "gold_post_search_only": True,
            "production_integration": False,
        },
        "gold_count": len(gold_rows),
        "missing_gold_count": sum(
            not row["existing_candidate_retrieved"] for row in gold_rows
        ),
        "existing_retrieved_gold_count": sum(
            bool(row["existing_candidate_retrieved"]) for row in gold_rows
        ),
        "identifier_available_count": sum(
            bool(row["identifier_available"]) for row in gold_rows
        ),
        "identifier_available_rate": (
            sum(bool(row["identifier_available"]) for row in gold_rows)
            / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "exact_title_recovered_count": sum(
            isinstance(row["exact_title_rank"], int) for row in gold_rows
        ),
        "exact_title_recovered_rate": (
            sum(isinstance(row["exact_title_rank"], int) for row in gold_rows)
            / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "normalized_title_recovered_count": sum(
            isinstance(row["normalized_title_rank"], int) for row in gold_rows
        ),
        "normalized_title_recovered_rate": (
            sum(isinstance(row["normalized_title_rank"], int) for row in gold_rows)
            / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "title_core_recovered_count": sum(
            isinstance(row["title_core_rank"], int) for row in gold_rows
        ),
        "title_core_recovered_rate": (
            sum(isinstance(row["title_core_rank"], int) for row in gold_rows)
            / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "current_query_oracle_recall_at_k": {
            str(k): recall_at_k(ranks, k) for k in (20, 50, 100)
        },
        "title_oracle_recall_at_k": {
            mode: {str(k): recall_at_k(mode_ranks, k) for k in (20, 50, 100)}
            for mode, mode_ranks in oracle_ranks.items()
        },
        "failure_reason_distribution": dict(sorted(failure_reasons.items())),
        "cause_family_distribution": cause_families,
        "lexical_mismatch_count": failure_reasons["lexical_mismatch"],
        "adapter_term_loss_count": failure_reasons["adapter_term_loss"],
        "ranked_below_limit_count": (
            failure_reasons["result_limit_truncation"]
            + failure_reasons["query_over_broad_ranked_below_limit"]
        ),
        "source_failure_count": (
            failure_reasons["source_error"]
            + failure_reasons["source_unavailable"]
        ),
        "source_failed_request_count": sum(
            entry.status == "failed" for entry in entries.values()
        ),
        "query_type_failure_distribution": query_type_failures,
        "snapshot_entry_count": len(entries),
        "snapshot_status_counts": dict(
            sorted(Counter(entry.status for entry in entries.values()).items())
        ),
        "recorded_cost": recorded.model_dump(mode="json"),
        "replay_execution_cost": replay_cost.model_dump(mode="json"),
        "primary_bottleneck": primary,
        "primary_bottleneck_family": primary_family,
        "next_experiment_recommendation": _recommendation(failure_reasons),
    }


def _facet_coverage(planning: dict[str, Any], document: str) -> dict[str, Any]:
    facets = ((planning.get("planning") or {}).get("facets") or [])
    document_tokens = set(content_tokens(document))
    output: dict[str, Any] = {}
    for facet in facets:
        facet_type = str(facet.get("facet_type") or "unknown")
        terms = [str(item) for item in facet.get("terms") or []]
        matched = [term for term in terms if set(content_tokens(term)) & document_tokens]
        output[facet_type] = {
            "terms": terms,
            "matched_terms": matched,
            "coverage": len(matched) / len(terms) if terms else 0.0,
        }
    return output


def _existing_candidate_has_gold(row: dict[str, Any], arxiv_id: str) -> bool:
    diagnostics = row.get("stage_diagnostics") or {}
    snapshot = next(
        (
            item
            for item in diagnostics.get("snapshots") or []
            if item.get("stage") == "initial_deduplicated"
        ),
        {},
    )
    return any(
        _normalize_arxiv_id(
            ((candidate.get("identifiers") or {}).get("arxiv_id") or "")
        )
        == arxiv_id
        for candidate in snapshot.get("candidates") or []
    )


def _minimum_rank(
    entries: Iterable[AuditSnapshotEntry | None],
    arxiv_id: str,
) -> int | None:
    ranks = [
        index
        for entry in entries
        if entry is not None and entry.status == "success"
        for index, paper in enumerate(entry.papers, start=1)
        if paper.arxiv_id == arxiv_id
    ]
    return min(ranks) if ranks else None


def _recommendation(reasons: Counter[str]) -> str:
    ranked = reasons["result_limit_truncation"] + reasons[
        "query_over_broad_ranked_below_limit"
    ]
    if ranked and ranked >= max(reasons.values(), default=0):
        return "在同一查询上受控比较 Top-20 与更深候选池，再评估成本和噪声。"
    if reasons["lexical_mismatch"] >= max(reasons.values(), default=0):
        return "在独立快照上比较通用语义/标题摘要扩展，不使用 gold 标题生成生产查询。"
    if reasons["adapter_term_loss"]:
        return "验证通用术语保留检查，确认 source adapter 是否丢失与 gold 元数据共享的查询词。"
    if reasons["source_error"] or reasons["source_unavailable"]:
        return "先复核来源可用性，并用同一请求集比较可复现的多源覆盖。"
    return "在冻结候选上比较通用查询放宽与分面组合，禁止使用单条 gold 信息。"


def _validate_holdout_queries(queries: list[EvalQuery]) -> None:
    if tuple(query.query_id for query in queries) != HOLDOUT_CASE_IDS:
        raise ValueError("retrieval audit requires fixed holdout30 case order")


def _kind_priority(kind: AuditRequestKind) -> int:
    return {
        "metadata_by_id": 0,
        "current_query": 1,
        "exact_title": 2,
        "normalized_title": 3,
        "title_core_terms": 4,
    }[kind]


def _normalize_arxiv_id(value: str) -> str:
    raw = str(value).strip().rstrip("/").split("/")[-1]
    return _ARXIV_VERSION_RE.sub("", raw).casefold()


def _normalize_space(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _element_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


def _parse_year(value: str) -> int | None:
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else None


def _all_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return _TOKEN_RE.findall(normalized)


def _strip_arxiv_field_syntax(value: str) -> str:
    return re.sub(r"\b(?:all|ti|abs|au|cat):", " ", value, flags=re.IGNORECASE)


def _sanitize(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+", r"\1=[redacted]", value)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
