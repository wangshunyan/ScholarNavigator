"""SciFact current-rules query-depth audit with isolated Record/Replay.

Gold metadata is introduced only while replaying recorded query responses.  The
record plan is reconstructed exclusively from the frozen current-rules
diagnostics and contains no case or gold identifiers.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request

from scholar_agent.connectors.arxiv import search_arxiv_detailed
from scholar_agent.connectors.openalex import search_openalex_detailed
from scholar_agent.connectors.pubmed import (
    PUBMED_ESEARCH_URL,
    _efetch_articles,
    _parse_article,
    _request_bytes as _pubmed_request_bytes,
    _with_api_key,
)
from scholar_agent.connectors.semantic_scholar import (
    SEARCH_FIELDS,
    SEMANTIC_SCHOLAR_SEARCH_URL,
    _parse_paper as _parse_semantic_paper,
    _request_json_detailed as _semantic_request_json,
    _semantic_scholar_headers,
)
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.diagnostics_schemas import (
    ConnectorDiagnostics,
    merge_connector_diagnostics,
)
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import identity_evidence
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.datasets.beir_scifact import load_beir_scifact_enriched
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    _rank_pool,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.metrics import canonical_paper_id
from scholar_agent.retrieval.query_adapter import adapt_query_for_source


AUDIT_SCHEMA_VERSION = "scifact-query-depth-v1"
SOURCES = ("openalex", "arxiv", "semantic_scholar", "pubmed")
PREFIXES = (20, 50, 100, 200)
PAGED_SOURCES = {"semantic_scholar", "pubmed"}
PAGE_SIZE = 100
MAX_DEPTH = 200
DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS = 35.0
PageStatus = Literal["success", "failed", "source_outage"]


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_results(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_request_plan(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    max_depth: int = MAX_DEPTH,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild every planned source/adapted-query list in frozen call order."""

    if max_depth != MAX_DEPTH:
        raise ValueError("scifact depth audit requires max_depth=200")
    sources = [str(item) for item in config.get("sources") or []]
    if sources != list(SOURCES):
        raise ValueError("frozen source order mismatch")
    by_case = {str(row["case_id"]): row for row in rows}
    case_ids = [str(item) for item in config.get("case_ids") or []]
    if set(by_case) != set(case_ids):
        raise ValueError("frozen result case set mismatch")

    requests: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        row = by_case[case_id]
        snapshots = row.get("stage_diagnostics", {}).get("snapshots", [])
        retrieval = next(
            (item for item in snapshots if item.get("stage") == "initial_retrieval"),
            None,
        )
        if retrieval is None:
            raise ValueError(f"initial retrieval diagnostics missing:{case_id}")
        lists: list[dict[str, Any]] = []
        for list_order, call in enumerate(retrieval.get("retrieval_calls") or []):
            source = str(call.get("source") or "")
            adapted_query = str(call.get("adapted_query") or "").strip()
            if source not in SOURCES or not adapted_query:
                raise ValueError(f"planned query list invalid:{case_id}:{list_order}")
            page_keys: list[str] = []
            for request in page_requests(source, adapted_query, max_depth=max_depth):
                key = stable_hash(request)
                requests.setdefault(key, {**request, "key": key})
                page_keys.append(key)
            lists.append(
                {
                    "list_order": list_order,
                    "source": source,
                    "origin_subquery": str(call.get("origin_subquery") or ""),
                    "adapted_query": adapted_query,
                    "baseline_logical_call_executed": bool(
                        call.get("logical_call_executed") is True
                    ),
                    "baseline_terminal_status": call.get("terminal_status"),
                    "page_keys": page_keys,
                }
            )
        cases.append(
            {
                "case_order": case_order,
                "case_id": case_id,
                "query": str(row.get("query") or ""),
                "lists": lists,
                "query_analysis": row["stage_diagnostics"][
                    "initial_query_planning"
                ]["query_analysis"],
            }
        )
    return requests, cases


def page_requests(
    source: str, adapted_query: str, *, max_depth: int = MAX_DEPTH
) -> list[dict[str, Any]]:
    if source in PAGED_SOURCES:
        offsets = range(0, max_depth, PAGE_SIZE)
        return [
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "source": source,
                "adapted_query": adapted_query,
                "offset": offset,
                "limit": min(PAGE_SIZE, max_depth - offset),
                "max_depth": max_depth,
            }
            for offset in offsets
        ]
    return [
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "source": source,
            "adapted_query": adapted_query,
            "offset": 0,
            "limit": max_depth,
            "max_depth": max_depth,
        }
    ]


def probe_request(request: dict[str, Any]) -> dict[str, Any]:
    """Build a low-cost, gold-free source probe from a planned query."""

    return {
        **{key: request[key] for key in ("schema_version", "source", "adapted_query")},
        "offset": 0,
        "limit": 1,
        "max_depth": 1,
    }


def fetch_page(request: dict[str, Any]) -> dict[str, Any]:
    """Fetch exactly one deterministic result page using the existing clients."""

    started = time.perf_counter()
    source = str(request["source"])
    query = str(request["adapted_query"])
    offset = int(request["offset"])
    limit = int(request["limit"])
    try:
        if source == "arxiv":
            if offset != 0:
                raise ValueError("arxiv offset is unsupported in this audit")
            result = search_arxiv_detailed(query, limit=limit, max_retries=1)
        elif source == "openalex":
            if offset != 0:
                raise ValueError("openalex offset is unsupported in this audit")
            result = search_openalex_detailed(query, limit=limit, max_retries=1)
        elif source == "semantic_scholar":
            return _fetch_semantic_page(query, offset, limit, started)
        elif source == "pubmed":
            return _fetch_pubmed_page(query, offset, limit, started)
        else:
            raise ValueError("unsupported audit source")
    except Exception as exc:  # noqa: BLE001 - isolate one page without aborting a batch
        return _failed_response(
            error_type=f"connector_exception:{type(exc).__name__}",
            latency_seconds=time.perf_counter() - started,
            request_count=0,
        )
    return _connector_response(result, started)


def _fetch_semantic_page(
    query: str, offset: int, limit: int, started: float
) -> dict[str, Any]:
    adapted = adapt_query_for_source(query, "semantic_scholar").query
    params = {
        "query": adapted,
        "offset": str(offset),
        "limit": str(limit),
        "fields": SEARCH_FIELDS,
    }
    request = Request(
        f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{urlencode(params)}",
        headers=_semantic_scholar_headers(),
    )
    payload, error, warnings, diagnostics = _semantic_request_json(
        request,
        max_retries=1,
    )
    if payload is None:
        return _failed_response_from_diagnostics(error, diagnostics, started)
    records = payload.get("data")
    if not isinstance(records, list):
        return _failed_response(
            error_type="response_schema_or_parse",
            latency_seconds=time.perf_counter() - started,
            request_count=diagnostics.request_count,
            retry_count=diagnostics.retry_count,
            rate_limit_wait_seconds=diagnostics.rate_limit_wait_seconds,
        )
    papers: list[Paper] = []
    parse_errors = 0
    for record in records:
        try:
            paper = _parse_semantic_paper(record) if isinstance(record, dict) else None
        except Exception:  # noqa: BLE001 - one malformed result is not a page failure
            parse_errors += 1
            continue
        if paper is not None:
            papers.append(paper)
    merged = diagnostics.model_copy(
        update={"error_count": diagnostics.error_count + parse_errors}
    )
    return _success_response(papers, warnings, merged, started)


def _fetch_pubmed_page(
    query: str, offset: int, limit: int, started: float
) -> dict[str, Any]:
    adapted = adapt_query_for_source(query, "pubmed").query
    params = _with_api_key(
        {
            "db": "pubmed",
            "term": adapted,
            "retmode": "json",
            "retstart": str(offset),
            "retmax": str(limit),
            "sort": "relevance",
        }
    )
    search_request = Request(
        f"{PUBMED_ESEARCH_URL}?{urlencode(params)}",
        headers={"User-Agent": "ScholarNavigator"},
    )
    payload, error, warnings, search_diagnostics = _pubmed_request_bytes(
        search_request,
        label="PubMed depth audit esearch",
        throttle_sleep=None,
        monotonic=None,
    )
    if payload is None:
        return _failed_response_from_diagnostics(error, search_diagnostics, started)
    try:
        data = json.loads(payload.decode("utf-8"))
        raw_ids = data.get("esearchresult", {}).get("idlist", [])
        if not isinstance(raw_ids, list):
            raise ValueError("idlist is not a list")
        ids = [str(value).strip() for value in raw_ids if str(value).strip()]
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError):
        return _failed_response(
            error_type="response_schema_or_parse",
            latency_seconds=time.perf_counter() - started,
            request_count=search_diagnostics.request_count,
            rate_limit_wait_seconds=search_diagnostics.rate_limit_wait_seconds,
        )
    if not ids:
        return _success_response([], warnings, search_diagnostics, started)
    fetch_payload, fetch_error, fetch_warnings, fetch_diagnostics = _efetch_articles(
        ids,
        throttle_sleep=None,
        monotonic=None,
    )
    diagnostics = merge_connector_diagnostics(
        [search_diagnostics, fetch_diagnostics]
    )
    if fetch_payload is None:
        return _failed_response_from_diagnostics(fetch_error, diagnostics, started)
    try:
        root = ET.fromstring(fetch_payload)
    except ET.ParseError:
        return _failed_response(
            error_type="response_schema_or_parse",
            latency_seconds=time.perf_counter() - started,
            request_count=diagnostics.request_count,
            retry_count=diagnostics.retry_count,
            rate_limit_wait_seconds=diagnostics.rate_limit_wait_seconds,
        )
    papers: list[Paper] = []
    parse_errors = 0
    for article in root.findall(".//PubmedArticle"):
        try:
            paper = _parse_article(article)
        except Exception:  # noqa: BLE001
            parse_errors += 1
            continue
        if paper is not None:
            papers.append(paper)
    merged = diagnostics.model_copy(
        update={"error_count": diagnostics.error_count + parse_errors}
    )
    return _success_response(
        papers,
        [*warnings, *fetch_warnings],
        merged,
        started,
    )


def _connector_response(result: Any, started: float) -> dict[str, Any]:
    if result.error_message:
        return _failed_response_from_diagnostics(
            result.error_message,
            result.diagnostics,
            started,
        )
    return _success_response(
        result.papers,
        result.warnings,
        result.diagnostics,
        started,
    )


def _success_response(
    papers: Sequence[Paper],
    warnings: Sequence[str],
    diagnostics: ConnectorDiagnostics,
    started: float,
) -> dict[str, Any]:
    return {
        "status": "success",
        "requested": True,
        "error_type": None,
        "http_status": 200,
        "warnings": [str(item) for item in warnings],
        "diagnostics": diagnostics.model_dump(mode="json"),
        "latency_seconds": max(
            float(diagnostics.latency_seconds), time.perf_counter() - started
        ),
        "papers": [paper.model_dump(mode="json") for paper in papers],
    }


def _failed_response_from_diagnostics(
    error: str | None,
    diagnostics: ConnectorDiagnostics,
    started: float,
) -> dict[str, Any]:
    status_match = re.search(r"(?:status:?\s*|HTTP Error\s+)(\d{3})", error or "")
    http_status = int(status_match.group(1)) if status_match else None
    if http_status == 429:
        error_type = "http_429_rate_limit"
    elif http_status in {401, 403}:
        error_type = f"http_{http_status}_authorization"
    elif http_status is not None and 500 <= http_status <= 599:
        error_type = "http_5xx"
    elif "timeout" in (error or "").casefold() or "timed out" in (
        error or ""
    ).casefold():
        error_type = "network_timeout"
    elif "parse" in (error or "").casefold() or "json" in (
        error or ""
    ).casefold():
        error_type = "response_schema_or_parse"
    else:
        error_type = "source_failure"
    return _failed_response(
        error_type=error_type,
        latency_seconds=max(
            float(diagnostics.latency_seconds), time.perf_counter() - started
        ),
        request_count=diagnostics.request_count,
        retry_count=diagnostics.retry_count,
        rate_limit_wait_seconds=diagnostics.rate_limit_wait_seconds,
        http_status=http_status,
    )


def _failed_response(
    *,
    error_type: str,
    latency_seconds: float,
    request_count: int,
    retry_count: int = 0,
    rate_limit_wait_seconds: float = 0.0,
    http_status: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "requested": bool(request_count),
        "error_type": error_type,
        "http_status": http_status,
        "warnings": [],
        "diagnostics": {
            "request_count": request_count,
            "retry_count": retry_count,
            "error_count": 1,
            "cache_hit_count": 0,
            "rate_limit_wait_seconds": rate_limit_wait_seconds,
            "latency_seconds": latency_seconds,
        },
        "latency_seconds": latency_seconds,
        "papers": [],
    }


class DepthSnapshotStore:
    """Strict flat store for audit-only page snapshots."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries_dir = self.root / "pages"

    def path(self, key: str) -> Path:
        return self.entries_dir / f"{key}.json"

    def contains(self, request: dict[str, Any]) -> bool:
        return self.path(str(request["key"])).is_file()

    def write(self, request: dict[str, Any], response: dict[str, Any]) -> bool:
        path = self.path(str(request["key"]))
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"request": request, "response": response}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return True

    def read(self, request: dict[str, Any]) -> dict[str, Any]:
        key = str(request["key"])
        path = self.path(key)
        if not path.is_file():
            raise ValueError(f"depth_snapshot_missing:{key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request") != request:
            raise ValueError(f"depth_snapshot_request_mismatch:{key}")
        expected_key = stable_hash({k: v for k, v in request.items() if k != "key"})
        if expected_key != key:
            raise ValueError(f"depth_snapshot_key_mismatch:{key}")
        response = payload.get("response")
        if not isinstance(response, dict) or response.get("status") not in {
            "success",
            "failed",
            "source_outage",
        }:
            raise ValueError(f"depth_snapshot_response_invalid:{key}")
        if response.get("status") == "source_outage":
            evidence = response.get("probe_evidence")
            diagnostics = response.get("diagnostics") or {}
            if (
                not isinstance(evidence, dict)
                or evidence.get("source") != request["source"]
                or response.get("probe_evidence_hash") != stable_hash(evidence)
                or response.get("requested") is not False
                or int(diagnostics.get("request_count") or 0) != 0
            ):
                raise ValueError(f"depth_snapshot_outage_evidence_invalid:{key}")
        return response


def preflight_evidence(source: str, response: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": response.get("status"),
        "error_type": response.get("error_type"),
        "http_status": response.get("http_status"),
        "diagnostics": response.get("diagnostics") or {},
    }


def validate_preflight(evidence: dict[str, Any], source: str) -> None:
    required = {
        "schema_version",
        "source",
        "timestamp",
        "status",
        "error_type",
        "http_status",
        "diagnostics",
    }
    if not required.issubset(evidence) or evidence.get("source") != source:
        raise ValueError(f"depth_preflight_invalid:{source}")


def source_outage_response(
    evidence: dict[str, Any], *, runtime: bool = False
) -> dict[str, Any]:
    return {
        "status": "source_outage",
        "requested": False,
        "error_type": "runtime_rate_limit_outage" if runtime else "preflight_outage",
        "http_status": evidence.get("http_status"),
        "warnings": ["page_not_started_due_to_source_outage"],
        "diagnostics": {
            "request_count": 0,
            "retry_count": 0,
            "error_count": 0,
            "cache_hit_count": 0,
            "rate_limit_wait_seconds": 0.0,
            "latency_seconds": 0.0,
        },
        "latency_seconds": 0.0,
        "papers": [],
        "probe_evidence": evidence,
        "probe_evidence_hash": stable_hash(evidence),
    }


def record_missing(
    requests: Iterable[dict[str, Any]],
    store: DepthSnapshotStore,
    preflights: dict[str, dict[str, Any]],
    *,
    runner: Callable[[dict[str, Any], float], dict[str, Any]],
    throttle: Callable[[str], float],
    wall_timeout_seconds: float,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Record every missing page serially; one failure never aborts later pages."""

    ordered = list(requests)
    counts: Counter[str] = Counter()
    active_sources = [
        source
        for source in SOURCES
        if any(item["source"] == source for item in ordered)
    ]
    for source in active_sources:
        evidence = preflights[source]
        validate_preflight(evidence, source)
        source_requests = [item for item in ordered if item["source"] == source]
        if evidence.get("status") != "success":
            for request in source_requests:
                if store.write(request, source_outage_response(evidence)):
                    counts["source_outage"] += 1
            continue
        streak = 0
        trigger: list[dict[str, Any]] = []
        runtime_evidence: dict[str, Any] | None = None
        for index, request in enumerate(source_requests, 1):
            if store.contains(request):
                counts["existing"] += 1
                continue
            if runtime_evidence is not None:
                if store.write(
                    request,
                    source_outage_response(runtime_evidence, runtime=True),
                ):
                    counts["source_outage"] += 1
                continue
            throttle(source)
            response = runner(request, wall_timeout_seconds)
            store.write(request, response)
            status = str(response.get("status") or "unknown")
            counts[status] += 1
            if progress is not None:
                progress(f"{source} [{index}/{len(source_requests)}] {status}")
            if response.get("error_type") == "http_429_rate_limit":
                streak += 1
                trigger.append(response)
            else:
                streak = 0
                trigger = []
            if streak >= 2:
                attempted = sum(
                    store.contains(item) for item in source_requests
                )
                runtime_evidence = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "source": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "error_type": "http_429_rate_limit",
                    "http_status": 429,
                    "diagnostics": trigger[-1].get("diagnostics") or {},
                    "source_level_trigger": {
                        "threshold_consecutive_http_429": 2,
                        "attempted_page_count": attempted,
                        "not_started_page_count": len(source_requests) - attempted,
                    },
                }
    counts["pending"] = sum(not store.contains(item) for item in ordered)
    return dict(sorted(counts.items()))


def load_oracle_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_results(path)
    if len(rows) != 42:
        raise ValueError("SciFact exact-oracle gold count mismatch")
    return rows


def replay_audit(
    *,
    queries: Sequence[EvalQuery],
    cases: Sequence[dict[str, Any]],
    requests: dict[str, dict[str, Any]],
    store: DepthSnapshotStore,
    oracle_rows: Sequence[dict[str, Any]],
    candidate_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay every page without network or snapshot writes."""

    responses = {key: store.read(request) for key, request in requests.items()}
    query_by_id = {query.query_id: query for query in queries}
    oracle_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        oracle_by_case[str(row["case_id"])].append(dict(row))
    for rows in oracle_by_case.values():
        rows.sort(key=lambda item: int(item["gold_index"]))

    detail: list[dict[str, Any]] = []
    depth_accumulator: dict[int, list[dict[str, Any]]] = defaultdict(list)
    classifications: Counter[str] = Counter()
    target_classifications: Counter[str] = Counter()
    list_status_by_source: dict[str, Counter[str]] = {
        source: Counter() for source in SOURCES
    }
    exact_hit_gold_by_source: dict[str, set[str]] = {
        source: set() for source in SOURCES
    }
    first_list_rank_histogram: Counter[str] = Counter()
    page_status: dict[str, Counter[str]] = {
        source: Counter() for source in SOURCES
    }
    page_status_by_offset: dict[str, dict[str, Counter[str]]] = {
        source: defaultdict(Counter) for source in SOURCES
    }
    unique_page_costs: dict[str, dict[str, float | int]] = {
        source: _empty_costs() for source in SOURCES
    }
    for key, request in requests.items():
        response = responses[key]
        source = str(request["source"])
        page_status[source][str(response["status"])] += 1
        page_status_by_offset[source][str(request["offset"])][
            str(response["status"])
        ] += 1
        _add_response_cost(unique_page_costs[source], response)

    for case in cases:
        case_id = str(case["case_id"])
        query = query_by_id[case_id]
        oracle = oracle_by_case.get(case_id, [])
        eval_gold = [query.gold_papers[int(item["gold_index"])] for item in oracle]
        per_source = _source_completeness(case, responses)
        case_list_statuses = Counter()
        for item in case["lists"]:
            terminal = list_terminal_status(item, responses)
            case_list_statuses[terminal] += 1
            list_status_by_source[str(item["source"])][terminal] += 1
        curves: dict[str, dict[str, Any]] = {}
        full_unbounded: list[Paper] = []
        for prefix in PREFIXES:
            unbounded, candidates, prefix_complete = build_prefix_pool(
                case,
                responses,
                prefix=prefix,
                candidate_limit=candidate_limit,
            )
            if prefix == 200:
                full_unbounded = unbounded
            metrics, formal, _ = _rank_pool(
                QueryAnalysis.model_validate(case["query_analysis"]),
                candidates,
                eval_gold,
            )
            retrieval_hits = _matched_gold_indexes(unbounded, eval_gold)
            candidate_hits = _matched_gold_indexes(candidates, eval_gold)
            returned_hits = _matched_gold_indexes(formal, eval_gold)
            curve = {
                "depth": prefix,
                "all_planned_lists_complete": prefix_complete,
                "retrieval_unique_candidate_count": len(unbounded),
                "retrieval_matched_gold_count": len(retrieval_hits),
                "retrieval_candidate_recall": (
                    len(retrieval_hits) / len(eval_gold) if eval_gold else None
                ),
                "budgeted_candidate_count": len(candidates),
                "candidate_matched_gold_count": len(candidate_hits),
                "candidate_recall": metrics["candidate_recall"],
                "returned_gold_count": len(returned_hits),
                "recall_at_20": metrics["recall_at_20"],
                "f1_at_20": metrics["f1_at_20"],
                "retrieval_gold_indexes": sorted(retrieval_hits),
                "candidate_gold_indexes": sorted(candidate_hits),
                "returned_gold_indexes": sorted(returned_hits),
            }
            curves[str(prefix)] = curve
            depth_accumulator[prefix].append(
                {"case_id": case_id, "gold_count": len(eval_gold), **curve}
            )

        gold_diagnostics: list[dict[str, Any]] = []
        for local_index, oracle_row in enumerate(oracle):
            gold = eval_gold[local_index]
            hit = first_hit_evidence(case, responses, gold)
            category = classify_gold_depth(
                first_rank=hit["first_list_rank"],
                oracle_row=oracle_row,
                source_completeness=per_source,
            )
            classifications[category] += 1
            if hit["first_list_rank"] is not None:
                first_list_rank_histogram[str(hit["first_list_rank"])] += 1
            for source in hit["hit_sources"]:
                exact_hit_gold_by_source[str(source)].add(
                    f"{case_id}:{int(oracle_row['gold_index'])}"
                )
            if oracle_row.get("classification") == "source_exactly_locatable_query_miss":
                target_classifications[category] += 1
            merged_rank = _first_exact_rank(full_unbounded, gold)
            gold_diagnostics.append(
                {
                    "gold_index": int(oracle_row["gold_index"]),
                    "audit_subject_id": oracle_row["audit_subject_id"],
                    "prior_oracle_classification": oracle_row["classification"],
                    "classification": category,
                    "first_hit_depth": _depth_bucket(hit["first_list_rank"]),
                    "first_list_rank": hit["first_list_rank"],
                    "first_merged_rank_at_200": merged_rank,
                    "hit_sources": hit["hit_sources"],
                    "hit_lists": hit["hit_lists"],
                    "exactly_locatable_sources": sorted(
                        source
                        for source, value in oracle_row["sources"].items()
                        if value.get("terminal") == "exact_hit"
                    ),
                }
            )
        increments: dict[str, int] = {}
        previous = 0
        for prefix in PREFIXES:
            current = int(curves[str(prefix)]["retrieval_matched_gold_count"])
            increments[str(prefix)] = current - previous
            previous = current
        detail.append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "case_order": int(case["case_order"]),
                "case_id": case_id,
                "query": case["query"],
                "evaluable_gold_count": len(eval_gold),
                "planned_list_count": len(case["lists"]),
                "list_terminal_status_counts": dict(sorted(case_list_statuses.items())),
                "source_completeness": per_source,
                "depth_curve": curves,
                "retrieval_gold_increment": increments,
                "gold_diagnostics": gold_diagnostics,
            }
        )

    aggregate_curves = {
        str(prefix): _aggregate_depth(depth_accumulator[prefix])
        for prefix in PREFIXES
    }
    recovered = (
        aggregate_curves["200"]["retrieval_matched_gold_count"]
        - aggregate_curves["20"]["retrieval_matched_gold_count"]
    )
    costs = _sum_costs(unique_page_costs.values())
    aggregate = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "snapshot_replay",
        "network_request_count": 0,
        "snapshot_write_count": 0,
        "case_count": len(cases),
        "evaluable_gold_count": len(oracle_rows),
        "planned_list_count": sum(len(case["lists"]) for case in cases),
        "unique_list_count": len(
            {
                (item["source"], item["adapted_query"])
                for case in cases
                for item in case["lists"]
            }
        ),
        "unique_page_key_count": len(requests),
        "depth_curve": aggregate_curves,
        "classification_counts": dict(sorted(classifications.items())),
        "indexed_query_miss_37_classification_counts": dict(
            sorted(target_classifications.items())
        ),
        "first_hit_depth_distribution": {
            key: classifications.get(key, 0)
            for key in (
                "current_depth_hit",
                "depth_21_50_hit",
                "depth_51_100_hit",
                "depth_101_200_hit",
                "depth_200_miss",
                "source_unavailable_or_incomplete",
                "identity_match_uncertain",
            )
        },
        "first_list_rank_histogram": dict(
            sorted(first_list_rank_histogram.items(), key=lambda item: int(item[0]))
        ),
        "page_status_by_source": {
            source: dict(sorted(values.items()))
            for source, values in page_status.items()
        },
        "page_status_by_source_offset": {
            source: {
                offset: dict(sorted(values.items()))
                for offset, values in sorted(
                    offsets.items(), key=lambda item: int(item[0])
                )
            }
            for source, offsets in page_status_by_offset.items()
        },
        "list_status_by_source": {
            source: dict(sorted(values.items()))
            for source, values in list_status_by_source.items()
        },
        "list_status_total": dict(
            sorted(
                sum((values for values in list_status_by_source.values()), Counter()).items()
            )
        ),
        "exact_query_hit_gold_count_by_source": {
            source: len(values) for source, values in exact_hit_gold_by_source.items()
        },
        "recorded_cost_by_source": unique_page_costs,
        "recorded_cost_total": costs,
        "additional_retrieval_gold_20_to_200": recovered,
        "cost_per_additional_gold_20_to_200": (
            {
                "http_requests": costs["request_count"] / recovered,
                "latency_seconds": costs["latency_seconds"] / recovered,
                "retry_count": costs["retry_count"] / recovered,
            }
            if recovered > 0
            else None
        ),
    }
    return detail, aggregate


def build_prefix_pool(
    case: dict[str, Any],
    responses: dict[str, dict[str, Any]],
    *,
    prefix: int,
    candidate_limit: int,
) -> tuple[list[Paper], list[Paper], bool]:
    """Take each list's same-response prefix, then mirror frozen merge order."""

    by_subquery: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subquery_order: list[str] = []
    for item in case["lists"]:
        origin = str(item["origin_subquery"])
        if origin not in by_subquery:
            subquery_order.append(origin)
        by_subquery[origin].append(item)
    outputs: list[Paper] = []
    all_complete = True
    for origin in subquery_order:
        raw: list[Paper] = []
        for item in by_subquery[origin]:
            papers, complete = list_prefix(item, responses, prefix)
            raw.extend(papers)
            all_complete = all_complete and complete
        outputs.extend(deduplicate_papers(raw))
    unbounded = deduplicate_papers(outputs)
    candidates = list(unbounded)
    if len(candidates) > candidate_limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=candidate_limit,
            source_order=list(SOURCES),
        )
    return unbounded, candidates, all_complete


def list_prefix(
    item: dict[str, Any],
    responses: dict[str, dict[str, Any]],
    prefix: int,
) -> tuple[list[Paper], bool]:
    papers: list[Paper] = []
    remaining = prefix
    complete = True
    for key in item["page_keys"]:
        response = responses[str(key)]
        if response.get("status") != "success":
            complete = False
            break
        page = [Paper.model_validate(value) for value in response.get("papers") or []]
        take = min(remaining, len(page))
        papers.extend(page[:take])
        remaining -= take
        if len(page) < int(response.get("page_limit") or _request_limit(key, item)):
            remaining = 0
        if remaining <= 0:
            break
    if remaining > 0:
        complete = False
    return papers, complete


def list_terminal_status(
    item: dict[str, Any], responses: dict[str, dict[str, Any]]
) -> str:
    statuses = [str(responses[str(key)]["status"]) for key in item["page_keys"]]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    if statuses and all(status == "source_outage" for status in statuses):
        return "source_outage"
    if statuses:
        return "incomplete"
    return "unsupported"


def _request_limit(key: str, item: dict[str, Any]) -> int:
    # Requests are fixed at 100 for paged sources and 200 otherwise. This helper
    # avoids embedding request payloads into per-case plans.
    del key
    return PAGE_SIZE if item["source"] in PAGED_SOURCES else MAX_DEPTH


def _source_completeness(
    case: dict[str, Any], responses: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        lists = [item for item in case["lists"] if item["source"] == source]
        statuses = [
            str(responses[key]["status"])
            for item in lists
            for key in item["page_keys"]
        ]
        result[source] = {
            "planned_list_count": len(lists),
            "planned_page_count": len(statuses),
            "page_status_counts": dict(sorted(Counter(statuses).items())),
            "complete_at_200": bool(statuses)
            and all(status == "success" for status in statuses),
            "unavailable": bool(statuses)
            and all(status == "source_outage" for status in statuses),
        }
    return result


def first_hit_evidence(
    case: dict[str, Any],
    responses: dict[str, dict[str, Any]],
    gold: EvalGoldPaper,
) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for item in case["lists"]:
        papers, _ = list_prefix(item, responses, MAX_DEPTH)
        rank = _first_exact_rank(papers, gold)
        if rank is None:
            continue
        hits.append(
            {
                "source": item["source"],
                "adapted_query": item["adapted_query"],
                "list_order": item["list_order"],
                "rank": rank,
            }
        )
    hits.sort(key=lambda item: (int(item["rank"]), int(item["list_order"])))
    return {
        "first_list_rank": int(hits[0]["rank"]) if hits else None,
        "hit_sources": sorted({str(item["source"]) for item in hits}),
        "hit_lists": hits,
    }


def classify_gold_depth(
    *,
    first_rank: int | None,
    oracle_row: dict[str, Any],
    source_completeness: dict[str, dict[str, Any]],
) -> str:
    bucket = _depth_bucket(first_rank)
    if bucket is not None:
        return bucket
    if oracle_row.get("classification") == "identity_evidence_insufficient":
        return "identity_match_uncertain"
    indexed_sources = [
        source
        for source, value in oracle_row.get("sources", {}).items()
        if value.get("terminal") == "exact_hit"
    ]
    if not indexed_sources:
        return "identity_match_uncertain"
    if any(
        source_completeness.get(source, {}).get("complete_at_200")
        for source in indexed_sources
    ):
        return "depth_200_miss"
    return "source_unavailable_or_incomplete"


def _depth_bucket(rank: int | None) -> str | None:
    if rank is None:
        return None
    if rank <= 20:
        return "current_depth_hit"
    if rank <= 50:
        return "depth_21_50_hit"
    if rank <= 100:
        return "depth_51_100_hit"
    if rank <= 200:
        return "depth_101_200_hit"
    return None


def _first_exact_rank(
    papers: Sequence[Any], gold: EvalGoldPaper
) -> int | None:
    for rank, paper in enumerate(papers, 1):
        evidence = identity_evidence(paper, gold)
        if evidence.equivalent and evidence.shared_identifiers:
            return rank
    return None


def _matched_gold_indexes(
    papers: Sequence[Any], gold: Sequence[EvalGoldPaper]
) -> set[int]:
    matched: set[int] = set()
    for paper in papers:
        for index, expected in enumerate(gold):
            if index in matched:
                continue
            evidence = identity_evidence(paper, expected)
            if evidence.equivalent and evidence.shared_identifiers:
                matched.add(index)
                break
    return matched


def _aggregate_depth(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold_denominator = sum(int(row["gold_count"]) for row in rows)
    retrieval_ids = {
        f"{row['case_id']}:{index}"
        for row in rows
        for index in row["retrieval_gold_indexes"]
    }
    candidate_ids = {
        f"{row['case_id']}:{index}"
        for row in rows
        for index in row["candidate_gold_indexes"]
    }
    returned_ids = {
        f"{row['case_id']}:{index}"
        for row in rows
        for index in row["returned_gold_indexes"]
    }
    evaluable = [row for row in rows if int(row["gold_count"]) > 0]
    return {
        "case_count": len(rows),
        "complete_case_count": sum(
            bool(row["all_planned_lists_complete"]) for row in rows
        ),
        "coverage_is_observed_lower_bound": any(
            not row["all_planned_lists_complete"] for row in rows
        ),
        "gold_denominator": gold_denominator,
        "retrieval_matched_gold_count": len(retrieval_ids),
        "retrieval_candidate_recall": (
            len(retrieval_ids) / gold_denominator if gold_denominator else None
        ),
        "candidate_matched_gold_count": len(candidate_ids),
        "candidate_recall": (
            len(candidate_ids) / gold_denominator if gold_denominator else None
        ),
        "returned_gold_count": len(returned_ids),
        "macro_recall_at_20": _mean(
            float(row["recall_at_20"]) for row in evaluable
        ),
        "macro_f1_at_20": _mean(float(row["f1_at_20"]) for row in evaluable),
        "average_retrieval_unique_candidate_count": _mean(
            float(row["retrieval_unique_candidate_count"]) for row in rows
        ),
        "average_budgeted_candidate_count": _mean(
            float(row["budgeted_candidate_count"]) for row in rows
        ),
    }


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _empty_costs() -> dict[str, float | int]:
    return {
        "page_snapshot_count": 0,
        "request_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "cache_hit_count": 0,
        "latency_seconds": 0.0,
        "rate_limit_wait_seconds": 0.0,
    }


def _add_response_cost(costs: dict[str, float | int], response: dict[str, Any]) -> None:
    diagnostics = response.get("diagnostics") or {}
    costs["page_snapshot_count"] += 1
    for field in (
        "request_count",
        "retry_count",
        "error_count",
        "cache_hit_count",
    ):
        costs[field] += int(diagnostics.get(field) or 0)
    costs["latency_seconds"] += float(response.get("latency_seconds") or 0.0)
    costs["rate_limit_wait_seconds"] += float(
        diagnostics.get("rate_limit_wait_seconds") or 0.0
    )


def _sum_costs(values: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    result = _empty_costs()
    for value in values:
        for key in result:
            result[key] += value[key]
    return result


def build_config(
    *,
    dataset_path: str | Path,
    sample_manifest_path: str | Path,
    crosswalk_path: str | Path,
    baseline_run_dir: str | Path,
    oracle_dir: str | Path,
    request_count: int,
    list_count: int,
    candidate_limit: int,
    request_wall_timeout_seconds: float,
) -> dict[str, Any]:
    baseline = Path(baseline_run_dir)
    oracle = Path(oracle_dir)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": "beir_scifact",
        "split": "test",
        "sample_query_count": 50,
        "max_depth": MAX_DEPTH,
        "prefixes": list(PREFIXES),
        "sources": list(SOURCES),
        "candidate_limit": candidate_limit,
        "top_k": 20,
        "query_source": "frozen_current_rules_retrieval_calls",
        "gold_access_during_record": False,
        "pagination": {
            "arxiv": "single max_results=200",
            "openalex": "single per-page=200",
            "semantic_scholar": "offset pages 0/100, limit=100",
            "pubmed": "retstart pages 0/100, retmax=100",
        },
        "request_policy": {
            "serial": True,
            "request_wall_timeout_seconds": request_wall_timeout_seconds,
            "max_retries": 1,
            "consecutive_429_outage_threshold": 2,
        },
        "planned_list_count": list_count,
        "unique_page_key_count": request_count,
        "inputs": {
            "dataset_sha256": file_sha256(dataset_path),
            "sample_manifest_sha256": file_sha256(sample_manifest_path),
            "crosswalk_sha256": file_sha256(crosswalk_path),
            "baseline_results_sha256": file_sha256(baseline / "results.jsonl"),
            "baseline_config_sha256": file_sha256(baseline / "config.json"),
            "oracle_gold_audit_sha256": file_sha256(oracle / "gold_audit.jsonl"),
            "oracle_aggregate_sha256": file_sha256(oracle / "aggregate.json"),
        },
    }


def write_replay_artifacts(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "config.json", config)
    (root / "gold_depth_audit.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    _write_json(root / "aggregate.json", aggregate)
    hashes = {
        name: file_sha256(root / name)
        for name in ("config.json", "gold_depth_audit.jsonl", "aggregate.json")
    }
    _write_json(root / "artifact_hashes.json", hashes)
    return hashes


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def candidate_display_ids(papers: Sequence[Paper]) -> list[str | None]:
    """Small deterministic helper used by tests and audit inspection."""

    return [canonical_paper_id(paper) for paper in papers]
