#!/usr/bin/env python3
"""四源 Gold 可检索性 oracle 审计（与生产检索路径隔离）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import sys
import time
from datetime import datetime, timezone
from queue import Empty
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scholar_agent.connectors.arxiv import (  # noqa: E402
    _throttle_arxiv_request,
    search_arxiv_detailed,
)
from scholar_agent.connectors.openalex import search_openalex_detailed  # noqa: E402
from scholar_agent.connectors.pubmed import (  # noqa: E402
    _throttle_pubmed_request,
    search_pubmed_detailed,
)
from scholar_agent.connectors.semantic_scholar import (  # noqa: E402
    _throttle_semantic_scholar_request,
    search_semantic_scholar_detailed,
)
from scholar_agent.core.dedup import normalize_title  # noqa: E402
from scholar_agent.evaluation.datasets import load_dataset  # noqa: E402

SOURCES = ("arxiv", "openalex", "semantic_scholar", "pubmed")
# Exact-title retrieval is executed at the fixed depth; identifier lookup is
# kept as a separate oracle request.  Normalized title matching is performed
# against the returned records, not by issuing a result-dependent query.
KINDS = ("identifier", "exact_title", "fixed_depth_title")
DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS = 30.0
SOURCE_ID_FIELDS = {
    "arxiv": ("arxiv_id",),
    "openalex": ("openalex_id", "doi", "arxiv_id"),
    "semantic_scholar": ("semantic_scholar_id", "doi", "arxiv_id"),
    "pubmed": ("pubmed_id", "doi"),
}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _title_tokens(value: str) -> set[str]:
    return set(normalize_title(value).split())


def _paper_dict(paper: Any) -> dict[str, Any]:
    return paper.model_dump(mode="json")


def _normalize_identifier(field: str, value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:", "", text).rstrip("/ ")
    if field in {"arxiv_id", "pubmed_id", "openalex_id", "semantic_scholar_id"}:
        text = text.rsplit("/", 1)[-1]
    if field == "arxiv_id":
        text = re.sub(r"^(arxiv:|abs/|pdf/)", "", text)
        text = re.sub(r"v\d+$", "", text)
    if field == "pubmed_id":
        text = re.sub(r"^pmid:", "", text)
        text = re.sub(r"\D", "", text)
    return text


def _identifier_aliases(identifiers: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for field, value in identifiers.items():
        normalized = _normalize_identifier(field, value)
        if normalized:
            aliases.add(f"{field}:{normalized}")
            if field == "doi" and normalized.startswith("10.48550/arxiv."):
                aliases.add("arxiv_id:" + normalized.rsplit("arxiv.", 1)[-1])
    return aliases


def _fetch(source: str, query: str | None, limit: int) -> dict[str, Any]:
    if not query:
        return {
            "status": "unavailable",
            "error_message": None,
            "warnings": ["no_source_appropriate_identifier"],
            "diagnostics": {"request_count": 0, "retry_count": 0, "error_count": 0},
            "latency_seconds": 0.0,
            "papers": [],
        }
    started = time.perf_counter()
    try:
        if source == "arxiv":
            result = search_arxiv_detailed(query, limit=limit, max_retries=1)
        elif source == "openalex":
            result = search_openalex_detailed(query, limit=limit, max_retries=1)
        elif source == "semantic_scholar":
            result = search_semantic_scholar_detailed(query, limit=limit, max_retries=1)
        else:
            result = search_pubmed_detailed(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - preserve external failure evidence
        return {
            "status": "failed",
            "error_message": f"audit_connector_exception:{type(exc).__name__}",
            "error_type": type(exc).__name__,
            "failure_layer": "connector_exception",
            "warnings": [],
            "diagnostics": {"request_count": 0, "retry_count": 0, "error_count": 1},
            "latency_seconds": time.perf_counter() - started,
            "papers": [],
        }
    error_message = result.error_message
    failure_layer = None
    if error_message:
        lowered = error_message.lower()
        if "429" in lowered:
            failure_layer = "http_429_rate_limit"
        elif "http error" in lowered or "non-2xx" in lowered or "status" in lowered:
            failure_layer = "http_error"
        elif "timeout" in lowered or "timed out" in lowered or "urlerror" in lowered:
            failure_layer = "network_error"
        elif "json" in lowered or "response missing" in lowered or "parse" in lowered:
            failure_layer = "response_schema_or_parse"
        else:
            failure_layer = "connector_error"
    return {
        "status": "failed" if error_message else "success",
        "error_message": error_message,
        "error_type": failure_layer,
        "failure_layer": failure_layer,
        "warnings": result.warnings,
        "diagnostics": result.diagnostics.model_dump(mode="json"),
        "latency_seconds": max(result.latency_seconds, time.perf_counter() - started),
        "papers": [_paper_dict(paper) for paper in result.papers],
    }


def _probe_evidence(response: dict[str, Any], source: str) -> dict[str, Any]:
    evidence = {
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": response["status"],
        "error_message": response.get("error_message"),
        "error_type": (response.get("error_message") or "").split(":", 1)[0] or None,
        "diagnostics": response.get("diagnostics") or {},
    }
    return evidence


def _validate_probe_evidence(raw: str | dict[str, Any], source: str) -> dict[str, Any]:
    evidence = json.loads(raw) if isinstance(raw, str) else raw
    required = {"source", "timestamp", "status", "error_message", "error_type", "diagnostics"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("probe_evidence_schema_invalid")
    if evidence["source"] != source:
        raise ValueError("probe_evidence_source_mismatch")
    return evidence


def _source_outage_response(source: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "source_outage",
        "error_message": "source_level_probe_outage",
        "warnings": ["no_per_gold_request_sent"],
        "diagnostics": {"request_count": 0, "retry_count": 0, "error_count": 0},
        "latency_seconds": 0.0,
        "papers": [],
        "requested": False,
        "source": source,
        "probe_evidence": evidence,
        "probe_evidence_hash": _hash(evidence),
    }


def _is_failed_request_status(status: str) -> bool:
    return status in {"failed", "source_outage", "request_aborted"}


def _request_aborted_response(source: str, evidence: dict[str, Any], *, requested: bool) -> dict[str, Any]:
    return {
        "status": "request_aborted",
        "error_message": "audit_request_aborted_after_stall",
        "warnings": ["attempted_then_terminated_after_stall" if requested else "no_additional_request_sent"],
        "diagnostics": {"request_count": 1 if requested else 0, "retry_count": 0, "error_count": 1 if requested else 0},
        "latency_seconds": float(evidence.get("stall_threshold_seconds", 0)) if requested else 0.0,
        "papers": [],
        "requested": requested,
        "source": source,
        "abort_evidence": evidence,
        "abort_evidence_hash": _hash(evidence),
    }


def _exception_response(exc: BaseException) -> dict[str, Any]:
    """Convert an unexpected worker failure into an auditable terminal response."""

    return {
        "status": "failed",
        "error_message": f"audit_worker_exception:{type(exc).__name__}",
        "error_type": type(exc).__name__,
        "failure_layer": "worker_exception",
        "warnings": [],
        "diagnostics": {"request_count": 0, "retry_count": 0, "error_count": 1},
        "latency_seconds": 0.0,
        "papers": [],
    }


def _process_entry(function: Any, args: tuple[Any, ...], result_queue: Any) -> None:
    try:
        result_queue.put({"ok": True, "value": function(*args)})
    except BaseException as exc:  # noqa: BLE001 - return child failure to parent
        result_queue.put({"ok": False, "value": _exception_response(exc)})


def _run_isolated(function: Any, args: tuple[Any, ...], timeout: float) -> Any:
    """Run one connector request in a killable process with a wall-clock bound."""

    # Fixed spawn avoids inheriting connector throttle globals, locks, or I/O
    # handles from the parent on Linux where the default is fork.
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_process_entry, args=(function, args, result_queue))
    process.start()
    try:
        # Read before join: joining first can deadlock when the child feeder
        # is still flushing a large successful response into this queue.
        try:
            result = result_queue.get(timeout=timeout)
        except Empty:
            if process.is_alive():
                process.terminate()
                process.join(2)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(2)
            return {
                "status": "failed",
                "error_message": "audit_request_wall_clock_timeout",
                "error_type": "TimeoutError",
                "failure_layer": "audit_wall_clock_timeout",
                "warnings": ["attempted_then_terminated_after_wall_clock_timeout"],
                "diagnostics": {"request_count": 1, "retry_count": 0, "error_count": 1},
                "latency_seconds": timeout,
                "papers": [],
                "requested": True,
            }
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(2)
        return result["value"]
    except Exception as exc:  # noqa: BLE001 - child exited without a response
        return _exception_response(exc)
    finally:
        result_queue.close()
        result_queue.join_thread()


def _fetch_isolated(request: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _run_isolated(_fetch, (request["source"], request["query"], request["limit"]), timeout)


def _parent_throttle(
    source: str,
    *,
    sleep: Any | None = None,
    monotonic: Any | None = None,
) -> float:
    """Preserve connector rate limits across the per-request child processes."""

    throttles = {
        "arxiv": _throttle_arxiv_request,
        "semantic_scholar": _throttle_semantic_scholar_request,
        "pubmed": _throttle_pubmed_request,
    }
    throttle = throttles.get(source)
    return throttle(sleep=sleep, monotonic=monotonic) if throttle else 0.0


def _should_record_missing(response: dict[str, Any] | None) -> bool:
    """Retry only per-request terminal failures; preserve source-level outage evidence."""

    if not response:
        return True
    return response.get("status") == "request_aborted"


def _request(source: str, kind: str, gold: dict[str, Any], limit: int) -> dict[str, Any]:
    identifier_field = next(
        (field for field in SOURCE_ID_FIELDS[source] if gold.get(field)), None
    )
    identifier = gold.get(identifier_field) if identifier_field else None
    title = " ".join(str(gold.get("title") or "").split())
    if kind == "identifier":
        query = identifier
        capability = "search_only"
        query_limit = 1
    elif kind == "exact_title":
        query = title
        capability = "search_only"
        query_limit = 1
    else:
        query = title
        capability = "search_only"
        query_limit = limit
    request = {
        "key": _hash({"source": source, "kind": kind, "query": query, "limit": query_limit, "identifier_field": identifier_field, "capability": capability if query else "unsupported"}),
        "source": source,
        "kind": kind,
        "query": query or None,
        "limit": query_limit,
        "identifier_field": identifier_field,
        "available": bool(query),
        "capability": capability if query else "unsupported",
        "gold_id": identifier or "" if query else "",
        "gold_title": title if query else "",
    }
    return request


def _matches(paper: dict[str, Any], gold: dict[str, Any]) -> tuple[bool, bool]:
    stable = bool(_identifier_aliases(paper.get("identifiers") or {}) & _identifier_aliases(gold))
    title_hit = normalize_title(str(paper.get("title") or "")) == normalize_title(str(gold.get("title") or ""))
    return stable, title_hit


def classify_source(*, oracle: dict[str, Any], current: dict[str, Any]) -> str:
    """互斥分类；只依据快照中的可观测信号。"""
    if not oracle.get("title_applicable", False):
        return "source_unavailable"
    if oracle.get("fixed_depth_outage"):
        return "external_failure"
    if current.get("candidate_identity"):
        if not current.get("returned"):
            return "entered_candidate_filtered_or_truncated"
        return "entered_candidate"
    if current.get("title_present_unmatched"):
        return "identity_normalization_or_dedup_miss"
    if oracle.get("fixed_depth_hit"):
        return "indexed_but_normalized_query_miss"
    if oracle.get("identifier_hit") and oracle.get("fixed_depth_complete"):
        return "indexed_but_normalized_query_miss"
    if oracle.get("identifier_hit") or oracle.get("exact_title_hit"):
        return "indexed_but_normalized_query_miss"
    if oracle.get("direct_identifier_not_found"):
        return "source_not_indexed"
    if oracle.get("fixed_depth_failed"):
        return "external_failure"
    if oracle.get("fixed_depth_complete"):
        return "fixed_depth_title_miss"
    return "inconclusive"


def potential_coverage_upper_bound(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """只统计 fixed-depth 成功命中且当前未进入该源候选的确定上界。"""
    result: dict[str, dict[str, int]] = {}
    for record in records:
        for source, data in record["sources"].items():
            bucket = result.setdefault(source, {"gold_denominator": 0, "evaluable_gold_denominator": 0, "outage_gold_count": 0, "external_failure_gold_count": 0, "definite_oracle_hits": 0, "current_missing": 0, "potential_new_gold_count": 0})
            bucket["gold_denominator"] += 1
            outage = bool(data["oracle"].get("fixed_depth_outage"))
            external = bool(
                data["oracle"].get("fixed_depth_failed")
                or data["oracle"].get("fixed_depth_outage")
                or not data["oracle"].get("fixed_depth_complete", True)
            )
            bucket["outage_gold_count"] += int(outage)
            bucket["external_failure_gold_count"] += int(external)
            bucket["evaluable_gold_denominator"] += int(not external)
            definite = bool(data["oracle"].get("fixed_depth_hit"))
            missing = not bool(data["current"].get("candidate_identity"))
            bucket["definite_oracle_hits"] += int(definite)
            bucket["current_missing"] += int(missing)
            if not external:
                bucket["potential_new_gold_count"] += int(definite and missing)
    for bucket in result.values():
        if bucket["evaluable_gold_denominator"] == 0:
            bucket["potential_new_gold_count"] = None
    return result


def _load_current(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    return rows


def _validate_snapshot_payload(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    key = expected["key"]
    if payload.get("request") != expected:
        raise ValueError(f"snapshot_request_mismatch:{key}")
    expected_key = _hash({
        "source": expected["source"],
        "kind": expected["kind"],
        "query": expected["query"],
        "limit": expected["limit"],
        "identifier_field": expected.get("identifier_field"),
        "capability": expected.get("capability"),
    })
    if expected_key != key:
        raise ValueError(f"snapshot_key_mismatch:{key}")


def _current_signal(row: dict[str, Any], gold: dict[str, Any], source: str) -> dict[str, Any]:
    diagnostics = row.get("stage_diagnostics") or {}
    initial = next((s for s in diagnostics.get("snapshots", []) if s.get("stage") == "initial_retrieval"), {})
    ranked = next((s for s in diagnostics.get("snapshots", []) if s.get("stage") == "initial_reranked"), {})
    final = next((s for s in diagnostics.get("snapshots", []) if s.get("stage") == "final_ranked"), {})
    returned = next((s for s in diagnostics.get("snapshots", []) if s.get("stage") == "final_returned"), {})
    initial_candidates = initial.get("candidates") or []

    def source_signals(candidates: list[dict[str, Any]]) -> list[tuple[dict[str, Any], bool, bool]]:
        return [
            (candidate, *_matches(candidate, gold))
            for candidate in candidates
            if source in (candidate.get("sources") or [])
        ]

    initial_signals = source_signals(initial_candidates)
    ranked_signals = source_signals(ranked.get("candidates") or [])
    final_signals = source_signals(final.get("candidates") or [])
    returned_signals = source_signals(returned.get("candidates") or [])
    identity_candidate = next((p for p, stable, _ in initial_signals if stable), None)
    title_candidate = next((p for p, _, title in initial_signals if title), None)
    ranked_candidate = next((p for p, stable, _ in ranked_signals if stable), None)
    final_candidate = next((p for p, stable, _ in final_signals if stable), None)
    returned_candidate = next((p for p, stable, _ in returned_signals if stable), None)
    category = (final_candidate or {}).get("category")
    drop_reason = None if returned_candidate else (
        "judged_weakly_relevant" if category == "weakly_relevant" else
        "outside_final_top_k" if final_candidate and (final_candidate.get("rank") or 0) > 20 else None
    )
    return {
        "returned": returned_candidate is not None,
        "candidate_identity": identity_candidate is not None,
        "title_present_unmatched": title_candidate is not None and identity_candidate is None,
        "drop_reason": drop_reason,
        "initial_rank": (ranked_candidate or {}).get("rank"),
        "final_rank": (final_candidate or {}).get("rank"),
    }


def build_request_plan(
    queries: list[Any],
    current: dict[str, dict[str, Any]],
    sources: tuple[str, ...],
    depth: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    requests: dict[str, dict[str, Any]] = {}
    gold_rows: list[dict[str, Any]] = []
    for query in queries:
        query_number = int(query.query_id.rsplit("_", 1)[-1])
        split = "dev" if query_number < 10 else "val"
        row = current[split][query.query_id]
        for gold in query.gold_papers:
            gold_data = gold.model_dump(mode="json")
            per_source = {}
            for source in sources:
                entries = {}
                for kind in KINDS:
                    req = _request(source, kind, gold_data, depth)
                    requests[req["key"]] = req
                    entries[kind] = req["key"]
                per_source[source] = entries
            gold_rows.append({"case_id": query.query_id, "query": query.query, "gold": gold_data, "requests": per_source, "current": row})
    return requests, gold_rows


def _run(args: argparse.Namespace) -> int:
    if args.workers != 1:
        raise SystemExit("audit_record_workers_must_be_1_for_process_isolation")
    out = Path(args.output).resolve()
    snap = Path(args.snapshot).resolve()
    out.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    queries = load_dataset("auto_scholar_query")[args.offset : args.offset + args.limit]
    current = {
        split: _load_current(Path(args.current_root) / f"source_ablation_d889_{split}_all_replay")
        for split in ("dev", "val")
    }
    sources_to_run = (args.source,) if args.source else SOURCES
    requests, gold_rows = build_request_plan(queries, current, sources_to_run, args.depth)
    if args.probe:
        source = args.source or SOURCES[0]
        probe = next((req for req in requests.values() if req["source"] == source and req["kind"] == "exact_title" and req["available"]), None)
        if probe is None:
            print(json.dumps({"source": source, "status": "unavailable"}))
        else:
            _parent_throttle(probe["source"])
            response = _fetch_isolated(probe, args.request_timeout)
            print(json.dumps({"source": source, "kind": probe["kind"], "probe_evidence": _probe_evidence(response, source)}, ensure_ascii=False))
        return 0
    if args.mode == "source-outage":
        if not args.source or not args.probe_evidence:
            raise SystemExit("source-outage-requires-source-and-probe-evidence")
        try:
            evidence = _validate_probe_evidence(args.probe_evidence, args.source)
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        for req in requests.values():
            response = _source_outage_response(args.source, evidence)
            (snap / f"{req['key']}.json").write_text(json.dumps({"request": req, "response": response}, ensure_ascii=False, indent=2))
        print(json.dumps({"source": args.source, "status": "source_outage", "request_count": 0, "affected_request_count": len(requests)}, ensure_ascii=False))
        return 0
    if args.mode == "request-abort":
        if not args.source or not args.abort_evidence:
            raise SystemExit("request-abort-requires-source-and-abort-evidence")
        evidence = json.loads(args.abort_evidence)
        required = {"source", "observed_stalled_request_key", "observed_stalled_kind", "observed_stalled_query_hash", "stall_threshold_seconds", "termination_reason"}
        if evidence.get("source") != args.source or not required.issubset(evidence):
            raise SystemExit("abort_evidence_source_mismatch")
        evidence["evidence_recorded_at"] = datetime.now(timezone.utc).isoformat()
        pending = []
        for req in requests.values():
            path = snap / f"{req['key']}.json"
            if not path.exists():
                pending.append(req)
            elif json.loads(path.read_text()).get("response", {}).get("status") == "request_aborted":
                pending.append(req)
        for req in pending:
            requested = req["key"] == evidence.get("observed_stalled_request_key")
            response = _request_aborted_response(args.source, evidence, requested=requested)
            (snap / f"{req['key']}.json").write_text(json.dumps({"request": req, "response": response}, ensure_ascii=False, indent=2))
        print(json.dumps({"source": args.source, "status": "request_aborted", "affected_request_count": len(pending)}, ensure_ascii=False))
        return 0
    if args.mode == "record-missing":
        pending = []
        for req in requests.values():
            path = snap / f"{req['key']}.json"
            if not path.exists():
                pending.append(req)
                continue
            try:
                existing = json.loads(path.read_text())
                if _should_record_missing(existing.get("response")):
                    pending.append(req)
            except (OSError, json.JSONDecodeError):
                pending.append(req)
        for index, req in enumerate(pending, 1):
            _parent_throttle(req["source"])
            response = _fetch_isolated(req, args.request_timeout)
            path = snap / f"{req['key']}.json"
            path.write_text(json.dumps({"request": req, "response": response}, ensure_ascii=False, indent=2))
            print(f"[{index}/{len(pending)}] {req['source']} {req['kind']} {response['status']}", flush=True)
        return 0
    if args.mode == "replay":
        if any(not (snap / f"{key}.json").exists() for key in requests):
            raise SystemExit("snapshot_missing")
        records = []
        for item in gold_rows:
            per_source = {}
            gold = item["gold"]
            for source, kinds in item["requests"].items():
                oracle = {
                    "hit": False,
                    "identifier_applicable": False,
                    "title_applicable": False,
                    "failed_count": 0,
                    "successful_count": 0,
                    "matches": {},
                    "depth": args.depth,
                }
                for kind, key in kinds.items():
                    payload = json.loads((snap / f"{key}.json").read_text())
                    expected = requests[key]
                    try:
                        _validate_snapshot_payload(payload, expected)
                    except ValueError as exc:
                        raise SystemExit(str(exc)) from exc
                    response = payload["response"]
                    request = payload["request"]
                    if response.get("status") == "source_outage":
                        evidence = _validate_probe_evidence(response.get("probe_evidence") or {}, request["source"])
                        if response.get("probe_evidence_hash") != _hash(evidence) or response.get("requested") is not False or (response.get("diagnostics") or {}).get("request_count") != 0:
                            raise SystemExit(f"source_outage_evidence_invalid:{key}")
                    if response.get("status") == "request_aborted":
                        evidence = response.get("abort_evidence") or {}
                        required = {"source", "observed_stalled_request_key", "observed_stalled_kind", "observed_stalled_query_hash", "stall_threshold_seconds", "termination_reason", "evidence_recorded_at"}
                        expected_attempted = request["key"] == evidence.get("observed_stalled_request_key")
                        if evidence.get("source") != request["source"] or not required.issubset(evidence) or response.get("abort_evidence_hash") != _hash(evidence) or response.get("requested") is not expected_attempted or (response.get("diagnostics") or {}).get("request_count") != int(expected_attempted):
                            raise SystemExit(f"request_abort_evidence_invalid:{key}")
                    if kind == "identifier" and request.get("available"):
                        oracle["identifier_applicable"] = True
                    if kind != "identifier" and request.get("available"):
                        oracle["title_applicable"] = True
                    if _is_failed_request_status(response["status"]) and request.get("available"):
                        oracle["failed_count"] += 1
                    if response["status"] == "success" and request.get("available"):
                        oracle["successful_count"] += 1
                    matches = [(_matches(p, gold), i + 1) for i, p in enumerate(response.get("papers") or [])]
                    stable = [rank for ((stable, _), rank) in matches if stable]
                    title = [rank for ((_, title), rank) in matches if title]
                    oracle["matches"][kind] = {"stable_rank": min(stable) if stable else None, "title_rank": min(title) if title else None}
                    if stable or title: oracle["hit"] = True
                    if kind == "identifier" and stable:
                        oracle["identifier_hit"] = True
                    if kind == "exact_title" and (stable or title):
                        oracle["exact_title_hit"] = True
                    if kind == "fixed_depth_title" and (stable or title):
                        oracle["fixed_depth_hit"] = True
                        oracle["fixed_depth_rank"] = min(stable or title)
                oracle["complete"] = oracle["title_applicable"] and oracle["failed_count"] == 0
                oracle["direct_identifier_not_found"] = False
                fixed_key = kinds.get("fixed_depth_title")
                if fixed_key:
                    fixed_response = json.loads((snap / f"{fixed_key}.json").read_text())["response"]
                    oracle["fixed_depth_failed"] = fixed_response["status"] in {"failed", "source_outage", "request_aborted"}
                    oracle["fixed_depth_outage"] = fixed_response["status"] == "source_outage"
                    oracle["fixed_depth_complete"] = fixed_response["status"] == "success"
                current_signal = _current_signal(item["current"], gold, source)
                oracle["classification"] = classify_source(oracle=oracle, current=current_signal)
                per_source[source] = {"oracle": oracle, "current": current_signal}
            records.append({"case_id": item["case_id"], "gold": gold, "requests": item["requests"], "sources": per_source})
        counts = {}
        by_source = {}
        request_status = {}
        abort_summary = {"attempted_aborted_requests": 0, "not_requested_aborted_requests": 0, "attempted_gold_source_pairs": 0, "not_requested_gold_source_pairs": 0}
        for record in records:
            for source, data in record["sources"].items():
                category = data["oracle"]["classification"]
                counts[category] = counts.get(category, 0) + 1
                by_source.setdefault(source, {})[category] = by_source.setdefault(source, {}).get(category, 0) + 1
                for kind, key in record["requests"].get(source, {}).items():
                    response = json.loads((snap / f"{key}.json").read_text())["response"]
                    request_status.setdefault(source, {}).setdefault(kind, {}).setdefault(response["status"], 0)
                    request_status[source][kind][response["status"]] += 1
                    if response["status"] == "request_aborted":
                        if response.get("requested"):
                            abort_summary["attempted_aborted_requests"] += 1
                        else:
                            abort_summary["not_requested_aborted_requests"] += 1
                source_abort = any(
                    json.loads((snap / f"{key}.json").read_text())["response"]["status"] == "request_aborted"
                    for key in record["requests"][source].values()
                )
                if source_abort:
                    attempted_pair = any(
                        (lambda response: response.get("status") == "request_aborted" and response.get("requested"))(json.loads((snap / f"{key}.json").read_text())["response"])
                        for key in record["requests"][source].values()
                    )
                    abort_summary["attempted_gold_source_pairs" if attempted_pair else "not_requested_gold_source_pairs"] += 1
        aggregate = {
            "gold_count": len(records),
            "source_count": sum(len(record["sources"]) for record in records),
            "classification_counts": dict(sorted(counts.items())),
            "by_source": {source: dict(sorted(values.items())) for source, values in sorted(by_source.items())},
            "request_status": request_status,
            "request_aborted_summary": abort_summary,
            "source_outage_gold_source_pairs": {
                source: sum(
                    data["oracle"].get("fixed_depth_failed", False)
                    and any(
                        json.loads((snap / f"{key}.json").read_text())["response"]["status"] == "source_outage"
                        for key in record["requests"][source].values()
                    )
                    for record in records
                    for data in [record["sources"][source]]
                )
                for source in SOURCES
            },
            "potential_coverage_upper_bound": potential_coverage_upper_bound(records),
            "capabilities": {
                source: {
                    kind: requests[key].get("capability")
                    for kind, key in records[0]["requests"][source].items()
                }
                for source in SOURCES
            } if records else {},
            "unobservable": {
                "returned_beyond_fixed_candidate_depth_count": 0,
                "reason": "fixed_depth_title uses a bounded generic search; no rank beyond depth is observable",
            },
            "replay_http_requests": 0,
            "depth": args.depth,
        }
        if not args.source and aggregate["source_count"] != aggregate["gold_count"] * 4:
            raise SystemExit("source_count_mismatch")
        (out / "gold_audit.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n")
        (out / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="四源 Gold 可检索性 oracle Record/Replay 审计")
    parser.add_argument("--mode", choices=("record-missing", "source-outage", "request-abort", "replay"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--snapshot", default="outputs/benchmark_snapshots/cross_source_gold_d889")
    parser.add_argument("--output", default="outputs/benchmark_runs/cross_source_gold_d889")
    parser.add_argument("--current-root", default="outputs/benchmark_runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS)
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-evidence")
    parser.add_argument("--abort-evidence")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
